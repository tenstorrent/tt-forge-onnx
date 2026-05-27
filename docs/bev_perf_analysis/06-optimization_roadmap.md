# Optimization Roadmap — tt-mlir Changes for 30 FPS BEV

**Date:** 2026-05-19  
**Baseline:** 2.07 FPS (483.76 ms/frame)  
**Target:** 30 FPS (33.3 ms/frame)  
**Required speedup:** 14.6×

This roadmap lists specific tt-mlir changes in priority order. Each item includes: what to change, which file/pass, expected speedup, test to run, and configuration.

---

## Priority Legend

- **P1 — Critical Path:** Changes to passes that directly reduce DRAM round-trips for conv2d weights and activations. Maximum expected speedup.
- **P2 — Sharding Improvement:** Changes to MLA/sharding propagation to reduce `<1x1>` placement.
- **P3 — Overhead Reduction:** Eliminate roundtrip pairs and unnecessary format conversions.
- **P4 — Kernel-Level:** Changes that require kernel modifications (longer lead time).

---

## P1-A: Enable L1 Weight Caching (`config_tensors_in_dram = false`)

### Problem
All 187 conv2d ops (148 in Block A, 39 in Block C) have `config_tensors_in_dram = true`. This forces weight tensors to be fetched from DRAM on every convolution call. The weights are never pinned in L1 between inferences or between the 4 serial camera branches in Block A.

### What to Change

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/OptimizerPasses/TTNNPrepareConv2dWeightsAndBias.cpp`

Look for the logic that sets `config_tensors_in_dram = true` on `prepare_conv2d_weights` ops. This is likely a conservative default for large models where L1 cannot hold all weights simultaneously.

The change needed: add a per-op analysis that checks if a given weight tensor can fit in L1 (weight_size <= L1_budget_for_weights), and sets `config_tensors_in_dram = false` for those ops that qualify.

**Alternative file:** `third_party/tt-mlir/lib/Dialect/TTNN/Analysis/L1SpillManagement.h` or `GreedyMemoryLayoutPropagation.cpp` — the decision may be made there during the MLA weight placement analysis.

### Expected Speedup

For ops where the weight fits in L1:
- Eliminates DRAM round-trip latency for weight fetch (estimated 100–200 ns per tile × N tiles per weight per call)
- For the 4 serial branches in Block A with shared weights, L1-cached weights would be fetched once and reused 4× — a 4× reduction in weight-fetch latency

Conservative estimate: **1.5–2.5× speedup for Block A and C** (the weight-fetch latency currently dominates op scheduling). Overall model speedup: **1.4–2.0×** (given A+C = 89% of time).

### Test to Run

```bash
# Test block A in isolation
pytest forge/test/mlir/test_ops_onnx.py::test_bev_block_A \
  --opt-level 2 --math-fidelity HiFi3 --fp32-dest-acc true \
  --enable-trace true --data-type Float16_b

# Full model
pytest forge/test/models/onnx/vision/bev/test_bev_full_model.py \
  --opt-level 2 --math-fidelity HiFi3 --fp32-dest-acc true \
  --enable-trace true --data-type Float16_b
```

### Configuration
- opt_level=2, HiFi3, fp32_dest_acc=true, trace=enabled, Float16_b
- Do NOT disable any passes or change math config
- Measure with: `--benchmark-iters 10` to get stable timing

---

## P1-B: Eliminate DRAM→L1→DRAM Roundtrip Pairs

### Problem
44 roundtrip pairs in full model (32 in Block A, 10 in Block C, 2 in Block E). Each pair moves data to DRAM only to immediately reload it, doubling the DRAM latency for affected tensors.

### What to Change

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyL1SpillManagement.cpp`

The Belady-optimal L1 spill algorithm in `TTNNGreedyL1SpillManagement` decides which tensors to spill to DRAM. When it spills a tensor that is the immediate input to the next L1-requiring op, a roundtrip pair is created.

The fix: after the spill decision pass, add a post-processing step that detects roundtrip pairs:
```
Pattern: to_memory_config(%t, DRAM) immediately followed by to_memory_config(%spilled_t, L1)
         where the L1 result is the direct successor's input
```
When this pattern is found, the spill can be eliminated by keeping `%t` in L1 and accepting a larger L1 footprint for that interval (if L1 budget permits).

**Alternative approach:** In `L1SpillManagement.h`, the `SumL1MemoryTracker` tracking can be extended to detect when a spill-then-reload would happen and choose a different eviction candidate.

### Expected Speedup

44 roundtrip pairs × 2 DRAM operations each = 88 to_memory_config ops eliminated.  
At ~0.3 ms per DRAM config op (estimated): ~26 ms savings.  
Overall model speedup: ~**1.05×** (modest but free).

### Test to Run

Same as P1-A tests. Verify by checking MEMDUMP counts in the build log — roundtrip pairs should drop from 44 to 0 (or near 0).

```bash
# Check MEMDUMP output in build log for:
# "DRAM->L1->DRAM pairs: N" — should be 0 after fix
```

### Configuration
Same as P1-A.

---

## P2-A: Improve Sharding Propagation to Reduce `<1x1>` Tensor Count

### Problem
73.5% of tensor layout annotations in Block A use `<1x1>` (single-core DRAM interleaved). This means ~73% of tensors are NOT distributed across the 64-core grid. Many of these tensors at intermediate resolutions (96×96, 48×48, etc.) would analytically fit in L1 if sharded, but MLA is being conservative.

### Root Cause
The MLA (`TTNNGreedyMemoryLayoutPropagation`) uses L1 footprint estimates that include circular buffer (CB) overhead. When CB fragmentation is estimated to exceed safe limits, MLA falls back to DRAM interleaved placement. The 73.5% DRAM placement suggests the CB fragmentation estimate is overly conservative.

### What to Change

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyMemoryLayoutPropagation.cpp`

1. Review the CB fragmentation estimate used in the L1 budget calculation. If the estimate uses a worst-case padding factor (e.g., 2× for CB alignment), try reducing it for ops where the actual CB usage is well-bounded.

2. The `ShardSolver` (referenced in MLA) determines valid sharding configurations. Check if it's correctly propagating sharding across reshape/concat boundaries — if a reshape breaks sharding propagation, downstream ops default to `<1x1>` even when they could be sharded.

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Analysis/L1SpillManagement.h`

The `SumL1MemoryTracker` tracks the sum of live tensor L1 sizes. Review whether it accounts for tensor reuse (a tensor that is freed before the next shard is allocated should not be double-counted in the L1 budget).

### Expected Speedup

If `<1x1>` percentage drops from 73.5% to 40% (by sharding tensors at 96×96 and below that currently default to DRAM):
- 33.5% more ops become L1-sharded → reduced DRAM round-trips for activations
- Combined with P1-A (weight caching): **1.5–2.0× additional speedup** beyond P1-A alone

### Test to Run

```bash
pytest forge/test/mlir/test_ops_onnx.py::test_bev_block_A \
  --opt-level 2 --math-fidelity HiFi3 --fp32-dest-acc true \
  --enable-trace true --data-type Float16_b
# Check <1x1> count in 09-mla.mlir: grep -c '<1x1>' BEV_MODEL_IRS/BLOCK_A/09-mla.mlir
# Should decrease from 2802 to <1400
```

### Configuration
opt_level=2, HiFi3, fp32_dest_acc=true, trace=enabled, Float16_b

---

## P2-B: Fix `dram_width` Conv2d at 1536×1536 (If L1 Is Sufficient)

### Problem
8 conv2d ops at 1536×1536 use `conv2d_slice_config = dram_width`. Analysis shows that at 3 input channels and 64 cores:
- Activation shard: 1536×1536×3 / 64 = 110 KB per core
- Weight: 3×3×3×C BF16 ≈ negligible for small C
- This could fit in 1.43 MB L1

These ops currently stream through DRAM even though L1 may suffice.

### What to Change

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyMemoryLayoutPropagation.cpp` or the slice config assignment logic.

The `conv2d_slice_config` is set to `dram_width` when the convolution cannot be fully sliced to fit in L1. Review the threshold computation: check if output channel doubling (e.g., 3→32) is being conservatively included in the L1 estimate for the slice that processes the input. The input-side analysis should only count the input slice and weight, not the output expansion.

### Expected Speedup

If the 8 `dram_width` ops at 1536×1536 transition to `l1_full` with proper sharding:
- Each op currently streaming 13.5 MB through DRAM → becomes L1-resident
- DRAM latency for these 8 ops reduced significantly
- **1.1–1.3× speedup** for Block A alone (the 8 ops represent ~8/148 = 5% of conv ops but are at maximum spatial resolution)

### Test to Run

```bash
pytest forge/test/mlir/test_ops_onnx.py::test_bev_block_A \
  --opt-level 2 --math-fidelity HiFi3 --fp32-dest-acc true \
  --enable-trace true --data-type Float16_b
# Verify: grep -c 'dram_width' BEV_MODEL_IRS/BLOCK_A/09-mla.mlir
# Should decrease from 8 to 0 (or fewer)
```

---

## P3-A: Fuse Grid Sample Format Conversion Chain

### Problem
Each of the 40 grid_sample ops (32 in Block B + 8 in Block D) is surrounded by 6 format conversion ops:
- 2 `to_memory_config` (L1↔DRAM)
- 2 `to_layout` (tile↔row_major)
- 2 additional ops for grid LUT preparation

These 6 ops are mandatory given the current kernel interface (ROW_MAJOR input/output constraint), but some can be fused or eliminated if the surrounding context allows.

### What to Change

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp` (grid_sample workaround)

The workaround currently inserts individual `to_layout` + `to_memory_config` ops before and after `grid_sample`. A potential optimization:

1. **Fuse the pre-grid_sample `to_memory_config` + `to_layout`** into a single op that converts tile+L1 → row_major+DRAM in one DMA operation if hardware supports it.

2. **Check if the feature input's preceding op outputs row_major naturally**: If the producer of the grid_sample input happens to output row_major (e.g., another grid_sample, or a `to_layout` from a different chain), the pre-conversion can be eliminated.

3. **Fuse the grid LUT slice → reshape → to_memory_config → to_layout** into a precomputed constant tensor that is already row_major+DRAM at compile time (the LUT is static).

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`

Add a `TTNNFuseGridSampleConversions` pass before the workarounds pass that:
- Detects `[to_memory_config] → [to_layout] → [grid_sample] → [to_layout] → [to_memory_config]` pattern
- Replaces it with `[combined_prep] → [grid_sample_with_builtin_conversion] → [combined_post]`

### Expected Speedup

For Block B (~30 ms estimated): eliminating 50% of format conversion ops (~96/192) → **~10–15% Block B speedup** (3–5 ms savings).  
For Block D (16.97 ms): similar proportional improvement.  
Overall model: **~1–2% speedup** (given B+D are ~5% of total time).

This is a low absolute-time gain but clean engineering: the format conversion overhead currently accounts for ~48% of all ops in Block B.

### Test to Run

```bash
pytest forge/test/mlir/test_ops_onnx.py::test_bev_block_B \
  --opt-level 2 --math-fidelity HiFi3 --fp32-dest-acc true \
  --enable-trace true --data-type Float16_b
# Note: Block B test currently fails PCC=0.988 vs threshold=0.99
# This is EXPECTED — run with --pcc-threshold 0.985 for timing analysis

pytest forge/test/mlir/test_ops_onnx.py::test_bev_block_D \
  --opt-level 2 --math-fidelity HiFi3 --fp32-dest-acc true \
  --enable-trace true --data-type Float16_b
```

---

## P3-B: Precompute Grid LUT in Row-Major DRAM Format

### Problem
For each of the 40 grid_sample calls, 4 ops prepare the sampling grid at inference time:
1. `slice_static` — extracts 2D grid from 5D LUT (static per call)
2. `reshape` — reformats for kernel API
3. `to_memory_config` — L1 → DRAM
4. `to_layout` — tile → row_major

Steps 1–4 are deterministic (the LUT is a precomputed constant). With `consteval` enabled, constant folding should propagate through these ops. If it does not, the fix is to ensure `slice_static + reshape + to_memory_config + to_layout` is recognized as a constant subgraph.

### What to Change

**File:** `forge/csrc/passes/constant_folding.cpp`

Check if `ttir.slice_static` on a constant tensor is being folded. If not, add it to the constant folding opset.

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`

Ensure the consteval (constant evaluation) pass runs after the workarounds pass so it can fold the grid LUT preparation chain (which is inserted by the workarounds pass and thus not visible during earlier consteval).

### Expected Speedup

If grid LUT preparation is fully constant-folded:
- 4 ops × 40 grid_sample calls = 160 ops eliminated from inference critical path
- Grid tensors become compile-time constants stored in row_major DRAM format
- At ~0.2 ms per op: ~32 ms theoretical savings in Blocks B+D
- **~5–7% overall model speedup** (more impactful than P3-A)

### Test to Run

```bash
# Verify consteval is folding the grid preparation
# After build, check 01-ttir-passes.mlir for absence of slice_static on LUT inputs
# (they should have been replaced by constant tensors)
grep -c 'slice_static' BEV_MODEL_IRS/BLOCK_B/01-ttir-passes.mlir  
# Before fix: 32
# After fix: 0 (all folded into constants)
```

---

## P4: Parallel Camera Branch Processing (Multi-Stream)

### Problem
Block A's 148 conv2d ops process 4 camera branches **serially**. The 4 branches are structurally independent (no data dependency between them). All 4 share the same weights and the same FPN architecture. Serial execution wastes 3/4 of available throughput for inter-branch parallelism.

### What to Change

This requires a fundamental change to the tt-mlir execution model. Options:

**Option 1: Op-level parallelism in TTNN (tt-metal multi-stream)**
- The 4 camera branches need to be submitted to the device simultaneously as separate command queues
- Requires `tt-metal` multi-CQ (command queue) support
- **File:** Not a compiler change — requires device runtime changes in `tt-metal`

**Option 2: Tile-level batching (batch all 4 cameras as batch=4)**
- Reshape the model to process 4×1536×1536 as a single batch-4 input
- conv2d on batch=4 would naturally use 4× the tiles and could benefit from better hardware utilization
- **File:** Model-level change in `forge/test/models/onnx/vision/bev/` — modify the block decomposition to batch the 4 camera inputs

**Option 3: Shared-weight detection in compiler**
- If the compiler detects that 4 serial blocks use identical weights, it could automatically batch them
- **File:** A new pass in `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/Passes/` — `TTNNBatchSerialBranches.cpp`
- High complexity, but potentially the cleanest solution

### Expected Speedup

If 4 camera branches become fully parallel: up to **4× speedup for Block A** (326 ms → ~82 ms). More realistic estimate (given memory bandwidth limits): **2–3× speedup** → Block A: 110–163 ms.

Overall model: **2–3× speedup** (given Block A is 67% of total).

This is the highest-leverage single change but also the most architecturally complex. Combined with P1-A and P2-A, target FPS could reach 10–15 FPS.

### Test to Run

```bash
# Model-level batching test (if Option 2 is implemented)
pytest forge/test/models/onnx/vision/bev/test_bev_block_A_batched.py \
  --opt-level 2 --math-fidelity HiFi3 --fp32-dest-acc true \
  --enable-trace true --data-type Float16_b --batch-size 4
```

---

## Summary Table: All Optimizations

| ID | Change | Primary File | Expected Block Speedup | Expected Model Speedup | Complexity |
|----|--------|-------------|------------------------|------------------------|------------|
| P1-A | L1 weight caching | `TTNNPrepareConv2dWeightsAndBias.cpp` | A: 1.5–2.5×, C: 1.5–2× | 1.4–2.0× | Medium |
| P1-B | Eliminate roundtrip pairs | `GreedyL1SpillManagement.cpp` | A: 1.05×, C: 1.05× | 1.05× | Medium |
| P2-A | Reduce `<1x1>` placement | `GreedyMemoryLayoutPropagation.cpp` | A: 1.3–1.8×, C: 1.3× | 1.2–1.6× | High |
| P2-B | dram_width→l1_full at 1536×1536 | `GreedyMemoryLayoutPropagation.cpp` | A: 1.1–1.3× | 1.05–1.2× | High |
| P3-A | Fuse grid_sample format chain | `TTNNWorkaroundsPass.cpp`, Pipelines | B: 1.1–1.15×, D: 1.1× | 1.01–1.02× | Low-Medium |
| P3-B | Consteval grid LUT prep | `constant_folding.cpp`, Pipelines | B: 1.2×, D: 1.2× | 1.05–1.07× | Medium |
| P4 | Parallel camera branches | `tt-metal` CQ or model batching | A: 2–4× | 2–3× | Very High |

### Achievable FPS Estimates

| Scenario | Optimizations Applied | Estimated Time (ms) | Estimated FPS |
|----------|----------------------|---------------------|---------------|
| Baseline | None | 483.76 | 2.07 |
| Quick wins | P1-A + P1-B | ~250–320 | 3–4 |
| Compiler improvements | P1-A + P1-B + P2-A + P2-B + P3-B | ~130–200 | 5–8 |
| + Grid fusion | + P3-A | ~125–195 | 5–8 |
| + Parallel branches | + P4 | ~40–100 | 10–25 |
| Theoretical max (all P1-P4) | All | ~30–50 | **20–33** |

**Reaching 30 FPS on a single device requires all optimizations (P1 through P4).** P1-A and P4 are the critical path. Without parallel branch execution (P4), the ceiling is ~8 FPS even with perfect compiler optimizations.

---

## Recommended Implementation Order

1. **First: P1-A** (weight caching) — straightforward pass change, high impact, low risk of PCC regression
2. **Then: P1-B** (roundtrip elimination) — follow-on to Belady pass, additive with P1-A
3. **Then: P3-B** (consteval grid LUT) — clean constant folding, no numerical risk
4. **Then: P2-A** (sharding propagation) — more complex, needs careful L1 budget validation
5. **Then: P3-A** (grid_sample format fusion) — dependent on P3-B being done first
6. **Then: P2-B** (dram_width→l1_full) — requires validating L1 budget at 1536×1536 accurately
7. **Last: P4** (parallel branches) — largest impact but requires cross-team effort with tt-metal

---

## Constraints Reminder

All changes must preserve:
- `trace = enabled` (do not disable trace)
- `opt_level = 2` (all optimization passes active)
- `HiFi3` math fidelity
- `fp32_dest_acc = true`
- `Float16_b` data type
- `consteval = enabled`
- Block B PCC = 0.988 is acceptable (nearest-neighbor BF16 behavior, NOT a regression target)
- Single device only (no multi-chip changes)
- TTNN GridSample ROW_MAJOR output constraint is a kernel fact, not a bug to fix
