# BEV Performance — Concrete Implementation Plan

**Date:** 2026-05-19  
**Baseline:** 2.07 FPS (484 ms)  
**Target:** 30 FPS (33.3 ms)  
**Required speedup:** 14.6×  
**Frozen config:** trace=True, opt_level=2, HiFi3, fp32_dest_acc=True, Float16_b, consteval=True

---

## Summary of Findings

| Block | Time | % | Conv2d | to_memory_config | Roundtrips | Root Cause |
|-------|------|---|--------|-----------------|------------|------------|
| A | 326 ms | 67% | 148 | 288 | 32 | DRAM weights, 73.5% <1x1> tensors |
| C | 107 ms | 22% | 39 | 87 | 10 | Same as A, smaller scale |
| B | ~30 ms | 6% | 8 | 104 | 0 | PCC fail (known), grid_sample overhead |
| D | 17 ms | 4% | 2 | 27 | 0 | Already good (58 FPS) |
| E | 6 ms | 1% | 18 | 41 | 2 | Already fast |
| F | 6 ms | 1% | 11 | 21 | 0 | Already fast |

**Single largest finding:** All 226 conv2d ops across all blocks use `config_tensors_in_dram=true` — weights fetched from DRAM on every inference call. This alone is the dominant performance bottleneck.

---

## Fix 1 (P1-A) — L1 Weight Caching for Conv2d

### What
`ttnn.prepare_conv2d_weights` emits `config_tensors_in_dram=true` for all ops. This forces every conv2d to fetch its weight tensor from DRAM at runtime. Weights should be pinned in L1 when they fit.

### Where
**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/OptimizerPasses/TTNNPrepareConv2dWeightsAndBias.cpp`

Look for where `config_tensors_in_dram` / `weights_on_device` is set.

**Alternative:** `third_party/tt-mlir/lib/Dialect/TTNN/Analysis/` — the decision may live in `ShardSolver` or weight placement analysis.

### How
Add per-op weight size check:
- Compute `weight_bytes = out_ch × in_ch/groups × kH × kW × sizeof(bf16)`
- If `weight_bytes / num_cores <= L1_WEIGHT_BUDGET` (suggest: 200 KB per core threshold), set `config_tensors_in_dram = false`
- The 8 large 1536×1536 stride-2 downsamplers (3→64 channels, kernel 3×3, `dram_width` slice) should stay in DRAM — weight is only ~3.4 KB but activation is DRAM-sliced anyway
- The 120+ mid/small conv2d at 96×96 and below (weight 20–300 KB, 58–64 cores) are the targets

### Expected Impact
- 1.5–2.5× speedup for Block A (weight-fetch latency eliminated for 140 of 148 ops)
- 1.3–2.0× for Block C (35 of 39 ops)
- **Combined estimated full-model speedup: 1.4–2.0× → ~240–350 ms → 2.9–4.1 FPS**

### Test Command
```bash
export TTMLIR_DUMP_PIPELINE_IR=1
tt-smi -r
unset FORGE_RELOAD_GENERATED_MODULES
mkdir -p BEV_MODEL_IRS/FIX1_BLOCK_A
export TTMLIR_DUMP_DIR=BEV_MODEL_IRS/FIX1_BLOCK_A
pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py \
  -k "block_A and opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled and enable_program_cache" \
  -vss &> BEV_MODEL_LOGS/fix1_block_A.log

# Verify in IR: weight ops should show config_tensors_in_dram=false
grep -c "config_tensors_in_dram = false" BEV_MODEL_IRS/FIX1_BLOCK_A/09-mla.mlir
```

### Verification
In `BEV_MODEL_IRS/FIX1_BLOCK_A/09-mla.mlir`:
- Count `config_tensors_in_dram = false` — should be ≥120 (vs 0 baseline)
- Count `config_tensors_in_dram = true` — should drop from 148 to ≤8

---

## Fix 2 (P1-B) — Eliminate DRAM→L1→DRAM Roundtrip Pairs

### What
44 roundtrip pairs in full model (32 in Block A, 10 in Block C). Pattern:
```
L1-sharded tensor
  → to_memory_config (L1→DRAM)   [forced by reshape needing DRAM input]
  → reshape                       [runs in DRAM]
  → to_memory_config (DRAM→L1)   [next conv needs L1 input]
```
This is pure overhead — data written to DRAM and immediately read back.

### Where
**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNDecomposeLayouts.cpp`

The `TTNNDecomposeLayouts` pass is what inserts `to_memory_config` ops. Adding a post-pass or extending it to detect and eliminate roundtrips here is the correct fix point.

**Alternative:** Re-enable the post-decompose `CanonicalizerPass` that we previously commented out in `TTNNPipelines.cpp`:
```cpp
// In createTTIRToTTNNCommonPipeline, after createTTNNPipelineLayoutDecompositionPass:
devicePm.addPass(mlir::createCanonicalizerPass());
```
This needs `ToMemoryConfigOp::fold` to detect and fold the roundtrip pairs. The `foldDRAMtoL1toDRAMRoundTrip` function (currently commented out in `TTNNOps.cpp`) handles exactly this.

**Specifically in** `third_party/tt-mlir/lib/Dialect/TTNN/IR/TTNNOps.cpp`:
```cpp
// The fold was commented out. Re-enable it, but only for non-trace paths,
// OR only for pairs that don't cross trace boundaries.
```

### How
1. In `TTNNOps.cpp`, re-enable `foldDRAMtoL1toDRAMRoundTrip` but add a guard:
   - Only fold if the two `to_memory_config` ops are adjacent (no ops between them)
   - Only fold if both are within the same trace region (not across `begin_trace/end_trace` boundary)
2. Add the post-decompose canonicalization back in `TTNNPipelines.cpp` but gated properly

### Expected Impact
- Eliminates 44 roundtrip pairs × 2 DRAM ops = 88 `to_memory_config` ops
- **Estimated speedup: ~1.05–1.10× → saves 24–50 ms → ~255–370 ms**

### Test Command
```bash
tt-smi -r
mkdir -p BEV_MODEL_IRS/FIX2_BLOCK_A BEV_MODEL_IRS/FIX2_BLOCK_C
export TTMLIR_DUMP_DIR=BEV_MODEL_IRS/FIX2_BLOCK_A
pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py \
  -k "block_A and opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled and enable_program_cache" \
  -vss &> BEV_MODEL_LOGS/fix2_block_A.log

# Verify: roundtrip pairs should drop from 32 to 0
grep "DRAM->L1->DRAM pairs" BEV_MODEL_LOGS/fix2_block_A.log
```

---

## Fix 3 (P2-A) — Improve Sharding Propagation (Reduce <1x1> Tensors)

### What
73.5% of tensor layout annotations in Block A are `<1x1>` (single-core DRAM interleaved). Many tensors at 96×96, 48×48 spatial resolutions could be L1-sharded across 64 cores but MLA defaults to DRAM.

Specific suboptimal case found: 4 `conv2d(12×12, 384→320, kernel=3x3)` ops land on `8x8/L1/interleaved` (all 64 cores but NOT sharded — interleaved means each core has random tiles). The follow-up conv2d uses `5x5/L1/block_sharded`, requiring a reshard `to_memory_config`.

### Where
**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyMemoryLayoutPropagation.cpp`

Look for CB fragmentation/overhead estimate. This estimate determines when MLA falls back from L1-sharded to DRAM-interleaved.

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Analysis/L1SpillManagement.h`

The `SumL1MemoryTracker` determines if a tensor can fit in L1. Review whether it double-counts tensors that are freed before the next allocation.

### How
1. Add diagnostic output showing WHY each op was placed in DRAM (which constraint triggered fallback)
2. Review CB overhead multiplier — if using 2× safety factor, try 1.5× for conv2d outputs
3. Fix the `8x8/interleaved` at 12×12 case: force `5x5/block_sharded` directly to match the follow-up op's layout and avoid the reshard

### Expected Impact
- If `<1x1>` percentage drops from 73.5% to 40%: **1.3–1.8× speedup**
- Fixing the 4 specific `interleaved → block_sharded` reshard pairs: saves 4 `to_memory_config` ops (minor)

### Test Command
```bash
tt-smi -r
mkdir -p BEV_MODEL_IRS/FIX3_BLOCK_A
export TTMLIR_DUMP_DIR=BEV_MODEL_IRS/FIX3_BLOCK_A
pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py \
  -k "block_A and opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled and enable_program_cache" \
  -vss &> BEV_MODEL_LOGS/fix3_block_A.log

# Measure sharding improvement:
grep -c "<1x1>" BEV_MODEL_IRS/FIX3_BLOCK_A/09-mla.mlir  # Should drop from 2802
```

---

## Fix 4 (P3) — Fuse to_layout + to_memory_config around grid_sample

### What
Each of the 40 `ttnn.grid_sample` ops (32 in Block B, 8 in Block D) is surrounded by:
```
to_memory_config (DRAM→L1)
to_layout (TILE→ROW_MAJOR)      ← required: kernel needs ROW_MAJOR input
grid_sample
to_layout (ROW_MAJOR→TILE)      ← output comes out ROW_MAJOR
to_memory_config (L1→DRAM)
```
The `to_layout + to_memory_config` pairs could be fused into a single `to_layout` call that simultaneously changes layout and memory location, saving 2 kernel dispatches per grid_sample.

### Where
**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNWorkaroundsPass.cpp`

The `createGridSampleOpOperandsWorkarounds` function already handles ROW_MAJOR requirements. Add a canonicalization pattern or workaround extension that fuses the surrounding format conversions.

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/IR/TTNNOps.cpp`

Add a fold or canonicalization pattern for `to_memory_config(to_layout(x))` → `to_layout(x, new_memcfg)`.

### Expected Impact
- 40 grid_sample ops × 2 ops saved = 80 fewer `to_memory_config` dispatches
- **Block D speedup: ~1.05–1.10×** (minor given 17ms baseline)
- **Block B: no timing data but structure improves**

### Test Command
```bash
tt-smi -r
mkdir -p BEV_MODEL_IRS/FIX4_BLOCK_D
export TTMLIR_DUMP_DIR=BEV_MODEL_IRS/FIX4_BLOCK_D
pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py \
  -k "block_D and opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled and enable_program_cache" \
  -vss &> BEV_MODEL_LOGS/fix4_block_D.log
```

---

## Fix 5 (P4) — Parallel Camera Branches (Multi-Stream)

### What
Block A processes 4 camera inputs **serially**. Each camera branch runs ~81 ms (326 ms / 4). If branches ran in parallel, Block A would take ~81 ms — a **4× speedup on Block A alone**.

### Options

**Option A — Model restructuring (batch dimension)**
Change the model to use batch=4 for Block A:
- Input: `4×3×1536×1536` (batched) instead of 4× `1×3×1536×1536` (serial)
- Requires model-level change in `create_bev_splits.py` or the ONNX export
- MLA can then distribute the batch across cores

**Option B — Multi-CQ (command queue) parallelism in tt-metal**
Use tt-metal's multi-queue API to dispatch Block A branches on separate command queues that run concurrently. This requires changes in the TTNN runtime, not just the compiler.

**Option C — TTIR-level batching transformation**
Add a pass in the TTIR pipeline that detects repeated identical subgraph patterns (the 4 camera branches share the same structure) and merges them into a single batched call.

### Expected Impact
- **Option A or C: up to 3–4× speedup on Block A → ~80–110 ms for Block A**
- Combined with fixes 1–4: estimated **200–250 ms total → 4–5 FPS**
- Combined with EVERYTHING: estimated **80–120 ms → 8–12 FPS** on single device

### Note on 30 FPS
30 FPS on a single device requires 33.3 ms/frame. Even with:
- Block A at 80 ms (4× parallel improvement)
- Block C at 50 ms (2× sharding improvement)
- Block B at 20 ms (estimated with fixes)
- Block D+E+F at ~29 ms

Sum = ~180 ms → 5.5 FPS. **30 FPS is not achievable on a single device with this model's compute requirements.** Multi-device or model compression would be needed for 30 FPS.

### Test Command
```bash
# After implementing batch dimension change:
tt-smi -r
mkdir -p BEV_MODEL_IRS/FIX5_BLOCK_A
export TTMLIR_DUMP_DIR=BEV_MODEL_IRS/FIX5_BLOCK_A
pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py \
  -k "block_A and opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled and enable_program_cache" \
  -vss &> BEV_MODEL_LOGS/fix5_block_A.log
```

---

## Realistic FPS Ceiling Analysis

| Fixes Applied | Block A | Block C | Total | FPS |
|---|---|---|---|---|
| Baseline | 326 ms | 107 ms | 484 ms | 2.1 |
| Fix 1 (weight caching) | ~180 ms | ~60 ms | ~290 ms | 3.4 |
| Fix 1+2 (roundtrips) | ~165 ms | ~54 ms | ~250 ms | 4.0 |
| Fix 1+2+3 (sharding) | ~110 ms | ~36 ms | ~175 ms | 5.7 |
| Fix 1+2+3+5 (parallel branches) | ~50 ms | ~36 ms | ~115 ms | 8.7 |
| Fix 1+2+3+5 + model compression | ~30 ms | ~20 ms | ~75 ms | 13.3 |
| 30 FPS ceiling | ~10 ms | ~7 ms | ~33 ms | 30.0 |

The gap between ~8.7 FPS (achievable with compiler fixes + model restructuring) and 30 FPS requires fundamental model compression, quantization, or multi-device deployment beyond the scope of compiler optimization alone.

---

## Implementation Order (Start Here)

```
Week 1: Fix 2 (roundtrip elimination) — low risk, quantifiable gain, no accuracy impact
  → Verify: MEMDUMP roundtrip count drops from 44 to 0
  → Verify: PCC ≥ 0.99 for blocks A, C, D, E, F

Week 2: Fix 1 (L1 weight caching) — medium risk, highest gain
  → Start with Block C (smaller, easier to validate)
  → Then apply to Block A
  → Verify: no OOM (monitor L1 usage via MEMDUMP)

Week 3: Fix 3 (sharding propagation)
  → Start with the specific 4 ops (12×12 interleaved→block_sharded fix)
  → Then review CB fragmentation estimate broadly

Week 4: Fix 4 (grid_sample fusion) + full model validation
  → Run full BEV benchmark with all fixes
  → Verify PCC ≥ 0.99 for full model
```

---

## Files in This Analysis

| File | Content |
|------|---------|
| `00-summary.md` | Executive summary, block breakdown, speedup estimates |
| `01-block_A_analysis.md` | Block A deep dive (MLA sharding, memory traffic, root causes) |
| `02-block_C_analysis.md` | Block C analysis |
| `02-block_C_B_analysis.md` | Block C + Block B combined analysis |
| `03-block_B_analysis.md` | Block B PCC failure root cause |
| `04-block_D_E_F_analysis.md` | Blocks D, E, F (already fast — reference state) |
| `05-memory_traffic_analysis.md` | DRAM↔L1 traffic quantification across all blocks |
| `06-optimization_roadmap.md` | 7 specific tt-mlir changes in priority order |
| `07-pipeline-passes-analysis.md` | All 40+ pipeline passes documented |
| `08-op-inventory-and-config.md` | All unique TTNN ops with configs, sharding, suboptimal patterns |
| `09-implementation-plan.md` | This file — concrete fixes with file locations and test commands |
