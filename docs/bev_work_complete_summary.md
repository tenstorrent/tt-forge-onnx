# BEV Model — Complete Work Summary

**Model:** Autonomous driving BEV (Bird's Eye View) deformed backbone  
**Hardware:** WH N150, `opt_level_2 · bfloat16 · HiFi3 · fp32_dest_acc · trace_enabled`  
**Repos:** `tt-forge-onnx` (3 commits) · `tt-mlir` (13 commits) · tt-metal (no kernel changes)

---

## Repositories and Branch Status

| Repo | Branch | Commits ahead of origin |
|------|--------|------------------------|
| `tt-forge-onnx` | `main` | 3 |
| `tt-mlir` | `pchandrasekaran/grid_sample` | 13 |
| `tt-metal` | upstream | 0 (no changes needed) |

---

## tt-forge-onnx Commits

### `3126f4af` — [tt-mlir] MaxPool2d BLOCK_SHARDED spill fix + SliceStaticOp HS-RM support
Bumps `third_party/tt-mlir` submodule pointer to `6d9ab8778`. Adds
`docs/maxpool2d_block_sharded_spill_fix.md`.

### `aaa434c3` — [forge] Add native GridSample op support for BEV models
Adds `GridSampleOp` as a first-class native op throughout the tt-forge-onnx stack,
replacing the TVM relay decomposition path (`DecomposeGridSample` removed):
- `forge/csrc/ops/op_grid_sample.cpp` (new): `eval()`, `shape()`, `attrs()` implementation
- `forge/csrc/ops/op.hpp/cpp`: GridSample enum + type-string registration
- `forge/csrc/ops/op_interface.hpp`, `python_bindings.cpp`: interface + Python binding
- `forge/csrc/ops/CMakeLists.txt`: adds op_grid_sample.cpp to build
- `forge/csrc/passes/lower_to_mlir.cpp`: `grid_sample` → `GridSampleOp` lowering handler
- `forge/forge/op/resize.py`: `GridSample()` Python API
- `forge/forge/tvm_to_python.py`: `populate_grid_sample_args()` relay→forge mapping
- `forge/forge/tvm_calls/relay/op/forge_passes.py`: removes `DecomposeGridSample()`
- `forge/csrc/passes/constant_folding.cpp`: `calculate_and_set_node_shape` before `consteval` (shape computation bug found during BEV work)
- `forge/csrc/passes/fuse_per_channel_ops.cpp`: remove `orig_shape_len` from Unsqueeze TM, calculate shape after `const_concat` (per-channel fuse bug found during BEV work)
- `forge/csrc/passes/mlir_compiler.cpp`: `TTMLIR_DUMP_PIPELINE_IR` env var for TTIR/TTNN diagnostic IR dumps to files
- `forge/test/mlir/test_ops_onnx.py`: GridSample ONNX op tests
- `forge/test/models/onnx/vision/bev/`: BEV benchmark, block tests, model split utilities

### `cba781a0` — [docs] BEV model optimization documentation + tt-mlir submodule update
Commits all BEV documentation (56 files) and bumps `third_party/tt-mlir` to the
final `613876801` (SliceStaticOp HS RM final fix).

---

## tt-mlir Commits

### `088021b2` — Add native grid_sample op support to tt-mlir
_(27 files, 691 insertions)_

**Background:** `grid_sample` had no native dialect ops. The TVM relay pass
`DecomposeGridSample` expanded each `image.grid_sample` into ~362 TTIR primitives.
With ~40 grid_sample ops in the BEV model, this produced ~14,480 TTIR ops from
grid_sample alone, causing the MLA (Memory Layout Analysis) pass to hang indefinitely
at opt_level_2.

**What was added across the stack:**
- `TTIR_GridSampleOp`: inputs `(N,C,H_in,W_in)` + grid `(N,2,H_out,W_out)`, `mode`/`padding_mode`/`align_corners` attrs, verifier
- `TTNN_GridSampleOp`: inputs `(N,H_in,W_in,C)` NHWC + grid `(N,H_out,W_out,2)`, `getOperandsWorkarounds` forces both inputs to ROW_MAJOR and grid to float32 (precomputed path) or BF16 (direct path)
- TTIR→TTNN conversion: permutes input NCHW→NHWC and grid `(N,2,Ho,Wo)`→`(N,Ho,Wo,2)` before calling `ttnn::grid_sample`
- Op model: `GridSampleOp::getOpConstraints` / `getOpRuntime` (from `6cc1beca6`)
- Batched fusion pass: `GridSampleBatchedFusePass` (from `9d67270ff`)

**Key format requirements enforced:**
- `tt-metal grid_sample` kernel: input must be NHWC ROW_MAJOR BF16; grid must be `(N,H_out,W_out,2)` ROW_MAJOR
- Channel count `C` must be divisible by 32 (kernel packs channels into hardware tiles)
- `align_corners=True`: use precomputed grid path (host computes pixel coords in float32 to avoid kernel's hardcoded `align_corners=False` formula)

---

### `a168c1971` — [runtime] GridSample: cache precomputed grid in ProgramContext to fix trace capture crash
_(2 files, 64 insertions, 27 deletions)_

During trace capture, the `ProgramContext` was re-creating the precomputed grid
tensor on every call, causing a crash when the trace tried to capture a tensor
that was re-allocated each time. Fixed by caching the precomputed grid in
`ProgramContext` keyed by op signature so it is stable across trace iterations.

---

### `435c17e3` — [opmodel] ConvTranspose2d: set weights_dtype in Conv2dConfig before kernel query
_(1 file, 40 insertions)_

`ConvTranspose2dOp` op model was querying the kernel with an empty `weights_dtype`
when the model used BF16 weights, causing the kernel to assume float32 → OOM.
Fixed by reading `weights_dtype` from the op's weight tensor type before the
kernel query.

**Impact:** Fixed BEV Block C `conv_transpose2d` OOM/hang at opt_level_2.

---

### `884ef96e3` — [opmodel] UpsampleOp: reject sharded L1 layouts for bilinear mode
_(1 file, 58 insertions, 8 deletions)_

`tt-metal ttnn::upsample` bilinear kernel does not support sharded L1 inputs. The
op model was allowing sharded candidates, causing a segfault at compile time.
Added early rejection of L1 sharded configs for bilinear UpsampleOp.

**Impact:** Fixed BEV Block B bilinear upsample segfault at opt_level_2.

---

### `2743a3e01` — [L1SpillManagement] Fix L1 CB dead zone clash for large-CB ops
_(3 files, 116 insertions, 7 deletions)_

When a large-CB op (e.g. 3×3 conv2d with HiFi3+fp32_acc → 400+ KB CBs) preceded
a live L1 tensor, `wouldCBsOverlapTensors` detected a real clash but
`ensureFitsL1` did not evict the conflicting tensor, leading to OOM or wrong results
at runtime. Extended `evictForCBOverlap` to correctly evict low-virtual-address
(low-physical) tensors that physically overlap with the CB zone.

---

### `6adf40b31` — [L1SpillManagement] Guard against L1 fragmentation OOM for large tensors
_(2 files, 93 insertions, 8 deletions)_

When L1 was heavily fragmented, allocating a large tensor could fail even when
total free L1 was sufficient. Added a guard that checks for fragmented-OOM before
applying the Belady eviction path, so ops are demoted to DRAM instead of hanging.

---

### `6cc1beca6` — [opmodel] GridSampleOp: implement getOpConstraints and getOpRuntime
_(3 files, 98 insertions, 8 deletions)_

Wired `GridSampleOp` into the TTNN op model, allowing `GreedyMemoryLayoutPropagation`
to correctly size CBs and estimate runtime for grid_sample in the optimization loop.

---

### `c0059473f` — [runtime] system_desc: clarify compute_with_storage_grid_size comment
_(minor documentation/comment fix)_

---

### `9d67270ff` — [ttir][ttnn] GridSample batched fusion for multi-camera BEV models
_(13 files, 204 insertions, 25 deletions)_

The BEV model repeats 8 per-camera `grid_sample` ops (one per camera) followed by
a `concat`. Each camera uses a different grid but the same input tensor. Running 8
separate grid_sample ops serially wastes execution bandwidth.

**Fix:** `GridSampleBatchedFusePass` in TTIR and TTNN:
- Pattern: N identical `grid_sample` ops on the same input with per-camera grids + `concat`
- Transform: concatenate grids along batch dim → single `grid_sample(input, batched_grid, batch_output_channels=True)` → slice outputs back
- The `batch_output_channels=True` flag tells the kernel to treat each "batch" as a separate camera, outputting `(N*cameras, H_out, W_out, C)` which is then split

**Impact:** Block B FPS 9.31 → 11.94 (+28%).

---

### `bd216ed5e` — [ttnn] Add ToMemoryConfigOp canonicalization to fold bounce spills
_(4 files, 183 insertions)_

`OperationValidationAndFallback` and `TTNNDecomposeLayouts` insert `to_memory_config`
chains that create redundant DRAM hops with no compute value:
```
Pattern 1 (bounce):    L1_sharded → DRAM → L1_sharded
Pattern 2 (bypass):    L1_sharded → DRAM → L1_interleaved  (consumer could read L1 directly)
```

**Three patterns added:**
1. `ToMemoryConfigOp::fold`: identity fold when input/output types are identical
2. `FoldConsecutiveToMemoryConfigOps`: collapses two chained `to_memory_config` ops
   when the intermediate is L1-sharded with no other compute users
3. `BypassDRAMForL1InterleavedConsumers`: collects ALL L1-interleaved consumers of a
   DRAM op and reroutes them to read the L1-sharded source directly; erases DRAM op
   when no compute users remain

**Two canonicalizer passes added to pipeline** (`TTNNPipelines.cpp`):
- After `OperationValidationAndFallback` — folds bounce spills from fallback
- After `TTNNDecomposeLayouts` — folds L1_sharded→DRAM→L1_interleaved patterns from decomposition

**Impact:** `total_ops` 1198→962 (−236 ops in Block A IR). No FPS change under trace
(tensor aliasing prevents remat), but IR cleanliness improves significantly.

---

### `bcb8cad73` — [L1SpillManagement] Fix false-positive CB_ZONE_EVICT, add tryReduceConv2dActBlockH
_(3 files, 460 insertions, 38 deletions)_

**CB_ZONE_EVICT fix:**
The `CB_ZONE_EVICT` block in `ensureFitsL1` evicted tensors with high virtual
addresses, based on an assumption that TTNN allocates L1 bottom-up (low-physical
first). `tt_metal/impl/buffers/buffer.cpp:289` proves this wrong:
```cpp
bottom_up_(bottom_up.value_or(this->is_dram()))
// L1: is_dram()==false → bottom_up=false → TOP-DOWN allocation
```
Both the simulator and tt-metal allocate top-down, so the virtual→physical mapping
is same-direction (`physical = l1_unreserved_base + virtual`). High-virtual tensors
are at HIGH physical addresses — far from the CB zone (which grows bottom-up from
`l1_unreserved_base`). The loop was systematically evicting the safest tensors
(56 false evictions per Block A compile).

**Fix:** Removed the `CB_ZONE_EVICT` block and `tryReduceConv2dCBForZoneEvict` helper
entirely (~153 lines). The existing `wouldCBsOverlapTensors` → `evictForCBOverlap`
path correctly targets low-virtual (= low-physical = dangerous) tensors.

**tryReduceConv2dActBlockH (OOM recovery):**
Added in the `handleOOM` path: before DRAM demotion, tries `act_block_h` values
{1024, 992, …, 64, 32} for L1Full Conv2d ops. A smaller `act_block_h` reduces CB
peak at marginal throughput cost but keeps the op in L1, avoiding the DRAM round-trip.

**Impact:** `effectively_sharded_ops` 100→208 (+2×), `sharded_and_spilled_ops` 108→24 (−78%).

---

### `6d9ab8778` — [TTNN optimizer] MaxPool2d BLOCK_SHARDED spill fix + SliceStaticOp HS-RM support

**MaxPool2d fix:**
MaxPool2d ops inheriting BLOCK_SHARDED output from an upstream Conv2d (via NULL hint)
were immediately spilled to DRAM because `SliceRmShardedWidthTrimProgramFactory`
only accepts HEIGHT_SHARDED RM or DRAM inputs.

Added `MaxPool2dRuleBook`: omits the NULL hint (prevents BLOCK_SHARDED inheritance),
offers DRAM/interleaved as primary and HEIGHT_SHARDED as fallback.

**SliceStaticOp HS RM (intermediate version, superseded by `613876801`):**
Initial attempt to offer HEIGHT_SHARDED RM output for last-dim-only slices —
contained an approach that was later refined.

**Impact:** `sharded_and_spilled_ops` 12→0 across all op types; `dram_spilled_ops` 108→96.

---

### `613876801` — [TTNN optimizer] SliceStaticOp HS RM output: HS TILED marker hint approach
_(3 files, 186 insertions, 11 deletions)_

**Background:** `SliceRmShardedWidthTrimProgramFactory` supports two output paths:
`output_is_dram = !output.is_sharded()`. The L1 HS RM path trims each shard
in-place into a globally-allocated output CB. No cross-core NOC reads.

**Two prior attempts failed:**
1. Search `legalConfigs` for HS RM entries → always empty (`rowMajorEnabled=false`
   causes `LegalOpLayoutAnalysis::fillTTNNLayoutAttrs` to skip the RM page layout loop)
2. Read `op->getOperand(0).getType()` for HS RM hint → reflects pre-MLP DRAM default,
   not MaxPool2d's beam-committed HS RM (`TTNNRowMajorLayoutPropagation` is a no-op
   for BF16 inputs — only propagates from integer-type function inputs)

**Final fix:** Find the first HEIGHT_SHARDED **TILED** entry in `legalConfigs`, convert
to RM via `.setLayout(Layout::RowMajor)`. Inherits a valid `coreRangeSet` from the
TILED template (required by `TTNNLayoutAttr::build()` for sharded layouts). The op
model's `outputIsHS` bypass (`TTNNOpModel.cpp`) builds the actual output grid from the
**INPUT** layout, ignoring the hint's grid — the marker hint only needs to carry the
HS + RowMajor flags.

`isValidOutputHintForInputs` gates the HS hint to HEIGHT_SHARDED ROW_MAJOR input
candidates (MaxPool2d beam state). DRAM inputs fall through to the NULL fallback.

**Known limitation (outer concat):** The outer concat after the slice still reshards to
DRAM because `ConcatRuleBook::isValidOutputHintForInputs` requires hint grid == input
grid (58×1), and `legalConfigs` has no 58×1 HS output hint. The solver adds a
`to_memory_config(DRAM→L1 HS RM)` before the downstream Conv2d, which is faster than
the Conv2d reading from DRAM directly.

**Impact:** `dram_spilled_ops` 96→24 (−75%); FPS 2.54→2.77 (+9%).

---

## tt-metal — No Changes Required

All tt-metal kernel changes were **not needed** — the `ttnn::grid_sample` kernel was
already present and correct. The work was purely in the compiler stack (tt-mlir,
tt-forge-onnx) to properly interface with the kernel.

**Key tt-metal kernel constraints handled by the compiler:**
- Input: NHWC ROW_MAJOR BF16; Grid: `(N, H_out, W_out, 2)` ROW_MAJOR
- Channel count `C` must be divisible by 32
- `align_corners=True` → must use precomputed grid path (host computes coords in float32 to bypass kernel's hardcoded `align_corners=False` formula)
- Grid dtype: BF16 for direct bilinear path, float32 for precomputed path

The PermuteOp sharding corruption fix (which affected the NCHW→NHWC permute before
`grid_sample`) was handled in tt-mlir by adding `PermuteRuleBook` restrictions that
prevent BLOCK_SHARDED and HEIGHT_SHARDED output for WH-involving permutations (these
layouts cause silent data corruption in `permute.cpp`'s `transpose_wh/hc` chain).

---

## Current Status

### What is Working

| Area | Status |
|------|--------|
| BEV GridSample — all 120 test cases at opt_level_2 | ✅ PASSING |
| Block A L1 sharding (effectively sharded %) | ✅ 24.5% → 51.0% (+26.5 pp) |
| Block A `sharded_and_spilled_ops` | ✅ 112 → 24 (−79%) |
| Block A `to_memory_config` ops | ✅ 324 → 120 (−63%) |
| Block A FPS | ✅ 2.77 (within noise of 2.79 baseline) |
| Block A PCC validation | ✅ > 0.99 |
| ConvTranspose2d at opt_level_2 (Block C) | ✅ Fixed |
| Bilinear upsample at opt_level_2 (Block B) | ✅ Fixed |
| GridSample trace capture crash | ✅ Fixed |
| ToMemoryConfigOp bounce spill fold | ✅ Committed, active |

### What Did Not Work / Investigations Closed

| Area | Outcome |
|------|---------|
| Bounce spill fold → FPS improvement | IR shrinks by 236 ops but trace tensor aliasing prevents remat; no FPS gain under `trace=True` |
| `act_block_h` reduction for 72 DRAM-spilled Conv2d | Root cause is Belady eviction by downstream ops, not OOM in the conv2d itself — `tryReduceConv2dActBlockH` fires in `handleOOM` but those 72 ops don't go through `handleOOM` |
| Outer concat HS RM output | `ConcatRuleBook::isValidOutputHintForInputs` requires hint grid == input grid (58×1); no legalConfigs entry has 58×1 — solver falls back to DRAM. FPS still improves because downstream Conv2d now reads L1 via prefetch. Fix 14 opportunity. |

### Known Remaining Opportunities

| Opportunity | Estimated Impact |
|------------|-----------------|
| **Fix 14**: Extend `ConcatRuleBook::getOutputHints` to offer input-grid-derived HS hint (same marker approach as Fix 13) | Eliminate 4×3 = 12 `to_memory_config` ops, outer concat stays in L1 |
| **Act_block_h in Belady path**: Invoke `tryReduceConv2dActBlockH` from inside `evictUntil` before selecting Belady victim | Could keep 32–72 Conv2d ops in L1 instead of spilling under memory pressure |
| **72 DRAM-checkpoint Conv2d**: Root cause is `conv_transpose2d` (no L1Full config) triggering Belady eviction of preceding conv2d outputs | Requires `conv_transpose2d` L1-sharded kernel support in tt-metal (hardware constraint) |

---

## Block A Metrics: Baseline vs Final

Config: `opt_level_2 · bfloat16 · HiFi3 · fp32_dest_acc · trace_enabled`  
Source: `BEV_MODEL_LOGS/LATEST/block_A_deformed_backbone_perf_metrics.json` (baseline)  
and `BEV_MODEL_LOGS/LATEST/block_A_deformed_backbone_AFTER_FIX_12_perf_metrics.json` (after all fixes)

| Metric | Baseline | After All Fixes | Delta |
|--------|:--------:|:---------------:|:-----:|
| `effectively_sharded_ops` | 100 | **204** | **+104 (+2×)** |
| `effectively_sharded_%` | 24.5% | **51.0%** | **+26.5 pp** |
| `sharded_and_spilled_ops` | 112 | **24** | **−88 (−79%)** |
| `sharded_ops` | 212 | **228** | +16 |
| `total_shardable_ops` | 408 | **400** | −8 |
| `ttnn.to_memory_config` ops | 324 | **120** | **−204 (−63%)** |
| `ttnn.to_layout` ops | 24 | **16** | −8 |
| `ttnn.deallocate` ops | 700 | **420** | −280 |
| FPS | 2.79 | **2.77** | within noise |
