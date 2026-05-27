# BEV Model — Complete Fix & Feature Summary

## Overview

All work targets the BEV (Bird's Eye View) model split into 6 logical blocks:
- Block A — `block_A_deformed_backbone`
- Block B — `block_B_camera_deformed_cylinder_bev_transform`
- Block C — `block_C_cylinder_backbone`
- Block D — `block_D_camera_cylinder_bev_transform`
- Block E — `block_E_bev_aggregator`
- Block F — Output Heads

Fixes span three repositories: `tt-forge-onnx`, `tt-mlir`, and `tt-metal` (vendored inside `tt-mlir/third_party/tt-metal`).

---

## Feature Additions

### 1. Native `grid_sample` Op (end-to-end)

**Problem:** Each ONNX `GridSample` node was decomposed into ~362 TTIR primitives by TVM relay fallback. With ~40 GridSample ops in the full BEV model this produced 14,480+ MLIR ops, causing MLA pipeline hangs at opt_level_2.

**Solution:** Added full native `ttnn::grid_sample` support through the entire compiler stack — from ONNX ingestion in tt-forge-onnx down through tt-mlir dialects, flatbuffer serialization, and runtime execution on device.

**Key layout transformations:**
- Input: NCHW → NHWC (tt-metal `grid_sample` requires NHWC)
- Grid: TVM relay format `(N, 2, H_out, W_out)` → `(N, H_out, W_out, 2)` (tt-metal convention)
- Precomputed-grid path for `nearest` mode and `align_corners=True` (BF16 precision insufficient for large grids — precompute in float32 on host CPU)

**Files changed:**

| Repo | File | Change |
|------|------|--------|
| **tt-forge-onnx** | `forge/forge/op/resize.py` | `GridSample` frontend function |
| **tt-forge-onnx** | `forge/csrc/ops/op_grid_sample.cpp` | C++ eval / shape / backward |
| **tt-forge-onnx** | `forge/csrc/ops/op.cpp` | OpType dispatch table entries |
| **tt-forge-onnx** | `forge/csrc/ops/op_interface.hpp` | `DECLARE_OP_INTERFACE(grid_sample)` |
| **tt-forge-onnx** | `forge/csrc/ops/CMakeLists.txt` | Add new source file |
| **tt-forge-onnx** | `forge/csrc/ops/python_bindings.cpp` | Python binding |
| **tt-forge-onnx** | `forge/csrc/passes/lower_to_mlir.cpp` | Emit `ttir::GridSampleOp` |
| **tt-forge-onnx** | `forge/forge/tvm_calls/relay/op/forge_passes.py` | TVM relay `image.grid_sample` → Forge op |
| **tt-forge-onnx** | `forge/forge/tvm_to_python.py` | `populate_grid_sample_args` |
| **tt-mlir** | `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` | `TTIR_GridSampleOp` tablegen definition + verifier |
| **tt-mlir** | `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` | `TTNN_GridSampleOp` tablegen definition + verifier |
| **tt-mlir** | `lib/Dialect/TTIR/IR/TTIROps.cpp` | TTIR verifier implementation |
| **tt-mlir** | `lib/Dialect/TTNN/IR/TTNNOps.cpp` | TTNN verifier implementation |
| **tt-mlir** | `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` | TTIR→TTNN: NCHW→NHWC permute for input; N2HW→NHW2 permute for grid |
| **tt-mlir** | `lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp` | EmitC lowering |
| **tt-mlir** | `lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp` | EmitPy lowering |
| **tt-mlir** | `include/ttmlir/Target/TTNN/operations/pool.fbs` | Flatbuffer `GridSampleOp` table |
| **tt-mlir** | `include/ttmlir/Target/TTNN/program.fbs` | Register in `OpType` union |
| **tt-mlir** | `lib/Target/TTNN/TTNNToFlatbuffer.cpp` | `createOp` + `emitTTNNOperation` dispatch |
| **tt-mlir** | `runtime/lib/ttnn/operations/pool/grid_sample.cpp` | Runtime: precomputed-grid path for nearest / align_corners=True |
| **tt-mlir** | `runtime/lib/ttnn/operations/pool/grid_sample.h` | Runtime header |
| **tt-mlir** | `runtime/lib/ttnn/operations/CMakeLists.txt` | Add source file |
| **tt-mlir** | `runtime/lib/ttnn/program_executor.cpp` | Dispatch case |
| **tt-mlir** | `runtime/lib/ttnn/runtime.cpp` | `getOpOutputRef` / `getOpInputRefs` cases |
| **tt-mlir** | `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp` | Conditional BF16 workaround for grid operand |
| **tt-mlir** | `include/ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h` | Updated factory method declaration |
| **tt-mlir** | `lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp` | GridSampleOp returns `OpNotSupportedError` |
| **tt-mlir** | `lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp` | Layout fallback for GridSampleOp |
| **tt-mlir** | `lib/Dialect/TTNN/Analysis/L1InterleavedFallbackAnalysis.cpp` | Interleaved fallback for GridSampleOp |

**Result:** All GridSample configurations (bilinear, nearest; zeros, border, reflection padding; align_corners true/false) pass at opt_level_0/1/2. TTIR op count reduced from ~362 primitives per GridSample to 1.

---

## Bug Fixes

---

### Fix 2 — Block D: PermuteOp BLOCK_SHARDED Data Corruption (opt_level_2)

**Symptom:** Silent wrong output; 0/40 tests pass at opt_level_2. No crash or error — just incorrect inference results.

**Root cause:** At opt_level_2, `TTNNGreedyMemoryLayoutPropagation` assigned BLOCK_SHARDED L1 output layouts to `PermuteOp` (the NCHW↔NHWC permutations surrounding `grid_sample`). Inside `permute.cpp`, WH-involving permutations on sharded input decompose into `transpose_wh` + `transpose_hc` chains. The `transpose_wh` kernel's sharded path is gated on:

```cpp
bool input_height_sharded = is_sharded && is_l1 &&
    shard_spec.shape[1] == logical_shape[-1];  // full-width shard required
bool use_sharded_wh = input_height_sharded && !input_cn_sharded;
```

For BLOCK_SHARDED, `shard_spec.shape[1] < logical_shape[-1]` → `use_sharded_wh = false` → non-sharded `TransposeWHProgramFactory` runs on sharded memory → corrupted output.

The op model could not detect this because `QUERY_OP_CONSTRAINTS` builds the input tensor from the previous op's layout (DRAM interleaved), so `a.is_sharded()` is always false during the query and no exception is raised.

**Fix:**

| File | Layer | Change |
|------|-------|--------|
| `lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp` | tt-mlir compiler | Filter all sharded L1 layouts from PermuteOp candidates at compile time |
| `ttnn/.../permute/permute.cpp` | tt-metal runtime | Route WH-involving permutations with non-full-width shards through `prim_permute` |
| `ttnn/.../permute/device/permute_device_operation.cpp` | tt-metal runtime | Add `TT_FATAL(!is_sharded)` in `validate_on_program_cache_miss` |

**Result:** 0/40 → **40/40 pass** at opt_level_2.

---

### Fix 3 — Block C: ConvTranspose2d Compile Crash (opt_level_1/2)

**Symptom:**
```
TT_FATAL @ conv2d_op_program_factory_common.cpp:91:
  get_cb_info expects conv_config.weights_dtype to be already set
```
Affects all four ConvTranspose2d ops (in_ch=192, out_ch=192, kernel=2×2, stride=2×2) on inputs 1×192×20×36 and larger.

**Root cause:** `prepare_conv_transpose2d_weights.cpp` computes `weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype())` but never writes the result back. When the DRAM slice path subsequently calls `determine_slice_config → get_L1_usage → calculate_L1_usage → get_cb_info`, the `Conv2dConfig` still has `weights_dtype = nullopt` → fatal.

**Fix:**

| File | Repo | Change |
|------|------|--------|
| `ttnn/.../conv_transpose2d/prepare_conv_transpose2d_weights.cpp` | tt-metal (via tt-mlir) | Write `weight_dtype` back into `conv_config.weights_dtype` in the missing path |
| `lib/OpModel/TTNN/TTNNOpModel.cpp` (getOpConstraints) | tt-mlir | Set `conv2dConfigConverted->weights_dtype = weightSpec.data_type()` before `QUERY_OP_CONSTRAINTS` |
| `lib/OpModel/TTNN/TTNNOpModel.cpp` (getPrepareConv2dWeightsOpOutputTensorSpec) | tt-mlir | When `transpose=true`, set `conv2dConfigConverted->weights_dtype` from `outputDtype` |

**Result:** All four ConvTranspose2d ops pass at opt_level_0/1/2. Block C opt_level_1: FAIL → **PASS**.

---

### Fix 4 — Block C: Conv2d L1 Circular Buffer Clash at Runtime (opt_level_2)

**Symptom:**
```
TT_THROW @ program.cpp:1366:
Statically allocated circular buffers in program 60394 clash with L1 buffers
on core range [0-0 - 7-6].
L1 buffer allocated at 171520 and static circular buffer region ends at 177440.
```
Call stack: `conv2d_L1 → prim::matmul → validate_circular_buffer_region → TT_THROW`

Does not reproduce at opt_level_1 (Conv2d uses DRAM at that level).

**Root cause — The Dead Zone:**

The L1SpillManagement simulation uses virtual addresses `[0, l1BudgetPerCore]` where virtual 0 maps to real address `l1_size − l1BudgetPerCore = 173,484`. Circular buffers physically start at `l1_unreserved_base = 103,712`. The region `[103,712 – 173,484)` (69,772 bytes) is the **dead zone** — below the simulation floor, invisible to the tracker.

The failing Conv2d had `cbPeakUsage = 73,728 bytes > dead zone (69,772 bytes)`, meaning its CB extended 3,956 bytes into untrackable territory. At runtime, actual L1 usage exceeded the budget by 1,964 bytes (due to lazy deallocation and program cache overhead), placing a tensor at real address 171,520 — inside the CB region ending at 177,440. The simulation's `wouldCBsOverlapTensors` check could not see this tensor (it was below the simulation floor) and did not demote the op.

**Fix:**

| File | Change |
|------|--------|
| `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h` | Add `usableL1Size` constructor parameter; add `l1DeadZone` member |
| `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp` | Compute `l1DeadZone = usableL1Size − l1BudgetPerCore`; add dead zone pre-check in `ensureFitsL1`: if `cbPeakUsage > l1DeadZone`, forcibly demote op to DRAM |
| `lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyL1SpillManagement.cpp` | Pass `chipDesc.getUsableL1Size()` to constructor |

```
Hardware constants (WH N150):
  l1_unreserved_base = 103,712 B
  l1_size            = 1,499,136 B
  usableL1Size       = 1,395,424 B
  l1BudgetPerCore    = 0.95 × 1,395,424 = 1,325,652 B
  l1DeadZone         = 1,395,424 − 1,325,652 = 69,772 B
```

**Result:** Block C opt_level_2: FAIL (CB clash) → **PASS**.

---

### Fix 5 — Block E: Bilinear Upsample Segfault (opt_level_2)

**Symptom:**
```
Fatal Python error: Segmentation fault
stacktrace:
 --- ttnn::operations::sliding_window::generate_halo_kernel_config_tensors(...)
 --- ttnn::prim::UntilizeWithHaloProgramFactory::create(...)
 --- ttnn::operations::upsample::upsample(...)
 --- mlir::tt::ttnn::op_model::OpModel<mlir::tt::ttnn::UpsampleOp>::getOpConstraints(...)
 --- mlir::tt::ttnn::MemoryLayoutPropagation::evaluateHint(...)
```
Occurs during MLA (Memory Layout Analysis), not at runtime. Does not reproduce at opt_level_1.

Block E has three bilinear upsample ops on inputs `1×16×8×256`, `1×32×16×128`, and `1×64×32×64`.

**Root cause:** MLA proposes HEIGHT_SHARDED L1 with an arbitrary shard spec and passes it to `QUERY_OP_CONSTRAINTS(::ttnn::upsample, ...)`. Inside `ttnn::upsample`, if the input is already sharded, the autoresharding step is skipped and `apply_bilinear_halo_preprocessing` runs with the wrong shard spec. `generate_halo_kernel_config_tensors` then accesses `tensor_metadata[global_idx]` where `global_idx` exceeds `tensor_metadata.size()` — out-of-bounds → segfault.

The autoshard formula (`compute_bilinear_autoshard_memory_config`):
```
total_nhw    = N × H × W
num_shards   = min(grid.x × grid.y, total_nhw)
shard_height = round_up(total_nhw, num_shards) / num_shards
shard_spec   = {shard_height, C}, HEIGHT_SHARDED, L1
```

| Upsample op | Input shape | shard_height | shard_spec |
|-------------|-------------|--------------|------------|
| #1 | `1×16×8×256` | 2 | `[2, 256]` |
| #2 | `1×32×16×128` | 8 | `[8, 128]` |
| #3 | `1×64×32×64` | 32 | `[32, 64]` |

**Fix (two stages):**

**Stage 1 — Reject sharded L1 (stop the segfault):** Early-return guard rejecting all sharded L1 for bilinear upsample → falls back to DRAM interleaved. Result: 116.58 FPS.

**Stage 2 — Recompute autoshard spec in OpModel (enable L1 sharded path):**

| Step | Change |
|------|--------|
| Reject non-HEIGHT_SHARDED | BLOCK_SHARDED / WIDTH_SHARDED → return error (metal `TT_FATAL`) |
| `computeBilinearAutoshardMemoryConfig` | Mirror autoshard formula to produce exact MemoryConfig the kernel expects |
| `buildBilinearAutoshardInputSpec` | Build TensorSpec with ROW_MAJOR page layout (TILE layout forces tile-aligned shard shapes; autoshard shard_height can be as small as 2) |
| Use corrected spec | Replace MLA's arbitrary TensorSpec with autoshard spec before `QUERY_OP_CONSTRAINTS` |

**File:** `lib/OpModel/TTNN/TTNNOpModel.cpp`

**Result:**

| Test | Before | Stage 1 | Stage 2 |
|------|--------|---------|---------|
| `opt_level_1-block_E` | PASS | PASS | PASS (132.80 FPS) |
| `opt_level_2-block_E` | **FAIL (segfault)** | PASS (116.58 FPS, DRAM) | **PASS (120.93 FPS, L1 sharded)** |

---

### Fix 6 — Block A: Conv2d L1 Fragmentation OOM (opt_level_2)

**Symptom:**
```
Not enough space to allocate 37748736 B L1 buffer across 64 banks,
where each bank needs to store 589824 B, but bank size is 1329888 B
(allocated: 606208 B, free: 723680 B, largest free block: 420576 B)
```
The simulation believed there was enough room; at runtime only 420,576 bytes of contiguous space was available.

**Root cause — Simulation-vs-runtime mismatch for ToLayoutOp outputs:**

By design, `ToLayoutOp` outputs are NOT added to `liveValues` in the `SumL1MemoryTracker` (they are "short-lived L1 tenants"). `ensureFitsL1` is called for them (to make room) but they are never tracked by `memoryTracker`. This creates an invisible fragmentation problem at runtime.

In Block A, three `ToLayoutOp` outputs (303,104 bytes/core each) are allocated in L1, the middle one is freed, leaving the heap fragmented:

```
L1 bank: 1,329,888 B

T1 = [1,026,784 – 1,329,888)   303,104 B  ← live
T2 = [  723,680 – 1,026,784)   303,104 B  ← freed (hole)
T3 = [  420,576 –   723,680)   303,104 B  ← live
free = [0 – 420,576)            420,576 B  ← max contiguous

Conv2d needs 589,824 B → OOM (420,576 < 589,824)
```

The simulation sees `getOccupiedL1() = 0` because:
1. `processDeadTensors(pos)` frees the conv2d input (also a ToLayoutOp result) before `ensureFitsL1` is called for the conv2d output
2. The three ToLayoutOp outputs (T1, T2, T3) were never added to `memoryTracker`

So the simulation concludes "589,824 bytes fits (budget = 1,325,652, occupied = 0)" → keeps conv2d output in L1 → OOM at runtime.

**Fix — Large-tensor fragmentation guard in `ensureFitsL1`:**

```cpp
static constexpr double kMaxSingleTensorFraction = 0.40;
if (l1Size > static_cast<uint64_t>(kMaxSingleTensorFraction *
                                   static_cast<double>(l1BudgetPerCore))) {
  // Evict live L1 inputs and demote to DRAM
  llvm::SmallVector<Value> toEvict;
  for (Value operand : op->getOperands())
    if (liveValues.count(operand)) toEvict.push_back(operand);
  for (Value victim : toEvict)
    evictValue(victim, pos, data);
  demoteToDram(op);
  evictForDramCBGrowth(op, pos, data);
  return 0;
}
```

**Threshold rationale (WH N150):**

| Tensor | Size/core | Fraction of budget | Guard fires? |
|--------|-----------|-------------------|--------------|
| Block A conv2d output | 589,824 B | 44.5% | **YES → DRAM** |
| ToLayoutOp outputs | 303,104 B | 22.9% | No → stay in L1 |

40% threshold (530,261 B) rejects the problematic conv2d output while leaving smaller tensors in L1.

**File:** `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`

**Result:** Block A opt_level_2: FAIL (L1 OOM) → **PASS**.

---

## Consolidated File Change Table

### tt-forge-onnx

| File | Change | Feature/Fix |
|------|--------|-------------|
| `forge/forge/op/resize.py` | `GridSample` frontend function | Feature 1 |
| `forge/csrc/ops/op_grid_sample.cpp` | C++ eval / shape / backward | Feature 1 |
| `forge/csrc/ops/op.cpp` | OpType dispatch entries | Feature 1 |
| `forge/csrc/ops/op_interface.hpp` | `DECLARE_OP_INTERFACE(grid_sample)` | Feature 1 |
| `forge/csrc/ops/CMakeLists.txt` | Add source file | Feature 1 |
| `forge/csrc/ops/python_bindings.cpp` | Python binding | Feature 1 |
| `forge/csrc/passes/lower_to_mlir.cpp` | Emit `ttir::GridSampleOp` | Feature 1 |
| `forge/forge/tvm_calls/relay/op/forge_passes.py` | TVM relay → Forge op | Feature 1 |
| `forge/forge/tvm_to_python.py` | `populate_grid_sample_args` | Feature 1 |

### tt-mlir

| File | Change | Feature/Fix |
|------|--------|-------------|
| `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` | `TTIR_GridSampleOp` definition | Feature 1 |
| `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` | `TTNN_GridSampleOp` definition | Feature 1 |
| `lib/Dialect/TTIR/IR/TTIROps.cpp` | TTIR verifier | Feature 1 |
| `lib/Dialect/TTNN/IR/TTNNOps.cpp` | TTNN verifier | Feature 1 |
| `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` | TTIR→TTNN with NCHW↔NHWC + N2HW↔NHW2 permutes | Feature 1 |
| `lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp` | EmitC lowering | Feature 1 |
| `lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp` | EmitPy lowering | Feature 1 |
| `include/ttmlir/Target/TTNN/operations/pool.fbs` | Flatbuffer table | Feature 1 |
| `include/ttmlir/Target/TTNN/program.fbs` | OpType union registration | Feature 1 |
| `lib/Target/TTNN/TTNNToFlatbuffer.cpp` | Serialization dispatch | Feature 1 |
| `runtime/lib/ttnn/operations/pool/grid_sample.cpp` | Runtime with precomputed-grid path | Feature 1 |
| `runtime/lib/ttnn/operations/pool/grid_sample.h` | Runtime header | Feature 1 |
| `runtime/lib/ttnn/operations/CMakeLists.txt` | Add source | Feature 1 |
| `runtime/lib/ttnn/program_executor.cpp` | Dispatch case | Feature 1 |
| `runtime/lib/ttnn/runtime.cpp` | `getOpOutputRef` / `getOpInputRefs` | Feature 1 |
| `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp` | BF16 workaround for grid operand | Feature 1 |
| `include/ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h` | Factory method declaration | Feature 1 |
| `lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp` | `OpNotSupportedError` for GridSampleOp | Feature 1 |
| `lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp` | Layout fallback + sharded L1 filter for PermuteOp | Feature 1, Fix 2 |
| `lib/Dialect/TTNN/Analysis/L1InterleavedFallbackAnalysis.cpp` | Interleaved fallback for GridSampleOp | Feature 1 |
| `lib/OpModel/TTNN/TTNNOpModel.cpp` | ConvTranspose2d `weights_dtype` guards; bilinear upsample autoshard recompute | Fix 3, Fix 5 |
| `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h` | `usableL1Size` param; `l1DeadZone` member | Fix 4 |
| `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp` | Dead zone pre-check; large-tensor fragmentation guard in `ensureFitsL1` | Fix 4, Fix 6 |
| `lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyL1SpillManagement.cpp` | Pass `usableL1Size` to constructor | Fix 4 |

### tt-metal (vendored inside `third_party/tt-mlir/third_party/tt-metal`)

| File | Change | Feature/Fix |
|------|--------|-------------|
| `ttnn/.../permute/permute.cpp` | Route WH-involving + non-full-width-shard through `prim_permute` | Fix 2 |
| `ttnn/.../permute/device/permute_device_operation.cpp` | `TT_FATAL(!is_sharded)` in `validate_on_program_cache_miss` | Fix 2 |
| `ttnn/.../conv_transpose2d/prepare_conv_transpose2d_weights.cpp` | Write `weight_dtype` back into `conv_config.weights_dtype` | Fix 3 |

---

## Test Results Summary

| Block | Model | opt_level_0 | opt_level_1 | opt_level_2 | Primary Fix(es) |
|-------|-------|:-----------:|:-----------:|:-----------:|-----------------|
| Block A | `block_A_deformed_backbone` | PASS | PASS | PASS ✓ | Fix 6 (L1 fragmentation OOM) |
| Block C | `block_C_cylinder_backbone` | PASS | PASS ✓ | PASS ✓ | Fix 3 (ConvT2d compile) + Fix 4 (CB clash) |
| Block D | `block_D_camera_cylinder_bev_transform` | PASS | PASS | PASS ✓ | Fix 2 (PermuteOp corruption) |
| Block E | `block_E_bev_aggregator` | PASS | PASS | PASS ✓ | Fix 5 (upsample segfault) |
| All blocks | GridSample (all) | PASS ✓ | PASS ✓ | PASS ✓ | Feature 1 (native grid_sample) |

✓ = was failing before fix
