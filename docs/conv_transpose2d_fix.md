# ConvTranspose2d opt_level_2 Fix — Block C BEV

## Problem Statement

`test_opt_sweep[enable_program_cache-opt_level_2-block_C]` (and opt_level_1) failed
for `block_C_cylinder_backbone` due to ConvTranspose2d operations.

Block C has four ConvTranspose2d ops with shared config:
- weight: in_ch=192, out_ch=192, kernel=2×2, stride=2×2, pad=0, dilation=1, groups=1, no bias, f32

| Op  | Input shape       | Output shape      |
|-----|-------------------|-------------------|
| op0 | 1×192×10×18       | 1×192×20×36       |
| op1 | 1×192×20×36       | 1×192×40×72       |
| op2 | 1×192×40×72       | 1×192×80×144      |
| op3 | 1×192×80×144      | 1×192×160×288     |

op0 passed; op1–op3 failed at opt_level_1 and opt_level_2.

---

## Bug 1 — ConvTranspose2d compilation failure (FIXED)

### Symptom

```
TT_FATAL @ conv2d_op_program_factory_common.cpp:91:
  get_cb_info expects conv_config.weights_dtype to be already set
```

### Root Cause

`prepare_conv_transpose2d_weights.cpp` computes `weight_dtype` from
`conv_config.weights_dtype.value_or(weight_tensor.dtype())` but does NOT write
the result back into `conv_config.weights_dtype`. When the DRAM path calls
`determine_slice_config → get_L1_usage → calculate_L1_usage → get_cb_info`,
the `Conv2dConfig` stored in `ConvT2DSliceAttr` still has `weights_dtype =
nullopt`, triggering the fatal.

Both `conv_transpose2d_L1` (lines 123-125) and `conv_transpose2d_DRAM`
(lines 1029-1031) already set `weights_dtype` early — `prepare_conv_transpose2d_weights`
was the only path that didn't.

### Fix 1 — `prepare_conv_transpose2d_weights.cpp`

File: `third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.cpp`

After computing `weight_dtype`, ensure `conv_config.weights_dtype` is set before
it is captured by the slice-config helper:

```cpp
DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype());
// Ensure weights_dtype is set so get_cb_info doesn't fatal in the DRAM path.
if (!conv_config.weights_dtype.has_value()) {
    conv_config.weights_dtype = weight_dtype;
}
```

Compiled into `_ttnncpp.so` (rebuild via `ninja _ttnncpp.so` in the tt-metal
`build_Release` dir, then copy to `third_party/tt-mlir/build/install/lib/`).

### Fix 2 — `TTNNOpModel.cpp` (defense-in-depth, two locations)

File: `third_party/tt-mlir/lib/OpModel/TTNN/TTNNOpModel.cpp`

**`getOpConstraints` for `ConvTranspose2dOp`** (~line 5636):
Sets `conv2dConfigConverted->weights_dtype = weightSpec.data_type()` before
`QUERY_OP_CONSTRAINTS(::ttnn::conv_transpose2d, ...)` is called.

**`getPrepareConv2dWeightsOpOutputTensorSpec`** (~line 655):
When `transpose=true`, ensures `conv2dConfigConverted->weights_dtype` is set
from `outputDtype` before the query closure captures `conv2dConfigConverted`:

```cpp
if (transpose && outputDtype.has_value()) {
    if (!conv2dConfigConverted.has_value()) {
        conv2dConfigConverted = ::ttnn::Conv2dConfig{};
    }
    if (!conv2dConfigConverted->weights_dtype.has_value()) {
        conv2dConfigConverted->weights_dtype = *outputDtype;
    }
}
```

Compiled into `libTTMLIRCompiler.so`.

### Verification

Standalone test `forge/test/mlir/test_conv_transpose2d_block_c.py`:
- **12/12 PASS** (all 4 ops × opt_level_0/1/2)

`test_opt_sweep[enable_program_cache-opt_level_1-block_C]`: **PASS**

---

## Bug 2 — Conv2d L1 circular buffer clash at runtime (OPEN)

### Symptom

After fixing Bug 1, `test_opt_sweep[enable_program_cache-opt_level_2-block_C]`
fails during inference with:

```
TT_THROW @ program.cpp:1366:
Statically allocated circular buffers in program 60394 clash with L1 buffers
on core range [0-0 - 7-6].
L1 buffer allocated at 171520 and static circular buffer region ends at 177440
```

Call stack: `conv2d_L1 → prim::matmul → validate_circular_buffer_region → TT_THROW`

opt_level_1 does **not** reproduce this — the Conv2d uses DRAM at opt_level_1.

### Root Cause Analysis

At opt_level_2 the Memory Layout Analysis (MLA) pass places Conv2d input/output
tensors in L1. The op model uses `QUERY_OP_CONSTRAINTS` in `NO_DISPATCH` mode:
tensor memory allocations are captured, but the kernel dispatch step (including
`validate_circular_buffer_region`) is **not** executed. Consequently the op model
does not see the 177440-byte static CB region that the matmul kernel inside
`conv2d_L1` will require, and it returns `ExecutionStatus::Success` for the L1
placement.

At runtime:
1. MLA-assigned L1 tensor lands at address 171520.
2. `EnqueueMeshWorkload → compile → validate_circular_buffer_region` checks
   that all L1 buffers lie above `cb_region_end` (177440).
3. 171520 < 177440 → `TT_THROW`.

### Related Prior Fixes

| Commit    | Description |
|-----------|-------------|
| `8ad446616` | `TTNNDecomposeLayouts`: fix L1 CB clash when untilizing L1 interleaved → DRAM (moves to DRAM first, then untilizes in DRAM) |
| `86a81049e` | Skip L1 ops in `ConstEvalHoistTransform` to prevent unaccounted L1 pressure (const-eval hoisting bypassed `L1SpillManagement`) |
| `4ee1b7aef` | Set `conv2d_slice_config` to DRAM for L1 OOM fallbacks |
| `dedf1b8d1` | Enable DRAM slice fallback for `ttnn::ConvTranspose2dOp` |

### Investigation Paths

1. **`TTNNDecomposeLayouts` pattern** — does the clashing tensor at 171520 come
   from a layout-decomposition step (e.g., untilize before/after Conv2d)? Commit
   `8ad446616` already fixes the L1-interleaved→DRAM case; check if this Conv2d
   hits a decomposition variant not yet covered.

2. **`ConstEvalHoistTransform` pattern** — does a const-eval hoisted op produce
   an L1 tensor whose lifetime spans the Conv2d, causing hidden L1 pressure that
   `L1SpillManagement` didn't account for? (`86a81049e` address this class.)

3. **CB overhead estimate** — add a Conv2d CB-size estimate (based on kernel
   config) to the op model's reported L1 usage so MLA leaves sufficient headroom
   and falls back to DRAM automatically.

4. **Force DRAM for Conv2d ops whose shard would land inside the CB region** —
   analogous to `4ee1b7aef`/`dedf1b8d1` but triggered by CB-region overlap
   rather than OOM.

### Block C Conv2d Ops (candidates for investigation)

Large spatial Conv2d ops in the model that are most likely to stress L1:

| Layer | In shape | Out shape | Kernel |
|-------|----------|-----------|--------|
| `_alignment_convs.0` | 1×128×160×288 | 1×192×160×288 | 1×1 |
| `_yolov4tiny_blocks.0._convblock_5` | 1×160×160×288 | 1×128×160×288 | 3×3 |
| `_yolov4tiny_blocks.1._convblock_5` | 1×224×80×144 | 1×192×80×144 | 3×3 |
| `_alignment_convs.1` | 1×192×80×144 | 1×192×80×144 | 1×1 |

---

## File Summary

| File | Change | Status |
|------|--------|--------|
| `third_party/tt-metal/.../prepare_conv_transpose2d_weights.cpp` | Set `weights_dtype` in DRAM path | Applied, compiled |
| `third_party/tt-mlir/lib/OpModel/TTNN/TTNNOpModel.cpp` | Set `weights_dtype` in `getOpConstraints` and `getPrepareConv2dWeightsOpOutputTensorSpec` | Applied, compiled |
| `_ttnncpp.so` (install) | Contains prepare_conv_transpose2d_weights fix | Updated 2026-05-14 |
| `libTTMLIRCompiler.so` (install) | Contains TTNNOpModel fixes | Updated 2026-05-14 |

## Test Results

| Test | Before | After |
|------|--------|-------|
| `test_conv_transpose2d_block_c[op0_10x18-opt_level_0/1/2]` | PASS/PASS/PASS | PASS/PASS/PASS |
| `test_conv_transpose2d_block_c[op1_20x36-opt_level_0/1/2]` | PASS/FAIL/FAIL | PASS/PASS/PASS |
| `test_conv_transpose2d_block_c[op2_40x72-opt_level_0/1/2]` | PASS/FAIL/FAIL | PASS/PASS/PASS |
| `test_conv_transpose2d_block_c[op3_80x144-opt_level_0/1/2]` | PASS/FAIL/FAIL | PASS/PASS/PASS |
| `test_opt_sweep[…-opt_level_1-block_C]` | FAIL (ConvT2d compile) | PASS |
| `test_opt_sweep[…-opt_level_2-block_C]` | FAIL (ConvT2d compile) | FAIL (Conv2d runtime CB clash) |
