# Block C BEV — Conv2d / ConvTranspose2d Fixes

## Overview

`test_opt_sweep[enable_program_cache-opt_level_1/2-block_C]` failed for the
`block_C_cylinder_backbone` BEV model due to two independent bugs.  Both fixes
are in `third_party/tt-mlir` only — no changes were made to the tt-forge-onnx
front-end or any MLA passes.

---

## Bug 1 — ConvTranspose2d compilation failure

### Symptom

```
TT_FATAL @ conv2d_op_program_factory_common.cpp:91:
  get_cb_info expects conv_config.weights_dtype to be already set
```

Affects: `test_opt_sweep[opt_level_1-block_C]` and `opt_level_2-block_C`.

Block C has four ConvTranspose2d ops (in_ch=192, out_ch=192, kernel=2×2,
stride=2×2) with inputs ranging from 1×192×10×18 up to 1×192×80×144.

### Root Cause

`prepare_conv_transpose2d_weights.cpp` computes `weight_dtype` from
`conv_config.weights_dtype.value_or(weight_tensor.dtype())` but never writes
the result back.  When the DRAM slice path calls
`determine_slice_config → get_L1_usage → calculate_L1_usage → get_cb_info`,
the `Conv2dConfig` stored in `ConvT2DSliceAttr` still has
`weights_dtype = nullopt`, triggering the fatal.

### Fix 1 — `prepare_conv_transpose2d_weights.cpp`

File:
`third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.cpp`

```cpp
DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype());
// Ensure weights_dtype is set so get_cb_info doesn't fatal in the DRAM path.
if (!conv_config.weights_dtype.has_value()) {
    conv_config.weights_dtype = weight_dtype;
}
```

### Fix 2 — `TTNNOpModel.cpp` (two locations)

File: `third_party/tt-mlir/lib/OpModel/TTNN/TTNNOpModel.cpp`

**`getOpConstraints` for `ConvTranspose2dOp`** (~line 5636): set
`conv2dConfigConverted->weights_dtype = weightSpec.data_type()` before
`QUERY_OP_CONSTRAINTS` is called.

**`getPrepareConv2dWeightsOpOutputTensorSpec`** (~line 655): when
`transpose=true`, ensure `conv2dConfigConverted->weights_dtype` is set from
`outputDtype` before the query closure captures `conv2dConfigConverted`:

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

### Result

`test_opt_sweep[opt_level_1-block_C]` **PASS**.

---

## Bug 2 — Conv2d L1 circular buffer clash at runtime

### Symptom

After Bug 1 was fixed, `test_opt_sweep[opt_level_2-block_C]` failed at
inference time with:

```
TT_THROW @ program.cpp:1366:
Statically allocated circular buffers in program 60394 clash with L1 buffers
on core range [0-0 - 7-6].
L1 buffer allocated at 171520 and static circular buffer region ends at 177440.
```

Call stack: `conv2d_L1 → prim::matmul → validate_circular_buffer_region → TT_THROW`

`opt_level_1` did **not** reproduce this (Conv2d uses DRAM at opt_level_1).

### Root Cause — The Dead Zone

The L1SpillManagement simulation uses a virtual address space
`[0, l1BudgetPerCore]` where:

```
l1BudgetPerCore  = tensorL1UsageCap × usableL1Size
               = 0.95 × 1,395,424 = 1,325,652 bytes
usableL1Size     = l1_size − l1_unreserved_base
               = 1,499,136 − 103,712 = 1,395,424 bytes
```

Virtual address 0 maps to real address
`l1_size − l1BudgetPerCore = 173,484`.  The region
**real [103,712 – 173,484)** (69,772 bytes) is the **dead zone** — it is
below the simulation floor but above `l1_unreserved_base`, where CBs
physically start.

The failing Conv2d had:

| Parameter | Value |
|-----------|-------|
| `cbPeakUsage` | 73,728 bytes |
| Dead zone | 69,772 bytes |
| Real CB end | 103,712 + 73,728 = **177,440** |
| Dead zone upper boundary | 173,484 |
| CB extends past simulation floor | 177,440 − 173,484 = **3,956 bytes** |

Because the CB extends 3,956 bytes into the dead zone, if total runtime L1
pressure exceeds the budget by even a tiny amount, a tensor will be allocated
inside the dead zone — within the CB region.

At runtime, this exact scenario occurred: actual L1 usage was 1,327,616 bytes
(1,964 bytes over budget), placing a tensor at real address **171,520** —
inside the CB region ending at 177,440.  The simulation's
`wouldCBsOverlapTensors` check reported `lowestExisting = 289,364` (virtual),
corresponding to real 462,848 — far from 171,520 — so no demotion was
triggered, and the op was kept in L1.

### Fix — Dead Zone Pre-Check in `ensureFitsL1`

**Files changed:**

| File | Change |
|------|--------|
| `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h` | Added `usableL1Size` constructor parameter; added `l1DeadZone` member |
| `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp` | Updated constructor; added dead zone pre-check in `ensureFitsL1` |
| `lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyL1SpillManagement.cpp` | Pass `chipDesc.getUsableL1Size()` to constructor |

**Constructor change (`L1SpillManagement.cpp`):**

```cpp
L1SpillManagement<MemoryTracker>::L1SpillManagement(
    func::FuncOp func, ttcore::GridAttr deviceGrid,
    uint64_t l1BudgetPerCore, uint64_t usableL1Size,
    std::unique_ptr<L1SpillObserver> observer)
    : ...,
      l1DeadZone(usableL1Size > l1BudgetPerCore
                     ? usableL1Size - l1BudgetPerCore : 0) { ... }
```

**Dead zone pre-check added to `ensureFitsL1`:**

```cpp
if (l1DeadZone > 0 && cbPeakUsage > l1DeadZone) {
    // CBs extend past the simulation floor into territory the simulation
    // cannot track.  Even a tiny runtime L1 overflow places a tensor in the
    // dead zone, directly under the CB region.  Force DRAM.
    // Evict live L1 inputs first (mirrors sibling-spill in handleFragmentation)
    // so that evictForDramCBGrowth's DRAM-output validation sees homogeneous
    // DRAM inputs.
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

**Why this works:**
- `l1DeadZone = usableL1 − l1BudgetPerCore` is the exact gap between the
  simulation floor and `l1_unreserved_base`.
- If `cbPeakUsage > l1DeadZone`, the CB physically extends into the dead zone.
  Any runtime overflow (lazy deallocation, program cache) creates a tensor
  there that the simulation cannot see and cannot evict.
- The L1 inputs are evicted first so that `evictForDramCBGrowth`'s DRAM
  output validation succeeds (ops like Concat require homogeneous input/output
  memory layouts).

### Result

`test_opt_sweep[opt_level_2-block_C]` **PASS**.

---

## Complete Test Results

| Test | Before | After |
|------|--------|-------|
| `test_opt_sweep[opt_level_0-block_C]` | PASS | PASS |
| `test_opt_sweep[opt_level_1-block_C]` | FAIL (ConvT2d compile) | **PASS** |
| `test_opt_sweep[opt_level_2-block_C]` | FAIL (Conv2d CB clash) | **PASS** |
| `test_ops_onnx.py` (17 tests incl. GridSample) | PASS | PASS |

---

## File Summary

| File | Change | Bug |
|------|--------|-----|
| `tt-metal/.../prepare_conv_transpose2d_weights.cpp` | Set `weights_dtype` in all paths | Bug 1 |
| `tt-mlir/lib/OpModel/TTNN/TTNNOpModel.cpp` | Set `weights_dtype` in `getOpConstraints` and `getPrepareConv2dWeightsOpOutputTensorSpec` | Bug 1 |
| `tt-mlir/include/.../L1SpillManagement.h` | Add `usableL1Size` constructor param, `l1DeadZone` member | Bug 2 |
| `tt-mlir/lib/.../L1SpillManagement.cpp` | Constructor init; dead zone pre-check in `ensureFitsL1` | Bug 2 |
| `tt-mlir/lib/.../GreedyL1SpillManagement.cpp` | Pass `chipDesc.getUsableL1Size()` to constructor | Bug 2 |
