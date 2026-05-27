# PermuteOp Sharding Fix

## Context

The BEV GridSample pipeline uses two 4D permutations:

- `[0, 2, 3, 1]` — NCHW → NHWC (before `grid_sample`)
- `[0, 3, 1, 2]` — NHWC → NCHW (after `grid_sample`)

Both permutations move the last dimension (WH-involving). At `opt_level_2`, the
`TTNNGreedyMemoryLayoutPropagation` pass with `enableL1ShardingLayouts = true` was
assigning BLOCK_SHARDED L1 output layouts to these `PermuteOp`s, causing silent
data corruption at runtime.

---

## Root Cause

### Why BLOCK_SHARDED corrupts PermuteOp output

In `permute.cpp`, WH-involving permutations on sharded input are handled by chaining
`transpose_wh` and `transpose_hc` calls. Inside `transpose_device_operation.cpp`,
the WH-sharded kernel path is gated on:

```cpp
bool input_height_sharded =
    is_sharded && is_l1 && shard_spec.shape[1] == logical_shape[-1];
bool use_sharded_wh = input_height_sharded && !input_cn_sharded;
```

For BLOCK_SHARDED, `shard_spec.shape[1] < logical_shape[-1]` (shard covers only
part of the width), so `input_height_sharded = false` → `use_sharded_wh = false`.
The non-sharded `TransposeWHProgramFactory` then runs on a sharded memory buffer,
producing a corrupted HEIGHT_SHARDED non-contiguous result.

For HEIGHT_SHARDED, `shard_spec.shape[1] == logical_shape[-1]` so `transpose_wh`
itself is fine, but the intermediate tensor produced by `transpose_hc` is not
HEIGHT_SHARDED — the subsequent `transpose_wh` cannot handle it correctly.

### Why `prim::permute` does not support sharded inputs

`PermuteDeviceOperation` (the primitive used by `prim::permute`) has no sharded
kernel path. Passing a sharded tensor to it would run the non-sharded kernel on
sharded memory. This constraint was undocumented before this fix.

### Why the op model could not detect the problem

`PermuteOp::getOpConstraints` calls `QUERY_OP_CONSTRAINTS(::ttnn::permute, input, ...)`.
The `input` tensor is created from the **previous op's output layout**, which is
typically DRAM interleaved (e.g. `grid_sample` always outputs DRAM). So
`a.is_sharded()` is `false` during the query, the sharded branch is never entered,
no exception is thrown, and the op model reports the BLOCK_SHARDED config as valid.

Attempting to add a static check to `PermuteOp::getOpConstraints` via
`issueErrorForGetOpConstraints` / `OpNotSupportedError` also fails: the
`OperationValidationAndFallback` pass interprets `OpNotSupportedError` as
"op model not implemented → skip this op entirely," which leaves the sharded
layout assigned by the greedy optimizer in place and causes the same corruption.

---

## Files Changed

| File | Layer | Change |
|---|---|---|
| `ttnn/cpp/ttnn/operations/data_movement/permute/permute.cpp` | tt-metal runtime | Route WH-involving + non-full-width-shard through `prim_permute` |
| `ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_device_operation.cpp` | tt-metal runtime | Add `TT_FATAL(!is_sharded)` in `validate_on_program_cache_miss` |
| `lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp` | tt-mlir compiler | Filter sharded L1 layouts from PermuteOp candidates |

---

## Fix 1 — `permute.cpp`

**File:** `third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/permute/permute.cpp`

In the `a.is_sharded()` branch, detect WH-involving permutations with a
non-full-width shard and route them to `prim_permute` (which now fails with
`TT_FATAL`) instead of silently running broken transpose chains.

```cpp
if (a.is_sharded()) {
    // WH-involving permutations require the shard to cover the full logical width
    // (HEIGHT_SHARDED-equivalent: shard_spec.shape[1] == logical_shape[-1]).
    // For BLOCK_SHARDED inputs, shard_spec.shape[1] < logical_shape[-1], so the
    // transpose_wh kernel's use_sharded_wh condition is false and the non-sharded
    // program factory runs on sharded memory, producing corrupted output.
    // Route these cases through prim_permute so validate_on_program_cache_miss
    // rejects the sharded input and the op model can exclude this config.
    bool involves_wh = (N == 0 && C == 1 && H == 3 && W == 2) ||
                       (N == 0 && C == 2 && H == 3 && W == 1) ||
                       (N == 0 && C == 3 && H == 1 && W == 2) ||
                       (N == 0 && C == 3 && H == 2 && W == 1);
    bool full_width_shard = a.shard_spec().has_value() &&
                            a.shard_spec()->shape[1] == static_cast<uint32_t>(a.logical_shape()[-1]);
    if (involves_wh && !full_width_shard) {
        output = prim_permute(formatted_input_tensor);
    } else if (N == 0 && C == 1 && H == 2 && W == 3) {
        // ... existing cases unchanged
    }
}
```

**WH-involving patterns** (for 4D tensors, `dims = [N, C, H, W]`):

| Pattern | Dims | Decomposition |
|---|---|---|
| `[0,1,3,2]` | N=0, C=1, H=3, W=2 | `transpose_wh` |
| `[0,2,3,1]` | N=0, C=2, H=3, W=1 | `transpose_wh(transpose_hc(...))` ← BEV NCHW→NHWC |
| `[0,3,1,2]` | N=0, C=3, H=1, W=2 | `transpose_hc(transpose_wh(...))` ← BEV NHWC→NCHW |
| `[0,3,2,1]` | N=0, C=3, H=2, W=1 | `transpose_wh(transpose_hc(transpose_wh(...)))` |

---

## Fix 2 — `permute_device_operation.cpp`

**File:** `third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_device_operation.cpp`

Add an explicit `TT_FATAL` in `validate_on_program_cache_miss` so that any sharded
tensor reaching `prim::permute` fails with a clear error instead of producing
silent garbage:

```cpp
void PermuteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.dims.size() == tensor_args.input_tensor.logical_shape().rank(),
        "Permute dimensions must match input tensor rank");
    TT_FATAL(
        !tensor_args.input_tensor.is_sharded(),
        "PermuteDeviceOperation (prim::permute) does not support sharded input tensors. "
        "Use ttnn::permute which handles sharded tensors via transpose decomposition for supported permutation patterns.");
}
```

---

## Fix 3 — `LegalOpLayoutAnalysis.cpp`

**File:** `third_party/tt-mlir/lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp`

Filter all sharded L1 layouts from PermuteOp's candidate output layout set during
the greedy optimizer beam search. This is the **correct compile-time constraint
layer**: `LegalOpLayoutAnalysis` is where layout validity is declared for each op,
preventing the optimizer from ever selecting invalid configurations.

```cpp
// PermuteOp: the metal permute and transpose kernels cannot correctly handle
// sharded L1 input/output for WH-involving permutations (any permutation
// where the last dimension is moved). For BLOCK_SHARDED the transpose_wh
// kernel's use_sharded_wh condition requires a full-width shard, which
// BLOCK_SHARDED never satisfies; for HEIGHT_SHARDED the intermediate
// HC-transpose emits a tensor that the subsequent WH-transpose cannot handle.
// Removing sharded L1 layouts here is the correct compile-time constraint —
// LegalOpLayoutAnalysis is the canonical place to declare what output layouts
// are valid for an op. The tt-metal permute op validates this at runtime via
// TT_FATAL in prim::permute's validate_on_program_cache_miss.
if (isa<PermuteOp>(op)) {
    analysisResult.erase(
        std::remove_if(analysisResult.begin(), analysisResult.end(),
                       [](const OpConfig &cfg) {
                         return cfg.outputLayout &&
                                cfg.outputLayout.hasShardedL1TensorMemoryLayout();
                       }),
        analysisResult.end());
}
```

### Why not in `TTNNOpModelInterface.cpp`?

Two approaches were attempted and rejected:

**Attempt 1 — `QUERY_OP_CONSTRAINTS` in `TTNNOpModel.cpp`:**
The query creates the input tensor from the previous op's layout (DRAM interleaved).
`a.is_sharded()` is always `false` during the query, so the sharded runtime path is
never reached and no exception is raised. The op model reports the config as valid.

**Attempt 2 — `issueErrorForGetOpConstraints` in `TTNNOpModelInterface.cpp`:**
Returning `OpNotSupportedError` from `getOpConstraints` for sharded candidates
caused 32 regressions. The `OperationValidationAndFallback` pass interprets
`OpNotSupportedError` (i.e. `isNotImplemented() == true`) as "op model is missing,
skip this op," leaving the sharded layout assigned by the greedy optimizer in place
rather than selecting a non-sharded alternative.

---

## Greedy Optimizer Pipeline Context

At `opt_level_2`, `enableGreedyOptimizer = true` and the following pipeline runs:

```
1. TTNNWorkaroundsPass        — forces ROW_MAJOR for GridSampleOp inputs
2. TTNNGreedyMemoryLayoutPropagation (enableL1ShardingLayouts = true)
     └── LegalOpLayoutAnalysis per op  ← fix applied here
3. TTNNGreedyL1SpillManagement
```

Without the filter, `LegalOpLayoutAnalysis` generates BLOCK_SHARDED and
HEIGHT_SHARDED L1 candidates for PermuteOp. The beam search selects the highest-
scoring sharded config, assigns it as PermuteOp's output layout, and
`insertReshardOp` inserts a `to_memory_config` before the PermuteOp's consumer,
propagating the sharded layout through the graph.

With the filter, no sharded L1 candidates are generated for PermuteOp. The optimizer
selects an interleaved DRAM or interleaved L1 layout instead, and the permute
operations run on non-sharded tensors as the metal kernels require.

---

## Test Results

**Test file:** `forge/test/models/onnx/vision/bev/test_bev_block_d_gridsample.py`

| Config | Before fix | After fix |
|---|---|---|
| default | 40/40 pass | 40/40 pass |
| opt_level_1 | 40/40 pass | 40/40 pass |
| opt_level_2 | 0/40 pass | 40/40 pass |
| **Total** | **80/120** | **120/120** |
