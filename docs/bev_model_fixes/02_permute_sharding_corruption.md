# Block D — PermuteOp BLOCK_SHARDED Silent Data Corruption

## 1. Affected Test Cases

- `test_bev_block_d_gridsample[opt_level_2-*]`: 0/40 pass (silent wrong output)
- Block D: `block_D_camera_cylinder_bev_transform`

## 2. Failure

No crash. The model compiled and ran successfully but produced numerically incorrect outputs. PCC (Pearson Correlation Coefficient) was near zero for all test configurations at opt_level_2.

## 3. Failure Reason

At opt_level_2, `TTNNGreedyMemoryLayoutPropagation` with `enableL1ShardingLayouts=true` assigned BLOCK_SHARDED L1 output layouts to the two `PermuteOp`s surrounding `grid_sample` (NCHW↔NHWC permutations).

Inside `permute.cpp`, WH-involving permutations (where the last dimension is moved) are decomposed into chains of `transpose_wh` and `transpose_hc`. The `transpose_wh` kernel's sharded path is gated on:

```cpp
bool input_height_sharded = is_sharded && is_l1 &&
    shard_spec.shape[1] == logical_shape[-1];  // full width of tensor
bool use_sharded_wh = input_height_sharded && !input_cn_sharded;
```

For BLOCK_SHARDED inputs, `shard_spec.shape[1] < logical_shape[-1]` (each shard covers only part of the width) — so `use_sharded_wh = false` — and the non-sharded `TransposeWHProgramFactory` runs on sharded memory, causing **silent data corruption** (no error, wrong results).

The op model could not detect this because `QUERY_OP_CONSTRAINTS` builds the input tensor from the previous op's output layout (which was DRAM interleaved since `grid_sample` always outputs DRAM), so `a.is_sharded()` was always false during the model query, and no error was raised.

Two rejected fix approaches:

1. Adding detection in `TTNNOpModel.cpp` — the query tensor was never sharded, so the bug was invisible to the model.
2. Returning `OpNotSupportedError` from `getOpConstraints` — `OperationValidationAndFallback` interprets this as "model not implemented, skip op" and leaves the sharded layout from the greedy optimizer in place (32 test regressions).

## 4. Fix Implementation Details

Three-layer fix:

**Fix A — `LegalOpLayoutAnalysis.cpp` (compile-time, correct layer)**

Filter all sharded L1 layouts from PermuteOp candidates before the optimizer beam search. This is the canonical fix: `LegalOpLayoutAnalysis` is where layout validity is declared per-op. By removing sharded L1 from the candidate set entirely, the greedy optimizer never proposes a sharded layout for PermuteOp and no runtime path reaches the broken `TransposeWHProgramFactory` sharded code.

**Fix B — `permute.cpp` (runtime routing)**

In the `a.is_sharded()` branch, detect WH-involving permutations with non-full-width shards and route them to `prim_permute` instead of the broken transpose chain. This ensures that if a sharded tensor somehow reaches the op at runtime (e.g., from a code path not covered by Fix A), the fallback is explicit and correct rather than silently corrupt.

**Fix C — `permute_device_operation.cpp` (runtime guard)**

Add `TT_FATAL(!is_sharded)` in `prim::permute`'s `validate_on_program_cache_miss` so any sharded tensor reaching the primitive fails with a clear error message instead of silent corruption. This turns the silent data corruption into a visible, debuggable failure if the guards in Fixes A and B are ever bypassed.

## 5. Files Changed with Diffs

**`lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp`** (tt-mlir)
```diff
+  if (isa<PermuteOp>(op)) {
+    analysisResult.erase(
+        std::remove_if(analysisResult.begin(), analysisResult.end(),
+                       [](const OpConfig &cfg) {
+                         return cfg.outputLayout &&
+                                cfg.outputLayout.hasShardedL1TensorMemoryLayout();
+                       }),
+        analysisResult.end());
+  }
```

**`ttnn/cpp/ttnn/operations/data_movement/permute/permute.cpp`** (tt-metal)
```diff
 if (a.is_sharded()) {
-    if (N == 0 && C == 1 && H == 2 && W == 3) {
+    bool involves_wh = (N == 0 && C == 1 && H == 3 && W == 2) ||
+                       (N == 0 && C == 2 && H == 3 && W == 1) ||
+                       (N == 0 && C == 3 && H == 1 && W == 2) ||
+                       (N == 0 && C == 3 && H == 2 && W == 1);
+    bool full_width_shard = a.shard_spec().has_value() &&
+                            a.shard_spec()->shape[1] == static_cast<uint32_t>(a.logical_shape()[-1]);
+    if (involves_wh && !full_width_shard) {
+        output = prim_permute(formatted_input_tensor);
+    } else if (N == 0 && C == 1 && H == 2 && W == 3) {
         output = formatted_input_tensor;
```

**`ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_device_operation.cpp`** (tt-metal)
```diff
 void PermuteDeviceOperation::validate_on_program_cache_miss(...) {
     TT_FATAL(attributes.dims.size() == tensor_args.input_tensor.logical_shape().rank(), ...);
+    TT_FATAL(
+        !tensor_args.input_tensor.is_sharded(),
+        "PermuteDeviceOperation (prim::permute) does not support sharded input tensors. "
+        "Use ttnn::permute which handles sharded tensors via transpose decomposition for supported permutation patterns.");
 }
```

## 6. After Fix — How It Works

At opt_level_2, `LegalOpLayoutAnalysis` now generates only non-sharded layout candidates for PermuteOp. The greedy optimizer's beam search is restricted to DRAM interleaved or L1 interleaved layouts for any PermuteOp. The NCHW→NHWC permute on the `grid_sample` input and the NHWC→NCHW permute on the `grid_sample` output both execute with non-sharded tensors, where the `TransposeWHProgramFactory` operates correctly on contiguous memory.

The runtime fix in `permute.cpp` provides defense-in-depth: if a sharded tensor reaches `permute.cpp` through any future code path, WH-involving permutations with partial-width shards are routed to `prim_permute` (the scalar fallback) instead of the decomposition into `transpose_wh` + `transpose_hc`, which is only valid for HEIGHT_SHARDED (full-width shard) inputs. The `TT_FATAL` in `permute_device_operation.cpp` ensures that any attempt to run the `prim::permute` primitive directly on a sharded tensor produces a clear diagnostic error.

## 7. Test Results

| Test | Before | After |
|------|--------|-------|
| `test_bev_block_d_gridsample[opt_level_2]` (40 tests) | **0/40 FAIL (wrong output)** | **40/40 PASS** |
| `test_bev_block_d_gridsample[opt_level_0/1]` | PASS | PASS |
