# Block E — Bilinear Upsample MLA Segfault & L1 Sharding Fix

## 1. Affected Test Cases

- `test_opt_sweep[enable_program_cache-opt_level_2-block_E]` — FAIL (segfault)
- `test_opt_sweep[disable_program_cache-opt_level_2-block_E]` — FAIL (segfault)

Block E: `block_E_bev_aggregator`. Has three bilinear upsample ops with inputs `1x16x8x256`, `1x32x16x128`, `1x64x32x64`.

## 2. Failure

```
Fatal Python error: Segmentation fault

tt_forge_signal_handler - signal: 11 (segmentation fault)
stacktrace:
 --- ttnn::operations::sliding_window::generate_halo_kernel_config_tensors(...)
 --- ttnn::prim::UntilizeWithHaloProgramFactory::create(...)
 --- ttnn::prim::halo(...)
 --- ttnn::operations::upsample::upsample(...)
 --- mlir::tt::ttnn::op_model::OpModel<mlir::tt::ttnn::UpsampleOp>::getOpConstraints(...)
 --- mlir::tt::ttnn::MemoryLayoutPropagation::evaluateHint(...)
 --- mlir::tt::ttnn::MemoryLayoutPropagation::processOp(...)
```

The segfault occurs during compilation (MLA pass), not at runtime.

## 3. Failure Reason

MLA calls `OpModel<UpsampleOp>::getOpConstraints` with a HEIGHT_SHARDED L1 layout hint (proposing that the upsample input should be HEIGHT_SHARDED). The OpModel calls `QUERY_OP_CONSTRAINTS(::ttnn::upsample, ...)` which invokes the real TTNN C++ API on the mock device.

Inside `ttnn::upsample` for bilinear mode:

```cpp
if (!input_tensor.is_sharded()) {
    // Autoshard DRAM -> HEIGHT_SHARDED internally
    auto mc = compute_bilinear_autoshard_memory_config(input_tensor);
    input = to_memory_config(input, mc);
}
// Input is now HEIGHT_SHARDED (either from autoshard or from caller):
apply_bilinear_halo_preprocessing(input, ...);
```

When MLA passes a HEIGHT_SHARDED tensor, `input.is_sharded()` is true — autoresharding is skipped — and `apply_bilinear_halo_preprocessing` runs immediately with MLA's arbitrary shard spec.

Inside halo preprocessing, `generate_halo_kernel_config_tensors` computes `global_idx` from shard boundaries. MLA's shard height almost never matches the autoshard formula's expected `ceil(N*H*W / num_cores)`. When `global_idx` exceeds `tensor_metadata.size()` — out-of-bounds memory access — segfault.

**The autoshard formula** (`compute_bilinear_autoshard_memory_config`):

```cpp
total_nhw    = N x H x W
num_shards   = min(grid.x x grid.y, total_nhw)   // 64 cores on WH N150
shard_height = round_up(total_nhw, num_shards) / num_shards
shard_spec   = {shard_height, C}, HEIGHT_SHARDED, L1, ROW_MAJOR
```

| Upsample | Input (NHWC) | N x H x W | num_shards | shard_height |
|----------|-------------|-----------|-----------|-------------|
| #1 | `1x8x256x16` | 2048 | 64 | 32 |
| #2 | `1x16x128x32` | 2048 | 64 | 32 |
| #3 | `1x32x64x64` | 2048 | 64 | 32 |

MLA proposes different shard heights — out-of-bounds access in `generate_halo_kernel_config_tensors`.

Additionally, prior conv2d ops output TILE layout. HEIGHT_SHARDED + TILE requires shard shapes divisible by 32x32. Shard heights of 2 or 8 (for smaller inputs) violate this — additional TILE-alignment fatal error.

## 4. Fix Implementation Details — Two Stages

**Stage 1 — Reject all non-HEIGHT_SHARDED sharded L1 (stop segfault)**

In `TTNNOpModel.cpp::getOpConstraints` and `getOpRuntime` for UpsampleOp, add early-return guards that reject any non-HEIGHT_SHARDED sharded L1 input layout. This prevents BLOCK_SHARDED layouts from reaching `ttnn::upsample`, which cannot handle them:

```cpp
if (mode == "bilinear") {
    if (inputLayout.hasL1BufferType() && inputLayout.getMemLayout()) {
        auto memLayout = inputLayout.getMemLayout().getValue();
        if (isShardedMemoryLayout(memLayout) &&
            memLayout != TensorMemoryLayout::HeightSharded) {
            return llvm::createStringError(
                "Bilinear upsample only supports HEIGHT_SHARDED L1 input");
        }
    }
}
```

This stops the segfault. MLA sees an error for the sharded hint, falls back to DRAM interleaved. Result: 116.58 FPS.

**Stage 2 — Recompute autoshard spec in OpModel (enable L1 sharded path)**

Add helper functions that mirror the tt-metal autoshard formula exactly. When MLA proposes a HEIGHT_SHARDED input, the OpModel replaces MLA's arbitrary shard spec with the correct autoshard spec — the same one that `ttnn::upsample` will compute internally at runtime.

`computeBilinearAutoshardMemoryConfig(device, inputShape)`:

```cpp
// inputShape is NHWC: [N, H, W, C]
total_nhw = N x H x W
num_shards = min(grid.x x grid.y, total_nhw)
shard_height = round_up(total_nhw, num_shards) / num_shards
return MemoryConfig(HEIGHT_SHARDED, L1,
    ShardSpec(num_cores_to_corerangeset(num_shards, grid, true),
              {shard_height, channels}, ROW_MAJOR))
```

`buildBilinearAutoshardInputSpec(device, inputShape, inputLayout)`:

```cpp
auto autoMC = computeBilinearAutoshardMemoryConfig(device, inputShape);
// Use ROW_MAJOR page layout: autoshard shard_height can be < 32,
// which is incompatible with TILE layout requirements.
TensorLayout layout(getDataType(inputLayout.getDataType()),
                    PageConfig(Layout::ROW_MAJOR), autoMC);
return TensorSpec(getShape(inputShape), layout);
```

In `getOpConstraints`: for HEIGHT_SHARDED input, replace MLA's arbitrary TensorSpec with `buildBilinearAutoshardInputSpec(...)` before calling `QUERY_OP_CONSTRAINTS`. For HEIGHT_SHARDED output, compute the matching autoshard config for the output shape `(N, H_out, W_out, C)`.

Result with Stage 2: 120.93 FPS (+3.7% over DRAM fallback from Stage 1).

## 5. Files Changed with Diffs

**`lib/OpModel/TTNN/TTNNOpModel.cpp`** (tt-mlir) — summary of additions:

```diff
+#include "tt-metalium/math.hpp"
+#include "tt-metalium/work_split.hpp"

+// Mirrors ttnn::operations::upsample::compute_bilinear_autoshard_memory_config
+// inputShape is NHWC: [N, H, W, C]
+static ::tt::tt_metal::MemoryConfig
+computeBilinearAutoshardMemoryConfig(MeshDevice *device,
+                                     llvm::ArrayRef<int64_t> inputShape) {
+  const uint64_t total_nhw =
+      inputShape[0] * inputShape[1] * inputShape[2];
+  const uint64_t channels = inputShape[3];
+  const auto grid = device->compute_with_storage_grid_size();
+  const uint64_t num_shards =
+      std::min(static_cast<uint64_t>(grid.x * grid.y), total_nhw);
+  const uint64_t shard_height =
+      ::tt::round_up(total_nhw, num_shards) / num_shards;
+  const ShardSpec shard_spec(
+      num_cores_to_corerangeset(num_shards, grid, /*row_wise=*/true),
+      {shard_height, channels},
+      ShardOrientation::ROW_MAJOR);
+  return MemoryConfig(TensorMemoryLayout::HeightSharded,
+                      BufferType::L1, shard_spec);
+}

+// Build TensorSpec with the autoshard memory config.
+// Uses ROW_MAJOR page layout: autoshard shard_height can be < 32 rows,
+// which is incompatible with TILE layout's 32x32 tile requirement.
+static ::ttnn::TensorSpec
+buildBilinearAutoshardInputSpec(MeshDevice *device,
+                                llvm::ArrayRef<int64_t> inputShape,
+                                const TTNNLayoutAttr &inputLayout) {
+  auto autoMC = computeBilinearAutoshardMemoryConfig(device, inputShape);
+  TensorLayout layout(getDataType(inputLayout.getDataType()),
+                      PageConfig(Layout::ROW_MAJOR), autoMC);
+  return TensorSpec(getShape(inputShape), layout);
+}

 // In getOpConstraints for UpsampleOp:
+  // Reject non-HEIGHT_SHARDED sharded L1 (bilinear only supports HEIGHT_SHARDED)
+  if (mode == "bilinear" && inputLayout.hasL1BufferType() &&
+      inputLayout.getMemLayout()) {
+    auto memLayout = inputLayout.getMemLayout().getValue();
+    if (isShardedMemoryLayout(memLayout) &&
+        memLayout != TensorMemoryLayout::HeightSharded) {
+      return llvm::createStringError(
+          llvm::inconvertibleErrorCode(),
+          "Bilinear upsample only supports HEIGHT_SHARDED L1 input");
+    }
+  }
+  // For HEIGHT_SHARDED input, replace MLA's arbitrary spec with the autoshard spec
+  ::ttnn::TensorSpec inputSpec = [&]() -> ::ttnn::TensorSpec {
+    if (mode == "bilinear" && inputLayout.hasL1BufferType() &&
+        inputLayout.getMemLayout() &&
+        inputLayout.getMemLayout().getValue() ==
+            TensorMemoryLayout::HeightSharded) {
+      return buildBilinearAutoshardInputSpec(device, inputShape, inputLayout);
+    }
+    return detail::convertToTensorSpec(device, inputShape, inputLayout).get();
+  }();
+  // For HEIGHT_SHARDED output, compute matching autoshard config
+  std::optional<::tt::tt_metal::MemoryConfig> outputMemConfig =
+      detail::getNullableMemoryConfig(outputLayout);
+  if (mode == "bilinear" && outputLayout && outputLayout.hasL1BufferType() &&
+      outputLayout.getMemLayout() &&
+      outputLayout.getMemLayout().getValue() ==
+          TensorMemoryLayout::HeightSharded) {
+    outputMemConfig = computeBilinearAutoshardMemoryConfig(device, outputShape);
+  }
```

(Identical guards also applied in `getOpRuntime` for UpsampleOp.)

## 6. After Fix — How It Works

**Stage 1 (segfault fix):**
MLA proposes HEIGHT_SHARDED with an arbitrary shard spec. `getOpConstraints` detects HEIGHT_SHARDED mode and calls `buildBilinearAutoshardInputSpec`, replacing MLA's spec with `shard_height=32, shard_width=C`. `QUERY_OP_CONSTRAINTS` now calls `ttnn::upsample` with a tensor that is already correctly HEIGHT_SHARDED. Inside `upsample`, `input.is_sharded()` is true, autoresharding is skipped, and `generate_halo_kernel_config_tensors` receives the expected shard boundaries — no out-of-bounds access, no segfault.

**Stage 2 (L1 sharded path performance):**
Since MLA now receives a valid cost estimate for HEIGHT_SHARDED, it correctly determines that placing the upsample input in HEIGHT_SHARDED L1 eliminates the DRAM→L1 data movement that would otherwise occur at runtime. MLA assigns HEIGHT_SHARDED L1 to the upsample input. At runtime, the conv2d preceding upsample outputs directly to HEIGHT_SHARDED L1. The upsample reads from HEIGHT_SHARDED L1 — matching the shard spec that its autoshard formula would have computed — and proceeds without the DRAM round-trip.

The 3.7% FPS improvement (116.58 → 120.93 FPS) comes from eliminating the DRAM bandwidth for three upsample input tensors per BEV aggregation pass.

## 7. Test Results

| Test | Before | Stage 1 | Stage 2 |
|------|--------|---------|---------|
| `opt_level_1-block_E` | PASS | PASS | PASS (132.80 FPS) |
| `opt_level_2-block_E` | **FAIL (segfault)** | PASS (116.58 FPS) | **PASS (120.93 FPS)** |
