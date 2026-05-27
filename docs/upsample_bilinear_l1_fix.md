# Block E BEV — Bilinear Upsample Fix

## Overview

`test_opt_sweep[enable_program_cache-opt_level_2-block_E]` failed for the
`block_E_bev_aggregator` BEV model due to a segfault during compilation.
Block E has three bilinear upsample ops with scale factor 2×2 on inputs of
size `1×16×8×256`, `1×32×16×128`, and `1×64×32×64`.

All fixes are in `third_party/tt-mlir` only — no changes were made to the
tt-forge-onnx front-end or any MLA passes.

---

## Bug — Bilinear Upsample Segfault at opt_level_2

### Symptom

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

The segfault occurred during the Memory Layout Analysis (MLA) pass, not at
runtime. `opt_level_1` did not reproduce this because MLA does not evaluate
L1 sharded layouts at that level.

### Root Cause

The MLA pass calls `OpModel<UpsampleOp>::getOpConstraints` with a sharded
L1 layout hint to evaluate whether the upsample op fits in L1. Inside the
OpModel, this invokes `QUERY_OP_CONSTRAINTS(::ttnn::upsample, ...)` which
calls the actual TTNN C++ API on the mock device.

For bilinear mode, `ttnn::upsample` checks if the input is already sharded.
When MLA's proposed layout is HEIGHT_SHARDED, the autoshard step is skipped
and `apply_bilinear_halo_preprocessing` runs with the arbitrary shard spec
MLA proposed. Inside halo preprocessing, `generate_halo_kernel_config_tensors`
accesses `tensor_metadata[global_idx]` where `global_idx` is computed from
the shard boundaries. Because MLA's shard height does not match the kernel's
expected distribution (`ceil(N*H*W / num_cores)`), `global_idx` exceeds
`tensor_metadata.size()` — an out-of-bounds access that segfaults.

### Fix — Stage 1: Reject Sharded L1 to Stop the Segfault

**File:** `lib/OpModel/TTNN/TTNNOpModel.cpp` — `getOpConstraints` and `getOpRuntime`

Added early-return guards before any device call to reject sharded L1 for
bilinear upsample, causing MLA to fall back to DRAM interleaved:

```cpp
if (mode == "bilinear") {
    if (inputLayout.hasL1BufferType() && inputLayout.getMemLayout() &&
        isShardedMemoryLayout(inputLayout.getMemLayout().getValue())) {
        return llvm::createStringError(
            "Bilinear upsample does not support sharded L1 input");
    }
}
```

This resolved the segfault and made `test_opt_sweep[opt_level_2-block_E]`
pass at 116.58 FPS.

---

## Improvement — Enable HEIGHT_SHARDED L1 for Bilinear Upsample

### What tt-metal Actually Supports

The bilinear path in `ttnn::upsample` (`upsample.cpp` lines 122–151):

- **DRAM interleaved input**: autoreshards to HEIGHT_SHARDED L1 internally
  via `compute_bilinear_autoshard_memory_config`, then runs halo + kernel.
- **HEIGHT_SHARDED L1 input**: runs directly to halo + kernel (no reshard).
- **BLOCK_SHARDED / WIDTH_SHARDED**: `TT_FATAL` in device validation.

The DRAM path is functionally correct but pays a DRAM→L1 copy at runtime.
With a correct HEIGHT_SHARDED input the copy is eliminated.

### The Autoshard Formula

`compute_bilinear_autoshard_memory_config` in `upsample.cpp`:

```cpp
total_nhw    = N * H * W
num_shards   = min(grid.x * grid.y, total_nhw)
shard_height = round_up(total_nhw, num_shards) / num_shards
shard_spec   = {shard_height, C}   // ROW_MAJOR, HEIGHT_SHARDED, L1
```

For Block E on WH N150 (8×8 = 64 cores):

| Op | Input shape | N·H·W | shard_height | shard_spec |
|----|-------------|--------|--------------|------------|
| upsample2d #1 | `1×16×8×256` | 128 | 2 | `[2, 256]` |
| upsample2d #2 | `1×32×16×128` | 512 | 8 | `[8, 128]` |
| upsample2d #3 | `1×64×32×64` | 2048 | 32 | `[32, 64]` |

### Why MLA's Shard Spec is Wrong

MLA proposes HEIGHT_SHARDED with a layout it derives from surrounding ops.
This shard spec almost never matches the autoshard formula, causing
`generate_halo_kernel_config_tensors` to go out of bounds.

Additionally, prior ops (Conv2d) output TILE layout. A HEIGHT_SHARDED +
TILE TensorSpec requires shard shapes to be multiples of 32×32. The
autoshard shard height can be as small as 2 (e.g., upsample #1), so a
TILE-based query crashes with:

```
TT_FATAL: Physical shard shape (2, 256) must be tile {32, 32} sized!
```

### Fix — Stage 2: Recompute Autoshard Spec in the OpModel

**File:** `lib/OpModel/TTNN/TTNNOpModel.cpp`

Replaced the blanket rejection with a targeted approach:

**Step 1 — Reject unsupported sharded types** (BLOCK/WIDTH sharded metal TT_FATAL):

```cpp
if (mode == "bilinear" && inputLayout.hasL1BufferType() &&
    inputLayout.getMemLayout()) {
    auto memLayout = inputLayout.getMemLayout().getValue();
    if (isShardedMemoryLayout(memLayout) &&
        memLayout != TensorMemoryLayout::HeightSharded) {
        return llvm::createStringError(
            "Bilinear upsample only supports HEIGHT_SHARDED L1 input");
    }
}
```

**Step 2 — Added `computeBilinearAutoshardMemoryConfig`**: mirrors the C++
autoshard formula to produce the exact MemoryConfig the kernel expects:

```cpp
static ::tt::tt_metal::MemoryConfig
computeBilinearAutoshardMemoryConfig(MeshDevice *device,
                                     llvm::ArrayRef<int64_t> inputShape) {
    const uint64_t total_nhw = N * H * W;
    const uint64_t num_shards = std::min(grid.x * grid.y, total_nhw);
    const uint64_t shard_height = ::tt::round_up(total_nhw, num_shards) / num_shards;

    ShardSpec shard_spec(
        num_cores_to_corerangeset(num_shards, grid, true),
        {shard_height, C}, ShardOrientation::ROW_MAJOR);

    return MemoryConfig(HEIGHT_SHARDED, L1, shard_spec);
}
```

**Step 3 — Added `buildBilinearAutoshardInputSpec`**: builds a TensorSpec
using the autoshard MemoryConfig with **ROW_MAJOR page layout** regardless
of the input's current layout. This is required because TILE layout forces
tile-aligned shard shapes, but the autoshard shard height may be less than 32:

```cpp
static ::ttnn::TensorSpec
buildBilinearAutoshardInputSpec(MeshDevice *device,
                                llvm::ArrayRef<int64_t> inputShape,
                                const TTNNLayoutAttr &inputLayout) {
    auto autoMC = computeBilinearAutoshardMemoryConfig(device, inputShape);
    TensorLayout layout(getDataType(inputLayout.getDataType()),
                        PageConfig(Layout::ROW_MAJOR), autoMC);
    return TensorSpec(getShape(inputShape), layout);
}
```

**Step 4 — Used corrected spec in query**: for HEIGHT_SHARDED input, replace
MLA's arbitrary TensorSpec with the autoshard one before calling
`QUERY_OP_CONSTRAINTS`. For HEIGHT_SHARDED output, compute the matching
autoshard config on the output shape (N × out_H × out_W × C).

### Result

`test_opt_sweep[opt_level_2-block_E]` **PASS** at **120.93 FPS**
(vs 116.58 FPS with DRAM fallback — +3.7% improvement).

---

## Complete Test Results

| Test | Before | After Stage 1 | After Stage 2 |
|------|--------|---------------|---------------|
| `test_opt_sweep[opt_level_1-block_E]` | PASS | PASS | PASS (132.80 FPS) |
| `test_opt_sweep[opt_level_2-block_E]` | **FAIL (segfault)** | PASS (116.58 FPS, DRAM) | **PASS (120.93 FPS, L1 sharded)** |

---

## File Summary

| File | Change |
|------|--------|
| `lib/OpModel/TTNN/TTNNOpModel.cpp` | Added `computeBilinearAutoshardMemoryConfig` and `buildBilinearAutoshardInputSpec` helpers; replaced blanket sharded-L1 rejection with HEIGHT_SHARDED support using recomputed autoshard spec; added `tt-metalium/math.hpp` and `tt-metalium/work_split.hpp` includes |
