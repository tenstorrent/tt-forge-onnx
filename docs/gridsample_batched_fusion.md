# GridSample Batched Fusion for BEV Block B

## Context

Block B of the BEV model (`CameraDeformedCylinder BEV Transform`) has 4 independent camera groups. Each group originally contained 8 per-camera chains:

```
index(lut5d, k, dim=3) → reshape → transpose(-3,-1) → transpose(-2,-1) → grid_sample(image, grid_k) → concat(8 results, dim=channel)
```

The LUT tensor shape is `(1, 128, 64, 8, 2)` — `(N, H, W, K=8, coords=2)`.

## Problem

Running 8 sequential `grid_sample` calls followed by a `concat` wastes kernel launch overhead and prevents the TTNN batched grid kernel from being used. The TTNN `grid_sample` op supports `batch_output_channels=true` + `use_precomputed_grid=true`, which processes all K cameras in one kernel call. It produces output `(N, C*K, H_out, W_out)` from input `(N, C, H_in, W_in)` and grid `(N, 2K, H_out, W_out)`.

## Transformation: `GridSampleBatchedFuse` Pattern

**Anchor:** `ConcatOp` whose K inputs all come from `GridSampleOp` results.

**Match conditions:**

1. All K inputs to the concat are `GridSampleOp` results, each with exactly one use.
2. All K `GridSampleOp`s share the same image input and identical `mode`/`padding_mode`/`align_corners` attributes.
3. Each grid chain is canonical: `permute([0,3,1,2]) ← reshape ← index(lut5d, dim=3, begin=k, end=k+1, step=1)` for k=0..K-1.
4. All K index ops draw from the same `lut5d` source.
5. `lut5d` has shape `(N, H, W, K, 2)`.

**Why the permute is `[0,3,1,2]`:** The original TTIR has `transpose(-3,-1)` then `transpose(-2,-1)` on rank-4. `CanonicalizerPass` converts `TransposeOp`s to `PermuteOp`s with positive dims, and `foldConsecutivePermute` folds the two permutes: `[0,3,2,1] ∘ [0,1,3,2] = [0,3,1,2]`.

**Replacement:**

1. `reshape(lut5d, (N, H, W, 2K))` — collapse K and coords dimensions.
2. `permute([0,3,1,2])` — single permute giving `(N, 2K, H, W)`.
3. `grid_sample(image, grid_2K)` — single batched call, output `(N, K*C, H_out, W_out)`.

## Files Changed

| File | Change |
|------|--------|
| `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` | Added `GridSampleBatchedFuse` rewrite pattern (146 lines) |
| `lib/Dialect/TTIR/IR/TTIROps.cpp` | `GridSampleOp::verify()` now allows grid dim 1 = 2K (K≥1) and checks output channels = K×C |
| `lib/Dialect/TTNN/IR/TTNNOps.cpp` | TTNN `GridSampleOp::verify()` checks `batch_output_channels` flag against grid last dim |
| `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` | Detects K from TTIR grid shape dim 1 (=2K), sets `batch_output_channels=true` when K>1 |
| `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` | Added `batch_output_channels` bool attr to TTNN `GridSampleOp` |
| `include/ttmlir/OpModel/TTNN/TTNNOpModel.h` | Added `bool batchOutputChannels` param to `getOpConstraints` and `getOpRuntime` |
| `lib/OpModel/TTNN/TTNNOpModel.cpp` | Passes `batchOutputChannels` to TTNN query instead of hardcoded `false` |
| `lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp` | Passes `getBatchOutputChannels()` to OpModel calls |
| `lib/Target/TTNN/TTNNToFlatbuffer.cpp` | Serializes `batch_output_channels` to flatbuffer |
| `runtime/lib/ttnn/operations/pool/grid_sample.cpp` | Passes `batch_output_channels` from flatbuffer to TTNN runtime call |
| `include/ttmlir/Target/TTNN/operations/pool.fbs` | Added `batch_output_channels` field to `GridSampleOp` flatbuffer table |
| `lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp` | Emits `batch_output_channels` arg in C++ codegen |
| `lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp` | Emits `batch_output_channels` kwarg in Python codegen |

## Why the OpModel Fix Was Critical

TTNN internally computes output shape and shard spec based on `batch_output_channels`. With `batch_output_channels=false` and grid last dim = 16 (K=8), TTNN computed `output_shard_width=64` while the tensor had physical width 512, causing:

```
Shard width 64 must match physical width 512 for height sharded
```

in `tensor_spec.cpp`. The fix passes the actual flag so TTNN computes `output_shard_width = 64*8 = 512` correctly.

## TTNN IR Result

```mlir
// Group 0 (of 4):
%12 = "ttnn.grid_sample"(%9, %11)
  <{align_corners = true, batch_output_channels = true, mode = "nearest", padding_mode = "zeros"}>
  : (tensor<1x96x96x64xbf16, ROW_MAJOR_DRAM>, tensor<1x128x64x16xbf16, ROW_MAJOR_DRAM>)
  -> tensor<1x128x64x512xbf16, HEIGHT_SHARDED_L1>
```

Input `(1,96,96,64)` NHWC, grid `(1,128,64,16)` where 16=2×8, output `(1,128,64,512)` where 512=64×8.

## Result

**11.94 FPS** at `opt_level_2 + BFloat16 + HiFi3 + fp32_dest_acc + trace_enabled` (was previously failing with shard mismatch error).
