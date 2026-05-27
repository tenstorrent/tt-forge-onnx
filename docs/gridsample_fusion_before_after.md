# GridSample Fusion: Before and After

## Overview

The BEV model uses a "LUT-based" grid sample where the sampling grid is precomputed from
camera calibration data and stored as a lookup table. For K cameras, the grid tensor has
shape `[N, 2K, H_grid, W_grid]`.

The compiler needed two separate transformations to handle this efficiently:
1. **TTIR Fusion** — collapse K independent slice+reshape ops into a single reshape
2. **TTNN Lowering** — detect batched K>1 grids and set `batch_output_channels=True`

---

## Part 1: LUT Preparation — K×(Slice+Reshape)+Concat → Single Reshape

### What the pattern looks like (TTIR, before fusion)

The BEV preprocessing extracts K slices from a combined grid tensor, reshapes each one,
then concatenates them along the spatial dimension.

**Before (K=8 example, 17 ops):**
```mlir
// Input: %grid_src : tensor<1×8×2×H×W×f32>  (combined grid for all 8 cameras)

%s0 = ttir.slice_static(%grid_src) [0:1, 0:1, :, :, :] -> tensor<1×1×2×H×W×f32>
%r0 = ttir.reshape(%s0) -> tensor<1×2×H×W×f32>

%s1 = ttir.slice_static(%grid_src) [0:1, 1:2, :, :, :] -> tensor<1×1×2×H×W×f32>
%r1 = ttir.reshape(%s1) -> tensor<1×2×H×W×f32>

// ... (s2/r2 through s7/r7) ...

%s7 = ttir.slice_static(%grid_src) [0:1, 7:8, :, :, :] -> tensor<1×1×2×H×W×f32>
%r7 = ttir.reshape(%s7) -> tensor<1×2×H×W×f32>

%lut = ttir.concat(%r0, %r1, ..., %r7, dim=1) -> tensor<1×16×H×W×f32>
//                                                         ^^^ = 2×K = 16
```

**Why this is equivalent to a reshape:**
The slice operation extracts one element along the `K` dimension (dim 1 of the 5D tensor).
The reshape eliminates that singleton. The concat re-assembles all K slices along dim 1.
Row-major layout means `[1, K, 2, H, W]` and `[1, 2K, H, W]` have identical memory
representations — no data movement occurs.

**After (1 op via `GridSampleLutSimplify`):**
```mlir
%lut = ttir.reshape(%grid_src) -> tensor<1×16×H×W×f32>
```

### Where the pattern fires

`TTIRFusing.cpp` — `GridSampleLutSimplify` rewrite pattern on `ConcatOp`.

**Pattern matching conditions:**
1. All K concat inputs are single-use `ReshapeOp`s
2. Each reshape's input is a single-use `SliceStaticOp`
3. All slice ops share the same source tensor
4. Slices are unit-width (`ends[d] - begins[d] == 1`) along exactly one dimension
5. Slice offsets are contiguous: slice `k` has `begins[sliceDim] = k`
6. Output shape matches `K × innerDim` on the concat dimension

### Impact

| Metric | Before | After |
|--------|--------|-------|
| Op count (LUT prep) | 17 ops (8 slice + 8 reshape + 1 concat) | 1 op (reshape) |
| Concat inputs | K=8 | 0 |
| Compilation time contribution | ~700s (cross-product) | <1s |
| Runtime data movement | 0 bytes (all in-place) | 0 bytes |

---

## Part 2: TTNN GridSampleOp — `batch_output_channels` for K>1

### What changed in lowering

**File:** `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

The TTIR `ttir.grid_sample` op takes a grid of shape `[N, 2K, H_grid, W_grid]`. For K>1, the
tt-metal kernel requires `batch_output_channels=True` to correctly interleave the K camera
outputs in the output tensor channels.

**Before:**
```cpp
// K was not computed; batch_output_channels was always false
rewriter.create<ttnn::GridSampleOp>(
    op.getLoc(), outputType, input, grid,
    rewriter.getStringAttr(mode),
    rewriter.getStringAttr(paddingMode),
    rewriter.getBoolAttr(alignCorners),
    rewriter.getBoolAttr(false),  // batch_output_channels always false!
    memoryConfigAttr);
```

**After:**
```cpp
// Detect K from grid shape dim[1] / 2
int64_t K = gridType.getShape()[1] / 2;
bool batchOutputChannels = (K > 1);

rewriter.create<ttnn::GridSampleOp>(
    op.getLoc(), outputType, input, grid,
    rewriter.getStringAttr(mode),
    rewriter.getStringAttr(paddingMode),
    rewriter.getBoolAttr(alignCorners),
    rewriter.getBoolAttr(batchOutputChannels),  // K>1 → true
    memoryConfigAttr);
```

**TTNN IR before:**
```mlir
%out = "ttnn.grid_sample"(%input, %grid) {
  mode = "nearest",
  padding_mode = "zeros",
  align_corners = false,
  batch_output_channels = false   // <-- wrong for K=8
} : (tensor<...>, tensor<1×16×H×W×f32>) -> tensor<...>
```

**TTNN IR after:**
```mlir
%out = "ttnn.grid_sample"(%input, %grid) {
  mode = "nearest",
  padding_mode = "zeros",
  align_corners = false,
  batch_output_channels = true    // <-- correct for K=8
} : (tensor<...>, tensor<1×16×H×W×f32>) -> tensor<...>
```

---

## Part 3: Runtime — Precomputed Grid Memory Config

**File:** `runtime/lib/ttnn/operations/pool/grid_sample.cpp`

### The precomputed grid path

The BEV model precomputes the grid once (at first trace capture) and caches it in device DRAM.
Subsequent calls reuse the cached grid.

**Before (incorrect):**
```cpp
// Problem: passing the MLA-selected shard spec to the kernel
// triggered "user-provided shard spec" in compute_output_specs,
// which produced an invalid TensorSpec (512 shards on 64 cores).
::ttnn::grid_sample(
    input, precomputedGridDevice,
    mode, paddingMode, alignCorners,
    /*use_precomputed_grid=*/true,
    /*batch_output_channels=*/false,  // always false (wrong for K>1)
    memoryConfig);                    // MLA shard spec → TT_FATAL
```

**After (correct):**
```cpp
// Let the device auto-compute its HEIGHT_SHARDED output layout.
// The output is immediately desharded to DRAM, so the auto-shard spec
// has no effect on downstream ops.
::ttnn::grid_sample(
    input, precomputedGridDevice,
    mode, paddingMode, alignCorners,
    /*use_precomputed_grid=*/true,
    batchOutputChannels,              // from flatbuffer attribute
    /*memory_config=*/std::nullopt);  // let kernel decide shard layout
```

---

## Full Data Flow: Before and After (K=8)

### Before all fixes

```
Python model (ONNX)
  │
  ▼
TTIR (after TVM→Python lowering):
  %grid_src : tensor<1×8×2×H×W×f32>
  %s0 = slice[k=0]  → %r0 = reshape  ─┐
  %s1 = slice[k=1]  → %r1 = reshape   │
  ...                                  ├── concat(dim=1) → %lut [1×16×H×W]
  %s7 = slice[k=7]  → %r7 = reshape  ─┘
  %out = grid_sample(%feat, %lut)   [wrong: batch_output_channels=False]
  │
  ▼
TTNN (at opt_level=2):
  - MemoryLayoutPropagation: 3^8 = 6561 candidates for concat → HANG (~700s)
  - validateOperation(concat, sharded) → GraphProcessor JSON → HANG
  - batch_output_channels=False → wrong output
  - Runtime: shard spec from flatbuffer → TT_FATAL @ tensor_spec.cpp:143
```

### After all fixes

```
Python model (ONNX)
  │
  ▼
TTIR (after GridSampleLutSimplify in TTIRFusing):
  %grid_src : tensor<1×8×2×H×W×f32>
  %lut = reshape(%grid_src) → tensor<1×16×H×W×f32>   // 17 ops → 1 op
  %out = grid_sample(%feat, %lut)   [batch_output_channels=True for K>1]
  │
  ▼
TTNN (at opt_level=2):
  - MemoryLayoutPropagation: no concat → no cross-product problem
  - batch_output_channels=True → correct 8-camera batched output
  - Runtime: memory_config=nullopt → auto shard → correct TensorSpec
  - Output desharded to DRAM → feeds downstream ops correctly
```

---

## Final TTIR and TTNN IR Paths (Block B)

The full IR dumps for Block B are available at:

| Stage | File |
|-------|------|
| After TTIR passes (fusing included) | `BEV_MODEL_IRS/BLOCK_B/ttir_block_B_deformed_bev_transform.mlir` |
| Final TTNN (after all passes) | `BEV_MODEL_IRS/BLOCK_B/ttnn_block_B_deformed_bev_transform.mlir` |
| After MLA | `BEV_MODEL_IRS/BLOCK_B/09-mla.mlir` |
| Final TTNN after dealloc | `BEV_MODEL_IRS/BLOCK_B/13-final-after-dealloc.mlir` |

All paths are relative to:
`/proj_sw/user_dev/pchandrasekaran/new/Forge2/tt-forge-onnx/`
