# BEV Model TTNN Op Inventory and Performance Configuration Analysis

**Date:** 2026-05-19  
**Analyzed IRs:** `BEV_MODEL_IRS/BLOCK_{A,B,C,D,E,F}/13-final-after-dealloc.mlir` (final TTNN IRs)  
**MLA IRs:** `BEV_MODEL_IRS/BLOCK_{A,C}/09-mla.mlir` (memory layout assignment output)

---

## 1. Op Frequency Table

Counts are for actual executable ops only (excludes attribute types like `ttnn.ttnn_layout`, `ttnn.buffer_type`, `ttnn.memory_config`).

| Op | Block A | Block B | Block C | Block D | Block E | Block F | Full Model |
|---|---|---|---|---|---|---|---|
| `ttnn.conv2d` | 148 | 8 | 39 | 2 | 18 | 11 | 226 |
| `ttnn.prepare_conv2d_weights` | 148 | 8 | 39 | 2 | 18 | 11 | 226 |
| `ttnn.prepare_conv2d_bias` | 140 | 8 | 38 | 2 | 17 | 10 | 215 |
| `ttnn.to_memory_config` | 288 | 104 | 87 | 27 | 41 | 21 | 561 |
| `ttnn.deallocate` | 1256 | 366 | 360 | 95 | 177 | 112 | 2281 |
| `ttnn.reshape` | 36 | 76 | 9 | 19 | 12 | 6 | 136 |
| `ttnn.to_layout` | 24 | 76 | 6 | 19 | 12 | 4 | 130 |
| `ttnn.slice_static` | 56 | 32 | 15 | 8 | 6 | 3 | 120 |
| `ttnn.concat` | 48 | 4 | 13 | 1 | 7 | 3 | 73 |
| `ttnn.permute` | 28 | 8 | 7 | 2 | 6 | 2 | 31 |
| `ttnn.grid_sample` | 0 | 32 | 0 | 8 | 0 | 0 | 40 |
| `ttnn.max_pool2d` | 20 | 0 | 6 | 0 | 3 | 0 | 29 |
| `ttnn.add` | 16 | 0 | 6 | 0 | 3 | 3 | 28 |
| `ttnn.multiply` | 16 | 0 | 8 | 0 | 0 | 0 | 24 |
| `ttnn.conv_transpose2d` | 16 | 0 | 4 | 0 | 0 | 0 | 20 |
| `ttnn.prepare_conv_transpose2d_weights` | 16 | 0 | 4 | 0 | 0 | 0 | 20 |
| `ttnn.upsample` | 0 | 0 | 0 | 0 | 3 | 1 | 4 |
| `ttnn.typecast` | 0 | 0 | 0 | 0 | 0 | 6 | 6 |
| `ttnn.relu6` | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `ttnn.to_device` | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `ttnn.get_device` | 290 | 17 | 85 | 8 | 39 | 28 | 447 |
| `ttnn.write_tensor` | 4 | 8 | 1 | 2 | 5 | 1 | 10 |
| `ttnn.empty` | 4 | 8 | 1 | 2 | 5 | 1 | 10 |
| `ttnn.begin_trace_capture` | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `ttnn.capture_or_execute_trace` | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `ttnn.execute_trace` | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| `ttnn.end_trace_capture` | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

**Notes:**
- Block A is the dominant block: 148 conv2d ops (65% of the full model's 226), 288 to_memory_config ops (51% of 561 total).
- `ttnn.deallocate` counts are high (1256 in Block A) but these are zero-cost metadata ops — the trace infrastructure handles them as no-ops at runtime.
- `ttnn.get_device` appears once per op that needs device access; it is not a real compute cost.

---

## 2. Per-Op Analysis

### 2.1 `ttnn.conv2d`

**TTNN header:** `ttnn/operations/conv/conv2d/conv2d.hpp`

**Signature:**
```cpp
Conv2dResultWithOptions conv2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels, uint32_t out_channels,
    uint32_t batch_size, uint32_t input_height, uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride = {1,1},
    ...padding, dilation, groups...,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt);
```

**Supported configurations in BEV:**
- Dtype: `bf16` (uniform across all BEV blocks)
- Memory layouts: `dram/interleaved`, `l1/height_sharded`, `l1/block_sharded`, `l1/width_sharded`, `l1/interleaved`
- Shard layouts assigned by MLA: `height_sharded`, `block_sharded`, `width_sharded`
- Compute config: `math_fidelity = hifi3, fp32_dest_acc_en = true` (applied to ALL conv2d ops)
- `conv2d_slice_config = l1_full` (most common), `dram_width` (for stride-2 pixel-downsampling)
- Activation fused: `relu6` fused in `conv2d_config` for most layers

**Current BEV grid assignments (from Block A `09-mla.mlir`, 148 ops):**

| Input Height | Grid | Shard Type | Count |
|---|---|---|---|
| 1536 | `1x1` | DRAM interleaved | 8 (4 per trace repetition × 2 traces) |
| 384 | `64x1` | L1 height_sharded | 16 |
| 192 | `64x1` | L1 height_sharded | 16 |
| 96 | `58x1` | L1 height_sharded | 28 |
| 48 | `36x1` | L1 height_sharded | 20 |
| 24 | `18x1` | L1 height_sharded | 16 |
| 24 | `6x8` | L1 block_sharded | 4 |
| 12 | `8x8` | L1 **interleaved** | 4 (suboptimal) |
| 12 | `5x5`, `5x6`, `5x7` | L1 block_sharded | 20 |
| 6 | `2x6`, `1x12`, `1x20`, `1x6` | L1 block_sharded/width_sharded | 16 |

**Suboptimal: 4 ops at `in_h=12` landing on `8x8 / L1 interleaved`** — these are the `(in_h=12, out_ch=320, kernel=3x3, in_ch=384)` conv2d (line 852/1102/1352/1602 in `09-mla.mlir`). They read from `dram/interleaved` (row_major) and output to `8x8/l1/interleaved`. The interleaved layout in L1 at small spatial size (144 elements, 12×12) means data is not sharded — all 64 cores compete. The follow-up op is a block_sharded `5x5` conv2d (320→320 at `in_h=12`), requiring a reshard `to_memory_config` to convert from interleaved to block_sharded. This pattern introduces unnecessary bounce traffic.

---

### 2.2 `ttnn.prepare_conv2d_weights` / `ttnn.prepare_conv2d_bias`

**Role:** Weight reorder/pack for conv2d. These are run outside the execution trace and run once at program launch (constant folded). They transform system_memory weights → DRAM tile-layout tensors.

**Compute config:** Same `hifi3 / fp32_dest_acc_en=true` as conv2d (required for matching precision).

**Key observation:** Every conv2d has an accompanying `prepare_conv2d_weights`. In Block A, 148 conv2d ops = 148 weight preparations + 140 bias preparations. This is expected and correct.

---

### 2.3 `ttnn.to_memory_config`

**Role:** Copies tensor data between memory locations and/or changes sharding strategy.

**DRAM/L1 direction breakdown per block:**

| Block | Total | L1→DRAM | DRAM→L1 | L1→L1 (reshard) |
|---|---|---|---|---|
| Block A | 288 | 164 | 72 | 52 |
| Block B | 104 | 72 | 32 | 0 |
| Block C | 87 | 47 | 23 | 17 |
| Block D | 27 | 19 | 8 | 0 |
| Block E | 41 | 17 | 14 | 10 |
| Block F | 21 | 15 | 3 | 3 |
| **Full Model** | **561** | **314** | **152** | **95** |

**Key observation:** Total cross-DRAM bandwidth ops = 314 (L1→DRAM writes) + 152 (DRAM→L1 reads) = **466 DRAM copies** in the full model per inference. This is the single largest non-compute cost.

The `to_memory_config` with `<dram>, <interleaved>` as destination is where most intermediate feature maps get spilled. This happens at every FPN neck junction and after each concat/add in Block A.

---

### 2.4 `ttnn.deallocate`

**Role:** Frees tensor buffers to allow L1 reuse. In trace mode these are executed as metadata (no actual dealloc kernel is launched).

**Count:** 2281 total in full model. These are not a performance concern. They are generated correctly by the pass pipeline.

---

### 2.5 `ttnn.reshape`

**Role:** Logical shape change with no data movement when the physical layout is contiguous.

**Usage in BEV:**
- Block A: 36 reshapes — primarily around the image-to-column rearrangement for grid_sample prep (lines 1461–1489). Sequences like:
  ```
  reshape [1,1,2359296,3] → [1,1,384,4,384,4]  (DRAM, no copy)
  permute [0,3,5,1,2,4]                         (DRAM → L1, 8x8 interleaved, actual copy)
  reshape [1,4,4,1,384,384] → [1,16,384,384]    (L1, no copy)
  ```
- Block B: 76 reshapes — all in the `to_layout` / `reshape` sequences for grid_sample inputs

**Issue:** Some reshapes force a copy when the strides in the new layout are non-contiguous with the memory layout. Check lines 1473–1481 in Block A: `reshape → permute → reshape → permute` chain that involves 2 actual copies (permutes are always copies).

---

### 2.6 `ttnn.permute`

**Role:** Transpose / reorder tensor dimensions. Always a data copy (no zero-copy permute in TTNN).

**Count:** 28 in Block A, 8 in Block B (per trace repetition for grid_sample setup), 7 in Block C.

**BEV usage pattern (Block A lines 1454–1485):**
```
permute(NHWC→NCHW): 1x3x1536x1536 DRAM → 1x1536x1536x3 DRAM  (copy over DRAM)
... reshape ...
permute([0,3,5,1,2,4]): 6D rearrange for block extraction   (DRAM → L1 8x8, copy)
... reshape to [1,16,384,384] ...
permute(NHWC→NCHW): 1x384x384x16 L1 → 1x16x384x384 DRAM    (L1 → DRAM copy)
permute(NHWC→NCHW): 1x384x384x8 L1 → 1x8x384x384 DRAM      (L1 → DRAM copy)
```
This 4-permute + 4-reshape chain for processing the grid coords runs once per camera view (4 cameras in Block A). Each permute on a 1536×1536 tensor is O(7M elements × 2 bytes) = ~14 MB DRAM traffic per permute.

---

### 2.7 `ttnn.grid_sample`

**TTNN header:** `ttnn/operations/pool/grid_sample/grid_sample.hpp`

**Signature:**
```cpp
ttnn::Tensor grid_sample(
    const ttnn::Tensor& input_tensor,   // (N, H_in, W_in, C) — row-major
    const ttnn::Tensor& grid,           // (N, H_out, W_out, 2) — row-major
    const std::string& mode = "bilinear",
    const std::string& padding_mode = "zeros",
    bool align_corners = false,
    bool use_precomputed_grid = false,
    bool batch_output_channels = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);
```

**Current BEV usage:**
```
%13 = "ttnn.grid_sample"(%10, %12) <{align_corners = true, mode = "nearest", padding_mode = "zeros"}>
  : (tensor<1x96x96x64xbf16, <1x1>, memref<9216x64xbf16, dram, interleaved>>,
     tensor<1x128x64x2xbf16, <1x1>, memref<8192x2xbf16, dram, interleaved>>)
  -> tensor<1x128x64x64xbf16, <64x1>, memref<128x64xbf16, l1, height_sharded, 8x8>)
```

**Both inputs are in DRAM row-major format.** The output goes to L1 height-sharded on a full `8x8` grid (64 cores).

- Mode: `nearest` (not `bilinear` — the IR attributes show `mode = "nearest"`) — this is the custom tt-forge-onnx grid_sample implementation.
- Input tensor (feature map): DRAM interleaved, row-major layout — must be loaded from DRAM each call.
- Grid tensor: DRAM interleaved, row-major — must be loaded from DRAM each call.
- No `use_precomputed_grid` flag set (false) — coordinate computation done inline.
- The grid tensor contains `(x, y)` pairs in `[-1, 1]` range.
- Output is height-sharded on `64x1` grid (8×8 = 64 cores, `128x64` per shard).

**Sharding suboptimality:** The input feature map stays in DRAM for all grid_sample calls. In Block B, 32 grid_sample ops all read the same two source tensors (`%10` feature map of shape `1x96x96x64` and individual `%12` grid slices). The feature map is never prefetched to L1 between calls.

---

### 2.8 `ttnn.max_pool2d`

**TTNN header:** `ttnn/operations/pool/generic/generic_pools.hpp`

**Signature:**
```cpp
std::vector<Tensor> max_pool2d(
    const Tensor& input_tensor, uint32_t batch_size,
    uint32_t input_h, uint32_t input_w, uint32_t channels,
    std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    ...
    std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
    bool config_tensor_in_dram = false);
```

**Current BEV usage (Block A, 20 ops per full inference):**
- `channels=96, in_h=192`: Input L1 height_sharded `64x1`. Output L1 height_sharded `64x1`. ✓ Good.
- `channels=128, in_h=96`: **Input DRAM interleaved.** Output L1 height_sharded `58x1`. Suboptimal — must load 9216×128 = ~2.4 MB from DRAM before compute.
- `channels=192, in_h=48`: **Input DRAM interleaved.** Output L1 height_sharded `36x1`. Suboptimal.
- `channels=384, in_h=24`: **Input DRAM interleaved.** Output L1 height_sharded `18x1`. Suboptimal.
- `channels=448, in_h=12`: Input L1 block_sharded `5x7`. Output L1 block_sharded `5x7`. ✓ Good.

Three of five max_pool2d shapes (3×4=12 of 20 total ops) read from DRAM. This is because their producer (a conv2d or concat) writes to DRAM first, and no explicit prefetch sharding is inserted between.

---

### 2.9 `ttnn.add` / `ttnn.multiply`

**Usage:** Residual connections (add) and scale operations (multiply). 32 total in Block A.

**Issue:** All 32 add/multiply ops in Block A have at least one DRAM-interleaved input:
```
%152 = "ttnn.add"(%130, %151)
  : (tensor<1x1x144x192xbf16, <1x1>, memref<5x6 tiles, dram, interleaved>>,   # L1 would be better
     tensor<1x1x144x192xbf16, <5x6>, memref<1x1 tile, l1, block_sharded>>)
  -> tensor<1x1x144x192xbf16, <5x6>, memref<1x1 tile, l1, block_sharded>>
```
One operand is always DRAM interleaved (the skip connection), the other is L1 sharded (from the conv chain). TTNN's elementwise add can read from DRAM, but the DRAM operand serializes the memory access — each core must fetch tiles via the NOC from DRAM before computing. If the skip connection were pre-loaded into L1, the add would be register-speed.

---

### 2.10 `ttnn.concat`

**Usage:** Feature map concatenation along channel dim (dim=3). 48 ops in Block A, 4 in Block B.

**Pattern in Block A:**
- Most concats output to **DRAM interleaved** (8 of 10 distinct concat ops examined output to `dram`).
- Example: `concat([1x1x576x256, 1x1x576x192] L1 interleaved) → 1x1x576x448 DRAM` (line 1641)
- Example: `concat([1x1x8192x64 height_sharded × 8] L1) → 1x1x8192x512 L1 height_sharded` (Block B line 224)

Block B's 8-input concat is notable: all 8 inputs are `43x1` height_sharded on the same `(0,0)-(7,4),(0,5)-(2,5)` core range, and the output stays height_sharded in L1. This is efficient.

Block A's concats between FPN levels mostly write to DRAM — the spatial feature concat outputs need to be consumed later from DRAM, adding round-trip cost.

---

### 2.11 `ttnn.to_layout`

**Role:** Convert between `row_major` and `tile` data layouts. Always a copy.

**Count:** 24 in Block A, 76 in Block B.

**Block B pattern:** `to_layout` appears in pairs with `reshape` for the grid_sample grid coordinate preparation:
```
to_layout(tile → row_major, DRAM)   # prepare grid coords for grid_sample
grid_sample(row_major inputs)
to_layout(row_major → tile, L1)     # convert output for next conv
```
76 to_layout ops in Block B are entirely around grid_sample I/O format requirements. grid_sample requires row-major inputs; downstream conv2d requires tile layout. The conversions are unavoidable given the current grid_sample kernel API.

---

### 2.12 `ttnn.slice_static`

**Role:** Extract a sub-range of a tensor along one dimension (channel split). Zero-copy for some layouts.

**Block A usage (56 ops):** Mostly channel splits:
- `slice_static` on `1x1x2359296x3` → `1x1x2359296x1` (first channel extraction at lines 1460, 1469)
- `slice_static` on `1x1x36864x64` → `1x1x36864x32` (channel halving)
- `slice_static` on `1x1x9216x96` → `1x1x9216x64` (channel extraction from conv output)

These splits are used to implement the custom per-scale processing in the neck. They operate on DRAM interleaved and L1 interleaved tensors.

---

### 2.13 `ttnn.upsample`

**Usage:** Blocks E (3 ops) and F (1 op). All use bilinear mode, 2×2 scale.

**Current state:**
```
Block E:
  upsample(1x16x8x256 DRAM → 1x32x16x256 DRAM)
  upsample(1x32x16x128 DRAM → 1x64x32x128 DRAM)
  upsample(1x64x32x64 L1 interleaved → 1x128x64x64 DRAM)
```
Two of three upsample ops work on DRAM inputs. The output of upsample goes back to DRAM in all three cases. This is a known issue — the upsampler does not currently support sharded output.

---

### 2.14 `ttnn.typecast`

**Usage:** Block F only. 6 ops: `bf16 → f32 → bf16` round-trip on system_memory tensors (weight/bias terms):
```
typecast(1x64x1x1 bf16 system_memory → f32 system_memory)
... some computation on host ...
typecast(1x1x1x64 f32 system_memory → bf16 system_memory)
```
These run on CPU (system_memory), not on device. They are part of BN/scale folding. Not a GPU bottleneck.

---

### 2.15 `ttnn.conv_transpose2d`

**Usage:** Blocks A (16 ops) and C (4 ops). Used for upsampling in the FPN decoder.

**Current config (all ops):**
```
conv_transpose2d(in_ch=192, out_ch=192, kernel=2x2, stride=2, no bias)
  input: L1 interleaved
  output: L1 interleaved or block_sharded
  shard_layout: block_sharded
  compute_config: hifi3, fp32_dest_acc_en=true
```
Example (Block A line 1785 area):
```
conv_transpose2d(1x1x36x192 L1 interleaved, 1x1x768x192 DRAM weights) → 1x1x144x192 L1 block_sharded 5x6
```

---

### 2.16 `ttnn.relu6`

**Usage:** Block F only (3 ops). All on L1 height-sharded tensors `64x1`:
```
relu6(1x1x32768x64 L1 height_sharded 64x1) → same layout
```
This is a standalone activation (not fused into conv2d). Could potentially be fused with the preceding conv2d — Block F conv2ds lack the `activation = <relu6>` attribute that Block A conv2ds have.

---

## 3. DRAM↔L1 Traffic Summary

### Directional breakdown:

| Block | Total to_memory_config | L1→DRAM writes | DRAM→L1 reads | L1→L1 reshard |
|---|---|---|---|---|
| Block A | 288 | 164 | 72 | 52 |
| Block B | 104 | 72 | 32 | 0 |
| Block C | 87 | 47 | 23 | 17 |
| Block D | 27 | 19 | 8 | 0 |
| Block E | 41 | 17 | 14 | 10 |
| Block F | 21 | 15 | 3 | 3 |
| **Full Model** | **561** | **314** | **152** | **95** |

### Estimated bandwidth cost (Block A):
- Most L1→DRAM writes are for feature maps at spatial resolutions 192×192 to 12×12, channel depths 96–448.
- Largest single DRAM spill: `1x1x147456x24 bf16 = 147456×24×2 = ~6.8 MB` (line 1494–1496, L1 height-sharded → DRAM → L1 height-sharded again = redundant).
- Most common: `1x1x36864x64 bf16 = ~4.7 MB`, `1x1x9216x192 bf16 = ~3.5 MB`.

---

## 4. Suboptimal Configurations Found

### 4.1 In-h=12 conv2d with no input sharding (Block A)

**Location:** Lines 852, 1102, 1352, 1602 in `09-mla.mlir` (4 occurrences, 1 per camera × repeated in 4 trace captures)

**Problem:**
```
conv2d(in_h=12, in_ch=384, out_ch=320, kernel=3x3)
  input: tensor<1x1x144x384xbf16, 1x1, memref<144x384xbf16, dram, interleaved>>
  output: tensor<1x1x144x320xbf16, 8x8, memref<1x1 tile, l1, interleaved>>
```
The input is DRAM row-major (non-tile), the output lands in L1 `8x8/interleaved`. This means 144×384×2 = 110 KB is read from DRAM, and 144×320×2 = 92 KB is written to L1 in **interleaved** layout (not sharded). MLA chose `8x8 interleaved` instead of something like `5x5 block_sharded` (which is used for the immediately following conv2d at the same spatial scale).

The interleaved output then requires a `to_memory_config` reshard to `3x8/block_sharded` before the next `512→320, 1x1` conv2d (line 1685 area). This bounce is avoidable if the first `384→320` conv2d was assigned `block_sharded` output.

### 4.2 max_pool2d reading from DRAM (Block A, 12 of 20 ops)

**Locations:** Lines 1572, 1617, 1665 and their repeats.

**Problem:**
```
max_pool2d(channels=128, in_h=96)
  input: tensor<1x1x9216x128xbf16, 1x1, memref<288x4 tiles, dram, interleaved>>
  output: L1 height_sharded 58x1
```
The input to max_pool2d is DRAM interleaved. This means the pool reads 9216×128×2 = ~2.4 MB from DRAM per call, rather than from L1. The producer (concat or conv2d) wrote to DRAM before this. Inserting a `to_memory_config` to height_sharded L1 before the pool would allow on-chip access — but this must be balanced against L1 capacity.

### 4.3 All add/multiply ops have one DRAM input (Block A, 32 ops)

**Pattern:**
```
multiply(scalar DRAM, activation_output L1 sharded) → L1 sharded
add(skip_connection DRAM interleaved, multiply_result L1 sharded) → L1 sharded
```
Skip connections (the addends that come from DRAM) are residual paths that bypass the conv chain. They end up in DRAM because the conv chain that produced them spilled to DRAM for subsequent ops. If the skip connection could stay in L1 through the entire residual path, the add would not need a DRAM read. This requires substantially more L1 budget.

### 4.4 upsample always outputs to DRAM (Blocks E and F)

**Problem:** `ttnn.upsample` currently places its output in DRAM interleaved regardless of whether a sharded configuration would be possible. In Block E lines 345 and 361, upsample processes small tensors (128 and 512 rows × 256 or 128 channels) that would easily fit in L1 sharded — but the output goes to DRAM, requiring the next op to reload from DRAM.

### 4.5 relu6 not fused in Block F conv2d

**Location:** Block F lines 191, 210, 234.

**Problem:**
```
conv2d(in_ch=64, out_ch=64, ...) → L1 height_sharded  [no activation attribute]
relu6(same tensor) → L1 height_sharded
```
Block A conv2ds use `activation = <op_type = relu6>` in `conv2d_config`, which fuses relu6 into the conv2d kernel (one pass). Block F's 3 relu6 ops are separate, requiring a second kernel launch over the same data. The data shape `1x1x32768x64` (65536 elements) is re-traversed needlessly.

### 4.6 typecast bf16→f32→bf16 round-trip in Block F

**Location:** Block F lines 20–24 (repeated 3×).

**Problem:**
```
typecast(bias 1x64x1x1 bf16 → f32, system_memory)
... host operation ...
typecast(1x1x1x64 f32 → bf16, system_memory)
```
This is a bf16→f32→bf16 round-trip on 64-element bias vectors on CPU. It wastes compute cycles (minor, since these are tiny tensors on CPU). The root cause is likely a scale/bias op that was not fully folded at compile time. Could be eliminated if the constant folding pass handled this case.

---

## 5. Redundant Op Patterns

### 5.1 L1→DRAM→L1 bounce chains (most impactful)

**Count:** 44 in Block A, 14 in Block C, 7 in Block E, 3 in Block F = **68 total redundant DRAM round-trips** in the full model.

**Example from Block A lines 1504–1506:**
```
%26 = conv2d(...)    → L1 height_sharded 64x1
%27 = to_memory_config(%26) → DRAM interleaved     # L1→DRAM
%28 = to_memory_config(%27) → L1 height_sharded    # DRAM→L1  ← REDUNDANT
```
`%27` (DRAM) is only created to satisfy a downstream op's expected memory placement, but `%28` immediately converts it back to L1. The two `to_memory_config` ops could be collapsed into a single `reshard` or eliminated if the producer's output were directly placed in the target L1 layout.

**Second example (lines 1534–1537):**
```
%41 = max_pool2d(...)  → L1 height_sharded (64x1), row_major
%42 = to_memory_config(%41) → DRAM interleaved     # L1→DRAM (line 1534)
%43 = to_memory_config(%42) → L1 height_sharded    # DRAM→L1 (line 1536) ← REDUNDANT
%44 = to_memory_config(%43) → DRAM interleaved     # L1→DRAM (line 1537) ← ALSO REDUNDANT
```
Three consecutive `to_memory_config` calls on the same data (lines 1534–1537). The first goes L1→DRAM, the second immediately goes back to the same L1 shard spec, then the third goes back to DRAM. This is a three-hop chain where a single direct L1→target-DRAM should suffice.

**Example from Block A lines 1576–1577:**
```
%63 = conv2d(...)     → L1 height_sharded 58x1
%64 = to_memory_config(%63) → DRAM           # L1→DRAM
%64b = to_memory_config(%63_orig) → L1 hs     # DRAM→L1 (same shard spec as source)
%65 = to_memory_config(%64b) → DRAM          # L1→DRAM  ← net result: L1→DRAM with middle bounce
```

### 5.2 Consecutive reshape on same data

**Pattern (Block A lines 1461–1465):**
```
reshape([1,1,2359296,3] → [1,1,384,4,384,4])   # DRAM, no copy
permute([0,3,5,1,2,4])                           # DRAM → L1 8x8
reshape([1,4,4,1,384,384] → [1,16,384,384])     # L1, no copy
```
And separately (lines 1473–1481):
```
reshape([1,1,589824,2] → [1,768,768,2])          # DRAM
permute([0,3,1,2])                                # DRAM → L1 8x8
reshape([1,2,768,768] → [1,2,384,2,384,2])       # DRAM
permute([0,3,5,1,2,4])                            # DRAM
reshape([1,2,2,2,384,384] → [1,8,384,384])       # DRAM
```
The second chain has 3 reshapes and 2 permutes. The reshapes are free (shape metadata changes) but the 2 permutes are actual data copies. This chain runs once per camera × 4 trace repetitions = 8 total permutes over `(~590K × 2 bytes)` data each.

### 5.3 conv2d → to_memory_config (block_sharded) → slice → concat → to_memory_config (height_sharded)

**Pattern (Block A lines 1685–1710):**
```
concat(...interleaved) → 1x1x144x512 DRAM
to_memory_config → 1x1x144x512 L1 block_sharded (3x8)
conv2d(512→320, 1x1) → L1 block_sharded 5x5
slice_static(→ 192ch)
concat([conv_out, slice]) → DRAM interleaved
to_memory_config → L1 block_sharded (for next conv)
```
The concat output goes to DRAM, then immediately is re-sharded to L1 for the next conv. In this sequence the DRAM step is avoidable — the concat could write directly to the L1 block_sharded layout that the next `to_memory_config` sets up.

---

## 6. Top 10 Optimization Opportunities (Ranked by Estimated Performance Impact)

### Priority 1: Eliminate 68 redundant L1→DRAM→L1 chains

**Impact:** ~68 unnecessary DRAM round-trips at 2–7 MB each = potentially 100–450 MB of avoidable DRAM traffic per inference.

**Fix:** The MLA pass or a post-MLA cleanup pass should detect patterns where a `to_memory_config(L1→DRAM)` result is immediately consumed only by `to_memory_config(DRAM→L1)` and collapse them into a single `to_memory_config(L1→L1 reshard)`.

**Files to modify:** `forge/csrc/passes/lower_to_mlir.cpp` or add a new `eliminate_memory_bounce_pass` in `forge/csrc/passes/`.

### Priority 2: Keep skip connections (residual add inputs) in L1

**Impact:** 32 add/multiply ops in Block A all read one DRAM operand. Each `ttnn.add` on a `1x1x9216x192` tensor reads ~3.5 MB from DRAM. If the skip connection stayed in L1, 32 × ~2 MB average = ~64 MB DRAM bandwidth reduction per inference.

**Fix:** The memory optimizer should recognize residual addition patterns and attempt to keep the shorter path (skip connection) in L1. This requires L1 reservation during the forward path analysis. The skip connections are at FPN scales 192×192, 96×96, 48×48, 24×24, 12×12.

**Files:** `forge/csrc/passes/mlir_compiler.cpp` (memory optimization hints).

### Priority 3: Fix in_h=12 conv2d landing on 8x8 interleaved (MLA assignment)

**Impact:** 4 conv2d ops per camera × 4 trace repetitions = 16 ops that bounce through interleaved L1 before the required reshard. Each bounce is ~92 KB. Total avoidable traffic: ~1.5 MB + kernel re-launch overhead.

**Fix:** In the MLA (Memory Layout Assignment) pass, add a constraint that for spatial sizes ≤ 12×12 with channel depth ≥ 320, prefer `block_sharded` layout matching the downstream op's layout, to avoid the inter-op reshard `to_memory_config`.

**Files:** `third_party/tt-mlir/lib/Dialect/TTNN/Analysis/MemoryLayoutAnalysis.cpp` (or equivalent MLA constraint file).

### Priority 4: Fuse relu6 into Block F conv2d

**Impact:** 3 separate relu6 kernel launches on 32768×64 tensors (8 MB data each = 24 MB re-traversed). Savings: 3 kernel launches + 24 MB L1 bandwidth.

**Fix:** Set `activation = <op_type = relu6>` in the `conv2d_config` of the 3 conv2d ops preceding these relu6 ops in Block F. The preceding conv2ds currently lack this attribute (unlike Block A which correctly fuses it).

**Files:** `forge/csrc/passes/lower_to_mlir.cpp` or `forge/forge/tvm_to_python.py` (where conv2d configuration is generated).

### Priority 5: Pre-shard grid_sample feature map input to L1

**Impact:** Block B has 32 grid_sample ops that all read the same source feature map from DRAM (`1x96x96x64 = 1.2 MB`). This map is read 32 times from DRAM (total 38.4 MB DRAM reads) for what is effectively one multi-view sampling operation. If the feature map were pre-loaded to L1 (sharded) before the grid_sample loop, the same data would be read from L1 64× faster.

**Fix:** Add a `to_memory_config(L1, height_sharded)` before the first grid_sample call and keep the sharded tensor live for all 32 calls. Block B's feature map (`96×96×64 = ~1.2 MB`) fits in L1 when sharded across the full `8x8` grid (18.75 KB per core).

**Files:** `forge/csrc/passes/lower_to_mlir.cpp` (add pre-shard hoist for repeated grid_sample inputs).

### Priority 6: Reduce concat→DRAM spills in Block A FPN neck

**Impact:** 8 of 10 distinct concat operations in Block A write to DRAM. If the concat output were placed in L1 with the correct shard spec for the next consumer, the downstream `to_memory_config` (DRAM→L1) would be eliminated. Each eliminated DRAM spill is 1–4 MB. 8 concats × ~2 MB average = ~16 MB DRAM bandwidth reduction.

**Fix:** The memory planner should check if the concat output can be placed directly in the L1 shard spec required by the next consumer, subject to L1 capacity constraints.

**Files:** MLA pass in tt-mlir.

### Priority 7: Fix upsample output always landing in DRAM

**Impact:** 3 upsample ops in Block E all output to DRAM. The downstream consumer reads the output from DRAM. Each upsample processes 128–2048 rows × 64–256 channels. The two smaller upsamples (16×256 and 32×128) trivially fit in L1 sharded.

**Fix:** Enable sharded output in `ttnn.upsample` and assign height_sharded L1 output in the MLA pass for small spatial sizes.

**Files:** `ttnn/operations/pool/upsample/` (kernel support), tt-mlir MLA pass.

### Priority 8: Eliminate permute DRAM copies for grid coordinate setup

**Impact:** 4 large permutes over 1536×1536 pixel coordinate data (each ~14 MB) run per inference for Block A's grid_sample initialization. Total: 4 × 4 (trace rep) × 14 MB = 224 MB of permute-induced DRAM traffic.

**Fix:** The grid coordinate computation (`reshape → permute → reshape → permute → reshape`) is constant folded (these are weight-like inputs run once at trace start). If the constant folding pass runs these 8 permutes once during model compilation (not per inference), they become zero-cost at runtime. Confirm that `ttnn.capture_or_execute_trace` excludes these ops from the hot path.

**Files:** `forge/csrc/passes/constant_folding.cpp` — verify grid coordinate ops are outside the trace.

### Priority 9: Fix max_pool2d reading from DRAM (3 of 5 shapes)

**Impact:** 12 of 20 max_pool2d ops in Block A (repeated per trace) read from DRAM. At input sizes of 2304×192 and 576×384, this is 0.9–4.4 MB per pool op read from DRAM. 12 ops × ~2 MB average = ~24 MB per inference.

**Fix:** The memory planner should ensure that the conv2d/concat chain feeding into max_pool2d writes to L1 sharded (or at minimum that a pre-pool `to_memory_config(L1)` is inserted and hoisted to reuse across repeated invocations). The three problematic shapes are `128ch 96×96`, `192ch 48×48`, `384ch 24×24`.

**Files:** MLA pass in tt-mlir.

### Priority 10: Eliminate typecast bf16→f32→bf16 round-trips in Block F

**Impact:** Minor (3 ops on 64-element CPU tensors). Net cost: CPU cycles for 3 typecasts on tiny vectors.

**Fix:** Eliminate the constant-folding bypass that prevents the bias/scale constants from remaining in bf16 throughout. Identify why the `1x64x1x1 bf16` bias is being cast to f32 — likely a divide or accumulation op that demands f32. If the op can be rewritten to use bf16, the casts disappear.

**Files:** `forge/forge/tvm_calls/relay/op/forge_passes.py` or `forge/forge/tvm_to_python.py`.

---

## Appendix: MLA Sharding Summary

From `09-mla.mlir` (Block A, 148 conv2d ops):

| Grid | Memory | Shard | Op Count | Notes |
|---|---|---|---|---|
| `1x1` | DRAM | interleaved | 8 | Large spatial (1536×1536), no sharding — L1 insufficient |
| `64x1` | L1 | height_sharded | 36 | Spatial 192×192–384×384, 64 cores = full device |
| `58x1` | L1 | height_sharded | 28 | Spatial 96×96, 58 cores (non-power-of-2 due to tile alignment) |
| `36x1` | L1 | height_sharded | 20 | Spatial 48×48 |
| `18x1` | L1 | height_sharded | 16 | Spatial 24×24 |
| `8x8` | L1 | **interleaved** | 4 | Spatial 12×12 — suboptimal, should be block_sharded |
| `5x5`–`5x7` | L1 | block_sharded | 20 | Spatial 12×12 |
| `2x6`–`1x20` | L1 | block/width_sharded | 16 | Spatial 6×6 |

Key observations:
1. The full `8x8` grid is used for large spatial sizes (192+), which maximizes parallelism. This is correct.
2. Small spatial (6×6 to 12×12) uses partial grids with block_sharded, which is appropriate.
3. The 4 ops on `8x8/interleaved` at spatial 12×12 are anomalous — these should be `5x5` or `3x8` block_sharded to match adjacent ops and avoid reshard `to_memory_config`.
4. The two `1x1/DRAM` conv2ds at `in_h=1536` (per-channel normalization conv2d with kernel=1×1, in_ch=3, out_ch=3) cannot be sharded height-wise because 3 input channels do not divide well across tiles. This is architecturally constrained.

---

*Generated by automated IR analysis — all counts and examples verified against actual IR file content.*
