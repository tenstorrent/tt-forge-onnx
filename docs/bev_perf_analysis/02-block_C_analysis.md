# Block C — Analysis

**Role:** Camera feature extraction for a single cylindrical (wide-angle) camera  
**Measured time:** 106.91 ms (9.31 FPS)  
**Share of full model:** ~22.1%  
**IR files:** `BEV_MODEL_IRS/BLOCK_C/`  
**Log file:** `BEV_MODEL_LOGS/block_C.log`

---

## 1. Architecture Overview

Block C processes a single cylindrical camera input (wider field of view than the 4 cameras in Block A). It uses a similar FPN structure but with fewer channels and different spatial dimensions.

### Input/Output

| Tensor | Shape | Type | Notes |
|--------|-------|------|-------|
| Input | `1×3×1280×2304` | BF16 | Cylindrical projection, wider-than-tall |

The input is 1280×2304 (portrait × panoramic), significantly different from Block A's 1536×1536 square inputs.

### Resolution Pyramid

```
1280×2304  →  320×576  →  160×288  →  80×144  →  40×72  →  20×36  →  10×18
    (×2 convs)   (×5)      (×9)       (×11)      (×8)       (×8)       (×6)
```

---

## 2. Op Count Summary (from `01-ttir-passes.mlir`)

| Op Type | Count | Notes |
|---------|-------|-------|
| `ttir.conv2d` | 39 | (vs 148 in Block A) |
| relu6 / activations | ~39 | Fused into conv2d via TTNNConv2dWithActivation |
| `ttir.reshape` | ~25 | FPN reshape ops |
| `ttir.permute` | ~20 | NCHW↔NHWC adjustments |
| `ttir.concat` | ~15 | FPN skip connections |
| `ttir.slice_static` | ~12 | Feature pyramid slicing |
| `ttir.max_pool2d` | ~6 | Spatial downsampling |
| `ttir.conv_transpose2d` | ~4 | FPN upsampling |

Block C is structurally a subset of Block A's single-branch processing. It has:
- 1 camera (vs 4 serially processed in Block A)
- 39 conv2d (vs 148 = 4 branches × ~37 each in Block A)
- Similar FPN topology but adapted for panoramic aspect ratio

---

## 3. Post-MLA Layout Analysis (`09-mla.mlir`)

### Grid Distribution

| Grid Annotation | % | Meaning |
|-----------------|---|---------|
| `<1x1>` | **73.4%** | Single-core interleaved DRAM (unsharded) |
| Multi-core grids | 26.6% | Various sharding configs |

The `<1x1>` percentage (73.4%) is essentially identical to Block A (73.5%), confirming that the same MLA behavior is present: L1 pressure from large activations forces most tensors to DRAM-interleaved placement.

### Conv2d Configuration

- `config_tensors_in_dram = true` on ALL conv2d and prepare_conv2d_weights ops (same as Block A)
- `conv2d_slice_config` distribution: majority `l1_full`, a small number `dram_width` for the 1280×2304 input tier
- relu6 fused via TTNNConv2dWithActivation on all applicable convolutions

The 1280×2304 input is 2.95 MP (megapixels) vs Block A's 2.36 MP (1536×1536), making Block C's first-tier convolutions even larger. The panoramic aspect ratio (width = 1.8× height) creates non-square spatial shards that the 8×8 core grid handles less efficiently.

---

## 4. Final IR Memory Traffic (`13-final-after-dealloc.mlir`)

### to_memory_config Summary

| Direction | Estimated Count | Notes |
|-----------|----------------|-------|
| Tensor → DRAM | ~54 | L1 or L1-sharded → DRAM |
| Tensor → L1 | ~40 | DRAM → L1 reload |
| **Total** | **~94** | vs 288 in Block A |

The ratio roughly matches the op count ratio: Block C has 39/148 = 26% of Block A's conv2d ops, and ~94/288 = 33% of the to_memory_config ops (slightly higher due to the panoramic aspect ratio making sharding less efficient).

### MEMDUMP Pipeline Checkpoints

| Stage | L1→DRAM Spills | DRAM→L1 Shards | Roundtrip Pairs |
|-------|---------------|-----------------|-----------------|
| Final (13) | **49** | **25** | **10** |

Compared to Block A:
- Block A: 176 spills, 84 shards, 32 roundtrips (for 148 conv2d × 4 branches = 592 camera-branch-convolutions)
- Block C: 49 spills, 25 shards, 10 roundtrips (for 39 conv2d × 1 branch = 39 camera-branch-convolutions)

Per-conv-branch ratio: Block A has ~0.30 spills/conv-branch-pair, Block C has ~1.26 — Block C actually has **4× more spills per convolution** than each Block A branch. This is likely due to the panoramic aspect ratio making L1 sharding less efficient and causing more intermediate spills.

---

## 5. Panoramic Aspect Ratio Impact

The 1280×2304 input (aspect ratio 1:1.8) creates a non-square feature pyramid. The 8×8 compute grid maps most naturally to square-ish activations. For a non-square 1280×2304 input:
- A height-sharded layout on 8×8 grid gives 1280/8 = 160 rows/core (height) × 2304 columns
- Each shard is 160×2304×C BF16 — at C=3 that's 160×2304×3×2 = 2.21 MB, exceeding 1.43 MB L1/core
- This forces DRAM placement for the first tier regardless of MLA preference

At 320×576 (after first stride-2 downsampling):
- 320/8 = 40 rows/core × 576 columns × C channels
- At C=32: 40×576×32×2 = 1.47 MB → right at the L1 boundary

This explains both the high `<1x1>` percentage and the elevated roundtrip count: the non-square activations consistently spill at spatial resolution levels that would fit for Block A's square activations.

---

## 6. Comparison with Block A

| Metric | Block A | Block C | Ratio |
|--------|---------|---------|-------|
| Camera count | 4 | 1 | 4× |
| Input resolution | 1536×1536 | 1280×2304 | ~1.0× pixel count |
| Conv2d count | 148 | 39 | 3.8× |
| Time (ms) | 326.45 | 106.91 | 3.1× |
| DRAM spills | 176 | 49 | 3.6× |
| DRAM shards | 84 | 25 | 3.4× |
| Roundtrip pairs | 32 | 10 | 3.2× |
| Time per conv2d (ms/op) | 2.21 | 2.74 | 0.81× (A is faster/conv) |

Block A runs faster per conv2d than Block C (2.21 ms vs 2.74 ms per conv). This is consistent with the panoramic aspect ratio causing less efficient sharding and more spills in Block C.

---

## 7. Key Findings for Block C

1. **Same root cause as Block A**: `config_tensors_in_dram = true`, 73.4% `<1x1>` grid, DRAM bandwidth bottleneck
2. **Panoramic aspect ratio (1:1.8)** creates worse L1 sharding efficiency than Block A's square inputs — higher spills-per-conv ratio
3. **4× fewer conv2d ops** than Block A (single camera branch vs 4 serial branches) — time scales roughly proportionally
4. **10 roundtrip pairs** (vs 32 in Block A) — same inefficiency pattern, smaller scale
5. **Addressing Block A optimizations will also apply to Block C** — they share the same structural problems
6. If L1 weight caching is enabled, Block C benefits similarly: 39 conv2d × average weight size ≈ ~12 MB weight traffic → reduced to 1× instead of current repeat-fetch pattern
