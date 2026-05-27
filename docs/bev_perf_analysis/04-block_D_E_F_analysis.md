# Blocks D, E, F — State Documentation

These three blocks together account for less than 6% of total inference time. They are documented here for completeness and to establish a baseline for future regression detection.

---

## Block D — Cylinder BEV Transform

**Role:** Cylindrical camera BEV projection (grid_sample based, similar to Block B but smaller scale)  
**Measured time:** 16.97 ms (58.37 FPS)  
**Share of full model:** ~3.5%  
**IR files:** `BEV_MODEL_IRS/BLOCK_D/`  
**Log file:** `BEV_MODEL_LOGS/block_D.log`

### Architecture

Block D transforms the cylindrical camera features (from Block C) into BEV coordinates. Unlike Block B which uses 4 camera × 8 feature levels = 32 grid_sample ops, Block D uses a simpler 8 grid_sample structure for the single cylindrical camera.

### Inputs

| Tensor | Shape | Type | Notes |
|--------|-------|------|-------|
| Camera features | `1×192×80×144` | BF16 | From Block C output |
| Sampling LUT | `1×128×64×8×2` | BF16 | Precomputed BEV coordinate grids |

### Op Summary (`01-ttir-passes.mlir`)

| Op Type | Count | Notes |
|---------|-------|-------|
| `ttir.grid_sample` | 8 | One per BEV pyramid level |
| `ttir.conv2d` | 2 | Post-projection refinement |
| `ttir.reshape` | 19 | Grid/feature preparation |
| `ttir.permute` | 19 | Layout adjustments |

### Memory Traffic (`13-final-after-dealloc.mlir`)

| Metric | Value |
|--------|-------|
| L1→DRAM Spills | 19 |
| DRAM→L1 Shards | 8 |
| Roundtrip Pairs | 0 |

The 19 spills and 8 shards correspond to:
- 8 grid_sample × ~2-3 format conversion ops each = ~19–24 expected ops
- Very clean: 0 roundtrip pairs (no wasted DRAM movement)

### Key Observations

1. **Already fast (58 FPS)** — no action required
2. **Same ROW_MAJOR kernel constraint as Block B** — each grid_sample requires the same 6-op format conversion chain
3. **8 grid_sample vs Block B's 32** — 4× fewer ops explains the 4–5× speed difference from Block B
4. **0 roundtrip pairs** — memory pattern is optimal given kernel constraints
5. **19/8 spill/shard ratio (2.4:1)** — consistent with Block B's 72/32 = 2.25:1, confirming same underlying structure

---

## Block E — Feature Aggregation

**Role:** Merge BEV features from Blocks B and D into a unified representation  
**Measured time:** 6.26 ms (156.05 FPS)  
**Share of full model:** ~1.3%  
**IR files:** `BEV_MODEL_IRS/BLOCK_E/`  
**Log file:** `BEV_MODEL_LOGS/block_E.log`

### Architecture

Block E takes the BEV features from Block B (multi-camera perspective) and Block D (cylindrical), aggregates them via convolution and feature merging, and produces a unified BEV feature representation.

### Op Summary

| Op Type | Count (estimated) | Notes |
|---------|-------------------|-------|
| `ttir.conv2d` | ~18 | Feature aggregation convolutions |
| `ttir.concat` | ~4 | Channel-wise feature merge |
| `ttir.add` | ~4 | Residual connections |
| `ttir.relu6` | ~18 | Fused into preceding conv2d |

### Memory Traffic (`13-final-after-dealloc.mlir`)

| Metric | Value |
|--------|-------|
| L1→DRAM Spills | 17 |
| DRAM→L1 Shards | 14 |
| Roundtrip Pairs | 2 |

The 2 roundtrip pairs are a minor inefficiency. At 6 ms total time, even eliminating both roundtrips would save < 0.5 ms.

### Key Observations

1. **Already fast (156 FPS)** — well within target
2. **Low spill count relative to op count** — good L1 utilization, BEV features (1×C×128×64) fit well in L1 across 64 cores
3. **2 roundtrip pairs** — minor, low priority
4. **At 128×64 spatial resolution**, each core sees 128×64/64 = 128 elements per core for height-sharded layout — trivially fits in L1
5. **No grid_sample ops** — no format conversion overhead

---

## Block F — Output Heads

**Role:** Final detection/segmentation heads producing model outputs  
**Measured time:** 5.61 ms (173.28 FPS)  
**Share of full model:** ~1.2%  
**IR files:** `BEV_MODEL_IRS/BLOCK_F/`  
**Log file:** `BEV_MODEL_LOGS/block_F.log`

### Architecture

Block F applies the final prediction heads to the aggregated BEV features from Block E, producing bounding boxes, class probabilities, velocity estimates, and other detection outputs.

### Op Summary

| Op Type | Count (estimated) | Notes |
|---------|-------------------|-------|
| `ttir.conv2d` | ~11 | Prediction head convolutions |
| `ttir.reshape` | ~8 | Output format reshaping |
| `ttir.permute` | ~6 | Layout adjustment |

### Memory Traffic (`13-final-after-dealloc.mlir`)

| Metric | Value |
|--------|-------|
| L1→DRAM Spills | 15 |
| DRAM→L1 Shards | 3 |
| Roundtrip Pairs | 0 |

Block F has the cleanest memory pattern of all blocks:
- Only 3 DRAM→L1 shards (most outputs stay in L1 or go directly to host)
- 0 roundtrip pairs
- 15 spills are the final output tensors moving to DRAM for host readback

### Key Observations

1. **Fastest block (173 FPS)** — well within target, no action required
2. **Cleanest memory pattern** — 0 roundtrips, only 3 DRAM shards
3. **Small spatial resolution** — output heads operate on 128×64 or smaller BEV features, which fit easily in L1
4. **11 conv2d** — smallest conv2d count of all blocks

---

## Comparative Summary: D vs E vs F

| Metric | Block D | Block E | Block F |
|--------|---------|---------|---------|
| Time (ms) | 16.97 | 6.26 | 5.61 |
| FPS | 58.37 | 156.05 | 173.28 |
| L1→DRAM Spills | 19 | 17 | 15 |
| DRAM→L1 Shards | 8 | 14 | 3 |
| Roundtrip Pairs | 0 | 2 | 0 |
| Primary ops | 8 grid_sample | ~18 conv2d | ~11 conv2d |

All three blocks run well above 30 FPS individually. They require no optimization. The combined time (28.84 ms) is already below the 33.3 ms per-frame target, meaning that if Blocks A, B, and C could be brought to ~4 ms combined (implausible), the overall system would hit 30 FPS. The realistic path is to reduce A+B+C from ~463 ms to ~4.5 ms — a 100× speedup — which is impossible without multi-device parallelism. The realistic 30 FPS target on a single device requires approximately:
- Block A: 326 ms → ~20 ms (16× speedup, requires weight caching + parallel branches)
- Block C: 107 ms → ~8 ms (13× speedup, same fixes)
- Block B: ~30 ms → ~3 ms (10× speedup, requires kernel or fusing improvements)

These targets require architectural changes beyond compiler optimization alone.
