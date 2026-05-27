# BEV Model Performance Analysis — Executive Summary

**Date:** 2026-05-19  
**Device:** Single Tenstorrent Wormhole B0 (8×8 compute grid, 64 cores, ~1.43 MB usable L1/core)  
**Pipeline:** tt-forge-onnx + tt-mlir (opt_level=2, HiFi3, fp32_dest_acc, Float16_b/BF16, trace enabled)  
**Target:** 30 FPS (33.3 ms/frame)  
**Baseline:** 2.07 FPS (483.76 ms/frame) — peak host RAM: 3806.67 MB  

---

## Baseline Breakdown by Block

| Block | Role | Time (ms) | FPS | % of Total | DRAM Spills | DRAM Shards | Roundtrips |
|-------|------|-----------|-----|------------|-------------|-------------|------------|
| A     | Camera feature extraction (4×1536×1536) | 326.45 | 3.05 | **67.5%** | 176 | 84 | 32 |
| C     | Camera feature extraction (1×1280×2304) | 106.91 | 9.31 | **22.1%** | 49 | 25 | 10 |
| B     | BEV view transform (grid_sample × 32) | ~30 est | — | ~6.2% | 72 | 32 | 0 |
| D     | Cylinder BEV transform (grid_sample × 8) | 16.97 | 58.37 | 3.5% | 19 | 8 | 0 |
| E     | Feature aggregation | 6.26 | 156.05 | 1.3% | 17 | 14 | 2 |
| F     | Output heads | 5.61 | 173.28 | 1.2% | 15 | 3 | 0 |
| **Total** | | **483.76** | **2.07** | 100% | 348 | 166 | 44 |

Note: Block B run exits with a PCC failure (PCC=0.9880, threshold=0.99) so its isolated timing is unavailable. The "~30 est" above is derived from full_model − (A+C+D+E+F).

**Required speedup to reach 30 FPS:** 14.6× (from 484 ms to 33.3 ms)

---

## Top Bottlenecks (ranked by impact)

### 1. Block A — DRAM-Bound Convolutions on 4× Serial Camera Branches (67% of time)

The model has 4 deformed-cylinder camera inputs (each 1×3×1536×1536 BF16). These are processed serially on a single device. Block A contains:
- **148 conv2d** ops + relu6 fused into 128 of them
- **16 conv_transpose2d** ops
- **8 "dram_width" stride-2 downsamplers** at 1536×1536 resolution (DRAM bandwidth bound — tensor does not fit in L1)
- **ALL conv2d weights fetched from DRAM on every forward pass** (`config_tensors_in_dram = true` on all 488 conv2d/prepare_conv2d_weights ops)
- **73.5% of layout annotations use `<1x1>` grid** (single-core interleaved DRAM, no distribution)
- **32 DRAM→L1→DRAM roundtrip pairs** (data written to DRAM, immediately read back to L1)

The combined effect: nearly every convolution cycle is throttled by DRAM latency rather than compute throughput.

### 2. Block C — Same Root Cause, Smaller Scale (22% of time)

Single cylindrical camera (1×3×1280×2304) with 39 conv2d ops. The identical structural problems as Block A: `config_tensors_in_dram = true`, 73.4% `<1x1>` grid, 10 roundtrip pairs, 74 total spill/shard events.

### 3. Weight Refetch Overhead (Blocks A + C together)

With `config_tensors_in_dram = true`, every one of the 187 conv2d ops (148+39) fetches its weight tensor fresh from DRAM each inference. For the 4 camera branches in Block A, if any branches share weights (shared feature extractors), those weights are fetched redundantly — up to 4× per shared weight per inference.

### 4. L1 Pressure from Large Spatial Tensors

The 8 largest conv2d ops at 1536×1536 use `dram_width` slice config rather than `l1_full`, confirming that L1 per core is insufficient to hold the full activation. This forces a streaming pattern through DRAM, reducing effective compute utilization.

### 5. Block B — Grid Sample Format Conversion Overhead

32 nearest-neighbor grid_sample ops, each surrounded by:
- 2 `to_layout` ops (tile↔row_major conversion — mandatory, kernel outputs row_major)
- 2 `to_memory_config` ops (L1↔DRAM movement)
- 4 additional ops for grid LUT preparation (slice → reshape → to_memory_config → to_layout)

This is ~8 ops of overhead per grid_sample. The PCC failure (0.9880 vs 0.99 threshold) is a known issue with nearest-neighbor interpolation in BF16 accumulation and is not a compiler defect.

---

## Optimization Opportunity Summary

| Priority | Optimization | Blocks | Estimated Speedup | Complexity |
|----------|-------------|--------|-------------------|------------|
| P1 | Enable L1 weight caching (`config_tensors_in_dram = false`) | A, C | 2–3× | Medium |
| P2 | Reduce `<1x1>` (DRAM-interleaved) tensor count via improved sharding propagation | A, C | 1.3–1.8× | High |
| P3 | Eliminate DRAM→L1→DRAM roundtrips (32 pairs in A, 10 in C) | A, C | 1.1–1.2× | Medium |
| P4 | Fuse to_memory_config + to_layout sequences around grid_sample | B, D | 1.05–1.1× | Low-Medium |
| P5 | Parallel camera branch processing (multi-stream) | A | Up to 4× theoretical | Very High |
| P6 | dram_width→l1_full for 1536×1536 ops if tensor tiling is improved | A | 1.1–1.3× | High |

**Combined achievable speedup (P1+P2+P3+P4):** 3–6× → projected 80–160 ms → 6–12 FPS  
**With P5 (parallel branches):** additional 2–4× → projected 20–80 ms → 12–50 FPS range  

Reaching 30 FPS on a single device requires addressing all categories. The biggest single lever is P1 (weight caching) combined with P2 (sharding propagation), which together address the DRAM bottleneck that dominates Block A and C.

---

## Architecture Context

```
Input: 4× cameras (1×3×1536×1536) + 1× cylindrical camera (1×3×1280×2304)
          ↓ Block A (4 serial branches × FPN)    ↓ Block C (1 branch × FPN)
    Camera features at [96×96 per cam]       Camera features [80×144]
          ↓ Block B (32 grid_sample)              ↓ Block D (8 grid_sample)
    BEV feature plane [128×64]               Cylinder BEV features [80×144]
          ↓                                        ↓
          └──────────── Block E (aggregation) ─────┘
                              ↓ Block F (output heads)
                    Detection outputs (boxes, classes, velocity)
```

All blocks execute sequentially on the single device. There is no current pipelining or parallel execution between blocks.

---

## Key Constraints (Must Not Be Changed)

- trace=enabled, opt_level=2
- HiFi3 math fidelity
- fp32_dest_acc
- Float16_b (BF16) data type
- consteval enabled
- All optimization passes enabled (MLA, fusing, constant folding, etc.)
- Block B PCC failure is an expected numerical property, not a target to fix
- Single device only (no multi-chip distribution)
- TTNN GridSample always outputs ROW_MAJOR — this is a kernel constraint

---

## Files Analyzed

| File | Lines | Purpose |
|------|-------|---------|
| `BEV_MODEL_LOGS/block_A.log` | 9773 | Block A execution: GREEDY decisions, MEMDUMP, benchmark |
| `BEV_MODEL_LOGS/block_B.log` | 2476 | Block B execution: PCC failure, MEMDUMP |
| `BEV_MODEL_LOGS/block_C.log` | ~3000 | Block C execution: MEMDUMP, benchmark |
| `BEV_MODEL_LOGS/block_D.log` | ~1500 | Block D: 8 grid_sample BEV |
| `BEV_MODEL_LOGS/block_E.log` | ~1200 | Block E: feature aggregation |
| `BEV_MODEL_LOGS/block_F.log` | ~1000 | Block F: output heads |
| `BEV_MODEL_LOGS/bev_full_model.log` | 16530 | Full model end-to-end |
| `BEV_MODEL_IRS/BLOCK_{A-F}/01-ttir-passes.mlir` | — | TTIR op counts, input shapes |
| `BEV_MODEL_IRS/BLOCK_{A-F}/09-mla.mlir` | — | Post-MLA layout decisions |
| `BEV_MODEL_IRS/BLOCK_{A-F}/13-final-after-dealloc.mlir` | — | Final IR: to_memory_config counts |
| `BEV_MODEL_IRS/FULL_MODEL/09-mla.mlir` | — | Full model MLA combined |
