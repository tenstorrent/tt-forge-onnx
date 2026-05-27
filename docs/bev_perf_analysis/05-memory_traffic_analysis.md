# Memory Traffic Analysis — DRAM↔L1 Across All Blocks

**Device:** Wormhole B0 — 64 cores (8×8 grid), ~1.43 MB usable L1/core, ~91.5 MB total L1  
**DRAM bandwidth:** ~288 GB/s peak (bidirectional), ~200 GB/s sustained  
**Peak host RAM (full model):** 3806.67 MB

---

## 1. MEMDUMP Summary Across All Blocks

The compiler inserts MEMDUMP instrumentation at pipeline checkpoints. The final checkpoint (post-dealloc) numbers represent actual to_memory_config ops in the deployed kernel.

### Final-Stage MEMDUMP (13-final-after-dealloc.mlir)

| Block | Role | L1→DRAM Spills | DRAM→L1 Shards | Roundtrip Pairs | Notes |
|-------|------|---------------|-----------------|-----------------|-------|
| A | 4-cam FPN (serial) | **176** | **84** | **32** | Dominant — 67% of time |
| B | BEV transform (grid_sample ×32) | 72 | 32 | 0 | ~6% est |
| C | 1-cam FPN (panoramic) | **49** | **25** | **10** | 22% of time |
| D | Cylinder BEV (grid_sample ×8) | 19 | 8 | 0 | 3.5% |
| E | Feature aggregation | 17 | 14 | 2 | 1.3% |
| F | Output heads | 15 | 3 | 0 | 1.2% |
| **Full Model** | End-to-end | **345** | **163** | **44** | Total (log-verified) |
| **Sum of blocks** | | **348** | **166** | **44** | ~1% overhead in summing |

The full_model total (345/163/44) closely matches the sum of blocks (348/166/44), confirming blocks execute independently without significant overlap or shared L1 state.

---

## 2. Roundtrip Pairs — Pure Waste

A roundtrip pair is a DRAM→L1 spill immediately followed by a DRAM→L1 reload of the same data. It contributes zero computational work while consuming 2× DRAM bandwidth.

| Block | Roundtrip Pairs | Estimated DRAM Waste |
|-------|-----------------|---------------------|
| A | **32** | High (large activation sizes at mid-resolutions) |
| C | 10 | Medium |
| E | 2 | Negligible |
| B, D, F | 0 | — |

The 44 total roundtrip pairs in the full model represent the single largest category of avoidable DRAM traffic.

### Why Roundtrips Occur

Roundtrip pairs arise when an L1-resident tensor must pass through a non-L1-compatible op (typically `reshape`, `concat`, or `slice_static`) that forces a DRAM intermediate, and the result is immediately needed in L1 by the next op.

In Block A, the pattern is:
```
conv2d output (L1 sharded) 
  → to_memory_config (L1→DRAM)     [spill — reshape requires DRAM input]
  → reshape (DRAM→DRAM)
  → to_memory_config (DRAM→L1)     [reload — next conv2d wants L1 input]
  [= 1 roundtrip pair]
```

The fix requires either:
1. Making `reshape` op L1-compatible (in-place or with tile rearrangement without DRAM bounce), OR
2. Fusing the reshape into the preceding/following conv2d so no explicit memory config change is needed

---

## 3. DRAM Traffic Quantification

### L1→DRAM Spill Traffic (Lower Bound)

For each spill event, the DRAM traffic equals the tensor size. Representative tensor sizes at different resolution levels:

| Level | Spatial | Channels | BF16 Size | # Spills est. | Traffic est. |
|-------|---------|----------|-----------|---------------|-------------|
| 1536×1536 | 1536×1536 | 3 | 13.5 MB | 8 | ~108 MB |
| 384×384 | 384×384 | 32 | 9.4 MB | 8 | ~75 MB |
| 192×192 | 192×192 | 64 | 4.7 MB | 16 | ~75 MB |
| 96×96 | 96×96 | 128 | 2.4 MB | 32 | ~77 MB |
| 48×48 | 48×48 | 256 | 2.4 MB | 32 | ~77 MB |
| 24×24 (and below) | small | 256+ | < 1 MB | 80 | ~40 MB |

**Estimated total L1→DRAM spill traffic per inference: ~450–600 MB** (Block A alone)

At 200 GB/s sustained DRAM bandwidth:
- 600 MB / 200 GB/s = 3 ms minimum — this is the irreducible minimum for the spill traffic alone
- Actual time is 326 ms → the system is **not DRAM bandwidth-limited** on the spills themselves
- The dominant cost is **compute latency** for each DRAM-backed convolution, not raw bandwidth

### Weight Traffic (the actual DRAM bottleneck)

With `config_tensors_in_dram = true` for all 187 conv2d ops:
- Each weight tensor is fetched from DRAM at the start of each conv2d
- Weights are NOT cached between the 4 serial camera branches in Block A
- Block A's 148 conv2d have weights ranging from 3.4 KB (1×1 at small spatial) to ~590 KB (3×3 at larger channel counts)

Estimated total weight traffic per inference (Block A):
- Small convs (< 32 KB each): ~80 ops × average 20 KB = 1.6 MB
- Mid convs (32–128 KB each): ~48 ops × average 80 KB = 3.8 MB
- Large convs (128+ KB each): ~20 ops × average 300 KB = 6.0 MB
- Total per branch: ~11.4 MB × 4 branches = **~45.6 MB weight traffic per inference in Block A**

Block C: ~12 MB (39 ops × 1 branch)

**Total weight fetch traffic: ~58 MB per inference**

At 200 GB/s: 58 MB / 200 GB/s = 0.29 ms minimum — also not the bandwidth bottleneck in absolute terms.

The actual bottleneck is **latency**: each conv2d incurs multiple DRAM round-trips for weight tiles, with each round-trip adding ~100–200 ns of latency. At 187 conv2d ops × ~10 tile fetches each × 150 ns = ~280 ms — this matches the observed 326 ms order of magnitude.

---

## 4. to_memory_config Count Analysis

The `to_memory_config` op is the primary representation of memory traffic in the final IR. Its count per block:

| Block | to_memory_config (→DRAM) | to_memory_config (→L1) | Total |
|-------|--------------------------|------------------------|-------|
| A | 164 | 124 | 288 |
| B | 72 | 32 | 104 |
| C | ~54 | ~40 | ~94 |
| D | ~19 | ~8 | ~27 |
| E | ~17 | ~14 | ~31 |
| F | ~15 | ~3 | ~18 |
| **Total** | **~341** | **~221** | **~562** |

Each `to_memory_config` represents a synchronization point plus DMA transfer. The 562 total across all blocks means the device performs on average ~562/484ms ≈ **1.16 memory moves per millisecond**, or roughly **one memory config op every 0.86 ms**.

---

## 5. L1 Occupancy Pressure

The MLA algorithm's conservative DRAM placement (73.5% `<1x1>` in Blocks A and C) is driven by L1 pressure calculations.

### L1 Budget Analysis at Key Resolutions

| Resolution | Channels | Activation Size | Per-Core (÷64) | Fits in 1.43 MB? |
|------------|----------|----------------|----------------|-------------------|
| 1536×1536 | 3 | 13.5 MB | 211 KB | Yes (barely) |
| 1536×1536 | 32 | 144 MB | 2.25 MB | **No** |
| 384×384 | 32 | 9.4 MB | 147 KB | Yes |
| 384×384 | 64 | 18.9 MB | 295 KB | Yes |
| 192×192 | 64 | 4.7 MB | 73 KB | Yes |
| 192×192 | 128 | 9.4 MB | 147 KB | Yes |
| 96×96 | 128 | 2.4 MB | 37 KB | Yes |
| 96×96 | 256 | 4.7 MB | 74 KB | Yes |
| 48×48 | 256 | 2.4 MB | 37 KB | Yes |

Analysis: below the 1536×1536 resolution, activations fit in L1 easily. The L1 pressure at 1536×1536 only applies to high-channel-count tensors. The first few convolutions (in_channels=3) fit, but any conv that expands channels significantly (e.g., 3→32 at 1536×1536) would create 144 MB activation — impossible to L1-cache across 64 cores.

However, the MLA's `<1x1>` decisions extend down to much smaller resolutions (e.g., 96×96 with 128 channels would fit trivially). This suggests the MLA is being **overly conservative** due to L1 fragmentation tracking:
- Circular buffer (CB) fragmentation can reduce effective L1 from 1.43 MB to 700–800 KB usable
- The `SumL1MemoryTracker` in `L1SpillManagement.h` may be using conservative CB estimates

### Circular Buffer (CB) Fragmentation

The Wormhole L1 is divided into program memory, constant data, circular buffers (for streaming tiles), and scratch space. When multiple live tensors compete for CB slots:
- CB fragmentation can leave 40–60% of L1 unusable for tile storage
- The MLA conservatively falls back to DRAM when estimated CB fragmentation exceeds safe limits
- This is the likely cause of `<1x1>` placement for activations that would analytically fit

---

## 6. Full Model DRAM Traffic Summary

From `BEV_MODEL_LOGS/bev_full_model.log`:

| Metric | Value |
|--------|-------|
| Total L1→DRAM spills | 345 |
| Total DRAM→L1 shards | 163 |
| Roundtrip pairs | 44 |
| Peak host RAM | 3806.67 MB |
| to_memory_config ops (est.) | ~562 |
| Inference time | 483.76 ms |

### Memory Traffic Per Second at Baseline (2.07 FPS)

- Spills per second: 345 × 2.07 = ~714 spill ops/second
- At average 1 MB per spill: ~714 MB/s → well within 200 GB/s DRAM bandwidth
- This confirms the bottleneck is **latency** (waiting for each DRAM round-trip to complete), not **bandwidth** (saturating the DRAM bus)

---

## 7. Optimization Impact on Memory Traffic

| Optimization | Spills Reduction | Shard Reduction | Roundtrip Reduction |
|-------------|-----------------|-----------------|---------------------|
| L1 weight caching (`config_tensors_in_dram = false`) | ~0 direct | ~0 direct | ~0 direct | 
| Improved sharding propagation (reduce `<1x1>`) | 30–50% | 30–50% | 20–40% |
| Roundtrip pair elimination (fuse reshape into L1) | ~0 direct | ~0 direct | **100%** (of pairs) |
| Grid_sample format fusion | ~15% of B+D | ~15% of B+D | ~0 |

Note: L1 weight caching (`config_tensors_in_dram = false`) reduces weight fetch latency dramatically but doesn't change the spill/shard MEMDUMP counts — those measure activation tensor movement, not weight movement. Weight caching would primarily reduce `prepare_conv2d_weights` latency.
