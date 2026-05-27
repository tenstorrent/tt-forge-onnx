# Hi @Andreas @Nikola Vukobrat
# Current status:


## BEV Model Performance — Optimization Work Update
After achieving 2.05 FPS end-to-end, profiling and optimization continued on the individual model blocks.

### a) BEV Transform (Block B) — GridSample Batched Fusion
**Issue:**
The BEV Transform block processes images from 8 cameras. The compiler was generating 32 separate grid sampling operations (one per camera per group), each preceded by 17 preparatory steps to set up the shared lookup table (8 slice + 8 reshape + 1 concat). This meant the hardware was executing 32 separate kernel launches and 17 redundant data-preparation steps, even though all 8 cameras within a group shared the exact same lookup table.

**Fix:**
Two pattern-matching optimizations were added to the TTIR canonicalization pass — the compiler stage that simplifies the operation graph before lowering to hardware:

- **LUT Simplification:** Detected the repeated broadcast pattern where the same source tensor was sliced, reshaped, and concatenated 8 times — once per camera. Verified that all 8 slices referenced the same source and replaced the entire 17-op sequence with a single reshape, since broadcasting to each camera individually was redundant.
- **Camera Batching:** Matched groups of grid_sample ops that fed into a concat and shared the same image and LUT source. Fused them into a single batched grid_sample call that processes all cameras in one kernel launch. The camera count is inferred from the grid tensor's shape.

**Result:**
~40 operations per group were reduced to ~5. Block B improved from **9.31 → 11.94 FPS (~28% improvement)**.

---

### b) DRAM → L1 → DRAM Round-Trip Fold — Investigated, Not Viable
**Issue:**
While profiling Block D (Cylinder BEV Transform), a pattern was found in the compiled IR where the compiler inserted back-to-back memory layout operations with no compute in between. On Tenstorrent hardware, tensors live either in DRAM (main memory — slow but large) or L1 (fast on-chip memory local to each core). Moving data between them has a bandwidth cost. The observed pattern was:

> Move tensor from DRAM into L1-sharded memory → immediately move it back to DRAM, with no op using it in L1

This round-trip was pure overhead. The data should have remained in DRAM.

**Fix tried:**
A new rewrite pattern was added to the compiler that runs after the layout decomposition stage. It detects consecutive memory-move operations where the intermediate L1-sharded result has no compute consumers and collapses the two moves into one. This correctly removed 3 round-trip pairs and 65 identity operations (68 ops total) across the full BEV model.

**Result:**
The optimization works on standard runs but has no effect when hardware tracing is enabled. With tracing enabled, a pass called TTNNTraceHoistTransform restructures the IR before layout decomposition runs, reordering the memory-move chains so the fold pattern never appears. Since the 2.05 FPS target requires trace mode (without it, the model runs at ~0.38–0.9 FPS), this optimization cannot contribute to the benchmark. **Investigation closed.**

---

### c) Deformed Backbone (Block A) — DRAM Bounce Spill Fix
**Issue:**
Block A is the pipeline bottleneck at **358.7 ms / 2.79 FPS (~56% of total execution time)**.

The compiler's Memory Layout Assignment (MLA) pass sometimes produces a wasteful pattern called a bounce spill — data is moved from DRAM into fast on-chip L1 memory and then immediately moved back to DRAM with no computation in between. This wastes bandwidth on two unnecessary memory transfers.

Profiling identified **52 bounce spills** in Block A, introduced at two points in the compiler pipeline where layout adjustments were made but no cleanup pass followed to remove the redundant moves. A secondary blocker was that soft memory-cleanup markers on the intermediate tensors were being mistaken for real data uses, preventing the fold from triggering.

**Fix:**
- Added a canonicalization pattern that detects the back-to-back memory-move structure, removes the intermediate L1 hop, and merges the two moves into one. It runs to fixpoint, so cascading chains are fully resolved.
- Inserted cleanup passes at the two pipeline points where bounce spills were being created, ensuring they are eliminated immediately rather than surviving to the final output.

**Result:**

| | Before | After |
|---|---|---|
| Bounce spill patterns | 52 | **0** |
| Memory-move operations | 324 | 228 (−96) |
| Total IR operations | 4007 | 3810 (−197) |
| Block A FPS | 2.79 | 2.62–2.74 |

**Note on FPS Impact:**
The GridSample batching improved Block B by 28% block-level but did not shift end-to-end FPS since Block A is the bottleneck. The bounce spill removal showed no measurable FPS change (within ±50 ms run noise) — the 8 bounces inside the hardware trace were already absorbed by the trace executor at runtime, and the remaining 44 removed overhead memory moves rather than compute ops. Both changes reduce compiler-generated overhead and make the IR cleaner for further analysis.

**Overall, the BEV model did see end-to-end FPS improvement (reaching 2.05 FPS), with the GridSample fusion delivering a real block-level gain for Block B. Block A (Deformed Backbone) remains the largest bottleneck at ~56% of total pipeline time, and investigation is currently ongoing to address the 116 compute ops that correctly run on a sharded grid but unnecessarily evict their outputs to DRAM — this is the primary lever for further end-to-end FPS improvement.**

---

## EVO50 Model Performance — Block-Level Profiling & FPN Optimization

Split the full EVO50 model into three independently compiled blocks (Backbone, FPN, Heads) and benchmarked each block separately to identify the primary bottlenecks.

**Block-Level Performance Results (Baseline)**

| Block    | Latency  | L1 Sharding |
|----------|----------|-------------|
| Backbone | ~45 ms   | ~82%        |
| FPN      | ~33.6 ms | ~14.9%      |
| Heads    | ~7 ms    | ~38%        |

The FPN block was identified as the primary bottleneck. Its L1 sharding rate (~14.9%) was significantly lower than the backbone (~82%). Root-cause analysis showed that all 5 bilinear ttnn.upsample ops were executing from DRAM.

---

### a) FPN — Bilinear Upsample: DRAM → L1 HEIGHT_SHARDED
**Issue:**
All 5 bilinear upsample ops were executing from DRAM despite the TTNN kernel natively supporting HEIGHT_SHARDED L1 layouts. Three blockers prevented the compiler from assigning sharded layouts:

- The op-model's QUERY_OP_CONSTRAINTS path internally called ttnn::halo(), which segfaulted in the dry-run device context, causing every HEIGHT_SHARDED candidate to be rejected.
- The layout deduplicator removed the exact grid selected by the runtime's compute_bilinear_autoshard_memory_config (num_cores = min(max_cores, N×H×W)), causing a shard-spec mismatch at execution time.
- Passing a TILED input to to_memory_config(..., HEIGHT_SHARDED_ROW_MAJOR_spec) produced a tile-misaligned shard height, triggering a TT_FATAL crash inside TensorLayout.

**Fix:**
- Replaced the crashing op-model path with an analytical CB-size formula that mirrors the runtime's auto-shard logic, bypassing the halo() call entirely.
- Added core-count validation that rejects any grid not matching the runtime autoshard count and generates the exact grid using buildWithCanonicalCorePlacement.
- Added a UpsampleRuleBook with a HEIGHT_SHARDED fallback hint and added UpsampleOp to the sharding eligibility list.
- Fixed the tt-metal runtime crash by inserting to_layout(ROW_MAJOR) before to_memory_config when the input was not already sharded.

**Result:**
All 5 bilinear upsample ops now execute from L1 HEIGHT_SHARDED:

| Op          | Grid | L1 HEIGHT_SHARDED | Spilled to DRAM |
|-------------|------|-------------------|-----------------|
| Resize2d_0  | 39×1 | Yes               | Yes             |
| Resize2d_12 | 64×1 | Yes               | Yes             |
| Resize2d_24 | 64×1 | Yes               | No              |
| Resize2d_36 | 64×1 | Yes               | No              |
| Resize2d_48 | 64×1 | Yes               | No              |

Resize2d_0 and Resize2d_12 spilled to DRAM — addressed in section (b) below.

| Metric      | Baseline | After Upsample Fix | Δ               |
|-------------|----------|--------------------|-----------------|
| Latency     | ~33.6 ms | ~30.2 ms           | −3.4 ms         |
| FPS         | ~29.8    | ~33.1              | +3.3 FPS (+11%) |
| L1 Sharding | ~14.9%   | ~23.7%             | +8.8 pp         |

---

### b) FPN — Eliminating DRAM Spill for Resize2d_0 and Resize2d_12 — Currently Investigating

Resize2d_0 and Resize2d_12 operate on 80-channel feature maps that require a pad → upsample → slice_static pattern (channels padded from 80 → 96 before upsample, then sliced back to 80 afterward).

Even after the upsample fix, a to_memory_config(DRAM) is still inserted between the HEIGHT_SHARDED upsample output and the slice_static op, causing the L1 result to be evicted to DRAM before any compute op can consume it.

Root-cause analysis showed that the compiler's hint-selection logic for slice_static never proposes a NULL output hint — which is required to trigger slice.cpp's auto-recompute path that inherits the HEIGHT_SHARDED layout from the upsample. As a result, the optimizer always falls back to a DRAM input layout for the slice.

**Currently working on:**
- Adding HEIGHT_SHARDED support for slice_static in the compiler (SliceRuleBook, DFShardingPolicy)
- Implementing a new tt-metal program factory that trims each core's local shard in-place without cross-core reads
- This is expected to fully eliminate the remaining DRAM spills for Resize2d_0 and Resize2d_12
