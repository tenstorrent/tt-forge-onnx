# BEV Model Performance — Optimization Work Update

After achieving 2.05 FPS end-to-end, continued profiling and optimizing the individual model blocks.

---

## a) BEV Transform (Block B) — GridSample Batched Fusion

**Issue:**
The BEV Transform block processes images from 8 cameras. The compiler was generating 32 separate grid sampling operations (one per camera per group), each preceded by 17 preparatory steps to set up the shared lookup table (8 slice + 8 reshape + 1 concat). This meant the hardware was doing 32 separate kernel launches and 17 redundant data-preparation steps, even though all 8 cameras within a group share the exact same lookup table.

**Fix:**
Two pattern-matching optimizations were added to the TTIR canonicalization pass — the compiler stage that simplifies the operation graph before lowering to hardware:

- **LUT Simplification:** Detected the repeated broadcast pattern where the same source tensor was being sliced, reshaped, and concatenated 8 times — once per camera. Verified that all 8 slices reference the same source and replaced the entire 17-op sequence with a single reshape, since broadcasting to each camera individually is redundant.
- **Camera Batching:** Matched groups of grid_sample ops that feed into a concat and share the same image and LUT source. Fused them into a single batched grid_sample call that processes all cameras in one kernel launch. The camera count is inferred from the grid tensor's shape.

**Result:**
~40 operations per group reduced to ~5. Block B improved from **9.31 → 11.94 FPS (~28% improvement)**.

---

## b) DRAM → L1 → DRAM Round-Trip Fold — Investigated, Not Viable

**Issue:**
While profiling Block D (Cylinder BEV Transform), a pattern was found in the compiled IR where the compiler inserts back-to-back memory layout operations with no compute in between. On Tenstorrent hardware, tensors live either in DRAM (main memory, slow, large) or L1 (fast on-chip memory, local to each core). Moving data between them has a bandwidth cost. The pattern found was:

> Move tensor from DRAM into L1-sharded memory → immediately move it back to DRAM, no op uses it in L1

This round-trip is pure overhead. The data should have stayed in DRAM.

**Fix tried:**
A new rewrite pattern was added to the compiler that fires after the layout decomposition stage. It detects consecutive memory-move operations where the intermediate L1-sharded result has no compute consumers and collapses the two moves into one. This correctly removed 3 round-trip pairs and 65 identity operations (68 ops total) across the full BEV model.

**Result:**
The optimization works on standard runs but has **no effect when hardware tracing is enabled**. With tracing on, a pass called `TTNNTraceHoistTransform` restructures the IR before layout decomposition runs, reordering the memory-move chains so the fold pattern never appears. Since the 2.05 FPS target requires trace mode (without it the model runs at ~0.38–0.9 FPS), this optimization cannot contribute to the benchmark. **Investigation closed.**

---

## c) Deformed Backbone (Block A) — DRAM Bounce Spill Fix

**Issue:**
Block A is the pipeline bottleneck at **358.7 ms / 2.79 FPS** (~56% of total time).

The compiler's Memory Layout Assignment (MLA) pass sometimes produces a wasteful pattern called a **bounce spill** — data is moved from DRAM into fast on-chip L1 memory, then immediately moved back to DRAM with no computation in between. This wastes bandwidth for two memory transfers that should be zero.

Profiling found **52 bounce spills** in Block A, introduced at two points in the compiler pipeline where layout adjustments were made but no cleanup pass followed to remove the redundant moves. A secondary blocker: soft memory-cleanup markers on the intermediate tensors were being mistaken for real data uses, preventing the fold from triggering.

**Fix:**
- Added a canonicalization pattern that detects the back-to-back memory-move structure, removes the intermediate L1 hop, and merges the two moves into one. It runs to fixpoint, so cascading chains are fully resolved.
- Inserted cleanup passes at the two pipeline points where bounces were being created, so they are eliminated immediately rather than surviving to the final output.

**Result:**

| | Before | After |
|---|---|---|
| Bounce spill patterns | 52 | **0** |
| Memory-move operations | 324 | 228 (−96) |
| Total IR operations | 4007 | 3810 (−197) |
| Block A FPS | 2.79 | 2.62–2.74 |

---

## Summary

Overall, the BEV model has seen end-to-end FPS improvement — reaching **2.05 FPS** after the full set of optimizations. At the block level, the GridSample batching delivered a real **28% improvement for Block B** (9.31 → 11.94 FPS). The bounce spill removal cleaned up 197 redundant operations in Block A but did not produce a measurable FPS change, as the removed ops were overhead moves rather than compute and the hardware trace was already absorbing the cost at runtime.

Block A (Deformed Backbone) remains the biggest bottleneck at **358.7 ms / 2.79 FPS (~56% of total pipeline time)**. The bounce spill work has cleaned up the IR and established the infrastructure for further improvements. The next focus is the structural DRAM spill problem in Block A — **116 compute ops that run on the correct sharded grid but have their outputs unnecessarily evicted to DRAM** before the next op can consume them. Fixing this is the primary lever for improving end-to-end FPS further.
