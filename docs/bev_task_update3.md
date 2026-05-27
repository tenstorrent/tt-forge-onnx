# Task updates from my side:

## 1. BEV Model Performance — Continued Optimization Work

After achieving 2.05 FPS end-to-end, I continued profiling and optimizing the individual model blocks.

---

### a) BEV Transform (Block B) — GridSample Batched Fusion

The BEV Transform block contained 32 grid_sample calls (4 groups × 8 cameras). Each group also required 17 LUT preparation ops (8 slice + 8 reshape + 1 concat) to broadcast a shared 5D LUT tensor to each individual camera before the grid_sample calls.

The TTNN grid_sample kernel supports batched execution through `batch_output_channels=True`, which processes all cameras in a single kernel launch. However, the compiler was not utilizing this because the frontend lowered one op per camera.

Fixed this in two stages within the TTIR canonicalization pass (`TTIRFusing.cpp`):

- **LUT Simplification (`GridSampleLutSimplify`):** Since all 8 cameras within a group share the same LUT, the 17-op broadcast sequence was replaced with a single reshape. The pattern matches the `concat(reshape(slice(x)), ..., reshape(slice(x)))` structure and verifies that all slices reference the same source.
- **Camera Batching (`GridSampleBatchedFuse`):** Matched groups of grid_sample ops feeding into a concat where all ops share the same image and LUT source, and fused them into a single grid_sample with `batch_output_channels=True`. The TTNN lowering detects the camera count K from grid shape dim 1 (= 2K) and sets the flag accordingly.

Net IR reduction: ~40 ops per group → ~5 ops per group.
Block B FPS improved from ~9.31 → **11.94 FPS (~28% improvement)**.

---

### b) DRAM → L1 → DRAM Round-Trip Fold — Investigated, Not Viable with Trace

While profiling the Cylinder BEV Transform (Block D), identified a pattern where the compiler generates back-to-back `to_memory_config` ops with no useful work in between:

```
to_memory_config(to_memory_config(x, L1_sharded), DRAM)
```

The intermediate L1 hop allocates and copies data, then immediately copies it back out to DRAM without any operation consuming it in L1.

Added `foldDRAMtoL1toDRAMRoundTrip` as a rewrite pattern firing after `TTNNDecomposeLayouts` to collapse these into a single move. On non-trace runs, this worked correctly and eliminated 3 round-trip pairs + 65 identity ops (68 total ops removed in the full BEV model).

However, the optimization has no effect with `trace=True`. When trace is enabled, `TTNNTraceHoistTransform` restructures the IR before decomposition runs, reordering the `ToMemoryConfigOp` chains such that the fold pattern never matches. Since the 2.05 FPS end-to-end result requires `trace=True` (without trace the model falls back to ~0.38–0.9 FPS), this optimization cannot contribute to the target benchmark. **Investigation closed.**

---

### c) Deformed Backbone (Block A) — DRAM Bounce Spill Fix ✅

**Background:**
The Deformed Backbone was the pipeline bottleneck at **358.7 ms / 2.79 FPS** (~56.4% of total time).

When the compiler assigns L1-sharded layouts, it sometimes produces a pattern like this:

```
op output (DRAM) → to_memory_config(L1_sharded) → to_memory_config(DRAM) → next op
```

This is called a **bounce spill** — the data makes a pointless round-trip through L1 without any computation happening there. It wastes memory bandwidth and L1 allocation for zero benefit.

**What we found:**
Profiling the TTNN IR revealed **52 bounce spill patterns** (not 8 as originally estimated — the initial count was from a limited detection script). The bounces came from two places in the compiler pipeline:

- **32 bounces** were introduced by `OperationValidationAndFallback` inside `DevicePassesWrapper`. A `canonicalize` pass ran before this pass but not after, so the bounce chains it created survived.
- **8 bounces** were introduced by `TTNNDecomposeLayouts` which runs after `TTNNTraceHoistTransform` moves ops into the hardware trace function. Again, no canonicalize ran after this step.
- A secondary blocker: soft `deallocate` ops on the intermediate result were counting as extra users, preventing the fold from triggering.

**What we implemented (changes in `third_party/tt-mlir/`):**

1. **`lib/Dialect/TTNN/IR/TTNNOps.cpp`** + **`include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`**
   Added a new canonicalization pattern `FoldConsecutiveToMemoryConfigOps`. It detects when a `to_memory_config` output is L1-sharded and its only consumer is another `to_memory_config` (plus optional soft deallocates), then collapses the two ops into one, skipping the L1 hop entirely. Soft deallocates on the intermediate are removed as part of the fold.

2. **`lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`**
   Added two `canonicalize` pass insertions — one after `OperationValidationAndFallback` inside `DevicePassesWrapper` (catches the 32 bounces in `@forward`), and one after `TTNNDecomposeLayouts` in the top-level pipeline (catches the 8 bounces in `@trace_0_forward`).

**Results:**

| Metric | Before | After |
|---|---|---|
| Bounce spill patterns | 52 | **0** |
| `to_memory_config` ops | 324 | 228 (−96) |
| `deallocate` ops | 1308 | 1212 (−96) |
| IR lines | 4007 | 3810 (−197) |
| FPS | 2.79 | 2.62–2.74 |

FPS change is within run-to-run measurement noise (±50 ms variance). The 8 bounces in the trace region were already being absorbed by the hardware trace execution at runtime, so their removal does not show up as latency improvement.

---

### d) Deformed Backbone (Block A) — Structural DRAM Spills (Next Steps)

With bounce spills fully eliminated, the remaining bottleneck is **structural DRAM spills** — cases where compute ops run on a sharded grid but their outputs are forced to DRAM because the next op cannot accept L1-sharded input. These are a fundamentally different class of problem from bounce spills.

Currently, only **23.5% of shardable ops are "effectively sharded"** (compute in L1 and output stays in L1). The other 116 sharded ops compute correctly on a multi-core grid, but their output is evicted to DRAM before the next op can use it.

```
effectively_sharded:    96 ops  (23.5%)  ← compute + output both in L1  ✅
sharded but spilled:   116 ops  (28.4%)  ← compute in L1, output evicted to DRAM ⚠️
not sharded:           196 ops  (48.0%)  ← single-core or DRAM-only
```

If the 116 spilled ops could stay in L1, effectively_sharded would go from **23.5% → 52%**.

The 116 break down by root cause:

| Root cause | Count | What needs to change |
|---|---|---|
| Conv2d output used by two consumers needing different layouts (conv2d + slice_static); MLA picks DRAM as common ground | 28 | MLA to route each consumer its own layout conversion instead of sharing a DRAM copy |
| Stride=2 conv2d kernel requires DRAM-interleaved activation input | 16 | tt-metal kernel support for L1-sharded activation in stride=2 conv2d |
| Concat consumer has incompatible shard spec; forces DRAM intermediate | 32 | MLA concat sharding for mixed-spec inputs |
| max_pool2d output spills immediately; downstream conv2d reads DRAM | 20 | MLA / tt-metal max_pool2d output chaining |
| conv_transpose2d is single-core only (1×1 grid) | 12 | Sharded conv_transpose2d in tt-metal |
| Skip-connection add/concat with shard spec mismatches | 20 | MLA |

The highest-priority fix is the **28 multi-consumer cases**. The pattern is:

```
Current — both consumers share a DRAM copy:
  %out = conv2d(...) → L1_sharded
  %dram = to_memory_config(%out, DRAM)
      ├── conv2d(%dram, ...)                     ← reads DRAM unnecessarily
      └── to_memory_config(%dram, L1_interleaved)
              └── slice_static(...)

Target — primary conv2d stays in L1, only slice gets a layout copy:
  %out = conv2d(...) → L1_sharded
      ├── conv2d(%out, ...)                      ← reads L1 directly ✅
      └── to_memory_config(%out, L1_interleaved)
              └── slice_static(...)
```

Fix location: `TTNNGreedyMemoryLayoutPropagation.cpp` — when a value has multiple consumers with conflicting layout requirements, keep the dominant consumer's preferred layout and insert per-consumer `to_memory_config` ops only for the others.
