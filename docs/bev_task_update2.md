Hi @Nikola Vukobrat
Task updates from my side (continued),
1. BEV model performance investigation
 After splitting the model into multiple blocks, worked on enabling compiler optimization inidiially for each block and fix and improving FPS

**a) CameraCylinder BEV Transform — PermuteOp Silent Data Corruption (opt_level_2)**

The CameraCylinder BEV Transform produced wrong outputs at opt_level_2 with no crash — all test cases failed with near-zero PCC. The MLA pass was assigning BLOCK_SHARDED L1 layouts to the PermuteOps surrounding each grid_sample. Inside permute.cpp, WH-involving permutations decompose into a `transpose_wh + transpose_hc` chain, and the sharded path in the wh kernel requires `shard_spec.shape[1] == logical_shape[-1]` (full-width shard). BLOCK_SHARDED violates this condition, so the non-sharded factory runs on sharded memory and silently produces corrupted output. The OpModel could not catch this because it builds the test input from the previous op's DRAM-interleaved layout, so `a.is_sharded()` is always false during the query. Fixed by filtering all sharded L1 layouts from PermuteOp candidates in `LegalOpLayoutAnalysis.cpp` and adding a `TT_FATAL(!is_sharded)` guard and a `prim_permute` fallback routing in `permute.cpp`. All test cases now pass at opt_level_2.


**b) BEV Aggregator — Bilinear Upsample Segfault During Compilation (opt_level_2)**

The BEV Aggregator segfaulted inside the opt_level_2 MLA pass — not at runtime — in `generate_halo_kernel_config_tensors` with an out-of-bounds array access. The MLA was proposing an arbitrary HEIGHT_SHARDED spec for the three bilinear upsample ops and passing it to `QUERY_OP_CONSTRAINTS`. When the input is already sharded, the kernel's autoresharding step is skipped, and `apply_bilinear_halo_preprocessing` runs with a spec that does not match what the kernel computed internally, causing the out-of-bounds. Fixed in two stages: first, rejected all sharded L1 for bilinear upsample to stop the crash and the block passed at 116.58 FPS on DRAM. Then, mirrored the kernel's own autoshard formula in `TTNNOpModel.cpp` — `shard_height = roundup(N×H×W, num_shards) / num_shards` with ROW_MAJOR page layout — so `QUERY_OP_CONSTRAINTS` receives the exact spec the kernel expects and validates successfully, bringing the block to 120.93 FPS with L1 sharding enabled.



**c) CameraDeformedCylinder Backbone — Conv2d L1 Fragmentation OOM (opt_level_2)**

The CameraDeformedCylinder Backbone failed at opt_level_2 with `Not enough space to allocate 37748736 B — largest free block: 420576 B` even though the simulation believed sufficient space existed. By design, `ToLayoutOp` outputs are not tracked in `liveValues` (they are short-lived L1 tenants). Three such outputs (303,104 B/core each) were allocated and freed in a non-ideal order, fragmenting the L1 heap. The simulation saw `getOccupiedL1() = 0` because these tenants were never tracked, so it approved placing a 589,824 B/core Conv2d output in L1 — at runtime the largest contiguous block was only 420,576 B and the allocation failed. Fixed by adding a fragmentation guard in `ensureFitsL1`: if a tensor exceeds 40% of `l1BudgetPerCore` (530,261 B on WH N150), all live L1 inputs for that op are evicted and the op is unconditionally demoted to DRAM. The backbone now passes cleanly at opt_level_2.



**d) CameraCylinder BEV Transform — GridSample Trace Capture Crash (trace_enabled)**

When running with `enable_trace=True`, the CameraCylinder BEV Transform crashed because the `nearest + align_corners=True` grid precomputation path reads the grid tensor from device to host for CPU-side float32 preprocessing — valid during the warmup run, but forbidden once trace capture is active. Fixed by caching the precomputed grid in the `ProgramContext` on the first invocation and reusing it on all trace-captured invocations, following the existing pattern for trace-incompatible resources in `runtime/lib/ttnn/operations/pool/grid_sample.cpp`. The block now runs correctly in trace mode.


**Full BEV Model — End-to-End Performance**

With all six blocks fixed, the full BEV model was compiled and benchmarked end-to-end. At opt_level_0 (no compiler optimizations) throughput was 0.38 FPS. With opt_level_2, program cache, trace, constant folding, Forge optimization passes, and bfloat16 data format override enabled, throughput reached 2.05 FPS — a 5.4× improvement over the unoptimized baseline.


**Current work — GridSample OpModel: improving L1 utilization at opt_level_2**

`grid_sample` is the core op in BEV transforms — it samples a feature map at locations specified by a grid to project camera features into bird's-eye-view coordinates. The BEV model contains 40 such ops. At opt_level_2 the compiler runs a Memory Layout Analyzer (MLA) that uses an OpModel — a cost estimator — to decide whether each op's output should live in fast on-chip L1 SRAM or in slower DRAM. The `GridSampleOp` OpModel is not yet fully implemented: it returns `OpNotSupportedError` unconditionally, meaning the compiler has no cost estimate for it and conservatively places all 40 GridSample outputs in DRAM.

The performance problem is a cascade triggered by this fallback. Because `grid_sample` writes its output in ROW_MAJOR format but all downstream ops expect TILE format, the `TTNNWorkaroundsPass` inserts a format-conversion op (`to_layout`) after every GridSample. MLA then tries to assign each of these 40 conversion ops to HEIGHT_SHARDED L1 — the fast path — but L1SpillManagement immediately evicts all of them back to DRAM because the actual L1 budget is already under pressure from other live tensors. The result is 40 round-trips of allocate-L1 → evict-to-DRAM per compilation, with no net gain in runtime performance.

To diagnose the full scope, an IR dump pass was added at six checkpoints across the compiler pipeline. The dump shows that 65 DRAM→L1→DRAM pairs are eliminated early by a new fold pass, and 3 more after layout decomposition, but 216 L1→DRAM spills remain. The remaining spills do not come from the GridSample `to_layout` ops — they originate from compute ops (conv2d, concat, etc.) whose outputs MLA assigned HEIGHT_SHARDED L1 that L1SpillManagement immediately evicted. This means the root problem is broader: MLA is assigning HEIGHT_SHARDED L1 to tensors across the model that will be spilled before any consumer reads them, allocating L1 space that is never productively used. The fix needs to be at the MLA level — either teaching it to predict which assignments will survive L1SpillManagement, or feeding back spill decisions so MLA avoids repeating assignments it knows will be evicted.
