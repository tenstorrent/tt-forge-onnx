# BEV Model — Progress Update

---

**1. Model Decomposition**

Split the BEV model into six logical blocks to enable independent compilation and debugging:

- CameraDeformedCylinder Backbone
- CameraDeformedCylinder BEV Transform
- CameraCylinder Backbone
- CameraCylinder BEV Transform
- BEV Aggregator
- Output Heads

Each block was tested independently with Forge optimization passes and program cache enabled. At `opt_level_1`, all blocks compiled and executed successfully. At `opt_level_2`, the two BEV Transform blocks were hanging during compilation. Since both blocks share the same architecture, the smaller one (CameraCylinder BEV Transform) was selected for deeper analysis.

---

**2. Root Cause — grid_sample Op Explosion**

The analysis showed that the model produced ~2,931 TTIR ops despite the ONNX graph containing only ~20 ops. By profiling individual ops, `grid_sample` was identified as the main bottleneck — a single `grid_sample` op was expanding into ~362 lower-level primitives. Since the full BEV model contains ~40 `grid_sample` ops, this accounted for ~14,480 out of the total ~15,769 TTIR ops. The root cause was that `grid_sample` had no native support in the tt-mlir compiler stack and was instead being decomposed into primitives through a TVM fallback path. This op explosion caused the memory layout analysis pass to time out at `opt_level_2`.

---

**3. Native grid_sample Support Added**

Since `ttnn::grid_sample` already exists in tt-metal, added native end-to-end `grid_sample` support through the entire tt-mlir compiler stack — from ONNX ingestion in tt-forge-onnx down through dialect definitions, compiler lowering passes, flatbuffer serialization, and runtime execution on device.

The tt-metal kernel requires a different tensor layout than what ONNX provides. The input tensor needs to be in NHWC format (ONNX gives NCHW) and the grid tensor needs shape `(N, H_out, W_out, 2)` (ONNX gives `(N, 2, H_out, W_out)`). Layout permutation ops are inserted automatically during lowering, transparent to the user. For `nearest` interpolation mode and `align_corners=True`, the sampling coordinates are precomputed on the host CPU in float32 precision — bfloat16 is not precise enough for large grid sizes — and transferred to device before each kernel invocation.

All `grid_sample` configurations were validated:
- Interpolation modes: bilinear, nearest
- Padding modes: zeros, border, reflection
- `align_corners`: true / false
- Validated at `opt_level_0`, `opt_level_1`, and `opt_level_2`

After adding native support, the TTIR op count dropped from ~362 primitives per `grid_sample` to 1, fully resolving the `opt_level_2` compilation hang.

---

**4. ConvTranspose2d Compile Crash — CameraCylinder Backbone (opt_level_1/2)**

After resolving `grid_sample`, continued testing all six blocks at `opt_level_2`. The CameraCylinder Backbone was crashing during compilation at both `opt_level_1` and `opt_level_2` due to a missing weight data-type propagation step in the tt-metal kernel preparation code for its ConvTranspose2d layers. A weight dtype was being computed but never written back to the op configuration, causing a later compilation stage to fail with a fatal error when it expected the dtype to already be set. Fixed by ensuring the computed dtype is correctly stored at all code paths. After the fix, all ConvTranspose2d ops compiled and executed successfully at both `opt_level_1` and `opt_level_2`.

---

**5. Silent Data Corruption in Permute Ops — CameraCylinder BEV Transform (opt_level_2)**

The CameraCylinder BEV Transform produced numerically incorrect outputs at `opt_level_2` with no crash — the model ran to completion but all outputs were wrong (near-zero PCC across all test cases). The memory layout analysis pass was assigning a block-sharded L1 memory layout to the permutation ops surrounding `grid_sample` (the NCHW↔NHWC reshapes). The underlying kernel only supports the sharded path when each shard covers the full tensor width, which block-sharding does not guarantee. When this condition was violated, the kernel silently produced corrupted output. Fixed by rejecting unsuitable sharded layouts for these ops at compile time. Result: 0/40 → 40/40 pass at `opt_level_2`.

---

**6. Conv2d L1 Circular Buffer Clash — CameraCylinder Backbone (opt_level_2)**

After fixing the ConvTranspose2d crash, the CameraCylinder Backbone still failed at `opt_level_2` with a runtime error where memory allocated for a convolution's circular buffers overlapped with another tensor in L1. The compiler's L1 budget estimator was unaware of a "dead zone" in the L1 address space that sits below its simulation floor, causing it to approve allocations that exceeded safe bounds at runtime. Fixed by adding a pre-check that forces large operations out of L1 when their buffer requirement exceeds the dead zone size.

---

**7. Bilinear Upsample Segfault During Compilation — BEV Aggregator (opt_level_2)**

The BEV Aggregator was segfaulting during the `opt_level_2` memory layout analysis pass — not at runtime. The pass was proposing a sharded memory layout for bilinear upsample ops and passing it to the op cost model to evaluate. The cost model invoked the tt-metal kernel in query mode with an incorrect shard specification, triggering an out-of-bounds memory access inside the kernel's halo configuration logic. Fixed by replicating the kernel's own shard specification formula inside the compiler's op model so it always queries with an internally consistent layout.

---

**8. Conv2d L1 Fragmentation OOM — CameraDeformedCylinder Backbone (opt_level_2)**

The CameraDeformedCylinder Backbone was failing at `opt_level_2` with a runtime out-of-memory error. The simulation reported sufficient L1 space but at runtime no contiguous block of the required size was actually available. Several layout conversion ops had left fragments in L1 that were no longer tracked by the budget estimator (they are treated as short-lived tenants by design), leaving the heap fragmented in a way the simulator could not see. Fixed by adding a guard that preemptively moves any tensor requiring more than 40% of the per-core L1 budget to DRAM, preventing the fragmentation scenario.

---

**9. GridSample Trace Capture Fix — CameraCylinder BEV Transform (trace_enabled)**

When running with TTNN trace enabled, the CameraCylinder BEV Transform was crashing because trace capture forbids any device-to-host memory reads. The `nearest` mode and `align_corners=True` grid precomputation path reads the grid tensor from device to host for CPU preprocessing — valid during the warmup run, but fatal once trace capture is active. Fixed by caching the precomputed grid in the program execution context during the warmup run and reusing it during trace capture, following the same pattern already used for other trace-incompatible resources in the runtime.

---

**10. Performance Results**

Running the full BEV model end-to-end:

- **Baseline (no compiler optimizations):** 0.38 FPS
- **With opt_level_2, program cache, trace, constant folding, Forge optimization passes, and bfloat16 data format override:** **2.05 FPS**

This represents a **~5.4× performance improvement** over the unoptimized baseline.
