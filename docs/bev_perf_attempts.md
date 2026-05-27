# BEV Model — Per-Block Performance Attempts & Fixes

Config locked for all runs: `opt_level=2`, `BFloat16`, `HiFi3`, `fp32_dest_acc=True`, `trace=True`, `program_cache=ON`

---

## Block A — `block_A_deformed_backbone`

**Baseline:** 326.45 ± 0.38 ms → **3.05 FPS**

### What was tried

| Attempt | Description | Result |
|---------|-------------|--------|
| Baseline | opt_level_2, BF16, HiFi3, fp32_acc, trace | 3.05 FPS — compile pass, runtime errors |
| fix1 | Conv2d L1 circular buffer clash fix (`usableL1Size` dead-zone guard) | 3.06 FPS — OOM resolved for small tensors |
| fix2 | Height-sharded L1 spill fix (large ToLayout outputs forced to DRAM) | 3.04 FPS — no regression |
| fix3 | Large-tensor fragmentation guard in `ensureFitsL1` (>40% L1 threshold → DRAM) | 3.07 FPS — Block A opt_level_2 now PASS |
| fix3b | Tuning fragmentation threshold | 3.06 FPS — no improvement |
| fix4 | Permute sharding fix (BLOCK_SHARDED data corruption in PermuteOp) | **3.08 FPS** — correctness fix |
| fix3_notrace | Same as fix3 but without trace (baseline comparison) | 3.05 FPS — trace adds ~0.7ms overhead |

### Best result

**3.08 FPS** — improvement minimal; Block A is bottlenecked by 8× deformable conv2d + heavy feature extraction. No further optimizations attempted.

### Fixes applied (in tt-mlir)

1. **Conv2d L1 CB clash** — `L1SpillManagement`: added `usableL1Size` dead-zone that reserves space for circular buffer peak; prevents CB from evicting live tensors mid-op.
2. **Large tensor fragmentation** — `ensureFitsL1`: if a tensor exceeds 40% of usable L1, force it to DRAM to prevent heap fragmentation causing subsequent allocation failures.
3. **PermuteOp BLOCK_SHARDED corruption** — `permute.cpp`: WH-involving permutations on non-full-width-sharded inputs now route through `prim_permute` to avoid silent data corruption.

---

## Block B — `block_B_deformed_bev_transform`

**Baseline:** 62.87 ± 0.61 ms → **15.71 FPS**
**Target:** ≤33.3 ms → 30 FPS

### What was tried

| Attempt | Description | Result |
|---------|-------------|--------|
| Baseline | opt_level_2, BF16, HiFi3, fp32_acc, trace | **15.71 FPS** |
| batch_fusion_v1 | `GridSampleBatchFusion` pass: fuse 32 grid_sample calls (4 cam × 8 LUT) into 4 batched calls using `batch_output_channels=True` | **Crashed** — flatbuffer segfault in `asJson()` |
| batch_fusion_v2 | Fix #1 (flatbuffer field/arg order) applied | **Hung 68+ min** — MLA stalled on 512-ch grid_sample output |
| batch_fusion_v3 | Fix #2 (GridSampleOp OpModel fast-path) applied | **Hung** — MLA stalled on 5D tensors (`1×128×64×8×2`) |
| batch_fusion_v4 | Fix #3a (ToLayoutOp + SliceStaticOp 5D fast-paths) applied; stale `.so` not rebuilt | **Hung** — `.so` was not actually rebuilt (ccache hit on old object) |
| batch_fusion_v5 | Fix #3b (ReshapeOp 5D fast-path) + forced `touch` rebuild | **In progress** |

### Root cause chain

The `GridSampleBatchFusion` TTIR pass correctly fuses 32→4 grid_sample calls, but introduced three cascading issues:

**Issue 1 — Flatbuffer argument order mismatch**
The `CreateGridSampleOp()` generated function places bool scalars before offset fields (flatbuffers convention), but our `createOp()` call passed arguments in schema declaration order. Result: `memoryConfig` offset was cast to `bool`, `out` TensorRef offset was assigned to `memory_config` field → corrupt flatbuffer binary → `asJson()` crash in `GenStruct`.

**Issue 2 — MLA hang on 512-channel grid_sample output**
The fused output is `(1,128,64,512)` — 512 channels = 64 features × 8 LUT levels. `QUERY_OP_CONSTRAINTS(::ttnn::grid_sample, ...)` in OpModel is extremely slow for this shape (metal kernel does internal buffer allocation proportional to output size). MLA calls this per layout candidate → 68-minute hang.

**Issue 3 — MLA hang on 5D tensors**
The grid LUT inputs arrive as `(1,128,64,8,2)` 5D tensors and are unpacked via `to_layout → slice_static → reshape` (32 ops each = 96 total). The metal kernels for these ops don't support 5D — `convertToTensorSpec` hangs for rank>4. MLA probes all layout candidates for all 96 ops → infinite hang.

### Fixes applied (all in tt-mlir)

| Fix | File | Change |
|-----|------|--------|
| **Flatbuffer arg order** | `lib/Target/TTNN/TTNNToFlatbuffer.cpp` | Pass `batchOutputChannels` before `memoryConfig` to match generated `CreateGridSampleOp` signature |
| **Flatbuffer field order** | `include/ttmlir/Target/TTNN/operations/pool.fbs` | `batch_output_channels` field added at END of table (backward compat with pre-built `_C.so`) |
| **GridSampleOp OpModel** | `lib/OpModel/TTNN/TTNNOpModel.cpp` | `batch_output_channels=True`: reject L1 layouts immediately, return DRAM constraint without kernel query |
| **ToLayoutOp OpModel** | `lib/OpModel/TTNN/TTNNOpModel.cpp` | `inputShape.size() > 4`: bypass metal kernel, reject L1, accept DRAM |
| **SliceStaticOp OpModel** | `lib/OpModel/TTNN/TTNNOpModel.cpp` | `inputShape.size() > 4`: same fast-path |
| **ReshapeOp OpModel** | `lib/OpModel/TTNN/TTNNOpModel.cpp` | `inputShape.size() > 4 \|\| outputShape.size() > 4`: same fast-path |
| **GridSampleBatchFusion pass** | `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` | New rewrite pattern: matches `concat(dim=1)` whose inputs are all `grid_sample` with the same feature tensor; fuses into `concat(grids,dim=1) + batched grid_sample` |
| **`prepare_grid_sample_grid`** | `ttnn/.../grid_sample/grid_sample_prepare_grid.cpp` | Support K>1 batched grids in nearest mode: outer loop over K coordinate pairs; assertion relaxed from `last_dim==2` to `last_dim % 2 == 0` |
| **GridSampleOp TTNN dialect** | `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` | Added `DefaultValuedAttr<BoolAttr, "false">:$batch_output_channels` |
| **GridSampleOp TTNN verifier** | `lib/Dialect/TTNN/IR/TTNNOps.cpp` | Relax `grid last_dim != 2` to `last_dim % 2 != 0` to accept 2K grids |
| **GridSampleOp TTIR verifier** | `lib/Dialect/TTIR/IR/TTIROps.cpp` | Same relaxation on dim[1] |
| **TTIR→TTNN conversion** | `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` | Auto-detect K>1 grid (dim[1] = 2K) and set `batch_output_channels=True` |
| **TTNNToEmitC** | `lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp` | Emit `batch_output_channels` attribute |
| **TTNNToEmitPy** | `lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp` | Emit `batch_output_channels` kwarg |
| **OpModelInterface** | `lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp` | Pass `getBatchOutputChannels()` to both cache lookup calls |
| **Benchmark timeout** | `forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py` | 15-min SIGALRM timeout on `_compile_block`; auto-diagnose from IR dumps on timeout |

### Minimal ONNX reproducer

`forge/test/models/onnx/vision/bev/test_5d_mla_hang_repro.py` — reproduces the 5D MLA hang with a minimal model: `input(1,128,64,8,2)` → 8× `Slice` → 8× `Reshape(→1,128,64,2)` → `Concat(dim=3)` → `output(1,128,64,16)`. Confirmed hang at opt_level_2 before fix; used to validate the OpModel fast-path fix.

### Validation note

Block B validation is **disabled** (known grid_sample PCC issue — timing only).

---

## Block C — `block_C_cylinder_backbone`

**Baseline:** 106.91 ± 0.21 ms → **9.31 FPS**

### What was tried

No performance optimization attempted yet. Block C was fixed for correctness as a side effect of Block A work. Performance optimization is on the roadmap.

### Fixes applied (correctness, from Block A work)

1. **ConvTranspose2d compile crash** — `prepare_conv_transpose2d_weights.cpp`: write `weight_dtype` back into `conv_config.weights_dtype` (was being lost, causing JIT type mismatch).
2. **Conv2d L1 CB clash** — same `usableL1Size` dead-zone fix as Block A.

---

## Block D — `block_D_camera_cylinder_bev_transform`

**Result:** 16.97 ± 0.08 ms → **58.37 FPS** ✓ (above 30 FPS target)

### What was tried

Block D correctness fix was the primary effort (silent wrong output at opt_level_2). After fix, performance was already above target — no further optimization needed.

### Fix applied

**PermuteOp BLOCK_SHARDED data corruption** — same fix as Block A. WH-involving permutations on non-full-width-sharded inputs now route through `prim_permute`.

---

## Block E — `block_E_bev_aggregator`

**Result:** 6.26 ± 0.10 ms → **156.05 FPS** ✓ (well above target)

### Fix applied

**Bilinear upsample segfault** — `TTNNOpModel.cpp`: bilinear upsample autoreshards DRAM interleaved → HEIGHT_SHARDED internally. Added OpModel guards to reject L1 input/output layouts so MLA never assigns L1 sharded layouts that the kernel then rejects at runtime.

---

## Block F — Output Heads

**Result:** 5.61 ± 0.45 ms → **173.28 FPS** ✓ (well above target)

No fixes needed — compiled and ran correctly at opt_level_2 from the start.

---

## Summary Table

| Block | Model | FPS | Status | Primary Fix |
|-------|-------|-----|--------|-------------|
| A | `block_A_deformed_backbone` | **3.08** | Correct ✓, perf limited by model | L1 fragmentation + PermuteOp |
| B | `block_B_deformed_bev_transform` | **15.71** baseline; batch fusion in progress | Optimization in progress | GridSample batch fusion (32→4 calls) |
| C | `block_C_cylinder_backbone` | **9.31** | Correct ✓, not yet optimized | ConvT2d crash + CB clash |
| D | `block_D_cylinder_bev_transform` | **58.37** | ✓ Above 30 FPS | PermuteOp corruption |
| E | `block_E_bev_aggregator` | **156.05** | ✓ Above 30 FPS | Upsample segfault |
| F | Output heads | **173.28** | ✓ Above 30 FPS | None |

**Blocks still needing work:** A (3.08 FPS), B (batch fusion in progress — expected >20 FPS), C (9.31 FPS).
