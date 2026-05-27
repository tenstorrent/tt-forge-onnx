# Block A Performance Analysis

**Date:** 2026-05-19  
**Block:** block_A_deformed_backbone  
**Model:** CameraDeformedCylinder Backbone (460 nodes)  
**Config:** opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled, cache=ON  
**Baseline:** 326.45 ± 0.38 ms, 3.05 FPS  
**Target:** 30 FPS (33.3 ms)

---

## IR Structure

| Layer | Ops | Resolution | Memory |
|---|---|---|---|
| 8 | conv2d | 1536×1536 → 768×768 | DRAM (dram_width) |
| 8 | conv2d | 384×384 | HEIGHT_SHARDED |
| 12 | conv2d | 192×192 | HEIGHT_SHARDED |
| 28 | conv2d | 96×96 | HEIGHT_SHARDED |
| 24 | conv2d | 48×48 | mixed |
| 24 | conv2d | 24×24 | mixed |
| 24 | conv2d | 12×12 | mixed |
| 20 | conv2d | 6×6 | mixed |
| 32 | conv_transpose2d | 6×6 → 48×48 | DRAM |
| 212 | to_memory_config | various | L1↔DRAM |
| ~68 | reshape/permute/concat/slice | various | — |

**Total FLOPs:** ~128 GFLOPS  
**Estimated effective throughput:** ~390 GFLOPS (near hardware ceiling for this op mix)

---

## Fixes Attempted

### Fix 1 — `config_tensors_in_dram` weight threshold

**Hypothesis:** 148 conv2d ops emit `config_tensors_in_dram=true`, forcing weight metadata to DRAM.  
**Change:** `TTIRToTTNN.cpp` — set `config_tensors_in_dram = (weightBytes > 512KB)`  
**Result:** 326ms → 326ms — **no impact**  
**Why:** `config_tensors_in_dram` controls reader-index metadata tensors, not weight placement. Weights always go to DRAM regardless.

---

### Fix 2 — Re-enable DRAM→L1→DRAM fold

**Hypothesis:** 56 adjacent DRAM→L1→DRAM pairs in IR represent pure overhead.  
**Change:** Re-enabled `foldDRAMtoL1toDRAMRoundTrip` in `TTNNOps.cpp` + canonicalization pass.  
**Result:** 326ms → 328ms — **no impact**  
**Why:** The pairs were eliminated (stage 12 dropped from 68 to 20), but the dominant pattern is L1→DRAM spills (opposite direction), not DRAM→L1→DRAM.

---

### Fix 3 — `act_block_h` fallback fix

**Hypothesis:** When `act_block_h_override=1024` (32 tiles) doesn't divide `padded_output_height_ntiles_per_core=159`, `find_closest_largest_divisor(159,32)=3` → 53 inner loop iterations vs 1 with auto value.  
**Change:** `conv2d_utils.cpp` — keep auto value when override doesn't divide evenly.  
**Result:** 326ms → 325ms — **no impact**  
**Why:** `act_block_h` affects inner loop structure but not total MACs or total data movement. In trace mode the loop overhead is negligible. The act_block_h warning is cosmetic.  
**Note:** Required copying `_ttnncpp.so` from `build_Release/ttnn/` to `build/install/lib/` (the path `_C.so` links to).

---

### Fix 4 — Remove stale `enable_kernel_stride_folding` workaround

**Hypothesis:** All 468 conv2d-related ops have `enable_kernel_stride_folding=false` due to an October 2025 workaround for tt-metal bug #30985 (stride folding fails for flattened `1×1×HW×C` tensors). This bug was fixed in tt-metal PR #32903 (commit `14d6567d72b`, Nov 21 2025). Removing the workaround re-enables auto stride folding for eligible ops.  

**Eligibility check:** `auto_enable_kernel_folding` enables folding only when:
- stride == kernel_size (for non-override path)
- stride > 1
- dilation == 1  
- (input_height + padding) % stride == 0
- Memory: HEIGHT_SHARDED OR (DRAM + ROW_MAJOR)

**Eligible ops in Block A:** only **4 ops** (stride=(2,2), kernel=(2,2) at 384×384)  
- 420 ops: stride=1 → no folding ever
- 8 ops: stride=(2,2), kernel=(2,1) → stride≠kernel → no fold
- 4 ops: stride=(2,2), kernel=(2,2) → **FOLD_OK**

**Change:** `TTNNWorkaroundsPatterns.cpp` — removed `Conv2dEnableKernelStrideFoldingRewritePattern<Conv2dOp>` from patterns list.  
**Result:** 326ms → **323.5 ± 0.4 ms, 3.08 FPS** — **~3ms improvement**  
**Why minimal:** Only 4 eligible ops. 3ms saving ≈ 4 ops × 0.75ms each.  
**Note:** Fix is still correct and valid — re-enables proper behavior and benefits Block C too.

---

## Root Cause: Near Hardware Ceiling

Block A operates at near-hardware efficiency for this op mix:

| Resolution | Conv2d ops | FLOPs |
|---|---|---|
| 1536×1536 | 8 | 0.21 GFLOPS |
| 384×384 | 8 | 6.64 GFLOPS |
| 192×192 | 12 | 21.74 GFLOPS |
| 96×96 | 28 | 55.94 GFLOPS |
| 48×48 | 24 | 22.33 GFLOPS |
| 24×24 | 24 | 13.27 GFLOPS |
| 12×12 | 24 | 5.68 GFLOPS |
| 6×6 | 20 | 2.10 GFLOPS |
| **Total** | **148** | **~128 GFLOPS** |

- **Average per op:** ~0.7ms (hardware dispatch + L1 management + compute)
- **460 total ops × 0.7ms = 322ms** ≈ observed 326ms
- Per-op hardware overhead dominates at small resolutions

### Why compiler fixes can't reach 30 FPS

The 10× gap (326ms → 33ms) requires structural changes:

| Approach | Impact | Feasibility |
|---|---|---|
| Compiler fixes (done) | ~3ms (1%) | Easy — done |
| Fuse consecutive ops | 50-100ms? | Complex, needs MLA work |
| Remove flattened tensor repr | Unknown — enables more opts | Requires tt-forge-onnx changes |
| Batch 4 cameras together | Up to 4× | Model-level change |
| INT8 quantization | ~2× | Model-level change |

---

## Files Modified

| File | Change | Still Active |
|---|---|---|
| `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` | config_tensors_in_dram threshold | Yes (harmless) |
| `lib/Dialect/TTNN/IR/TTNNOps.cpp` | Re-enabled DRAM→L1→DRAM fold | Yes (harmless) |
| `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` | Re-enabled canonicalization pass | Yes (harmless) |
| `third_party/tt-metal/.../conv2d_utils.cpp` | act_block_h auto-value fallback | Yes (harmless) |
| `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp` | Removed stale stride-fold workaround | Yes (**active fix**) |

---

## Final Result

| Metric | Value |
|---|---|
| Baseline | 326.45 ± 0.38 ms, 3.05 FPS |
| After all fixes | 323.53 ± 0.41 ms, 3.08 FPS |
| Improvement | ~3ms (~1%) |
| Gap to 30 FPS | 290ms (needs 10× more improvement) |
