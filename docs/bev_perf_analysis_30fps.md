# BEV Model Performance Analysis — Path to 30 FPS

**Date:** 2026-05-18  
**Current baseline:** 2.04 FPS (490 ms/frame, opt_level_2, trace, float16_b, HiFi3, program cache)  
**Target:** 30 FPS (33 ms/frame)  
**Required speedup:** 14.8×

---

## 1. Benchmark Configuration

```python
MLIRConfig()
  .set_enable_consteval(True)
  .set_optimization_level(2)
  .set_compute_cfg_math_fidelity(HiFi3)
  .set_compute_cfg_fp32_dest_acc_en(True)
  .set_enable_trace(True)

compiler_cfg.default_df_override = DataFormat.Float16_b
compiler_cfg.enable_optimization_passes = True
device_settings.enable_program_cache = True
```

Results: 490.23 ± 7.78 ms inference (H2D + run + D2H), PCC > 0.99 on all 3 outputs.

---

## 2. Op-Type Breakdown (Full Graph)

| Op | Count | Notes |
|---|---|---|
| `ttnn.deallocate` | 1370 | Memory management overhead |
| `ttnn.to_memory_config` | 567 | Reshards / spill-restore — **primary overhead** |
| `ttnn.conv2d` | 226 | Main compute |
| `ttnn.reshape` | 136 | Format utility |
| `ttnn.to_layout` | 127 | ROW_MAJOR ↔ TILE conversions |
| `ttnn.slice_static` | 120 | Index/slice ops |
| `ttnn.concat` | 73 | Feature aggregation |
| `ttnn.grid_sample` | 40 | BEV transform — **single core, DRAM** |
| `ttnn.permute` | 31 | NCHW ↔ NHWC transposes |
| `ttnn.max_pool2d` | 29 | All spill to DRAM |
| `ttnn.add` | 28 | Residuals |
| `ttnn.multiply` | 24 | Scale/activation |
| `ttnn.conv_transpose2d` | 20 | Decoder upsampling |
| `ttnn.upsample` | 4 | Bilinear upsample — single core |
| `ttnn.relu6` | 3 | Activation |

**Overhead vs compute ratio:**

| Category | Count |
|---|---|
| Overhead ops (`to_memory_config` + `to_layout` + `deallocate`) | 2064 |
| Compute ops (`conv2d` + `add` + `mul` + `grid_sample` + `pool` + `concat`) | 424 |

**5:1 overhead-to-compute ratio** is the primary performance problem.

---

## 3. Sharding Effectiveness Analysis

Out of 861 shardable ops:

| Status | Count | Percentage |
|---|---|---|
| Sharded (L1, any) | 336 | 39.0% |
| **Effectively sharded** (L1, not spilled) | **162** | **18.8%** |
| Spilled to DRAM | 477 | 55.4% |
| Sharded then immediately spilled | 174 | 20.2% |

Only **18.8% of ops genuinely stay in L1**. The rest go to DRAM or transition through it.

### Per-op sharding breakdown:

| Op | Total | Sharded | Effectively in L1 | DRAM spilled |
|---|---|---|---|---|
| `ttnn.conv2d` | 226 | 209 (92%) | 103 (46%) | 106 (47%) |
| `ttnn.max_pool2d` | 29 | 29 (100%) | **0 (0%)** | 29 (100%) |
| `ttnn.add` | 28 | 28 (100%) | 16 (57%) | 12 (43%) |
| `ttnn.multiply` | 24 | 23 (96%) | 21 (88%) | 2 (8%) |
| `ttnn.concat` | 73 | 30 (41%) | 13 (18%) | 17 (23%) |
| `ttnn.conv_transpose2d` | 20 | 5 (25%) | 5 (25%) | 0 |
| `ttnn.grid_sample` | 40 | **0 (0%)** | **0 (0%)** | 0 |
| `ttnn.upsample` | 4 | **0 (0%)** | **0 (0%)** | 0 |
| `ttnn.reshape` | 136 | 9 (7%) | 4 (3%) | 5 (4%) |
| `ttnn.permute` | 31 | 0 (0%) | 0 (0%) | 0 |
| `ttnn.slice_static` | 120 | 0 (0%) | 0 (0%) | 0 |
| `ttnn.to_layout` | 127 | 0 (0%) | 0 (0%) | 0 |

---

## 4. Root Cause Analysis by Bottleneck

### 4.1 `grid_sample` — No OpModel, Forced to Single Core

**Impact: HIGH — 40 ops, each serialized to 1 core with DRAM**

The OpModel in `lib/OpModel/TTNN/TTNNOpModel.cpp` returns an unimplemented error:
```cpp
llvm::Expected<OpConstraints> OpModel<GridSampleOp>::getOpConstraints(...) {
  return llvm::createStringError(..., "GridSampleOp op model not implemented");
}
```

Because MLA has no cost model, it cannot assign a sharded layout and falls back to DRAM interleaved with a 1×1 core grid.

**The kernel already supports multi-core.** In `grid_sample_nearest_program_factory.cpp`:
```cpp
// Case 2: Grid not sharded — auto-shard across full compute grid
auto [num_cores_used, all_cores_range, ...] =
    split_work_to_cores(compute_grid_size, total_grid_points);
output_memory_config = MemoryConfig(HEIGHT_SHARDED, L1, output_shard_spec);
```

For the BEV model (grid 128×64 = 8192 points, 64 cores): 128 points/core, output shard `[128, 64]` ≈ 16 KB/core → easily fits in L1.

**Current layout of all 40 grid_sample ops:**
```
<1x1>, memref<8192x64xbf16, #dram>, interleaved
```

**What it should be with a proper OpModel:**
```
<8x8>, memref<128x64xbf16, #l1>, height_sharded
```

Since there is also no `sharded_reader` path for the input tensor in the bilinear kernel, the nearest-mode path uses a `start_id` reader that strides through DRAM. Giving the kernel a proper memory config allows the HEIGHT_SHARDED path and the sharded reader (`reader_grid_sample_sharded.cpp`) which avoids global DRAM broadcast.

### 4.2 `to_memory_config` — 567 Reshards Dominating Execution

**Impact: HIGH — reshards outnumber compute ops 2.5:1**

Every conv2d output that spills to DRAM (106 ops) generates:
- 1 `to_memory_config` for the spill itself
- 1+ `to_memory_config` per downstream consumer to restore from DRAM to L1

With each reshard costing ~0.3–0.5 ms (DRAM → L1 conversion for typical feature map sizes), 567 reshards contribute an estimated **170–280 ms** of the 490 ms total — **35–57% of inference time is data movement, not compute**.

Root cause: the L1 spill manager (`L1SpillManagement.cpp`) evicts aggressively because:
1. `kMaxSingleTensorFraction = 0.40` — any tensor >40% of per-core L1 budget is preemptively demoted to DRAM
2. Dead zone guard — ops with CB > dead zone are demoted before they run
3. ToLayoutOp outputs (format conversion intermediates) are not tracked by the budget estimator, fragmenting L1 in ways the simulator cannot see

### 4.3 `max_pool2d` — Computes in L1, Immediately Spills (29/29)

**Impact: MEDIUM — 29 unnecessary DRAM round-trips in the encoder**

Every max_pool2d computes with HEIGHT_SHARDED input (correct) but its output is flagged for DRAM. Downstream consumers (the next conv2d block) then read from DRAM and reshard back. This adds 29 avoidable DRAM round-trips in the backbone.

Likely cause: when `evictForDramCBGrowth` runs after a preceding conv2d demotes to DRAM (because conv CB footprint grows when output switches from globally-allocated to locally-allocated), the pool output is evicted as part of the CB overlap clearance.

### 4.4 `conv_transpose2d` — 75% Unsharded (15/20 at DRAM)

**Impact: MEDIUM — decoder path runs mostly at single core**

The 5 sharded conv_transpose2d are the small ones (`6x6` grid, small output). The 15 unsharded ones are the larger decoder ops. These run without L1 sharding and produce DRAM output, adding reshards for all consumers.

### 4.5 `upsample` — All 4 at Single Core

**Impact: LOW-MEDIUM — 4 bilinear upsample ops, largest outputs 4 MB**

All 4 upsample ops run at `1x1, DRAM`. The largest (`32768x64`) is 4 MB. These were intentionally blocked from sharding (to fix the earlier segfault from incorrect shard spec in the OpModel query), but the block is too conservative — it rejects all sharded L1 instead of finding the valid shard shape.

### 4.6 Narrow Grid Shapes in conv2d

**Impact: MEDIUM — suboptimal core utilization for some conv layers**

148/226 conv2d ops use `Nx1` grids (height-only sharding). Representative distribution:
```
64x1 → 48 ops   (uses all 64 cores but no width parallelism)
58x1 → 32 ops   (uses 90% of cores)
36x1 → 20 ops   (uses 56% of cores)
18x1 → 16 ops   (uses 28% of cores)
 1x1 → 11 ops   (single core)
```

The `Nx1` pattern is correct for HEIGHT_SHARDED conv2d, but 18×1 and 1×1 grids indicate insufficient work for full device utilization on small feature maps.

---

## 5. Compiler Improvements (tt-mlir)

### Fix 1: Implement `GridSampleOp` OpModel — **Highest Priority**

**File:** `lib/OpModel/TTNN/TTNNOpModel.cpp`

Replace the stub with a real implementation that models the kernel's HEIGHT_SHARDED output:

```cpp
llvm::Expected<OpConstraints> OpModel<GridSampleOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> gridShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr gridLayout, llvm::StringRef mode,
    llvm::StringRef paddingMode, bool alignCorners,
    TTNNLayoutAttr outputLayout) {
  // Nearest mode: kernel auto-shards H_out*W_out across full compute grid
  const uint64_t numGridPoints = gridShape[1] * gridShape[2];  // H_out * W_out
  const uint64_t numCores = deviceGrid.getNumRows() * deviceGrid.getNumCols();
  const uint64_t pointsPerCore = llvm::divideCeil(numGridPoints, numCores);
  const uint64_t channels = inputShape[3];  // NHWC

  // CB: precomputed grid (2 elements × bf16) + output shard
  const uint64_t gridCBBytes   = pointsPerCore * 2 * sizeof(uint16_t);
  const uint64_t outputShard   = pointsPerCore * channels * sizeof(uint16_t);
  const uint64_t cbPeak        = gridCBBytes + outputShard;
  const uint64_t outputL1Usage = outputShard;

  return OpConstraints{cbPeak, cbPeak, outputL1Usage, /*valid=*/true};
}
```

This tells MLA that grid_sample outputs HEIGHT_SHARDED L1, letting it chain sharded layouts through the entire BEV transform section without DRAM round-trips.

**Expected gain:** Eliminates DRAM spill for 40 grid_sample outputs + enables downstream ops to receive data already in L1. Estimated: removes ~40–80 `to_memory_config` reshards.

### Fix 2: Tune `kMaxSingleTensorFraction` — **High Priority**

**File:** `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`

Current value: `0.40` (40% of per-core budget triggers preemptive DRAM demotion).

This was set conservatively to prevent fragmentation OOM. The analysis shows it causes 106/226 conv2d outputs (47%) to spill. Tuning it upward to `0.55`–`0.60` would allow larger tensors to remain in L1.

The risk is re-introducing the fragmentation OOM fixed earlier. A safer approach: combine a relaxed threshold with the existing address-simulation guard — only demote if both the size threshold AND the contiguous-fit check fail.

### Fix 3: Fix max_pool2d Output Spill

**File:** `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`

After `evictForDramCBGrowth` runs for a conv2d demotion, the max_pool2d output that follows in the schedule gets evicted. Add a protected-window heuristic: if an op's output fits in L1 and its next consumer is within 2 schedule positions, do not evict it during CB overlap clearance.

### Fix 4: Relax Upsample Sharding Restriction

**File:** `lib/OpModel/TTNN/TTNNOpModel.cpp` (UpsampleOp section)

The current fix rejects all sharded L1 layouts for bilinear upsample. The underlying issue was an incorrect shard height passed to the kernel's halo calculation. The valid shard height formula is `input_H / num_cores_H` with the kernel's own alignment rules. Implement the formula in the OpModel so that only valid shard specs are proposed, instead of rejecting all sharding.

### Fix 5: `getOpRuntime` for grid_sample

**File:** `lib/OpModel/TTNN/TTNNOpModel.cpp`

Along with `getOpConstraints`, implement `getOpRuntime` with a bandwidth model:

```cpp
llvm::Expected<size_t> OpModel<GridSampleOp>::getOpRuntime(...) {
  // Input is broadcast DRAM read, output is L1 write
  // Each core reads full input (H_in*W_in*C bytes) + grid shard + writes output shard
  const uint64_t inputBytes  = inputShape[1] * inputShape[2] * inputShape[3] * 2;
  const uint64_t outputBytes = gridShape[1] * gridShape[2] * inputShape[3] * 2;
  // Approximate: bottlenecked by DRAM read of input per core
  return (inputBytes + outputBytes) * 1000 / kDRAMBandwidthBytesPerUs;
}
```

This allows the scheduler to make informed decisions when multiple layout configs are compared.

---

## 6. What Single-Chip Can Realistically Achieve

With all compiler fixes above applied:

| Improvement | Estimated Saved Time | New Total |
|---|---|---|
| Baseline | — | 490 ms |
| grid_sample HEIGHT_SHARDED (Fix 1) | ~60–100 ms | ~390 ms |
| Reduced spills via better L1 budget (Fix 2) | ~60–80 ms | ~310 ms |
| max_pool2d stop spilling (Fix 3) | ~20–30 ms | ~280 ms |
| Upsample sharding (Fix 4) | ~10–20 ms | ~265 ms |
| **Optimistic single-chip total** | — | **~250–300 ms** |
| **New FPS** | — | **~3.3–4.0 FPS** |

**Realistic single-chip ceiling: ~4–5 FPS** after all compiler improvements.

---

## 7. Path to 30 FPS — What Is Required

Reaching 30 FPS (33 ms/frame) from ~4–5 FPS after compiler work requires an additional **6–8× speedup**. This cannot come from the compiler alone. The following approaches are required:

### 7.1 Multi-Device Pipeline Parallelism (most realistic path)

The BEV model has a natural 2-branch structure that can run in parallel:
- **Branch A:** CameraDeformedCylinder Encoder (Backbone + BEV Transform) 
- **Branch B:** CameraCylinder Encoder (Backbone + BEV Transform)

Both branches are independent until the BEV Aggregator. With 2 Wormhole cards running the two branches concurrently, effective throughput doubles.

With a 4-card pipeline (2 encoders + aggregator + output heads), pipelining can achieve throughput ≈ `1 / max(T_stage)`. If each stage is ~80 ms, pipeline throughput ≈ 12 FPS.

For 30 FPS with pipelining: stages need to average ~33 ms each, which requires further per-stage optimization (quantization or model pruning).

### 7.2 INT8 Quantization

Switching from bfloat16 to INT8 for backbone conv layers:
- 2× weight bandwidth reduction (half the DRAM reads)
- 2× arithmetic throughput (INT8 ops are faster than bfloat16)
- Expected: 2–3× speedup on conv-heavy backbone paths

Tenstorrent supports INT8 TTNN kernels. The key challenge is maintaining PCC > 0.99 after quantization.

### 7.3 Reduce `grid_sample` Invocation Count via Batching

Currently 40 independent `grid_sample` calls. The BEV model applies each camera's grid to the same feature map. All 40 share the same input tensor (`1x64x96x96` or `1x64x80x144`). 

The `batch_output_channels=True` and grid batching flags in the TTNN kernel allow K grid sets to be processed in a single kernel invocation. If 8 cameras can be batched into a single call with K=8, the 40 ops reduce to 5, cutting kernel dispatch overhead 8×.

This requires changes to both the ONNX ingestion pass (to recognize the batching opportunity) and the runtime (to pass the batched grid).

### 7.4 Depthwise Conv / Model Architecture Changes

Many of the 226 conv2d ops in the backbone are separable. Replacing 3×3 standard convolutions with depthwise-separable convolutions reduces multiply-accumulate count by ~9× with minimal accuracy loss. This is a model-level change (outside the compiler scope) but has the largest potential impact.

---

## 8. Recommended Action Plan

### Immediate (compiler, 1–2 weeks)

1. Implement `GridSampleOp::getOpConstraints` + `getOpRuntime` in `TTNNOpModel.cpp`
2. Tune `kMaxSingleTensorFraction` from 0.40 → 0.55 with guarded fallback
3. Protect max_pool2d outputs from CB-overlap eviction

**Expected outcome:** ~3.5–4.5 FPS

### Short-term (compiler + runtime, 2–4 weeks)

4. Implement valid shard-height formula for bilinear upsample in OpModel
5. Enable `batch_output_channels` batching for multiple grid_sample calls sharing the same input feature map
6. Profile per-op timing with TTNN perf trace to identify the true heaviest single ops

**Expected outcome:** ~5–6 FPS

### Medium-term (system-level, 1–2 months)

7. Multi-device pipeline: split the two encoder branches across 2 Wormhole cards
8. INT8 quantization for backbone conv layers (post-training quantization)

**Expected outcome:** ~12–20 FPS with 4 devices

### Long-term (model + system)

9. Depthwise-separable conv replacement for 3×3 backbone convolutions
10. Further multi-chip scaling (8 cards)

**Expected outcome:** 30 FPS is achievable with model architecture changes + 4–8 cards
