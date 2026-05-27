# BEV Block B FPS Improvement Proposals

**Baseline:** 11.94 FPS at `opt_level_2 + BFloat16 + HiFi3 + fp32_dest_acc + trace_enabled`

---

## Current TTNN Execution Structure

Block B has 4 independent camera groups, each executing sequentially:

```
arg[g] (1×192×96×96 bf16)
  → permute [0,2,3,1]           NCHW→NHWC
  → reshape (1×1×9216×192)
  → conv2d(192→64, relu6, w=w_exp[g])  ← different weight per group
  → reshape (1×96×96×64)
  → to_dram + to_layout(row_major)

lut[g] (1×128×64×8×2 bf16)
  → reshape (1×128×64×16)       5D→4D, 2K=16
  → to_dram + to_layout(row_major)

grid_sample(feat_g, lut_g, batch_output_channels=true, K=8)
  → (1×128×64×512) L1 height_sharded

  → to_layout(tile) + to_dram
  → reshape (1×1×8192×512)
  → to_l1(height_sharded <192×512>)
  → conv2d(512→64, w=w_red)     ← SAME weight for ALL 4 groups
  → reshape (1×128×64×64)
  → permute [0,3,1,2]           NHWC→NCHW
  → to_dram                     → out[g] (1×64×128×64)
```

**Key finding:** The reduce-conv weight `model._backbone..._reduce_conv.weight` is the same tensor for all 4 groups (confirmed from `forward_const_eval_1,2,5,10` all calling `load_cached(%arg10)`).

**Estimated time breakdown at 65ms inference:**

| Operation | Time |
|-----------|------|
| 4× conv2d(192→64) | ~10ms |
| 8× layout/DRAM round-trips (input prep) | ~8ms |
| 4× grid_sample(K=8) | ~20ms |
| 8× layout/DRAM round-trips (post-GS) | ~8ms |
| 4× conv2d(512→64) | ~14ms |
| 4× permute + DRAM | ~5ms |

---

## Proposal 1: Cross-Group Super-Batch (grid_sample + reduce-conv)

**Status: IN PROGRESS (current implementation)**

**Concept:** Batch all 4 groups' grid_sample calls into one (batch=4) and all 4 reduce-convs into one (batch=4, same weight).

**TTIR transformation:**

```
BEFORE:
  conv(feat0,w0)→f0,  conv(feat1,w1)→f1,  conv(feat2,w2)→f2,  conv(feat3,w3)→f3
  grid_sample(f0, lut0) → g0    conv2d(g0_flat, w_red) → out0
  grid_sample(f1, lut1) → g1    conv2d(g1_flat, w_red) → out1
  grid_sample(f2, lut2) → g2    conv2d(g2_flat, w_red) → out2
  grid_sample(f3, lut3) → g3    conv2d(g3_flat, w_red) → out3

AFTER:
  conv(feat0,w0)→f0,  conv(feat1,w1)→f1,  conv(feat2,w2)→f2,  conv(feat3,w3)→f3
  concat([f0,f1,f2,f3], dim=0)         → (4×96×96×64)
  concat([lut0,lut1,lut2,lut3], dim=0) → (4×128×64×16)
  grid_sample((4×96×96×64), (4×128×64×16), K=8)  → (4×128×64×512)   [1 launch]
  reshape → (4×1×8192×512)
  conv2d((4×1×8192×512), w_red)        → (4×1×8192×64)               [1 launch]
  split → 4×(1×1×8192×64)
  4× reshape+permute                   → 4×(1×64×128×64)
```

**Why it works:**
- `ttnn.grid_sample` supports batch=N; the batch dimension is processed independently per element (confirmed from `grid_sample_device_operation.cpp`)
- `batch_output_channels=true` operates per-batch-element: each of the 4 inputs independently maps 8 cameras
- All 4 reduce-convs share the same weight, so batching to `batch=4` is semantically identical to 4 independent calls

**Implementation location:** New `GridSampleGroupBatchFuse` pattern in `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`. Pattern anchored on 4 `GridSampleOp` → reshape → `Conv2DOp(same_weight)` chains.

**Estimated gain:** Save 3 grid_sample launches (~15ms) + 3 reduce-conv launches (~11ms) = ~26ms saved → **65ms → ~39ms → ~18 FPS**

---

## Proposal 2: Batch the 4 Identical Reduce-Convs Only

**Status: Proposed**

**Concept:** Simpler, standalone version of Proposal 1 targeting only the 4 reduce-convs (without batching the grid_sample). Since all 4 share the same weight, 4 sequential `conv2d(1×1×8192×512, 512→64)` are fused into 1 batched `conv2d(4×1×8192×512, 512→64)`.

**TTIR transformation:**

```
BEFORE:
  conv2d(g0_nhwc, w_red) → out0
  conv2d(g1_nhwc, w_red) → out1
  conv2d(g2_nhwc, w_red) → out2
  conv2d(g3_nhwc, w_red) → out3

AFTER:
  concat([g0,g1,g2,g3], dim=0)          → (4×1×8192×512)
  conv2d((4×1×8192×512), w_red)         → (4×1×8192×64)
  split(4)                              → 4×(1×1×8192×64)
```

**Match conditions:**
- N `ttir.conv2d` ops with identical weight SSA value
- Same `kernel_size=[1,1]`, stride, padding, groups, dilation
- Inputs are independent (no data dependency between them)

**Implementation:** New `BatchIdenticalConv2DFuse` pattern in `TTIRFusing.cpp`.

**Estimated gain:** Save 3 reduce-conv launches (~11ms) → **65ms → ~54ms → ~14 FPS**

**Note:** This is a safe precursor to Proposal 1. If Proposal 1's grid_sample batching has numerical issues, Proposal 2 alone still provides a measurable benefit.

---

## Proposal 3: Eliminate DRAM Round-Trip After Grid-Sample

**Status: Proposed**

**Concept:** After each `grid_sample` output (L1 height-sharded, row_major), the current IR executes:

```
to_layout(tile) → to_memory_config(DRAM) → reshape → to_memory_config(L1 height-sharded)
```

The reshape `(1,128,64,512) → (1,1,8192,512)` is logically a no-op on the flat row-major layout since 128×64=8192. The DRAM eviction and reload exists only because the reshape op does not directly support height-sharded L1 in the current memory planning. This causes 2 unnecessary DRAM round-trips per group × 4 groups = 8 extra memory operations.

**Approach:** In the TTNN memory planning pass, recognize when a reshape on height-sharded L1 produces a logically equivalent layout (rows×cols preserved in the physical representation) and elide the DRAM eviction. The reshaped tensor remains in L1 height-sharded.

**Implementation:** Modify the TTNN memory planning / layout assignment pass to detect and elide reshape-induced DRAM round-trips when the reshape is shape-preserving in the height-sharded physical layout.

**Estimated gain:** ~8ms saved → **65ms → ~57ms → ~13 FPS**

**Note:** This optimization is orthogonal to Proposals 1 and 2 and can be applied independently. It also applies broadly to any model with height-sharded tensors followed by a shape-preserving reshape.

---

## Summary

| # | Proposal | FPS Gain | Complexity | Status |
|---|----------|----------|------------|--------|
| 1 | Cross-group super-batch (GS + reduce-conv) | ~6 FPS (→18) | High | In Progress |
| 2 | Batch 4 identical reduce-convs only | ~2 FPS (→14) | Medium | Proposed |
| 3 | Elim. DRAM round-trip post grid-sample | ~1 FPS (→13) | Medium | Proposed |
| — | Combined (1+3) | ~7 FPS (→19) | — | — |

**Baseline:** 11.94 FPS at `opt_level_2 + BFloat16 + HiFi3 + fp32_dest_acc + trace_enabled`.
