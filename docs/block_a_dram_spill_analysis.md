# Block A (Deformed Backbone) — DRAM Spill Root Cause Analysis

Source IR: `BEV_MODEL_IRS/LATEST/BLOCK_A/ttnn_block_A_deformed_backbone.mlir`

## Key Finding

All 72 spilled conv2d ops compute in L1 (`L1=True`, height_sharded). The spill is **NOT
capacity-related** — even 1.8 KB/core outputs are evicted to DRAM. The `to_memory_config(DRAM)`
is inserted because the **CONSUMER** cannot accept L1 sharded input at that grid configuration.

## Block A Structure

- 148 conv2d ops, 20 max_pool2d, 56 slice_static, 48 concat
- 4× camera-group replicas (patterns repeat 4×)
- Spatial scales: 1536×1536 → 384×384 → 192×192 → 96×96 → 48×48 → 24×24 → 12×12 → 6×6

## Spill Group Breakdown (72 total spills)

| Output shape | Tensor size | Per-core KB | Count | Consumer pattern |
|---|---|---|---|---|
| 1×384×384×64 (1×1 conv) | 18.0 MB | 288 KB | 4 | conv2d (stride=2) |
| 1×192×192×64 (dw stride=2) | 4.5 MB | 72 KB | 4 | to_memory_config(L1) |
| 1×96×96×192 (1×1) | 3.4 MB | 60 KB | 8 | conv2d |
| 1×192×192×32 (3×3) | 2.25 MB | 36 KB | 8 | unknown |
| 1×96×96×128 (3×3) | 2.25 MB | 40 KB | 4 | conv2d |
| 1×96×96×96 (3×3) | 1.69 MB | 30 KB | 4 | conv2d |
| 1×48×48×192 (3×3 + 1×1) | 0.84 MB | — | 12 | conv2d |
| 1×48×48×160 (3×3) | 0.70 MB | — | 4 | conv2d |
| 1×24×24×{384,256,192} | 0.05–0.42 MB | — | 12 | unknown |
| **1×12×12×192, 1×6×6×192** | **0.013–0.053 MB** | **1.8 KB** | **8** | unknown |

## Three Distinct Spill Patterns

### Pattern 1 — Chain Spill (64 / 72 cases)

```
conv2d(input=L1_sharded, output=L1_height_sharded, 64 cores)
  → to_memory_config(DRAM)          ← MLA forces eviction
  → conv2d(input=DRAM, stride=2, output=DRAM)
  → to_memory_config(DRAM)
  → conv2d(input=DRAM, ...)
  ...
```

Once an op lands in DRAM, the entire downstream chain stays in DRAM.
Repeated 3–4× per spatial scale. Each hop costs full DRAM bandwidth on 4–18 MB tensors.

**Example (IR lines 1695–1712):**
```
%207 = "ttnn.conv2d"(%182, ...) -> tensor<1x384x384x64xbf16, #ttnn.tensor_memory_layout<...
  height_sharded, shards=64, L1=true>>
%208 = "ttnn.to_memory_config"(%207, ...) -> tensor<..., DRAM>
%209 = "ttnn.conv2d"(%208, ..., stride=[2,2]) -> tensor<1x192x192x64xbf16, DRAM>
```

### Pattern 2 — Bounce Spill (8 cases)

```
conv2d → L1_height_sharded
  → to_memory_config(DRAM)               ← unnecessary eviction
  → to_memory_config(L1_interleaved)     ← re-load into L1
  → to_memory_config(L1_interleaved)     ← redundant
  → slice_static(L1_interleaved)         ← slice requires interleaved
  → to_memory_config(DRAM)               ← final spill
```

Three unnecessary data copies. The root cause is `slice_static` requiring L1-interleaved layout,
which forces a DRAM transit to change the sharding mode.

### Pattern 3 — Tiny Spill (8 cases, deep layers)

```
conv2d(1×12×12×192, 6 cores) → L1, 1.8 KB/core → to_memory_config(DRAM)
conv2d(1×6×6×192, 6 cores)   → L1, 1.8 KB/core → to_memory_config(DRAM)
```

At 1.8 KB/core, there is no capacity justification. MLA is being overly conservative at small
spatial scales.

**Example (IR line 1907):**
```
%294 = "ttnn.conv2d"(%293, ...) -> tensor<1x12x12x192xbf16, #ttnn.tensor_memory_layout<...
  height_sharded, shards=6, L1=true>>
%295 = "ttnn.to_memory_config"(%294, ...) -> tensor<..., DRAM>
```

## Root Causes (Ranked by Impact)

| Rank | Cause | Spill count | Fix complexity |
|------|-------|-------------|----------------|
| 1 | Strided conv2d (stride=2) cannot accept height_sharded L1 input | 4 | Medium |
| 2 | `slice_static` requires L1-interleaved → forces DRAM transit | 8 | Medium |
| 3 | MLA over-evicts tiny tensors at 12×12 / 6×6 spatial scales | 8 | Low |
| 4 | Chain propagation: once one op spills, all downstream ops stay in DRAM | 60 | Blocked by #1 |

## Proposed Fixes

### Fix A — Eliminate Bounce Spill (8 cases, Pattern 2)

Detect L1→DRAM→L1 bounce in `TTNNDecomposeLayouts`. If a `to_memory_config(DRAM)` is immediately
consumed by a `to_memory_config(L1)`, collapse the round trip.

**Expected impact:** 8 fewer DRAM round trips, removes redundant bandwidth on ~4 MB tensors.

**Location:** `third_party/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNDecomposeLayouts.cpp`

### Fix B — Suppress Tiny Spills (8 cases, Pattern 3)

In MLA or post-MLA cleanup: if output tensor < 32 KB/core AND the consumer can read L1, do not
insert `to_memory_config(DRAM)`.

**Expected impact:** 8 fewer DRAM round trips at 12×12 / 6×6 scales.

**Location:** MLA pass or `TTNNDecomposeLayouts.cpp` post-layout fixup.

### Fix C — L1→L1 Chaining for Sequential Conv2d (60 cases, Pattern 1)

Allow sequential height_sharded conv2ds with compatible shard grids to chain directly in L1,
skipping the DRAM intermediate. Requires verifying that the TTNN metal conv2d kernel accepts
L1-sharded input (not just DRAM-interleaved).

**Expected impact:** This is the highest-leverage fix. If 60 chained-spill ops can be avoided,
Block A inference time could drop significantly.

**Location:** TTNN metal conv2d kernel input support, then MLA sharding propagation.

### Fix D — Sharding-Compatible Stride=2 Conv2d (4 cases)

For stride=2 conv2d where the input is height_sharded: reorganize the input shard spec so that
the stride-2 downsampling preserves shard alignment (input 2× rows per shard → output 1× rows
per shard at same core count).

**Expected impact:** 4 cases at large spatial scales (384×384), removes the root cause of chain
propagation for the largest tensors.

**Location:** TTNN memory layout pass, stride=2 shard spec calculation.

## Recommended Fix Order

1. **Fix B** (tiny spills) — low risk, no kernel changes, clear win
2. **Fix A** (bounce elimination) — pattern is already partially handled in `TTNNDecomposeLayouts`
3. **Fix C** (L1→L1 chaining) — highest impact, needs metal kernel validation first
4. **Fix D** (stride=2 input sharding) — architectural, requires careful shard math
