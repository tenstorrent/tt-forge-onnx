# GridSample Batched Fusion Plan

## What the TTIR Currently Looks Like

### Inputs to Block B

```
%arg0..%arg3 : tensor<1x192x96x96xbf16>   — 4 camera feature maps (from backbone)
%arg4..%arg7 : tensor<1x128x64x8x2xbf16>  — 4 LUT grids, 8 cameras each (K=8)
%arg8..%arg17: conv weights and biases
```

### Step 1 — Feature projection (per camera group)

```
%0  = transpose(%arg0, -3, -2)  → [1,96,192,96]
%1  = transpose(%0, -2, -1)     → [1,96,96,192]
%2  = conv2d(%1, %arg8, %arg9)  → [1,96,96,64]
%3  = transpose(%2, -2, -1)     → [1,96,64,96]
%4  = transpose(%3, -3, -2)     → [1,64,96,96]
%5  = clamp(%4, 0, 6)           → [1,64,96,96]   ← feature tensor for group 0
```

Same pattern gives `%57` (group 1), `%109` (group 2), `%161` (group 3).

---

### Step 2 — 8× (index → reshape → T → T → grid_sample)   [per LUT group]

For **group 0** (`%arg4: [1,128,64,8,2]`, feature `%5: [1,64,96,96]`):

```
k=0:
  %6  = index(%arg4, dim=3, 0:1)       → [1,128,64,1,2]
  %7  = reshape(%6)                     → [1,128,64,2]
  %8  = transpose(%7, -3, -1)          → [1,2,64,128]
  %9  = transpose(%8, -2, -1)          → [1,2,128,64]
  %10 = grid_sample(%5, %9)            → [1,64,128,64]

k=1:
  %11 = index(%arg4, dim=3, 1:2)       → [1,128,64,1,2]
  %12 = reshape(%11)                    → [1,128,64,2]
  %13 = transpose(%12, -3, -1)         → [1,2,64,128]
  %14 = transpose(%13, -2, -1)         → [1,2,128,64]
  %15 = grid_sample(%5, %14)           → [1,64,128,64]

  ... (k=2..6 same pattern) ...

k=7:
  %41 = index(%arg4, dim=3, 7:8)       → [1,128,64,1,2]
  %42 = reshape(%41)                    → [1,128,64,2]
  %43 = transpose(%42, -3, -1)         → [1,2,64,128]
  %44 = transpose(%43, -2, -1)         → [1,2,128,64]
  %45 = grid_sample(%5, %44)           → [1,64,128,64]
```

**Op count per group:** 8×index + 8×reshape + 8×transpose + 8×transpose + 8×grid_sample = **40 ops**

---

### Step 3 — Concat + transpose to channel-last + conv2d

```
%46 = concat(%10,%15,%20,%25,%30,%35,%40,%45, dim=1)  → [1,512,128,64]
%47 = transpose(%46, -3, -2)                          → [1,128,512,64]
%48 = transpose(%47, -2, -1)                          → [1,128,64,512]
%49 = conv2d(%48, %arg10, %arg11)                     → [1,128,64,64]
%50 = transpose(%49, -2, -1)                          → [1,128,64,64]
%51 = transpose(%50, -3, -2)                          → [1,64,128,64]   ← output 0
```

Same structure for groups 1–3 using `%arg5/%57`, `%arg6/%109`, `%arg7/%161`,
producing outputs `%103`, `%155`, `%207`.

---

## Why Fusion Is Valid

### Memory layout argument

`%arg4` has shape `[1, 128, 64, 8, 2]` stored row-major.
The 8 camera slices sit contiguously in memory as:
```
[cam0_xy, cam1_xy, ..., cam7_xy]   (innermost two dims: 8×2 = 16 values per spatial point)
```

A reshape to `[1, 128, 64, 16]` produces an identical memory buffer — no data copy.

After `reshape → [1,128,64,16]`, the two transposes are **exactly the same** as applying
them to any single camera slice after `reshape → [1,128,64,2]`. The transpose is a
pure view operation (no data copy) in both cases.

### batch_output_channels semantic

`ttnn::grid_sample` with `batch_output_channels=True` and grid `[N, 2K, H, W]`
samples the input `[N, C, H_in, W_in]` for each of K camera slices independently
and concatenates along the channel dim, producing `[N, K×C, H, W]`.
This is **exactly** what the 8 separate `grid_sample` + `concat` does.

---

## Fused TTIR (after pattern)

For group 0:

```
%fused_grid = reshape(%arg4, [1,128,64,16])       → [1,128,64,16]   // merge K and xy
%fused_t1   = transpose(%fused_grid, -3, -1)      → [1,16,64,128]
%fused_t2   = transpose(%fused_t1, -2, -1)        → [1,16,128,64]   // grid [N,2K,H,W]
%fused_out  = grid_sample(%5, %fused_t2,
                batch_output_channels=True)        → [1,512,128,64]  // K×C channels
```

**Concat is eliminated.**
`%fused_out` feeds directly into `%47 = transpose(%fused_out, -3, -2)`.

---

## Op Count: Before vs After

| | Before | After |
|--|--------|-------|
| index | 8 per group × 4 = **32** | 0 |
| reshape (LUT) | 8 per group × 4 = **32** | 1 per group × 4 = **4** |
| transpose (LUT prep) | 16 per group × 4 = **64** | 2 per group × 4 = **8** |
| grid_sample | 8 per group × 4 = **32** | 1 per group × 4 = **4** |
| concat | 1 per group × 4 = **4** | 0 |
| **Total (LUT section)** | **164** | **16** |

---

## Fusion Rewrite Pattern (TTIRFusing.cpp)

**Pattern to match** (on `ConcatOp`):

1. All K inputs to concat are single-use `GridSampleOp`s
2. Each `GridSampleOp`'s grid comes from `transpose(-2,-1)` of
3. a `transpose(-3,-1)` of
4. a single-use `ReshapeOp` that squeezes a singleton dim
5. whose input is a single-use `IndexOp` on the **same source tensor**
6. `IndexOp`s slice dim=3, offsets 0,1,2,...,K-1 contiguously (step=1, width=1)
7. All `GridSampleOp`s share the **same feature tensor** and same attributes
   (`mode`, `padding_mode`, `align_corners`)

**Replacement:**

```
reshape(lut_src, [..., K*2])       // flatten last two dims
transpose(-3, -1)
transpose(-2, -1)
grid_sample(feature, fused_grid,   // add batch_output_channels=True at TTIR level
            batch_output_channels=True)
```

---

## batch_output_channels in TTIR

Currently `ttir.grid_sample` has no `batch_output_channels` attribute.
Options:

**Option A — Add attribute to TTIR op** (cleaner, explicit in IR):
- Add `DefaultValuedAttr<BoolAttr, "false">:$batch_output_channels` to `TTIROps.td`
- Fusion pattern sets it to `true` when K>1
- `TTIRToTTNN.cpp` passes it through directly (no K-detection needed)

**Option B — Detect at TTNN lowering** (current approach):
- Keep TTIR unchanged
- `TTIRToTTNN.cpp` infers K from grid shape dim[1]/2
- Sets `batch_output_channels = (K > 1)` during lowering

**Recommended: Option A** — makes the IR self-documenting and removes the implicit
inference logic from the lowering pass.

---

## Fixes to Revert (no longer needed after fusion)

Once the concat is eliminated, these workarounds become dead code:

| Fix | File | Reason to Revert |
|-----|------|-----------------|
| `rejectAllSharded` filter | `DataMovementRules.cpp` | No more K=8 concat |
| `shouldExploreReshards()=false` | `DataMovementRules.cpp` | No more K=8 concat |
| `shouldSkipOpModelQuery()=true` | `DataMovementRules.cpp` + `OpRuleBook.h` | No more K=8 concat |
| `evaluateHint` bypass | `MemoryLayoutPropagation.cpp` | No more K=8 concat |
| Cross-product guard | `MemoryLayoutPropagation.cpp` | Can keep as general safety, but concat-specific motivation gone |

---

## Block B Test Split Plan

Each Block B subgraph is an independent op or op group.
Split `test_bev_block_b_ops.py` into per-op test files:

| Test file | Op(s) tested | IR inputs |
|-----------|-------------|-----------|
| `test_block_b_conv2d_encoder.py` | `conv2d + clamp` (feature projection) | `%arg0`, weights |
| `test_block_b_gridsample_fused.py` | fused `reshape+T+T+grid_sample` (K=1 and K=8) | `%arg4`, `%5` |
| `test_block_b_reduce_conv.py` | `conv2d` after grid_sample (`%49`) | `%48`, `%arg10` |
| `test_block_b_conv_transpose2d.py` | `conv_transpose2d` upsampler | upsample inputs |

---

## Implementation Order

1. Add `batch_output_channels` attr to `TTIROps.td` + `TTIROps.cpp` verifier
2. Implement `GridSampleBatchedFuse` pattern in `TTIRFusing.cpp`
3. Update `TTIRToTTNN.cpp` — pass `batch_output_channels` directly (remove K inference)
4. Revert concat workaround diffs in `DataMovementRules.cpp`, `OpRuleBook.h`, `MemoryLayoutPropagation.cpp`
5. Split `test_bev_block_b_ops.py` into per-op test files
6. Run and validate
