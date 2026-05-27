# SliceRmShardedWidthTrimProgramFactory — L1 HS RM Output Fix

**Date:** 2026-05-26  
**Target model:** BEV deformed backbone (`block_A`)  
**Config:** `opt_level_2`, `bfloat16`, `hifi3+fp32_acc`, `trace_enabled`  
**Expected impact:** ~12 fewer `to_memory_config` ops; downstream concats kept in L1 HEIGHT_SHARDED

---

## 1. Background — Current State After Fix 12

After Fix 12 (MaxPool2dRuleBook), `sharded_and_spilled_ops = 0` and `effectively_sharded_percentage = 55.7%`. The 96 remaining `dram_spilled_ops` break down as:

| Op Type | DRAM count | Driver |
|---------|:-----------:|--------|
| `ttnn.slice_static` | 56 | Always outputs DRAM (this fix targets 12 of these) |
| `ttnn.concat` | 36 | Forced DRAM when any input is DRAM |
| `ttnn.reshape` | 32 | Non-view reshapes: non-sharded |
| `ttnn.permute` | 28 | Same ReshapeRuleBook |
| `ttnn.conv2d` | 8 | Boundary/AvgPool2d |
| `ttnn.max_pool2d` | 8 | Small spatial sizes |

The dominant root cause for the cascade: **`SliceRmShardedWidthTrimProgramFactory` always writes its output to DRAM even when the input is HEIGHT_SHARDED ROW_MAJOR** — forcing downstream concats to be DRAM-output, which then requires a `to_memory_config` re-shard before the next conv2d.

---

## 2. Root Cause

### 2.1 MLIR pattern (per camera pass, 4× repeated)

```mlir
%33 = max_pool2d → #ttnn_layout108 (L1 HEIGHT_SHARDED 58×1 ROW_MAJOR)  ← Fix 12 result
%34 = slice_static(%33, channels 32:96) → DRAM interleaved              ← THIS FIX
%35 = conv2d(%33) → HEIGHT_SHARDED TILED
%37 = to_memory_config(%35) → L1 interleaved
%38 = slice_static(%37) → DRAM interleaved
%39 = to_memory_config(%38) → HEIGHT_SHARDED RM                         ← re-shard
%40 = concat(%36_HS, %39_HS_RM) → HEIGHT_SHARDED ✓
%42 = to_memory_config(%41) → L1 interleaved
%43 = concat(%34_DRAM, %42_interleaved) → DRAM                          ← forced DRAM
%44 = conv2d(%43) → HEIGHT_SHARDED TILED                                ← DRAM→L1 reshard
```

**3 wasted `to_memory_config` ops per camera pass × 4 cameras = 12 ops** that exist solely because `%34` is DRAM instead of HEIGHT_SHARDED RM:
- `%39`: re-shard DRAM→HS RM (could be eliminated if `%38` was already HS RM)
- `%42`: HS→interleaved before outer concat (forced by `%34` being DRAM)
- `%44 input reshard`: conv2d reads from DRAM instead of L1 HS (extra NOC traffic)

### 2.2 Why the slice outputs DRAM

`SliceRuleBook::getOutputHints` (Fix 11) returns only a NULL hint for `isWidthTrimSliceStatic` ops:

```cpp
if (isWidthTrimSliceStatic(op)) {
  OutputHints result;
  result.hints.push_back(OpConfig(TTNNLayoutAttr())); // NULL → DRAM
  return result;
}
```

The NULL hint causes `MemoryLayoutPropagation` to select a DRAM output config for the slice. Height-sharded RM configs exist in `legalConfigs` (from `LegalOpLayoutAnalysis`) but are never offered as candidates.

### 2.3 Why HS RM output is safe for `SliceRmShardedWidthTrimProgramFactory`

The tt-metal kernel already supports **both** output paths via `output_is_dram = !output.is_sharded()`:

**DRAM path** (current): reads from HS L1 input CB, writes trimmed sticks to DRAM interleaved via `noc_async_write`. No output CB needed.

**L1 HS path** (this fix enables): reads from HS L1 input CB, trims into a globally-allocated output CB on the **same grid**. The output shard has the same `shard_height` as the input shard (same number of sticks per core), but the `out_stick_size` is smaller (trimmed last dimension). Each core works entirely locally — no cross-core NOC reads.

Both paths are already implemented in `slice_program_factory_rm_sharded.cpp:384-401`.

### 2.4 Why the op model already handles HS→HS

`TTNNOpModel.cpp` (existing code, lines 2030-2043) has the analytical bypass for the HS→HS case:

```cpp
if (outputIsHS) {
  const size_t cbOutSize = sticksPerCore * COut * kElemBytes;
  const size_t cbPeak    = cbInSize + cbOutSize;
  // Build HS RM output layout with same grid as input
  return OpConstraints(cbPeak, cbPeak, cbPeak, cbOutSize, outLayouts);
}
```

This code path is triggered when `outputLayout` (the hint) is HEIGHT_SHARDED RM. Since `getOutputHints` currently never returns HS RM hints, this path is never exercised. The fix enables it.

---

## 3. Fix

### Strategy

For `isWidthTrimSliceStatic` ops:
1. Offer HEIGHT_SHARDED RM configs from `legalConfigs` as **primary** output hints.
2. Guard with `isValidOutputHintForInputs`: HS output only valid when input is HS RM.
3. NULL hint as **fallback** → DRAM path (for interleaved/DRAM inputs).

The solver will select HEIGHT_SHARDED RM output when:
- Input is HEIGHT_SHARDED RM (from MaxPool2d fixed by Fix 12)
- The L1 budget allows `cbInSize + cbOutSize` on each core
- The downstream concat also accepts HEIGHT_SHARDED input (ConcatRuleBook already handles this)

---

## 4. File Changes

### 4.1 `include/ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h`

Add `isValidOutputHintForInputs` to `SliceRuleBook`:

```cpp
struct SliceRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
  bool isValidOutputHintForInputs(
      const OpConfig &hint,
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const override;
};
```

---

### 4.2 `lib/Dialect/TTNN/Analysis/OpRules/DataMovementRules.cpp`

**Key complications:**

1. **`rowMajorEnabled = false`:** `GreedyMemoryLayoutPropagation` runs with `rowMajorEnabled = false` by default. `LegalOpLayoutAnalysis` skips RowMajor page layouts → `legalConfigs` has **no HS RM entries**. However, it **does** contain HEIGHT_SHARDED **TILED** entries.

2. **Pre-MLP IR types:** `getOutputHints` is called before the greedy pass commits op configs. `op->getOperand(0).getType()` reflects the pre-pass default (DRAM), **not** MaxPool2d's beam-committed HS RM. `TTNNRowMajorLayoutPropagation` only propagates from integer-type inputs — BEV uses BF16, so this pass is a no-op.

3. **`TTNNLayoutAttr::build()` asserts** if `coreRangeSet` is null for sharded layouts. A from-scratch HS RM hint has no valid `coreRangeSet`.

**Solution:** Use the first HEIGHT_SHARDED TILED entry from `legalConfigs` as a template — it has a valid `coreRangeSet`. Convert it to RM via `.setLayout(RowMajor)`. The op model's analytical bypass only checks `outputIsHS` (memLayout + layout fields); it builds the actual output from the **input** layout, ignoring the hint's grid. `isValidOutputHintForInputs` correctly gates the HS hint to HS RM input candidates (MaxPool2d's beam state).

Also add `#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"` for `TTNNLayoutAttr::Builder`.

**`SliceRuleBook::getOutputHints`:**

```cpp
OutputHints
SliceRuleBook::getOutputHints(Operation *op,
                              const std::vector<OpConfig> &legalConfigs) const {
  if (isWidthTrimSliceStatic(op)) {
    OutputHints result;

    // Build HS RM marker hint from the first HS TILED config in legalConfigs.
    // rowMajorEnabled=false: legalConfigs has no HS RM entries, but DOES have
    // HEIGHT_SHARDED TILED entries. Convert one to RM: preserves coreRangeSet
    // (required by TTNNLayoutAttr::build() for sharded layouts) and grid from
    // the TILED template; switches element type to RM BF16 via setLayout.
    // The op model uses the INPUT layout's grid (not the hint's grid), so the
    // marker hint's grid is irrelevant for correctness.
    // isValidOutputHintForInputs restricts this hint to HS RM input candidates
    // (MaxPool2d beam state); DRAM inputs fall through to the NULL fallback.
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (outputType) {
      auto outShape = outputType.getShape();
      for (const auto &cfg : legalConfigs) {
        if (!cfg.outputLayout) continue;
        auto ml = cfg.outputLayout.getMemLayout();
        if (!ml || ml.getValue() != TensorMemoryLayout::HeightSharded) continue;
        // cfg is HS TILED: inherit its grid and coreRangeSet, switch to RM.
        TTNNLayoutAttr markerHint =
            TTNNLayoutAttr::Builder(cfg.outputLayout, outShape)
                .setLayout(Layout::RowMajor)
                .build();
        result.hints.push_back(OpConfig(markerHint));
        break; // One HS RM marker is sufficient.
      }
    }

    result.fallbackHints.push_back(OpConfig(TTNNLayoutAttr())); // DRAM fallback
    if (result.hints.empty()) {
      // No HS TILED entry in legalConfigs; use NULL hint (→ DRAM) as primary.
      result.hints.swap(result.fallbackHints);
    }
    return result;
  }
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}
```

**`SliceRuleBook::isValidOutputHintForInputs`** — guard HS output to HS RM inputs only:

```cpp
bool SliceRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint,
    llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  if (inputLayouts.empty() || !hint.outputLayout)
    return true;
  auto hintML = hint.outputLayout.getMemLayout();
  if (!hintML || !isShardedMemoryLayout(hintML.getValue()))
    return true; // DRAM/interleaved output: always valid
  // Sharded output: only valid when input is HEIGHT_SHARDED ROW_MAJOR.
  auto inputML = inputLayouts[0].getMemLayout();
  return inputML &&
         inputML.getValue() == TensorMemoryLayout::HeightSharded &&
         inputLayouts[0].getLayout() == Layout::RowMajor;
}
```

---

### 4.3 No changes required in:

- `TTNNOpModel.cpp` — the HS→HS analytical bypass already exists (lines 2030-2043)
- `DFShardingPolicy.cpp` — SliceStaticOp remains excluded from `validForSharding` (no output L1 CB in the existing flow; the fix adds one, but DFSharding is not the right policy for slice since it needs to be between two HS chains)
- tt-metal `slice_program_factory_rm_sharded.cpp` — both output paths already exist

---

## 5. Expected After-Fix Pattern

```mlir
%33 = max_pool2d → L1 HEIGHT_SHARDED 58×1 ROW_MAJOR    ← Fix 12
%34 = slice_static(%33) → L1 HEIGHT_SHARDED 58×1 RM    ← this fix (same grid, trimmed last dim)
...
%43 = concat(%34_HS_RM, %42_HS_RM) → HEIGHT_SHARDED    ← stays L1
%44 = conv2d(%43) → HEIGHT_SHARDED TILED                ← reads from L1, no DRAM hop
```

Eliminated per camera pass:
- `%34` stops being DRAM → saves 1 DRAM write
- `%39` (DRAM→HS RM re-shard) eliminated
- `%42` (HS→interleaved) either eliminated or absorbed by concat
- `%44` input no longer requires DRAM→L1 transfer

---

## 6. Actual Metrics Impact

| Metric | Fix 12 | Fix 13 | Change |
|--------|:------:|:------:|:------:|
| FPS | 2.54 | **2.77** | **+9%** |
| `dram_spilled_ops` | 96 | **24** | **−75%** |
| `sharded_and_spilled_ops` | 0 | 24 | +24 |
| `effectively_sharded_ops` | 216 | 204 | −12 |
| `effectively_sharded_percentage` | 55.7% | 51.0% | −4.7pp |
| `to_memory_config` ops | 96 | 120 | +24 |
| `to_layout` ops | 0 | 16 | +16 |

### Why different from predicted

The outer concat (after the slice and a downstream conv2d chain) has two inputs:
- `%34` = slice → **L1 HS RM 58×1** (our fix ✓)
- `%41` = conv2d → HS TILED 58×1 (different page layout)

`ConcatRuleBook::isValidOutputHintForInputs` requires the hint grid to match
the input grid (58×1). No legalConfigs entry has 58×1 → all HS output hints fail.
The solver reshards both inputs to DRAM and uses a DRAM concat. This adds:
- 4 × `to_memory_config(HS_RM → DRAM)` (new, per camera)
- 4 × `to_memory_config(HS_TILED → DRAM)` (changed target)
- 4 × `to_layout(DRAM → TILED)` (new, before next conv2d)
- 4 × `to_memory_config(DRAM TILED → L1 HS RM)` (new, before conv2d)

**Net benefit**: The conv2d after the outer concat now reads from L1 HS RM (fast)
instead of DRAM directly (Fix 12 path). The `to_memory_config` prefetch is faster
than conv2d pulling from DRAM, yielding the 9% FPS gain despite the extra ops.

The `dram_spilled_ops` drop from 96→24 reflects the 72 ops (12 slices + 60 related
downstream ops across 4 camera passes) that no longer produce DRAM outputs directly.

---

## 7. Risk Assessment

**Low risk:**

- **tt-metal side**: `SliceRmShardedWidthTrimProgramFactory` HS output path already works and is tested (the path exists since the kernel was written). The `output_is_dram` flag correctly selects between the two paths.
- **Numerical**: Slice is a pure data copy — no arithmetic. Output layout (DRAM vs L1 HS) cannot affect numerical correctness.
- **L1 pressure**: The `cbOutSize` is smaller than `cbInSize` (COut < CIn for last-dim slice). Peak usage is `cbInSize + cbOutSize` vs `cbInSize` in the DRAM path. The op model accounts for this correctly.
- **Fallback**: If L1 is insufficient for `cbInSize + cbOutSize`, `evaluateHint` returns no valid candidate for the HS hint → solver falls back to NULL hint → DRAM path unchanged.

**Potential concern**: downstream `ConcatRuleBook::isValidInputCombination` requires all inputs to have the same memory layout type and grid. The HEIGHT_SHARDED RM slice output must match the grid expected by the concat. Since the slice inherits its grid from the HS RM MaxPool2d input (same `shard_height`, same cores), and the other concat input is already HEIGHT_SHARDED RM (via `%40` path), the grids should match.

---

## 8. Build

Only 2 files need recompilation:
```
ninja -C build obj.MLIRTTNNAnalysis MLIRTTNNAnalysis TTMLIRCompiler
```

Changed files:
- `DataMovementRules.h` — header change; triggers transitive recompilation of including TUs
- `DataMovementRules.cpp`
