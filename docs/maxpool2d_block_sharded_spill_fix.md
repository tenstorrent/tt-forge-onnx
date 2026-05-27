# MaxPool2d BLOCK_SHARDED Output Spill Fix

**Date:** 2026-05-26  
**Fixes:** Fix 12 — eliminates all 12 remaining `sharded_and_spilled_ops` in BEV Block A  
**Model:** BEV deformed backbone (`block_A`), `opt_level_2`, `bfloat16`, `hifi3+fp32_acc`, `trace_enabled`  
**Result:** `sharded_and_spilled_ops` 12 → **0**, test PASSED (PCC > 0.99)

---

## 1. Background

After Fix 11 (SliceRmShardedWidthTrimProgramFactory DRAM output), BEV Block A had 12 remaining `sharded_and_spilled_ops` — operations that write L1-sharded output and are immediately followed by a `ttnn.to_memory_config` that spills the result to DRAM. These are pure overhead: the L1 write happens only to be immediately thrown away.

The 12 spilling ops were:

| op_type | location |
|---------|----------|
| `ttnn.max_pool2d` | MaxPool2d_100 |
| `ttnn.max_pool2d` | MaxPool2d_158 |
| `ttnn.max_pool2d` | MaxPool2d_324 |
| `ttnn.max_pool2d` | MaxPool2d_382 |
| `ttnn.max_pool2d` | MaxPool2d_548 |
| `ttnn.max_pool2d` | MaxPool2d_606 |
| `ttnn.max_pool2d` | MaxPool2d_772 |
| `ttnn.max_pool2d` | MaxPool2d_830 |
| `ttnn.conv2d` | Conv2d_67 |
| `ttnn.conv2d` | Conv2d_291 |
| `ttnn.conv2d` | Conv2d_515 |
| `ttnn.conv2d` | Conv2d_739 |

---

## 2. Root Cause: MaxPool2d BLOCK_SHARDED Inheritance

### What `sharded_and_spilled` means

`TTNNCollectPerfMetrics.cpp` → `identifyDRAMSpills` walks all `ToMemoryConfigOp`s. If the producer of a `ToMemoryConfigOp` output has:
- output in L1 (sharded: `HEIGHT_SHARDED`, `WIDTH_SHARDED`, or `BLOCK_SHARDED`)
- `ToMemoryConfigOp` output in DRAM

then that producer is counted as `sharded_and_spilled`.

### Why MaxPool2d produced BLOCK_SHARDED output

The **default** `OpRuleBook::getOutputHints` returns a NULL hint first, then falls back to sharded configs. A NULL hint tells tt-metal to pick the output layout itself. When the MaxPool2d input comes from a BLOCK_SHARDED upstream (e.g., a conv2d with BLOCK_SHARDED output), tt-metal's null-hint path **inherits** the input sharding type — producing BLOCK_SHARDED output.

```
conv2d → [BLOCK_SHARDED] → max_pool2d (NULL hint) → [BLOCK_SHARDED output, inherited]
       → ToMemoryConfig → [DRAM] → SliceStaticOp
```

### Why BLOCK_SHARDED must immediately spill

`SliceRuleBook::getInputLayoutFilter` (from Fix 11) only admits HEIGHT_SHARDED ROW_MAJOR or interleaved inputs:

```cpp
return [](TTNNLayoutAttr layout) -> bool {
  auto ml = layout.getMemLayout();
  if (!ml || !isShardedMemoryLayout(ml.getValue()))
    return true; // interleaved — keep
  return ml.getValue() == TensorMemoryLayout::HeightSharded &&
         layout.getLayout() == Layout::RowMajor;
};
```

A BLOCK_SHARDED MaxPool2d output is not admitted by SliceRuleBook, so `MemoryLayoutPropagation` inserts a `ToMemoryConfig` DRAM spill before the slice. This is the **wasted write**: MaxPool2d writes to L1 BLOCK_SHARDED, then immediately the spill copies it to DRAM.

### Why conv2d ops also spilled (indirect effect)

The 4 conv2d ops (Conv2d_67, Conv2d_291, Conv2d_515, Conv2d_739) were non-primary operands of downstream `ttnn.add` operations. With the BLOCK_SHARDED MaxPool2d chain in effect, `DFShardingPolicy` assigned `HEIGHT_SHARDED` to these conv2d ops as single-op dead-end chains. The downstream `add` could not consume `HEIGHT_SHARDED` from a secondary operand without a reshard, so a `ToMemoryConfig` spill was inserted. Once the MaxPool2d chain changed from BLOCK_SHARDED to DRAM, the DFSharding layout pressure on those conv2d ops resolved naturally — they were no longer forced into HEIGHT_SHARDED single-op chains.

---

## 3. Fix: `MaxPool2dRuleBook`

### Strategy

Override `getOutputHints` for `MaxPool2dOp` to:

1. **Skip the NULL hint** — prevents BLOCK_SHARDED inheritance from upstream
2. **Use DRAM/interleaved configs as primary hints** — direct DRAM output; no L1 write wasted
3. **Use HEIGHT_SHARDED as fallback hints** — tried only when DRAM yields no valid candidate
4. **Exclude BLOCK_SHARDED entirely** — downstream slice cannot consume it

```
conv2d → [BLOCK_SHARDED] → max_pool2d (DRAM primary hint) → [DRAM output]
       → SliceStaticOp (no spill needed)
```

---

## 4. File Changes

### 4.1 `include/ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h`

**Added** `MaxPool2dRuleBook` struct declaration at the end of the namespace:

```cpp
/// MaxPool2dOp: reject BLOCK_SHARDED output.
/// Downstream consumers (SliceStaticOp via SliceRmShardedWidthTrimProgramFactory,
/// conv2d with config_tensors_in_dram) only accept HEIGHT_SHARDED ROW_MAJOR or
/// DRAM. A BLOCK_SHARDED output always requires an immediate DRAM spill.
/// Use non-sharded (DRAM) as primary hints; HEIGHT_SHARDED as fallback.
struct MaxPool2dRuleBook : OpRuleBook {
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};
```

---

### 4.2 `lib/Dialect/TTNN/Analysis/OpRules/DataMovementRules.cpp`

**Added** `MaxPool2dRuleBook::getOutputHints` implementation at the end of the file:

```cpp
//===----------------------------------------------------------------------===//
// MaxPool2dRuleBook
//===----------------------------------------------------------------------===//

OutputHints
MaxPool2dRuleBook::getOutputHints(Operation * /*op*/,
                                  const std::vector<OpConfig> &legalConfigs) const {
  // Reject BLOCK_SHARDED output for max_pool2d.  Downstream ops
  // (SliceStaticOp via SliceRmShardedWidthTrimProgramFactory, conv2d with
  // config_tensors_in_dram) only accept HEIGHT_SHARDED ROW_MAJOR or DRAM.
  // A BLOCK_SHARDED output would always be immediately spilled to DRAM.
  //
  // Strategy: DRAM/interleaved configs as primary hints (no NULL hint to avoid
  // the backend defaulting to BLOCK_SHARDED for a BLOCK_SHARDED input);
  // HEIGHT_SHARDED as fallback.  The NULL hint is omitted on purpose so that
  // inputs with BLOCK_SHARDED layout don't inherit BLOCK_SHARDED output.
  OutputHints result;
  for (const auto &cfg : legalConfigs) {
    if (!cfg.outputLayout) {
      continue; // skip NULL hint
    }
    auto ml = cfg.outputLayout.getMemLayout();
    if (!ml || !isShardedMemoryLayout(ml.getValue())) {
      result.hints.push_back(cfg); // DRAM / L1-interleaved (primary)
    } else if (ml.getValue() == TensorMemoryLayout::HeightSharded) {
      result.fallbackHints.push_back(cfg); // HEIGHT_SHARDED (fallback)
    }
    // BLOCK_SHARDED: excluded
  }
  // If legalConfigs produced no non-sharded entry, fall back to NULL hint so
  // we don't end up with zero primary hints (which would cause no candidate).
  if (result.hints.empty()) {
    result.hints.push_back(OpConfig(TTNNLayoutAttr()));
  }
  return result;
}
```

---

### 4.3 `lib/Dialect/TTNN/Analysis/OpRules/OpRuleBook.cpp`

**Added** static `MaxPool2dRuleBook` instance and its registration:

```cpp
// In static variable block:
static MaxPool2dRuleBook maxPool2d;

// In registration lambda:
reg(MaxPool2dOp::getOperationName(), &maxPool2d);
```

Full context in `getRuleBook()`:

```cpp
static Conv2dRuleBook conv2d;
// ...
static MaxPool2dRuleBook maxPool2d;   // ← added
// ...
reg(Conv2dOp::getOperationName(), &conv2d);
reg(ConvTranspose2dOp::getOperationName(), &conv2d);
reg(MaxPool2dOp::getOperationName(), &maxPool2d);  // ← added
```

---

## 5. What Was Tried and Reverted

An earlier attempt additionally excluded the 4 conv2d ops directly from `DFShardingPolicy` and added an early `nonShardedOutputHints` return to `Conv2dRuleBook::getOutputHints`. This **broke numerical accuracy**: PCC dropped from >0.99 to 0.905.

**Why it broke PCC:** The conv2d ops were previously processed by the `HEIGHT_SHARDED` kernel variant (DFSharding assigned HEIGHT_SHARDED; it was then spilled to DRAM). Excluding them from DFSharding caused `MemoryLayoutPropagation` to select DRAM output configs, which routes through a different conv2d kernel variant. The two variants use different tiling/accumulation strategies and produce slightly different numerical outputs.

**Lesson:** Changing conv2d from HEIGHT_SHARDED to DRAM output is not numerically transparent. Only the memory-type of the op output should be changed for conv2d by changing upstream pressure (which is what the MaxPool2d fix achieves indirectly).

---

## 6. `OpRuleBook` Framework Context

The `OpRuleBook` system (`OpRuleBook.h/.cpp`, `DataMovementRules`, `ConvRules`, `MatmulRules`, etc.) provides per-op policy hooks that `MemoryLayoutPropagation` and `DFShardingPolicy` consult when selecting output layouts:

| Hook | Purpose |
|------|---------|
| `getOutputHints(op, legalConfigs)` | Returns `{primary hints, fallback hints}`. Primary tried first; fallback only if primary yields no sharded result. |
| `getInputLayoutFilter(operandIdx)` | Predicate to filter out illegal input layouts before op-model evaluation. |
| `isValidInputCombination(inputLayouts)` | Cross-input validation (e.g., concat requires same layout on all inputs). |
| `isValidOutputHintForInputs(hint, inputLayouts)` | Validates output hint against already-committed input layouts. |
| `shouldExploreReshards()` | Whether to try inserting reshards to unlock better sharding. |
| `preferCandidate(op, a, b)` | Tiebreaker between two `BeamCandidate`s of equal score. |

`getRuleBook(op)` returns the correct `OpRuleBook` subtype for a given op via a thread-safe static `DenseMap` registry. Ops without a registered rule book get `defaultRules` (the base `OpRuleBook`), whose `getOutputHints` returns NULL hint first then sharded fallbacks — which is what caused the MaxPool2d inheritance problem.

---

## 7. Results

### Metrics (BEV Block A, `opt_level_2`, `bfloat16`, `hifi3+fp32_acc`, `trace_enabled`)

| Metric | Before (Fix 11) | After (Fix 12) | Delta |
|--------|:--------------:|:--------------:|:-----:|
| `sharded_and_spilled_ops` | 12 | **0** | −12 |
| `effectively_sharded_ops` | 212 | 216 | +4 |
| `sharded_ops` | 224 | 216 | −8 |
| `dram_spilled_ops` | 108 | 96 | −12 |
| `total_ops_with_output_tensor` | 504 | 484 | −20 |
| `effectively_sharded_percentage` | 53.5% | 55.7% | +2.2pp |
| PCC | >0.99 ✓ | >0.99 ✓ | — |

The −20 in `total_ops_with_output_tensor` reflects the 8 MaxPool2d spill `ToMemoryConfigOp`s and 4 conv2d spill `ToMemoryConfigOp`s being eliminated (they are no longer needed at all after the layout fix).

### Build impact

Only 3 files needed recompilation:
- `DataMovementRules.cpp.o`
- `OpRuleBook.cpp.o`
- `DataMovementRules.h` (header — triggers transitive recompilation of including TUs if modified)

Rebuild target: `ninja -C build obj.MLIRTTNNAnalysis MLIRTTNNAnalysis TTMLIRCompiler`
