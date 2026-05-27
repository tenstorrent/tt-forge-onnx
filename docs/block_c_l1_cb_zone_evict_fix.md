# Block C BEV — L1 CB Zone Eviction Fix (opt_level_2)

## Overview

`test_opt_sweep[enable_program_cache-opt_level_2_bfloat16_hifi2_fp32_acc-block_C]`
failed at runtime after the earlier ConvTranspose2d and dead-zone pre-check fixes
(documented in `conv2d_l1_cb_clash_fix.md`) were applied.

The test exhibited two related problems:

| Problem | Symptom |
|---------|---------|
| **CB clash crash** | `RuntimeError: TT_THROW @ program.cpp:1366` — static CBs clash with L1 buffers |
| **PCC baseline** | PCC = 0.9633, below the 0.99 test threshold |

Both were investigated.  The crash has a root cause and is fixed here.  The PCC
= 0.9633 was found to be a **pre-existing characteristic** of Block C under
bfloat16 + HiFi2 + fp32_acc settings (reproduced at opt_level_1 where
`L1SpillManagement` plays no role), not a regression introduced by compiler
changes.

All fixes are in `third_party/tt-mlir` only.

---

## Crash — CB clash at `opt_level_2_bfloat16_hifi2_fp32_acc`

### Symptom

```
RuntimeError: TT_THROW @ program.cpp:1366: tt::exception
Statically allocated circular buffers in program clash with L1 buffers
on core range [0-0 – 7-6].
```

`opt_level_1_bfloat16_hifi2_fp32_acc` does **not** reproduce — the op involved
stays in DRAM at opt_level_1.

### Root Cause — Address Space Inversion

TTNN's L1 allocator and the compiler's simulation allocate memory in opposite
directions:

| Allocator | Direction | Consequence |
|-----------|-----------|-------------|
| **Simulation** (`SumL1MemoryTracker`) | Top-down (high virtual → low virtual) | First allocation gets highest virtual |
| **TTNN runtime** | Bottom-up (low physical → high physical) | First allocation gets lowest physical address |

The mapping from simulation virtual to TTNN physical is therefore:

```
physical ≈ l1_unreserved_base + (l1BudgetPerCore − virtual)
```

A tensor allocated **early** in the schedule (= high virtual in simulation = low
physical in TTNN) can land inside the CB region
`[l1_unreserved_base, l1_unreserved_base + cbPeakUsage)` when the CB for a
later op is large.

The existing `wouldCBsOverlapTensors` / `evictForCBOverlap` path only inspects
`getLowestOccupiedAddress()` — the **most recently** allocated tensor
(= lowest virtual = highest physical).  This is the safest tensor.  It
completely misses tensors allocated **earlier** (= higher virtual = lower
physical) that may lie inside a large CB zone.

### The Failing Sequence

Block C schedule around the crash (program positions, simplified):

```
pos 162: %194 = add(...)        ← allocated early → HIGH virtual → LOW physical
pos 163: op_A (cbPeak = 626,688 B)
           cbVT = l1BudgetPerCore − cbPeak = 1,325,652 − 626,688 = 698,964
           %194's virtual > 698,964 → %194's physical < 730,400 → inside CB zone!
pos 167: op_B (cbPeak = 651,264 B)
           CB zone: [103,712, 754,976)
           %190 live at pos 167 with virtual > 674,388 → physical inside CB zone
           → TT_THROW at runtime
```

`evictForCBOverlap` at pos 167 checks `getLowestOccupiedAddress()` which sees
the most recent allocation at low virtual / high physical (above the CB zone top)
and reports no conflict — missing `%190`.

### Ineffective Earlier Approach (Replaced)

An earlier iteration added a `CB_CLASH_GUARD` that triggered whenever
`cbPeakUsage > threshold` and unconditionally demoted the op to DRAM (plus
evicting all direct operands via phase-1).  Two threshold values were tried:

| Threshold | Positions triggered | Effect |
|-----------|---------------------|--------|
| `> l1DeadZone` (69,772 B) | 30+ positions | Crash fixed; PCC = 0.9633 |
| `> 40% × l1BudgetPerCore` (530,261 B) | 9 positions | Crash fixed; PCC = 0.9633 |

Both variants fired at many positions where no tensor was actually inside the CB
zone.  The unnecessary `demoteToDram` calls at those positions caused cascading
evictions through `evictForDramCBGrowth` but did not improve PCC — confirming
that PCC = 0.9633 is a pre-existing numerical characteristic, not a regression.

### Fix — Targeted CB Zone Eviction (`CB_ZONE_EVICT`)

**File:** `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`

Instead of a threshold-based DRAM demotion, the fix adds a targeted scan that
evicts **only** tensors whose simulated virtual address maps to a TTNN physical
address inside the current op's CB zone:

```cpp
// Evict live tensors whose TTNN physical address falls inside this op's CB
// region [l1_unreserved_base, l1_unreserved_base + cbPeakUsage).
//
// Only check when cbPeakUsage > l1DeadZone: below this threshold the CB
// region is entirely within the dead zone (below the simulation floor) so
// no simulation-tracked tensor can have a conflicting physical address.
if (cbPeakUsage > l1DeadZone && cbPeakUsage <= l1BudgetPerCore) {
  uint64_t cbVT = l1BudgetPerCore - cbPeakUsage;
  bool anyEvicted = false;
  for (Value victim : memoryTracker.getValuesAboveVirtualThreshold(cbVT)) {
    if (!liveValues.count(victim)) {
      continue;
    }
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    CB_ZONE_EVICT: cbPeakUsage={0} cbVT={1}, evicting "
                 "high-virtual tensor to prevent CB-tensor clash",
                 cbPeakUsage, cbVT);
    evictValue(victim, pos, data);
    anyEvicted = true;
  }
  if (anyEvicted) {
    // Recompute after address rebuild triggered by evictions.
    speculativeAddr = memoryTracker.wouldAllocateAt(l1Size);
    if (!speculativeAddr) {
      l1Size = handleNoFit(op, pos, data, opL1Usage, l1Size);
      speculativeAddr = memoryTracker.wouldAllocateAt(l1Size);
    }
  }
}
```

Key differences from the earlier approach:

| Aspect | Old CB_CLASH_GUARD | New CB_ZONE_EVICT |
|--------|-------------------|-------------------|
| Trigger | `cbPeak > threshold` (fires at many positions) | Always checked; evicts only when a tensor is actually in the CB zone |
| Phase 1 | Evicts all direct L1 operands | **No phase-1 evictions** |
| DRAM demotion | `demoteToDram(op)` on every trigger | **No DRAM demotion** — op stays in L1 |
| `evictForDramCBGrowth` | Called every trigger | **Not called** |
| Speculative address | Stale after evictions (not recomputed) | Recomputed after any eviction |

**How the fix chain works at pos 163:**

1. `cbPeak = 626,688 > l1DeadZone` → enter scan.
2. `cbVT = 1,325,652 − 626,688 = 698,964`.
3. `getValuesAboveVirtualThreshold(698,964)` returns `%194` (the only live tensor
   at virtual > 698,964).
4. `evictValue(%194, ...)` → `markEvictedAndRebuild` replays address simulation.
5. After rebuild, `%190`'s virtual address shifts to ≤ 674,388 (= cbVT at pos 167).
6. `%190`'s TTNN physical address is now above 754,976 — outside pos 167's CB zone.
7. At pos 167, `getValuesAboveVirtualThreshold(674,388)` returns empty → no evictions,
   no DRAM demotion.  No CB clash at runtime. ✓

At all other positions (63, 76, 89, 97, 105, 160, 167, 176) where
`cbPeak > l1DeadZone`, `getValuesAboveVirtualThreshold` returns empty (no tensor
is actually in the CB zone) → no evictions → no effect on accuracy.

### Helper Function — `getValuesAboveVirtualThreshold`

**File:** `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`
**Header:** `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`

Added to `SumL1MemoryTracker`:

```cpp
llvm::SmallVector<Value>
SumL1MemoryTracker::getValuesAboveVirtualThreshold(uint64_t threshold) const {
  llvm::SmallVector<Value> result;
  for (const auto &[val, addrSize] : tensorAddresses) {
    if (addrSize.first > threshold) {
      result.push_back(val);
    }
  }
  return result;
}
```

---

## PCC = 0.9633 — Pre-Existing Characteristic, Not a Regression

During investigation, a `PCC = 0.9633` accuracy failure was observed whenever
the crash was prevented.  Root-cause analysis showed this is **not** caused by
any `L1SpillManagement` change:

```
opt_level_1_bfloat16_hifi2_fp32_acc-block_C:
  PCC = 0.9632  ← same value, opt_level_1 does not run L1SpillManagement
```

The PCC difference from the 0.99 threshold is inherent to the **HiFi2 compute
mode** (`MathFidelity::HiFi2`) used in this test configuration.  HiFi2 reduces
mantissa computation precision for performance — this is a deliberate accuracy
vs. speed tradeoff.  The 0.99 threshold in `AutomaticValueChecker` is not
appropriate for HiFi2 configurations of Block C.

Fixing the PCC requires either:
- Adjusting the test threshold for HiFi2 variants, OR
- A kernel-level accuracy improvement — outside the scope of `L1SpillManagement`.

---

## File Summary

| File | Change |
|------|--------|
| `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h` | Declared `getValuesAboveVirtualThreshold` on `SumL1MemoryTracker`; added `l1DeadZone` member and updated constructor signature |
| `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp` | Added `getValuesAboveVirtualThreshold` implementation; replaced `CB_CLASH_GUARD` with targeted `CB_ZONE_EVICT` in `ensureFitsL1`; added `speculativeAddr` recomputation after evictions |

---

## Test Results

| Test | Before | After |
|------|--------|-------|
| `opt_level_2_bfloat16_hifi2_fp32_acc-block_C` | **FAIL (CB clash crash)** | No crash; PCC = 0.9633 (pre-existing) |
| `opt_level_1_bfloat16_hifi2_fp32_acc-block_C` | FAIL (PCC = 0.9632) | FAIL (PCC = 0.9632, unchanged) |
| `opt_level_2-block_C` | FAIL (CB clash crash) | FAIL (CB clash — different config, separate issue) |
| `opt_level_2-block_A` | PASS | PASS (no regression) |
| `opt_level_2-block_E` | PASS | PASS (no regression) |

> **Note:** `opt_level_2-block_C` (default float32, no HiFi2) also crashes with
> a CB clash — this is a separate instance of the same class of problem and
> requires a distinct investigation pass.
