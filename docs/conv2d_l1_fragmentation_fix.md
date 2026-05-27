# Block A BEV — Conv2d L1 Fragmentation Fix

## Overview

`test_opt_sweep[enable_program_cache-opt_level_2-block_A]` and
`test_opt_sweep[disable_program_cache-opt_level_2-block_A]` failed for
`block_A_deformed_backbone` with an L1 OOM during compilation at opt_level_2.

All fixes are in `third_party/tt-mlir` only — no changes were made to the
tt-forge-onnx front-end or any MLA passes.

---

## Failure — L1 OOM at opt_level_2

### Symptom

```
Not enough space to allocate 37748736 B L1 buffer across 64 banks,
where each bank needs to store 589824 B, but bank size is 1329888 B
(allocated: 606208 B, free: 723680 B, largest free block: 420576 B)
```

The error occurs during TTNN op execution on the device. The L1 spill
management simulation (`SumL1MemoryTracker`) believed there was enough room to
keep the conv2d output in L1, but at runtime only 420,576 bytes of contiguous
space was available — not the 589,824 bytes/core the kernel needed.

### Root Cause

The root cause is a simulation-vs-runtime mismatch in how `L1SpillManagement`
tracks L1 occupancy for `ToLayoutOp` (workaround-inserted layout conversion ops).

**The design decision that causes the gap:**

By design, `ToLayoutOp` outputs are NOT added to `liveValues` in the simulation
(see `L1SpillManagement.cpp` lines 1013–1041). The comment explains: they are
"short-lived L1 tenants" that will be consumed before the next eviction decision.
`ensureFitsL1` is still called for them to make room, but they are never tracked
by `memoryTracker`.

**The fragmentation pattern at runtime (Block A, 4 camera inputs):**

Block A contains a 1×1 conv2d on HEIGHT_SHARDED inputs. Prior to the conv2d,
three `ToLayoutOp` outputs are allocated in L1 (one per camera at 303,104 bytes/core
each), then the middle one is consumed and freed. This creates a fragmented L1
heap:

```
L1 bank size: 1,329,888 bytes

Top of L1 (base of top-down allocator):
  T1 = ToLayoutOp output:  [1,026,784 – 1,329,888)  303,104 B  ← still live
  T2 = ToLayoutOp output:  [  723,680 – 1,026,784)  303,104 B  ← freed (hole)
  T3 = ToLayoutOp output:  [  420,576 –   723,680)  303,104 B  ← still live
  free:                    [        0 –   420,576)  420,576 B
  allocated: 606,208 B     largest free block: 420,576 B
```

When the conv2d's output (589,824 bytes/core) is allocated, the largest
contiguous free block is only 420,576 bytes → OOM.

**Why the simulation missed it:**

The simulation's `getOccupiedL1()` returns 0 when `ensureFitsL1` is called for
the conv2d output, because:

1. `processDeadTensors(pos)` frees the conv2d's input (also a `ToLayoutOp` result)
   before `ensureFitsL1` is called.
2. The three `ToLayoutOp` outputs (T1, T2, T3 above) were NEVER added to
   `memoryTracker` — the simulation's free list always shows a full 1,329,888 B
   budget.

So the simulation concludes "589,824 bytes fits (budget = 1,325,652 bytes, occupied = 0)"
and keeps the conv2d output in L1. At runtime, however, 606,208 bytes are already
occupied by ToLayoutOp outputs, causing the OOM.

---

## Fix — Large-Tensor Fragmentation Guard in `ensureFitsL1`

**File:** `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`

Added a guard inside `ensureFitsL1` that unconditionally sends any single tensor
exceeding 40% of `l1BudgetPerCore` to DRAM. The guard fires AFTER the dead-zone
check and BEFORE the `wouldCBsOverlapTensors` check:

```cpp
static constexpr double kMaxSingleTensorFraction = 0.40;
if (l1Size > static_cast<uint64_t>(kMaxSingleTensorFraction *
                                   static_cast<double>(l1BudgetPerCore))) {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    LARGE_TENSOR_FRAG: l1Size={0} > 40%% budget={1}, "
               "forcing DRAM to prevent fragmentation OOM",
               l1Size, l1BudgetPerCore);
  llvm::SmallVector<Value> toEvict;
  for (Value operand : op->getOperands()) {
    if (liveValues.count(operand)) {
      toEvict.push_back(operand);
    }
  }
  for (Value victim : toEvict) {
    evictValue(victim, pos, data);
  }
  demoteToDram(op);
  evictForDramCBGrowth(op, pos, data);
  return 0;
}
```

### Threshold Rationale

On WH N150:
- `bank_size` = 1,329,888 bytes
- `usableL1Size` = 1,395,424 bytes
- `l1BudgetPerCore` = 0.95 × usableL1 = 1,325,652 bytes
- **40% threshold** = 530,261 bytes

| Tensor | Size/core | Fraction | Guard fires? |
|--------|-----------|----------|--------------|
| Block A conv2d output (`%308`) | 589,824 B | 44.5% | **YES → DRAM** |
| ToLayoutOp outputs (T1, T2, T3) | 303,104 B | 22.9% | No → L1 |

The threshold is chosen to reject the 589,824-byte conv2d output (which is
nearly half the budget and risks fragmentation OOM) while leaving smaller tensors
(≤303,104 B, 22.9% of budget) in L1 where they benefit from the fast path.

### Why This Is Conservative but Correct

A tensor at 40–50% of the L1 budget is inherently risky:
- Even a single other untracked allocation (one `ToLayoutOp` output = 22.9%) can
  halve the effective contiguous space.
- If fragmented, no amount of Belady eviction can help — the simulation cannot
  see or evict untracked `ToLayoutOp` outputs.
- The conv2d's output is immediately followed by a `to_memory_config → DRAM` op
  in Block A; there is no downstream L1 consumer to benefit from keeping it in L1.

Sending such tensors to DRAM is correct: the L1 bandwidth savings are minimal
when the tensor won't stay resident, and the risk of fragmentation-induced OOM is
real and undetectable by the simulation.

---

## Affected Op in Block A

The op that OOMs without the fix:

| Attribute | Value |
|-----------|-------|
| Op | `%308 = ttnn.conv2d(...)` (1×1 depthwise-style conv on camera features) |
| Layout | `#ttnn_layout88` — HEIGHT_SHARDED, L1 |
| Output size | 589,824 bytes/core (44.5% of l1BudgetPerCore) |
| Follows | Three `ToLayoutOp` outputs of 303,104 B/core each |
| Consumers | `to_memory_config → DRAM` immediately after |

The guard fires 4 times in Block A (once per camera input) and moves all four
instances of `#ttnn_layout88` from L1 to DRAM.

---

## Build Process

**Critical:** Python uses the INSTALLED copy of the compiler, not the build directory copy.

```bash
cd /path/to/third_party/tt-mlir
cmake --build build          # compiles → build/lib/libTTMLIRCompiler.so
cmake --install build        # installs → build/install/lib/libTTMLIRCompiler.so  ← Python uses this
```

Forgetting `cmake --install build` causes the fix to appear to have no effect
(same OOM as before), even if the source was correctly modified and compiled.

---

## Test Results

| Test | Before Fix | After Fix |
|------|------------|-----------|
| `test_opt_sweep[opt_level_1-block_A]` | PASS | PASS |
| `test_opt_sweep[enable_program_cache-opt_level_2-block_A]` | **FAIL (L1 OOM)** | **PASS** |
| `test_opt_sweep[disable_program_cache-opt_level_2-block_A]` | **FAIL (L1 OOM)** | **PASS** |
| `test_opt_sweep[enable_program_cache-opt_level_2-block_C]` | PASS | PASS (no regression) |
| `test_opt_sweep[enable_program_cache-opt_level_2-block_E]` | PASS | PASS (no regression) |

---

## File Summary

| File | Change |
|------|--------|
| `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp` | Added large-tensor fragmentation guard in `ensureFitsL1` — unconditionally demotes tensors > 40% of `l1BudgetPerCore` to DRAM to prevent simulation-invisible fragmentation OOM |
