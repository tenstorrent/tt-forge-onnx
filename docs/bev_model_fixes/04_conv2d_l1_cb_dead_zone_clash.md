# Block C — Conv2d L1 Circular Buffer Dead Zone Clash

## 1. Affected Test Cases

- `test_opt_sweep[enable_program_cache-opt_level_2-block_C]` — FAIL (after Fix 3 was applied)
- `test_opt_sweep[disable_program_cache-opt_level_2-block_C]` — FAIL

## 2. Failure

```
TT_THROW @ program.cpp:1366:
Statically allocated circular buffers in program 60394 clash with L1 buffers
on core range [0-0 - 7-6].
L1 buffer allocated at 171520 and static circular buffer region ends at 177440.
```

Call stack: `conv2d_L1 -> prim::matmul -> validate_circular_buffer_region -> TT_THROW`

## 3. Failure Reason — The Dead Zone

The `L1SpillManagement` simulation works in a virtual address space `[0, l1BudgetPerCore]`. Virtual address 0 maps to real address `l1_size - l1BudgetPerCore`.

```
Hardware (WH N150):
  l1_size            = 1,499,136 bytes
  l1_unreserved_base = 103,712 bytes   <- where CBs physically start
  usableL1Size       = 1,395,424 bytes (l1_size - l1_unreserved_base)
  l1BudgetPerCore    = 0.95 x 1,395,424 = 1,325,652 bytes
  Simulation floor   = 1,499,136 - 1,325,652 = 173,484 (real addr)

Dead zone = [103,712, 173,484) = 69,772 bytes
```

The simulation tracks tensors in virtual space `[0, 1,325,652]`. Circular buffers physically start at `l1_unreserved_base = 103,712`, below the simulation's real floor at `173,484`. The **dead zone** is the region `[103,712, 173,484)` — it is:
- Below the simulation floor (invisible to the tracker)
- Above `l1_unreserved_base` (physically usable for CBs)

The failing Conv2d had `cbPeakUsage = 73,728 bytes > l1DeadZone = 69,772 bytes`. This means the CB physically extends `73,728 - 69,772 = 3,956 bytes` past the simulation floor into the dead zone. At runtime, if any tensor is placed at a real address between `173,484 - 3,956 = 169,528` and `173,484`, it collides with the CB region.

At runtime, actual L1 usage exceeded the budget by 1,964 bytes (due to lazy deallocation and program cache overhead), placing a tensor at real address `171,520` — exactly inside the CB region ending at `177,440`. The simulation's `wouldCBsOverlapTensors` check operates in virtual space and reported `lowestExisting = 289,364` virtual (= `462,848` real) — far from the actual collision address. So no demotion was triggered and the op stayed in L1, causing the `TT_THROW` at runtime.

## 4. Fix Implementation Details

Added `l1DeadZone` field and a dead zone pre-check in `ensureFitsL1`:

**Step 1 — Add `usableL1Size` to constructor and compute `l1DeadZone`**

The dead zone is the gap between `usableL1Size` (the full physical L1 minus system reservations) and `l1BudgetPerCore` (the fraction of usable L1 we allow the simulation to assign). This gap is the physical space that the simulation assumes is always free but which CBs can expand into.

```cpp
l1DeadZone = (usableL1Size > l1BudgetPerCore)
             ? usableL1Size - l1BudgetPerCore : 0;
```

**Step 2 — Pass `usableL1Size` from `GreedyL1SpillManagement.cpp`**

The chip descriptor already exposes `getUsableL1Size()`. This value is threaded from the call site into the `L1SpillManagement` constructor:

```cpp
L1SpillManagement<SumL1MemoryTracker> spill(
    func, deviceGrid, l1BudgetPerCore, chipDesc.getUsableL1Size(), std::move(observer));
```

**Step 3 — Pre-check in `ensureFitsL1`** (before `wouldCBsOverlapTensors`)

The pre-check fires before the existing virtual-space overlap analysis. If the CB size exceeds the dead zone size, the CB will physically overlap the dead zone regardless of where tensors are placed — demotion to DRAM is the only safe choice:

```cpp
if (l1DeadZone > 0 && cbPeakUsage > l1DeadZone) {
    // CB extends into the dead zone -- any runtime overflow puts a tensor there.
    // Evict live L1 inputs first (homogeneous inputs for downstream validation).
    for (Value operand : op->getOperands())
        if (liveValues.count(operand)) evictValue(operand, pos, data);
    demoteToDram(op);
    evictForDramCBGrowth(op, pos, data);
    return 0;
}
```

## 5. Files Changed with Diffs

**`include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`** (tt-mlir)
```diff
-  L1SpillManagement(func::FuncOp func, ttcore::GridAttr deviceGrid,
-                    uint64_t l1BudgetPerCore,
-                    std::unique_ptr<L1SpillObserver> observer = nullptr);
+  L1SpillManagement(func::FuncOp func, ttcore::GridAttr deviceGrid,
+                    uint64_t l1BudgetPerCore, uint64_t usableL1Size,
+                    std::unique_ptr<L1SpillObserver> observer = nullptr);

+  /// Dead zone = usableL1Size - l1BudgetPerCore.
+  /// This is the L1 region that CBs can physically use but the simulation
+  /// does not track as available for tensor placement.
+  uint64_t l1DeadZone;
```

**`lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`** (tt-mlir)
```diff
 L1SpillManagement<MemoryTracker>::L1SpillManagement(
-    func::FuncOp func, ttcore::GridAttr deviceGrid, uint64_t l1BudgetPerCore,
-    std::unique_ptr<L1SpillObserver> observer)
+    func::FuncOp func, ttcore::GridAttr deviceGrid, uint64_t l1BudgetPerCore,
+    uint64_t usableL1Size, std::unique_ptr<L1SpillObserver> observer)
     : func(func), deviceGrid(deviceGrid), l1BudgetPerCore(l1BudgetPerCore),
       cbFragCushion(static_cast<uint64_t>(kCBFragCushionFraction * l1BudgetPerCore)),
+      l1DeadZone(usableL1Size > l1BudgetPerCore ? usableL1Size - l1BudgetPerCore : 0) {

 // In ensureFitsL1, replacing previous simple condition:
-  if (l1Size > 0 && speculativeAddr &&
-      wouldCBsOverlapTensors(op, pos, cbPeakUsage, *speculativeAddr)) {
-    l1Size = handleFragmentation(op, pos, data, opL1Usage, cbPeakUsage, l1Size);
-  }
+  if (l1Size > 0 && speculativeAddr) {
+    if (l1DeadZone > 0 && cbPeakUsage > l1DeadZone) {
+      // CB physically extends into the dead zone below the simulation floor.
+      // No virtual-space check can detect this; demotion to DRAM is mandatory.
+      llvm::SmallVector<Value> toEvict;
+      for (Value operand : op->getOperands())
+        if (liveValues.count(operand)) toEvict.push_back(operand);
+      for (Value victim : toEvict) evictValue(victim, pos, data);
+      demoteToDram(op); evictForDramCBGrowth(op, pos, data);
+      return 0;
+    }
+    if (wouldCBsOverlapTensors(op, pos, cbPeakUsage, *speculativeAddr)) {
+      l1Size = handleFragmentation(op, pos, data, opL1Usage, cbPeakUsage, l1Size);
+    }
+  }
```

**`lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyL1SpillManagement.cpp`** (tt-mlir)
```diff
-      L1SpillManagement<SumL1MemoryTracker> spill(func, deviceGrid, l1BudgetPerCore, std::move(observer));
+      L1SpillManagement<SumL1MemoryTracker> spill(func, deviceGrid, l1BudgetPerCore,
+                                                   chipDesc.getUsableL1Size(), std::move(observer));
```

## 6. After Fix — How It Works

The constructor computes `l1DeadZone = 1,395,424 - 1,325,652 = 69,772 bytes` for WH N150.

In `ensureFitsL1`, before checking CB overlap in virtual space, the dead zone guard checks whether the CB physically extends into the dead zone. For the Conv2d with `cbPeakUsage = 73,728 > 69,772`, the guard fires:

1. All live L1 inputs of the Conv2d are evicted (moved to DRAM) — this keeps the eviction set homogeneous for downstream layout validation.
2. `demoteToDram(op)` marks the Conv2d output as DRAM interleaved.
3. `evictForDramCBGrowth(op, pos, data)` adjusts the simulation's free tracking to account for the DRAM CB size (which is larger than the L1 CB).

The Conv2d runs from DRAM, which has a larger but non-overlapping CB region. The TT_THROW at `program.cpp:1366` is avoided because the tensor at address `171,520` is no longer placed there — the DRAM path allocates output buffers outside the CB region.

## 7. Test Results

| Test | Before | After |
|------|--------|-------|
| `test_opt_sweep[opt_level_2-block_C]` | **FAIL (CB clash at 171520)** | **PASS** |
| `test_opt_sweep[opt_level_1-block_C]` | PASS | PASS |
