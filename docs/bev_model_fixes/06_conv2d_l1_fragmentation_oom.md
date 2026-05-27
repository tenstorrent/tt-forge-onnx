# Block A — Conv2d L1 Fragmentation OOM (ToLayoutOp Simulation Blind Spot)

## 1. Affected Test Cases

- `test_opt_sweep[enable_program_cache-opt_level_2-block_A]` — FAIL
- `test_opt_sweep[disable_program_cache-opt_level_2-block_A]` — FAIL

Block A: `block_A_deformed_backbone`. 4-camera inputs, 1x1 conv2d on HEIGHT_SHARDED features.

## 2. Failure

```
Not enough space to allocate 37748736 B L1 buffer across 64 banks,
where each bank needs to store 589824 B, but bank size is 1329888 B
(allocated: 606208 B, free: 723680 B, largest free block: 420576 B)
```

## 3. Failure Reason — Simulation-vs-Runtime Mismatch

**The design decision that causes the gap:**

By design, `ToLayoutOp` outputs (workaround-inserted layout conversion ops) are NOT added to `liveValues` in the simulation. The comment in `L1SpillManagement.cpp` explains: they are "short-lived L1 tenants" that will be consumed before the next eviction decision. `ensureFitsL1` is called for them (to make room if needed), but `memoryTracker` never tracks them.

**What happens at runtime in Block A:**

Four cameras produce four execution paths. In each path, three `ToLayoutOp` outputs of 303,104 bytes/core each are allocated in L1 top-down. The middle one is freed (its consumer completes), creating a fragmented heap:

```
L1 bank (1,329,888 bytes):

T1 = ToLayoutOp output:  [1,026,784 - 1,329,888)  303,104 B  <- live
T2 = ToLayoutOp output:  [  723,680 - 1,026,784)  303,104 B  <- freed (hole)
T3 = ToLayoutOp output:  [  420,576 -   723,680)  303,104 B  <- live
free:                    [        0 -   420,576)  420,576 B  <- largest contiguous block
```

The conv2d output needs 589,824 bytes/core contiguous. Since `420,576 < 589,824`, allocation fails with OOM.

**Why the simulation missed it:**

When `ensureFitsL1` is called for the conv2d output:

1. `processDeadTensors(pos)` has already freed the conv2d input (also a ToLayoutOp result).
2. The three ToLayoutOp outputs (T1, T2, T3) were never added to `memoryTracker`.
3. So `getOccupiedL1() = 0` and the simulation's free list shows the full budget as available.

The simulation concludes: "589,824 bytes fits (budget = 1,325,652, occupied = 0)" — assigns HEIGHT_SHARDED L1 — OOM at runtime. The `wouldCBsOverlapTensors` check in virtual space reports `lowestExisting = 289,364` virtual (= `462,848` real), which is far from the actual collision, so no demotion is triggered.

## 4. Fix Implementation Details

Added a **large-tensor fragmentation guard** in `ensureFitsL1`, placed after the dead zone check and before `wouldCBsOverlapTensors`:

```cpp
static constexpr double kMaxSingleTensorFraction = 0.40;
if (l1Size > static_cast<uint64_t>(kMaxSingleTensorFraction *
                                   static_cast<double>(l1BudgetPerCore))) {
    // Single tensor exceeds 40% of L1 budget.
    // Untracked ToLayoutOp L1 allocations can halve contiguous free space,
    // making a fit in simulation become OOM at runtime.
    // Evict live L1 inputs first (homogeneous inputs for downstream validation).
    llvm::SmallVector<Value> toEvict;
    for (Value operand : op->getOperands())
        if (liveValues.count(operand)) toEvict.push_back(operand);
    for (Value victim : toEvict) evictValue(victim, pos, data);
    demoteToDram(op);
    evictForDramCBGrowth(op, pos, data);
    return 0;
}
```

**Threshold rationale (WH N150):**

- `l1BudgetPerCore` = 1,325,652 bytes
- 40% threshold = 530,261 bytes
- Conv2d output: 589,824 bytes (44.5%) — **rejected, demoted to DRAM**
- ToLayoutOp outputs: 303,104 bytes (22.9%) — allowed in L1 (below 40%)

A tensor at 44.5% of the budget is inherently fragmentation-prone. Even a single untracked ToLayoutOp output (22.9%) can halve the effective contiguous space. With three concurrent ToLayoutOp outputs totalling 67.7% of the budget, placing a 44.5% tensor in L1 guarantees OOM when any of those ToLayoutOp outputs are still live.

Since the conv2d output is immediately followed by a `to_memory_config -> DRAM` op anyway (no downstream L1 consumer in Block A), there is no net performance loss from this demotion. The guard fires 4 times (once per camera input) and moves all four instances of the HEIGHT_SHARDED conv2d output to DRAM.

## 5. Files Changed with Diffs

**`lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`** (tt-mlir)
```diff
   if (l1Size > 0 && speculativeAddr) {
     if (l1DeadZone > 0 && cbPeakUsage > l1DeadZone) {
       // ... dead zone check (see Fix 4) ...
     }
+    // Large-tensor fragmentation guard: if a single tensor exceeds ~40% of
+    // the L1 budget, runtime fragmentation from untracked L1 allocations
+    // (e.g. ToLayoutOp outputs that are excluded from liveValues by design)
+    // can leave insufficient contiguous space even when the simulation's free
+    // list shows a fit. Threshold = 40% x budget (530,261 B on WH N150).
+    //
+    // Example (Block A): three concurrent ToLayoutOp outputs of 303,104 B
+    // each fragment L1 into a largest-free-block of 420,576 B. A conv2d
+    // output of 589,824 B (44.5%) cannot fit despite the simulation reporting
+    // full budget available (liveValues = empty due to ToLayoutOp exclusion).
+    static constexpr double kMaxSingleTensorFraction = 0.40;
+    if (l1Size > static_cast<uint64_t>(kMaxSingleTensorFraction *
+                                       static_cast<double>(l1BudgetPerCore))) {
+      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
+                   "    LARGE_TENSOR_FRAG: l1Size={0} > 40%% budget={1}, "
+                   "forcing DRAM to prevent fragmentation OOM",
+                   l1Size, l1BudgetPerCore);
+      llvm::SmallVector<Value> toEvict;
+      for (Value operand : op->getOperands()) {
+        if (liveValues.count(operand)) {
+          toEvict.push_back(operand);
+        }
+      }
+      for (Value victim : toEvict) {
+        evictValue(victim, pos, data);
+      }
+      demoteToDram(op);
+      evictForDramCBGrowth(op, pos, data);
+      return 0;
+    }
     if (wouldCBsOverlapTensors(op, pos, cbPeakUsage, *speculativeAddr)) {
       l1Size = handleFragmentation(op, pos, data, opL1Usage, cbPeakUsage, l1Size);
     }
   }
```

## 6. After Fix — How It Works

The guard fires for the conv2d output (589,824 > 530,261 bytes). The execution path is:

1. All live L1 inputs of the conv2d (if any are tracked in `liveValues`) are evicted to DRAM. This keeps the eviction set homogeneous for downstream layout validation.
2. `demoteToDram(op)` marks the conv2d's output layout as DRAM interleaved in the simulation's layout map.
3. `evictForDramCBGrowth(op, pos, data)` adjusts the simulation's memory tracker to account for the DRAM CB size difference (DRAM CBs are larger than L1 CBs for the same op, so existing L1 tenants may need to be evicted to make room for CB growth).

At runtime:
- The conv2d op reads its HEIGHT_SHARDED input from L1 (or DRAM, depending on preceding ops).
- The conv2d writes its output to DRAM interleaved.
- The subsequent `to_memory_config` op reads from DRAM — this is the same op that would have been emitted anyway (Block A's next consumer is not HEIGHT_SHARDED L1).
- No OOM occurs because the 589,824-byte/core output is now placed in DRAM, where the allocator operates on full-chip DRAM with no fragmentation constraint.

The guard fires 4 times in Block A (once per camera path). All four conv2d instances run from DRAM. The block compiles and runs correctly at opt_level_2.

## 7. Test Results

| Test | Before | After |
|------|--------|-------|
| `test_opt_sweep[opt_level_1-block_A]` | PASS | PASS |
| `test_opt_sweep[enable_program_cache-opt_level_2-block_A]` | **FAIL (OOM: 589824 B needs 420576 B avail)** | **PASS** |
| `test_opt_sweep[disable_program_cache-opt_level_2-block_A]` | **FAIL (OOM)** | **PASS** |
| Blocks B, C, E (regression check) | PASS | PASS (no regression) |
