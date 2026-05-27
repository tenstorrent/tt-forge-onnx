# Block A L1 Sharding Work — 2026-05-22

## Context

Block A (`block_A_deformed_backbone`) is the largest sub-graph of the BEV model.
Config: `opt_level=2, BFloat16, HiFi3, fp32_dest_acc=True, trace=True`.
Baseline performance: **~2.76 FPS** (337 ms inference).

All compiler fixes are in `third_party/tt-mlir`, not in tt-forge-onnx.

---

## Fix_7: Eliminate L1-int-via-DRAM bounce patterns (DONE ✓)

### Problem
20 `to_memory_config` ops with the pattern:
```
to_memory_config(%src_L1_sharded → DRAM) → to_memory_config(DRAM → L1_int)
```
The DRAM hop is structurally wasteful: L1_sharded → L1_int is a single TTNN op.

### Root cause (Fix_6 limitation)
The old `BypassDRAMForL1InterleavedFromL1Sharded` pattern fired **per-consumer**.
When a DRAM op had TWO L1_int consumers:
1. First rerouting sees the second as `hasOtherComputeUser=true` → fires
2. After rerouting, DRAM becomes single-consumer → blocked by the guard

### Fix (tt-mlir: `lib/Dialect/TTNN/IR/TTNNOps.cpp`)
Replaced per-consumer pattern with `BypassDRAMForL1InterleavedConsumers`:
- Fires on the **DRAM-producing op** itself
- Collects ALL L1_int consumers at once and reroutes them simultaneously
- Erases DRAM op entirely when no other compute users remain
- Guard: skips if L1_sharded source already has a `DeallocateOp` user

### Result
| Metric | Before | After |
|---|---|---|
| L1_int-via-DRAM bounces | 20 | **0** |
| `to_memory_config` ops | 228 | 224 (−4) |
| IR lines | 3810 | 3803 (−7) |
| FPS | 2.76 | 2.76 (no change) |

The bounce ops are in the trace critical path but too small to produce measurable FPS gain.

---

## IR Analysis After Fix_7

### `to_memory_config` breakdown (224 total)

| Pattern | Count | Meaning |
|---|---|---|
| `l1_sharded → dram` | 112 | L1 capacity checkpoints |
| `l1_sharded → l1_int` | 44 | Direct, correct conversions |
| `l1_int → dram` | 32 | Interleaved checkpoints (permute/slice) |
| `dram → l1_sharded` | 28 | DRAM reloads |
| `l1_int → l1_sharded` | 8 | Resharding |

### Conv2d analysis (148 ops)

| Input layout | Count |
|---|---|
| DRAM | 120 (81%) |
| L1_sharded | 40 (27%) |
| L1_int | 4 (3%) |

**72 sharded-but-spilled**: conv2d produces L1_sharded output that is
immediately followed by `to_memory_config(L1_sharded → DRAM)`.

All 72 have `conv2d_slice_config = <l1_full, 0>` (full L1 activation block).

### Producers of the 112 L1_sharded→DRAM spills

| Producer | Count |
|---|---|
| `ttnn.conv2d` | 72 |
| `ttnn.max_pool2d` | 20 |
| `ttnn.concat` | 12 |
| `ttnn.add` | 8 |

### Non-sharded ops summary

| Op | Count not L1-sharded | Root cause | Fixable? |
|---|---|---|---|
| `conv2d` | 120 | DRAM checkpoint between stages | Potentially |
| `conv_transpose2d` | 16 | Hardware constraint (no L1-sharded kernel) | No |
| `permute` | 28 | Arbitrary permutation breaks sharding | No |
| `reshape` | 32 | Shape change breaks sharding geometry | No |
| `slice_static` | 56 | Slice breaks sharding continuity | No |
| `concat` | 32 | Multi-input sharding alignment | Partial |
| `add` | 16 | One residual branch arrives from DRAM | Yes if upstream fixed |
| `max_pool2d` | 12 | DRAM checkpoint before pool | Yes if upstream fixed |

---

## Fix_8 Attempt: Reduce `act_block_h` in L1SpillManagement (REVERTED)

### Hypothesis
The 72 sharded-but-spilled conv2d ops are caused by L1 capacity pressure.
With `l1_full, act_block_h=0`, each conv2d uses the full L1 for its CB
(circular buffers). Reducing `act_block_h` to a smaller value would shrink
the CB peak, allowing consecutive conv2d ops to coexist in L1.

### Implementation
Added `tryReduceConv2dActBlockH()` to `L1SpillManagement` in:
- `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`
- `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`

Called from `handleOOM()` as the first step before Belady eviction.
Tries `act_block_h` values from 1024 down to 32 (step 32).

### What happened
- Fixed 4 ops (got `act_block_h=1024`) — but these were `dram_width` side-path ops, not from the 72 spills
- The 72 sharded-but-spilled count **unchanged**
- FPS: 2.70 (slightly worse, within noise ±10ms)
- Reverted the `handleOOM` call site; method kept as dead code with comment

### Root cause of failure

The 72 spills are **NOT** caused by conv2d ops entering `handleOOM`.
They are caused by **Belady eviction triggered by downstream ops**:

| Downstream op triggering eviction | Count | Why act_block_h can't help |
|---|---|---|
| `conv_transpose2d` | 12 | No `L1Full` config; fix returns false |
| `concat` | 8 | No `act_block_h` concept |
| downstream `conv2d` | 32 | Conv2d validation PASSES (no `handleOOM`); Belady picks predecessor later |
| other paths | 20 | Same Belady path |

The fix fires when the **current failing op IS a conv2d with L1Full**.
But the actual evictions happen when a downstream op fails OOM and
Belady's algorithm picks the conv2d output as the farthest-last-use victim.

---

## Next Steps

### Option 1: Fix the Belady eviction path (deeper change)

Instead of firing `tryReduceConv2dActBlockH` only in `handleOOM`, also invoke it
from **inside `evictUntil`**: before evicting a victim that is a conv2d L1_sharded
output, try reducing the **current failing op's** footprint first.

This requires a new callback into `evictUntil`:
```
evictUntil(pos, data, [&]() {
  // Before picking Belady victim, try reducing the failing op's config
  if (tryReduceConv2dActBlockH(op, ...)) return true;
  // Otherwise fall through to standard eviction
  evictFarthestUse();
  return validate().isSuccess();
});
```

Challenge: `evictUntil` currently just evicts and re-validates. Adding a
"try config reduction" step requires passing the op context into `evictUntil`.

### Option 2: Pre-emptive act_block_h assignment before SpillManagement

Run a new pass BEFORE `GreedyL1SpillManagement` that:
1. Identifies conv2d → conv2d chains
2. Estimates if the intermediate would be evicted (L1 budget heuristic)
3. Pre-assigns smaller `act_block_h` to the downstream conv2d

Pro: no changes to complex spill management internals.
Con: requires accurate L1 budget estimation without running SpillManagement.

### Option 3: Accept DRAM checkpoints, improve DRAM bandwidth

If the 72 DRAM checkpoints are genuinely necessary (L1 can't hold them),
optimise the DRAM read/write bandwidth:
- Ensure `config_tensors_in_dram=true` is set (already done)
- Look for opportunities to reduce tensor sizes (different tile shapes)
- Profile the actual DRAM bottleneck with perf metrics

### Option 4: Move to a different Block A bottleneck

The 112 L1_sharded→DRAM spills account for ~224 `to_memory_config` ops.
Even if eliminated, the FPS gain may be modest (hardware trace amortizes
much of the overhead). Other bottlenecks to investigate:
- `conv_transpose2d` (16 ops, always DRAM, potentially slow kernels)
- `concat` (48 ops) — see if any can be eliminated by reshaping upstream
- Overall op count: 148 conv2d + 16 conv_transpose2d is the dominant cost

---

## Files Changed (still in-tree, not committed)

| File | Change | Status |
|---|---|---|
| `lib/Dialect/TTNN/IR/TTNNOps.cpp` | Fix_7: `BypassDRAMForL1InterleavedConsumers` | **Active** |
| `test/ttmlir/Dialect/TTNN/Canonicalizer/bypass_dram_for_l1_interleaved_from_l1_sharded.mlir` | Fix_7: updated test cases | **Active** |
| `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp` | Fix_8: `tryReduceConv2dActBlockH` method (not called) + Conv2dConfigParams include | Kept but **not wired** |
| `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h` | Fix_8: method declaration | Kept but **not wired** |

IR dumps:
- Fix_7 result: `BEV_MODEL_IRS/LATEST/BLOCK_A_AFTER_FIX_6/ttnn_block_A_deformed_backbone.mlir`
- Fix_8 result: `BEV_MODEL_IRS/LATEST/BLOCK_A_AFTER_FIX_8/ttnn_block_A_deformed_backbone.mlir`
- Logs: `BEV_MODEL_LOGS/LATEST/block_A_after_fix_6.log`, `block_A_after_fix_8.log`
