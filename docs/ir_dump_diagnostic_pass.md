# TTNN IR Dump Diagnostic Pass

## Purpose

When a memory-layout fix does not produce the expected improvement, we need to see
exactly what the TTNN IR looks like after each pass in the pipeline. This document
describes the `TTNNDumpMemoryOpsPass` diagnostic pass that was added to
`lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` to make this possible.

---

## How to Enable

Set the environment variable `TTMLIR_DUMP_MEMORY_OPS=1` before running compilation:

```bash
TTMLIR_DUMP_MEMORY_OPS=1 python3 my_compile_script.py 2>&1 | tee /tmp/memdump.txt
```

> **Note:** `llvm::errs()` output is captured by pytest's `-q` flag. Always run with
> `python3` directly and `2>&1 | tee` to see the dump output.

---

## What It Prints

At each instrumented checkpoint the pass prints every `to_memory_config` op that moves
a tensor between L1 and DRAM, then a summary line:

```
[MEMDUMP:5-after-DecomposeLayouts] ========================================
  [L1->DRAM SHARDED ROUNDTRIP] loc("gridsample_0_workaround") @ func.mlir:1234
    in  : tensor<1x64x200x400xbf16, #ttnn.ttnn_layout<...HEIGHT_SHARDED L1...>>
    out : tensor<1x64x200x400xbf16, #ttnn.ttnn_layout<...DRAM interleaved...>>
  [DRAM->L1 SHARDED] loc("gridsample_0_workaround") @ func.mlir:1233
    in  : tensor<1x64x200x400xbf16, #ttnn.ttnn_layout<...DRAM interleaved...>>
    out : tensor<1x64x200x400xbf16, #ttnn.ttnn_layout<...HEIGHT_SHARDED L1...>>
  ...
[MEMDUMP:5-after-DecomposeLayouts] SUMMARY:
  L1->DRAM spills=40  DRAM->L1 shards=40  DRAM->L1->DRAM pairs=40
[MEMDUMP:5-after-DecomposeLayouts] ========================================
```

### Tag meanings

| Tag | Meaning |
|-----|---------|
| `[L1->DRAM]` | Spill: tensor evicted from any L1 to DRAM |
| `[L1->DRAM SHARDED]` | Spill from HEIGHT_SHARDED L1 specifically |
| `[L1->DRAM ROUNDTRIP]` | Spill whose input came directly from a DRAM→L1 shard (wasteful pair) |
| `[L1->DRAM SHARDED ROUNDTRIP]` | Both: sharded L1 spill that is also half of a DRAM→L1→DRAM pair |
| `[DRAM->L1]` | Shard: tensor moved from DRAM into any L1 |
| `[DRAM->L1 SHARDED]` | Shard into HEIGHT_SHARDED L1 specifically |

---

## Checkpoint Locations

Six checkpoints are inserted in the pipeline:

```
PASS ORDER (opt_level_2, greedy optimizer)
══════════════════════════════════════════

[INSIDE DevicePassesWrapper]
  TTNNGreedyL1SpillManagement
  ★ checkpoint 1-after-L1Spill          ← how many ops were spilled?
  CanonicalizerPass #1
  ★ checkpoint 2-after-Canon1           ← did Pattern A/B fire?
  TTNNOperationValidationAndFallback
  CanonicalizerPass #2
  ★ checkpoint 3-after-Canon2           ← did OPVF-related folds fire?

[OUTSIDE DevicePassesWrapper]
  TTNNWorkarounds  (inserts GridSample to_layout revert ops)
  ...
  ★ checkpoint 4-before-DecomposeLayouts ← state before decomposition
  TTNNDecomposeLayouts
  ★ checkpoint 5-after-DecomposeLayouts  ← new DRAM→L1→DRAM pairs visible here
  CanonicalizerPass #3  (our new pass)
  ★ checkpoint 6-after-PostDecompCanon  ← should be 0 pairs if fix works
```

---

## Example: Fix Working Correctly

```
[MEMDUMP:1-after-L1Spill]    SUMMARY: L1->DRAM spills=40  DRAM->L1 shards=0   pairs=0
[MEMDUMP:2-after-Canon1]     SUMMARY: L1->DRAM spills=40  DRAM->L1 shards=0   pairs=0
[MEMDUMP:3-after-Canon2]     SUMMARY: L1->DRAM spills=40  DRAM->L1 shards=0   pairs=0
[MEMDUMP:4-before-Decompose] SUMMARY: L1->DRAM spills=40  DRAM->L1 shards=0   pairs=0
[MEMDUMP:5-after-Decompose]  SUMMARY: L1->DRAM spills=40  DRAM->L1 shards=40  pairs=40
                                                                ↑                   ↑
                              TTNNDecomposeLayouts created 40 new DRAM→L1 shards
                              and revealed 40 DRAM→L1→DRAM round-trip pairs

[MEMDUMP:6-after-PostDecompCanon] SUMMARY: L1->DRAM spills=0  DRAM->L1 shards=0  pairs=0
                                                               ↑
                              foldDRAMtoL1toDRAMRoundTrip eliminated all 40 pairs
```

---

## Example: Fix Not Working — fold blocked by non-adjacency

```
[MEMDUMP:5-after-Decompose]      SUMMARY: ... pairs=40
[MEMDUMP:6-after-PostDecompCanon] SUMMARY: ... pairs=40   ← still 40, fold didn't fire
```

This means the adjacency check in `foldDRAMtoL1toDRAMRoundTrip` is failing. There is
at least one op between `to_memory_config(DRAM→L1)` and `to_memory_config(L1→DRAM)`.

**Fix direction:** Relax the adjacency check, or inspect the IR to see what op is
between them and whether it is safe to bypass.

---

## Example: Fix Not Working — producer has multiple uses

```
[MEMDUMP:5-after-Decompose]      SUMMARY: ... pairs=40
[MEMDUMP:6-after-PostDecompCanon] SUMMARY: ... pairs=30   ← only 10 eliminated
```

10 pairs remain because `producerOp->hasOneUse()` is failing for some of them.
The `to_memory_config(DRAM→HEIGHT_SHARDED L1)` output is consumed by more than just
the spill — another op also reads the HEIGHT_SHARDED L1 tensor.

**Fix direction:** These cannot be simply eliminated. Need to rethink whether to
keep the L1 tensor for those consumers or replace it with a separate copy.

---

## Example: Wrong pattern — spills from compute ops, not to_memory_config pairs

```
[MEMDUMP:5-after-Decompose] SUMMARY: L1->DRAM spills=40  DRAM->L1 shards=0  pairs=0
```

Spills=40 but shards=0 and pairs=0. This means the 40 spills come from compute ops
(conv2d, concat, etc.) whose outputs are HEIGHT_SHARDED L1 — NOT from decomposed
`to_layout` ops. The `foldDRAMtoL1toDRAMRoundTrip` fold does not help here.

**Fix direction:** The root cause is MLA assigning HEIGHT_SHARDED L1 to compute op
outputs that then get immediately spilled. Fix must be either in
`TTNNGreedyMemoryLayoutPropagation` (don't assign HEIGHT_SHARDED if L1 pressure is
too high) or in `TTNNGreedyL1SpillManagement` (when evicting a compute op output to
DRAM, update the op's output type directly rather than inserting a to_memory_config).

---

## Implementation Notes

The pass is defined inline in `TTNNPipelines.cpp` as a `PassWrapper<..., OperationPass<ModuleOp>>`.
It does not need tablegen registration — it is instantiated directly in the pipeline.

Key design decisions:
- **Gated by env var** (`TTMLIR_DUMP_MEMORY_OPS`) so there is zero overhead in normal builds
- **Reads Value types directly** (not via `getDefiningOp` helper functions) so it works
  for block arguments as well as op results
- **Adjacency is NOT required** for ROUNDTRIP detection in the dump — the dump shows
  ALL DRAM→L1→DRAM patterns regardless of adjacency, while the fold only eliminates
  adjacent pairs. This helps distinguish between "fold could fire but has a bug" vs
  "pattern exists but is non-adjacent"
- **Counts are per-module** (walks the entire module op), so numbers reflect the
  total across all functions in the device module

---

## Files Changed

| File | Change |
|------|--------|
| `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` | Added `TTNNDumpMemoryOpsPass` struct and `createDumpMemoryOpsPass()` factory; inserted at 6 checkpoints gated by `TTMLIR_DUMP_MEMORY_OPS` |

No new files, no CMakeLists changes, no tablegen changes required.

---

## Status

| Item | Status |
|------|--------|
| Pass implemented and compiling | Done |
| `.so` built (310 MB) and installed | Done |
| BEV benchmark run with `TTMLIR_DUMP_MEMORY_OPS=1` | Running |
| Analysis of dump output | Pending |
| Fix confirmed working or revised based on dump | Pending |
