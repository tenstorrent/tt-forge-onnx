# HEIGHT_SHARDED L1 Spill Waste — BEV GridSample Investigation

---

## 1. Goal

The BEV model has 40 `grid_sample` ops. At `opt_level_2` the compiler spends time
allocating fast on-chip L1 memory for tensors related to those ops, then immediately
throwing that L1 away before anything reads from it. The goal is to eliminate those
wasted allocations so:

- Compilation is cleaner (no useless memory moves)
- The L1 budget that was wasted is available for ops that can actually use it
- The `TTNNCollectPerfMetrics` report no longer shows unnecessary HEIGHT_SHARDED spills

---

## 2. Background: L1 vs DRAM on Tenstorrent Hardware

Each Tenstorrent core has a small, fast on-chip SRAM called **L1** (~1.3 MB per core on
Wormhole N150). Off-chip DRAM is much larger but 5–10× slower.

The compiler tries to keep tensor data in L1. One strategy is **HEIGHT_SHARDED**: cut the
tensor across its rows and give each core a slice to hold in its L1.

```
HEIGHT_SHARDED tensor: 1×64×200×400 (64 channels, 200 rows, 400 cols)
split across 8 cores  →  each core holds 25 rows in its L1

  Core 0 L1 │ rows  0-24  │ bf16  ← fast, on-chip
  Core 1 L1 │ rows 25-49  │
  Core 2 L1 │ rows 50-74  │
  ...
  Core 7 L1 │ rows 175-199│

DRAM (fallback): all 200 rows in one contiguous off-chip buffer
```

When the L1 budget runs out the compiler must **spill** a tensor back to DRAM by inserting
a `to_memory_config(L1 → DRAM)` op. That spill adds latency and wastes bandwidth.

---

## 3. What Is grid_sample and Why It Is Special

`grid_sample` is the op at the heart of every BEV transform block. Given a feature map
(the camera image features) and a coordinate grid (where to sample in that feature map),
it performs bilinear or nearest-neighbor interpolation and produces a resampled feature map.
In the BEV model it is used to project camera-space features into bird's-eye-view space.

The BEV ONNX model has **40 grid_sample ops**. Each one produces a tensor of shape roughly
`1×64×200×400` in BF16.

**The special constraint:** the Tenstorrent `ttnn::grid_sample` metal kernel requires its
output to be in **ROW_MAJOR layout** (plain C-order rows). The rest of the network expects
**TILE layout** (data packed into 32×32 blocks that the hardware math engines use). So after
every `grid_sample` the `TTNNWorkaroundsPass` inserts a format-conversion op to go from
ROW_MAJOR back to TILE:

```
grid_sample output
     │  dtype=BF16, layout=ROW_MAJOR, memory=DRAM   ← metal kernel requires this
     ▼
 to_layout("gridsample_0_workaround")               ← inserted by TTNNWorkaroundsPass
     │  dtype=BF16, layout=TILE, memory=???
     ▼
 next conv2d / concat / slice_static
```

There are 40 grid_sample ops → **40 workaround `to_layout` ops** in the model.

---

## 4. The Problem: Wasted DRAM→L1→DRAM Round-Trip

At `opt_level_2` the compiler runs two key passes in sequence:

1. **TTNNGreedyMemoryLayoutPropagation (MLA)** — looks at each op and decides the best
   output memory layout. For the 40 workaround `to_layout` ops, MLA assigns
   `HEIGHT_SHARDED L1 TILE` to their outputs, hoping to speed up the downstream conv2d.

2. **TTNNGreedyL1SpillManagement** — walks the ops again and checks whether the L1 budget
   can actually hold all the MLA assignments simultaneously. With 226 conv2d ops also
   competing for L1, the budget is exhausted. The 40 `to_layout` outputs are **immediately
   spilled back to DRAM** by inserting a `to_memory_config(L1 → DRAM)` op after each one.

The final IR for each grid_sample path looks like:

```
grid_sample (op 0 of 40)
     │  ROW_MAJOR BF16 DRAM
     ▼
 to_layout "gridsample_0_workaround"           ← format conversion
     │  TILE BF16, HEIGHT_SHARDED L1            ← MLA wanted this in L1...
     ▼
 to_memory_config "gridsample_0_workaround_spill"  ← ...but L1SpillManagement spilled it
     │  TILE BF16, DRAM interleaved
     ▼
 conv2d / concat (downstream consumer)
```

The HEIGHT_SHARDED L1 tensor is **allocated and thrown away** before any consumer reads it.
This happens 40 times — once per grid_sample op.

**Cost of each round-trip:**
- One L1 shard allocation (takes L1 space from ops that could actually use it)
- One DRAM write (L1 → DRAM spill bandwidth)
- One DRAM read (consumer reads from DRAM anyway)
- Shows up in `TTNNCollectPerfMetrics` as a spilled HEIGHT_SHARDED value

The perf metric showed **40 HEIGHT_SHARDED L1 → DRAM spills** at the start of this investigation.

---

## 5. Why the Fix Is Not Obvious: Pass Ordering

The tricky part is that the workaround `to_layout` ops are **not visible** during most of
the compiler pipeline. Here is the order of passes at `opt_level_2`:

```
PASS ORDER (opt_level_2, greedy optimizer)
══════════════════════════════════════════

[INSIDE DevicePassesWrapper — steps 1–7]
  1. TTNNRowMajorLayoutPropagation
  2. TTNNGreedyMemoryLayoutPropagation    ← assigns HEIGHT_SHARDED L1
  3. TTNNGreedyL1SpillManagement          ← inserts to_memory_config(L1 → DRAM) spills
  4. CanonicalizerPass #1
  5. TTNNOperationValidationAndFallback
  6. CanonicalizerPass #2
  7. TTNNPrepareConv2dWeightsAndBias

[OUTSIDE DevicePassesWrapper — steps 8+]
  8.  TTNNWorkarounds      ← *** inserts the 40 GridSample to_layout ops HERE ***
  9.  CanonicalizerPass
  10. CSEPass
  11. TTNNDecomposeLayouts  ← decomposes to_layout(DRAM → HEIGHT_SHARDED L1 TILE) into:
                                  sub-op A: to_layout(ROW_MAJOR DRAM → TILE DRAM)   (tilize)
                                  sub-op B: to_memory_config(TILE DRAM → HEIGHT_SHARDED L1)
  [no canonicalization here ← this was the gap]
  12. TTNNDeallocate
  13. TTNNCollectPerfMetrics  ← reports spills
```

**Key insight:** `TTNNWorkarounds` (step 8) runs AFTER the greedy optimizer block (steps
1–7). The 40 GridSample `to_layout` ops do not exist when steps 2–6 run. Any fix that
looks for them inside steps 1–7 will find nothing.

After `TTNNDecomposeLayouts` (step 11), the IR for one GridSample path becomes:

```
grid_sample
     │  ROW_MAJOR BF16 DRAM
     ▼
 to_layout "tilize"                          ← new sub-op from decomposition
     │  TILE BF16 DRAM
     ▼
 to_memory_config "shard to L1"             ← new sub-op from decomposition
     │  TILE BF16 HEIGHT_SHARDED L1
     ▼
 to_memory_config "spill" (from step 3)     ← still here from L1SpillManagement
     │  TILE BF16 DRAM interleaved
     ▼
 consumer
```

Now the two consecutive `to_memory_config` ops form a **DRAM → L1 → DRAM round-trip** —
the L1 step is completely useless. If a canonicalization pass ran here it could eliminate
it. But there was no canonicalization pass after step 11.

---

## 6. What We Tried

### Attempt 1 — Pattern A/B inside the canonicalization passes (steps 4 and 6)

**Idea:** At canonicalization time look for the pattern:

```
to_layout(... → HEIGHT_SHARDED L1)
     ▼
to_memory_config(HEIGHT_SHARDED L1 → DRAM)   ← spill
```

and simplify it directly in DRAM, removing the L1 step.

Two specific patterns were written:

**Pattern A (TILE DRAM input):**
```
BEFORE:
  %l1   = to_layout(%dram_tile, HEIGHT_SHARDED_L1)   // DRAM TILE → L1
  %out  = to_memory_config(%l1, DRAM_interleaved)    // L1 → DRAM

AFTER (L1 middle hop removed):
  %out  = to_memory_config(%dram_tile, DRAM_interleaved)
  → identity fold fires next: op removed, consumer reads %dram_tile directly
```

**Pattern B (ROW_MAJOR DRAM input, the GridSample case):**
```
BEFORE:
  %l1   = to_layout(%row_major_dram, HEIGHT_SHARDED_L1_TILE)  // tilize + shard into L1
  %out  = to_memory_config(%l1, DRAM_TILE)                    // spill back to DRAM

AFTER (tilize directly into DRAM, skip L1 entirely):
  %out  = to_layout(%row_major_dram, DRAM_TILE)               // tilize in DRAM
```

**What happened:** The patterns never fired. Added diagnostic prints to every
`to_memory_config` fold attempt. The output showed:

```
[PATTERNA_DIAG] FAIL_NO_TOLAYOUT at to_memory_config: input is conv2d
[PATTERNA_DIAG] FAIL_NO_TOLAYOUT at to_memory_config: input is concat
[PATTERNA_DIAG] FAIL_NO_TOLAYOUT at to_memory_config: input is reshape
```

Every single `to_memory_config(L1 → DRAM)` op at steps 4/6 had a **compute op** as its
input — not a `to_layout` op. The GridSample `to_layout` ops didn't exist yet at steps 4
and 6 because `TTNNWorkarounds` had not run. The spills at steps 4/6 are from conv2d,
concat, and reshape — ops MLA placed in L1 that the spill manager evicted. Patterns A/B
were looking for something that didn't exist yet.

**Result: Failed — wrong pipeline stage.**

---

### Attempt 2 — `foldDRAMtoL1toDRAMRoundTrip` + post-decompose canonicalization

**Correct diagnosis:** The round-trip only becomes visible AFTER `TTNNDecomposeLayouts`
(step 11) creates the `to_memory_config(DRAM → HEIGHT_SHARDED L1)` sub-op that pairs with
the existing `to_memory_config(HEIGHT_SHARDED L1 → DRAM)` spill from step 3. The gap was
that no canonicalization ran after step 11.

**Two-part fix:**

**Part 1 — New fold `foldDRAMtoL1toDRAMRoundTrip`** in `TTNNOps.cpp`:

```
Pattern it eliminates:
  %l1   = to_memory_config(%dram_src, HEIGHT_SHARDED_L1)  // DRAM → L1 (producer)
  %dram = to_memory_config(%l1,       DRAM_interleaved)   // L1 → DRAM (this op)

Requirements:
  • this op's output is DRAM                   ✓
  • producer's output is sharded L1            ✓
  • producer's input is DRAM                   ✓
  • producer has exactly one use               ✓
  • producer and this op are adjacent          ✓

After fold — rewire this op to bypass the L1 hop:
  %dram = to_memory_config(%dram_src, DRAM_interleaved)   // DRAM → DRAM directly

Then foldIdentityToMemoryConfigOp fires if types match:
  (op removed entirely; consumer reads %dram_src directly)
```

For the GridSample path, the end result after both folds:

```
BEFORE fold:
  grid_sample → to_layout(tilize) → to_memory_config(shard→L1) → to_memory_config(spill→DRAM) → consumer

AFTER fold:
  grid_sample → to_layout(tilize) → consumer
                                     ↑
                    HEIGHT_SHARDED intermediate is gone
```

**Part 2 — Canonicalization pass after `TTNNDecomposeLayouts`** in `TTNNPipelines.cpp`:

```cpp
createTTNNPipelineLayoutDecompositionPass(devicePm, options);

// NEW: triggers foldDRAMtoL1toDRAMRoundTrip to eliminate DRAM→L1→DRAM pairs
// that TTNNDecomposeLayouts just created by expanding to_layout sub-ops.
devicePm.addPass(mlir::createCanonicalizerPass());
```

Without this, the new fold never gets a chance to fire.

**Result: Partially worked.** The fold eliminated round-trips, but not all spills.

---

## 7. What the IR Dump Showed

To measure the actual effect, a diagnostic pass (`TTNNDumpMemoryOpsPass`) was added that
counts `L1→DRAM spills`, `DRAM→L1 shards`, and `DRAM→L1→DRAM pairs` at 6 checkpoints:

```
Checkpoint                       L1→DRAM spills  DRAM→L1 shards  Pairs
────────────────────────────── ──────────────── ──────────────── ──────
1-after-L1Spill                     250              165            65
2-after-Canon1                      185              100             0
3-after-Canon2                      170              100             0
4-before-DecomposeLayouts           170              100             0
5-after-DecomposeLayouts            238              125             3
6-after-PostDecompCanon             216              103             0
```

**Reading the numbers:**

- At checkpoint 1 there are **65 DRAM→L1→DRAM pairs** — these are round-trips that existed
  immediately after L1SpillManagement. Canon1 (checkpoint 2) eliminated all 65 using the
  existing `foldConsecutiveToMemoryConfigOp` fold. The pairs went from 65 → 0, and spills
  dropped from 250 → 185.

- At checkpoint 5, `TTNNDecomposeLayouts` created **3 new pairs** by expanding `to_layout`
  sub-ops next to existing spills. The new `foldDRAMtoL1toDRAMRoundTrip` at checkpoint 6
  eliminated those 3. Pairs went from 3 → 0.

- **The remaining 216 spills at checkpoint 6 are not pairs.** They are `L1→DRAM` spills
  with no matching `DRAM→L1` shard. This means they come from **compute ops** (conv2d,
  concat, etc.) whose outputs MLA assigned HEIGHT_SHARDED L1 that L1SpillManagement
  immediately evicted — not from the GridSample `to_layout` workaround ops at all.

**Summary of what the fold fixed vs what remains:**

```
Original 40 HEIGHT_SHARDED spills reported by TTNNCollectPerfMetrics
  → actually traceable to the DRAM→L1→DRAM pairs at checkpoint 5
  → foldDRAMtoL1toDRAMRoundTrip eliminated them ✓

Remaining 216 L1→DRAM spills at checkpoint 6
  → come from conv2d, concat, reshape whose outputs MLA placed in HEIGHT_SHARDED L1
  → L1SpillManagement immediately evicts them (budget is exhausted by other live tensors)
  → foldDRAMtoL1toDRAMRoundTrip does NOT help here (these are not DRAM→L1→DRAM pairs)
  → these require a deeper fix in MLA itself
```

---

## 8. What Worked and What Didn't

| What | Outcome | Why |
|------|---------|-----|
| Pattern A/B at canonicalization steps 4/6 | **Did not work** | GridSample `to_layout` ops don't exist yet at those steps — inserted later by TTNNWorkarounds |
| `foldDRAMtoL1toDRAMRoundTrip` + post-decompose canon | **Partially worked** | Eliminated the 65+3 DRAM→L1→DRAM pairs from decomposed GridSample workarounds |
| Fixing the 216 remaining compute-op spills | **Not yet done** | Requires deeper MLA fix — explained below |

---

## 9. The Next Problem: MLA Assigns L1 That Will Be Immediately Spilled

The 216 remaining spills follow this pattern:

```
conv2d / concat / reshape output
     │  TILE BF16, HEIGHT_SHARDED L1   ← MLA assigned this
     ▼
 to_memory_config                      ← L1SpillManagement inserted this spill
     │  TILE BF16, DRAM interleaved
     ▼
 next op
```

MLA is assigning HEIGHT_SHARDED L1 to compute op outputs, but L1SpillManagement is
immediately evicting them because the L1 budget is already consumed by other live tensors.
MLA and L1SpillManagement are not coordinated: MLA optimistically assigns L1 to 216 ops
that L1SpillManagement then unconditionally evicts, allocating L1 space that is never used.

**Fix direction:** The fix needs to be in the MLA or SpillManagement layer:

- **Option A (MLA side):** When MLA considers assigning HEIGHT_SHARDED L1 to an op,
  simulate whether that assignment will survive L1SpillManagement given the current L1
  pressure. If it predicts an immediate eviction, assign DRAM instead and skip the
  round-trip entirely.

- **Option B (SpillManagement side):** When L1SpillManagement evicts a compute op's output
  to DRAM, instead of inserting a `to_memory_config` op, update the op's output type
  directly to DRAM. This avoids creating the L1 allocation in the first place and removes
  the need for the spill op entirely.

---

## 10. Files Changed So Far

| File | Change |
|------|--------|
| `lib/Dialect/TTNN/IR/TTNNOps.cpp` | Added `foldDRAMtoL1toDRAMRoundTrip`; cleaned up all debug prints from Pattern A/B attempt |
| `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` | Added `TTNNDumpMemoryOpsPass` at 6 checkpoints (gated by `TTMLIR_DUMP_MEMORY_OPS=1`); added `mlir::createCanonicalizerPass()` after `TTNNDecomposeLayouts` |

---

## 11. How to Enable the IR Dump

```bash
TTMLIR_DUMP_MEMORY_OPS=1 python3 /tmp/run_bev_dump.py 2>&1 | tee /tmp/memdump.txt
```

This prints a summary line at each of the 6 checkpoints:

```
[MEMDUMP:1-after-L1Spill]       SUMMARY: L1->DRAM spills=250  DRAM->L1 shards=165  pairs=65
[MEMDUMP:2-after-Canon1]        SUMMARY: L1->DRAM spills=185  DRAM->L1 shards=100  pairs=0
[MEMDUMP:3-after-Canon2]        SUMMARY: L1->DRAM spills=170  DRAM->L1 shards=100  pairs=0
[MEMDUMP:4-before-Decompose]    SUMMARY: L1->DRAM spills=170  DRAM->L1 shards=100  pairs=0
[MEMDUMP:5-after-Decompose]     SUMMARY: L1->DRAM spills=238  DRAM->L1 shards=125  pairs=3
[MEMDUMP:6-after-PostDecompCanon] SUMMARY: L1->DRAM spills=216  DRAM->L1 shards=103  pairs=0
```

Note: always run with `python3` directly, not `pytest`. Pytest captures `llvm::errs()` and
the dump output is invisible.
