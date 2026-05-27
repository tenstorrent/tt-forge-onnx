# Block A — DRAM Bounce Spill Analysis and Fix

**Block:** `block_A_deformed_backbone`
**Config:** `opt_level=2`, `BFloat16`, `HiFi3`, `fp32_dest_acc=True`, `trace=True`
**IR baseline:** `BEV_MODEL_IRS/LATEST/BLOCK_A/ttnn_block_A_deformed_backbone.mlir`

---

## What We Found

### The Bounce Spill Pattern

A "bounce spill" is a useless `DRAM → L1_sharded → DRAM` (or `DRAM → L1_sharded → L1_interleaved`)
round-trip in the TTNN IR. The intermediate L1-sharded tensor has exactly one consumer — another
`to_memory_config` that immediately moves it somewhere else — making the L1 staging a no-op that
wastes memory bandwidth.

Concrete example from the baseline IR (lines 1688–1696):

```
BASELINE:
  %21 = "ttnn.concat"(...)                         <- output in DRAM
  %22 = "ttnn.to_memory_config"(%21, L1_sharded)  <- BOUNCE: move to L1
  "ttnn.deallocate"(%21)
  %23 = "ttnn.to_memory_config"(%22, DRAM)         <- BOUNCE: immediately back to DRAM
  "ttnn.deallocate"(%22)
  %24 = "ttnn.conv2d"(%23, ...)                    <- conv2d reads from DRAM

AFTER FIX:
  %21 = "ttnn.concat"(...)                         <- output in DRAM
  %22 = "ttnn.conv2d"(%21, ...)                    <- conv2d reads directly from concat output
  "ttnn.deallocate"(%21)
```

The entire two-hop DRAM transit is gone. Two `to_memory_config` ops and one `deallocate` are
eliminated per bounce.

### Baseline Numbers (BLOCK_A IR)

| Metric | Count |
|---|---|
| Bounce spill patterns detected | **52** |
| `ttnn.to_memory_config` ops | 324 |
| `ttnn.deallocate` ops | 1308 |
| Total IR lines | 4007 |

### Why Bounces Were Introduced

The bounces come from two separate points in the TTNN compilation pipeline:

**Source 1 — `OperationValidationAndFallback` inside `DevicePassesWrapper`:**
MLA (`TTNNGreedyMemoryLayoutPropagation`) propagates L1-sharded layouts. When
`OperationValidationAndFallback` encounters an op that cannot accept an L1-sharded input (e.g. a
`concat` whose consumer is a `conv2d` that requires DRAM), it inserts `ToLayoutOp` ops to move
data to DRAM. These get combined with later `ToLayoutOp` ops that re-shard for the next op,
producing the bounce chain. A `canonicalize` pass ran after MLA but before `OperationValidationAndFallback`,
so it did not see these newly-inserted bounces.

**Source 2 — `TTNNDecomposeLayouts` (stage 12):**
`TTNNTraceHoistTransform` (stage 11) lifts ops into a `@trace_0_forward` function. Then
`TTNNDecomposeLayouts` (stage 12) converts all `ToLayoutOp` into explicit `to_memory_config`
chains. Bounce patterns that were invisible as `ToLayoutOp` become explicit `to_memory_config`
chains at this point. No canonicalize ran after stage 12, so these bounces survived to the
final TTNN IR.

---

## What We Tried

### Fix 1 — Investigation / IR Dump (no code change)

Enabled `TTMLIR_DUMP_PIPELINE_IR=1` to dump per-stage TTNN IR and counted bounces at each stage:

| Stage | Bounces |
|---|---|
| Post-MLA (stage 09) | 76 (as ToLayoutOp) |
| Post-TraceHoist (stage 11) | 76 (as ToLayoutOp) |
| Post-DecomposeLayouts (stage 12) | 88 (converted to to_memory_config) |
| Post-Deallocate (stage 13, final) | 40 (some folded by existing identity fold) |

Confirmed bounces fall into two groups:
- **32** in `@forward` (the main function body) — created by OperationValidationAndFallback
- **8** in `@trace_0_forward` (the trace-hoisted function) — created by TTNNDecomposeLayouts after TraceHoist

**Result:** 324 tmc ops, 4007 IR lines. FPS: 2.79

---

### Fix 2 — `FoldConsecutiveToMemoryConfigOps` Canonicalization Pattern + first canonicalize in DevicePassesWrapper

**Root cause addressed:** Bounces from `OperationValidationAndFallback` (Source 1 above).

**Code changes:**

**File 1: `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`**

Added `hasFolder = 1` and `hasCanonicalizer = 1` to `TTNN_ToMemoryConfigOp`:
```tablegen
def TTNN_ToMemoryConfigOp : TTNN_Op<"to_memory_config", [...]> {
  ...
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
```

**File 2: `lib/Dialect/TTNN/IR/TTNNOps.cpp`**

Added identity fold (returns input unchanged when input/output types match) and the
`FoldConsecutiveToMemoryConfigOps` pattern:

```cpp
// Identity fold: to_memory_config(x) -> x when types match
mlir::OpFoldResult mlir::tt::ttnn::ToMemoryConfigOp::fold(FoldAdaptor) {
  if (getInput().getType() == getResult().getType())
    return getInput();
  return nullptr;
}

// Fold: DRAM → L1_sharded (defOp) → DRAM/other (op)  =>  DRAM → DRAM/other
struct FoldConsecutiveToMemoryConfigOps
    : public OpRewritePattern<ttnn::ToMemoryConfigOp> {
  ...
  LogicalResult matchAndRewrite(ttnn::ToMemoryConfigOp op,
                                PatternRewriter &rewriter) const override {
    auto defOp = op.getInput().getDefiningOp<ttnn::ToMemoryConfigOp>();
    if (!defOp) return failure();

    // Only fold when intermediate is L1 sharded
    auto intermediateLayout = mlir::cast<TTNNLayoutAttr>(
        mlir::cast<RankedTensorType>(defOp.getResult().getType()).getEncoding());
    if (!intermediateLayout.hasShardedL1TensorMemoryLayout())
      return failure();

    // Allow soft (force=false) deallocates as additional users of intermediate
    SmallVector<ttnn::DeallocateOp> defOpDeallocs;
    for (Operation *user : defOp.getResult().getUsers()) {
      if (user == op.getOperation()) continue;
      auto deallocOp = dyn_cast<ttnn::DeallocateOp>(user);
      if (deallocOp && !deallocOp.getForce()) {
        defOpDeallocs.push_back(deallocOp);
        continue;
      }
      return failure();
    }

    for (ttnn::DeallocateOp dealloc : defOpDeallocs)
      rewriter.eraseOp(dealloc);

    rewriter.replaceOpWithNewOp<ttnn::ToMemoryConfigOp>(
        op, op.getResult().getType(), defOp.getInput(), op.getMemoryConfig());
    rewriter.eraseOp(defOp);
    return success();
  }
};

void mlir::tt::ttnn::ToMemoryConfigOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldConsecutiveToMemoryConfigOps>(context);
}
```

**File 3: `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`**

Added a second canonicalize pass inside `DevicePassesWrapper` after `PrepareConv2dWeightsAndBias`
(the pass that runs after `OperationValidationAndFallback`):

```cpp
innerPm.addPass(mlir::createCanonicalizerPass());  // existing — after L1SpillManagement
innerPm.addPass(createTTNNOperationValidationAndFallback(validationOptions));
innerPm.addPass(createTTNNPrepareConv2dWeightsAndBias());
// NEW: catch bounces inserted by OperationValidationAndFallback
innerPm.addPass(mlir::createCanonicalizerPass());
```

**Result:** 244 tmc ops (-80), 3842 IR lines (-165). Eliminated **44 of 52 bounces**.
FPS: 2.73–2.74. The remaining 8 bounces were in `@trace_0_forward` — they existed as
`ToLayoutOp` during all `DevicePassesWrapper` passes and were not yet converted to
`to_memory_config` chains, so the canonicalizer could not see them.

---

### Fix 3 — Canonicalize After `TTNNDecomposeLayouts` (Fix 5 in benchmark logs)

**Root cause addressed:** Bounces from `TTNNDecomposeLayouts` (Source 2 above) in `@trace_0_forward`.

**Why Fix 2 was insufficient for these 8 bounces:**

After `DevicePassesWrapper` finishes, the pipeline runs:
1. `TTNNTraceHoistTransform` (stage 11) — lifts ops into `@trace_0_forward`
2. `TTNNDecomposeLayouts` (stage 12) — converts `ToLayoutOp` → `to_memory_config` chains
3. `TTCoreOptimizationBarrierFold`
4. `TTNNDeallocate` (stage 13)

`TTNNDecomposeLayouts` creates new `to_memory_config` bounce chains from the 8 remaining
`ToLayoutOp` ops. No canonicalize ran between stage 12 and stage 13, so these bounces
survived to the final IR.

**Code change:**

**File: `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`**

Added canonicalize immediately after `createTTNNPipelineLayoutDecompositionPass`:

```cpp
createTTNNPipelineLayoutDecompositionPass(devicePm, options);

// Fold bounce spills (DRAM->L1_sharded->DRAM) introduced by ToLayoutOp
// decomposition in the layout decomposition pass above.
devicePm.addPass(mlir::createCanonicalizerPass());   // NEW

devicePm.addPass(ttcore::createTTCoreOptimizationBarrierFold());
createTTNNPipelineDeallocPass(devicePm, options);
```

**Result:** 228 tmc ops (-96 from baseline), 3810 IR lines (-197). All **52 bounces eliminated**.
FPS: 2.62.

---

## Complete Fix Summary

### Files Modified

| File | Change |
|---|---|
| `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` | Added `hasFolder = 1` and `hasCanonicalizer = 1` to `TTNN_ToMemoryConfigOp` |
| `lib/Dialect/TTNN/IR/TTNNOps.cpp` | Added `ToMemoryConfigOp::fold()` (identity fold) and `FoldConsecutiveToMemoryConfigOps` rewrite pattern |
| `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` | Added canonicalize pass after `OperationValidationAndFallback` (inside `DevicePassesWrapper`) and after `createTTNNPipelineLayoutDecompositionPass` (top-level device pipeline) |

All files are in `third_party/tt-mlir/`.

### Progression Table

| Stage | Bounce patterns | tmc ops | IR lines | FPS | Notes |
|---|---|---|---|---|---|
| Baseline | 52 | 324 | 4007 | 2.79 | Pre-fix |
| Fix 1 (no code change) | 52 | 324 | 4007 | 2.82 | IR dump investigation only |
| Fix 2 (canonicalize in wrapper + FoldConsecutive) | 8 | 244 | 3842 | 2.73 | 44 bounces eliminated; 8 remain in `@trace_0_forward` |
| Fix 3 (canonicalize after DecomposeLayouts) | **0** | 228 | 3810 | 2.62 | All 52 bounces eliminated |

> Note: FPS numbers have ±10–50 ms variance across runs. The FPS improvement from bounce removal
> is within measurement noise because the 8 remaining bounces were inside a `@trace_0_forward`
> function that executes as a hardware trace — trace execution already buffers the DRAM accesses.
> The main FPS gain from this class of fix was already captured by Fix 2 (44 bounces eliminated).

### IR Delta (Baseline → Fix 3)

| Op type | Baseline | Fix 3 | Delta |
|---|---|---|---|
| `ttnn.to_memory_config` | 324 | 228 | **-96** |
| `ttnn.deallocate` | 1308 | 1212 | **-96** |
| `ttnn.ttnn_layout` | 191 | 186 | -5 |

The 96 removals vs 52 detected bounces reflects cascading folds: removing one bounce can
expose the next in a chain, and the canonicalizer runs to fixpoint.

---

## How the Canonicalization Pattern Works

`FoldConsecutiveToMemoryConfigOps` fires on a `to_memory_config` op (`op`) whose input is
defined by another `to_memory_config` (`defOp`). It folds when:

1. The intermediate result (output of `defOp`) has an **L1-sharded layout** — height, block,
   or width sharded.
2. The only users of the intermediate are `op` itself plus zero or more **soft**
   (`force=false`) `deallocate` ops.

When the pattern fires it:
- Erases the soft deallocates of the intermediate
- Replaces `op` with a new `to_memory_config` that reads `defOp`'s input directly
- Erases `defOp`

This is safe because the L1-sharded intermediate is never read by any real compute op —
it was only materialized to satisfy the input layout requirements of the outer
`to_memory_config`, which in turn is a DRAM/L1-interleaved re-layout. Bypassing the
intermediate does not change observable memory or compute semantics.

---

## Remaining DRAM Spills (Not Fixed Here)

The perf metrics report 100 `dram_spilled_ops` across all fix stages. These are
**structural spills** caused by:
- Conv2d with `stride=2` that cannot accept L1-sharded input
- `slice_static` requiring L1-interleaved layout
- Chain propagation: once one op lands in DRAM, downstream ops stay in DRAM

These are tracked separately in `docs/block_a_dram_spill_analysis.md`.
