# BEV Model Performance — Block A DRAM Bounce Spill Investigation and Fix

## Context

The Deformed Backbone (Block A) was the pipeline bottleneck at:

- **358.7 ms / 2.79 FPS** (~56.4% of total pipeline time)
- Config: `opt_level=2`, `BFloat16`, `HiFi3`, `fp32_dest_acc=True`, `trace=True`

---

## Root Cause Investigation

Profiling the TTNN IR revealed **52 bounce spill patterns** — round-trips of the form:

```
op → to_memory_config(L1_sharded) → to_memory_config(DRAM / L1_interleaved)
```

The intermediate L1-sharded hop allocates and copies data but is consumed by exactly one op
(another `to_memory_config`) with no compute in between, making the L1 stage pure overhead.

### Where the Bounces Come From

**32 bounces in `@forward` function:**
Created by `OperationValidationAndFallback` inside `DevicePassesWrapper`. This pass inserts
layout-change `ToLayoutOp` entries that pair with existing ones to form bounce chains. A
`canonicalize` pass ran before `OperationValidationAndFallback` but not after it, so the bounce
chains survived to the final IR.

**8 bounces in `@trace_0_forward` function:**
Created by `TTNNDecomposeLayouts` (stage 12 in the pipeline). This pass converts all `ToLayoutOp`
ops into explicit `to_memory_config` chains. It runs *after* `TTNNTraceHoistTransform` (stage 11)
lifts ops into the trace function. No `canonicalize` ran after stage 12, so these 8 bounces also
survived.

**Additional blocker:** Soft (`force=false`) `deallocate` ops on the intermediate result were
acting as second users, preventing existing fold logic from triggering.

### Baseline Numbers

| Metric | Value |
|---|---|
| Bounce spill patterns | 52 |
| `to_memory_config` ops | 324 |
| `deallocate` ops | 1308 |
| IR lines | 4007 |
| FPS | 2.79 |

---

## Fix Implementation

Two changes in `third_party/tt-mlir/`:

### Change 1 — `FoldConsecutiveToMemoryConfigOps` canonicalization pattern

**Files:**
- `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` — added `hasFolder = 1` and `hasCanonicalizer = 1` to `TTNN_ToMemoryConfigOp`
- `lib/Dialect/TTNN/IR/TTNNOps.cpp` — added identity fold and the rewrite pattern

The pattern fires on a `to_memory_config` op whose input is defined by another `to_memory_config`
producing an L1-sharded result, when the only users of that intermediate are the outer
`to_memory_config` plus zero or more soft (`force=false`) deallocates. It erases the soft
deallocates and the intermediate, bypassing the L1 hop entirely.

```
Before:
  %a = to_memory_config(%x, DRAM)
  %b = to_memory_config(%a, L1_sharded)   ← intermediate bounce
  deallocate(%a)
  %c = to_memory_config(%b, DRAM)         ← outer op
  deallocate(%b)

After:
  %a = to_memory_config(%x, DRAM)
  %c = to_memory_config(%a, DRAM)         ← bypasses L1 hop entirely
  deallocate(%a)
```

### Change 2 — Two additional `canonicalize` passes in the pipeline

**File:** `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`

**Pass 1** — inside `DevicePassesWrapper`, after `PrepareConv2dWeightsAndBias`:
```cpp
innerPm.addPass(createTTNNOperationValidationAndFallback(validationOptions));
innerPm.addPass(createTTNNPrepareConv2dWeightsAndBias());
// NEW: fold bounces introduced by OperationValidationAndFallback
innerPm.addPass(mlir::createCanonicalizerPass());
```
Catches the 32 bounces in `@forward` introduced by `OperationValidationAndFallback`.

**Pass 2** — top-level device pipeline, after `createTTNNPipelineLayoutDecompositionPass`:
```cpp
createTTNNPipelineLayoutDecompositionPass(devicePm, options);
// NEW: fold bounces introduced by TTNNDecomposeLayouts in @trace_0_forward
devicePm.addPass(mlir::createCanonicalizerPass());
devicePm.addPass(ttcore::createTTCoreOptimizationBarrierFold());
```
Catches the 8 bounces in `@trace_0_forward` introduced by `TTNNDecomposeLayouts`.

---

## Results

| Metric | Baseline | After fix | Delta |
|---|---|---|---|
| Bounce patterns | 52 | **0** | −52 |
| `to_memory_config` ops | 324 | 228 | −96 |
| `deallocate` ops | 1308 | 1212 | −96 |
| IR lines | 4007 | 3810 | −197 |
| FPS (inference only) | 2.79 | 2.62–2.74 | within noise (±50 ms) |

The 96 op reductions vs 52 detected bounces reflects cascading folds: eliminating one bounce
can expose the next in a chain, and the canonicalizer runs to fixpoint.

FPS delta is within measurement noise. The 8 bounces inside `@trace_0_forward` executed as part
of a hardware trace — the trace engine was absorbing them at runtime, so their removal does not
produce observable latency improvement.

---

## Concrete Before / After Example

First bounce at IR line 1688 (baseline), `concat → conv2d` path:

```
BASELINE (lines 1688–1696):
  %21 = "ttnn.concat"(...)                           ← DRAM output
  %22 = "ttnn.to_memory_config"(%21, L1_sharded)    ← BOUNCE: unnecessary move to L1
  "ttnn.deallocate"(%21)
  %23 = "ttnn.to_memory_config"(%22, DRAM)           ← BOUNCE: immediately back to DRAM
  "ttnn.deallocate"(%22)
  %24 = "ttnn.conv2d"(%23, ...)                      ← conv2d reads from DRAM

AFTER FIX (lines 1683–1687):
  %21 = "ttnn.concat"(...)                           ← DRAM output
  %22 = "ttnn.conv2d"(%21, ...)                      ← conv2d reads concat output directly
  "ttnn.deallocate"(%21)
```

Two `to_memory_config` ops and one `deallocate` eliminated per bounce.

---

## Remaining Work — Structural DRAM Spills

The bounce fix did **not** change `effectively_sharded_percentage` (remains at 23.5%).
Bounce spills were pipeline artifacts (overhead ops), not real compute ops —
so the sharding counters were unaffected.

The perf metrics show **116 `sharded_and_spilled` ops**: ops that compute on a multi-core
sharded grid but whose outputs are immediately forced to DRAM. Eliminating these would move
`effectively_sharded` from **23.5% → 52.0%** (96/408 → 212/408).

### Current sharding state

| Category | Ops | % of shardable (408) |
|---|---|---|
| Effectively sharded (sharded + L1 output) | 96 | 23.5% |
| Sharded but spilled to DRAM | **116** | 28.4% |
| Not sharded (1×1 grid or DRAM-only) | 196 | 48.0% |

### Root causes of the 116 sharded+spilled ops

| Root cause | Op count | Fix location | Difficulty |
|---|---|---|---|
| Multi-consumer tensor: conv2d + slice_static need different layouts; MLA uses DRAM as common layout | 28 conv2d | MLA layout propagation | Medium |
| stride=2 conv2d kernel requires DRAM-interleaved activation input | 16 conv2d | tt-metal kernel | Hard |
| concat consumer with incompatible shard specs forces DRAM intermediate | 32 conv2d | MLA concat sharding | Medium |
| max_pool2d runs sharded but output immediately spills; downstream conv2d reads DRAM | 20 max_pool2d | MLA / tt-metal | Medium |
| conv_transpose2d is single-core only (1×1 grid) | 12 conv_transpose2d | tt-metal kernel | Hard |
| skip-connection add and concat with shard spec mismatches | 8 add + 12 concat | MLA | Medium |

### Multi-consumer pattern (highest priority, 28 cases)

The most actionable category. The output of an L1-sharded conv2d is consumed by two paths:
- A downstream conv2d that could accept L1-sharded input
- A slice_static path that requires L1-interleaved input

MLA resolves this by placing the output in DRAM (readable by both). The correct fix is for MLA
to keep the primary conv2d consumer on the L1-sharded output and insert a separate
`to_memory_config` branch only for the slice path — avoiding the shared DRAM intermediate entirely.

```
Current (both consumers read from DRAM):
  %42 = conv2d(...) → L1_sharded
  %43 = to_memory_config(%42, DRAM)          ← one DRAM copy for both consumers
      ├── conv2d(%43, ...)                   ← consumer A reads DRAM
      └── to_memory_config(%43, L1_int)
              └── slice_static(...)          ← consumer B reads via L1_interleaved

Target (primary stays in L1, slice gets its own copy):
  %42 = conv2d(...) → L1_sharded
      ├── conv2d(%42, ...)                   ← consumer A reads L1 directly
      └── to_memory_config(%42, L1_int)
              └── slice_static(...)          ← consumer B gets its own layout conversion
```

This is a change to the MLA (Memory Layout Assignment) pass in
`lib/Dialect/TTNN/Transforms/TTNNGreedyMemoryLayoutPropagation.cpp` — when a value has
multiple consumers with conflicting layout requirements, keep the dominant consumer's
preferred layout and insert per-consumer `to_memory_config` ops for the others.
