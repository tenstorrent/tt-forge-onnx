# BEV Block B — Fix Summary for tt-mlir

## Context

BEV (Bird's Eye View) model Block B exercises:
- `test_conv2d_encoder` — standard Conv2d stack
- `test_gridsample_single` — GridSample with K=1 (single camera, no batching)
- `test_gridsample_k8_batched` — GridSample with K=8 (8 cameras, batched LUT)
- `test_conv2d_bottleneck` — Conv2d bottleneck after GridSample
- `test_conv_transpose2d` — ConvTranspose2d upsampler

All tests run with `opt_level=2` (MLA + DFShardingPolicy + OperationValidation), `HiFi3`,
`fp32_dest_acc=True`, `trace=True`, `BFloat16`.

---

## Fix 1: GridSample — `batch_output_channels` for Batched K>1 Grids

**File:** `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

**Problem:**
The TTIR GridSample op's grid tensor has shape `[N, 2K, H, W]` where `K` = number of cameras.
For K>1 the tt-metal kernel requires `batch_output_channels=true` to correctly map the batched
LUT channels. Without this flag the output was silently wrong (wrong channel ordering).

**Fix:**
```cpp
// Detect batched grid: TTIR grid dim[1] = 2K; set batch_output_channels when K > 1.
int64_t K = gridType.getShape()[1] / 2;
bool batchOutputChannels = (K > 1);

rewriter.create<ttnn::GridSampleOp>(
    ...,
    rewriter.getBoolAttr(batchOutputChannels),
    ...);
```

**File:** `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`
```tablegen
DefaultValuedAttr<BoolAttr, "false">:$batch_output_channels,
```

---

## Fix 2: GridSample Runtime — Pass `nullopt` Memory Config for Precomputed Grid Path

**File:** `runtime/lib/ttnn/operations/pool/grid_sample.cpp`

**Problem:**
When using precomputed grid, the MLA-selected shard spec from the flatbuffer was passed as
`memory_config` to `ttnn::grid_sample`. This triggered the "user-provided shard spec" code path
in tt-metal's `compute_output_specs`, which produced an invalid `TensorSpec` with 512 shards on
64 cores (shape mismatch due to K>1 padding). The result was:
```
TT_FATAL @ tensor_spec.cpp:143: !shard_grid_fit_error.has_value()
```

**Fix:** Pass `memory_config=std::nullopt` for the precomputed grid path. The device op
auto-computes its own valid `HEIGHT_SHARDED` output layout. The output is immediately
desharded to DRAM after the op, so the shard spec has no observable effect downstream.

```cpp
// Was:
::ttnn::grid_sample(input, precomputedGridDevice, mode, paddingMode,
                    alignCorners, /*use_precomputed_grid=*/true,
                    /*batch_output_channels=*/false, memoryConfig);

// After:
::ttnn::grid_sample(input, precomputedGridDevice, mode, paddingMode,
                    alignCorners, /*use_precomputed_grid=*/true,
                    batchOutputChannels,
                    /*memory_config=*/std::nullopt);
```

---

## Fix 3: ConcatOp — Block Cross-Product Explosion (K=8 → 3^8 = 6561 Candidates)

**File:** `lib/Dialect/TTNN/Analysis/OpRules/DataMovementRules.cpp`

**Problem:**
`TTNNGreedyMemoryLayoutPropagation` generates input layout candidates for each operand and takes
their cross-product to enumerate all combinations. With K=8 concat inputs and ~3 candidates per
input (DRAM, L1 interleaved, L1 sharded), the cross-product is 3^8 = 6,561 combinations. Each
combination invoked `getOpConstraints` (tt-metal's GraphProcessor). This caused ~700s compilation.

**Fix:** `getInputLayoutFilter` returns `rejectAllSharded` and `shouldExploreReshards()` returns
`false`, reducing each operand to at most 2 candidates (DRAM, L1 interleaved) and eliminating
the N^K reshard cross-product entirely.

```cpp
LayoutFilterFn ConcatRuleBook::getInputLayoutFilter(unsigned) const {
  // Sharded concat inputs hang in tt-metal add_kernel (_M_copy) for K>=8.
  return layout_filter_utils::rejectAllSharded;
}

bool ConcatRuleBook::shouldExploreReshards() const {
  return false;  // N^K cross-product for K operands; always interleaved.
}
```

---

## Fix 4: ConcatOp — Bypass `getOpConstraints` (GraphProcessor JSON Hang)

**File:** `lib/Dialect/TTNN/Analysis/OpRules/DataMovementRules.cpp`
**File:** `include/ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h`
**File:** `lib/Dialect/TTNN/Analysis/MemoryLayoutPropagation.cpp`

**Problem:**
tt-metal's `GraphProcessor` accumulates a vertex list across all `getOpConstraints` calls within
the same Python process. After K=1 compilation, the list already contained thousands of vertices.
The K=8 `getOpConstraints` call on ConcatOp triggered `ScopedGraphCapture::~ScopedGraphCapture()`
which called `nlohmann::json::json_value::destroy()` recursively on the entire accumulated list.
On NFS with large memory, this recursive free hung indefinitely:

```
#0  _int_free / malloc_consolidate
#5  nlohmann::json::json_value::destroy
#6  ScopedGraphCapture::~ScopedGraphCapture
#7  GraphProcessor::get_graph_json
#8  ConcatOp::getOpConstraints  (via validateOperation)
```

**Fix:** Add `shouldSkipOpModelQuery()` to `OpRuleBook` base class and override in
`ConcatRuleBook` to return `true`. In `MemoryLayoutPropagation::evaluateHint()`, bypass
`validateOperation` entirely for these ops and synthesize a DRAM-interleaved result:

```cpp
// OpRuleBook.h
virtual bool shouldSkipOpModelQuery() const { return false; }

// DataMovementRules.cpp
bool ConcatRuleBook::shouldSkipOpModelQuery() const { return true; }

// MemoryLayoutPropagation.cpp
const bool skipOpModel = mlir::isa<ConcatOp>(op) ||
                         getRuleBook(op).shouldSkipOpModelQuery();
if (skipOpModel) {
  // Derive DRAM interleaved output from input shape; return synthetic success.
  TTNNLayoutAttr outLayout = TTNNLayoutAttr::Builder(inputLayouts[0], outputShape)
      .setBufferType(BufferType::DRAM)
      .setMemoryLayout(TensorMemoryLayout::Interleaved)
      .build();
  auto result = op_constraint_validation::ValidationResult::success(
      hintIdx, llvm::SmallVector<TTNNLayoutAttr>{outLayout});
  // ... build BeamCandidate and return it ...
  return candidate;
}
```

---

## Fix 5: ConcatOp — Guard Against Cross-Product Explosion (General)

**File:** `lib/Dialect/TTNN/Analysis/MemoryLayoutPropagation.cpp`

**Problem:**
Even with `rejectAllSharded`, a K=8 concat with 2 candidates per input has 2^8 = 256 combinations.
For other ops with many operands, the cross-product could be unbounded.

**Fix:** After `getInputCandidateSets()`, compute the total cross-product size. If it exceeds
`kMaxCrossProduct = 10000`, recompute a per-operand cap as the N-th root of the max:

```cpp
static constexpr size_t kMaxCrossProduct = 10000;
if (totalCrossProduct > kMaxCrossProduct) {
  size_t perOperandCap = std::max(1UL,
      (size_t)std::pow((double)kMaxCrossProduct,
                       1.0 / (double)numSets));
  for (auto &set : inputCandidateSets) {
    if (set.size() > perOperandCap) set.resize(perOperandCap);
  }
}
```

---

## Fix 6: ConcatOp — OperationValidationAndFallback Bypass

**File:** `lib/Dialect/TTNN/Transforms/OptimizerPasses/OperationValidationAndFallback.cpp`

**Problem:**
`DFShardingPolicy` (inside `TTNNOptimizer`) assigned a sharded layout to the ConcatOp output.
`OperationValidationAndFallback` then called `getOpConstraints` to validate that layout, hitting
the same GraphProcessor hang/crash path:
```
TT_FATAL @ tensor_spec.cpp:143: !shard_grid_fit_error.has_value()
```
This path is independent of the `MemoryLayoutPropagation` bypass.

**Fix:** In `OperationValidationAndFallback`, add an early skip for ops where
`getRuleBook(op).shouldSkipOpModelQuery()` is true. If the current output is sharded, demote it
to DRAM interleaved before returning. *(Pending as of last test run.)*

---

## Fix 7: ToMemoryConfigOp — Fold Round-Trips (L1→DRAM→L1, DRAM→L1→DRAM)

**File:** `lib/Dialect/TTNN/IR/TTNNOps.cpp`
**File:** `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`

**Problem:**
After `TTNNDecomposeLayouts` + `L1SpillManagement`, the IR contained patterns like:
```
%l1   = to_memory_config(%dram_src, #DRAM → HEIGHT_SHARDED L1)
%dram = to_memory_config(%l1, #L1 → DRAM)   // immediately spilled
```
These wasted DRAM transfers increased runtime and reduced performance.

**Fix:** Added `fold()` and `getCanonicalizationPatterns()` to `ToMemoryConfigOp`:
1. `foldIdentityToMemoryConfigOp` — same input/output layout → remove op entirely
2. `foldConsecutiveToMemoryConfigOp` — L1→DRAM→L1 → direct L1→L1 (or identity)
3. `foldDRAMtoL1toDRAMRoundTrip` — DRAM→L1→DRAM → direct DRAM→DRAM (or identity)

Removed up to 68 wasted DRAM transfers in Block A/B.

---

## Fix 8: Conv2d/ConvTranspose2d — Config Tensors in L1 for Small Weights

**File:** `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

**Problem:**
All conv config tensors were unconditionally allocated in DRAM
(`withConfigTensorsInDram(true)`). For small weights, this forced unnecessary DRAM reads,
hurting performance.

**Fix:** Use a 512 KB threshold: weights ≤ 512 KB stay in L1; larger weights go to DRAM
(to avoid L1 OOM).

```cpp
constexpr int64_t kL1WeightThresholdBytes = 512 * 1024;
auto conv2dConfigAttr = ttnn::Conv2dConfigAttr::get(ctx)
    .withConfigTensorsInDram(weightBytes > kL1WeightThresholdBytes);
```

---

## Fix 9: MaxPool2d / AvgPool2d — Allow ROW_MAJOR in LegalOpLayoutAnalysis

**File:** `lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp`
**File:** `lib/Dialect/TTNN/Analysis/L1InterleavedFallbackAnalysis.cpp`

**Problem:**
MaxPool2d and AvgPool2d always output ROW_MAJOR per the tt-metal kernel API. The legal layout
analysis did not generate ROW_MAJOR configs for them, so MLA could not assign L1 ROW_MAJOR
interleaved layouts.

**Fix:** `rowMajorAllowed = true` for these ops in `fillTTNNLayoutAttrs`. The L1 fallback
analysis skip for these ops was also removed (no longer needed).

---

## Fix 10: GridSample LUT Simplification (TTIRFusing)

**File:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`

**Problem:**
The BEV LUT preparation pattern produced K independent `slice_static → reshape` chains followed
by a K-input concat. For K=8 this was 8 slice + 8 reshape + 1 concat = 17 ops.

**Fix:** `GridSampleLutSimplify` rewrite pattern: detects the pattern and replaces the entire
K×(slice+reshape)+concat chain with a single `reshape`. Valid because the row-major layout of
`[.., K, d_{S+1}, ..]` is identical to `[.., K*d_{S+1}, ..]`.

See `docs/gridsample_fusion_before_after.md` for detailed before/after IR.

---

## Summary Table

| Fix | File | Issue | Status |
|-----|------|-------|--------|
| 1 | TTIRToTTNN.cpp | `batch_output_channels` for K>1 grids | ✅ Done |
| 2 | grid_sample.cpp runtime | Pass `nullopt` memcfg for precomputed path | ✅ Done |
| 3 | DataMovementRules.cpp | K=8 cross-product explosion (3^8 → 2^8) | ✅ Done |
| 4 | MemoryLayoutPropagation.cpp | GraphProcessor JSON hang bypass | ✅ Done |
| 5 | MemoryLayoutPropagation.cpp | General N^K cross-product guard | ✅ Done |
| 6 | OperationValidationAndFallback.cpp | DFShardingPolicy sharded concat crash | ⚠️ Pending |
| 7 | TTNNOps.cpp | L1→DRAM→L1 round-trip fold | ✅ Done |
| 8 | TTIRToTTNN.cpp | Conv config tensors L1 threshold | ✅ Done |
| 9 | LegalOpLayoutAnalysis.cpp | Pool ops ROW_MAJOR legal layouts | ✅ Done |
| 10 | TTIRFusing.cpp | GridSample LUT K-slice simplification | ✅ Done |
