# TTIR-to-TTNN Backend Pipeline: Exhaustive Pass Analysis

**Document scope:** Complete audit of every pass in `createTTIRToTTNNCommonPipeline`
as defined in
`third_party/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`.

**BEV context:** All measurements and recommendations assume `trace=True`,
`opt_level=2`, `HiFi3`, `fp32_dest_acc=True`, `Float16_b` dtype.

---

## Section 1: Pipeline Overview (Ordered Pass List)

The pipeline runs on the **Device module** nested inside the top-level
`ttcore::DeviceModuleOp`. The CPU module is touched only by CPU-hoisting passes.

All calls resolve via three helper functions:
`createTTNNPipelineTTIRPasses` (pre-lowering TTIR work),
`createTTNNPipelineLoweringPasses` (TTIR -> TTNN dialect switch),
and a set of TTNN-level passes.

### Phase 0 – Top-level module scaffolding (on root module)

| # | Pass | Function |
|---|------|----------|
| 0a | `TTCoreMarkFunctionsAsForwardPass` | Tag public functions as Device Forward |
| 0b | `TTCoreWrapDeviceModulePass` | Wrap IR in `ttcore.device_module` |
| 0c | `CPUHoistManuallyTaggedOpsTransform` | Hoist `ttir.should_hoist` ops to CPU |

### Phase 1 – TTIR Preparation (inside DeviceModule)

| # | Pass | Condition |
|---|------|-----------|
| 1a | `ElementTypeNormalization` | Always |
| 1b | `TTCoreRegisterDevicePass` | Always (sets up device descriptor) |
| 1c | `TTPopulateArgumentTypes` | Always |
| 1d | `CanonicalizerPass` | Always |
| 1e | `TTIRFusing` (first instance) | `enableFusing=true` |
| 1f | `TTIRQuantDequantConversion` | `enableQuantDequantConversion=true` |
| 1g | `TTIRToTTIRDecompositionPass` | Always |
| 1h | `CanonicalizerPass` | Always |
| 1i | `TTIRImplicitBroadcastFold` (first) | `implicitBroadcastFoldingEnabled=true` |
| 1j | `TTIRFusing` (second instance) | `enableFusing=true` |
| 1k | `CanonicalizerPass` | Always |
| 1l | `InlinerPass` | Always |
| 1m | `TTIRInferKVCacheArgumentTypes` | Always |
| 1n | `TTIRPropagateWeightDtype` | Always |
| 1o | `TTIRFlattenSlidingWindow` | Always |
| 1p | `TTIRExplicateTMs` + `TTIREraseInverseOps` | `eraseInverseOpsEnabled=true` |
| 1q | `TTIRImplicitBroadcastFold` (second) | `implicitBroadcastFoldingEnabled=true` |
| 1r | `TTIRFusing` (third instance) | `enableFusing=true` |
| 1s | `TTIRFoldFullToScalar` | Always |
| 1t | `TTIRQuantDataTypeConversionPass` | Always (controls quant bit-width) |
| 1u | `CSEPass` | Always |
| 1v | `ConstEvalHoistTransform` (first) | `enableConstEval=true` |

### Phase 2 – CPU Hoisting of Const-Eval (on root module)

| # | Pass |
|---|------|
| 2a | `CPUHoistConstEvalTransform` | `enableCPUHoistedConstEval=true` |

### Phase 3 – Lowering and TTNN Preparation (inside DeviceModule)

| # | Pass | Condition |
|---|------|-----------|
| 3a | `TTNNLayout` | Always |
| 3b | `ConvertTTIRToTTNNPass` | Always |
| 3c | `RemoveDeadValuesPass` | `removeDeadValuesEnabled=true` |
| 3d | `TTNNFusing` | `enableFusing=true` |
| 3e | `TTNNDecomposition` | Always |
| 3f | `TTNNMemoryManagement` | `dramSpaceSavingOptimizationEnabled=true` |
| 3g | `TTNNWorkarounds` + `Canonicalizer` + `CSE` | `!disableWorkarounds` |
| 3h | `TTNNWeightDtypeConversion` | Always (no-op if no annotations) |
| 3i | `TTNNKVCacheDtypeConversion` | `experimentalKVCacheDtype != None` |
| 3j | `TTNNSetComputeKernelConfig` | `fp32DestAccEn=true` or `mathFidelity != Undefined` |
| 3k | `TTNNCreateD2MSubgraphs` | `enableCreateD2MSubgraphs=true` |
| 3l | `ConstEvalHoistTransform` (second) | `enableConstEval=true` |

### Phase 4 – Memory Layout Analysis / Optimizer (inside DeviceModule)

**Default path (chain optimizer, `!enableGreedyOptimizer`):**

| # | Pass |
|---|------|
| 4a | `TTNNConfigureCCLOps` |
| 4b | `TTNNUniqueLocations` (optional) |
| 4c | `TTNNRowMajorLayoutPropagation` |
| 4d | `TTNNOptimizer` (chain-based MLA) |
| 4e | `CanonicalizerPass` |
| 4f | `TTNNOperationValidationAndFallback` |
| 4g | `TTNNPrepareConv2dWeightsAndBias` |

**Greedy path (`enableGreedyOptimizer=true`):**

| # | Pass |
|---|------|
| 4a | `TTNNConfigureCCLOps` |
| 4b | `TTNNUniqueLocations` (optional) |
| 4c | `TTNNRowMajorLayoutPropagation` |
| 4d | `TTNNGreedyMemoryLayoutPropagation` |
| 4e | `TTNNGreedyL1SpillManagement` (`memoryLayoutAnalysisEnabled=true`) |
| 4f | `CanonicalizerPass` |
| 4g | `TTNNOperationValidationAndFallback` |
| 4h | `CanonicalizerPass` (round-trip fold) |
| 4i | `TTNNPrepareConv2dWeightsAndBias` |

### Phase 5 – Post-MLA and Output (inside DeviceModule)

| # | Pass | Condition |
|---|------|-----------|
| 5a | `ConstEvalHoistTransform` (third) | `enableConstEval=true` |
| 5b | `TTNNConstEvalInputsToSystemMemory` + `Canonicalizer` | `enableConstEvalInputsToSystemMemory=true` |
| 5c | `TTNNTraceHoistTransform` | `enableTrace=true` |
| 5d | `TTNNDecomposeLayouts` | Always |
| 5e | `TTCoreOptimizationBarrierFold` | Always |
| 5f | `TTNNDeallocate` | Always |
| 5g | `TTNNCollectPerfMetrics` | `ttnnPerfMetricsEnabled=true` |

---

## Section 2: Per-Pass Deep Dive

### Pass 1a – ElementTypeNormalization

**File:** `lib/Dialect/TTIR/Transforms/ElementTypeNormalization.cpp`

**What it does:** Rewrites unsupported element types (`i64`, `f64`, `i8`, etc.)
to the nearest supported hardware type (typically `bf16` or `f32`). This is
mandatory because MLIR default integer promotions may produce 64-bit types that
have no hardware kernel equivalent on Tenstorrent silicon.

**When it runs:** First pass in the Device module, before any other
transformation. The pipeline comment explains this ordering: "canonicalization
patterns assume normalized types (e.g. no i64/f64) and may produce incorrect
results otherwise."

**Before/After:** `tensor<...xi64>` → `tensor<...xi32>` or `tensor<...xbf16>`.
`tensor<...xf64>` → `tensor<...xf32>`.

**Performance impact:** None directly, but downstream passes depend on legal
types.

**Known limitations:** Type narrowing from f64 to f32 loses precision. There is
no user override to keep f64 intermediates as f32 for high-precision subgraphs
— everything is normalized the same way.

**Optimization opportunity:** Selective normalization for accumulation buffers:
for BEV Block A, attention softmax denominators could benefit from staying at
f32 instead of bf16 even when the rest of the graph is bf16.

---

### Pass 1e / 1j / 1r – TTIRFusing (three instances)

**File:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`

**What it does:** Fuses consecutive TTIR operations into single ops to reduce
kernel dispatch overhead:
- `ConvAddBias` pattern: `conv2d(x, w) + bias` → `conv2d(x, w, bias)`. Works
  for Conv2d, ConvTranspose2d, and Conv3d. Also handles the case where the
  existing conv already has a bias: `conv2d(x, w, b1) + b2` → `conv2d(x, w, b1+b2)`.
- `PermuteMatmulFusion` (enabled by `enablePermuteMatmulFusion`): fuses a
  permute feeding a matmul into a single matmul op.
- `Conv2dWithMultiply` pattern (enabled by `enableFusingConv2dWithMultiplyPattern`).

**When it runs:** Three times — before decomposition, after decomposition
(when new fusable patterns may be created), and after inverse-op erasure. The
repeated application increases the chance of catching patterns that only appear
after earlier transforms expose them.

**BEV relevance:** BEV Block A and Block C are both conv-heavy. The
`ConvAddBias` fusion is significant: it avoids a separate elementwise-add
kernel dispatch after each conv, reducing the critical-path length.

**Known limitations:**
- Requires `convOp.getResult().hasOneUse()`: if the conv output feeds more than
  one consumer the fusion is skipped. BEV attention blocks sometimes use the
  same conv output for skip connections and the next layer simultaneously,
  preventing fusion.
- The conv-with-multiply pattern is behind a separate flag
  (`enableFusingConv2dWithMultiplyPattern`) and is not on by default.
- No fusion of back-to-back elementwise ops at the TTIR level (that is handled
  separately at TTNN by the D2M pipeline).

---

### Pass 1g – TTIRToTTIRDecompositionPass

**File:** `lib/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecompositionPass.cpp`

**What it does:** Lowers TTIR composite ops into combinations of primitive TTIR
ops that have direct TTNN/TTMetal kernel equivalents. In `DecompMode::TTNN`
mode (the BEV path), this decomposes:
- `IndexOp` → slice/gather sequence
- `GetDimensionSizeOp` → constant
- `DotGeneralOp` → matmul (with permutes as needed)
- `IndexSelectOp` → embedding or gather
- `ReduceAndOp` / `ReduceOrOp` → reduction trees
- `QuantizeOp` / `RequantizeOp` / `DequantizeOp` → typecast chains
- `ReverseOp` → concat/slice sequence
- Conv2d / ConvTranspose2d: if not already in NHWC format, wraps in permutes
  to produce NHWC input for the TTNN conv kernel

**Before/After example (conv2d non-NHWC):**
```
ttir.conv2d(%input_NCHW, %weight) -> %out_NCHW
```
becomes:
```
ttir.permute(%input_NCHW, [0,2,3,1]) -> %input_NHWC
ttir.conv2d(%input_NHWC, %weight)    -> %out_NHWC
ttir.permute(%out_NHWC,  [0,3,1,2]) -> %out_NCHW
```

**BEV relevance:** BEV Block A uses many Conv2d ops. If the framework passes
them in NCHW (the PyTorch default), this pass adds flanking permutes. Those
permutes become real TTNN ops (reshapes + transposes) that consume L1 bandwidth.

**Known limitations:** The surrounding permutes for non-NHWC convs are
necessary but expensive. If the upstream graph already produces NHWC layout
(possible with tt-forge-onnx transforms), these permutes would be
eliminated — but whether this happens depends on the ONNX exporter.

**Optimization opportunity:** An earlier TTIR pass that normalizes all
conv2d inputs to NHWC before they reach this decomposition could eliminate
the redundant permutes entirely. Alternatively, the TTNN conv2d op could
accept NCHW natively with an internal permute fused into the weight
pre-processing.

---

### Pass 1i / 1q – TTIRImplicitBroadcastFold

**File:** `lib/Dialect/TTIR/Transforms/Broadcast.cpp`

**What it does:** Walks the graph and folds explicit `ttir.broadcast` ops that
feed operations which support implicit broadcasting. When a broadcast feeds a
multiply, add, etc. that already handles broadcast semantics, the explicit
broadcast op is removed and the smaller operand is passed directly.

**Before/After:**
```
%0 = ttir.broadcast(%arg, ...) : (tensor<1x1x32xf32>, ...) -> tensor<1x16x32xf32>
%1 = ttir.multiply(%x, %0, %dps)
```
becomes:
```
%1 = ttir.multiply(%x, %arg, %dps)
```

**Performance impact:** Eliminates a full-tensor materialization for the
broadcast result. In BEV's attention blocks, query-key-value operations involve
repeated broadcasting of scale and position-embedding terms; folding these
removes several intermediate allocations.

**Known limitations:** Cannot fold broadcasts that feed ops with no implicit
broadcast support (e.g., reshape, some custom ops). A second pass runs after
eraseInverseOps to catch broadcasts exposed by TM commutation.

---

### Pass 1o – TTIRFlattenSlidingWindow

**File:** `lib/Dialect/TTIR/Transforms/FlattenSlidingWindow.cpp`

**What it does:** Reshapes the batch/height/width dimensions of Conv2d,
ConvTranspose2d, MaxPool2d, and AvgPool2d inputs into a single spatial
dimension before conversion to TTNN. The TTNN conv kernel requires a 4D tensor
in `[1, 1, N*H*W, C]` shape.

**Before:**
```
ttir.conv2d(%input: tensor<3x32x64x8xbf16>, ...)
                                          -> tensor<3x15x31x16xbf16>
```
**After:**
```
ttir.reshape(%input, [1, 1, 6144, 8])   -> tensor<1x1x6144x8xbf16>
ttir.conv2d(%reshaped, ...)              -> tensor<1x1x1395x16xbf16>
ttir.reshape(%out, [3, 15, 31, 16])     -> tensor<3x15x31x16xbf16>
```

The original shape is preserved in a `ttir.FlattenedCompatInfoAttr` for use
by later passes (in particular `TTIREraseInverseOps`).

**Performance impact:** The added reshapes are free (metadata-only for
contiguous tensors). The flattened shape enables TTNN's optimized
1D-conv kernels that are significantly faster than a general N-D path.

**Known limitations:** The `FlattenedCompatInfoAttr` is the trigger for
`TTIREraseInverseOps`; if the conv op does not get this attribute (edge case:
conv skipped for some reason), the erase-inverse pass won't run.

**BEV relevance:** Block A has ~20+ Conv2d ops; all of them go through this
flattening. The reshapes produced here are cheap but each one becomes a TTNN
op, adding to the op count. Combining with the ConvAddBias fusion reduces
overall dispatch count.

---

### Pass 1p – TTIRExplicateTMs + TTIREraseInverseOps

**Files:** `lib/Dialect/TTIR/Transforms/ExplicateTMs.cpp`,
`lib/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.cpp`

**What it does:** Together, these two passes eliminate redundant
tensor-manipulation pairs:
1. `TTIRExplicateTMs` makes implicit broadcasts explicit so they can
   participate in commutation analysis.
2. `TTIREraseInverseOps` finds pairs of TM ops that are mathematical inverses
   (e.g., `permute(A) → permute(A⁻¹)`, `reshape(S1) → reshape(S1)`) and
   removes both. It commutes TMs through elementwise ops and reductions to bring
   them adjacent.

The pass counts TMs before and after and verifies the count decreases; it only
runs when `FlattenedCompatInfoAttr` is present (indicating a sliding window
flattening added inverse permutes).

**Commutation patterns registered:** Broadcast commute, concat commute,
elementwise commute, RMSNorm commute, reduce commute, slice commute, softmax
commute.

**Performance impact:** High. The flattening pass adds a `reshape` before each
conv and a corresponding `reshape` after. Those two reshapes cancel each other
if they are adjacent, which they often become after the elementwise canonicalize
step. Without this pass, every conv in BEV Block A/C would carry two
superfluous reshape ops.

**Known limitations:**
- Guarded by `eraseInverseOpsEnabled` flag (default on).
- Only commutes through ops in the commutation-pattern whitelist. Novel
  composite ops (e.g., group-norm, custom attention) that are not in the list
  will block commutation and prevent cancellation.
- Does not commute through the SDPA (Scaled Dot-Product Attention) op, meaning
  the surrounding permutes in BEV's multi-head attention may not cancel.

---

### Pass 3a – TTNNLayout

**File:** `lib/Dialect/TTNN/Transforms/TTNNLayout.cpp`

**What it does:** Annotates every tensor in the IR with a `TTNNLayoutAttr`
encoding. This is the first pass to attach hardware-specific layout information.
The pass runs in two sub-phases:

1. **Type converter phase:** Applies `TTNNLayoutTensorTypeConverter` uniformly
   over all tensors. The default encoding is `DRAM interleaved tiled` on a
   `1x1` single-core grid. The global default buffer type is
   `g_defaultMemorySpaceDevice = BufferType::DRAM`.

2. **Rewriter phase:** Applies `TTNNLayoutRewriter` to every `TTIROp`. For each
   operand that does not already have the right layout, a `ttir::ToLayoutOp` is
   inserted to convert to `DRAM interleaved tiled`. The result type is updated
   to reflect the same. Special cases:
   - `ReshapeOp`: output tile-ness matches input (no implicit tilization change).
   - `Conv3dOp`: output forced to `ROW_MAJOR` (experimental op).
   - `MeshShardOp` with non-identity shard type: forced to `SystemMemory`.
   - Function input arguments typed as `ArgumentType::Input` (activations): set
     to `ROW_MAJOR` (for later propagation by `TTNNRowMajorLayoutPropagation`).
   - Conv2d weight arguments: forced to `SystemMemory` (to be
     const-eval-prepared off-device).

**Before/After:** A plain `tensor<64x128xbf16>` becomes
`tensor<64x128xbf16, #ttnn_layout<..., DRAM, interleaved, tiled>>`.

**Performance impact:** This pass sets conservative defaults; the optimizer
passes that come later will override these defaults with better sharding
decisions. However, the defaults ensure compilation always succeeds even without
the optimizer.

**Known limitations:**
- Default is `1x1 grid` (single-core). Without the optimizer, every tensor
  lands on a single core in DRAM, which is highly suboptimal for large tensors.
- The pass does not consult op models or live-range information — it has no
  knowledge of whether a tensor could fit in L1.

---

### Pass 3b – ConvertTTIRToTTNNPass

**File:** `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

**What it does:** The main dialect conversion pass. Replaces every TTIR op with
its TTNN equivalent using conversion patterns:
- `ttir::EmptyOp` → `ttnn::EmptyOp` (device) or `ttnn::ZerosOp` (host/system)
- `ttir::ToLayoutOp` → `ttnn::ToLayoutOp`
- Arithmetic, reduction, and nn ops: `ttir.add` → `ttnn.add`, `ttir.conv2d` →
  `ttnn.conv2d`, etc.
- Shape manipulation: `ttir.reshape` → `ttnn.reshape`, `ttir.permute` →
  `ttnn.permute`
- Memory ops: various `to_device`, `from_device` patterns

The conversion carries `TTNNLayoutAttr` through from the type annotations added
by `TTNNLayout`. The `ttnn::EmptyOp` path inspects `layoutAttr.getBufferType()`
to decide whether to emit a device empty (with memory config) or a host zeros.

**BEV relevance:** This is the boundary where TTIR ops become concrete TTNN ops
with attached memory configs. After this point the graph is all TTNN dialect.

**Known limitations:**
- ToLayoutOp conversion does not yet fully inline into adjacent op memory
  configs — that is left to `TTNNDecomposeLayouts` later.
- Some TTIR ops without TTNN equivalents must have been decomposed by the TTIR
  passes; if any slip through, this pass will fail with an illegal op error.

---

### Pass 3d – TTNNFusing

**File:** `lib/Dialect/TTNN/Transforms/TTNNFusing.cpp`

**What it does:** Fuses TTNN ops into higher-level fused kernels:

1. **Conv2d + activation** (`TTNNConv2dWithActivation`): Fuses an activation op
   (relu, gelu, silu, etc.) that immediately follows a Conv2d into the
   `conv2d_config.activation` field. The activation op is erased.
   Requires `conv.result.hasOneUse()`. Also handles the reshape-between-conv-
   and-activation case introduced by `TTIRFlattenSlidingWindow`.

2. **Matmul/Linear + activation** (`TTNNMatmulAndLinearWithActivation`):
   Sets `activation` field on matmul/linear ops.

3. **SDPA fusing** (`SDPAFusingPattern`, `TTMLIR_ENABLE_OPMODEL` required).

4. **RoPE fusing** (`RoPEFusingPattern`): Fuses rotary positional embedding
   sequence into `ttnn.rotary_embedding`.

5. **SplitQKV fusing** (`SplitQKVFusingPatterns`): Merges split Q/K/V linear
   projections.

6. **TopK fusing** (`TopKFusingPattern`): Merges top-k with a sort pattern.

7. **NLPConcatHeadsDecode fusing** (`NLPConcatHeadsDecodeFusing`): Merges
   `permute([1,2,0,3]) + reshape` into `nlp_concat_heads_decode` for LLM
   decode phase. Guarded by tile-alignment check (headDim % 32 == 0).

When `enableOpConstraints=true` (requires `TTMLIR_ENABLE_OPMODEL`), fused ops
are validated against hardware constraints in an isolated module before being
applied. If validation fails, the pattern is not applied.

**BEV relevance:** BEV Block A uses conv2d + relu heavily. The `TTNNConv2dWithActivation`
fusion eliminates a relu dispatch per conv layer. In Block A with ~20 conv ops,
this can save 20 kernel launches. BEV does not use LLM-style attention, so
SDPA/RoPE/SplitQKV fusing patterns are not relevant.

**Known limitations:**
- The activation fusion requires a single use of the conv output. BEV skip
  connections (residual adds consuming conv output and another path) will block
  this fusion.
- `NLPConcatHeadsDecodeFusing` has the tile-alignment guard (`headDim % 32`,
  `batchSize % 32`). Non-aligned BEV attention heads would miss this fusion.
- No elementwise chain fusion at this level (that requires D2M path or a
  dedicated elementwise fusion pass).

---

### Pass 3e – TTNNDecomposition

**File:** `lib/Dialect/TTNN/Transforms/Decomposition/TTNNDecompositionPass.cpp`

**What it does:** Decomposes TTNN composite ops that lack hardware kernel
support into primitive TTNN ops. The current patterns:
- `DistributedRMSNormDecompositionRewritePattern`: Decomposes
  `ttnn.distributed_rms_norm` into a sequence of `ttnn.mul`,
  `ttnn.reduce_scatter`, `ttnn.rms_norm`, `ttnn.all_gather` ops for
  multi-device setups.
- `DistributedLayerNormDecompositionRewritePattern`: Similar for distributed
  layer norm.

**BEV relevance:** BEV runs on single-device. Distributed norm decompositions
are no-ops for BEV. However, the pass still walks the module on every
compilation.

**Optimization opportunity:** Gate the walk with a multi-device check to avoid
unnecessary traversal on single-device models.

---

### Pass 3f – TTNNMemoryManagement (conditional)

**File:** `lib/Dialect/TTNN/Transforms/TTNNMemoryManagement.cpp`

**What it does:** Moves `slice` ops to reduce peak memory usage and reorganizes
`repeat` ops. Specifically: if a value's only consumers are `slice` ops, the
pass rearranges the computation to defer allocation until after slicing. Also
handles dimension-group mapping for repeat-then-slice patterns.

**Condition:** `dramSpaceSavingOptimizationEnabled=true` (controlled by
`opt_level`). At opt_level_2, this is typically enabled.

**BEV relevance:** BEV uses slice ops extensively in deformable convolution
attention sampling. This pass may help reduce DRAM allocation when large
feature maps are immediately sliced.

---

### Pass 3g – TTNNWorkarounds

**File:** `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`

**What it does:** A large collection of hardware workarounds that modify op
configurations or insert additional ops to work around runtime bugs, missing
kernel support, or hardware constraints. Workarounds are organized into two
categories:

**Layout workarounds** (gated by `layoutWorkaroundsEnabled`):
- `Conv2dRewritePattern`: Ensures conv2d has proper activation, weight dtype,
  and memory config set. Checks `optimizationLevel` — at opt_level_2 more
  aggressive configs are attempted.
- `Conv2dEnableKernelStrideFoldingRewritePattern`: Enables kernel stride folding
  optimization for certain conv2d configs.
- `UpsampleOpRewritePattern`: Fixes upsample memory layout.
- `EmbeddingOpSqueezeWeightRewritePattern`: Squeezes the weight tensor for
  embedding ops.
- `LinearOpOutputShapeRewritePattern` and `LinearOpRewritePattern`: Fix linear
  op output shapes and convert linear to matmul+bias in some cases.
- `GroupNormAffineReshapeRewritePattern`: Reshapes affine parameters for
  group_norm.
- `RMSNormConfigRewritePattern`: Sets RMSNorm memory config.
- `DistributedRMSNormWidthShardInputRewritePattern`: Handles width-shard input
  for distributed RMSNorm.
- Various SDPA workarounds (decode attention sink, broadcast mask, pad tile
  dims).

**Decomposition workarounds** (gated by `decompositionWorkaroundsEnabled`):
- `SliceStaticOpRewritePattern`: Decomposes certain static slice configurations.
- `PadHighDimRewritePattern`: Decomposes high-dimensional pad ops.
- `ScatterOpRewritePattern`: Decomposes scatter into supported primitives.
- `AllGatherOpRewritePattern`, `ReduceScatterOpRewritePattern`,
  `ReduceScatterConfigRewritePattern`: Handle multi-device collective ops.
- `PagedUpdateCacheOpRewritePattern`: Handles paged attention cache updates.
- `ArgMaxOpDimRewritePattern`: Fixes argmax when `dim` attribute is missing.
- `TopKRouterGptDecompositionRewritePattern`: Decomposes TopK router pattern.

After the workaround rewrites, `Canonicalizer` and `CSE` are run to clean up
any redundant ops introduced.

**BEV relevance:** The Conv2d workarounds (`Conv2dRewritePattern`,
`Conv2dEnableKernelStrideFoldingRewritePattern`) directly affect BEV Blocks A
and C. The `optimizationLevel=2` path enables stride-folding which can
significantly reduce the number of multiply-accumulate cycles for strided
convolutions.

**Known limitations:**
- Workarounds are hardware-specific and tied to the current tt-metal release.
  A workaround that was needed for a bug may be retained even after the bug is
  fixed upstream, silently adding overhead.
- No auditing mechanism exists to detect which workarounds are actually
  exercised for a given model. The `TTMLIR_DUMP_PIPELINE_IR` flag dumps IR but
  does not annotate which workarounds fired.

---

### Pass 3h – TTNNWeightDtypeConversion

**File:** `lib/Dialect/TTNN/Transforms/TTNNWeightDtypeConversion.cpp`

**What it does:** Inserts `ttnn.typecast` ops before matmul/linear/sparse_matmul
operations to convert weight tensors from bf16/f32 to a lower-precision BFP
format (bfp_bf8 or bfp_bf4). Only operates on weights that trace back to
constant or parameter function arguments (checked via
`ttcore::valueTracesToConstantArgs`).

Priority resolution:
1. Per-op `ttcore.weight_dtype` annotation (propagated by
   `TTIRPropagateWeightDtype`) takes precedence.
2. Falls back to the global `targetDtype` pass option.
3. Is a complete no-op when neither source provides a dtype.

**Performance impact:** Reducing weight dtype from bf16 to bfp_bf8 cuts matmul
operand memory bandwidth in half, potentially improving throughput on
bandwidth-bound matmul kernels.

**BEV relevance:** BEV Block A has multi-head attention with Q/K/V projections
(linear layers). If `experimentalWeightDtype=bfp_bf8` is set, these projection
weights will be quantized. However, BEV uses `Float16_b` throughout; mixing
bfp_bf8 with Float16_b requires careful accuracy verification.

**Known limitations:** Only applies to matmul/linear families. Conv2d weights
are not handled here (they are prepared via `TTNNPrepareConv2dWeightsAndBias`
which runs later inside the optimizer wrapper).

---

### Pass 3j – TTNNSetComputeKernelConfig

**File:** `lib/Dialect/TTNN/Transforms/TTNNSetComputeKernelConfig.cpp`

**What it does:** Walks all ops implementing `TTNNComputeKernelConfigOpInterface`
and sets default `DeviceComputeKernelConfigAttr` fields that are not already
set. The pass applies `withX()` setters with the following merge logic:
- If the field is already set (e.g., by a prior pass), it is preserved.
- If the field is `std::nullopt` (unset), the pass-level override is applied.

Fields that can be set: `math_fidelity`, `math_approx_mode`,
`fp32_dest_acc_en`, `packer_l1_acc`, `dst_full_sync_en`.

**Condition:** Only added to the pipeline if `fp32DestAccEn=true` OR
`mathFidelity != Undefined`.

**BEV relevance:** BEV uses `fp32_dest_acc=True` and `HiFi3`. This pass is
responsible for stamping those values on every eligible op (matmul, linear,
conv2d, etc.). Without this pass, ops would use their default compute config
which may be `HiFi4` (more accurate but slower) or `LoFi` (faster but less
accurate).

**Impact of HiFi3 vs HiFi4:** HiFi3 trades some matrix multiplication
precision for ~15% throughput improvement. For BEV detection accuracy this
is typically safe; for LLMs it may cause instability.

**Known limitations:** The pass runs with a module-wide walk, setting the same
config on every op. There is no mechanism to set different fidelity levels for
different ops (e.g., HiFi4 for the final classification head, HiFi3 for the
backbone). Per-op override would require the `overrideOutputLayout` mechanism
to also carry compute config.

---

### Pass 4c – TTNNRowMajorLayoutPropagation

**File:** `lib/Dialect/TTNN/Transforms/OptimizerPasses/RowMajorLayoutPropagation.cpp`

**What it does:** Propagates ROW_MAJOR layout from function input arguments
downstream through the dataflow graph, removing unnecessary `ToLayoutOp`s that
convert from ROW_MAJOR to TILE immediately on input. The pass:
1. Identifies `Input`-type block arguments (not KV cache, not mesh-shard).
2. Finds redundant `ttir::ToLayoutOp` chains that immediately tilize these
   inputs.
3. Bypasses those ToLayoutOps (`bypassRedundantToLayoutOps`).
4. Propagates the ROW_MAJOR constraint through subsequent operations until
   hitting an op that requires TILE (matmul, conv, etc.).
5. Sets `opLayoutConstraints` on those downstream ops.

This pass runs before the optimizer so it provides better initial layout hints
that reduce unnecessary tilize/untilize round-trips.

**Requires:** `TTMLIR_ENABLE_OPMODEL` (uses op constraint validation).

**BEV relevance:** BEV activation tensors arrive as ROW_MAJOR (from the host).
Without this pass, each input tensor would be immediately tilized by the default
`TTNNLayout` assignment, consuming L1 budget for a tilize kernel. With this
pass, the tilize is deferred to the point where it is actually required.

---

### Pass 4d – TTNNOptimizer (chain-based, default path)

**File:** `lib/Dialect/TTNN/Transforms/OptimizerPasses/Optimizer.cpp`

**What it does:** The legacy Memory Layout Analysis (MLA) optimizer. Performs
a chain-based dataflow analysis to determine optimal tensor layouts (L1 sharded
vs DRAM interleaved) and assigns them to each op result. Steps:
1. `ScalarDataTypeAnalysis`: collects all scalar types in the graph.
2. `LegalTensorLayoutAnalysis`: generates valid sharding configurations for
   each tensor type (up to `maxLegalLayouts=64` candidates per tensor).
3. `LegalOpLayoutAnalysis`: filters per-op, per-operand legal layout sets using
   `OpModel` constraint validation.
4. `MemoryLayoutAnalysis`: runs a graph-level solver (`ShardSolver` or
   `DFSharding` policy) to find a globally consistent L1 sharding assignment.
5. Applies the selected layouts by modifying op result types and inserting
   `ToLayoutOp`s for cross-op layout transitions (memory reconfig ops).
6. Optionally inserts `L1InterleavedFallbackAnalysis` as a fallback when
   sharding is infeasible.

**Options:**
- `memoryLayoutAnalysisEnabled=true` (required for L1 sharding; false = DRAM
  only).
- `memoryLayoutAnalysisPolicy`: `DFSharding` (dataflow) or `BFS`.
- `maxLegalLayouts=64`: caps the candidate count per tensor.
- `memReconfigEnabled`: enables insertion of memory-reconfig ops.
- `insertMemReconfig`: manual overrides for specific op operands.
- `overrideOutputLayout`: per-op layout overrides by location string.
- `overrideConv2dConfig`: per-op conv2d config overrides.

**Performance impact:** This is the most impactful pass in the pipeline for
throughput. Assigning L1 sharding to the right ops can deliver 3-5x speedup
compared to DRAM-only operation. The key decision is: which ops can have their
outputs live in L1 sharded form without overflowing L1 budget?

**Known limitations:**
- Chain-based solver is O(N²) or worse on large op sequences. Large BEV models
  (Block A, 326ms runtime, many ops) can have long compile times.
- The `ShardSolver` is greedy and may not find the globally optimal assignment.
- Conv2d configs (parallelization strategy: 1x1 vs NxM grid) must be set
  before this pass via `overrideConv2dConfig` if the auto-config is wrong.
- The optimizer does not model the trace capture overhead, which can be
  significant when many layout transitions cross the trace boundary.

**BEV note:** At `opt_level=2`, `memoryLayoutAnalysisEnabled=true`.
This pass is responsible for the L1 sharding decisions seen in the
BEV IR dumps (e.g., `height_sharded` on the conv2d output tensors in Block C).

---

### Pass 4d (greedy path) – TTNNGreedyMemoryLayoutPropagation

**File:** `lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyMemoryLayoutPropagation.cpp`

**What it does:** A beam-search replacement for the chain-based
`TTNNOptimizer`. Uses edge-based propagation with configurable beam width
(default `beamWidth=8`). Steps:
1. `ScalarDataTypeAnalysis`.
2. `LegalTensorLayoutAnalysis` (same as optimizer).
3. Applies `applyConvSliceConfig` first (sets `L1Full` slice config on Conv2d ops).
4. Per-op `LegalOpLayoutAnalysis` using `OpModel` constraint validation.
5. Greedy or beam-search propagation: for each op, keeps the `beamWidth` best
   candidate output layouts and propagates forward.
6. Applies decisions, inserting `ToLayoutOp` (→ `ToMemoryConfigOp` after
   decompose) for transitions.

**Options beyond optimizer:** `beamWidth=8`, `maxInputCandidatesPerOperand=64`,
`maxReshardCandidatesPerType=4`, `enableDecisionTrace`, `decisionTraceDir`,
`enableCompileTimeStats`.

**When used:** Only when `enableGreedyOptimizer=true` pipeline option is set.
This is a newer, experimental path.

**Performance advantages over chain optimizer:**
- Beam search explores more combinations per op.
- Better handles long chains where early decisions constrain later ops.
- `decisionTrace` output enables post-compilation layout analysis.

**BEV relevance:** If the greedy optimizer is enabled for BEV, the
`decisionTrace` JSON output is available at `ttrt-artifacts/decision_trace/`
and provides per-op layout decisions for debugging.

---

### Pass 4e (greedy path) – TTNNGreedyL1SpillManagement

**File:** `lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyL1SpillManagement.cpp`

**What it does:** After `TTNNGreedyMemoryLayoutPropagation` may have assigned
many tensors to L1, this pass enforces the L1 memory budget using Belady's
optimal page replacement algorithm. Steps:
1. Reads L1 budget: `chipDesc.getUsableL1Size() * tensorL1UsageCap`.
2. For each forward function, creates a `L1SpillManagement<SumL1MemoryTracker>`
   instance.
3. Runs Belady's algorithm: iterates the op sequence and spills the L1 tensor
   with the furthest next use when L1 budget would be exceeded.
4. Spilling inserts `ToLayoutOp` from L1 to DRAM at the spill point.
5. After the pass, `d2m_optimizer_utils::syncAllD2MFuncTypes` updates D2M
   subgraph function types to reflect changed operand layouts.

**Only runs when:** `memoryLayoutAnalysisEnabled=true`.

**Performance impact:** This pass is critical for BEV correctness on large
models. Without it, the greedy propagation may assign more L1 than physically
available, causing runtime OOM errors. With it, spills are inserted at points
where the data is least immediately needed.

**Known limitation:** Belady's algorithm requires future-use information which
is computed from a static sequential scan. Dynamic branching (not present in
BEV but present in LLMs with KV cache) cannot be handled exactly.

---

### Pass 4f / 4g – TTNNOperationValidationAndFallback

**File:** `lib/Dialect/TTNN/Transforms/OptimizerPasses/OperationValidationAndFallback.cpp`

**What it does:** Validates each TTNN op against hardware constraints using
`OpModel`. For ops that fail validation, applies fallback strategies by
transforming input layouts to find a working configuration. The pass:
1. Walks all forward-device ops.
2. For each op, calls `op_constraint_validation::validateOperation` with the
   current input layouts.
3. If validation fails, generates `createFallbackTransforms` — a set of
   candidate layout changes (buffer type changes, memory layout changes, dtype
   changes) ranked by `cost` distance.
4. Tries up to `maxFallbackAttempts=10000` combinations.
5. On success, inserts `ToLayoutOp` ops to transform inputs to the valid
   configuration.
6. On failure (no valid config found), emits an error.

**Cost model:**
- `NO_COST (0.0)`: no change
- `LOW_COST (1.0)`: minor layout change (e.g., DRAM→DRAM different tiling)
- `MID_COST (2.0)`: moderate change (e.g., L1→DRAM)
- `HIGH_COST (3.0)`: major change (e.g., L1 sharded → DRAM)

**BEV relevance:** This pass is the safety net that catches incorrect layout
assignments from the optimizer. For BEV Block A, if the optimizer assigns L1
height-sharded to a conv2d that requires a different sharding for correctness,
this pass finds the fallback config.

**Known limitations:** The fallback search is combinatorial (up to 10000
attempts). For ops with many operands (e.g., conv2d with input + weight + bias
+ memory_config) the search space can be very large and may not converge to
the optimal solution.

---

### Pass 4i – TTNNPrepareConv2dWeightsAndBias

**File:** `lib/Dialect/TTNN/Transforms/OptimizerPasses/TTNNPrepareConv2dWeightsAndBias.cpp`

**What it does:** Inserts `ttnn.prepare_conv2d_weights` and
`ttnn.prepare_conv2d_bias` ops before each `ttnn.conv2d`, and
`ttnn.prepare_conv_transpose2d_weights` / `ttnn.prepare_conv_transpose2d_bias`
before each `ttnn.conv_transpose2d`. The prepared ops preprocess weights
(transposing, packing, tiling) off-line into the exact memory format the TTNN
conv kernel expects. These ops are then const-evaluated in a subsequent
`ConstEvalHoistTransform` pass.

**Requires:** `TTMLIR_ENABLE_OPMODEL`. The pass calls
`op_model::getPreparedConv2dWeightsOutputTensor` to determine the exact output
tensor type for the prepared weights.

**Performance impact:** Weight preprocessing at compile time rather than
runtime. For BEV Block A and C, this can save several milliseconds of startup
time on first-inference runs. On repeated inference with trace=True, the
weights are loaded from const-eval results and the prepare ops are not
re-executed.

**Known limitations:** Requires the OpModel library. If building without
`TTMLIR_ENABLE_OPMODEL`, this pass is skipped and conv2d must preprocess
weights at runtime.

---

### Pass 5b – TTNNConstEvalInputsToSystemMemory

**File:** `lib/Dialect/TTNN/Transforms/TTNNConstEvalInputsToSystemMemory.cpp`

**What it does:** Forces all inputs of const-eval functions to `SystemMemory`
(host). Rewrites:
1. The const-eval function signature: input tensor types changed to
   `SystemMemory + RowMajor`.
2. The forward function arguments corresponding to const-eval inputs: also
   forced to `SystemMemory`.
3. Inserts `ToLayoutOp` (to_device) inside the const-eval function after the
   system-memory argument, to bring data back to device for actual computation.

The pass only operates on arguments whose sole user is a `ttcore.load_cached`
op (to avoid unnecessary transfers for tensors with other consumers).

**Performance impact:** Const-eval inputs (model weights) are kept on host
memory rather than being transferred to device memory on every call. This
reduces DRAM allocation at program start.

**BEV relevance:** BEV has many const-eval weight tensors (all conv kernels,
attention weights). This pass ensures that they are loaded from CPU memory
directly into device memory via the const-eval cache path, not double-buffered
through DRAM.

---

### Pass 5c – TTNNTraceHoistTransform

**File:** `lib/Dialect/TTNN/Transforms/TTNNTraceHoistTransform.cpp`

**What it does:** Splits the forward function into multiple functions to support
TTNN's metal trace infrastructure:

1. **`trace_N_funcname`**: Contains the traceable ops (all device ops that can
   be replayed without host involvement). Excluded ops: `ReturnOp`,
   `LoadCachedOp`, `CaptureOrExecuteTraceOp`, `GetDeviceOp`, `MeshShardOp`,
   `TTCoreCreationOp` traits.

2. **`run_and_capture_trace_N_funcname`**: Sets up pre-allocated trace input
   tensors. For activation inputs (non-const, non-parameter):
   - Allocates an `ttnn.empty` tensor as a scratch buffer.
   - Calls `ttnn.from_device` to bring the activation to host.
   - Calls `ttnn.write_tensor` to copy host data into the pre-allocated device
     buffer.
   - Then runs the trace function twice: once to warm up, once with
     `begin_trace_capture` / `end_trace_capture`.
   - Returns trace_id, warmup outputs, and the pre-allocated input tensors.

   For constant/parameter inputs (`shouldKeepArgOnDevice`): passed through
   directly without copying.

3. **`execute_trace_N_funcname`**: Re-uses the captured trace to replay ops
   with `ttnn.execute_trace`.

4. **`ttnn.capture_or_execute_trace`** op: Conditionally calls the capture
   function on first call and the execute function on subsequent calls.

**When it runs:** After all layout decisions are finalized (post-optimizer),
before `TTNNDecomposeLayouts`. This ordering is critical: trace hoisting works
at the abstract `ToLayoutOp` level; after `DecomposeLayouts` the ops would be
individual `to_device/from_device/to_memory_config` ops that are much harder
to analyze.

**Performance impact:** Trace mode is the primary performance feature for BEV.
With trace=True, all ~326ms of Block A's device ops are captured once and
replayed each frame with minimal host overhead. The trace mechanism eliminates
the latency of dispatching hundreds of individual TTNN ops per frame.

**Known limitations:**
- Ops excluded from trace (LoadCachedOp, GetDeviceOp, MeshShardOp) create
  "holes" in the trace that require host re-entry. If these ops appear in the
  middle of a long chain, they split the traceable region into multiple shorter
  traces.
- `write_tensor` for each activation input adds a host-to-device copy per
  input per frame. For BEV, if the input feature map is large (e.g., full
  camera frame), this copy can be significant.
- The pass uses `static std::atomic<uint64_t> traceFunctionIndex` — a global
  counter. In parallel compilation contexts this is thread-safe, but function
  names may have non-contiguous indices.
- Trace does not support dynamic shapes. All traced ops must have static shapes,
  which is satisfied for BEV.

**BEV measurement context:** The 326ms Block A latency is measured WITH trace.
The cost breakdown is approximately: trace-warmup on first frame (amortized),
then ~1-2ms per frame for `write_tensor` + `execute_trace`. The actual compute
is captured in the trace.

---

### Pass 5d – TTNNDecomposeLayouts

**File:** `lib/Dialect/TTNN/Transforms/TTNNDecomposeLayouts.cpp`

**What it does:** Expands all `ttnn::ToLayoutOp` instances into concrete memory
management ops. This is the final pass that materializes abstract layout
transitions into specific hardware operations.

The pass walks all `ttnn::ToLayoutOp`s and for each one determines, using
`determineRequiredOps`, which combination of ops is needed:

- `createToDeviceOp`: when input is on host (SystemMemory) and output is on
  device (DRAM or L1). Becomes `ttnn.to_device` op.
- `createFromDeviceOp`: when input is on device and output is on host. Becomes
  `ttnn.from_device` op.
- `createToLayoutOp`: when tile format changes (TILE ↔ ROW_MAJOR). Becomes
  `ttnn.to_layout` op (tilize or untilize).
- `createDataTypeCastOp`: when element dtype changes. Becomes `ttnn.typecast`.
- `createToMemoryConfigOp`: when buffer type or sharding changes while staying
  on device (DRAM→L1, L1→DRAM, or reshard between L1 configs). Becomes
  `ttnn.to_memory_config` op.

Reshard triggers:
- `input.bufferType == DRAM && output.bufferType == L1`: loads from DRAM into
  L1 sharded.
- `input.bufferType == L1 && output.bufferType == DRAM`: spills from L1 to DRAM.
- `input.isL1Sharded() && output.isL1Sharded() && (grid differs || CRS differs)`:
  reshard between different L1 configurations.

**Type support:** `canTilizeDataTypeOnDevice`: bf16, f32, uint32, uint16, int32.
`canUntilizeDataTypeOnDevice`: bf16, f32, uint32, int32.

**Performance impact:** This pass determines how many actual `to_memory_config`
ops appear in the final IR. Each such op is a real runtime cost. The
`TTMLIR_DUMP_MEMORY_OPS` diagnostic (checkpoint 4 and 5 in the pipeline) can
be used to count L1→DRAM spills, DRAM→L1 loads, and DRAM→L1→DRAM round-trips.

**Known limitations:**
- The pass has no global view of concurrent L1 occupancy. It decomposes each
  `ToLayoutOp` independently. Two adjacent `to_memory_config` ops that could
  be combined (e.g., reshard + dtype change) remain as two separate ops.
- The canonicalizer that was intended to run after this pass to eliminate
  DRAM→L1→DRAM round-trips has been disabled for the trace path (see comment
  in pipeline: "Post-decompose canonicalization disabled"). This means any
  round-trips introduced by OPVF restores after `DecomposeLayouts` remain in
  the IR.

---

### Pass 5e – TTCoreOptimizationBarrierFold

**File:** `lib/Dialect/TTCore/Transforms/`

**What it does:** Folds `ttcore.optimization_barrier` ops. These are
placeholder ops inserted by earlier passes to prevent reordering across a
critical boundary. After all layout decisions are made and trace hoisting is
done, these barriers are no longer needed and can be removed.

---

### Pass 5f – TTNNDeallocate

**File:** `lib/Dialect/TTNN/Transforms/` (defined in `Passes.td`)

**What it does:** Inserts `ttnn.deallocate` ops after each tensor value's last
use. This allows the runtime to reclaim device memory (L1 or DRAM) promptly
rather than holding it until function return.

**Performance impact:** Correct placement of deallocates is essential for
fitting large models in L1 and DRAM. Without deallocates, peak memory usage
equals total memory of all live tensors, which quickly exceeds device DRAM.

**Known limitations:** The pass operates on SSA liveness but cannot account
for in-place ops or aliasing. Aggressive deallocation may cause correctness
issues in rare aliasing scenarios.

---

## Section 3: Pass Interaction Effects

### Interaction 1: TTIRFlattenSlidingWindow + TTIREraseInverseOps

`TTIRFlattenSlidingWindow` adds a `reshape` before and after each conv2d,
storing the original shape in `FlattenedCompatInfoAttr`. This attribute is the
gating condition for `TTIREraseInverseOps`. The erase-inverse pass then looks
for the pattern:
```
reshape([N,H,W,C] → [1,1,N*H*W,C]) ... conv2d ... reshape([1,1,N*H*W,C] → [N,H,W,C])
```
and eliminates the reshape pair when they are adjacent and cancel out.

**Net effect:** The two passes together have zero overhead if the reshapes
cancel — which they do for all simple conv sequences. They leave useful reshapes
when the convolution output is consumed by multiple ops that prevent the
inverse pattern from forming.

**BEV implication:** In BEV Block A, some conv outputs are used both for the
next layer and for a residual skip connection. In this case the first reshape
cannot be cancelled because it has multiple users, resulting in one extra
reshape op per such conv. This is inherent to the architecture, not a compiler
bug.

### Interaction 2: TTNNLayout + TTNNOptimizer/GreedyMLA

`TTNNLayout` assigns conservative `DRAM interleaved tiled` defaults to all
tensors on a `1x1` grid. The optimizer then overrides these with L1 sharded
configs where feasible. The layout pass creates `ToLayoutOp` placeholders
for all needed conversions; the optimizer replaces the attributes; and
`TTNNDecomposeLayouts` finally materializes the ops.

If the optimizer does not change a tensor's layout, the ToLayoutOp from
`TTNNLayout` may become a no-op (same buffer type / same layout). In that case
`TTNNDecomposeLayouts` would fail with "Redundant ttnn::ToLayoutOp" — this is
prevented by the canonicalizer passes that fold identity ToLayoutOps before
decompose runs.

### Interaction 3: TTNNTraceHoistTransform + TTNNDecomposeLayouts

`TTNNTraceHoistTransform` must run **before** `TTNNDecomposeLayouts`. The trace
pass reorganizes functions at the `ToLayoutOp` abstraction level. After
decompose, the ops are `to_device/from_device/to_memory_config` — a much more
fine-grained representation where the trace boundary logic (which ops can be
inside a trace?) would need to understand each op individually.

The pipeline comment explicitly documents this: "Trace hoisting must run before
layout decomposition because it adjusts layouts of function arguments
(e.g. moving inputs to system_memory). It is much easier to work at the layout
abstraction level than on individual ops after they have been decomposed."

**BEV implication:** The trace boundary for BEV is set after the MLA optimizer
has determined all sharding decisions. The trace function therefore contains
ops with optimal L1 layouts. If the MLA optimizer is suboptimal for some ops,
those suboptimal layouts are baked into the trace and cannot be changed without
recompilation.

### Interaction 4: ConstEvalHoistTransform (three passes) + TTNNPrepareConv2dWeightsAndBias

The pipeline runs `ConstEvalHoistTransform` three times:
1. **First** (pass 1v): hoists subgraphs computable entirely from constants.
2. **Second** (pass 3l): picks up new const-eval-able ops created by
   `TTNNWorkarounds` and `TTNNWeightDtypeConversion` (e.g., typecast chains on
   constant weights).
3. **Third** (pass 5a): picks up the `PrepareConv2dWeights` ops inserted by
   the optimizer.

The third hoisting is necessary because `PrepareConv2dWeightsAndBias` runs
inside the `DevicePassesWrapper` (with the device open), but const-eval hoisting
needs the device closed. The pipeline works around this by running the third
hoist outside the wrapper.

**Performance implication for BEV:** Conv weight preprocessing (the expensive
shuffle/pack operations) is hoisted to CPU execution at compile time. On first
inference, the CPU precomputes the packed weights and stores them. On subsequent
calls with `trace=True`, the weights are already in the correct format and
loaded via the const-eval cache. This is a significant contributor to BEV's low
per-frame latency.

### Interaction 5: TTNNGreedyL1SpillManagement + TTNNDecomposeLayouts

`TTNNGreedyL1SpillManagement` inserts `ToLayoutOp` ops (L1 → DRAM) at spill
points. These appear in the IR as abstract layout ops. `TTNNDecomposeLayouts`
then materializes them as `to_memory_config` ops.

The `TTMLIR_DUMP_MEMORY_OPS` checkpoints 1 (after-L1Spill), 2 (after-Canon1),
3 (after-Canon2), 4 (before-DecomposeLayouts), and 5 (after-DecomposeLayouts)
provide a step-by-step view of how spills accumulate and transform. This
diagnostic was specifically added to BEV debugging work.

### Interaction 6: TTNNFusing + conv2d activation + TTNNWorkarounds

`TTNNFusing` fuses activation into conv2d by setting `conv2d_config.activation`.
`TTNNWorkarounds` then runs `Conv2dRewritePattern` which may re-examine and
adjust the conv2d config. If the activation is set by fusing and then the
workaround tries to set a different activation config, there is a potential for
conflict. The workaround pattern checks `srcOp.getConv2dConfig()->hasActivation()`
and skips re-setting it, so they are compatible.

---

## Section 4: Missing Passes / Optimization Gaps

### Gap 1: No global dead-code elimination pass after TTNNWorkarounds

The `CSEPass` after `TTNNWorkarounds` eliminates common subexpressions but not
dead code (ops whose results are never used). The `RemoveDeadValuesPass` is
gated on `removeDeadValuesEnabled`. If workarounds create new ops that replace
existing ones but leave the old ones unreachable, they remain in the IR until
the next canonicalization. For large models like BEV Block A, this inflates IR
size and compile time.

### Gap 2: No elementwise chain fusion pass at TTNN level (non-D2M path)

`TTNNFusing` fuses activation into matmul/conv but does not fuse chains of
elementwise ops (e.g., `add → mul → relu → add`). The D2M pipeline handles
this but only when `enableCreateD2MSubgraphs=true`. For the standard TTNN path,
a chain of 4 elementwise ops generates 4 separate kernel dispatches. A simple
pattern-based fuser (similar to XLA's elemental fusion) could reduce this to a
single fused kernel.

### Gap 3: No layout-aware op splitting pass

When an op's output tensor is too large for L1 sharding (e.g., a 512-channel
conv with a 160x160 feature map on Wormhole), the optimizer falls back to DRAM.
A pass that splits ops spatially (tiling along the spatial dimension) could
enable L1 computation for ops that are currently DRAM-bound. This is analogous
to loop tiling in traditional compilers.

### Gap 4: No cross-block memory reuse analysis

BEV runs Blocks A, B, C, D as separate traces. At each block boundary,
outputs are transferred host→device (written via `write_tensor`). A pass
that analyzes block boundaries and keeps inter-block tensors in device DRAM
(avoiding the round-trip through host) could reduce the boundary overhead.

### Gap 5: No stride-merge optimization for convolution chains

In BEV Block A, a strided conv followed by a regular conv could be fused by
absorbing the stride of the first conv into the second (adjusting the kernel
layout). This would reduce the total number of passes over the feature map
from 2 to 1 and roughly halve the memory bandwidth. This requires a TTIR-level
op pattern (before flattening) that recognizes strided-conv + conv sequences.

### Gap 6: No automatic operator parallelism selection for multi-chip

For multi-chip BEV deployment, the pipeline does not automatically distribute
the block A/C conv workload across chips. The CCL configure pass sets topology
but the MLA optimizer does not use mesh-topology information when choosing
sharding strategies. An inter-chip sharding pass could parallelize the heavy
conv2d blocks across devices.

### Gap 7: No compile-time profiling feedback loop

The `TTNNCollectPerfMetrics` pass collects JSON metrics after compilation.
These metrics (sharded ops count, spilled ops count) are not fed back into the
optimizer to improve future compilations. A profile-guided optimization pass
could use these metrics to adjust `tensorL1UsageCap` or the beam width of
`TTNNGreedyMemoryLayoutPropagation`.

### Gap 8: No fused conv2d+BatchNorm (BN folding) at TTIR level

BEV models trained with batch normalization are typically exported with folded
BN (absorbed into conv weights). However, if BN is not folded before tt-forge-
onnx translation, it appears as a separate elementwise pass after each conv.
A TTIR-level BN-folding pass would merge `conv2d + multiply + add` (BN scale
and shift) into a single fused conv2d with modified weights. This is standard
practice in TFLite/OpenVINO.

---

## Section 5: Priority Improvements for BEV Performance

The following improvements are ranked by estimated impact on BEV Block A
and C latency.

### Priority 1: Reduce DRAM→L1→DRAM round-trips (High Impact)

**Current state:** The post-`DecomposeLayouts` canonicalization is disabled for
trace workloads (commented out in pipeline). The `TTMLIR_DUMP_MEMORY_OPS`
diagnostic shows the count of round-trips at checkpoint 5 (after-DecomposeLayouts).
Any non-zero count represents unnecessary memory bandwidth.

**What to do:** Re-enable the post-decompose canonicalization pass (or add a
new `TTNNFoldRedundantMemConfig` pass) that pattern-matches
`to_memory_config(DRAM→L1) → to_memory_config(L1→DRAM)` pairs introduced by
OPVF restores and folds them into no-ops. This requires verifying that the
pair introduces no type changes between the two ops.

**Files to modify:** `TTNNPipelines.cpp` (re-enable disabled canonicalizer),
possibly add a new dedicated folding pass in
`lib/Dialect/TTNN/Transforms/`.

**Estimated impact:** 2-5% runtime reduction depending on how many round-trips
exist in Block A/C.

### Priority 2: NHWC layout normalization before TTIRToTTIRDecomposition (High Impact)

**Current state:** BEV Block A conv2d ops arrive in NCHW format.
`TTIRToTTIRDecompositionPass` wraps each conv in `permute` ops.
`TTIREraseInverseOps` cancels adjacent pairs but leaves residual permutes at
skip-connection boundaries.

**What to do:** Add a TTIR pass (before `TTIRFlattenSlidingWindow`) that
rewrites the entire graph to use NHWC as the canonical data format. This would
transpose all function inputs and outputs once, then commute the transpositions
through the graph so that all conv2d ops see NHWC input natively. The net
result: zero flanking permutes for conv2d ops, and all skip connections operate
in NHWC without extra transposes.

**Files to modify:** New file
`lib/Dialect/TTIR/Transforms/NHWCLayoutNormalization.cpp`.

**Estimated impact:** Elimination of ~2-4 permute ops per conv layer in Block
A (20+ conv layers → 40-80 fewer ops). Each permute is a memory-bandwidth-
bound op. Estimated 5-10% reduction in Block A/C latency.

### Priority 3: Fuse consecutive elementwise ops in TTNN (Medium-High Impact)

**Current state:** After `ConvertTTIRToTTNNPass`, chains like `add → mul →
relu` dispatch three separate kernels. The D2M path fuses them but requires
`enableCreateD2MSubgraphs=True`.

**What to do:** Add a simple `TTNNElementwiseFusion` pass (between
`TTNNFusing` and `TTNNDecomposition`) that combines chains of elementwise ops
on the same tensor into a single `ttnn.unary_chain` or equivalent. Alternatively,
enable D2M subgraph creation for all models (`enableCreateD2MSubgraphs=True`
as a default at opt_level_2).

**Files to modify:** New pattern in `TTNNFusing.cpp` or
enable `TTNNCreateD2MSubgraphs` by default.

**Estimated impact:** BEV attention blocks (Block A) have several elementwise
post-processing chains. Estimated 3-7% reduction depending on elementwise chain
length.

### Priority 4: Improve TTNNGreedyL1SpillManagement (Medium Impact)

**Current state:** Belady's algorithm uses `SumL1MemoryTracker` which sums all
L1 tensors and spills when the sum exceeds budget. It does not account for
per-core alignment waste or inter-op memory-config transition costs.

**What to do:**
1. Add a more accurate `AlignedL1MemoryTracker` that accounts for tile alignment
   overhead (tensors must be aligned to 32-element tile boundaries, which can
   waste 15-30% of allocated space).
2. Add a cost model for `to_memory_config` ops to the Belady decision: if
   spilling saves less than the cost of the spill+reload ops, don't spill.

**Files to modify:**
`lib/Dialect/TTNN/Analysis/L1SpillManagement.h/cpp`.

**Estimated impact:** Fewer unnecessary spills → fewer `to_memory_config` ops
→ 2-5% improvement in memory-intensive blocks.

### Priority 5: Conv2d weight dtype compression for BEV (Medium Impact)

**Current state:** `TTNNWeightDtypeConversion` is not used for BEV (BEV uses
the default Float16_b without per-arg `weight_dtype` annotations).

**What to do:** Enable `experimentalWeightDtype=bfp_bf8` for BEV matmul
operations in Block A attention layers and verify accuracy. bfp_bf8 weights
are 2x smaller than bf16, reducing weight loading bandwidth. For attention with
large head dimensions, this could be significant.

**Verification required:** Run BEV detection accuracy benchmarks with bfp_bf8
weights to confirm <0.1% mAP drop.

**Estimated impact:** 5-10% throughput improvement for attention projection ops
in Block A if accuracy is acceptable.

### Priority 6: Add `packer_l1_acc` for large reductions (Low-Medium Impact)

**Current state:** `TTNNSetComputeKernelConfig` sets `fp32_dest_acc=True` and
`HiFi3` globally. `packer_l1_acc` (pack directly to L1 after computation) is
not set.

**What to do:** Identify reduce/matmul ops where the output is immediately
consumed by the next op (no cross-op lifetime overlap) and set `packer_l1_acc=
True` for those ops. This allows the packer to write directly to L1 without
going through DRAM, reducing one round-trip per applicable op.

**Files to modify:** `TTNNSetComputeKernelConfig.cpp` — add a per-op heuristic
that checks output liveness before setting this flag.

**Estimated impact:** 1-3% improvement for ops where packer-L1 is safe.

### Priority 7: Beam width tuning for TTNNGreedyMemoryLayoutPropagation (Low-Medium)

**Current state:** `beamWidth=8` is hardcoded in the pipeline for the greedy
optimizer path. Beam width directly controls the quality-vs-compile-time
tradeoff: higher beam width finds better layouts but compiles slower.

**What to do:** Expose `beamWidth` as a pipeline option and profile BEV
compilation with different values (1=greedy, 4, 8, 16). For BEV, the heavy
conv blocks in A and C likely benefit from a higher beam width, while the
light blocks (D/E/F) can use beam=1.

**Files to modify:** `TTNNPipelines.cpp` — expose the option. Profile harness
to measure compile time vs runtime quality.

**Estimated impact:** Potentially 3-8% improvement in Block A/C layout quality
at the cost of longer compile time (acceptable for a model that is compiled
once and deployed thousands of times).

---

*Document generated by compiler analysis pass audit, 2026-05-19.*
*Pipeline source: `third_party/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`*
*Pass declarations: `third_party/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms/Passes.td`*
