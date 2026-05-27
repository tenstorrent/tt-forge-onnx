# `grid_sample` Changes in tt-mlir

## Context

These changes add full support for `ttnn::grid_sample` through the tt-mlir compiler stack,
including layout transformations, precision-preserving workarounds, and correct op model
error signalling so the optimizer can skip the op gracefully.

---

## Files changed

| File | Change |
|---|---|
| `runtime/lib/ttnn/operations/pool/grid_sample.cpp` | Runtime rewrite with precomputed grid path |
| `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp` | Conditional BF16 workaround for grid operand |
| `include/ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h` | Updated factory method signature |
| `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` | Updated `extraClassDeclaration` |
| `lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp` | GridSampleOp returns `OpNotSupportedError` |

---

## 1. Runtime — `grid_sample.cpp`

**File:** `runtime/lib/ttnn/operations/pool/grid_sample.cpp`

### Problem

The previous runtime passed the grid directly to `ttnn::grid_sample` in all cases. This fails for:

1. **Nearest mode** — the tt-metal kernel requires a precomputed grid; there is no runtime nearest
   path without it.
2. **Bilinear + `align_corners=True`** — the kernel hardcodes the `align_corners=False` coordinate
   formula. Using it directly with `align_corners=True` produces PCC ~0.8 for large inputs.
3. **Nearest mode output** — the kernel emits HEIGHT_SHARDED L1 output with shard shape `(1, C)`,
   which is incompatible with the subsequent `permute` op (requires tile-aligned shards ≥ 32×32).

### Solution

```cpp
bool needsPrecomputedGrid = (mode == "nearest") || alignCorners;

if (needsPrecomputedGrid) {
    // 1. Move grid to host. Keep float32 — BF16 precision is insufficient for
    //    large grids (H=96, scale ~47.5 → ~0.38 error → wrong nearest-neighbor).
    ::ttnn::Tensor hostGrid = ::ttnn::from_device(grid);
    ::ttnn::Tensor hostGridF32 =
        (hostGrid.dtype() == ::ttnn::DataType::FLOAT32)
            ? hostGrid
            : ::ttnn::typecast(hostGrid, ::ttnn::DataType::FLOAT32);

    // 2. Precompute pixel coordinates + interpolation weights on the host CPU.
    //    Returns (N, H_out, W_out, 6) for bilinear or (N, H_out, W_out, 2) for nearest.
    ::ttnn::Tensor precomputedGrid =
        ::ttnn::prepare_grid_sample_grid(hostGridF32, inputShapeNHWC, mode,
                                         paddingMode, alignCorners,
                                         ::ttnn::DataType::BFLOAT16);

    // 3. Move precomputed grid to device DRAM.
    ::ttnn::MemoryConfig dramInterleaved{
        ::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM};
    ::ttnn::Tensor precomputedGridDevice =
        ::ttnn::to_device(precomputedGrid, &device, dramInterleaved);

    // 4. Run kernel with precomputed grid.
    ::ttnn::Tensor output =
        ::ttnn::grid_sample(input, precomputedGridDevice, mode, paddingMode,
                            alignCorners, /*use_precomputed_grid=*/true,
                            /*batch_output_channels=*/false, memoryConfig);

    // 5. Nearest mode produces HEIGHT_SHARDED L1 output (shard=(1,C)).
    //    Subsequent permute ops require tile-aligned shards. Collect to DRAM.
    if (output.memory_config().is_sharded()) {
        output = ::ttnn::to_memory_config(output, dramInterleaved);
    }
    tensorPool.insertTTNNTensorAndValidate(op->out(), output);
} else {
    // bilinear + align_corners=False: kernel formula is correct, use directly.
    ::ttnn::Tensor output =
        ::ttnn::grid_sample(input, grid, mode, paddingMode, alignCorners,
                            /*use_precomputed_grid=*/false,
                            /*batch_output_channels=*/false, memoryConfig);
    tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
```

---

## 2. Workarounds pass — `TTNNWorkaroundsPass.cpp`

**File:** `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp`

### Problem

The workarounds pass inserts `to_layout` / `typecast` ops before each kernel call to convert
tensors to the format the kernel expects. Previously it always converted the grid to BF16 +
ROW_MAJOR. For the precomputed path the grid must arrive at the runtime in **float32** because
`prepare_grid_sample_grid` requires float32 input — converting to BF16 here introduces coordinate
quantization error before the precomputation even happens.

### Solution

The factory method was changed from a no-arg function to one that takes `mlir::Operation *op`,
allowing it to inspect the op's `mode` and `align_corners` attributes at compile time:

```cpp
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createGridSampleOpOperandsWorkarounds(
    mlir::Operation *op) {

  auto gridSampleOp = mlir::cast<mlir::tt::ttnn::GridSampleOp>(op);
  std::string mode = gridSampleOp.getMode().str();
  bool alignCorners = gridSampleOp.getAlignCorners();

  // Precomputed path: nearest mode always, or bilinear + align_corners=True.
  bool usesPrecomputedGrid = (mode == "nearest") || alignCorners;

  // Input + output: always ROW_MAJOR + BF16 (kernel requirement).
  TTNNOperandWorkarounds rowMajorBF16;
  rowMajorBF16.tensorLayoutWorkaround = Layout::RowMajor;
  rowMajorBF16.tensorDataTypeWorkaround = ttcore::DataType::BFloat16;

  // Grid: always ROW_MAJOR. BF16 only on the direct path.
  // On the precomputed path we must preserve float32 so that
  // prepare_grid_sample_grid can operate at full precision.
  TTNNOperandWorkarounds gridWorkaround;
  gridWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  if (!usesPrecomputedGrid) {
    gridWorkaround.tensorDataTypeWorkaround = ttcore::DataType::BFloat16;
  }

  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(rowMajorBF16)    // input tensor
      .addInputOperandWorkaround(gridWorkaround)   // grid tensor
      .addOutputOperandWorkaround(rowMajorBF16);   // output tensor
}
```

### Operand summary

| Operand | Layout | dtype (direct) | dtype (precomputed) |
|---|---|---|---|
| input | ROW_MAJOR | BF16 | BF16 |
| grid | ROW_MAJOR | **BF16** | **float32** (no conversion) |
| output | ROW_MAJOR | BF16 | BF16 |

---

## 3. Workarounds pass header — `TTNNWorkaroundsPass.h`

**File:** `include/ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h`

The factory method declaration was updated to accept `mlir::Operation *op`:

```cpp
// Before:
static TTNNOperandsWorkarounds createGridSampleOpOperandsWorkarounds();

// After:
static TTNNOperandsWorkarounds createGridSampleOpOperandsWorkarounds(mlir::Operation *op);
```

---

## 4. TTNN op tablegen — `TTNNOps.td`

**File:** `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`

The `extraClassDeclaration` on `TTNN_GridSampleOp` was updated to forward `getOperation()` to the
factory method:

```tablegen
def TTNN_GridSampleOp : TTNN_Op<"grid_sample"> {
    // ...

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createGridSampleOpOperandsWorkarounds(
            getOperation());
      }
    }];
}
```

Before this change the workaround method took no arguments; the op had to pass `getOperation()`
for the factory to inspect mode/align_corners attributes.

---

## 5. OpModel interface — `TTNNOpModelInterface.cpp`

**File:** `lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp`

### Problem

The `OperationValidationAndFallback` pass queries the op model to determine whether an op is
feasible with a given tensor layout. When the query returns an error the pass checks:

```cpp
if (originalResult.isNotImplemented()) {
    return WalkResult::skip();    // Op model missing — skip this op, move on
}
// Otherwise: treat as MetalBackendError — FAIL compilation
```

`isNotImplemented()` returns `true` only when the error is an `OpNotSupportedError`. Previously
`GridSampleOp::getOpConstraints` called into `TTNNOpModel::getOpConstraints` which returned a
generic `llvm::createStringError(...)`. This was caught as `MetalBackendError`, causing
compilation to fail with:

```
error: OperationValidationAndFallback: Operation ttnn.grid_sample failed validation
       (original error: MetalBackendError - GridSampleOp op model not implemented)
```

This broke all BEV model tests (16 test cases).

### Solution

Replace the body with calls to `issueErrorForGetOpConstraints` / `issueErrorForGetOpRuntime`,
which create an `OpNotSupportedError` that the pass recognises as `isNotImplemented()`:

```cpp
// Before — cascaded through TTNNOpModel::getOpConstraints → createStringError → MetalBackendError
llvm::Expected<op_model::OpConstraints>
GridSampleOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  // ... called opConstraintsCache().getOrCompute(OpModel<GridSampleOp>::getOpConstraints, ...)
}

// After — returns OpNotSupportedError → isNotImplemented() == true → pass skips op
llvm::Expected<op_model::OpConstraints>
GridSampleOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return detail::issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
GridSampleOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}
```

### Why not fix `TTNNOpModel.cpp` directly?

`TTNNOpModel.cpp` is compiled into `libTTNNOpModelLib.a`, a static archive built with RTTI
restrictions. Instantiating `make_error<OpNotSupportedError>` there triggers:

```
ld.lld: error: undefined symbol: typeinfo for llvm::ErrorInfoBase
>>> referenced by TTNNOpModel.cpp
>>>   TTNNOpModel.cpp.o:(typeinfo for llvm::ErrorInfo<OpNotSupportedError, ErrorInfoBase>)
```

`TTNNOpModelInterface.cpp` does not have this restriction — it already uses `OpNotSupportedError`
via the `issueError<T>` template and compiles and links without issues.

---

---

## 6. Greedy optimizer — PermuteOp sharded layout fix

### Problem

At `opt_level_2`, the `TTNNGreedyMemoryLayoutPropagation` pass runs with `enableL1ShardingLayouts = true`. It assigns sharded L1 output layouts (BLOCK_SHARDED / HEIGHT_SHARDED / WIDTH_SHARDED) to ops in the graph to maximise on-chip tensor reuse. However, the metal TTNN `PermuteOp` kernel cannot handle sharded L1 input or output — it produces a corrupted HEIGHT_SHARDED non-contiguous result at runtime instead of a correctly permuted tensor.

Because `PermuteOp` sits between a convolution and `GridSampleOp` in the BEV graph, this caused the following 40 BEV GridSample tests to fail at `opt_level_2`:

```
test_bev_gridsample[cam{0-4}-gs{0-7}-opt_level_2]
```

The failure mode was PCC mismatch (or TT_THROW if the MLA-assigned sharded output was also sharded). The failures only appeared in batch execution (not solo) because in solo runs, zero-initialised device L1 memory coincidentally matched the expected all-zero GridSample output for out-of-bounds grid coordinates.

### Solution

**File:** `lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp`

In `LegalOpLayoutAnalysis::analysisImplementation()`, sharded L1 layouts are filtered out for `PermuteOp` after the initial candidate layout set is built. This prevents the greedy optimizer from ever selecting sharded output for `PermuteOp`, so it also never inserts a sharded `to_memory_config` before `PermuteOp`'s input.

```cpp
// PermuteOp: the metal permute kernel cannot handle sharded L1 input or
// output (BLOCK_SHARDED / HEIGHT_SHARDED / WIDTH_SHARDED). Removing sharded
// layouts here prevents the greedy optimizer from ever selecting them for
// PermuteOp, which avoids corrupted non-contiguous sharded output at runtime.
if (isa<PermuteOp>(op)) {
  analysisResult.erase(
      std::remove_if(analysisResult.begin(), analysisResult.end(),
                     [](const OpConfig &cfg) {
                       return cfg.outputLayout &&
                              cfg.outputLayout.hasShardedL1TensorMemoryLayout();
                     }),
      analysisResult.end());
}
```

**File:** `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`

`GridSampleOp` was also added to `enabledOpsForWorkaroundWithOptimizer` so that the `TTNNWorkaroundsPass` inserts `to_layout(ROW_MAJOR)` for its data and grid inputs even when `opt_level >= 1`. Without this, the workarounds pass skips ROW_MAJOR insertion for ops that are "optimizer-managed".

```cpp
const std::set<mlir::StringRef>
    TTNNWorkarounds::enabledOpsForWorkaroundWithOptimizer = {
        ttnn::WhereOp::getOperationName(),
        ttnn::FullOp::getOperationName(),
        ttnn::EmbeddingOp::getOperationName(),
        ttnn::ScatterOp::getOperationName(),
        ttnn::TopKOp::getOperationName(),
        // GridSampleOp requires ROW_MAJOR layout for both data and grid inputs.
        ttnn::GridSampleOp::getOperationName()};
```

### Pipeline context

At `opt_level >= 1`, `enableGreedyOptimizer = true` and the following pipeline is used (NOT `TTNNOptimizer`):

1. `TTNNWorkaroundsPass` — inserts `to_layout(ROW_MAJOR)` for `enabledOpsForWorkaroundWithOptimizer`
2. `TTNNGreedyMemoryLayoutPropagation` — uses `LegalOpLayoutAnalysis` per op
3. `TTNNGreedyL1SpillManagement` (opt_level_2 only)

The `LegalOpLayoutAnalysis` fix is applied during step 2, filtering sharded layouts before the beam search assigns them to `PermuteOp`.

---

## End-to-end compiler flow for `ttnn.grid_sample`

```
TTIR (NCHW input, N2HW grid)
  │
  │  TTIRToTTNN.cpp: GridSampleOpConversionPattern
  │  • permute input: NCHW → NHWC  [0,2,3,1]
  │  • permute grid:  N2HW → NHW2  [0,2,3,1]
  │  • insert ttnn.grid_sample (NHWC output)
  │  • permute output: NHWC → NCHW [0,3,1,2]
  ▼
TTNN (NHWC input, NHW2 grid, NHWC output wrapped in permutes)
  │
  │  TTNNWorkaroundsPass: createGridSampleOpOperandsWorkarounds(op)
  │  • input:  insert to_layout(ROW_MAJOR) + typecast(BF16)
  │  • grid:   insert to_layout(ROW_MAJOR)
  │            + typecast(BF16)  [direct path only]
  │  • output: insert to_layout(ROW_MAJOR) + typecast(BF16)
  ▼
TTNN with layout fixup ops
  │
  │  OperationValidationAndFallback
  │  • getOpConstraints() → OpNotSupportedError → isNotImplemented() → skip
  ▼
Flatbuffer serialization
  │
  ▼
Runtime: grid_sample.cpp
  • needsPrecomputedGrid = (mode=="nearest") || alignCorners
  • if precomputed: from_device → prepare_grid_sample_grid → to_device → grid_sample(use_precomputed=true)
  •                 + to_memory_config if sharded (nearest mode)
  • else:           grid_sample(use_precomputed=false)
```
