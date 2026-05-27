# Native GridSample Op — Full Stack Implementation

## 1. Affected Test Cases

All BEV blocks at all opt_levels failed or hung when grid_sample was present.

- `test_opt_sweep[opt_level_2-block_B]` — compilation hang (MLA timeout on 14,480 TTIR ops)
- `test_opt_sweep[opt_level_2-block_D]` — compilation hang
- All `test_bev_block_d_gridsample.py` configs: 0/120 passing at opt_level_2

## 2. Failure

No explicit error — the compiler hung indefinitely during the Memory Layout Analysis (MLA) pass at opt_level_2. At opt_level_0/1, the model ran but produced wrong results because the decomposed primitives were not equivalent to the native op.

## 3. Failure Reason

`grid_sample` had no native `TTIR_GridSampleOp` or `TTNN_GridSampleOp` in the tt-mlir compiler. The TVM relay pass `DecomposeGridSample` was decomposing each `image.grid_sample` node into approximately 362 primitive TTIR ops (gather, interpolate, pad, etc.). The BEV model has approximately 40 `grid_sample` ops. This produced:

- ~14,480 TTIR ops from grid_sample alone out of 15,769 total
- MLA must solve a layout assignment problem for every op. At opt_level_2 with sharding enabled, this combinatorial explosion caused the pass to run indefinitely.

Additionally, `tt-metal` already has a native `ttnn::grid_sample` kernel that handles bilinear and nearest interpolation directly on device. The decomposed path could not match this kernel's performance or precision.

**Key format difference the implementation must handle:**
- TVM relay outputs grid as `(N, 2, H_out, W_out)` — coordinate dimension first
- `tt-metal grid_sample` expects input as `NHWC (N, H_in, W_in, C)` and grid as `(N, H_out, W_out, 2)` — coordinate dimension last
- TTIR uses `NCHW (N, C, H_in, W_in)` for inputs
- So TTIR→TTNN conversion must permute both input and grid

## 4. Fix Implementation Details

Added full native support for `grid_sample` through all compiler layers:

**Layer 1 — tt-forge-onnx (Python/C++ frontend)**
- Remove `DecomposeGridSample` relay pass so the op passes through natively
- Add `GridSample` op type to the Forge op enum and all dispatch tables
- Implement `grid_sample::eval`, `shape`, `backward` C++ functions
- Wire TVM relay `image.grid_sample` → `forge.op.GridSample` via `tvm_to_python.py`
- Emit `ttir::GridSampleOp` in `lower_to_mlir.cpp`

**Layer 2 — tt-mlir TTIR dialect**
- `TTIR_GridSampleOp` tablegen: inputs `(N,C,H_in,W_in)` and grid `(N,2,H_out,W_out)`, attributes `mode`, `padding_mode`, `align_corners`
- Verifier checks: 4D tensors, grid dim[1]=2, valid mode/padding strings, batch dimension match

**Layer 3 — tt-mlir TTNN dialect**
- `TTNN_GridSampleOp` tablegen: inputs `(N,H_in,W_in,C)` NHWC and grid `(N,H_out,W_out,2)`, plus `memory_config`
- Verifier checks same constraints but grid dim[3]=2 (last dim convention)
- `getOperandsWorkarounds`: input and grid forced to ROW_MAJOR; grid dtype is float32 for precomputed path (nearest or align_corners=True), BF16 for direct bilinear path

**Layer 4 — TTIR→TTNN conversion**
- Insert `ttnn::PermuteOp [0,2,3,1]` on input (NCHW→NHWC)
- Insert `ttnn::PermuteOp [0,2,3,1]` on grid (N2HW→NHW2)
- Create `ttnn::GridSampleOp` with NHWC shapes
- Insert `ttnn::PermuteOp [0,3,1,2]` on output (NHWC→NCHW, inverse permutation)

**Layer 5 — LegalOpLayoutAnalysis**
- Filter all sharded L1 layouts from PermuteOp candidates. The NCHW↔NHWC permutes inserted by the GridSample conversion are WH-involving permutations that fail with sharded inputs.

**Layer 6 — EmitC and EmitPy lowerings**
- Map TTNN op to `ttnn::grid_sample(input, grid, mode, padding_mode, align_corners, use_precomputed_grid=false, batch_output_channels=false, memory_config)`

**Layer 7 — Flatbuffer serialization**
- Add `GridSampleOp` table in `pool.fbs`
- Register in `OpType` union in `program.fbs`
- Add `createOp` and `emitTTNNOperation` dispatch in `TTNNToFlatbuffer.cpp`

**Layer 8 — Runtime**
- New `grid_sample.cpp`: two execution paths:
  - **Direct bilinear path** (align_corners=False): pass grid directly to `ttnn::grid_sample`
  - **Precomputed path** (nearest mode or align_corners=True): move grid to host CPU as float32, call `ttnn::prepare_grid_sample_grid` to precompute coordinates, move result back to device, call `ttnn::grid_sample` with `use_precomputed_grid=true`
  - After nearest mode, collect sharded output to DRAM interleaved (nearest mode produces HEIGHT_SHARDED L1 with shard shape (1,C) incompatible with subsequent permute)

**Layer 9 — OpModel stub**
- GridSampleOp returns `OpNotSupportedError` (MissingMetalDefinition) — the op is excluded from MLA, which falls back to DRAM interleaved for this op

**Layer 10 — Tools**
- `ttir_builder.py`: `grid_sample` builder method
- `mapping.py`: `grid_sample_golden` function mapping TTIR golden to `torch.nn.functional.grid_sample`
- `ttnn-precompiled.hpp`: include `grid_sample.hpp`

## 5. Files Changed with Diffs

### REPO: tt-forge-onnx

**`forge/forge/op/__init__.py`**
```diff
-from .resize import Resize1d, Resize2d, Upsample2d, Downsample2d
+from .resize import Resize1d, Resize2d, Upsample2d, Downsample2d, GridSample
```

**`forge/forge/op/resize.py`**
```diff
+def GridSample(
+    name: str,
+    operandA: Tensor,
+    operandB: Tensor,
+    mode: str = "bilinear",
+    padding_mode: str = "zeros",
+    align_corners: bool = False,
+) -> Tensor:
+    """
+    Grid Sample 2D operation.
+    Samples input (N, C, H_in, W_in) at grid coordinates (N, H_out, W_out, 2).
+    Returns (N, C, H_out, W_out).
+    """
+    assert mode in ["bilinear", "nearest"], f"GridSample only supports bilinear/nearest mode, got {mode}"
+    assert padding_mode in ["zeros"], f"GridSample only supports zeros padding mode, got {padding_mode}"
+    result: Tensor = op(
+        OpType.GridSample, name, operandA, operandB,
+        mode=mode, padding_mode=padding_mode, align_corners=align_corners,
+    ).get_tensor()
+    return result
```

**`forge/forge/tvm_calls/relay/op/forge_passes.py`**
```diff
-            DecomposeGridSample(),
```
(Line removed from `run_forge_compile_passes` — the decomposition pass is disabled so the op passes through natively.)

**`forge/forge/tvm_to_python.py`**
```diff
+def populate_grid_sample_args(graph, nid, compiler_cfg):
+    args = []
+    node = graph["nodes"][nid]
+    method = node["attrs"].get("method", [["bilinear"]])[0][0]
+    if method == "bilinear":
+        mode = "bilinear"
+    elif method == "nearest_neighbor":
+        mode = "nearest"
+    else:
+        mode = method
+    padding_mode = node["attrs"].get("padding_mode", [["zeros"]])[0][0]
+    if "align_corners" in node["attrs"]:
+        ac_val = node["attrs"]["align_corners"][0][0]
+        align_corners = "True" if ac_val in ("True", "true", "1") else "False"
+    else:
+        coord_mode = node["attrs"].get("coordinate_transformation_mode", [["half_pixel"]])[0][0]
+        align_corners = "True" if coord_mode == "align_corners" else "False"
+    args.append(("mode", f'"{mode}"'))
+    args.append(("padding_mode", f'"{padding_mode}"'))
+    args.append(("align_corners", f"{align_corners}"))
+    return args

 tvm_to_forge_op_map = {
     ...
+    "image.grid_sample": "grid_sample",
     ...
 }

 forge_op_to_function_name = {
     ...
+    "grid_sample": "forge.op.GridSample",
     ...
 }

 forge_ops_needing_arguments = {
     ...
+    "grid_sample": populate_grid_sample_args,
     ...
 }
```

**`forge/csrc/ops/CMakeLists.txt`**
```diff
+    op_grid_sample.cpp
```

**`forge/csrc/ops/op.hpp`**
```diff
+    GridSample,
```
(Added to `OpType` enum before `Where`.)

**`forge/csrc/ops/op_interface.hpp`**
```diff
+DECLARE_OP_INTERFACE(grid_sample);
```

**`forge/csrc/ops/python_bindings.cpp`**
```diff
+        .value("GridSample", ops::OpType::GridSample)
```

**`forge/csrc/ops/op.cpp`**
```diff
 // In OpTypeToString mapping:
+        mapping_[OpType::GridSample] = "grid_sample";

 // In StringToOpType mapping:
+        mapping_["grid_sample"] = OpType::GridSample;

 // In Op::eval switch:
+        case OpType::GridSample: return grid_sample::eval(*this, tensors);

 // In Op::shape switch:
+        case OpType::GridSample: return grid_sample::shape(*this, inputs);

 // In Op::backward switch:
+        case OpType::GridSample: return grid_sample::backward(*this, context, operand, inputs, output, gradient);

 // In Op::decompose_initial, decompose_post_optimize, decompose_post_autograd:
+        case OpType::GridSample: return;

 // In Op::initial_flops_estimate:
+        case OpType::GridSample: return 0;

 // In Op::is_tm, is_eltwise, is_eltwise_unary, is_eltwise_binary, is_eltwise_nary:
+        case OpType::GridSample: return false;
```

**`forge/csrc/passes/lower_to_mlir.cpp`**
```diff
+        lowering_handler_map["grid_sample"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::GridSampleOp>;
```

**`forge/csrc/ops/op_grid_sample.cpp`** — new file implementing `grid_sample::eval`, `shape`, `backward`

---

### REPO: tt-mlir

**`include/ttmlir/Dialect/TTIR/IR/TTIROps.td`**
```diff
+def TTIR_GridSampleOp : TTIR_NamedOp<"grid_sample"> {
+    let summary = "Grid sample operation.";
+    let description = [{
+      Samples an input tensor at grid locations.
+      Input tensor has shape (N, C, H_in, W_in).
+      Grid tensor has shape (N, 2, H_out, W_out) with coordinates normalized to [-1, 1].
+      This format matches the TVM relay image.grid_sample convention.
+      Output tensor has shape (N, C, H_out, W_out).
+      Attributes:
+      - `mode` (str): Interpolation mode. Supported: "bilinear", "nearest".
+      - `padding_mode` (str): Padding mode. Supported: "zeros".
+      - `align_corners` (bool): If true, -1/1 map to input corners.
+    }];
+    let arguments = (ins AnyRankedTensor:$input,
+                         AnyRankedTensor:$grid,
+                         DefaultValuedAttr<StrAttr, "\"bilinear\"">:$mode,
+                         DefaultValuedAttr<StrAttr, "\"zeros\"">:$padding_mode,
+                         DefaultValuedAttr<BoolAttr, "false">:$align_corners);
+    let results = (outs AnyRankedTensor:$result);
+    let hasVerifier = 1;
+}
```

**`include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`**
```diff
+def TTNN_GridSampleOp : TTNN_Op<"grid_sample"> {
+    let summary = "Grid sample operation.";
+    let description = [{
+      Input tensor has shape (N, H_in, W_in, C) in NHWC format.
+      Grid tensor has shape (N, H_out, W_out, 2) with coordinates normalized to [-1, 1].
+      Output tensor has shape (N, H_out, W_out, C) in NHWC format.
+    }];
+    let arguments = (ins AnyRankedTensor:$input,
+                         AnyRankedTensor:$grid,
+                         DefaultValuedAttr<StrAttr, "\"bilinear\"">:$mode,
+                         DefaultValuedAttr<StrAttr, "\"zeros\"">:$padding_mode,
+                         DefaultValuedAttr<BoolAttr, "false">:$align_corners,
+                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
+    let results = (outs AnyRankedTensor:$result);
+    let extraClassDeclaration = [{
+      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
+        return wa::TTNNOperandsWorkaroundsFactory::createGridSampleOpOperandsWorkarounds(getOperation());
+      }
+    }];
+    let hasVerifier = 1;
+}
```

**`include/ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h`**
```diff
+  static TTNNOperandsWorkarounds
+  createGridSampleOpOperandsWorkarounds(mlir::Operation *op);
```

**`include/ttmlir/OpModel/TTNN/TTNNOpModel.h`**
```diff
+template <>
+struct OpModel<GridSampleOp> {
+  static llvm::Expected<OpConstraints>
+  getOpConstraints(ttcore::GridAttr deviceGrid,
+                   llvm::ArrayRef<int64_t> inputShape,
+                   llvm::ArrayRef<int64_t> gridShape,
+                   TTNNLayoutAttr inputLayout, TTNNLayoutAttr gridLayout,
+                   llvm::StringRef mode, llvm::StringRef paddingMode,
+                   bool alignCorners, TTNNLayoutAttr outputLayout);
+  static llvm::Expected<size_t>
+  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
+               llvm::ArrayRef<int64_t> gridShape,
+               TTNNLayoutAttr inputLayout, TTNNLayoutAttr gridLayout,
+               llvm::StringRef mode, llvm::StringRef paddingMode,
+               bool alignCorners, TTNNLayoutAttr outputLayout);
+};
```

**`include/ttmlir/Target/TTNN/operations/pool.fbs`**
```diff
+table GridSampleOp {
+  input: tt.target.ttnn.TensorRef;
+  grid: tt.target.ttnn.TensorRef;
+  mode: string;
+  padding_mode: string;
+  align_corners: bool;
+  memory_config: tt.target.ttnn.MemoryConfig;
+  out: tt.target.ttnn.TensorRef;
+}
```

**`include/ttmlir/Target/TTNN/program.fbs`**
```diff
+  GridSampleOp,
```
(Added to `OpType` union, alphabetically between `GetDeviceOp` and `GroupNormOp`.)

**`lib/Dialect/TTIR/IR/TTIROps.cpp`**
```diff
+::mlir::LogicalResult mlir::tt::ttir::GridSampleOp::verify() {
+  // Validate 4D ranks, grid dim[1]==2, legal mode/padding_mode strings,
+  // and matching batch dimensions between input, grid, and output.
+  ...
+}
```
(55 lines added — full rank/dim/mode validation)

**`lib/Dialect/TTNN/IR/TTNNOps.cpp`**
```diff
+::mlir::LogicalResult GridSampleOp::verify() {
+  // Same validation as TTIR but for NHWC convention: grid dim[3]==2.
+  ...
+}
```
(53 lines added)

**`lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp`**
```diff
+TTNNOperandsWorkarounds
+TTNNOperandsWorkaroundsFactory::createGridSampleOpOperandsWorkarounds(
+    mlir::Operation *op) {
+  auto gridSampleOp = mlir::cast<mlir::tt::ttnn::GridSampleOp>(op);
+  std::string mode = gridSampleOp.getMode().str();
+  bool alignCorners = gridSampleOp.getAlignCorners();
+  bool usesPrecomputedGrid = (mode == "nearest") || alignCorners;
+
+  TTNNOperandWorkarounds rowMajorLayoutBF16Workaround;
+  rowMajorLayoutBF16Workaround.tensorLayoutWorkaround = Layout::RowMajor;
+  rowMajorLayoutBF16Workaround.tensorDataTypeWorkaround = ttcore::DataType::BFloat16;
+
+  TTNNOperandWorkarounds gridWorkaround;
+  gridWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
+  if (!usesPrecomputedGrid) {
+    gridWorkaround.tensorDataTypeWorkaround = ttcore::DataType::BFloat16;
+  }
+  // For precomputed path (nearest or align_corners=True), grid stays float32
+  // because prepare_grid_sample_grid requires float32 precision.
+
+  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
+      .addInputOperandWorkaround(rowMajorLayoutBF16Workaround)
+      .addInputOperandWorkaround(gridWorkaround)
+      .addOutputOperandWorkaround(rowMajorLayoutBF16Workaround);
+}
```

**`lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp`**
```diff
+llvm::Expected<op_model::OpConstraints>
+GridSampleOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
+                               const OpConfig &opConfig) {
+  return detail::issueErrorForGetOpConstraints(
+      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
+}
+llvm::Expected<size_t>
+GridSampleOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
+                           const OpConfig &opConfig) {
+  return detail::issueErrorForGetOpRuntime(
+      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
+}
```
(GridSampleOp reports MissingMetalDefinition — MLA skips it and falls back to DRAM interleaved, which is the correct conservative default.)

**`lib/Dialect/TTNN/Analysis/L1InterleavedFallbackAnalysis.cpp`**
```diff
-    if (isa<ttnn::MaxPool2dOp, ttnn::UpsampleOp>(op)) {
+    if (isa<ttnn::MaxPool2dOp, ttnn::UpsampleOp, ttnn::GridSampleOp>(op)) {
```
(GridSampleOp excluded from L1 interleaved fallback analysis — it uses ROW_MAJOR workaround.)

**`lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp`**
```diff
+  if (isa<PermuteOp>(op)) {
+    analysisResult.erase(
+        std::remove_if(analysisResult.begin(), analysisResult.end(),
+                       [](const OpConfig &cfg) {
+                         return cfg.outputLayout &&
+                                cfg.outputLayout.hasShardedL1TensorMemoryLayout();
+                       }),
+        analysisResult.end());
+  }
```
(Removes sharded L1 candidates for PermuteOp — see Fix 2 for detailed explanation.)

**`lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`**
```diff
+        ttnn::GridSampleOp::getOperationName(),
```
(Added to the set of ops that require operand workarounds to be applied before layout optimization.)

**`lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`**
```diff
+class GridSampleOpConversionPattern : public OpConversionPattern<ttir::GridSampleOp> {
+  LogicalResult matchAndRewrite(...) const override {
+    // 1. Permute input NCHW -> NHWC: [0,2,3,1]
+    // 2. Permute grid N2HW -> NHW2: [0,2,3,1]
+    // 3. Create ttnn::GridSampleOp with NHWC shapes
+    // 4. Permute output NHWC -> NCHW: [0,3,1,2]
+  }
+};
// Registered in populateTTIRToTTNNPatterns
```

**`lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp`**
```diff
+class GridSampleOpConversionPattern : public TTNNToEmitCBaseOpConversionPattern<GridSampleOp> {
+  // Emits: ttnn::grid_sample(input, grid, mode, padding_mode, align_corners,
+  //                          use_precomputed_grid=nullopt, batch_output_channels=nullopt, memory_config)
+};
```

**`lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp`**
```diff
+class GridSampleOpConversionPattern : public TTNNToEmitPyBaseOpConversionPattern<GridSampleOp> {
+  // Emits: ttnn.grid_sample(input, grid, mode=..., padding_mode=..., align_corners=..., memory_config=...)
+};
```

**`lib/OpModel/TTNN/TTNNOpModel.cpp`**
```diff
+llvm::Expected<OpConstraints> OpModel<GridSampleOp>::getOpConstraints(...) {
+  return llvm::createStringError(..., "GridSampleOp op model not implemented");
+}
+llvm::Expected<size_t> OpModel<GridSampleOp>::getOpRuntime(...) {
+  return llvm::createStringError(..., "GridSampleOp op model not implemented");
+}
```

**`lib/Target/TTNN/TTNNToFlatbuffer.cpp`**
```diff
+::flatbuffers::Offset<::tt::target::ttnn::GridSampleOp>
+createOp(FlatbufferObjectCache &cache, GridSampleOp op) {
+  // Serialize input, grid, mode, padding_mode, align_corners, memory_config, output
+  return ::tt::target::ttnn::CreateGridSampleOp(*cache.fbb, input, grid, mode,
+                                                paddingMode, alignCorners, memoryConfig, output);
+}
// dispatch in emitTTNNOperation:
+  if (auto gridSampleOp = dyn_cast<GridSampleOp>(op); gridSampleOp) {
+    return createOperation(cache, createOp(cache, gridSampleOp), ...);
+  }
```

**`runtime/include/tt/runtime/detail/ttnn/ttnn.h`**
```diff
+#include "ttnn/operations/pool/grid_sample/grid_sample.hpp"
```

**`runtime/lib/ttnn/operations/CMakeLists.txt`**
```diff
+  ${CMAKE_CURRENT_SOURCE_DIR}/pool/grid_sample.cpp
```

**`runtime/lib/ttnn/operations/pool/grid_sample.h`** — new file (15 lines, declares `run()`)

**`runtime/lib/ttnn/operations/pool/grid_sample.cpp`** — new file (114 lines)

The runtime handles two paths:
- **Precomputed path** (`mode=="nearest"` or `alignCorners==true`): Grid is moved to host as float32, `prepare_grid_sample_grid` precomputes coordinates, result moved back to device, `grid_sample` called with `use_precomputed_grid=true`. Output is collected from HEIGHT_SHARDED L1 to DRAM interleaved.
- **Direct path** (bilinear, align_corners=False): Grid passed directly to `grid_sample` with `use_precomputed_grid=false`.

**`runtime/lib/ttnn/program_executor.cpp`**
```diff
+#include "operations/pool/grid_sample.h"
// in switch:
+  case ::tt::target::ttnn::OpType::GridSampleOp: {
+    return operations::pool::run(op->type_as_GridSampleOp(), getContext());
+  }
```

**`runtime/lib/ttnn/runtime.cpp`**
```diff
// in getOpOutputRef:
+  case ::tt::target::ttnn::OpType::GridSampleOp: {
+    tensorRef = opContext.type_as_GridSampleOp()->out(); break;
+  }
// in getOpInputRefs:
+  case ::tt::target::ttnn::OpType::GridSampleOp: {
+    tensorRefs = {opContext.type_as_GridSampleOp()->input(),
+                  opContext.type_as_GridSampleOp()->grid()}; break;
+  }
```

**`tools/builder/ttir/ttir_builder.py`**
```diff
+    def grid_sample(self, in0, in1, output, mode="bilinear", padding_mode="zeros", align_corners=False, unit_attrs=None):
+        # TTIRBuilder method for constructing grid_sample TTIR ops in tests
```

**`tools/golden/mapping.py`**
```diff
+def grid_sample_golden(in0, in1, output, mode="bilinear", padding_mode="zeros", align_corners=False):
+    # TTIR grid is (N, 2, H_out, W_out); torch expects (N, H_out, W_out, 2)
+    grid = in1.permute(0, 2, 3, 1).contiguous()
+    return torch.nn.functional.grid_sample(in0, grid, mode=mode, padding_mode=padding_mode, align_corners=bool(align_corners))

+    ttir.GridSampleOp: grid_sample_golden,
```

**`tools/ttnn-standalone/ttnn-precompiled.hpp`**
```diff
+#include "operations/pool/grid_sample/grid_sample.hpp"
```

## 6. After Fix — How It Works

```
ONNX GridSample node
  -> TVM relay: image.grid_sample(data, grid)     [grid: N x H x W x 2]
  -> tvm_to_python.py: forge.op.GridSample(...)    [attributes extracted]
  -> lower_to_mlir.cpp: ttir.grid_sample           [NCHW input, N2HW grid]
  -> TTIR->TTNN conversion:
       input: PermuteOp [0,2,3,1] -> NHWC
       grid:  PermuteOp [0,2,3,1] -> NHW2
       ttnn.grid_sample(NHWC_input, NHW2_grid, mode, padding_mode, align_corners)
       output: PermuteOp [0,3,1,2] -> NCHW
  -> Flatbuffer: GridSampleOp table serialized
  -> Runtime: grid_sample.cpp dispatched
       bilinear/align_corners=False -> direct kernel call
       nearest or align_corners=True -> precomputed grid path
  -> Device: ttnn::grid_sample executes on hardware
```

## 7. Test Results

| Test | Before | After |
|------|--------|-------|
| `test_bev_block_d_gridsample[default]` (40 tests) | PASS | PASS |
| `test_bev_block_d_gridsample[opt_level_1]` (40 tests) | PASS | PASS |
| `test_bev_block_d_gridsample[opt_level_2]` (40 tests) | **0/40 FAIL (hang)** | **40/40 PASS** |
| `test_ops_onnx.py::test_gridsample_*` | N/A (new) | PASS |
