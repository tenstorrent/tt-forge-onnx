# `grid_sample` Changes in tt-forge-onnx

## Context

These changes wire the native `ttnn::grid_sample` kernel through the forge-onnx compiler stack,
replacing the old `DecomposeGridSample` relay pass that expanded each GridSample into ~360
primitive ops. For the BEV model (40 GridSamples) the decomposition produced 14,480+ MLIR ops and
caused MLA pipeline hangs. The native path reduces this to a single `ttnn.grid_sample` per call.

---

## Architecture overview

The path from an ONNX `GridSample` node to device execution goes through these layers:

```
ONNX model
  │  onnx → TVM relay (via ONNX importer)
  ▼
TVM relay: image.grid_sample(data, grid)
  │  tvm_to_python.py: populate_grid_sample_args → forge.op.GridSample(...)
  ▼
Forge graph: Op(OpType::GridSample, ...)
  │  op_grid_sample.cpp: eval / shape / backward
  │  lower_to_mlir.cpp:  emit_mlir_ttforge_op<ttir::GridSampleOp>
  ▼
TTIR: ttir.grid_sample  [NCHW input, N2HW grid]
  │  (passed to tt-mlir compiler — see grid_sample_tt_mlir.md)
  ▼
TTNN: ttnn.grid_sample  [NHWC input, NHW2 grid, with surrounding permutes]
  │
  ▼
Device execution (see grid_sample_tt_metal.md)
```

---

## Grid tensor format through the stack

One important subtlety is the grid shape convention at each layer:

| Layer | Grid shape | Convention |
|---|---|---|
| ONNX | `(N, H_out, W_out, 2)` | Standard ONNX |
| TVM relay | `(N, 2, H_out, W_out)` | TVM transposes the grid |
| TTIR | `(N, 2, H_out, W_out)` | Preserves TVM convention |
| TTNN (after permute) | `(N, H_out, W_out, 2)` | Back to ONNX / tt-metal convention |

The permute `[0,2,3,1]` in `TTIRToTTNN.cpp` converts `N2HW → NHW2` before the TTNN op.

---

## Files changed

| File | Change |
|---|---|
| `forge/forge/op/resize.py` | `GridSample` op frontend function |
| `forge/csrc/ops/op_grid_sample.cpp` | C++ eval / shape / backward |
| `forge/csrc/ops/op.cpp` | OpType dispatch table entries |
| `forge/csrc/ops/op_interface.hpp` | `DECLARE_OP_INTERFACE(grid_sample)` |
| `forge/csrc/ops/python_bindings.cpp` | Python enum binding |
| `forge/csrc/passes/lower_to_mlir.cpp` | MLIR lowering handler |
| `forge/forge/tvm_to_python.py` | TVM relay → forge op mapping |
| `forge/forge/tvm_calls/relay/op/forge_passes.py` | `DecomposeGridSample` (kept but not used) |
| `forge/test/mlir/test_ops_onnx.py` | Unit tests |

---

## 1. Frontend op — `resize.py`

**File:** `forge/forge/op/resize.py`

The public Python API for grid sampling. Accepts NCHW input and ONNX-convention grid
`(N, H_out, W_out, 2)`:

```python
def GridSample(
    name: str,
    operandA: Tensor,           # (N, C, H_in, W_in)
    operandB: Tensor,           # (N, H_out, W_out, 2) — ONNX convention
    mode: str = "bilinear",     # "bilinear" | "nearest"
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:                    # (N, C, H_out, W_out)
    assert mode in ["bilinear", "nearest"]
    assert padding_mode in ["zeros"]
    result = op(OpType.GridSample, name, operandA, operandB,
                mode=mode, padding_mode=padding_mode,
                align_corners=align_corners).get_tensor()
    return result
```

---

## 2. C++ op implementation — `op_grid_sample.cpp`

**File:** `forge/csrc/ops/op_grid_sample.cpp`

Implements CPU-side evaluation (used for golden reference during testing), shape inference, and
the backward stub.

### `eval` — CPU reference

```cpp
at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors) {
    std::string mode         = op.attr_as<std::string>("mode");
    std::string padding_mode = op.attr_as<std::string>("padding_mode");
    bool align_corners       = op.attr_as<bool>("align_corners");

    const at::Tensor &input = tensors[0];  // (N, C, H_in, W_in)

    // Grid arrives in TVM relay format (N, 2, H_out, W_out).
    // torch::grid_sample expects (N, H_out, W_out, 2).
    at::Tensor grid = tensors[1].permute({0, 2, 3, 1}).contiguous();

    GridSampleFuncOptions options;
    options.align_corners(align_corners);
    options.padding_mode(torch::kZeros);
    options.mode(mode == "bilinear" ? torch::kBilinear : torch::kNearest);

    return torch::nn::functional::grid_sample(input, grid, options);
}
```

The `permute({0, 2, 3, 1})` on the grid is necessary because TVM relay stores grids in
`(N, 2, H_out, W_out)` order, while PyTorch's `grid_sample` expects `(N, H_out, W_out, 2)`.

### `shape` — output shape inference

```cpp
std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes) {
    const auto &input_shape = in_shapes[0];  // (N, C, H_in, W_in)
    const auto &grid_shape  = in_shapes[1];  // (N, 2, H_out, W_out) — TVM format

    // Output: (N, C, H_out, W_out) where H_out/W_out come from grid dims 2 and 3.
    return {Shape::create({input_shape[0], input_shape[1],
                           grid_shape[2], grid_shape[3]}), {}};
}
```

### `backward`

Not implemented — throws `TT_THROW`. GridSample is inference-only in the current flow.

---

## 3. TVM relay → forge mapping — `tvm_to_python.py`

**File:** `forge/forge/tvm_to_python.py`

The TVM relay op `image.grid_sample` is mapped to `forge.op.GridSample`. The attribute
population function handles two naming conventions for `align_corners` (TVM native vs. ONNX
`coordinate_transformation_mode`):

```python
# Op name mapping:
"image.grid_sample" → "grid_sample"          # relay op → internal name
"grid_sample"       → "forge.op.GridSample"  # internal name → Python call

def populate_grid_sample_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    # TVM uses "nearest_neighbor"; forge uses "nearest"
    method = node["attrs"].get("method", [["bilinear"]])[0][0]
    mode = "nearest" if method == "nearest_neighbor" else method

    padding_mode = node["attrs"].get("padding_mode", [["zeros"]])[0][0]

    # TVM native: align_corners attribute directly
    # ONNX-sourced: coordinate_transformation_mode = "align_corners"
    if "align_corners" in node["attrs"]:
        ac_val = node["attrs"]["align_corners"][0][0]
        align_corners = "True" if ac_val in ("True", "true", "1") else "False"
    else:
        coord_mode = node["attrs"].get(
            "coordinate_transformation_mode", [["half_pixel"]])[0][0]
        align_corners = "True" if coord_mode == "align_corners" else "False"

    return [("mode", f'"{mode}"'),
            ("padding_mode", f'"{padding_mode}"'),
            ("align_corners", f"{align_corners}")]
```

---

## 4. MLIR lowering — `lower_to_mlir.cpp`

**File:** `forge/csrc/passes/lower_to_mlir.cpp`

A single line registers the generic template lowering handler for grid_sample:

```cpp
lowering_handler_map["grid_sample"] =
    &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::GridSampleOp>;
```

The template emits a `ttir.grid_sample` op with the op's `mode`, `padding_mode`, and
`align_corners` attributes forwarded directly. The TTIR op then follows the tt-mlir lowering
pipeline (see `grid_sample_tt_mlir.md`).

---

## 5. `DecomposeGridSample` — `forge_passes.py`

**File:** `forge/forge/tvm_calls/relay/op/forge_passes.py`

`DecomposeGridSample` is a TVM `DFPatternCallback` that rewrites `image.grid_sample` into ~360
primitive relay ops (split → unnormalise → gather → bilinear blend → concat). It was the original
workaround before the native kernel was available.

The class is **retained in the file** but is no longer instantiated or registered in any pass
pipeline. The native path in `tvm_to_python.py` handles `image.grid_sample` via the op name
mapping before any pattern callbacks run.

---

## 6. Unit tests — `test_ops_onnx.py`

**File:** `forge/test/mlir/test_ops_onnx.py`

### What changed

1. **Removed `xfail` marks** from nearest-mode test cases — they now pass with the native kernel.
2. **Updated test shapes** so that the channel dimension C is always divisible by 32 (TILE_WIDTH
   constraint of the tt-metal NHWC kernel).
3. **Added BEV-representative shape** `(1, 64, 80, 144)` as a test case.

```python
@pytest.mark.push
@pytest.mark.parametrize(
    "data_shape, grid_shape, mode, padding_mode, align_corners",
    [
        # data_shape is NCHW (N, C, H_in, W_in).
        # C must be divisible by 32 (tt-metal NHWC tile constraint).
        pytest.param((1, 32, 8, 8),    (1, 4, 4, 2),    "bilinear", "zeros", 1),
        pytest.param((1, 32, 8, 8),    (1, 4, 4, 2),    "bilinear", "zeros", 0),
        pytest.param((1, 32, 8, 8),    (1, 4, 4, 2),    "nearest",  "zeros", 1),
        pytest.param((1, 32, 8, 8),    (1, 4, 4, 2),    "nearest",  "zeros", 0),
        pytest.param((1, 64, 96, 96),  (1, 128, 64, 2), "bilinear", "zeros", 1),
        pytest.param((1, 64, 96, 96),  (1, 128, 64, 2), "nearest",  "zeros", 1),
        pytest.param((1, 64, 80, 144), (1, 128, 64, 2), "bilinear", "zeros", 1),  # BEV shape
    ],
)
def test_gridsample(data_shape, grid_shape, mode, padding_mode, align_corners):
    # Build ONNX model, compile with forge, verify against PyTorch reference.
    ...
```

### Shape constraint rationale

The ONNX test inputs use `data_shape = (N, C, H_in, W_in)`. The forge compiler applies a
`NCHW → NHWC` permute before the kernel, so the channel dimension C becomes the last dim of the
NHWC input tensor. The tt-metal kernel requires this last dim to be a multiple of TILE_WIDTH=32.

---

## Test results

| Test suite | Tests | Result |
|---|---|---|
| `forge/test/mlir/test_ops_onnx.py::test_gridsample` | 7 | ✅ All passed |
| `forge/test/models/onnx/vision/bev/test_bev_block_d_gridsample.py` | 16 | ✅ All passed |

---

## Known limitations

| Limitation | Detail |
|---|---|
| C must be divisible by 32 | Models with C % 32 ≠ 0 need padding or must use the decomposition fallback |
| `padding_mode` | Only `"zeros"` tested; `"border"` and `"reflection"` pass through to kernel but are unvalidated |
| No backward pass | `OpType::GridSample::backward` throws — inference only |
| OpModel not implemented | Layout optimizer skips `ttnn.grid_sample`; full L1 estimation is future work |
