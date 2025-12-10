# AveragePool2d

**class** `forge.transpiler.ir.operations.pooling.AveragePool2dNode`

Applies a 2D average pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size (N, C, H, W), output (N, C, H_out, W_out) and `kernel_size` (kH, kW) can be precisely described as:

```
out(N_i, C_j, h, w) = (1 / (kH * kW)) * Σ(m=0 to kH-1) Σ(n=0 to kW-1) 
                       input(N_i, C_j, stride[0] × h + m, stride[1] × w + n)
```

If `padding` is non-zero, then the input is implicitly zero-padded on both sides for `padding` number of points.

**Note**

When `ceil_mode=True`, sliding windows are allowed to go off-bounds if they start within the left padding or the input. Sliding windows that would start in the right padded region are ignored.

**Note**

`padding` should be at most half of effective kernel size.

The parameters `kernel_size`, `stride`, `padding` can either be:

* a single `int` or a single-element tuple – in which case the same value is used for the height and width dimension
* a `tuple` of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension

## Parameters

* **kernel_size** (`Union[int, Tuple[int, int]]`) – the size of the window
* **stride** (`Union[int, Tuple[int, int]]`, optional) – the stride of the window. Default value is `kernel_size`
* **padding** (`Union[int, Tuple[int, int]]`, default: `0`) – implicit zero padding to be added on both sides
* **ceil_mode** (`bool`, default: `False`) – when `True`, will use `ceil` instead of `floor` to compute the output shape
* **count_include_pad** (`bool`, default: `True`) – when `True`, will include the zero-padding in the averaging calculation

## Shape

* **Input**: `(N, C, H_in, W_in)` or `(C, H_in, W_in)`
* **Output**: `(N, C, H_out, W_out)` or `(C, H_out, W_out)`, where

```
H_out = floor((H_in + 2 × padding[0] - kernel_size[0]) / stride[0] + 1)
W_out = floor((W_in + 2 × padding[1] - kernel_size[1]) / stride[1] + 1)
```

Per the note above, if `ceil_mode` is `True` and `(H_out - 1) × stride[0] ≥ H_in + padding[0]`, we skip the last window as it would start in the bottom padded region, resulting in `H_out` being reduced by one. The same applies for `W_out`.

## Examples

```python
# Pool of square window of size=3, stride=2
from forge.transpiler.ir.operations.pooling import AveragePool2dNode
from forge.transpiler.ir.types import TensorInfo

# Create node
pool_node = AveragePool2dNode.create(
    name="avgpool2d",
    inputs=["input"],
    outputs=["output"],
    input_tensors={"input": TensorInfo(shape=(20, 16, 50, 32), dtype=torch.float32)},
    output_tensors={"output": TensorInfo(shape=None, dtype=torch.float32)},
    kernel_size=3,
    stride=2
)

# Pool of non-square window
pool_node = AveragePool2dNode.create(
    name="avgpool2d",
    inputs=["input"],
    outputs=["output"],
    input_tensors={"input": TensorInfo(shape=(20, 16, 50, 32), dtype=torch.float32)},
    output_tensors={"output": TensorInfo(shape=None, dtype=torch.float32)},
    kernel_size=(3, 2),
    stride=(2, 1)
)
```

## Implementation Details

### TIR Node

The `AveragePool2dNode` is a TIR (Tenstorrent Intermediate Representation) node that represents a 2D average pooling operation in the Forge transpiler graph. It uses PyTorch's `torch.nn.functional.avg_pool2d` for evaluation.

**Location**: `forge/forge/transpiler/ir/operations/pooling.py`

**Forge Operation**: `forge.op.AvgPool2d`

### ONNX Conversion

The `AveragePoolConverter` handles conversion from ONNX `AveragePool` operations to `AveragePool2dNode` instances.

**Location**: `forge/forge/transpiler/frontends/onnx/converters/pooling.py`

**Supported Opset Versions**: v1+

**Features**:
- Automatic dimension detection (1D, 2D, or 3D pooling)
- Auto-pad support (`SAME_UPPER`, `SAME_LOWER`, `VALID`)
- Explicit padding conversion from ONNX format to PyTorch format
- Support for `ceil_mode` and `count_include_pad` attributes

### Auto-Pad Handling

When `auto_pad` is set to `SAME_UPPER`, `SAME_LOWER`, or `VALID`, the converter:

1. Computes padding values using `compute_autopad_padding()` utility function
2. Creates a `PadNode` to apply the computed padding
3. Sets `padding=0` for the `AveragePool2dNode` (since padding is already applied)

This approach is similar to `onnx2pytorch`'s `AutoPad` module, but adapted for the graph-based TIR architecture.

### Padding Format Conversion

ONNX uses the format: `[pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]`

PyTorch uses the format: `(padLeft, padRight, padTop, padBottom)`

The converter automatically converts between these formats:
```python
# ONNX: [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]
# PyTorch: (pad_W_begin, pad_W_end, pad_H_begin, pad_H_end)
padding = (padding[1], padding[3], padding[0], padding[2])
```

### Evaluation

The `eval()` method uses PyTorch's `torch.nn.functional.avg_pool2d`:

```python
def eval(self, input_tensors):
    x = input_tensors[self.inputs[0]]
    kernel_size = self.attrs['kernel_size']
    stride = self.attrs.get('stride', kernel_size)
    padding = self.attrs.get('padding', 0)
    ceil_mode = self.attrs.get('ceil_mode', False)
    count_include_pad = self.attrs.get('count_include_pad', True)
    return {self.outputs[0]: F.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}
```

## Differences from PyTorch

1. **Graph-based Architecture**: `AveragePool2dNode` is a node in the TIR graph, not a `nn.Module`
2. **Static Creation**: Uses `create()` static method instead of `__init__()`
3. **TensorInfo**: Requires `TensorInfo` objects for input/output tensor metadata
4. **No `divisor_override`**: The Forge implementation does not support `divisor_override` parameter

## Related Operations

- `AveragePool1dNode` - 1D average pooling
- `AveragePool3dNode` - 3D average pooling (evaluation only, not supported in Forge codegen)
- `MaxPool2dNode` - 2D max pooling
- `GlobalAveragePoolNode` - Global average pooling

## References

- [PyTorch AvgPool2d Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)
- [ONNX AveragePool Specification](https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool)

