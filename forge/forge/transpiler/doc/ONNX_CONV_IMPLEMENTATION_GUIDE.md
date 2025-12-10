# ONNX Conv2d Implementation Guide

## Overview

This document describes the implementation plan for the ONNX Conv operator converter (`ConvConverter`) and the TIR `Conv2dNode`. The implementation focuses on **Conv2d only** for now, with plans to extend to Conv1d and Conv3d later.

**Reference Implementation**: This guide is based on the `onnx2pytorch` implementation in `forge/forge/onnx2pytorch`, which provides a proven approach for ONNX to PyTorch conversion.

## Architecture

```
ONNX Model (Conv node)
    ↓
ConvConverter (ONNX → TIR conversion)
    ↓
[PadNode] (if auto_pad or asymmetric padding) → Conv2dNode (TIR node with eval using torch.nn.functional.conv2d)
    ↓
Forge Code Generation (forge.op.convolution.Conv2d)
```

## Key Design Decisions (Based on onnx2pytorch)

1. **Auto Pad Handling**: Use `PadNode` before `Conv2dNode` when `auto_pad != 'NOTSET'` (similar to `onnx2pytorch`'s `AutoPad` module)
2. **Padding Format**: Check for symmetric padding and simplify to `(int, int)` format when possible
3. **Asymmetric Padding**: Use `PadNode` for asymmetric padding (similar to `onnx2pytorch`'s `ConstantPad2d` layer)
4. **Conv Implementation**: Use `torch.nn.functional.conv2d` in `eval()` function

---

## Comparison with onnx2pytorch Approach

### onnx2pytorch Strategy

The `onnx2pytorch` library uses the following approach for Conv conversion:

1. **Auto Pad**: Creates an `AutoPad` module (nn.Module) that computes padding dynamically in forward pass
   - Wraps in `nn.Sequential(AutoPad(...), Conv2d(...))`
   - `AutoPad` computes padding based on input shape at runtime

2. **Padding Conversion**: Uses `extract_padding_params_for_conv_layer()`:
   - If symmetric: Returns first half `[1,1,1,1]` → `[1,1]`
   - If asymmetric: Creates `nn.ConstantPad2d` layer and wraps in Sequential

3. **Conv Layer**: Uses `nn.Conv2d` (not `F.conv2d`)

### Our Strategy (TIR Graph-Based)

1. **Auto Pad**: Use `PadNode` before `Conv2dNode`:
   - Compute padding during graph construction (when input shape is known)
   - Create separate `PadNode` in the graph
   - Set `Conv2dNode` padding to 0

2. **Padding Conversion**: Similar logic but adapted for graph nodes:
   - If symmetric: Simplify to `int` or `(int, int)`
   - If asymmetric: Convert to `(left, right, top, bottom)` tuple for `F.conv2d`
   - Or use `PadNode` for complex cases

3. **Conv Node**: Use `F.conv2d` in `eval()` function

### Key Differences

| Aspect | onnx2pytorch | Our Approach |
|--------|--------------|--------------|
| **Architecture** | `nn.Sequential` layers | Separate nodes in graph |
| **Auto Pad** | `AutoPad` module (forward pass) | `PadNode` (graph construction) |
| **Asymmetric Padding** | `nn.ConstantPad2d` layer | `PadNode` or tuple for `F.conv2d` |
| **Conv Implementation** | `nn.Conv2d` | `F.conv2d` in `eval()` |
| **Padding Computation** | Runtime (forward pass) | Graph construction time |

### Why Our Approach?

1. **Graph-Based**: Our TIR graph represents operations as nodes, not sequential layers
2. **Explicit Nodes**: `PadNode` is a first-class node type, making the graph structure clear
3. **Flexibility**: Can optimize/reorder nodes independently
4. **Consistency**: Matches our overall architecture (TIR graph → Forge code generation)

---

## 1. Conv2dNode Implementation

### 1.1 Current Structure

The `Conv2dNode` is located in `forge/forge/transpiler/ir/operations/conv.py` and follows the TIR node pattern:

```python
class Conv2dNode(TIRNode):
    """
    PyTorch-like Conv2d operation.
    """
    @staticmethod
    def create(name, inputs, outputs, input_tensors, output_tensors,
               stride=1, padding=0, dilation=1, groups=1):
        # Creates Conv2dNode with PyTorch-compatible attributes
        
    def eval(self, input_tensors):
        # Executes using torch.nn.functional.conv2d
```

### 1.2 Eval Function Implementation

The `eval` function uses `torch.nn.functional.conv2d` as specified in the [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html).

#### Function Signature

```python
torch.nn.functional.conv2d(
    input,          # (N, C_in, H, W)
    weight,         # (C_out, C_in/groups, kH, kW)
    bias=None,      # (C_out,) optional
    stride=1,       # int or (int, int)
    padding=0,      # int, str ('valid', 'same'), or (int, int) or (int, int, int, int)
    dilation=1,     # int or (int, int)
    groups=1        # int
)
```

#### Current Implementation

```python
def eval(self, input_tensors):
    x = input_tensors[self.inputs[0]]  # Input tensor
    w = input_tensors[self.inputs[1]]    # Weight tensor
    b = input_tensors[self.inputs[2]] if len(self.inputs) > 2 else None  # Optional bias
    
    return {self.outputs[0]: F.conv2d(x, w, bias=b, **self.attrs)}
```

#### Required Updates

The current implementation is mostly correct, but we need to ensure:

1. **Padding Format Conversion**: ONNX padding format must be converted to PyTorch format before calling `F.conv2d`
2. **Auto Pad Handling**: If `auto_pad` was used, padding should already be handled by a `PadNode`, so `padding=0` should be passed
3. **Parameter Validation**: Ensure all parameters are in the correct format

#### Updated Eval Function

```python
def eval(self, input_tensors):
    """
    Execute Conv2d operation using torch.nn.functional.conv2d.
    
    Args:
        input_tensors: Dictionary mapping input names to torch.Tensor
        
    Returns:
        Dictionary mapping output name to torch.Tensor
    """
    import torch.nn.functional as F
    
    # Extract inputs
    x = input_tensors[self.inputs[0]]  # Input: (N, C_in, H, W)
    w = input_tensors[self.inputs[1]]   # Weight: (C_out, C_in/groups, kH, kW)
    b = input_tensors[self.inputs[2]] if len(self.inputs) > 2 else None  # Bias: (C_out,) optional
    
    # Extract attributes (already in PyTorch format from converter)
    stride = self.attrs.get('stride', 1)
    padding = self.attrs.get('padding', 0)
    dilation = self.attrs.get('dilation', 1)
    groups = self.attrs.get('groups', 1)
    
    # Validate and normalize parameters
    # stride: int or (int, int)
    if isinstance(stride, int):
        stride = (stride, stride)
    elif isinstance(stride, (list, tuple)):
        if len(stride) == 1:
            stride = (stride[0], stride[0])
        elif len(stride) == 2:
            stride = tuple(stride)
        else:
            raise ValueError(f"Conv2dNode '{self.name}': stride must be int or (int, int), got {stride}")
    
    # dilation: int or (int, int)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    elif isinstance(dilation, (list, tuple)):
        if len(dilation) == 1:
            dilation = (dilation[0], dilation[0])
        elif len(dilation) == 2:
            dilation = tuple(dilation)
        else:
            raise ValueError(f"Conv2dNode '{self.name}': dilation must be int or (int, int), got {dilation}")
    
    # padding: int, str, or (int, int) or (int, int, int, int)
    # Note: PyTorch F.conv2d accepts:
    #   - int: same padding on all sides
    #   - (int, int): (padH, padW) - symmetric padding
    #   - (int, int, int, int): (padLeft, padRight, padTop, padBottom) - asymmetric padding
    #   - str: 'valid' (no padding) or 'same' (same padding, stride=1 only)
    # The converter should have already converted ONNX padding to PyTorch format
    
    # Call PyTorch conv2d
    output = F.conv2d(
        input=x,
        weight=w,
        bias=b,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    
    return {self.outputs[0]: output}
```

---

## 2. ConvConverter Implementation

### 2.1 Current Structure

The `ConvConverter` is located in `forge/forge/transpiler/frontends/onnx/converters/conv.py` and handles ONNX Conv operator conversion.

### 2.2 Key Responsibilities

1. **Extract ONNX Attributes**: Parse ONNX node attributes (`auto_pad`, `strides`, `pads`, `dilations`, `group`, `kernel_shape`)
2. **Handle Auto Pad**: Convert `auto_pad` to explicit padding using `PadNode` if needed
3. **Convert Padding Format**: Convert ONNX padding format to PyTorch format
4. **Create Conv2dNode**: Instantiate `Conv2dNode` with PyTorch-compatible attributes

### 2.3 ONNX to PyTorch Parameter Mapping

| ONNX | PyTorch | Conversion Notes |
|------|---------|------------------|
| `strides` | `stride` | Direct mapping, normalize to `(int, int)` |
| `pads` | `padding` | **Format conversion required** (see below) |
| `dilations` | `dilation` | Direct mapping, normalize to `(int, int)` |
| `group` | `groups` | Direct mapping |
| `auto_pad` | N/A | **Handled separately** (converted to explicit padding) |
| `kernel_shape` | N/A | Used for validation/inference only |

### 2.4 Padding Format Conversion

#### ONNX Padding Format

ONNX `pads` attribute format: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`

For 2D convolution: `[pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]`

**Example:**
```python
# ONNX pads: [1, 2, 3, 4]
# Meaning:
#   - pad_H_begin = 1 (top)
#   - pad_W_begin = 2 (left)
#   - pad_H_end = 3 (bottom)
#   - pad_W_end = 4 (right)
```

#### PyTorch Padding Format

PyTorch `F.conv2d` accepts padding in multiple formats:

1. **Symmetric padding (int)**: Same padding on all sides
   ```python
   padding = 1  # pad(1, 1, 1, 1)
   ```

2. **Symmetric padding (tuple of 2)**: `(padH, padW)`
   ```python
   padding = (1, 2)  # pad(1, 1, 2, 2)
   ```

3. **Asymmetric padding (tuple of 4)**: `(padLeft, padRight, padTop, padBottom)`
   ```python
   padding = (2, 4, 1, 3)  # left=2, right=4, top=1, bottom=3
   ```

4. **String padding**: `'valid'` (no padding) or `'same'` (same padding, stride=1 only)

#### Conversion Function (Based on onnx2pytorch)

The `onnx2pytorch` implementation uses `extract_padding_params_for_conv_layer()` which:
1. Checks if padding is symmetric (all values equal)
2. If symmetric: Returns first half `params[:len(params)//2]` (e.g., `[1,1,1,1]` → `[1,1]`)
3. If asymmetric: Creates a `ConstantPad2d` layer (we use `PadNode` instead)

**Our Implementation:**

```python
def convert_onnx_pads_to_pytorch(onnx_pads: List[int]) -> Union[int, Tuple[int, int], Tuple[int, int, int, int], None]:
    """
    Convert ONNX pads format to PyTorch padding format for Conv2d.
    Based on onnx2pytorch's extract_padding_params_for_conv_layer().
    
    ONNX format: [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]
    PyTorch format options:
        - int: symmetric padding (all sides equal)
        - (int, int): (padH, padW) symmetric per dimension
        - (int, int, int, int): (padLeft, padRight, padTop, padBottom) asymmetric
        - None: Use PadNode for asymmetric padding (similar to ConstantPad2d in onnx2pytorch)
    
    Args:
        onnx_pads: List of 4 integers [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]
        
    Returns:
        PyTorch-compatible padding format, or None if asymmetric (should use PadNode)
    """
    if not onnx_pads or len(onnx_pads) != 4:
        return 0  # Default to no padding
    
    pad_H_begin, pad_W_begin, pad_H_end, pad_W_end = onnx_pads
    
    # Check if symmetric (all sides equal) - return int
    if pad_H_begin == pad_H_end == pad_W_begin == pad_W_end:
        return pad_H_begin
    
    # Check if symmetric per dimension (H symmetric and W symmetric) - return (int, int)
    if pad_H_begin == pad_H_end and pad_W_begin == pad_W_end:
        return (pad_H_begin, pad_W_begin)
    
    # Asymmetric padding: Return None to indicate PadNode should be used
    # Similar to onnx2pytorch's ConstantPad2d layer approach
    return None  # Will be handled by creating PadNode
```

**Alternative Approach (Direct Conversion):**

If we want to support asymmetric padding directly in `F.conv2d` (PyTorch supports it):

```python
def convert_onnx_pads_to_pytorch(onnx_pads: List[int]) -> Union[int, Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Convert ONNX pads format to PyTorch padding format for Conv2d.
    
    ONNX format: [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]
    PyTorch F.conv2d supports:
        - int: symmetric padding
        - (int, int): (padH, padW) symmetric per dimension
        - (int, int, int, int): (padLeft, padRight, padTop, padBottom) asymmetric
    
    Args:
        onnx_pads: List of 4 integers [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]
        
    Returns:
        PyTorch-compatible padding format
    """
    if not onnx_pads or len(onnx_pads) != 4:
        return 0
    
    pad_H_begin, pad_W_begin, pad_H_end, pad_W_end = onnx_pads
    
    # Symmetric (all sides equal)
    if pad_H_begin == pad_H_end == pad_W_begin == pad_W_end:
        return pad_H_begin
    
    # Symmetric per dimension
    if pad_H_begin == pad_H_end and pad_W_begin == pad_W_end:
        return (pad_H_begin, pad_W_begin)
    
    # Asymmetric: PyTorch format is (left, right, top, bottom)
    # ONNX: [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]
    # PyTorch: (padLeft, padRight, padTop, padBottom)
    return (pad_W_begin, pad_W_end, pad_H_begin, pad_H_end)
```

**Recommendation**: Use the direct conversion approach since PyTorch `F.conv2d` supports asymmetric padding as a tuple of 4 integers.

**Conversion Examples:**

```python
# Example 1: Symmetric padding (all sides equal)
onnx_pads = [1, 1, 1, 1]
pytorch_padding = 1  # int

# Example 2: Symmetric per dimension
onnx_pads = [2, 3, 2, 3]  # H: 2 on both sides, W: 3 on both sides
pytorch_padding = (2, 3)  # (padH, padW)

# Example 3: Asymmetric padding
onnx_pads = [1, 2, 3, 4]  # H: top=1, bottom=3; W: left=2, right=4
pytorch_padding = (2, 4, 1, 3)  # (left, right, top, bottom)
```

### 2.5 Auto Pad Handling (Based on onnx2pytorch)

The `auto_pad` attribute in ONNX has three modes:
- `NOTSET`: Use explicit `pads` attribute
- `SAME_UPPER`: Pad so that `output_shape[i] = ceil(input_shape[i] / strides[i])`, extra padding at end
- `SAME_LOWER`: Pad so that `output_shape[i] = ceil(input_shape[i] / strides[i])`, extra padding at beginning
- `VALID`: No padding

#### Strategy: Use PadNode for Auto Pad (Similar to onnx2pytorch's AutoPad Module)

The `onnx2pytorch` implementation creates an `AutoPad` module (nn.Module) that computes padding dynamically in the forward pass, then wraps it in `nn.Sequential(AutoPad(...), Conv2d(...))`.

**Our Approach**: Use `PadNode` before `Conv2dNode` (similar concept, but as separate nodes in the graph):

1. **Compute padding values** using `AutoPad.compute_padding()` (same algorithm as onnx2pytorch)
2. **Create PadNode** to apply padding
3. **Set Conv2dNode padding to 0** (padding already applied)

**Key Insight from onnx2pytorch**: The `AutoPad` module computes padding dynamically based on input shape in the forward pass. Our `AutoPad.compute_padding()` does the same calculation, but we apply it during graph construction (when we know the input shape).

#### Implementation Flow

```python
# In ConvConverter._convert_conv_impl()

auto_pad = attrs.get('auto_pad', 'NOTSET')

if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
    # Step 1: Compute padding values
    pads = []
    for each spatial dimension:
        pad_before, pad_after = AutoPad.compute_padding(
            input_size, kernel_size, stride, dilation, auto_pad
        )
        pads.extend([pad_before, pad_after])
    
    # Step 2: Convert to PyTorch F.pad format (reverse order)
    # ONNX: [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]
    # F.pad: [padLeft, padRight, padTop, padBottom]
    pad_list = [pads[1], pads[3], pads[0], pads[2]]  # For 2D
    
    # Step 3: Create PadNode
    pad_node = PadNode.create(
        name=f"{node_name}_pad",
        inputs=[input_name],
        outputs=[padded_output_name],
        pad=tuple(pad_list),
        mode='constant',
        value=0.0
    )
    nodes.append(pad_node)
    
    # Step 4: Update conv inputs to use padded output
    conv_inputs = [padded_output_name]
    conv_padding = 0  # No padding needed, already padded
    
else:  # auto_pad == 'NOTSET'
    # Convert explicit pads to PyTorch format
    conv_padding = convert_onnx_pads_to_pytorch(attrs.get('pads', [0, 0, 0, 0]))
```

### 2.6 Complete Converter Implementation

```python
@classmethod
def _convert_conv_impl(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                      output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                      node_index: int, graph_proto=None) -> List:
    """
    Common implementation for Conv conversion (Conv2d only for now).
    """
    nodes = []
    node_name = node_proto.name if node_proto.name else f"Conv_{node_index}"
    
    # Step 1: Determine kernel shape and dimension
    kernel_shape = attrs.get('kernel_shape', None)
    if kernel_shape is None:
        # Infer from weight tensor
        weight_name = node_proto.input[1]
        weight_shape = input_tensors[weight_name].shape
        if weight_shape and len(weight_shape) >= 2:
            kernel_shape = weight_shape[2:]  # [out_ch, in_ch, kH, kW] -> [kH, kW]
        else:
            raise ValueError(f"Cannot infer kernel_shape for Conv {node_name}")
    
    # For now, only support Conv2d (2D kernel)
    if len(kernel_shape) != 2:
        raise ValueError(f"Conv2d only supports 2D kernels, got kernel_shape {kernel_shape}")
    
    kernel_h, kernel_w = kernel_shape
    
    # Step 2: Handle AUTO_PAD
    auto_pad = attrs.get('auto_pad', 'NOTSET')
    conv_inputs = list(node_proto.input)
    conv_input_tensors = input_tensors.copy()
    conv_padding = 0
    
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
        # Create PadNode for auto_pad
        pad_name = f"{node_name}_pad"
        pad_output = f"{node_name}_padded"
        
        # Get input shape
        input_shape = input_tensors[conv_inputs[0]].shape
        if input_shape is None:
            raise ValueError(f"Cannot compute auto_pad for Conv {node_name} with unknown input shape")
        
        # Extract stride and dilation
        stride = attrs.get('strides', 1)
        dilation = attrs.get('dilations', 1)
        
        # Normalize to tuples
        if isinstance(stride, int):
            stride = (stride, stride)
        elif isinstance(stride, (list, tuple)):
            stride = tuple(stride[:2]) if len(stride) >= 2 else (stride[0], stride[0])
        
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        elif isinstance(dilation, (list, tuple)):
            dilation = tuple(dilation[:2]) if len(dilation) >= 2 else (dilation[0], dilation[0])
        
        # Compute padding for each spatial dimension
        pads = []
        spatial_dims = input_shape[2:]  # Skip batch and channel: (H, W)
        
        for in_size, k_size, s, d in zip(spatial_dims, (kernel_h, kernel_w), stride, dilation):
            pad_before, pad_after = AutoPad.compute_padding(
                in_size, k_size, s, d, auto_pad
            )
            pads.extend([pad_before, pad_after])
        
        # Convert to PyTorch F.pad format: [left, right, top, bottom]
        # ONNX pads: [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]
        # F.pad: [padLeft, padRight, padTop, padBottom]
        pad_list = [pads[1], pads[3], pads[0], pads[2]]
        
        # Create pad output tensor info
        pad_output_tensors = {pad_output: input_tensors[conv_inputs[0]]}
        
        pad_node = PadNode.create(
            name=pad_name,
            inputs=[conv_inputs[0]],
            outputs=[pad_output],
            input_tensors={conv_inputs[0]: input_tensors[conv_inputs[0]]},
            output_tensors=pad_output_tensors,
            pad=tuple(pad_list),
            mode='constant',
            value=0.0
        )
        nodes.append(pad_node)
        
        # Conv will use padded output
        conv_inputs = [pad_output]
        conv_input_tensors = {pad_output: pad_output_tensors[pad_output]}
        conv_padding = 0  # No padding needed, already padded
        
    else:  # auto_pad == 'NOTSET'
        # Convert explicit pads to PyTorch format
        onnx_pads = attrs.get('pads', [0, 0, 0, 0])
        conv_padding = cls._convert_onnx_pads_to_pytorch(onnx_pads)
    
    # Step 3: Extract and normalize other attributes
    stride = attrs.get('strides', 1)
    if isinstance(stride, int):
        stride = (stride, stride)
    elif isinstance(stride, (list, tuple)):
        stride = tuple(stride[:2]) if len(stride) >= 2 else (stride[0], stride[0])
    
    dilation = attrs.get('dilations', 1)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    elif isinstance(dilation, (list, tuple)):
        dilation = tuple(dilation[:2]) if len(dilation) >= 2 else (dilation[0], dilation[0])
    
    groups = attrs.get('group', 1)
    
    # Step 4: Create Conv2dNode
    conv_output_tensors = output_tensors.copy()
    conv_node = Conv2dNode.create(
        name=node_name,
        inputs=conv_inputs,
        outputs=[node_proto.output[0]],
        input_tensors=conv_input_tensors,
        output_tensors=conv_output_tensors,
        stride=stride,
        padding=conv_padding,
        dilation=dilation,
        groups=groups
    )
    nodes.append(conv_node)
    
    return nodes

@staticmethod
def _convert_onnx_pads_to_pytorch(onnx_pads: List[int]) -> Union[int, Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Convert ONNX pads format to PyTorch padding format for Conv2d.
    
    Args:
        onnx_pads: List of 4 integers [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end]
        
    Returns:
        PyTorch-compatible padding (int, (int, int), or (int, int, int, int))
    """
    if not onnx_pads or len(onnx_pads) != 4:
        return 0  # Default to no padding
    
    pad_H_begin, pad_W_begin, pad_H_end, pad_W_end = onnx_pads
    
    # Symmetric padding (all sides equal)
    if pad_H_begin == pad_H_end == pad_W_begin == pad_W_end:
        return pad_H_begin
    
    # Symmetric per dimension
    if pad_H_begin == pad_H_end and pad_W_begin == pad_W_end:
        return (pad_H_begin, pad_W_begin)
    
    # Asymmetric padding: (left, right, top, bottom)
    return (pad_W_begin, pad_W_end, pad_H_begin, pad_H_end)
```

---

## 3. Parameter Validation and Normalization

### 3.1 Stride

- **ONNX**: `strides` attribute (list of ints)
- **PyTorch**: `stride` parameter (int or tuple of 2 ints)
- **Normalization**: Convert to `(int, int)` tuple

```python
stride = attrs.get('strides', 1)
if isinstance(stride, int):
    stride = (stride, stride)
elif isinstance(stride, (list, tuple)):
    if len(stride) == 1:
        stride = (stride[0], stride[0])
    elif len(stride) >= 2:
        stride = tuple(stride[:2])
```

### 3.2 Dilation

- **ONNX**: `dilations` attribute (list of ints)
- **PyTorch**: `dilation` parameter (int or tuple of 2 ints)
- **Normalization**: Convert to `(int, int)` tuple

```python
dilation = attrs.get('dilations', 1)
if isinstance(dilation, int):
    dilation = (dilation, dilation)
elif isinstance(dilation, (list, tuple)):
    if len(dilation) == 1:
        dilation = (dilation[0], dilation[0])
    elif len(dilation) >= 2:
        dilation = tuple(dilation[:2])
```

### 3.3 Groups

- **ONNX**: `group` attribute (int)
- **PyTorch**: `groups` parameter (int)
- **Direct mapping**: No conversion needed

```python
groups = attrs.get('group', 1)
```

### 3.4 Padding

- **ONNX**: `pads` attribute (list of 4 ints) or `auto_pad` attribute
- **PyTorch**: `padding` parameter (int, tuple, or str)
- **Conversion**: Use `_convert_onnx_pads_to_pytorch()` function

---

## 4. Input/Output Handling

### 4.1 Inputs

ONNX Conv has 2-3 inputs:
1. **X** (required): Input tensor `(N, C_in, H, W)`
2. **W** (required): Weight tensor `(C_out, C_in/groups, kH, kW)`
3. **B** (optional): Bias tensor `(C_out,)`

```python
# In Conv2dNode.eval()
x = input_tensors[self.inputs[0]]  # X
w = input_tensors[self.inputs[1]]  # W
b = input_tensors[self.inputs[2]] if len(self.inputs) > 2 else None  # B (optional)
```

### 4.2 Outputs

ONNX Conv has 1 output:
- **Y**: Output tensor `(N, C_out, H_out, W_out)`

```python
# Output shape calculation:
H_out = floor((H_in + 2*pad_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1)
W_out = floor((W_in + 2*pad_w - dilation_w*(kernel_w-1) - 1) / stride_w + 1)
```

---

## 5. Examples

### Example 1: Basic Conv2d with Explicit Padding

**ONNX Model:**
```python
# Attributes:
#   kernel_shape = [3, 3]
#   strides = [1, 1]
#   pads = [1, 1, 1, 1]  # Symmetric padding
#   dilations = [1, 1]
#   group = 1
```

**Conversion:**
```python
# Padding conversion:
onnx_pads = [1, 1, 1, 1]
pytorch_padding = 1  # Symmetric, all sides = 1

# Conv2dNode attributes:
stride = (1, 1)
padding = 1
dilation = (1, 1)
groups = 1
```

**PyTorch Call:**
```python
F.conv2d(x, w, bias=b, stride=(1, 1), padding=1, dilation=(1, 1), groups=1)
```

### Example 2: Conv2d with Auto Pad SAME_UPPER

**ONNX Model:**
```python
# Attributes:
#   kernel_shape = [3, 3]
#   strides = [1, 1]
#   auto_pad = 'SAME_UPPER'
#   dilations = [1, 1]
#   group = 1
```

**Conversion:**
```python
# Step 1: Compute padding using AutoPad
# Input: (1, 3, 32, 32), kernel=(3, 3), stride=(1, 1), dilation=(1, 1)
# AutoPad.compute_padding(32, 3, 1, 1, 'SAME_UPPER') -> (1, 1) for H
# AutoPad.compute_padding(32, 3, 1, 1, 'SAME_UPPER') -> (1, 1) for W
# pads = [1, 1, 1, 1]

# Step 2: Create PadNode
pad_node = PadNode.create(
    name="conv_pad",
    pad=(1, 1, 1, 1),  # (left, right, top, bottom)
    mode='constant',
    value=0.0
)

# Step 3: Create Conv2dNode with padding=0
conv_node = Conv2dNode.create(
    name="conv",
    stride=(1, 1),
    padding=0,  # Already padded by PadNode
    dilation=(1, 1),
    groups=1
)
```

### Example 3: Conv2d with Asymmetric Padding

**ONNX Model:**
```python
# Attributes:
#   pads = [1, 2, 3, 4]  # H: top=1, bottom=3; W: left=2, right=4
```

**Conversion:**
```python
# Padding conversion:
onnx_pads = [1, 2, 3, 4]
# pad_H_begin=1, pad_W_begin=2, pad_H_end=3, pad_W_end=4
# PyTorch format: (left, right, top, bottom)
pytorch_padding = (2, 4, 1, 3)
```

**PyTorch Call:**
```python
F.conv2d(x, w, bias=b, padding=(2, 4, 1, 3))
```

### Example 4: Grouped Convolution

**ONNX Model:**
```python
# Attributes:
#   group = 2
#   Input: (1, 64, 32, 32)
#   Weight: (128, 32, 3, 3)  # 64/2 = 32 per group
```

**Conversion:**
```python
groups = 2
# Each group processes 32 input channels independently
# Output: (1, 128, 30, 30)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

1. **Padding Conversion Tests**:
   - Symmetric padding (all sides equal)
   - Symmetric per dimension
   - Asymmetric padding
   - Edge cases (zero padding, large padding)

2. **Auto Pad Tests**:
   - `SAME_UPPER` with various strides
   - `SAME_LOWER` with various strides
   - `VALID` (no padding)
   - Edge cases (stride > kernel_size)

3. **Conv2dNode Eval Tests**:
   - Basic convolution
   - With bias
   - Without bias
   - Grouped convolution
   - Dilated convolution
   - Strided convolution

4. **Converter Tests**:
   - Full ONNX model conversion
   - Different opset versions (1, 11, 22)
   - Various attribute combinations

### 6.2 Integration Tests

1. **ONNX Runtime Comparison**:
   - Compare TIR graph execution with ONNX Runtime
   - Verify output shapes match
   - Verify output values match (within tolerance)

2. **End-to-End Tests**:
   - Convert ONNX model → TIR graph → Execute
   - Compare with reference PyTorch model

---

## 7. Future Extensions

### 7.1 Conv1d Support

- Add `Conv1dNode` (currently raises `NotImplementedError`)
- Extend padding conversion for 1D: `[pad_begin, pad_end]` → `(pad,)` or `(pad_begin, pad_end)`
- Update converter to handle 1D kernels

### 7.2 Conv3d Support

- Add `Conv3dNode` (currently raises `NotImplementedError`)
- Extend padding conversion for 3D: `[pad_D_begin, pad_H_begin, pad_W_begin, pad_D_end, pad_H_end, pad_W_end]` → `(padLeft, padRight, padTop, padBottom, padFront, padBack)`
- Update converter to handle 3D kernels

### 7.3 Additional Features

- Support for `padding='same'` string in PyTorch (requires stride=1)
- Support for complex data types
- Optimizations for specific padding patterns

---

## 8. References

- [PyTorch F.conv2d Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)
- [ONNX Conv Operator Documentation](https://onnx.ai/onnx/operators/onnx__Conv.html)
- [ONNX Conv Complete Summary](./ONNX_CONV_COMPLETE_SUMMARY.md)
- [Convolution Detailed Guide](../CONVOLUTION_DETAILED_GUIDE.md)
- **onnx2pytorch Implementation** (Reference):
  - `forge/forge/onnx2pytorch/onnx2pytorch/convert/layer.py` - `convert_layer()` function
  - `forge/forge/onnx2pytorch/onnx2pytorch/operations/autopad.py` - `AutoPad` module
  - `forge/forge/onnx2pytorch/onnx2pytorch/utils.py` - `extract_padding_params_for_conv_layer()` function
  - `forge/forge/onnx2pytorch/onnx2pytorch/convert/attribute.py` - Attribute extraction

---

## Summary

This implementation guide provides:

1. **Conv2dNode**: Uses `torch.nn.functional.conv2d` in `eval()` with proper parameter normalization
2. **ConvConverter**: Handles ONNX → TIR conversion with:
   - Auto pad handling via `PadNode` (similar to onnx2pytorch's `AutoPad` module)
   - Padding format conversion (ONNX → PyTorch) based on `extract_padding_params_for_conv_layer()`
   - Parameter normalization and validation
3. **Padding Conversion**: 
   - Symmetric padding: Simplified to `int` or `(int, int)` format
   - Asymmetric padding: Converted to `(left, right, top, bottom)` tuple for `F.conv2d`
   - Based on onnx2pytorch's approach but adapted for our graph-based architecture
4. **Auto Pad Strategy**: Use `PadNode` to apply padding before convolution when `auto_pad != 'NOTSET'` (similar to onnx2pytorch's `nn.Sequential(AutoPad, Conv2d)` approach)

**Key Differences from onnx2pytorch**:
- onnx2pytorch uses `nn.Sequential` to chain layers, we use separate nodes in the graph
- onnx2pytorch uses `nn.ConstantPad2d` for asymmetric padding, we use `PadNode`
- onnx2pytorch's `AutoPad` computes padding in forward pass, we compute it during graph construction

The implementation focuses on **Conv2d only** for now, with clear extension points for Conv1d and Conv3d in the future.

