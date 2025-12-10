# ONNX Conv Operator - Complete Summary

## Overview

The **Conv** operator performs convolution operations on input tensors. It consumes an input tensor (X) and a filter/weight tensor (W), and optionally a bias tensor (B), to compute the output tensor (Y). The operator supports various attributes including kernel size, strides, padding, dilation, and groups for controlling the convolution behavior.

The operator is fundamental to convolutional neural networks (CNNs) and is used extensively in image processing, computer vision, and deep learning applications.

## Version History

| Version | Since | Shape Inference | Function | Key Changes |
|---------|-------|----------------|----------|-------------|
| 1 | 1 | ✅ | ❌ | Initial version, basic attributes (auto_pad, dilations, group, kernel_shape, pads, strides), float types only |
| 11 | 11 | ✅ | ❌ | Extended type support (bfloat16, int types), improved auto_pad behavior for SAME_UPPER/SAME_LOWER |
| 22 | 22 | ✅ | ❌ | Extended type support (complex types, extended float types), improved documentation |

---

## Conv - Version 1

**Since Version:** 1  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

The convolution operator consumes an input tensor and a filter, and computes the output.

**Example:** Basic 2D convolution with 3×3 kernel, stride 1, padding 1.

```
Input X: [1, 3, 32, 32]  (batch=1, channels=3, height=32, width=32)
Weight W: [16, 3, 3, 3]  (out_channels=16, in_channels=3, kernel_h=3, kernel_w=3)
Output Y: [1, 16, 32, 32]  (batch=1, channels=16, height=32, width=32)
```

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input. In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. |
| `dilations` | INTS | ❌ | - | dilation value along each spatial axis of the filter. |
| `group` | INT | ❌ | `1` | number of groups input channels and output channels are divided into. |
| `kernel_shape` | INTS | ❌ | - | The shape of the convolution kernel. If not present, should be inferred from input W. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 … x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |
| `W` | T | The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x … x kn), where (k1 x k2 x … kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL …]. X.shape[1] == (W.shape[1] * group) == C (assuming zero based indices for the shape array). Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. |
| `B` | T (optional) | Optional 1D bias to be added to the convolution, has size of M. |

**Input Count:** Between 2 and 3 inputs.

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths. |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Shape Inference:** Supported, allowing automatic shape propagation
- **Weight Shape:** Weight tensor W has shape `[M, C/group, kH, kW]` where:
  - `M` = number of output channels (feature maps)
  - `C/group` = number of input channels per group
  - `kH, kW` = kernel height and width
- **Group Convolution:** When `group > 1`, input and output channels are divided into groups. Each group processes independently.
- **Auto Pad:** 
  - `NOTSET`: Use explicit `pads` attribute
  - `SAME_UPPER` / `SAME_LOWER`: Pad to match input size (output spatial size = input spatial size)
  - `VALID`: No padding
- **Padding Format:** `pads` format is `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]` where `xi_begin` is padding at the beginning of axis i and `xi_end` is padding at the end of axis i.
- **Stride and Dilation:** If not specified, default to 1 along each spatial axis.
- **Kernel Shape:** If not specified, inferred from weight tensor W shape.

### Output Shape Calculation

For 2D convolution, the output height and width are calculated as:

```
H_out = floor((H_in + 2*pad_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1)
W_out = floor((W_in + 2*pad_w - dilation_w*(kernel_w-1) - 1) / stride_w + 1)
```

Where:
- `H_in`, `W_in`: Input height and width
- `kernel_h`, `kernel_w`: Kernel height and width
- `stride_h`, `stride_w`: Stride along height and width
- `pad_h`, `pad_w`: Padding along height and width (sum of begin and end padding)
- `dilation_h`, `dilation_w`: Dilation along height and width

---

## Conv - Version 11

**Since Version:** 11  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

The convolution operator consumes an input tensor and a filter, and computes the output.

**Key Improvements:**
- Extended type support (bfloat16, integer types)
- Improved `auto_pad` behavior documentation for `SAME_UPPER` and `SAME_LOWER`
- Better validation for group convolution

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that output_shape[i] = ceil(input_shape[i] / strides[i]) for each axis i. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER. |
| `dilations` | INTS | ❌ | - | dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis. |
| `group` | INT | ❌ | `1` | number of groups input channels and output channels are divided into. |
| `kernel_shape` | INTS | ❌ | - | The shape of the convolution kernel. If not present, should be inferred from input W. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 … x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |
| `W` | T | The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x … x kn), where (k1 x k2 x … kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL …]. Assuming zero based indices for the shape array, X.shape[1] == (W.shape[1] * group) == C and W.shape[0] mod G == 0. Or in other words FILTER_IN_CHANNEL multiplied by the number of groups should be equal to DATA_CHANNEL and the number of feature maps M should be a multiple of the number of groups G. |
| `B` | T (optional) | Optional 1D bias to be added to the convolution, has size of M. |

**Input Count:** Between 2 and 3 inputs.

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths. |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(int16)`
- `tensor(int32)`
- `tensor(int64)`
- `tensor(int8)`
- `tensor(uint16)`
- `tensor(uint32)`
- `tensor(uint64)`
- `tensor(uint8)`

**Total:** 12 types

**Description:** Constrain input and output types to float tensors.

### Changes from v1

1. ✅ **Extended Type Support:** Added support for:
   - `bfloat16` (brain floating point 16-bit)
   - Integer types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`
2. 📝 **Improved Auto Pad Documentation:** Better description of `SAME_UPPER` and `SAME_LOWER` behavior:
   - Output shape formula: `output_shape[i] = ceil(input_shape[i] / strides[i])`
   - Padding split equally or almost equally
   - Extra padding placement (end for SAME_UPPER, beginning for SAME_LOWER)
3. ✅ **Default Values:** Explicit defaults for `dilations` and `strides` (1 along each spatial axis)
4. ✅ **Group Validation:** Explicit validation that `W.shape[0] mod G == 0` (output channels must be multiple of groups)

### Notes

- **Type Support:** Now supports both floating-point and integer types
- **Auto Pad Formula:** `SAME_UPPER` and `SAME_LOWER` now use `ceil(input_shape[i] / strides[i])` formula instead of matching input size exactly
- **Group Convolution:** More explicit validation that output channels must be divisible by groups
- **Weight Shape Validation:** `W.shape[0] mod G == 0` ensures proper group division

---

## Conv - Version 22

**Since Version:** 22  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

The convolution operator consumes an input tensor and a filter, and computes the output.

**Key Improvements:**
- Extended type support (complex types, extended float types)
- Improved documentation and type constraints

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that output_shape[i] = ceil(input_shape[i] / strides[i]) for each axis i. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER. |
| `dilations` | INTS | ❌ | - | dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis. |
| `group` | INT | ❌ | `1` | number of groups input channels and output channels are divided into. |
| `kernel_shape` | INTS | ❌ | - | The shape of the convolution kernel. If not present, should be inferred from input W. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 … x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |
| `W` | T | The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x … x kn), where (k1 x k2 x … kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL …]. Assuming zero based indices for the shape array, X.shape[1] == (W.shape[1] * group) == C and W.shape[0] mod G == 0. Or in other words FILTER_IN_CHANNEL multiplied by the number of groups should be equal to DATA_CHANNEL and the number of feature maps M should be a multiple of the number of groups G. |
| `B` | T (optional) | Optional 1D bias to be added to the convolution, has size of M. |

**Input Count:** Between 2 and 3 inputs.

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths. |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 4 types

**Description:** Constrain input and output types to float tensors.

**Note:** The v22 documentation shows only 4 float types, but this may be a documentation simplification. The actual implementation may support more types similar to v11. Please refer to the ONNX specification for the complete type list.

### Changes from v11

1. 📝 **Documentation Update:** Improved documentation clarity
2. 🔄 **Type Constraints:** Documentation shows simplified type list (float types only), but actual support may vary by implementation

### Notes

- **Type Support:** Documentation shows float types only, but implementations may support additional types
- **Backward Compatibility:** Functionally equivalent to v11 for most use cases
- **Documentation Focus:** Emphasis on float tensor types in documentation

---

## Summary of Changes Across Versions

### Type Support Evolution

| Version | New Types Added | Total Types | Description |
|---------|----------------|-------------|-------------|
| **v1** | Base float types | 3 | `double`, `float`, `float16` |
| **v11** | `bfloat16` + Integer types | 12 | Added `bfloat16`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64` |
| **v22** | Documentation update | 4+ | Documentation shows float types, but implementation may support more |

### Attribute Evolution

| Attribute | v1 | v11 | v22 | Notes |
|-----------|----|-----|-----|-------|
| `auto_pad` | ✅ | ✅ | ✅ | Improved documentation in v11+ |
| `dilations` | ✅ | ✅ | ✅ | Default value documented in v11+ |
| `group` | ✅ | ✅ | ✅ | No changes |
| `kernel_shape` | ✅ | ✅ | ✅ | No changes |
| `pads` | ✅ | ✅ | ✅ | No changes |
| `strides` | ✅ | ✅ | ✅ | Default value documented in v11+ |

### Auto Pad Behavior Evolution

| Version | SAME_UPPER/SAME_LOWER Formula | Notes |
|---------|------------------------------|-------|
| **v1** | Output spatial size matches input | Simple matching |
| **v11+** | `output_shape[i] = ceil(input_shape[i] / strides[i])` | Accounts for stride, more accurate |

### Key Behavioral Notes

1. **Input Requirements:**
   - **Minimum:** 2 inputs (X, W)
   - **Maximum:** 3 inputs (X, W, B)
   - All inputs must have the same type T

2. **Weight Tensor Shape:**
   - Format: `[M, C/group, kH, kW]` for 2D
   - `M` = number of output channels (must be multiple of `group`)
   - `C/group` = number of input channels per group
   - `kH, kW` = kernel height and width
   - Relationship: `X.shape[1] == (W.shape[1] * group) == C`

3. **Bias Tensor:**
   - Optional 1D tensor
   - Shape: `[M]` where M is the number of output channels
   - Added element-wise to each output channel

4. **Group Convolution:**
   - When `group = 1`: Standard convolution (each output sees all inputs)
   - When `group > 1`: Grouped convolution (input/output channels divided into groups)
   - When `group = C`: Depthwise convolution (each input channel has its own filter)
   - Constraint: `W.shape[0] mod group == 0` (output channels must be divisible by groups)

5. **Padding:**
   - Format: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`
   - Cannot be used simultaneously with `auto_pad`
   - Default: 0 along all axes if not specified

6. **Stride and Dilation:**
   - Default: 1 along each spatial axis if not specified
   - Stride controls step size (downsampling)
   - Dilation controls spacing between kernel elements (increases receptive field)

7. **Kernel Shape:**
   - Can be specified via `kernel_shape` attribute
   - If not specified, inferred from weight tensor W shape
   - Must match the spatial dimensions of W

---

## Implementation Considerations

### For Converter Implementation

1. **Input Extraction:**
   - `X` (input data): Always present at `node_proto.input[0]`
   - `W` (weights): Always present at `node_proto.input[1]`
   - `B` (bias): Optional at `node_proto.input[2]` (if present)

2. **Attribute Extraction:**
   - Extract all attributes from `attrs` dictionary
   - Handle defaults:
     - `auto_pad`: Default `'NOTSET'`
     - `group`: Default `1`
     - `dilations`: Default `1` per spatial axis (if not present)
     - `strides`: Default `1` per spatial axis (if not present)
     - `pads`: Default `0` per axis (if not present)
   - `kernel_shape`: Infer from W if not present

3. **Auto Pad Handling:**
   - If `auto_pad != 'NOTSET'`, compute padding values:
     - `SAME_UPPER` / `SAME_LOWER`: Use formula `ceil(input_shape[i] / strides[i])`
     - `VALID`: No padding
   - Convert to explicit padding format for PyTorch
   - May need to create a PadNode before ConvNode (as done in `conv.py`)

4. **Padding Format Conversion:**
   - ONNX format: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`
   - PyTorch format (2D): `(left, right, top, bottom)`
   - Conversion: `(pads[1], pads[3], pads[0], pads[2])` for 2D
   - For 1D: `(pads[0], pads[1])`
   - For 3D: `(pads[1], pads[4], pads[0], pads[3], pads[2], pads[5])`

5. **Dimension Detection:**
   - Determine conv dimension from `kernel_shape` or weight tensor shape
   - 1D: `kernel_shape` has 1 element or W has shape `[M, C/group, k1]`
   - 2D: `kernel_shape` has 2 elements or W has shape `[M, C/group, kH, kW]`
   - 3D: `kernel_shape` has 3 elements or W has shape `[M, C/group, kD, kH, kW]`

6. **Node Creation:**
   - Create appropriate Conv node based on dimension:
     - `Conv1dNode` for 1D
     - `Conv2dNode` for 2D
     - `Conv3dNode` for 3D
   - Pass converted attributes (stride, padding, dilation, groups)

7. **Type Support:**
   - Ensure converter handles all supported types for target opset version
   - Type support is additive (newer versions support all previous types)

8. **Validation:**
   - Verify `X.shape[1] == (W.shape[1] * group)`
   - Verify `W.shape[0] mod group == 0` (v11+)
   - Verify `pads` and `auto_pad` are not both specified
   - Verify bias shape matches output channels if present

---

## Comparison with PyTorch

The ONNX Conv operator maps to PyTorch's `torch.nn.Conv1d`, `torch.nn.Conv2d`, or `torch.nn.Conv3d`:

### PyTorch Conv2d Parameters

```python
torch.nn.Conv2d(
    in_channels,      # From X.shape[1] or W.shape[1] * group
    out_channels,      # From W.shape[0]
    kernel_size,       # From kernel_shape or W.shape[2:]
    stride=1,          # From strides attribute
    padding=0,          # From pads attribute (converted format)
    dilation=1,        # From dilations attribute
    groups=1,          # From group attribute
    bias=True           # If B input is present
)
```

### Mapping Table

| ONNX | PyTorch | Notes |
|------|---------|-------|
| `X` | `input` | Input tensor |
| `W` | `weight` | Weight tensor (same shape) |
| `B` | `bias` | Bias tensor (optional) |
| `kernel_shape` | `kernel_size` | Kernel dimensions |
| `strides` | `stride` | Stride values |
| `pads` | `padding` | Padding (format conversion needed) |
| `dilations` | `dilation` | Dilation values |
| `group` | `groups` | Number of groups |
| `auto_pad` | N/A | Must be converted to explicit padding |

### Key Differences

1. **Padding Format:**
   - ONNX: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`
   - PyTorch: `(left, right, top, bottom)` for 2D
   - Conversion required (see Implementation Considerations)

2. **Auto Pad:**
   - ONNX: Supports `auto_pad` attribute
   - PyTorch: No direct equivalent, must compute padding explicitly
   - Solution: Compute padding values and use explicit `padding` parameter

3. **Type Support:**
   - ONNX v1: Float types only
   - ONNX v11+: Float + Integer types
   - PyTorch: Supports all numeric types

4. **Bias:**
   - ONNX: Optional input tensor `B`
   - PyTorch: Optional parameter `bias` (boolean) with learnable bias tensor
   - Mapping: If `B` input present, set `bias=True`; otherwise `bias=False`

---

## Examples

### Example 1: Basic 2D Convolution (v1)

```python
# ONNX Model
# Input X: [1, 3, 32, 32]
# Weight W: [16, 3, 3, 3]
# Attributes: kernel_shape=[3, 3], stride=[1, 1], pads=[1, 1, 1, 1]
# Output Y: [1, 16, 32, 32]

# PyTorch Equivalent
import torch.nn as nn
conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,  # pads=[1,1,1,1] → padding=1
    dilation=1,
    groups=1,
    bias=True
)
```

### Example 2: Grouped Convolution (v11)

```python
# ONNX Model
# Input X: [1, 64, 32, 32]
# Weight W: [128, 32, 3, 3]  # 64/2 = 32 per group
# Attributes: group=2, kernel_shape=[3, 3], stride=[1, 1]
# Output Y: [1, 128, 30, 30]

# PyTorch Equivalent
conv = nn.Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=0,
    dilation=1,
    groups=2,  # Input/output divided into 2 groups
    bias=True
)
```

### Example 3: Depthwise Convolution (v11)

```python
# ONNX Model
# Input X: [1, 64, 32, 32]
# Weight W: [64, 1, 3, 3]  # 64/64 = 1 per group
# Attributes: group=64, kernel_shape=[3, 3], stride=[1, 1]
# Output Y: [1, 64, 30, 30]

# PyTorch Equivalent
conv = nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=0,
    dilation=1,
    groups=64,  # Depthwise: each input channel has its own filter
    bias=True
)
```

### Example 4: Strided Convolution with Auto Pad (v11)

```python
# ONNX Model
# Input X: [1, 3, 32, 32]
# Weight W: [16, 3, 3, 3]
# Attributes: auto_pad='SAME_UPPER', stride=[2, 2]
# Output Y: [1, 16, 16, 16]  # ceil(32/2) = 16

# PyTorch Equivalent
# Compute padding for SAME_UPPER with stride=2
# padding = (kernel_size - 1) // 2 = (3 - 1) // 2 = 1
conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=2,
    padding=1,  # Computed from auto_pad
    dilation=1,
    groups=1,
    bias=True
)
```

### Example 5: Dilated Convolution (v11)

```python
# ONNX Model
# Input X: [1, 3, 32, 32]
# Weight W: [16, 3, 3, 3]
# Attributes: dilations=[2, 2], stride=[1, 1], pads=[2, 2, 2, 2]
# Output Y: [1, 16, 32, 32]

# PyTorch Equivalent
conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=2,  # pads=[2,2,2,2] → padding=2
    dilation=2,  # dilations=[2,2] → dilation=2
    groups=1,
    bias=True
)
```

---

## References

- [ONNX Conv Operator Documentation](https://onnx.ai/onnx/operators/onnx__Conv.html)
- [PyTorch Conv2d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch Conv1d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- [PyTorch Conv3d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
- [Convolution in Deep Learning - Detailed Guide](../CONVOLUTION_DETAILED_GUIDE.md)

