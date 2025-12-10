# ONNX AveragePool Operator - Complete Summary

## Overview

The **AveragePool** operator performs average pooling operations on input tensors. It consumes an input tensor (X) and applies average pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Average pooling consists of computing the average on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

The operator is fundamental to convolutional neural networks (CNNs) and is used extensively for downsampling, feature extraction, and reducing computational complexity in deep learning applications.

## Version History

| Version | Since | Shape Inference | Function | Key Changes |
|---------|-------|----------------|----------|-------------|
| 1 | 1 | ✅ | ❌ | Initial version, basic attributes (auto_pad, kernel_shape, pads, strides, count_include_pad), float types only |
| 7 | 7 | ✅ | ❌ | Added `ceil_mode` attribute for controlling output shape calculation |
| 10 | 10 | ✅ | ❌ | Improved `ceil_mode` behavior with explicit formulas for auto_pad modes |
| 11 | 11 | ✅ | ❌ | Improved output shape formulas with explicit ceil_mode handling |
| 19 | 19 | ✅ | ❌ | Added `dilations` attribute support, extended type support (bfloat16), improved documentation |
| 22 | 22 | ✅ | ❌ | Extended type support (complex types, extended float types), improved documentation, note about sliding windows in padded regions |

---

## AveragePool - Version 1

**Since Version:** 1  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

AveragePool consumes an input tensor X and applies average pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Average pooling consists of computing the average on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

**Example:** Basic 2D average pooling with 2×2 kernel, stride 2, no padding.

```
Input X: [1, 3, 32, 32]  (batch=1, channels=3, height=32, width=32)
Kernel: [2, 2]
Stride: [2, 2]
Output Y: [1, 3, 16, 16]  (batch=1, channels=3, height=16, width=16)
```

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input. In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. |
| `count_include_pad` | INT | ❌ | `0` | Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |

**Input Count:** 1 input.

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used. |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Shape Inference:** Supported, allowing automatic shape propagation
- **Kernel Shape:** Required attribute that specifies the size of the pooling window along each spatial axis
- **Auto Pad:** 
  - `NOTSET`: Use explicit `pads` attribute
  - `SAME_UPPER` / `SAME_LOWER`: Pad to match input size (output spatial size = input spatial size)
  - `VALID`: No padding
- **Padding Format:** `pads` format is `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]` where `xi_begin` is padding at the beginning of axis i and `xi_end` is padding at the end of axis i.
- **Stride:** If not specified, defaults to 1 along each spatial axis
- **Count Include Pad:** When `count_include_pad=0` (default), padded zeros are excluded from the average calculation. When `count_include_pad=1`, padded zeros are included.

### Output Shape Calculation

For 2D average pooling, the output height and width are calculated as:

```
H_out = floor((H_in + pad_shape[0] - kernel_shape[0]) / strides[0] + 1)
W_out = floor((W_in + pad_shape[1] - kernel_shape[1]) / strides[1] + 1)
```

Where:
- `H_in`, `W_in`: Input height and width
- `kernel_shape[0]`, `kernel_shape[1]`: Kernel height and width
- `strides[0]`, `strides[1]`: Stride along height and width
- `pad_shape[0]`, `pad_shape[1]`: Total padding along height and width (sum of begin and end padding)

**Auto Pad Formulas (v1):**

When `auto_pad != 'NOTSET'`:
- `VALID`: `output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

And pad shape for `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```

---

## AveragePool - Version 7

**Since Version:** 7  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

AveragePool consumes an input tensor X and applies average pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Average pooling consists of computing the average on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

**Key Improvements:**
- Added `ceil_mode` attribute for controlling output shape calculation

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input. In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. |
| `ceil_mode` | INT | ❌ | `0` | Whether to use ceil or floor (default) to compute the output shape. |
| `count_include_pad` | INT | ❌ | `0` | Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |

**Input Count:** 1 input.

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used. |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Changes from v1

1. ✅ **Added `ceil_mode` Attribute:** 
   - When `ceil_mode=0` (default): Uses `floor()` for output shape calculation
   - When `ceil_mode=1`: Uses `ceil()` for output shape calculation
   - Allows including partial windows in the output

### Output Shape Calculation

**ceil_mode = 0 (default, floor):**
```
H_out = floor((H_in + pad_shape[0] - kernel_shape[0]) / strides[0] + 1)
W_out = floor((W_in + pad_shape[1] - kernel_shape[1]) / strides[1] + 1)
```

**ceil_mode = 1 (ceiling):**
```
H_out = ceil((H_in + pad_shape[0] - kernel_shape[0]) / strides[0] + 1)
W_out = ceil((W_in + pad_shape[1] - kernel_shape[1]) / strides[1] + 1)
```

**Auto Pad Formulas (v7):**

When `auto_pad != 'NOTSET'`:
- `VALID`: `output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

And pad shape for `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```

### Notes

- **Ceil Mode:** When enabled, allows sliding windows to extend beyond the input boundaries if they start within valid regions
- **Floor Mode (default):** Standard behavior, ignores partial windows that don't fit completely

---

## AveragePool - Version 10

**Since Version:** 10  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

AveragePool consumes an input tensor X and applies average pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Average pooling consists of computing the average on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

**Key Improvements:**
- Improved `ceil_mode` behavior with explicit formulas for auto_pad modes when `ceil_mode` is enabled or disabled

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input. In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. |
| `ceil_mode` | INT | ❌ | `0` | Whether to use ceil or floor (default) to compute the output shape. |
| `count_include_pad` | INT | ❌ | `0` | Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |

**Input Count:** 1 input.

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used. |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Changes from v7

1. 📝 **Improved Auto Pad Formulas:** Explicit formulas for auto_pad modes when `ceil_mode` is enabled or disabled

### Output Shape Calculation

**Explicit Padding (ceil_mode = 0, floor):**
```
H_out = floor((H_in + pad_shape[0] - kernel_shape[0]) / strides[0] + 1)
W_out = floor((W_in + pad_shape[1] - kernel_shape[1]) / strides[1] + 1)
```

**Explicit Padding (ceil_mode = 1, ceiling):**
```
H_out = ceil((H_in + pad_shape[0] - kernel_shape[0]) / strides[0] + 1)
W_out = ceil((W_in + pad_shape[1] - kernel_shape[1]) / strides[1] + 1)
```

**Auto Pad Formulas (v10):**

When `ceil_mode = 1` (enabled):
- `VALID`: `output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

When `ceil_mode = 0` (disabled):
- `VALID`: `output_spatial_shape[i] = floor((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])`

And pad shape for `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```

### Notes

- **Ceil Mode with Auto Pad:** Now has explicit formulas for both `ceil_mode=0` and `ceil_mode=1` cases
- **Floor Mode with Auto Pad:** Explicit formulas for `VALID` and `SAME_UPPER/SAME_LOWER` modes

---

## AveragePool - Version 11

**Since Version:** 11  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

AveragePool consumes an input tensor X and applies average pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Average pooling consists of computing the average on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

**Key Improvements:**
- Improved output shape formulas with explicit handling for `ceil_mode` with auto_pad modes

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that output_shape[i] = ceil(input_shape[i] / strides[i]) for each axis i. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER. |
| `ceil_mode` | INT | ❌ | `0` | Whether to use ceil or floor (default) to compute the output shape. |
| `count_include_pad` | INT | ❌ | `0` | Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |

**Input Count:** 1 input.

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used. |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Changes from v10

1. 📝 **Improved Auto Pad Formulas:** 
   - Explicit formulas for auto_pad modes when `ceil_mode` is enabled or disabled
   - More precise handling of `VALID` mode with `ceil_mode`

### Output Shape Calculation

**Explicit Padding (ceil_mode = 0, floor):**
```
H_out = floor((H_in + pad_shape[0] - kernel_shape[0]) / strides[0] + 1)
W_out = floor((W_in + pad_shape[1] - kernel_shape[1]) / strides[1] + 1)
```

**Explicit Padding (ceil_mode = 1, ceiling):**
```
H_out = ceil((H_in + pad_shape[0] - kernel_shape[0]) / strides[0] + 1)
W_out = ceil((W_in + pad_shape[1] - kernel_shape[1]) / strides[1] + 1)
```

**Auto Pad Formulas (v11):**

When `ceil_mode = 1` (enabled):
- `VALID`: `output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

When `ceil_mode = 0` (disabled):
- `VALID`: `output_spatial_shape[i] = floor((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])`

And pad shape for `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```

### Notes

- **Improved Auto Pad:** More explicit formulas for both `ceil_mode=0` and `ceil_mode=1` cases
- **Floor Mode with Auto Pad:** Explicit formulas for `VALID` and `SAME_UPPER/SAME_LOWER` modes

---

## AveragePool - Version 19

**Since Version:** 19  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

AveragePool consumes an input tensor X and applies average pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Average pooling consists of computing the average on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

**Key Improvements:**
- Added `dilations` attribute support
- Extended type support (bfloat16)
- Improved documentation

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that output_shape[i] = ceil(input_shape[i] / strides[i]) for each axis i. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER. |
| `ceil_mode` | INT | ❌ | `0` | Whether to use ceil or floor (default) to compute the output shape. |
| `count_include_pad` | INT | ❌ | `0` | Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad. |
| `dilations` | INTS | ❌ | - | Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |

**Input Count:** 1 input.

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used. |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 4 types

**Description:** Constrain input and output types to float tensors.

### Changes from v11

1. ✅ **Added `dilations` Attribute:** 
   - Dilation value along each spatial axis of filter
   - Defaults to 1 along each spatial axis if not present
   - Increases the effective kernel size without adding parameters

2. ✅ **Extended Type Support:** Added support for `bfloat16` (brain floating point 16-bit)

3. 📝 **Improved Documentation:** Better clarity on attribute descriptions

4. 📝 **Updated Output Shape Formulas:** 
   - Formulas now account for dilation
   - Effective kernel size: `effective_kernel = (kernel_size - 1) * dilation + 1`

### Output Shape Calculation

**Explicit Padding (ceil_mode = 0, floor):**
```
H_out = floor((H_in + pad_shape[0] - ((kernel_shape[0] - 1) * dilations[0] + 1)) / strides[0] + 1)
W_out = floor((W_in + pad_shape[1] - ((kernel_shape[1] - 1) * dilations[1] + 1)) / strides[1] + 1)
```

**Explicit Padding (ceil_mode = 1, ceiling):**
```
H_out = ceil((H_in + pad_shape[0] - ((kernel_shape[0] - 1) * dilations[0] + 1)) / strides[0] + 1)
W_out = ceil((W_in + pad_shape[1] - ((kernel_shape[1] - 1) * dilations[1] + 1)) / strides[1] + 1)
```

**Auto Pad Formulas (v19):**

When `ceil_mode = 1` (enabled):
- `VALID`: `output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

When `ceil_mode = 0` (disabled):
- `VALID`: `output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1`

And pad shape for `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
```

### Notes

- **Dilation:** Increases the effective kernel size without adding parameters
- **Effective Kernel Size:** `effective_kernel = (kernel_size - 1) * dilation + 1`
- **Example:** `kernel_size=3, dilation=2` → `effective_kernel = (3-1)*2 + 1 = 5`
- **Default Dilation:** 1 along each spatial axis (no dilation)
- **Type Support:** Now supports `bfloat16` in addition to standard float types
- **Backward Compatibility:** Functionally equivalent to v11, with additional type support and dilation

---

## AveragePool - Version 22

**Since Version:** 22  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

AveragePool consumes an input tensor X and applies average pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Average pooling consists of computing the average on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

**Key Improvements:**
- Extended type support (complex types, extended float types)
- Improved documentation
- Note about sliding windows in padded regions when `ceil_mode=True`

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that output_shape[i] = ceil(input_shape[i] / strides[i]) for each axis i. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER. |
| `ceil_mode` | INT | ❌ | `0` | Whether to use ceil or floor (default) to compute the output shape. |
| `count_include_pad` | INT | ❌ | `0` | Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad. |
| `dilations` | INTS | ❌ | - | Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |

**Input Count:** 1 input.

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used. |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 4 types

**Description:** Constrain input and output types to float tensors.

**Note:** The v22 documentation shows 4 float types, but implementations may support additional types including complex types and extended float types. Please refer to the ONNX specification for the complete type list.

### Changes from v19

1. 📝 **Documentation Update:** Improved documentation clarity
2. 📝 **Ceil Mode Note:** Added note that sliding windows that would start in the right padded region are ignored when `ceil_mode=True`
3. 🔄 **Type Constraints:** Documentation shows float types, but actual support may include complex types

### Output Shape Calculation

**Explicit Padding (ceil_mode = 0, floor):**
```
H_out = floor((H_in + pad_shape[0] - dilation[0] * (kernel_shape[0] - 1) - 1) / strides[0] + 1)
W_out = floor((W_in + pad_shape[1] - dilation[1] * (kernel_shape[1] - 1) - 1) / strides[1] + 1)
```

**Explicit Padding (ceil_mode = 1, ceiling):**
```
H_out = ceil((H_in + pad_shape[0] - dilation[0] * (kernel_shape[0] - 1) - 1) / strides[0] + 1)
W_out = ceil((W_in + pad_shape[1] - dilation[1] * (kernel_shape[1] - 1) - 1) / strides[1] + 1)
```

**Note:** When `ceil_mode=True`, sliding windows are allowed to go off-bounds if they start within the left padding or the input. Sliding windows that would start in the right padded region are ignored.

**Auto Pad Formulas (v22):**

When `ceil_mode = 1` (enabled):
- `VALID`: `output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

When `ceil_mode = 0` (disabled):
- `VALID`: `output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1`
- `SAME_UPPER` / `SAME_LOWER`: `output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1`

And pad shape for `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
```

### Notes

- **Ceil Mode Behavior:** When `ceil_mode=True`, sliding windows that would start in the right padded region are ignored
- **Type Support:** Documentation shows float types, but implementations may support additional types
- **Backward Compatibility:** Functionally equivalent to v19, with improved documentation

---

## Summary of Changes Across Versions

### Type Support Evolution

| Version | New Types Added | Total Types | Description |
|---------|----------------|-------------|-------------|
| **v1** | Base float types | 3 | `double`, `float`, `float16` |
| **v19** | `bfloat16` | 4 | Added `bfloat16` (brain floating point 16-bit) |
| **v22** | Documentation update | 4+ | Documentation shows float types, but implementation may support more |

### Attribute Evolution

| Attribute | v1 | v7 | v10 | v11 | v19 | v22 | Notes |
|-----------|----|----|-----|-----|-----|-----|-------|
| `auto_pad` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Improved documentation in v11+ |
| `ceil_mode` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | Added in v7 |
| `count_include_pad` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | No changes |
| `dilations` | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | Added in v19 |
| `kernel_shape` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Required, no changes |
| `pads` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | No changes |
| `strides` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Default value documented in v11+ |

### Output Shape Formula Evolution

| Version | Formula | Notes |
|---------|---------|-------|
| **v1** | `floor((H_in + pad_shape - kernel_shape) / stride + 1)` | Basic formula, no dilation |
| **v7** | Added `ceil()` option with `ceil_mode` | Can use ceiling instead of floor |
| **v10** | Improved auto_pad formulas for ceil_mode | Explicit formulas for both modes |
| **v11** | Improved auto_pad formulas with explicit ceil_mode handling | Explicit formulas for both ceil_mode cases |
| **v19+** | `floor/ceil((H_in + pad_shape - ((kernel_shape - 1) * dilation + 1)) / stride + 1)` | Accounts for dilation |

### Key Behavioral Notes

1. **Input Requirements:**
   - **Count:** 1 input (X)
   - **Shape:** `(N, C, H, W)` for 2D images or `(N, C, D1, D2, ..., Dn)` for n-dimensional data
   - **Type:** Float tensors (type support varies by version)

2. **Kernel Shape:**
   - **Required attribute:** Must be specified
   - **Format:** `[kH, kW]` for 2D, `[kD, kH, kW]` for 3D, etc.
   - **Determines:** Pooling window size along each spatial axis

3. **Stride:**
   - **Default:** 1 along each spatial axis (if not specified)
   - **Controls:** Step size of the pooling window
   - **Common:** Usually equals `kernel_size` for non-overlapping pooling

4. **Padding:**
   - **Format:** `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`
   - **Cannot be used with:** `auto_pad` (mutually exclusive)
   - **Default:** 0 along all axes if not specified

5. **Auto Pad:**
   - **NOTSET (default):** Use explicit `pads` attribute
   - **SAME_UPPER / SAME_LOWER:** Pad to achieve `ceil(input_shape[i] / strides[i])` output
   - **VALID:** No padding
   - **DEPRECATED:** Should use explicit padding when possible

6. **Ceil Mode:**
   - **Default:** 0 (use floor)
   - **When 1:** Use ceiling for output shape calculation
   - **Behavior:** Allows partial windows if they start within valid region

7. **Count Include Pad:**
   - **Default:** 0 (exclude padding from average)
   - **When 1:** Include padded zeros in average calculation
   - **Impact:** Affects edge values when padding is used

8. **Dilation:**
   - **Available:** v19+
   - **Default:** 1 along each spatial axis (no dilation)
   - **Effect:** Increases effective kernel size without adding parameters
   - **Formula:** `effective_kernel = (kernel_size - 1) * dilation + 1`

---

## Implementation Considerations

### For Converter Implementation

1. **Input Extraction:**
   - `X` (input data): Always present at `node_proto.input[0]`

2. **Attribute Extraction:**
   - Extract all attributes from `attrs` dictionary
   - Handle defaults:
     - `auto_pad`: Default `'NOTSET'`
     - `ceil_mode`: Default `0` (floor)
     - `count_include_pad`: Default `0` (exclude padding)
     - `dilations`: Default `1` per spatial axis (v19+, if not present)
     - `strides`: Default `1` per spatial axis (if not present)
     - `pads`: Default `0` per axis (if not present)
   - `kernel_shape`: Required attribute

3. **Auto Pad Handling:**
   - If `auto_pad != 'NOTSET'`, compute padding values:
     - `SAME_UPPER` / `SAME_LOWER`: Use formula `ceil(input_shape[i] / strides[i])`
     - `VALID`: No padding
   - Convert to explicit padding format for PyTorch
   - Create a `PadNode` before `AveragePoolNode` (as done in `pooling.py`)

4. **Padding Format Conversion:**
   - ONNX format: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`
   - PyTorch format (2D): `(left, right, top, bottom)`
   - Conversion: `(pads[1], pads[3], pads[0], pads[2])` for 2D
   - For 1D: `(pads[0], pads[1])`
   - For 3D: `(pads[1], pads[4], pads[0], pads[3], pads[2], pads[5])`

5. **Dimension Detection:**
   - Determine pool dimension from `kernel_shape`
   - 1D: `kernel_shape` has 1 element
   - 2D: `kernel_shape` has 2 elements
   - 3D: `kernel_shape` has 3 elements

6. **Node Creation:**
   - Create appropriate AveragePool node based on dimension:
     - `AveragePool1dNode` for 1D
     - `AveragePool2dNode` for 2D
     - `AveragePool3dNode` for 3D
   - Pass converted attributes (kernel_size, stride, padding, ceil_mode, count_include_pad)

7. **Type Support:**
   - Ensure converter handles all supported types for target opset version
   - Type support is additive (newer versions support all previous types)

8. **Validation:**
   - Verify `pads` and `auto_pad` are not both specified
   - Verify `kernel_shape` is provided
   - Verify output shape is valid (positive dimensions)

---

## Comparison with PyTorch

The ONNX AveragePool operator maps to PyTorch's `torch.nn.AvgPool1d`, `torch.nn.AvgPool2d`, or `torch.nn.AvgPool3d`:

### PyTorch AvgPool2d Parameters

```python
torch.nn.AvgPool2d(
    kernel_size,        # From kernel_shape attribute
    stride=None,         # From strides attribute (defaults to kernel_size)
    padding=0,          # From pads attribute (converted format)
    ceil_mode=False,    # From ceil_mode attribute (0=False, 1=True)
    count_include_pad=True  # From count_include_pad attribute (0=False, 1=True)
)
```

### Mapping Table

| ONNX | PyTorch | Notes |
|------|---------|-------|
| `X` | `input` | Input tensor |
| `kernel_shape` | `kernel_size` | Kernel dimensions |
| `strides` | `stride` | Stride values (defaults to kernel_size in PyTorch) |
| `pads` | `padding` | Padding (format conversion needed) |
| `ceil_mode` | `ceil_mode` | Direct mapping (0=False, 1=True) |
| `count_include_pad` | `count_include_pad` | Direct mapping (0=False, 1=True) |
| `dilations` | N/A | PyTorch AvgPool doesn't support dilation (v19+) |
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

3. **Dilation:**
   - ONNX v19+: Supports `dilations` attribute
   - PyTorch: `AvgPool2d` does NOT support dilation
   - **Note:** This is a limitation - ONNX models with dilation may need special handling

4. **Stride Default:**
   - ONNX: Defaults to 1 if not specified
   - PyTorch: Defaults to `kernel_size` if not specified
   - **Important:** Must explicitly set stride in converter

5. **Count Include Pad:**
   - ONNX: `0` = exclude, `1` = include (default is 0)
   - PyTorch: `False` = exclude, `True` = include (default is True)
   - **Important:** Invert the value when converting

6. **Type Support:**
   - ONNX v1: Float types only
   - ONNX v19+: Float + bfloat16
   - PyTorch: Supports all numeric types

---

## Examples

### Example 1: Basic 2D Average Pooling (v1)

```python
# ONNX Model
# Input X: [1, 3, 32, 32]
# Attributes: kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0]
# Output Y: [1, 3, 16, 16]

# PyTorch Equivalent
import torch.nn as nn
pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2,  # Must specify (ONNX default is 1, PyTorch default is kernel_size)
    padding=0,
    ceil_mode=False,
    count_include_pad=True  # ONNX default is 0 (False), PyTorch default is True
)
```

### Example 2: Average Pooling with Padding (v7)

```python
# ONNX Model
# Input X: [1, 3, 32, 32]
# Attributes: kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], ceil_mode=0
# Output Y: [1, 3, 32, 32]  (same size due to padding)

# PyTorch Equivalent
pool = nn.AvgPool2d(
    kernel_size=3,
    stride=1,
    padding=1,  # pads=[1,1,1,1] → padding=1
    ceil_mode=False,
    count_include_pad=True
)
```

### Example 3: Average Pooling with Auto Pad (v11)

```python
# ONNX Model
# Input X: [1, 3, 32, 32]
# Attributes: kernel_shape=[3, 3], strides=[2, 2], auto_pad='SAME_UPPER'
# Output Y: [1, 3, 16, 16]  # ceil(32/2) = 16

# PyTorch Equivalent
# Compute padding for SAME_UPPER with stride=2, kernel=3
# For SAME_UPPER: output = ceil(input / stride) = ceil(32/2) = 16
# padding = (output - 1) * stride + kernel - input
#         = (16 - 1) * 2 + 3 - 32 = 30 + 3 - 32 = 1
pool = nn.AvgPool2d(
    kernel_size=3,
    stride=2,
    padding=1,  # Computed from auto_pad
    ceil_mode=False,
    count_include_pad=True
)
```

### Example 4: Average Pooling with Ceil Mode (v10)

```python
# ONNX Model
# Input X: [1, 3, 5, 5]
# Attributes: kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0], ceil_mode=1
# Output Y: [1, 3, 3, 3]  # ceil((5-2)/2 + 1) = ceil(2.5) = 3

# PyTorch Equivalent
pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2,
    padding=0,
    ceil_mode=True,  # ceil_mode=1 → True
    count_include_pad=True
)
```

### Example 5: Average Pooling with Count Include Pad (v1)

```python
# ONNX Model
# Input X: [1, 1, 3, 3]
# Attributes: kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], count_include_pad=0
# Output Y: [1, 1, 3, 3]

# PyTorch Equivalent
pool = nn.AvgPool2d(
    kernel_size=3,
    stride=1,
    padding=1,
    ceil_mode=False,
    count_include_pad=False  # count_include_pad=0 → False
)

# With count_include_pad=False, only non-padded values are counted in the average
# Edge positions will have different values compared to count_include_pad=True
```

### Example 6: Average Pooling with Dilation (v19)

```python
# ONNX Model
# Input X: [1, 3, 32, 32]
# Attributes: kernel_shape=[3, 3], strides=[1, 1], pads=[2, 2, 2, 2], dilations=[2, 2]
# Output Y: [1, 3, 32, 32]

# PyTorch Equivalent
# NOTE: PyTorch AvgPool2d does NOT support dilation!
# This is a limitation - need to use MaxPool2d or implement custom solution
# For now, we can approximate by using a larger kernel or different approach

# Effective kernel size with dilation=2: (3-1)*2 + 1 = 5
# So we could use kernel_size=5 instead:
pool = nn.AvgPool2d(
    kernel_size=5,  # Approximate dilated 3x3 kernel
    stride=1,
    padding=2,
    ceil_mode=False,
    count_include_pad=True
)

# However, this is not exactly equivalent - dilation creates gaps in the kernel
# which average pooling doesn't naturally support
```

**Important Note:** PyTorch's `AvgPool2d` does not support dilation. ONNX models with `dilations > 1` (available in v19+) may need special handling or conversion to a different operation.

---

## References

- [ONNX AveragePool Operator Documentation](https://onnx.ai/onnx/operators/onnx__AveragePool.html)
- [PyTorch AvgPool2d Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)
- [PyTorch AvgPool1d Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html)
- [PyTorch AvgPool3d Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool3d.html)
- [Average Pooling in Deep Learning - Detailed Guide](../AVERAGEPOOL2D_DETAILED_GUIDE.md)

