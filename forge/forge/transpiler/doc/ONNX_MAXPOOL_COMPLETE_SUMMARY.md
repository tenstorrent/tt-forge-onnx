# ONNX MaxPool Operator - Complete Summary

**Reference:** [ONNX MaxPool Operator Documentation](https://onnx.ai/onnx/operators/onnx__MaxPool.html)

## Overview

The **MaxPool** operator performs max pooling operations on input tensors. It consumes an input tensor X and applies max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Max pooling consists of computing the maximum on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

The output spatial shape is calculated differently depending on whether explicit padding is used (where `pads` is employed) or auto padding is used (where `auto_pad` is utilized).

---

## Version History

| Version | Since | Key Changes |
|---------|-------|-------------|
| **1** | 1 | Initial version. Basic attributes: `auto_pad`, `kernel_shape`, `pads`, `strides`. Single output Y. Float types only. |
| **8** | 8 | Added optional `Indices` output. Added `storage_order` attribute. |
| **10** | 10 | Added `ceil_mode` attribute. Added `dilations` attribute. Updated formulas to account for dilation. |
| **11** | 11 | Improved output shape formulas with explicit `ceil_mode` handling for `auto_pad` modes. Better documentation. |
| **12** | 12 | Extended type support: added `int8` and `uint8` tensors. |
| **22** | 22 | Extended type support: added `bfloat16`. Improved documentation. Note about sliding windows in padded regions being ignored. |

---

## MaxPool - Version 1

**Since Version:** 1  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

MaxPool consumes an input tensor X and applies max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. The output spatial shape will be following:

```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
```

Where `pad_shape[i]` is the sum of pads along axis i.

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:

- **VALID:** `output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])`
- **SAME_UPPER or SAME_LOWER:** `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:

```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```

The output of each pooling window is maximum number of elements exclude pad.

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input. In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. |
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
| `Y` | T | Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used. |

**Output Count:** 1 output.

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Description:** Constrain input and output types to float tensors.

---

## MaxPool - Version 8

**Since Version:** 8  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

MaxPool consumes an input tensor X and applies max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. The output spatial shape will be following:

```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
```

Where `pad_shape[i]` is the sum of pads along axis i.

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:

- **VALID:** `output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])`
- **SAME_UPPER or SAME_LOWER:** `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:

```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```

The output of each pooling window is maximum number of elements exclude pad.

**Key Changes from v1:**
- Added optional `Indices` output tensor
- Added `storage_order` attribute for index computation

**Note:** `ceil_mode` attribute was **NOT** introduced in v8. It was introduced in v10.

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input. In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `storage_order` | INT | ❌ | `0` | The storage order of the tensor. 0 is row major, and 1 is column major. This attribute is used only to convert an n-tuple index value into a single integer value for producing the second output. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |

**Input Count:** 1 input.

### Outputs

Between 1 and 2 outputs.

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used. |
| `Indices` (optional) | I | Indices tensor from max pooling across the input tensor. The dimensions of indices are the same as output tensor. The values in indices are the indices of the selected values during pooling. The indices are computed as flatten 1-D tensor, and the indices do not consider padding. So the values in indices are in [0, N x C x D1 x … x Dn). |

**Output Count:** 1 or 2 outputs (Indices is optional).

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**I** in:
- `tensor(int64)`

**Description:** Constrain input and output types to float tensors. Constrain index tensor to int64.

---

## MaxPool - Version 10

**Since Version:** 10  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

MaxPool consumes an input tensor X and applies max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. The output spatial shape will be following:

```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
```

or

```
output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
```

if `ceil_mode` is enabled.

Where `pad_shape[i]` is the sum of pads along axis i.

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:

- **VALID:** `output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])`
- **SAME_UPPER or SAME_LOWER:** `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:

```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
```

The output of each pooling window is maximum number of elements exclude pad.

**Key Changes from v8:**
- Added `ceil_mode` attribute for controlling output shape calculation
- Added `dilations` attribute support for dilated max pooling
- Updated output shape formulas to account for dilation

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input. In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. |
| `ceil_mode` | INT | ❌ | `0` | Whether to use ceil or floor (default) to compute the output shape. |
| `dilations` | INTS | ❌ | - | Dilation value along each spatial axis of filter. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `storage_order` | INT | ❌ | `0` | The storage order of the tensor. 0 is row major, and 1 is column major. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. |

### Inputs

Same as v8.

### Outputs

Same as v8.

### Type Constraints

Same as v8.

---

## MaxPool - Version 11

**Since Version:** 11  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

MaxPool consumes an input tensor X and applies max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. The output spatial shape will be following:

```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
```

or

```
output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
```

if `ceil_mode` is enabled.

Where `pad_shape[i]` is the sum of pads along axis i.

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:

- **VALID:** `output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])`
- **SAME_UPPER or SAME_LOWER:** `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:

```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
```

The output of each pooling window is maximum number of elements exclude pad.

**Key Changes from v10:**
- Improved output shape formulas with explicit `ceil_mode` handling for `auto_pad` modes
- Better documentation of `auto_pad` behavior with `ceil_mode`

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input. In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. |
| `ceil_mode` | INT | ❌ | `0` | Whether to use ceil or floor (default) to compute the output shape. |
| `dilations` | INTS | ❌ | - | Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `storage_order` | INT | ❌ | `0` | The storage order of the tensor. 0 is row major, and 1 is column major. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis. |

### Inputs

Same as v10.

### Outputs

Same as v10.

### Type Constraints

Same as v10.

---

## MaxPool - Version 12

**Since Version:** 12  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

MaxPool consumes an input tensor X and applies max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. The output spatial shape calculation is the same as v11.

**Key Changes from v11:**
- Extended type support to include `int8` and `uint8` tensors

### Attributes

Same as v11.

### Inputs

Same as v11.

### Outputs

Same as v11.

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(int8)`
- `tensor(uint8)`

**I** in:
- `tensor(int64)`

**Description:** Constrain input and output types to float and 8 bit tensors. Constrain index tensor to int64.

---

## MaxPool - Version 22

**Since Version:** 22  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

MaxPool consumes an input tensor X and applies max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Max pooling consists of computing the max on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

The output spatial shape is calculated differently depending on whether explicit padding is used, where `pads` is employed, or auto padding is used, where `auto_pad` is utilized.

**With explicit padding** (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):

```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
```

or

```
output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
```

if `ceil_mode` is enabled.

Where `pad_shape[i]` is the sum of pads along axis i. **Sliding windows that would start in the right padded region are ignored.**

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following **when `ceil_mode` is enabled**:

- **VALID:** `output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])`
- **SAME_UPPER or SAME_LOWER:** `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])`

or **when `ceil_mode` is disabled** (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):

- **VALID:** `output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1`
- **SAME_UPPER or SAME_LOWER:** `output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1`

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:

```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
```

The output of each pooling window is maximum number of elements exclude pad.

**Key Changes from v12:**
- Extended type support to include `bfloat16`
- Improved documentation
- Note about sliding windows in padded regions being ignored

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_pad` | STRING | ❌ | `'NOTSET'` | auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that output_shape[i] = ceil(input_shape[i] / strides[i]) for each axis i. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER. |
| `ceil_mode` | INT | ❌ | `0` | Whether to use ceil or floor (default) to compute the output shape. |
| `dilations` | INTS | ❌ | - | Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis. |
| `kernel_shape` | INTS | ✅ | - | The size of the kernel along each axis. |
| `pads` | INTS | ❌ | - | Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis. |
| `storage_order` | INT | ❌ | `0` | The storage order of the tensor. 0 is row major, and 1 is column major. This attribute is used only to convert an n-tuple index value into a single integer value for producing the second output. |
| `strides` | INTS | ❌ | - | Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE …]. |

**Input Count:** 1 input.

### Outputs

Between 1 and 2 outputs.

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used. |
| `Indices` (optional) | I | Indices tensor from max pooling across the input tensor. The dimensions of indices are the same as output tensor. The values in indices are the indices of the selected values during pooling. The indices are computed as flatten 1-D tensor, and the indices do not consider padding. So the values in indices are in [0, N x C x D1 x … x Dn). |

**Output Count:** 1 or 2 outputs (Indices is optional).

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(int8)`
- `tensor(uint8)`

**I** in:
- `tensor(int64)`

**Description:** Constrain input and output types to float and 8 bit tensors. Constrain index tensor to int64.

### Important Notes

1. **Sliding Windows in Padded Regions:** Sliding windows that would start in the right padded region are ignored. This affects output shape calculation when using explicit padding with `ceil_mode=1`.

2. **Auto Pad with Ceil Mode:** In v22, `auto_pad` modes respect `ceil_mode`:
   - When `ceil_mode=1`: Uses ceil formulas (same as v10-v11)
   - When `ceil_mode=0`: Uses floor formulas (new in v22)

3. **Explicit Padding:** When using explicit padding (`auto_pad='NOTSET'`), `ceil_mode` controls whether to use floor or ceil for output shape calculation.

---

## Detailed Changes Between Opset Versions

### Version 1 → Version 8

**What Changed:**

1. **New Output: `Indices` (optional)**
   - Added optional second output tensor of type `int64`
   - Contains the indices of the maximum values selected during pooling
   - Dimensions match the output tensor Y
   - Indices are computed as a flattened 1-D tensor and do not consider padding
   - Values range from `[0, N x C x D1 x … x Dn)`

2. **New Attribute: `storage_order`**
   - Type: `INT`, Default: `0`
   - Used only to convert n-tuple index values into single integer values for the `Indices` output
   - `0` = row major (C-style), `1` = column major (Fortran-style)

**What Stayed the Same:**
- Output shape calculation formulas (unchanged)
- `auto_pad` behavior (unchanged)
- Type constraints (float, float16, double only)
- All other attributes (`auto_pad`, `kernel_shape`, `pads`, `strides`)

**Impact:**
- Enables max unpooling operations (using indices to reverse pooling)
- No breaking changes for existing models using only the Y output

---

### Version 8 → Version 10

**What Changed:**

1. **New Attribute: `ceil_mode`**
   - Type: `INT`, Default: `0`
   - Controls output shape calculation: `0` = floor (default), `1` = ceil
   - **Important:** Only affects explicit padding (`auto_pad='NOTSET'`)
   - When `auto_pad` is set, `ceil_mode` is ignored (always uses ceil behavior in v10-v11)

2. **New Attribute: `dilations`**
   - Type: `INTS`
   - Dilation value along each spatial axis of filter
   - If not present, defaults to 1 along each spatial axis
   - Increases effective kernel size: `effective_kernel = (kernel_size - 1) * dilation + 1`

3. **Updated Output Shape Formulas:**
   - **Explicit padding with `ceil_mode=0`:**
     ```
     output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
     ```
   - **Explicit padding with `ceil_mode=1`:**
     ```
     output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
     ```
   - **Auto pad formulas updated to account for dilation:**
     - VALID: `ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])`
     - SAME_UPPER/SAME_LOWER: `ceil(input_spatial_shape[i] / strides_spatial_shape[i])`
   - **Padding calculation updated:**
     ```
     pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
     ```

**What Stayed the Same:**
- Type constraints (float, float16, double only)
- `Indices` output behavior
- `storage_order` attribute

**Impact:**
- Enables more flexible output shape control with `ceil_mode`
- Enables dilated max pooling for larger receptive fields
- Formulas now account for dilation in all calculations

---

### Version 10 → Version 11

**What Changed:**

1. **Improved Documentation:**
   - Better explanation of `auto_pad` behavior with `ceil_mode`
   - Clarified that `auto_pad` modes always use ceil behavior regardless of `ceil_mode` (in v10-v11)

2. **Enhanced Attribute Descriptions:**
   - `dilations`: Added note "If not present, the dilation defaults to 1 along each spatial axis"
   - `strides`: Added note "If not present, the stride defaults to 1 along each spatial axis"

**What Stayed the Same:**
- All formulas (same as v10)
- All attributes (same as v10)
- Type constraints (same as v10)
- Output behavior (same as v10)

**Impact:**
- Primarily documentation improvements
- No functional changes
- Better clarity for implementers

---

### Version 11 → Version 12

**What Changed:**

1. **Extended Type Support:**
   - Added `tensor(int8)` to type constraints
   - Added `tensor(uint8)` to type constraints
   - Enables quantization-aware max pooling operations

**What Stayed the Same:**
- All formulas (same as v11)
- All attributes (same as v11)
- Output behavior (same as v11)

**Impact:**
- Enables quantized models to use MaxPool
- Supports 8-bit integer operations
- No breaking changes for float models

---

### Version 12 → Version 22

**What Changed:**

1. **Extended Type Support:**
   - Added `tensor(bfloat16)` to type constraints
   - Enables bfloat16 precision operations

2. **Improved Documentation:**
   - Better explanation of output shape calculation
   - Clearer distinction between explicit padding and auto padding

3. **New Behavior: Sliding Windows in Padded Regions**
   - **Important:** Sliding windows that would start in the right padded region are ignored
   - Affects output shape calculation when using explicit padding with `ceil_mode=1`

4. **Changed Behavior: Auto Pad with Ceil Mode**
   - **Breaking Change:** `auto_pad` modes now respect `ceil_mode` attribute
   - **When `ceil_mode=1` (enabled):**
     - VALID: `ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])`
     - SAME_UPPER/SAME_LOWER: `ceil(input_spatial_shape[i] / strides_spatial_shape[i])`
   - **When `ceil_mode=0` (disabled):**
     - VALID: `floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1`
     - SAME_UPPER/SAME_LOWER: `floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1`
   - **Note:** This is different from v10-v11, where `auto_pad` always used ceil behavior regardless of `ceil_mode`

5. **Updated Output Shape Formula for Explicit Padding:**
   - Changed from: `((kernel_spatial_shape[i] - 1) * dilations[i] + 1)`
   - To: `dilation[i] * (kernel_shape[i] - 1)`
   - These are mathematically equivalent, but the notation is clearer

**What Stayed the Same:**
- All attributes (same as v12)
- `Indices` output behavior
- `storage_order` attribute

**Impact:**
- **Breaking Change:** Models using `auto_pad` with `ceil_mode=0` will produce different output shapes in v22 compared to v10-v11
- Enables bfloat16 precision
- More consistent behavior between explicit padding and auto padding

---

## Summary Table of Changes

| Transition | New Attributes | New Outputs | Formula Changes | Type Changes | Behavior Changes |
|-----------|---------------|-------------|----------------|--------------|------------------|
| **1 → 8** | `storage_order` | `Indices` (optional) | None | None | None |
| **8 → 10** | `ceil_mode`, `dilations` | None | Updated to account for dilation and ceil_mode | None | Ceil mode support for explicit padding |
| **10 → 11** | None | None | None | None | Documentation only |
| **11 → 12** | None | None | None | Added int8, uint8 | None |
| **12 → 22** | None | None | Notation improved | Added bfloat16 | Auto pad now respects ceil_mode; sliding windows in padded regions ignored |

---

## Key Behavioral Differences

### Auto Pad Behavior

**Versions 1-11:**
- `auto_pad` modes (SAME_UPPER/SAME_LOWER/VALID) always use ceil behavior, regardless of `ceil_mode`
- `ceil_mode` only affects explicit padding (`auto_pad='NOTSET'`)

**Version 22:**
- `auto_pad` modes respect `ceil_mode`:
  - When `ceil_mode=1`: Uses ceil formulas (same as v10-v11)
  - When `ceil_mode=0`: Uses floor formulas (new behavior)

### Explicit Padding

**All versions with `ceil_mode` (v10+):**
- When `ceil_mode=0`: Uses `floor()` for output shape calculation
- When `ceil_mode=1`: Uses `ceil()` for output shape calculation

**Versions without `ceil_mode` (v1, v8):**
- Always uses `floor()` for output shape calculation

---

## Comparison with PyTorch

### PyTorch MaxPool Operations

- `torch.nn.MaxPool1d` - 1D max pooling
- `torch.nn.MaxPool2d` - 2D max pooling  
- `torch.nn.MaxPool3d` - 3D max pooling
- `torch.nn.functional.max_pool1d`, `max_pool2d`, `max_pool3d` - Functional API

### Key Differences

1. **Indices Output:**
   - **ONNX:** Optional second output `Indices` available from v8+
   - **PyTorch:** `return_indices` parameter, returns tuple `(output, indices)`

2. **Dilation:**
   - **ONNX:** `dilations` attribute available from v10+
   - **PyTorch:** `dilation` parameter supported in all MaxPool operations

3. **Type Support:**
   - **ONNX v22:** Supports float, float16, bfloat16, double, int8, uint8
   - **PyTorch:** Supports float, float16, bfloat16, double, int8, uint8 (similar)

4. **Storage Order:**
   - **ONNX:** `storage_order` attribute for index computation (v8+)
   - **PyTorch:** Uses row-major (C-style) ordering by default

5. **Padding:**
   - **ONNX:** `pads` format: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`
   - **PyTorch:** `padding` can be int or tuple, symmetric padding

6. **Auto Pad:**
   - **ONNX:** `auto_pad` attribute with modes: NOTSET, SAME_UPPER, SAME_LOWER, VALID
   - **PyTorch:** No direct equivalent, need to compute padding manually

7. **Ceil Mode with Auto Pad:**
   - **ONNX v1-11:** When `auto_pad` is set, always uses ceil behavior regardless of `ceil_mode`
   - **ONNX v22:** When `auto_pad` is set, respects `ceil_mode`
   - **PyTorch:** `ceil_mode` always applies when set

### Attribute Mapping

| ONNX Attribute | PyTorch Parameter | Notes |
|----------------|-------------------|-------|
| `kernel_shape` | `kernel_size` | Required in both |
| `strides` | `stride` | Default: 1 in both |
| `pads` | `padding` | Format conversion needed |
| `dilations` | `dilation` | Available from v10+ in ONNX |
| `ceil_mode` | `ceil_mode` | Available from v10+ in ONNX |
| `auto_pad` | N/A | PyTorch doesn't have auto_pad, need to compute padding manually |
| `storage_order` | N/A | PyTorch uses row-major by default |
| `Indices` output | `return_indices=True` | ONNX v8+, PyTorch always available |

---

## Examples

### Example 1: Basic 2D Max Pooling (v1)

```python
import onnx
from onnx import helper

node = helper.make_node(
    'MaxPool',
    inputs=['X'],
    outputs=['Y'],
    kernel_shape=[2, 2],
    strides=[2, 2],
    pads=[0, 0, 0, 0]
)

# Input: [1, 3, 32, 32]
# Output: [1, 3, 16, 16]
```

### Example 2: Max Pooling with Indices (v8+)

```python
import onnx
from onnx import helper

node = helper.make_node(
    'MaxPool',
    inputs=['X'],
    outputs=['Y', 'Indices'],
    kernel_shape=[3, 3],
    strides=[2, 2],
    pads=[1, 1, 1, 1],
    storage_order=0
)

# Input: [1, 3, 32, 32]
# Output Y: [1, 3, 16, 16]
# Output Indices: [1, 3, 16, 16] (int64)
```

### Example 3: Max Pooling with Ceil Mode (v10+)

```python
import onnx
from onnx import helper

node = helper.make_node(
    'MaxPool',
    inputs=['X'],
    outputs=['Y'],
    kernel_shape=[3, 3],
    strides=[2, 2],
    pads=[0, 0, 0, 0],
    ceil_mode=1  # Use ceil for output shape
)

# Input: [1, 3, 31, 31]
# With ceil_mode=0: Output would be [1, 3, 15, 15]
# With ceil_mode=1: Output is [1, 3, 16, 16]
```

### Example 4: Max Pooling with Dilation (v10+)

```python
import onnx
from onnx import helper

node = helper.make_node(
    'MaxPool',
    inputs=['X'],
    outputs=['Y'],
    kernel_shape=[3, 3],
    strides=[1, 1],
    pads=[2, 2, 2, 2],
    dilations=[2, 2],
    ceil_mode=0
)

# Input: [1, 3, 32, 32]
# Effective kernel: 5x5 (with dilation 2)
# Output: [1, 3, 32, 32]
```

### Example 5: Max Pooling with Auto Pad (v1+)

```python
import onnx
from onnx import helper

# v1-v11: auto_pad always uses ceil behavior
node = helper.make_node(
    'MaxPool',
    inputs=['X'],
    outputs=['Y'],
    kernel_shape=[3, 3],
    strides=[1, 1],
    auto_pad='SAME_UPPER'
)

# v22: auto_pad respects ceil_mode
node_v22 = helper.make_node(
    'MaxPool',
    inputs=['X'],
    outputs=['Y'],
    kernel_shape=[3, 3],
    strides=[1, 1],
    auto_pad='SAME_UPPER',
    ceil_mode=0  # Uses floor formulas in v22
)
```

---

## Implementation Considerations

1. **Indices Output:** When implementing MaxPool with indices, ensure proper handling of the `storage_order` attribute for converting multi-dimensional indices to flat indices. The indices are computed as a flattened 1-D tensor and do not consider padding.

2. **Dilation Support:** Dilation increases the effective kernel size. Ensure proper calculation of output shapes when dilation > 1. The effective kernel size is `(kernel_size - 1) * dilation + 1`.

3. **Type Support:** Different versions support different types. Ensure type checking based on opset version:
   - v1-v11: float, float16, double
   - v12+: float, float16, double, int8, uint8
   - v22+: float, float16, bfloat16, double, int8, uint8

4. **Auto Pad:** When `auto_pad` is used, padding values are computed automatically. Do not use explicit `pads` attribute simultaneously.
   - **v1-v11:** `auto_pad` always uses ceil behavior, regardless of `ceil_mode`
   - **v22:** `auto_pad` respects `ceil_mode` (uses floor formulas when `ceil_mode=0`)

5. **Ceil Mode:** The `ceil_mode` attribute (available from v10+) affects output shape calculation:
   - When using explicit padding (`auto_pad='NOTSET'`): `ceil_mode` controls floor vs ceil
   - When using `auto_pad` (v1-v11): `ceil_mode` is ignored, always uses ceil
   - When using `auto_pad` (v22): `ceil_mode` is respected

6. **Sliding Windows:** In v22+, sliding windows that would start in the right padded region are ignored. This affects the output shape calculation when using explicit padding with `ceil_mode=1`.

7. **Padding Format:** ONNX uses `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]` format, while PyTorch uses different formats. Ensure proper conversion when mapping to PyTorch operations.

8. **Storage Order:** The `storage_order` attribute (v8+) is only used for computing the Indices output. It determines how multi-dimensional indices are converted to flat indices (0 = row-major, 1 = column-major).

---

## References

- [ONNX MaxPool Operator Documentation](https://onnx.ai/onnx/operators/onnx__MaxPool.html)
- [PyTorch MaxPool2d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
- [PyTorch MaxPool1d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html)
- [PyTorch MaxPool3d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html)
