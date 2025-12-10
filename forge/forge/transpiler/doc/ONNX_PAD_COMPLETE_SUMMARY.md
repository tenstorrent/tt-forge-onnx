# ONNX Pad Operator - Complete Summary

## Overview

The **Pad** operator adds padding to tensors along specified axes. It supports multiple padding modes (`constant`, `reflect`, `edge`, `wrap`) and allows selective padding of specific axes (since opset 18+).

The operator takes an input tensor, a `pads` tensor specifying padding amounts, an optional `constant_value` for constant mode, an optional `axes` tensor to specify which axes to pad, and a `mode` attribute to select the padding strategy.

## Version History

| Version | Since | Shape Inference | Function | Key Changes |
|---------|-------|----------------|----------|-------------|
| 1 | 1 | ❌ | ❌ | Initial version, `paddings` attribute, 3 modes (constant, reflect, edge), float types only |
| 2 | 2 | ✅ | ❌ | `paddings` → `pads` input, added `constant_value` input, extended to numeric types |
| 11 | 11 | ✅ | ❌ | Extended type support (bool, complex, bfloat16), improved `constant_value` defaults |
| 13 | 13 | ✅ | ❌ | Added string type support |
| 18 | 18 | ✅ | ❌ | Added `axes` input for selective axis padding, added `wrap` mode |
| 19 | 19 | ✅ | ❌ | Added `wrap` mode support, extended float type support |
| 21 | 21 | ✅ | ❌ | Extended float type support (float8e5m2fnuz, float8e8m0) |
| 23 | 23 | ✅ | ❌ | Extended integer type support (int2, int4, uint2, uint4) |
| 24 | 24 | ✅ | ❌ | Extended to IRv12 types |
| 25 | 25 | ✅ | ❌ | Extended to IRv13 types |

---

## Pad - Version 1

**Since Version:** 1  
**Shape Inference:** ❌ False  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Given data tensor, paddings, mode, and value. The operator adds padding elements to the input tensor according to the specified padding amounts and mode.

**Example:** Insert 0 paddings to the beginning of the second dimension.

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]
paddings = [0, 0, 2, 0]
output = [
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 0.0, 2.3, 3.4],
    [0.0, 0.0, 4.5, 5.7],
]
```

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | STRING | ❌ | `'constant'` | Three modes: constant(default), reflect, edge |
| `paddings` | INTS | ✅ | - | List of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D it is the number of pixels. paddings rank should be double of the input's rank. paddings format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added at the beginning of axis i and xi_end, the number of pixels added at the end of axis i. |
| `value` | FLOAT | ❌ | `0.0` | One float, indicates the value to be filled, default is 0 |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | Input tensor |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | Tensor after padding |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Shape Inference:** Not supported in v1
- **Padding Format:** `paddings` is an attribute (not an input), format: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`
- **Modes:** Supports `constant`, `reflect`, and `edge` modes
- **Type Support:** Limited to floating-point types only (double, float, float16)
- **Negative Padding:** Not explicitly mentioned but may be supported

---

## Pad - Version 2

**Since Version:** 2  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Given a tensor containing the data to be padded (data), a tensor containing the number of start and end pad values for axis (pads), (optionally) a mode, and (optionally) constant_value, a padded tensor (output) is generated.

The three supported modes are (similar to corresponding modes supported by numpy.pad):
- **constant(default)** - pads with a given constant value as specified by constant_value (which defaults to 0)
- **reflect** - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
- **edge** - pads with the edge values of array

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | STRING | ❌ | `'constant'` | Supported modes: constant(default), reflect, edge |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | Input tensor |
| `pads` | tensor(int64) | Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. pads should be a 1D tensor of shape [2 * input_rank]. pads format should be: [x1_begin, x2_begin,…,x1_end, x2_end,…], where xi_begin is the number of pad values added at the beginning of axis i and xi_end, the number of pad values added at the end of axis i. |
| `constant_value` | T (optional) | (Optional) A scalar value to be used if the mode chosen is constant (by default it is 0) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | Tensor after padding |

### Type Constraints

**T** in:
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

**Total:** 11 types

**Description:** Constrain input and output to only numeric types.

### Changes from v1

1. ✅ **Shape Inference:** Added shape inference support
2. 🔄 **`paddings` → `pads`:** Changed from attribute to input tensor
3. ➕ **Added `constant_value` Input:** Made `constant_value` an optional input instead of attribute
4. 📊 **Extended Type Support:** Extended from 3 float types to 11 numeric types (added int8, int16, int32, int64, uint8, uint16, uint32, uint64)
5. 📝 **Improved Documentation:** Better description of padding modes

### Notes

- **Shape Inference:** Now supported, allowing automatic shape propagation
- **Input-based Pads:** `pads` is now a tensor input, making it more flexible
- **Type Flexibility:** Supports integer types in addition to float types
- **Negative Padding:** Explicitly supports negative padding values to remove elements

---

## Pad - Version 11

**Since Version:** 11  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Given a tensor containing the data to be padded (data), a tensor containing the number of start and end pad values for axis (pads), (optionally) a mode, and (optionally) constant_value, a padded tensor (output) is generated.

The three supported modes are (similar to corresponding modes supported by numpy.pad):
- **constant(default)** - pads with a given constant value as specified by constant_value (which defaults to 0, empty string, or False)
- **reflect** - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
- **edge** - pads with the edge values of array

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | STRING | ❌ | `'constant'` | Supported modes: constant(default), reflect, edge |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | Input tensor |
| `pads` | tensor(int64) | Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. pads should be a 1D tensor of shape [2 * input_rank]. pads format should be: [x1_begin, x2_begin,…,x1_end, x2_end,…], where xi_begin is the number of pad values added at the beginning of axis i and xi_end, the number of pad values added at the end of axis i. |
| `constant_value` | T (optional) | (Optional) A scalar value to be used if the mode chosen is constant (by default it is 0, empty string or False) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | Tensor after padding |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(bool)`
- `tensor(complex128)`
- `tensor(complex64)`
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

**Total:** 15 types

**Description:** Constrain input and output types to all tensor types.

### Changes from v2

1. ➕ **Added `bfloat16` Type:** Extended type support to include Brain Floating Point 16-bit format
2. ➕ **Added `bool` Type:** Added boolean tensor support
3. ➕ **Added Complex Types:** Added `complex64` and `complex128` support
4. 📊 **Type Count:** Increased from 11 to 15 types
5. 📝 **Improved `constant_value` Defaults:** Clarified that `constant_value` defaults to 0 (numeric), empty string (string), or False (bool)

### Notes

- **bfloat16 Support:** Added support for the bfloat16 floating-point format
- **Boolean Support:** Can pad boolean tensors with False as default
- **Complex Number Support:** Supports padding of complex-valued tensors
- **Type-Specific Defaults:** `constant_value` defaults vary by type (0 for numeric, empty string for string, False for bool)

---

## Pad - Version 13

**Since Version:** 13  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Given a tensor containing the data to be padded (data), a tensor containing the number of start and end pad values for axis (pads), (optionally) a mode, and (optionally) constant_value, a padded tensor (output) is generated.

The three supported modes are (similar to corresponding modes supported by numpy.pad):
- **constant(default)** - pads with a given constant value as specified by constant_value (which defaults to 0, empty string, or False)
- **reflect** - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
- **edge** - pads with the edge values of array

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | STRING | ❌ | `'constant'` | Supported modes: constant(default), reflect, edge |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | Input tensor |
| `pads` | tensor(int64) | Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. pads should be a 1D tensor of shape [2 * input_rank]. pads format should be: [x1_begin, x2_begin,…,x1_end, x2_end,…], where xi_begin is the number of pad values added at the beginning of axis i and xi_end, the number of pad values added at the end of axis i. |
| `constant_value` | T (optional) | (Optional) A scalar value to be used if the mode chosen is constant (by default it is 0, empty string or False) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | Tensor after padding |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(bool)`
- `tensor(complex128)`
- `tensor(complex64)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(int16)`
- `tensor(int32)`
- `tensor(int64)`
- `tensor(int8)`
- `tensor(string)`
- `tensor(uint16)`
- `tensor(uint32)`
- `tensor(uint64)`
- `tensor(uint8)`

**Total:** 16 types

**Description:** Constrain input and output types to all tensor types.

### Changes from v11

1. ➕ **Added `string` Type:** Extended type support to include string tensors
2. 📊 **Type Count:** Increased from 15 to 16 types
3. 📝 **No Functional Changes:** Behavior remains the same, only type support extended

### Notes

- **String Support:** Can pad string tensors with empty string as default `constant_value`
- **Type-Specific Defaults:** `constant_value` defaults to empty string for string tensors

---

## Pad - Version 18

**Since Version:** 18  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Given a tensor containing the data to be padded (data), a tensor containing the number of start and end pad values for axis (pads), (optionally) a mode, and (optionally) constant_value, a padded tensor (output) is generated.

The three supported modes are (similar to corresponding modes supported by numpy.pad):
- **constant(default)** - pads with a given constant value as specified by constant_value (which defaults to 0, empty string, or False)
- **reflect** - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
- **edge** - pads with the edge values of array

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | STRING | ❌ | `'constant'` | Supported modes: constant(default), reflect, edge |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | Input tensor |
| `pads` | tensor(int64) | Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. pads should be a 1D tensor of shape [2 * num_axes] where num_axes refers to the number of elements in the axes input or the input rank if axes are not provided explicitly. pads format should be: [x1_begin, x2_begin, …, x1_end, x2_end,…], where xi_begin is the number of pad values added at the beginning of axis axes[i] and xi_end, the number of pad values added at the end of axis axes[i]. |
| `constant_value` | T (optional) | (Optional) A scalar value to be used if the mode chosen is constant (by default it is 0, empty string or False) |
| `axes` | Tind (optional) | 1-D tensor of axes that pads apply to. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an axis is repeated. If not provided, all axes are assumed ([0, 1, ..., input_rank-1]). |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | Tensor after padding |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(bool)`
- `tensor(complex128)`
- `tensor(complex64)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(int16)`
- `tensor(int32)`
- `tensor(int64)`
- `tensor(int8)`
- `tensor(string)`
- `tensor(uint16)`
- `tensor(uint32)`
- `tensor(uint64)`
- `tensor(uint8)`

**Total:** 16 types

**Tind** in:
- `tensor(int32)`
- `tensor(int64)`

**Description:** Constrain input and output types to all tensor types. Constrain indices to integer types.

### Changes from v13

1. ➕ **Added `axes` Input:** Allows selective padding of specific axes instead of all axes
2. 📝 **Updated `pads` Description:** `pads` length is now `2 * num_axes` where `num_axes` comes from `axes` input or input rank
3. 🔄 **Selective Padding:** Can now pad only specific axes, making the operator more flexible

### Notes

- **Selective Axis Padding:** The `axes` input allows padding only specific dimensions
- **Flexible Pads Length:** `pads` length depends on the number of axes being padded
- **Backward Compatible:** If `axes` is not provided, behavior is the same as v13 (pads all axes)
- **Negative Axes:** Supports negative axis values counting from the back

---

## Pad - Version 19

**Since Version:** 19  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Given a tensor containing the data to be padded (data), a tensor containing the number of start and end pad values for axis (pads), (optionally) a mode, and (optionally) constant_value, a padded tensor (output) is generated.

The four supported modes are (similar to corresponding modes supported by numpy.pad):
- **constant(default)** - pads with a given constant value as specified by constant_value (which defaults to 0, empty string, or False)
- **reflect** - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
- **edge** - pads with the edge values of array
- **wrap** - wrap-around padding as if the data tensor forms a torus

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | STRING | ❌ | `'constant'` | Supported modes: constant(default), reflect, edge, wrap |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | Input tensor |
| `pads` | tensor(int64) | Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. pads should be a 1D tensor of shape [2 * num_axes] where num_axes refers to the number of elements in the axes input or the input rank if axes are not provided explicitly. pads format should be: [x1_begin, x2_begin, …, x1_end, x2_end,…], where xi_begin is the number of pad values added at the beginning of axis axes[i] and xi_end, the number of pad values added at the end of axis axes[i]. |
| `constant_value` | T (optional) | (Optional) A scalar value to be used if the mode chosen is constant (by default it is 0, empty string or False) |
| `axes` | Tind (optional) | 1-D tensor of axes that pads apply to. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an axis is repeated. If not provided, all axes are assumed ([0, 1, ..., input_rank-1]). |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | Tensor after padding |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(bool)`
- `tensor(complex128)`
- `tensor(complex64)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(float4e2m1)`
- `tensor(float8e4m3fn)`
- `tensor(float8e4m3fnuz)`
- `tensor(float8e5m2)`
- `tensor(float8e5m2fnuz)`
- `tensor(int16)`
- `tensor(int32)`
- `tensor(int64)`
- `tensor(int8)`
- `tensor(string)`
- `tensor(uint16)`
- `tensor(uint32)`
- `tensor(uint64)`
- `tensor(uint8)`

**Total:** 21 types

**Tind** in:
- `tensor(int32)`
- `tensor(int64)`

**Description:** Constrain input and output types to all tensor types up to IRv10. Constrain indices to integer types.

### Changes from v18

1. ➕ **Added `wrap` Mode:** New padding mode that wraps around (treats tensor as periodic)
2. ➕ **Extended Float Type Support:** Added support for float4e2m1, float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz
3. 📊 **Type Count:** Increased from 16 to 21 types
4. 📝 **Updated Mode Documentation:** Added `wrap` mode to the summary and attributes

### Notes

- **Wrap Mode:** Useful for periodic signals or circular padding
- **Extended Float Types:** Supports various 4-bit, 5-bit, and 8-bit float formats
- **Mode Count:** Now supports 4 padding modes (was 3)

---

## Pad - Version 21

**Since Version:** 21  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Same as v19, with extended type support.

### Attributes

Same as v19.

### Inputs

Same as v19.

### Outputs

Same as v19.

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(bool)`
- `tensor(complex128)`
- `tensor(complex64)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(float4e2m1)`
- `tensor(float8e4m3fn)`
- `tensor(float8e4m3fnuz)`
- `tensor(float8e5m2)`
- `tensor(float8e5m2fnuz)`
- `tensor(float8e8m0)`
- `tensor(int16)`
- `tensor(int32)`
- `tensor(int64)`
- `tensor(int8)`
- `tensor(string)`
- `tensor(uint16)`
- `tensor(uint32)`
- `tensor(uint64)`
- `tensor(uint8)`

**Total:** 22 types

**Tind** in:
- `tensor(int32)`
- `tensor(int64)`

**Description:** Constrain input and output types to all tensor types up to IRv10. Constrain indices to integer types.

### Changes from v19

1. ➕ **Added `float8e8m0` Type:** Extended float type support
2. 📊 **Type Count:** Increased from 21 to 22 types

### Notes

- **Extended Float Support:** Added float8e8m0 format

---

## Pad - Version 23

**Since Version:** 23  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Same as v21, with extended type support.

### Attributes

Same as v21.

### Inputs

Same as v21.

### Outputs

Same as v21.

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(bool)`
- `tensor(complex128)`
- `tensor(complex64)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(float4e2m1)`
- `tensor(float8e4m3fn)`
- `tensor(float8e4m3fnuz)`
- `tensor(float8e5m2)`
- `tensor(float8e5m2fnuz)`
- `tensor(float8e8m0)`
- `tensor(int16)`
- `tensor(int2)`
- `tensor(int32)`
- `tensor(int4)`
- `tensor(int64)`
- `tensor(int8)`
- `tensor(string)`
- `tensor(uint16)`
- `tensor(uint2)`
- `tensor(uint32)`
- `tensor(uint4)`
- `tensor(uint64)`
- `tensor(uint8)`

**Total:** 26 types

**Tind** in:
- `tensor(int32)`
- `tensor(int64)`

**Description:** Constrain input and output types to all tensor types up to IRv11. Constrain indices to integer types.

### Changes from v21

1. ➕ **Added Integer Sub-byte Types:** Added int2, int4, uint2, uint4 support
2. 📊 **Type Count:** Increased from 22 to 26 types

### Notes

- **Sub-byte Integers:** Supports 2-bit and 4-bit integer formats
- **Extended Integer Support:** More compact integer representations

---

## Pad - Version 24

**Since Version:** 24  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Same as v23, with extended type support.

### Type Constraints

**T** in: All types from v23, extended to IRv12 types.

**Total:** 26+ types (IRv12)

**Description:** Constrain input and output types to all tensor types up to IRv12.

### Changes from v23

1. 📊 **Extended to IRv12:** Type constraints extended to IRv12 specification

---

## Pad - Version 25

**Since Version:** 25  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Given a tensor containing the data to be padded (data), a tensor containing the number of start and end pad values for axis (pads), (optionally) a mode, and (optionally) constant_value, a padded tensor (output) is generated.

The four supported modes are (similar to corresponding modes supported by numpy.pad):
- **constant(default)** - pads with a given constant value as specified by constant_value (which defaults to 0, empty string, or False)
- **reflect** - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
- **edge** - pads with the edge values of array
- **wrap** - wrap-around padding as if the data tensor forms a torus

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | STRING | ❌ | `'constant'` | Supported modes: constant(default), reflect, edge, wrap |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | Input tensor |
| `pads` | tensor(int64) | Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. pads should be a 1D tensor of shape [2 * num_axes] where num_axes refers to the number of elements in the axes input or the input rank if axes are not provided explicitly. pads format should be: [x1_begin, x2_begin, …, x1_end, x2_end,…], where xi_begin is the number of pad values added at the beginning of axis axes[i] and xi_end, the number of pad values added at the end of axis axes[i]. |
| `constant_value` | T (optional) | (Optional) A scalar value to be used if the mode chosen is constant (by default it is 0, empty string or False) |
| `axes` | Tind (optional) | 1-D tensor of axes that pads apply to. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an axis is repeated. If not provided, all axes are assumed ([0, 1, ..., input_rank-1]). |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | Tensor after padding |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(bool)`
- `tensor(complex128)`
- `tensor(complex64)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(float4e2m1)`
- `tensor(float8e4m3fn)`
- `tensor(float8e4m3fnuz)`
- `tensor(float8e5m2)`
- `tensor(float8e5m2fnuz)`
- `tensor(float8e8m0)`
- `tensor(int16)`
- `tensor(int2)`
- `tensor(int32)`
- `tensor(int4)`
- `tensor(int64)`
- `tensor(int8)`
- `tensor(string)`
- `tensor(uint16)`
- `tensor(uint2)`
- `tensor(uint32)`
- `tensor(uint4)`
- `tensor(uint64)`
- `tensor(uint8)`

**Total:** 26 types (IRv13)

**Tind** in:
- `tensor(int32)`
- `tensor(int64)`

**Description:** Constrain input and output types to all tensor types up to IRv13. Constrain indices to integer types.

### Changes from v24

1. 📊 **Extended to IRv13:** Type constraints extended to IRv13 specification

### Notes

- **Latest Version:** This is the current latest version of the Pad operator
- **Comprehensive Type Support:** Supports all tensor types up to IRv13

---

## Version Comparison Summary

### Type Support Evolution

| Version | Float Types | Integer Types | Other Types | Total |
|---------|------------|--------------|-------------|-------|
| v1 | double, float, float16 | - | - | 3 |
| v2 | double, float, float16 | int8, int16, int32, int64, uint8, uint16, uint32, uint64 | - | 11 |
| v11 | bfloat16, double, float, float16 | int8, int16, int32, int64, uint8, uint16, uint32, uint64 | bool, complex64, complex128 | 15 |
| v13 | Same as v11 | Same as v11 | Same as v11 + string | 16 |
| v19 | + float4e2m1, float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz | Same as v13 | Same as v13 | 21 |
| v21 | + float8e8m0 | Same as v19 | Same as v19 | 22 |
| v23 | Same as v21 | + int2, int4, uint2, uint4 | Same as v21 | 26 |
| v25 | Same as v23 | Same as v23 | Same as v23 | 26 (IRv13) |

### Feature Evolution

| Feature | v1 | v2 | v11 | v13 | v18 | v19 | v21+ |
|---------|----|----|-----|-----|-----|-----|------|
| Shape Inference | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `pads` as Input | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `constant_value` Input | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `axes` Input | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `wrap` Mode | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| String Type | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Complex Types | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| bfloat16 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Attribute/Input Changes

| Version | `paddings`/`pads` | `value`/`constant_value` | `axes` | `mode` |
|---------|-------------------|--------------------------|--------|--------|
| v1 | Attribute (`paddings`) | Attribute (`value`) | N/A | constant, reflect, edge |
| v2 | Input (`pads`) | Input (`constant_value`) | N/A | constant, reflect, edge |
| v11 | Input (`pads`) | Input (`constant_value`) | N/A | constant, reflect, edge |
| v13 | Input (`pads`) | Input (`constant_value`) | N/A | constant, reflect, edge |
| v18 | Input (`pads`) | Input (`constant_value`) | Input (`axes`) | constant, reflect, edge |
| v19+ | Input (`pads`) | Input (`constant_value`) | Input (`axes`) | constant, reflect, edge, wrap |

---

## Behavioral Notes

### Padding Modes

#### 1. `constant` Mode (Default)
Pads with a constant value specified by `constant_value`.

**Example:**
```
Input shape: [3, 2]
Input: [[1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7]]

pads = [0, 2, 0, 0]  # [axis1_begin, axis1_end, axis0_begin, axis0_end]
mode = 'constant'
constant_value = 0.0

Output shape: [3, 4]
Output: [[0.0, 0.0, 1.0, 1.2],
         [0.0, 0.0, 2.3, 3.4],
         [0.0, 0.0, 4.5, 5.7]]
```

#### 2. `reflect` Mode
Pads by reflecting the tensor values at the boundaries (mirroring).

**Example:**
```
Input shape: [3, 2]
Input: [[1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7]]

pads = [0, 2, 0, 0]
mode = 'reflect'

Output shape: [3, 4]
Output: [[1.0, 1.2, 1.0, 1.2],
         [2.3, 3.4, 2.3, 3.4],
         [4.5, 5.7, 4.5, 5.7]]
```

#### 3. `edge` Mode
Pads by replicating the edge values.

**Example:**
```
Input shape: [3, 2]
Input: [[1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7]]

pads = [0, 2, 0, 0]
mode = 'edge'

Output shape: [3, 4]
Output: [[1.0, 1.0, 1.0, 1.2],
         [2.3, 2.3, 2.3, 3.4],
         [4.5, 4.5, 4.5, 5.7]]
```

#### 4. `wrap` Mode (opset 19+)
Pads by wrapping around (treating the tensor as periodic).

**Example:**
```
Input shape: [3, 2]
Input: [[1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7]]

pads = [2, 1, 1, 1]  # [axis1_begin, axis1_end, axis0_begin, axis0_end]
mode = 'wrap'

Output shape: [5, 5]
Output: [[3.4, 2.3, 3.4, 2.3],
         [5.7, 4.5, 5.7, 4.5],
         [1.2, 1.0, 1.2, 1.0],
         [3.4, 2.3, 3.4, 2.3],
         [5.7, 4.5, 5.7, 4.5]]
```

### Pads Format

The `pads` tensor has format: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`

Where:
- `xi_begin`: Number of pad values added at the **beginning** of axis `i`
- `xi_end`: Number of pad values added at the **end** of axis `i`
- All begin values come first, then all end values
- Length: `2 * num_axes` (where `num_axes` is from `axes` input or input rank)

**Example for 2D `[H, W]`:**
```
pads = [1, 2, 3, 4]  # [axis1_begin, axis1_end, axis0_begin, axis0_end]
                    # = [W_begin, W_end, H_begin, H_end]
                    # = [left, right, top, bottom]
```

**Example for 3D `[D, H, W]`:**
```
pads = [1, 1, 2, 2, 3, 3]  # [axis2_begin, axis2_end, axis1_begin, axis1_end, axis0_begin, axis0_end]
                          # = [W_begin, W_end, H_begin, H_end, D_begin, D_end]
```

### Output Shape Calculation

For a tensor of shape `[D0, D1, ..., Dn-1]` with `pads = [p0_begin, p1_begin, ..., p0_end, p1_end, ...]`:

**Output shape**: `[D0 + p0_begin + p0_end, D1 + p1_begin + p1_end, ..., Dn-1 + p(n-1)_begin + p(n-1)_end]`

**Example:**
```
Input shape: [5, 10]
pads = [2, 3, 1, 1]  # [axis1_begin, axis1_end, axis0_begin, axis0_end]
Output shape: [5+1+1, 10+2+3] = [7, 15]
```

### Selective Axis Padding (opset 18+)

When `axes` input is provided, only the specified axes are padded.

**Example:**
```
Input shape: [2, 3, 4, 5]
axes = [2, 3]  # Only pad axes 2 and 3
pads = [1, 1, 2, 2]  # [axis2_begin, axis2_end, axis3_begin, axis3_end]
Output shape: [2, 3, 4+1+1, 5+2+2] = [2, 3, 6, 9]
```

---

## Implementation Considerations

### Pads Format Conversion

When converting from ONNX Pad to PyTorch `F.pad`, the dimension order must be reversed:

- **ONNX**: Pads in axis order (0, 1, 2, ...) - first dimension first
- **PyTorch**: Pads in reverse axis order (last dimension first)

**Conversion Algorithm:**
```python
def onnx_pads_to_pytorch_pad(onnx_pads, input_rank):
    """
    Convert ONNX pads format to PyTorch pad format.
    
    ONNX: [axis0_begin, axis1_begin, ..., axis0_end, axis1_end, ...]
    PyTorch: (axis(n-1)_begin, axis(n-1)_end, ..., axis0_begin, axis0_end)
    """
    num_axes = len(onnx_pads) // 2
    begins = onnx_pads[:num_axes]
    ends = onnx_pads[num_axes:]
    
    # PyTorch pads in reverse order (last dimension first)
    pytorch_pad = []
    for i in range(num_axes - 1, -1, -1):
        pytorch_pad.extend([begins[i], ends[i]])
    
    return tuple(pytorch_pad)
```

### Mode Mapping

The padding modes have different names between ONNX and PyTorch, but the behavior is equivalent:

| ONNX Mode | PyTorch Mode | Behavior | Notes |
|-----------|--------------|----------|-------|
| `constant` | `'constant'` | Pads with constant value | Direct mapping, same name |
| `reflect` | `'reflect'` | Mirrors values at boundaries | Direct mapping, same name |
| `edge` | `'replicate'` | Replicates edge values | **Different names, same behavior** |
| `wrap` | `'circular'` | Wraps around (periodic) | **Different names, same behavior** |

**Key Points:**
- `constant` and `reflect` have the same names in both frameworks
- `edge` (ONNX) = `replicate` (PyTorch): Both replicate the edge values
- `wrap` (ONNX) = `circular` (PyTorch): Both treat the tensor as periodic/wrapping

### Mode Conversion Functions

**Converting ONNX Pad Mode to PyTorch:**
```python
def onnx_mode_to_pytorch(onnx_mode: str) -> str:
    """
    Convert ONNX Pad mode to PyTorch F.pad mode.
    
    Args:
        onnx_mode: ONNX padding mode ('constant', 'reflect', 'edge', 'wrap')
    
    Returns:
        PyTorch padding mode ('constant', 'reflect', 'replicate', 'circular')
    
    Raises:
        ValueError: If onnx_mode is not supported
    """
    mode_mapping = {
        'constant': 'constant',
        'reflect': 'reflect',
        'edge': 'replicate',      # ONNX edge = PyTorch replicate
        'wrap': 'circular',       # ONNX wrap = PyTorch circular
    }
    
    if onnx_mode not in mode_mapping:
        raise ValueError(f"Unsupported ONNX mode: {onnx_mode}. "
                        f"Supported modes: {list(mode_mapping.keys())}")
    
    return mode_mapping[onnx_mode]
```

**Converting PyTorch Pad Mode to ONNX:**
```python
def pytorch_mode_to_onnx(pytorch_mode: str) -> str:
    """
    Convert PyTorch F.pad mode to ONNX Pad mode.
    
    Args:
        pytorch_mode: PyTorch padding mode ('constant', 'reflect', 'replicate', 'circular')
    
    Returns:
        ONNX padding mode ('constant', 'reflect', 'edge', 'wrap')
    
    Raises:
        ValueError: If pytorch_mode is not supported
    """
    mode_mapping = {
        'constant': 'constant',
        'reflect': 'reflect',
        'replicate': 'edge',      # PyTorch replicate = ONNX edge
        'circular': 'wrap',       # PyTorch circular = ONNX wrap
    }
    
    if pytorch_mode not in mode_mapping:
        raise ValueError(f"Unsupported PyTorch mode: {pytorch_mode}. "
                        f"Supported modes: {list(mode_mapping.keys())}")
    
    return mode_mapping[pytorch_mode]
```

### Complete Conversion Example

**Converting ONNX Pad to PyTorch F.pad:**
```python
import torch
import torch.nn.functional as F

def convert_onnx_pad_to_pytorch(data, onnx_pads, onnx_mode='constant', 
                                 onnx_constant_value=0.0, axes=None):
    """
    Convert ONNX Pad operation to PyTorch F.pad.
    
    Args:
        data: Input tensor
        onnx_pads: ONNX pads format [axis0_begin, axis1_begin, ..., axis0_end, axis1_end, ...]
        onnx_mode: ONNX padding mode ('constant', 'reflect', 'edge', 'wrap')
        onnx_constant_value: Constant value for constant mode
        axes: Optional axes to pad (if None, pads all axes)
    
    Returns:
        Padded tensor using PyTorch F.pad
    """
    # Convert mode
    pytorch_mode = onnx_mode_to_pytorch(onnx_mode)
    
    # Determine number of axes
    if axes is not None:
        num_axes = len(axes)
    else:
        num_axes = len(data.shape)
    
    # Convert pads format
    begins = onnx_pads[:num_axes]
    ends = onnx_pads[num_axes:]
    
    # PyTorch pads in reverse order (last dimension first)
    pytorch_pad = []
    for i in range(num_axes - 1, -1, -1):
        pytorch_pad.extend([begins[i], ends[i]])
    
    # Apply padding
    if pytorch_mode == 'constant':
        return F.pad(data, tuple(pytorch_pad), mode=pytorch_mode, value=onnx_constant_value)
    else:
        return F.pad(data, tuple(pytorch_pad), mode=pytorch_mode)

# Example usage:
# ONNX Pad: pads=[0, 2, 0, 0], mode='edge', input shape [3, 2]
# PyTorch equivalent:
data = torch.tensor([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]])
onnx_pads = [0, 2, 0, 0]  # [axis1_begin, axis1_end, axis0_begin, axis0_end]
result = convert_onnx_pad_to_pytorch(data, onnx_pads, onnx_mode='edge')
# Equivalent to: F.pad(data, (0, 2, 0, 0), mode='replicate')
```

**Converting PyTorch F.pad to ONNX Pad:**
```python
def convert_pytorch_pad_to_onnx(data, pytorch_pad, pytorch_mode='constant',
                                 pytorch_value=0.0):
    """
    Convert PyTorch F.pad operation to ONNX Pad format.
    
    Args:
        data: Input tensor
        pytorch_pad: PyTorch pad format (last_dim_begin, last_dim_end, ..., first_dim_begin, first_dim_end)
        pytorch_mode: PyTorch padding mode ('constant', 'reflect', 'replicate', 'circular')
        pytorch_value: Constant value for constant mode
    
    Returns:
        Dictionary with ONNX Pad parameters: {'pads': [...], 'mode': ..., 'constant_value': ...}
    """
    # Convert mode
    onnx_mode = pytorch_mode_to_onnx(pytorch_mode)
    
    # Convert pads format
    num_axes = len(pytorch_pad) // 2
    begins = []
    ends = []
    
    # PyTorch pads in reverse order, ONNX pads in forward order
    for i in range(num_axes - 1, -1, -1):
        begins.append(pytorch_pad[2 * i])
        ends.append(pytorch_pad[2 * i + 1])
    
    # ONNX format: all begins, then all ends
    onnx_pads = begins + ends
    
    result = {
        'pads': onnx_pads,
        'mode': onnx_mode,
    }
    
    if onnx_mode == 'constant':
        result['constant_value'] = pytorch_value
    
    return result

# Example usage:
# PyTorch: F.pad(data, (1, 2, 1, 1), mode='replicate')
# ONNX equivalent:
pytorch_pad = (1, 2, 1, 1)  # (W_begin, W_end, H_begin, H_end) for [H, W]
onnx_params = convert_pytorch_pad_to_onnx(data, pytorch_pad, pytorch_mode='replicate')
# Result: {'pads': [1, 1, 1, 2], 'mode': 'edge'}
#         [H_begin, H_end, W_begin, W_end] = [1, 1, 1, 2]
```

### Behavioral Verification

To ensure the conversion is correct, you can verify that both implementations produce the same results:

```python
import torch
import torch.nn.functional as F
import numpy as np

def verify_pad_conversion():
    """Verify ONNX and PyTorch pad modes produce equivalent results."""
    
    # Test data
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # Test cases: (pytorch_mode, onnx_mode, pads)
    test_cases = [
        ('constant', 'constant', (1, 1, 1, 1)),
        ('reflect', 'reflect', (1, 1, 1, 1)),
        ('replicate', 'edge', (1, 1, 1, 1)),
        ('circular', 'wrap', (1, 1, 1, 1)),
    ]
    
    for pytorch_mode, onnx_mode, pads in test_cases:
        # PyTorch result
        pytorch_result = F.pad(data, pads, mode=pytorch_mode)
        
        # Convert to ONNX format and back
        onnx_params = convert_pytorch_pad_to_onnx(data, pads, pytorch_mode)
        converted_pads = tuple(onnx_params['pads'])
        converted_mode = onnx_mode_to_pytorch(onnx_params['mode'])
        
        # Convert back to PyTorch
        reconverted_result = F.pad(data, converted_pads, mode=converted_mode)
        
        # Verify they match
        assert torch.allclose(pytorch_result, reconverted_result), \
            f"Mismatch for {pytorch_mode}/{onnx_mode}"
        
        print(f"✓ {pytorch_mode} <-> {onnx_mode}: Conversion verified")
```

### Important Notes

1. **Mode Availability:**
   - `wrap`/`circular` mode is only available in ONNX opset 19+ and PyTorch
   - For older ONNX opsets, `wrap` mode is not supported

2. **Behavioral Equivalence:**
   - `edge` (ONNX) and `replicate` (PyTorch) are **functionally identical**
   - `wrap` (ONNX) and `circular` (PyTorch) are **functionally identical**
   - The only difference is the naming convention

3. **Implementation Check:**
   - Always verify the conversion with test cases
   - Some edge cases (e.g., very small tensors, large padding) may have numerical differences
   - The pads format conversion is critical for correctness

### Negative Padding

ONNX Pad supports negative padding values (since v2) to remove elements from the tensor, effectively cropping.

---

## Differences from Similar Operators

### Pad vs. Concat

| Aspect | Pad | Concat |
|--------|-----|--------|
| Purpose | Add padding to tensor | Join tensors along axis |
| Input | Single tensor | Multiple tensors |
| Output Shape | Larger than input | Combined size of inputs |
| Use Case | Prepare for convolution, maintain size | Combine data from multiple sources |

### Pad vs. Slice

| Aspect | Pad | Slice |
|--------|-----|-------|
| Purpose | Add elements | Remove elements |
| Output Size | Larger | Smaller |
| Negative Values | Can crop (negative padding) | Uses indices/ranges |
| Relationship | Pad with negative values ≈ Slice | Slice can achieve similar results |

---

## Common Use Cases

1. **Image Padding:** Add borders around images for convolutional operations
2. **Sequence Padding:** Pad sequences to fixed length for batch processing
3. **Maintain Spatial Dimensions:** Keep spatial dimensions constant through network layers
4. **Reflection Padding:** Preserve edge information in image processing
5. **Circular Padding:** Handle periodic signals or wrap-around patterns

---

## Testing Considerations

### Test Cases to Cover

1. **Basic Functionality:**
   - 1D, 2D, 3D, 4D tensors
   - All padding modes (constant, reflect, edge, wrap)
   - Various padding amounts
   - Verify output shapes

2. **Type Coverage:**
   - All supported types (float, int, bool, string, complex)
   - Type-specific `constant_value` defaults

3. **Mode Coverage:**
   - `constant` with different `constant_value`
   - `reflect` mode behavior
   - `edge` mode behavior
   - `wrap` mode behavior (opset 19+)

4. **Axis Coverage:**
   - All axes padded (default)
   - Selective axis padding with `axes` input (opset 18+)
   - Negative axis values
   - Edge cases (axis 0, last axis)

5. **Edge Cases:**
   - Zero padding
   - Negative padding (cropping)
   - Large padding amounts
   - Single-element tensors
   - Empty tensors

6. **Version-Specific:**
   - v1: `paddings` attribute format
   - v2+: `pads` input format
   - v18+: `axes` input functionality
   - v19+: `wrap` mode

7. **Shape Inference:**
   - Verify output shapes match expected calculations
   - Test with dynamic shapes (v2+)

---

## References

- [ONNX Pad Operator Documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad)
- Related Operators: Concat, Slice, Reshape
- NumPy pad function (similar modes)

---

**Document Version:** 1.0  
**Last Updated:** Based on ONNX opset versions 1, 2, 11, 13, 18, 19, 21, 23, 24, 25
