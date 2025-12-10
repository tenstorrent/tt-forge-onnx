# ONNX Squeeze Complete Opset Version Summary

Based on the [official ONNX Squeeze documentation](https://onnx.ai/onnx/operators/onnx__Squeeze.html), this document provides a comprehensive summary of all opset versions.

## Overview

Squeeze removes single-dimensional entries from the shape of a tensor. Takes an input `axes` with a list of axes to squeeze. If `axes` is not provided, all the single dimensions will be removed from the shape. If an axis is selected with shape entry not equal to one, an error is raised.

**Default Behavior**: If the `axes` parameter is omitted, all dimensions with size 1 are removed from the tensor shape.

**Example**: 
- Input shape `(1, 3, 1, 5)` with `axes=[0, 2]` → Output shape `(3, 5)` (removes dims 0 and 2)
- Input shape `(1, 3, 1, 5)` with no axes → Output shape `(3, 5)` (removes all size-1 dims)
- Input shape `(2, 3, 4)` with `axes=[0]` → Error (dimension 0 has size 2, not 1)

---

## Version-by-Version Breakdown

### **Squeeze v1** (since version 1)

**Key Characteristics:**
- **Axes**: Attribute (`axes` as INTS attribute)
  - List of **non-negative integers** indicating the dimensions to squeeze
  - If not provided, all single dimensions will be removed
  - **Important**: Only non-negative indices are supported (no negative indexing)
- **Inputs**: 
  - `data` (T): Tensors with at least max(dims) dimensions
- **Outputs**:
  - `squeezed` (T): Reshaped tensor with same data as input
- **Type Constraints**: 
  - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`

**Supported Types (v1):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

**Limitations:**
- Axes must be non-negative integers only (no negative indexing)

---

### **Squeeze v11** (since version 11)

**Key Characteristics:**
- **Axes**: Attribute (`axes` as INTS attribute)
  - List of integers indicating the dimensions to squeeze
  - **Negative value means counting dimensions from the back**
  - Accepted range is `[-r, r-1]` where `r = rank(data)`
  - If not provided, all single dimensions will be removed
- **Inputs**: 
  - `data` (T): Tensors with at least max(dims) dimensions
- **Outputs**:
  - `squeezed` (T): Reshaped tensor with same data as input
- **Type Constraints**: 
  - Same as v1: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v1:**
- **Negative indexing support**: Axes can now be negative integers (counting from the back)
- Example: For a 4D tensor, `axes=[-1]` refers to the last dimension, `axes=[-2]` refers to the second-to-last, etc.

**Supported Types (v11):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Squeeze v13** (since version 13)

**Key Characteristics:**
- **Axes**: **Now an optional input tensor** (`axes` as `tensor(int64)` input)
  - 1D tensor of integers indicating the dimensions to squeeze
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(data)`
  - If not provided, all single dimensions will be removed
- **Inputs**: 
  - Between 1 and 2 inputs:
    - `data` (T): Tensors with at least max(dims) dimensions
    - `axes` (optional, `tensor(int64)`): 1D tensor of integers indicating the dimensions to squeeze
- **Outputs**:
  - `squeezed` (T): Reshaped tensor with same data as input
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(bfloat16)`, `tensor(int4)`, `tensor(uint4)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v11:**
- **BREAKING CHANGE**: `axes` changed from attribute to optional input tensor
  - Previously: `axes` was an attribute (static at model creation time)
  - Now: `axes` is an optional input tensor (can be dynamic at runtime)
- **Type support expanded**: Added `bfloat16`, `int4`, `uint4`

**Supported Types (v13):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16` ⭐ (new)
- Integer (signed): `int4` ⭐ (new), `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4` ⭐ (new), `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

**Implementation Note:**
- When `axes` is provided as an input tensor, it must be a 1D tensor of type `int64`
- The converter must handle both cases: when `axes` is provided and when it's omitted

---

### **Squeeze v21** (since version 21)

**Key Characteristics:**
- **Axes**: Optional input tensor (`axes` as `tensor(int64)` input)
  - 1D tensor of integers indicating the dimensions to squeeze
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(data)`
  - If not provided, all single dimensions will be removed
- **Inputs**: 
  - Between 1 and 2 inputs:
    - `data` (T): Tensors with at least max(dims) dimensions
    - `axes` (optional, `tensor(int64)`): 1D tensor of integers indicating the dimensions to squeeze
- **Outputs**:
  - `squeezed` (T): Reshaped tensor with same data as input
- **Type Constraints**: 
  - **EXPANDED**: Adds support for multiple new float types
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)` ⭐, `tensor(float8e4m3fn)` ⭐, `tensor(float8e4m3fnuz)` ⭐, `tensor(float8e5m2)` ⭐, `tensor(float8e5m2fnuz)` ⭐, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v13:**
- **Type support expanded**: Added `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`
- No functional changes to `axes` handling

**Supported Types (v21):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1` ⭐ (new), `float8e4m3fn` ⭐ (new), `float8e4m3fnuz` ⭐ (new), `float8e5m2` ⭐ (new), `float8e5m2fnuz` ⭐ (new)
- Integer (signed): `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Squeeze v23** (since version 23)

**Key Characteristics:**
- **Axes**: Optional input tensor (`axes` as `tensor(int64)` input)
  - 1D tensor of integers indicating the dimensions to squeeze
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(data)`
  - If not provided, all single dimensions will be removed
- **Inputs**: 
  - Between 1 and 2 inputs:
    - `data` (T): Tensors with at least max(dims) dimensions
    - `axes` (optional, `tensor(int64)`): 1D tensor of integers indicating the dimensions to squeeze
- **Outputs**:
  - `squeezed` (T): Reshaped tensor with same data as input
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(float8e8m0)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)` ⭐, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v21:**
- **Type support expanded**: Added `float8e8m0`
- No functional changes to `axes` handling

**Supported Types (v23):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0` ⭐ (new)
- Integer (signed): `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Squeeze v24** (since version 24)

**Key Characteristics:**
- **Axes**: Optional input tensor (`axes` as `tensor(int64)` input)
  - 1D tensor of integers indicating the dimensions to squeeze
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(data)`
  - If not provided, all single dimensions will be removed
- **Inputs**: 
  - Between 1 and 2 inputs:
    - `data` (T): Tensors with at least max(dims) dimensions
    - `axes` (optional, `tensor(int64)`): 1D tensor of integers indicating the dimensions to squeeze
- **Outputs**:
  - `squeezed` (T): Reshaped tensor with same data as input
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(int2)`, `tensor(uint2)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)`, `tensor(int16)`, `tensor(int2)` ⭐, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint2)` ⭐, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v23:**
- **Type support expanded**: Added `int2`, `uint2`
- No functional changes to `axes` handling

**Supported Types (v24):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0`
- Integer (signed): `int2` ⭐ (new), `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint2` ⭐ (new), `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Squeeze v25** (since version 25)

**Key Characteristics:**
- **Axes**: Optional input tensor (`axes` as `tensor(int64)` input)
  - 1D tensor of integers indicating the dimensions to squeeze
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(data)`
  - If not provided, all single dimensions will be removed
- **Inputs**: 
  - Between 1 and 2 inputs:
    - `data` (T): Tensors with at least max(dims) dimensions
    - `axes` (optional, `tensor(int64)`): 1D tensor of integers indicating the dimensions to squeeze
- **Outputs**:
  - `squeezed` (T): Reshaped tensor with same data as input
- **Type Constraints**: 
  - Same as v24: All types from v24 are supported
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)`, `tensor(int16)`, `tensor(int2)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint2)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v24:**
- No functional changes - v25 is identical to v24

**Supported Types (v25):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0`
- Integer (signed): `int2`, `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint2`, `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

## Summary of Changes Across Versions

### Type Support Evolution

| Version | New Types Added | Total Types |
|---------|----------------|-------------|
| **v1** | Base types | 15 |
| **v11** | None (same as v1) | 15 |
| **v13** | `bfloat16`, `int4`, `uint4` | 18 |
| **v21** | `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz` | 23 |
| **v23** | `float8e8m0` | 24 |
| **v24** | `int2`, `uint2` | 26 |
| **v25** | None (same as v24) | 26 |

### Axes Parameter Evolution

| Version | Axes Format | Negative Indexing | Notes |
|---------|-------------|-------------------|-------|
| **v1** | Attribute (INTS) | ❌ No | Only non-negative integers |
| **v11** | Attribute (INTS) | ✅ Yes | Supports negative indexing |
| **v13-v25** | Optional Input (`tensor(int64)`) | ✅ Yes | **BREAKING CHANGE**: Axes is now a runtime input tensor |

### Key Behavioral Notes

1. **Default Behavior**: If `axes` is not provided:
   - **v1-v11**: All dimensions with size 1 are removed
   - **v13-v25**: All dimensions with size 1 are removed (same behavior)

2. **Axes Examples**:
   - Input shape `(1, 3, 1, 5)` with `axes=[0, 2]` → Output shape `(3, 5)` (removes dims 0 and 2)
   - Input shape `(1, 3, 1, 5)` with `axes=[-4, -2]` → Output shape `(3, 5)` (same, using negative indices)
   - Input shape `(1, 3, 1, 5)` with no axes → Output shape `(3, 5)` (removes all size-1 dims)
   - Input shape `(2, 3, 4)` with `axes=[0]` → **Error** (dimension 0 has size 2, not 1)

3. **Error Conditions**:
   - If an axis is selected with shape entry not equal to one, an error is raised
   - If an axis index is out of range `[-r, r-1]`, an error is raised

4. **Type Support**: The operator supports a wide range of types, expanding significantly from v1 to v25, with additions of:
   - Low-precision integers (`int2`, `int4`, `uint2`, `uint4`)
   - Low-precision floats (`bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0`)
   - Complex numbers (`complex64`, `complex128`)
   - Strings (`string`)

---

## Implementation Considerations

### For Converter Implementation

1. **Axes Handling**:
   - **v1**: `axes` is an attribute (INTS) - only non-negative integers
   - **v11**: `axes` is an attribute (INTS) - supports negative integers
   - **v13-v25**: `axes` is an optional input tensor (`tensor(int64)`) - must handle both cases:
     - When `axes` is provided: Extract from input tensor (must be constant or handle dynamically)
     - When `axes` is omitted: Remove all dimensions with size 1

2. **Negative Index Normalization**:
   - For v11+, normalize negative indices to positive: `idx = idx + rank if idx < 0`
   - Example: For rank 4, `-1` → `3`, `-2` → `2`, etc.

3. **Validation**:
   - Check that all specified axes have size 1 in the input shape
   - Check that axis indices are in valid range `[-r, r-1]`
   - Remove duplicate axes (if any)

4. **Type Support**:
   - Ensure the converter handles all supported types for the target opset version
   - Type support is additive - newer versions support all types from previous versions

5. **Edge Cases**:
   - **No size-1 dimensions**: If input has no size-1 dims and `axes` is provided, validate that all axes have size 1
   - **All dimensions size-1**: Input shape `(1, 1, 1)` → Output shape `()` (scalar)
   - **Empty axes list**: If `axes=[]`, output shape equals input shape (no-op)

6. **Optimization Opportunities**:
   - If input has no size-1 dimensions and `axes` is omitted, use `Identity` operator
   - If `axes` is provided but empty `[]`, use `Identity` operator
   - If all dimensions are size-1, can optimize to create a scalar tensor

---

## Comparison with NumPy

The ONNX Squeeze operator is similar to `numpy.squeeze()`:

```python
import numpy as np

# ONNX: axes=[0, 2] on shape (1, 3, 1, 5)
# NumPy equivalent:
arr = np.array(...)  # shape (1, 3, 1, 5)
result = np.squeeze(arr, axis=(0, 2))  # shape (3, 5)

# ONNX: no axes (remove all size-1 dims) on shape (1, 3, 1, 5)
# NumPy equivalent:
arr = np.array(...)  # shape (1, 3, 1, 5)
result = np.squeeze(arr)  # shape (3, 5)

# ONNX: axes=[-1] on shape (2, 3, 1)
# NumPy equivalent:
arr = np.array(...)  # shape (2, 3, 1)
result = np.squeeze(arr, axis=-1)  # shape (2, 3)
```

**Key Differences**:
- NumPy `squeeze()` always supports negative indexing
- ONNX v1 only supports non-negative indices
- ONNX v13+ uses input tensor for `axes` (can be dynamic), while NumPy uses function parameter (static)

---

## References

- [ONNX Squeeze Operator Documentation](https://onnx.ai/onnx/operators/onnx__Squeeze.html)
- [NumPy squeeze Documentation](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html)

