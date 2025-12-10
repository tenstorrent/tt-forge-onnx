# ONNX Unsqueeze Complete Opset Version Summary

Based on the [official ONNX Unsqueeze documentation](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html), this document provides a comprehensive summary of all opset versions.

## Overview

Unsqueeze inserts single-dimensional entries into the shape of a tensor. Takes an input `axes` with a list of axes to unsqueeze. The `axes` parameter specifies the dimensions in the **output tensor** where size-1 dimensions should be inserted. Unlike Squeeze, `axes` is **required** for Unsqueeze (cannot be omitted).

**Key Behavior**: The `axes` values refer to positions in the **output tensor**, not the input tensor. For example, if input shape is `(3, 4, 5)` and `axes=[0, 4]`, the output shape will be `(1, 3, 4, 5, 1)` - inserting size-1 dimensions at positions 0 and 4 in the output.

**Example**: 
- Input shape `(3, 4, 5)` with `axes=[0, 4]` → Output shape `(1, 3, 4, 5, 1)` (inserts dims at positions 0 and 4)
- Input shape `(3, 4)` with `axes=[1]` → Output shape `(3, 1, 4)` (inserts dim at position 1)
- Input shape `(3, 4)` with `axes=[-1]` → Output shape `(3, 4, 1)` (inserts dim at last position using negative index)

**Important**: 
- `axes` must be provided (unlike Squeeze where it's optional)
- `axes` values must be unique (no duplicates allowed)
- `axes` values refer to output tensor dimensions, not input dimensions
- The order of values in `axes` does not matter

---

## Version-by-Version Breakdown

### **Unsqueeze v1** (since version 1)

**Key Characteristics:**
- **Axes**: Attribute (`axes` as INTS attribute) - **REQUIRED**
  - List of **non-negative integers** indicating the dimensions in the output tensor where size-1 dimensions should be inserted
  - **Important**: Only non-negative indices are supported (no negative indexing)
  - Must be unique (no duplicates)
  - Values refer to positions in the **output tensor**
- **Inputs**: 
  - `data` (T): Input tensor
- **Outputs**:
  - `expanded` (T): Reshaped tensor with same data as input, with additional size-1 dimensions inserted
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
- Axes is required (cannot be omitted)

---

### **Unsqueeze v11** (since version 11)

**Key Characteristics:**
- **Axes**: Attribute (`axes` as INTS attribute) - **REQUIRED**
  - List of integers indicating the dimensions in the output tensor where size-1 dimensions should be inserted
  - **Negative value means counting dimensions from the back** (in the output tensor)
  - Accepted range is `[-r, r-1]` where `r = rank(output) = rank(input) + len(axes)`
  - Must be unique (no duplicates)
  - Values refer to positions in the **output tensor**
- **Inputs**: 
  - `data` (T): Input tensor
- **Outputs**:
  - `expanded` (T): Reshaped tensor with same data as input, with additional size-1 dimensions inserted
- **Type Constraints**: 
  - Same as v1: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v1:**
- **Negative indexing support**: Axes can now be negative integers (counting from the back in the output tensor)
- Example: For input shape `(3, 4)` with `axes=[-1]`, output shape is `(3, 4, 1)` (inserts at last position)

**Supported Types (v11):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Unsqueeze v13** (since version 13)

**Key Characteristics:**
- **Axes**: **Now an input tensor** (`axes` as `tensor(int64)` input) - **REQUIRED**
  - List of integers indicating the dimensions in the output tensor where size-1 dimensions should be inserted
  - Negative value means counting dimensions from the back (in the output tensor)
  - Accepted range is `[-r, r-1]` where `r = rank(output) = rank(input) + len(axes)`
  - Must be unique (no duplicates)
  - Values refer to positions in the **output tensor**
- **Inputs**: 
  - 2 inputs:
    - `data` (T): Input tensor
    - `axes` (`tensor(int64)`): List of integers indicating the dimensions to unsqueeze
- **Outputs**:
  - `expanded` (T): Reshaped tensor with same data as input, with additional size-1 dimensions inserted
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(bfloat16)`, `tensor(int4)`, `tensor(uint4)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
  - Constrain input and output types to all tensor types

**Changes from v11:**
- **BREAKING CHANGE**: `axes` changed from attribute to required input tensor
  - Previously: `axes` was an attribute (static at model creation time)
  - Now: `axes` is a required input tensor (can be dynamic at runtime, but typically provided as constant initializer)
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
- The converter must extract `axes` from the constant initializer or handle it dynamically
- Unlike Squeeze, `axes` cannot be omitted for Unsqueeze

---

### **Unsqueeze v21** (since version 21)

**Key Characteristics:**
- **Axes**: Required input tensor (`axes` as `tensor(int64)` input)
  - List of integers indicating the dimensions in the output tensor where size-1 dimensions should be inserted
  - Negative value means counting dimensions from the back (in the output tensor)
  - Accepted range is `[-r, r-1]` where `r = rank(output) = rank(input) + len(axes)`
  - Must be unique (no duplicates)
  - Values refer to positions in the **output tensor**
- **Inputs**: 
  - 2 inputs:
    - `data` (T): Input tensor
    - `axes` (`tensor(int64)`): List of integers indicating the dimensions to unsqueeze
- **Outputs**:
  - `expanded` (T): Reshaped tensor with same data as input, with additional size-1 dimensions inserted
- **Type Constraints**: 
  - **EXPANDED**: Adds support for new float8 types
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)` ⭐, `tensor(float8e4m3fnuz)` ⭐, `tensor(float8e5m2)` ⭐, `tensor(float8e5m2fnuz)` ⭐, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
  - Constrain input and output types to all tensor types up to IRv10

**Changes from v13:**
- **Type support expanded**: Added `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`
- No functional changes to `axes` handling

**Supported Types (v21):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float8e4m3fn` ⭐ (new), `float8e4m3fnuz` ⭐ (new), `float8e5m2` ⭐ (new), `float8e5m2fnuz` ⭐ (new)
- Integer (signed): `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Unsqueeze v23** (since version 23)

**Key Characteristics:**
- **Axes**: Required input tensor (`axes` as `tensor(int64)` input)
  - 1D tensor of integers indicating the dimensions in the output tensor where size-1 dimensions should be inserted
  - Negative value means counting dimensions from the back (in the output tensor)
  - Accepted range is `[-r, r-1]` where `r = rank(output) = rank(input) + len(axes)`
  - Must be unique (no duplicates)
  - Values refer to positions in the **output tensor**
- **Inputs**: 
  - 2 inputs:
    - `data` (T): Input tensor
    - `axes` (`tensor(int64)`): 1D tensor of integers indicating the dimensions to unsqueeze
- **Outputs**:
  - `expanded` (T): Reshaped tensor with same data as input, with additional size-1 dimensions inserted
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(float4e2m1)` and `tensor(float8e8m0)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)` ⭐, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)` ⭐, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
  - Constrain input and output types to all tensor types up to IRv11

**Changes from v21:**
- **Type support expanded**: Added `float4e2m1` and `float8e8m0`
- No functional changes to `axes` handling

**Supported Types (v23):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1` ⭐ (new), `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0` ⭐ (new)
- Integer (signed): `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Unsqueeze v24** (since version 24)

**Key Characteristics:**
- **Axes**: Required input tensor (`axes` as `tensor(int64)` input)
  - 1D tensor of integers indicating the dimensions in the output tensor where size-1 dimensions should be inserted
  - Negative value means counting dimensions from the back (in the output tensor)
  - Accepted range is `[-r, r-1]` where `r = rank(output) = rank(input) + len(axes)`
  - Must be unique (no duplicates)
  - Values refer to positions in the **output tensor**
- **Inputs**: 
  - 2 inputs:
    - `data` (T): Input tensor
    - `axes` (`tensor(int64)`): 1D tensor of integers indicating the dimensions to unsqueeze
- **Outputs**:
  - `expanded` (T): Reshaped tensor with same data as input, with additional size-1 dimensions inserted
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(int2)`, `tensor(uint2)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)`, `tensor(int16)`, `tensor(int2)` ⭐, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint2)` ⭐, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
  - Constrain input and output types to all tensor types up to IRv12

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

### **Unsqueeze v25** (since version 25)

**Key Characteristics:**
- **Axes**: Required input tensor (`axes` as `tensor(int64)` input)
  - 1D tensor of integers indicating the dimensions in the output tensor where size-1 dimensions should be inserted
  - Negative value means counting dimensions from the back (in the output tensor)
  - Accepted range is `[-r, r-1]` where `r = rank(output) = rank(input) + len(axes)`
  - Must be unique (no duplicates)
  - Values refer to positions in the **output tensor**
- **Inputs**: 
  - 2 inputs:
    - `data` (T): Input tensor
    - `axes` (`tensor(int64)`): 1D tensor of integers indicating the dimensions to unsqueeze
- **Outputs**:
  - `expanded` (T): Reshaped tensor with same data as input, with additional size-1 dimensions inserted
- **Type Constraints**: 
  - Same as v24: All types from v24 are supported
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)`, `tensor(int16)`, `tensor(int2)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint2)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
  - Constrain input and output types to all tensor types up to IRv13

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
| **v21** | `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz` | 22 |
| **v23** | `float4e2m1`, `float8e8m0` | 24 |
| **v24** | `int2`, `uint2` | 26 |
| **v25** | None (same as v24) | 26 |

### Axes Parameter Evolution

| Version | Axes Format | Negative Indexing | Notes |
|---------|-------------|-------------------|-------|
| **v1** | Attribute (INTS) | ❌ No | Only non-negative integers, **REQUIRED** |
| **v11** | Attribute (INTS) | ✅ Yes | Supports negative indexing, **REQUIRED** |
| **v13-v25** | Required Input (`tensor(int64)`) | ✅ Yes | **BREAKING CHANGE**: Axes is now a runtime input tensor, **REQUIRED** |

### Key Behavioral Notes

1. **Axes is Required**: Unlike Squeeze, `axes` must always be provided for Unsqueeze (cannot be omitted)

2. **Axes Refer to Output Tensor**: The `axes` values specify positions in the **output tensor**, not the input tensor. This is a critical difference from Squeeze.
   - Input shape `(3, 4, 5)` with `axes=[0, 4]` → Output shape `(1, 3, 4, 5, 1)`
   - Input shape `(3, 4)` with `axes=[1]` → Output shape `(3, 1, 4)`
   - Input shape `(3, 4)` with `axes=[-1]` → Output shape `(3, 4, 1)` (negative index refers to output tensor)

3. **Axes Examples**:
   - Input shape `(3, 4, 5)` with `axes=[0, 4]` → Output shape `(1, 3, 4, 5, 1)` (inserts at positions 0 and 4)
   - Input shape `(3, 4)` with `axes=[1]` → Output shape `(3, 1, 4)` (inserts at position 1)
   - Input shape `(3, 4)` with `axes=[-1]` → Output shape `(3, 4, 1)` (inserts at last position using negative index)
   - Input shape `(3, 4)` with `axes=[0, 2]` → Output shape `(1, 3, 1, 4)` (inserts at positions 0 and 2)

4. **Error Conditions**:
   - If `axes` contains duplicate values, an error is raised
   - If an axis index is out of range `[-r, r-1]` where `r = rank(output)`, an error is raised
   - If `axes` is not provided, an error is raised (unlike Squeeze)

5. **Order Independence**: The order of values in `axes` does not matter. `axes=[0, 4]` and `axes=[4, 0]` produce the same result.

6. **Type Support**: The operator supports a wide range of types, expanding significantly from v1 to v25, with additions of:
   - Low-precision integers (`int2`, `int4`, `uint2`, `uint4`)
   - Low-precision floats (`bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0`)
   - Complex numbers (`complex64`, `complex128`)
   - Strings (`string`)

---

## Implementation Considerations

### For Converter Implementation

1. **Axes Handling**:
   - **v1**: `axes` is a required attribute (INTS) - only non-negative integers
   - **v11**: `axes` is a required attribute (INTS) - supports negative integers
   - **v13-v25**: `axes` is a required input tensor (`tensor(int64)`) - must extract from input:
     - Extract from constant initializer (most common case)
     - Handle dynamically if needed (though current implementation may require constant)

2. **Negative Index Normalization**:
   - For v11+, normalize negative indices to positive based on **output tensor rank**
   - Output rank = input rank + len(axes)
   - Example: For input shape `(3, 4)` with `axes=[-1]`, output rank is 3, so `-1` → `2`

3. **Axes Validation**:
   - Check that `axes` is provided (required, unlike Squeeze)
   - Check that all axes values are unique (no duplicates)
   - Check that axis indices are in valid range `[-r, r-1]` where `r = rank(output)`
   - Sort axes in ascending order for consistent processing

4. **Type Support**:
   - Ensure the converter handles all supported types for the target opset version
   - Type support is additive - newer versions support all types from previous versions

5. **Edge Cases**:
   - **Empty axes list**: If `axes=[]`, output shape equals input shape (no-op, but unusual)
   - **Multiple axes**: Must insert dimensions in sorted order to avoid index shifting issues
   - **Negative indices**: Must normalize based on output tensor rank, not input rank

6. **Output Shape Calculation**:
   - Start with input shape
   - For each axis value in sorted order, insert a dimension of size 1 at that position
   - Example: Input `(3, 4)`, `axes=[1, 3]` → Insert at pos 1 → `(3, 1, 4)` → Insert at pos 3 → `(3, 1, 4, 1)`

---

## Comparison with NumPy

The ONNX Unsqueeze operator is similar to `numpy.expand_dims()`:

```python
import numpy as np

# ONNX: axes=[0, 4] on shape (3, 4, 5)
# NumPy equivalent:
arr = np.array(...)  # shape (3, 4, 5)
result = np.expand_dims(arr, axis=0)  # shape (1, 3, 4, 5)
result = np.expand_dims(result, axis=4)  # shape (1, 3, 4, 5, 1)

# ONNX: axes=[1] on shape (3, 4)
# NumPy equivalent:
arr = np.array(...)  # shape (3, 4)
result = np.expand_dims(arr, axis=1)  # shape (3, 1, 4)

# ONNX: axes=[-1] on shape (3, 4)
# NumPy equivalent:
arr = np.array(...)  # shape (3, 4)
result = np.expand_dims(arr, axis=-1)  # shape (3, 4, 1)
```

**Key Differences**:
- NumPy `expand_dims()` takes a single axis at a time, while ONNX Unsqueeze can take multiple axes
- NumPy `expand_dims()` axis refers to the input tensor, while ONNX Unsqueeze axes refer to the output tensor
- ONNX v1 only supports non-negative indices
- ONNX v13+ uses input tensor for `axes` (can be dynamic), while NumPy uses function parameter (static)

**PyTorch Comparison**:
```python
import torch

# ONNX: axes=[1] on shape (3, 4)
# PyTorch equivalent:
x = torch.tensor(...)  # shape (3, 4)
result = torch.unsqueeze(x, dim=1)  # shape (3, 1, 4)

# ONNX: axes=[0, 2] on shape (3, 4)
# PyTorch equivalent (multiple calls):
x = torch.tensor(...)  # shape (3, 4)
result = torch.unsqueeze(x, dim=0)  # shape (1, 3, 4)
result = torch.unsqueeze(result, dim=2)  # shape (1, 3, 1, 4)
```

**Note**: PyTorch `unsqueeze()` takes a single dimension at a time, while ONNX Unsqueeze can handle multiple dimensions in one operation.

---

## Differences from Squeeze

| Aspect | Squeeze | Unsqueeze |
|--------|---------|-----------|
| **Operation** | Removes size-1 dimensions | Inserts size-1 dimensions |
| **Axes Required** | ❌ Optional (can omit to remove all size-1 dims) | ✅ Required (must specify) |
| **Axes Reference** | Input tensor dimensions | **Output tensor dimensions** |
| **Default Behavior** | Remove all size-1 dims if axes omitted | No default (axes required) |
| **Output Rank** | `rank(input) - len(axes)` | `rank(input) + len(axes)` |
| **Axes Range** | `[-r, r-1]` where `r = rank(input)` | `[-r, r-1]` where `r = rank(output)` |

---

## References

- [ONNX Unsqueeze Operator Documentation](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html)
- [NumPy expand_dims Documentation](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)
- [PyTorch unsqueeze Documentation](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)

