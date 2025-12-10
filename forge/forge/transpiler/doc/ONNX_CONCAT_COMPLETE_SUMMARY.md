# ONNX Concat Complete Opset Version Summary

Based on the [official ONNX Concat documentation](https://onnx.ai/onnx/operators/onnx__Concat.html), this document provides a comprehensive summary of all opset versions.

## Overview

Concat concatenates a list of tensors into a single tensor along a specified axis. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.

**Key Behavior**: The `axis` parameter specifies which dimension to concatenate along. All input tensors must have identical shapes in all dimensions except the concatenation axis. The output tensor's size along the concatenation axis is the sum of all input tensors' sizes along that axis.

**Example**: 
- Input shapes `(3, 4)`, `(3, 5)`, `(3, 2)` with `axis=1` → Output shape `(3, 11)` (concatenates along dimension 1: 4+5+2=11)
- Input shapes `(2, 3)`, `(4, 3)` with `axis=0` → Output shape `(6, 3)` (concatenates along dimension 0: 2+4=6)
- Input shapes `(2, 3, 4)`, `(2, 3, 5)` with `axis=-1` → Output shape `(2, 3, 9)` (concatenates along last dimension using negative index)

**Important**: 
- All input tensors must have the same rank
- All input tensors must have the same shape in all dimensions except the concatenation axis
- The number of inputs can range from 1 to 2,147,483,647
- The output shape along the concatenation axis is the sum of all input shapes along that axis

---

## Version-by-Version Breakdown

### **Concat v1** (since version 1)

**Key Characteristics:**
- **Axis**: Attribute (`axis` as INT attribute) - **OPTIONAL**
  - Which axis to concat on
  - **Default value is 1** (if not provided)
  - **Important**: Only non-negative indices are supported (no negative indexing)
  - Accepted range is `[0, r-1]` where `r = rank(inputs)`
- **Inputs**: 
  - Between 1 and 2,147,483,647 inputs (variadic)
  - `inputs` (variadic, T): List of tensors for concatenation
- **Outputs**:
  - `concat_result` (T): Concatenated tensor
- **Type Constraints**: 
  - **LIMITED**: Only float types supported
  - `tensor(double)`, `tensor(float)`, `tensor(float16)`
- **Shape Inference**: ❌ No (shape inference: False)

**Supported Types (v1):**
- Floating point: `float`, `double`, `float16`

**Limitations:**
- Axis must be non-negative integer only (no negative indexing)
- Only supports float tensor types
- No shape inference
- Axis defaults to 1 if not provided

---

### **Concat v4** (since version 4)

**Key Characteristics:**
- **Axis**: Attribute (`axis` as INT attribute) - **REQUIRED**
  - Which axis to concat on
  - **Important**: Only non-negative indices are supported (no negative indexing)
  - Accepted range is `[0, r-1]` where `r = rank(inputs)`
- **Inputs**: 
  - Between 1 and 2,147,483,647 inputs (variadic)
  - `inputs` (variadic, T): List of tensors for concatenation
- **Outputs**:
  - `concat_result` (T): Concatenated tensor
- **Type Constraints**: 
  - **EXPANDED**: Adds support for many new types
  - Full list: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)

**Changes from v1:**
- **BREAKING CHANGE**: `axis` is now **required** (no default value)
- **Type support expanded**: Added `bool`, `complex64`, `complex128`, `int8`, `int16`, `int32`, `int64`, `string`, `uint8`, `uint16`, `uint32`, `uint64`
- **Shape inference enabled**: Now supports shape inference

**Supported Types (v4):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Concat v11** (since version 11)

**Key Characteristics:**
- **Axis**: Attribute (`axis` as INT attribute) - **REQUIRED**
  - Which axis to concat on
  - **Negative value means counting dimensions from the back**
  - Accepted range is `[-r, r-1]` where `r = rank(inputs)`
- **Inputs**: 
  - Between 1 and 2,147,483,647 inputs (variadic)
  - `inputs` (variadic, T): List of tensors for concatenation
- **Outputs**:
  - `concat_result` (T): Concatenated tensor
- **Type Constraints**: 
  - Same as v4: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)

**Changes from v4:**
- **Negative indexing support**: Axis can now be negative integer (counting from the back)
- Example: For a 3D tensor, `axis=-1` refers to the last dimension, `axis=-2` refers to the second-to-last, etc.
- No type changes

**Supported Types (v11):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Concat v13** (since version 13)

**Key Characteristics:**
- **Axis**: Attribute (`axis` as INT attribute) - **REQUIRED**
  - Which axis to concat on
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(inputs)`
- **Inputs**: 
  - Between 1 and 2,147,483,647 inputs (variadic)
  - `inputs` (variadic, T): List of tensors for concatenation
- **Outputs**:
  - `concat_result` (T): Concatenated tensor
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(bfloat16)`, `tensor(int4)`, `tensor(uint4)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)

**Changes from v11:**
- **Type support expanded**: Added `bfloat16`, `int4`, `uint4`
- No functional changes to `axis` handling

**Supported Types (v13):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16` ⭐ (new)
- Integer (signed): `int4` ⭐ (new), `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4` ⭐ (new), `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Concat v21** (since version 21)

**Key Characteristics:**
- **Axis**: Attribute (`axis` as INT attribute) - **REQUIRED**
  - Which axis to concat on
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(inputs)`
- **Inputs**: 
  - Between 1 and 2,147,483,647 inputs (variadic)
  - `inputs` (variadic, T): List of tensors for concatenation
- **Outputs**:
  - `concat_result` (T): Concatenated tensor
- **Type Constraints**: 
  - **EXPANDED**: Adds support for new float8 types
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)` ⭐, `tensor(float8e4m3fnuz)` ⭐, `tensor(float8e5m2)` ⭐, `tensor(float8e5m2fnuz)` ⭐, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)

**Changes from v13:**
- **Type support expanded**: Added `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`
- No functional changes to `axis` handling

**Supported Types (v21):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float8e4m3fn` ⭐ (new), `float8e4m3fnuz` ⭐ (new), `float8e5m2` ⭐ (new), `float8e5m2fnuz` ⭐ (new)
- Integer (signed): `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Concat v23** (since version 23)

**Key Characteristics:**
- **Axis**: Attribute (`axis` as INT attribute) - **REQUIRED**
  - Which axis to concat on
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(inputs)`
- **Inputs**: 
  - Between 1 and 2,147,483,647 inputs (variadic)
  - `inputs` (variadic, T): List of tensors for concatenation
- **Outputs**:
  - `concat_result` (T): Concatenated tensor
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(float4e2m1)` and `tensor(float8e8m0)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)` ⭐, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)` ⭐, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)

**Changes from v21:**
- **Type support expanded**: Added `float4e2m1` and `float8e8m0`
- No functional changes to `axis` handling

**Supported Types (v23):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1` ⭐ (new), `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0` ⭐ (new)
- Integer (signed): `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Concat v24** (since version 24)

**Key Characteristics:**
- **Axis**: Attribute (`axis` as INT attribute) - **REQUIRED**
  - Which axis to concat on
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(inputs)`
- **Inputs**: 
  - Between 1 and 2,147,483,647 inputs (variadic)
  - `inputs` (variadic, T): List of tensors for concatenation
- **Outputs**:
  - `concat_result` (T): Concatenated tensor
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(int2)`, `tensor(uint2)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)`, `tensor(int16)`, `tensor(int2)` ⭐, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint2)` ⭐, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)

**Changes from v23:**
- **Type support expanded**: Added `int2`, `uint2`
- No functional changes to `axis` handling

**Supported Types (v24):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0`
- Integer (signed): `int2` ⭐ (new), `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint2` ⭐ (new), `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Concat v25** (since version 25)

**Key Characteristics:**
- **Axis**: Attribute (`axis` as INT attribute) - **REQUIRED**
  - Which axis to concat on
  - Negative value means counting dimensions from the back
  - Accepted range is `[-r, r-1]` where `r = rank(inputs)`
- **Inputs**: 
  - Between 1 and 2,147,483,647 inputs (variadic)
  - `inputs` (variadic, T): List of tensors for concatenation
- **Outputs**:
  - `concat_result` (T): Concatenated tensor
- **Type Constraints**: 
  - Same as v24: All types from v24 are supported
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)`, `tensor(int16)`, `tensor(int2)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint2)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)

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
| **v1** | Base float types | 3 |
| **v4** | `bool`, `complex64`, `complex128`, `int8`, `int16`, `int32`, `int64`, `string`, `uint8`, `uint16`, `uint32`, `uint64` | 15 |
| **v11** | None (same as v4) | 15 |
| **v13** | `bfloat16`, `int4`, `uint4` | 18 |
| **v21** | `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz` | 22 |
| **v23** | `float4e2m1`, `float8e8m0` | 24 |
| **v24** | `int2`, `uint2` | 26 |
| **v25** | None (same as v24) | 26 |

### Axis Parameter Evolution

| Version | Axis Format | Default Value | Negative Indexing | Shape Inference |
|---------|-------------|---------------|-------------------|-----------------|
| **v1** | Attribute (INT) | ✅ Yes (default=1) | ❌ No | ❌ No |
| **v4** | Attribute (INT) | ❌ Required | ❌ No | ✅ Yes |
| **v11-v25** | Attribute (INT) | ❌ Required | ✅ Yes | ✅ Yes |

### Key Behavioral Notes

1. **Axis Default Value**: 
   - **v1**: `axis` defaults to `1` if not provided
   - **v4+**: `axis` is **required** (no default value)

2. **Input Requirements**:
   - All input tensors must have the same rank
   - All input tensors must have the same shape in all dimensions except the concatenation axis
   - The number of inputs can range from 1 to 2,147,483,647 (variadic)

3. **Output Shape Calculation**:
   - Output shape is identical to input shapes in all dimensions except the concatenation axis
   - Output size along concatenation axis = sum of all input sizes along that axis
   - Example: Inputs `(3, 4)`, `(3, 5)`, `(3, 2)` with `axis=1` → Output `(3, 11)` (4+5+2=11)

4. **Axis Examples**:
   - Input shapes `(3, 4)`, `(3, 5)` with `axis=1` → Output shape `(3, 9)` (concatenates along dim 1)
   - Input shapes `(2, 3)`, `(4, 3)` with `axis=0` → Output shape `(6, 3)` (concatenates along dim 0)
   - Input shapes `(2, 3, 4)`, `(2, 3, 5)` with `axis=-1` → Output shape `(2, 3, 9)` (concatenates along last dim)
   - Input shapes `(2, 3)`, `(2, 3)` with `axis=0` → Output shape `(4, 3)` (concatenates along dim 0)

5. **Error Conditions**:
   - If input tensors have different ranks, an error is raised
   - If input tensors have different shapes in non-concatenation dimensions, an error is raised
   - If axis index is out of range `[-r, r-1]` where `r = rank(inputs)`, an error is raised
   - If only one input is provided, the output is the same as the input (no-op)

6. **Type Support**: The operator supports a wide range of types, expanding significantly from v1 to v25, with additions of:
   - Low-precision integers (`int2`, `int4`, `uint2`, `uint4`)
   - Low-precision floats (`bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0`)
   - Complex numbers (`complex64`, `complex128`)
   - Strings (`string`)
   - Boolean (`bool`)

---

## Implementation Considerations

### For Converter Implementation

1. **Axis Handling**:
   - **v1**: `axis` is optional attribute (defaults to 1 if not provided) - only non-negative integers
   - **v4**: `axis` is required attribute - only non-negative integers
   - **v11+**: `axis` is required attribute - supports negative integers
   - Normalize negative indices: `axis = axis + rank if axis < 0`

2. **Input Validation**:
   - Verify all input tensors have the same rank
   - Verify all input tensors have the same shape in all dimensions except the concatenation axis
   - Validate axis is in range `[-r, r-1]` where `r = rank(inputs)`

3. **Output Shape Calculation**:
   - Start with shape of first input tensor
   - Sum sizes along concatenation axis: `output_shape[axis] = sum(input_shapes[i][axis] for all i)`
   - Example: Inputs `(3, 4)`, `(3, 5)`, `(3, 2)` with `axis=1` → Output `(3, 11)`

4. **Type Support**:
   - Ensure the converter handles all supported types for the target opset version
   - Type support is additive - newer versions support all types from previous versions
   - All input tensors must have the same type

5. **Edge Cases**:
   - **Single input**: If only one input, output equals input (can use Identity)
   - **Empty inputs**: Should not occur (minimum 1 input required)
   - **Same shape on all dims**: Inputs `(3, 4)`, `(3, 4)` with `axis=0` → Output `(6, 4)`

6. **Variadic Inputs**:
   - Concat supports 1 to 2,147,483,647 inputs
   - Converter must handle any number of inputs dynamically
   - All inputs are passed to the ConcatNode

---

## Comparison with NumPy

The ONNX Concat operator is similar to `numpy.concatenate()`:

```python
import numpy as np

# ONNX: axis=1 on shapes (3, 4), (3, 5), (3, 2)
# NumPy equivalent:
arr1 = np.array(...)  # shape (3, 4)
arr2 = np.array(...)  # shape (3, 5)
arr3 = np.array(...)  # shape (3, 2)
result = np.concatenate([arr1, arr2, arr3], axis=1)  # shape (3, 11)

# ONNX: axis=0 on shapes (2, 3), (4, 3)
# NumPy equivalent:
arr1 = np.array(...)  # shape (2, 3)
arr2 = np.array(...)  # shape (4, 3)
result = np.concatenate([arr1, arr2], axis=0)  # shape (6, 3)

# ONNX: axis=-1 on shapes (2, 3, 4), (2, 3, 5)
# NumPy equivalent:
arr1 = np.array(...)  # shape (2, 3, 4)
arr2 = np.array(...)  # shape (2, 3, 5)
result = np.concatenate([arr1, arr2], axis=-1)  # shape (2, 3, 9)
```

**Key Differences**:
- NumPy `concatenate()` always supports negative indexing
- ONNX v1 only supports non-negative indices
- ONNX v1 has default axis=1, while NumPy requires explicit axis parameter
- ONNX v4+ requires axis to be specified (no default)

**PyTorch Comparison**:
```python
import torch

# ONNX: axis=1 on shapes (3, 4), (3, 5), (3, 2)
# PyTorch equivalent:
x1 = torch.tensor(...)  # shape (3, 4)
x2 = torch.tensor(...)  # shape (3, 5)
x3 = torch.tensor(...)  # shape (3, 2)
result = torch.cat([x1, x2, x3], dim=1)  # shape (3, 11)

# ONNX: axis=-1 on shapes (2, 3, 4), (2, 3, 5)
# PyTorch equivalent:
x1 = torch.tensor(...)  # shape (2, 3, 4)
x2 = torch.tensor(...)  # shape (2, 3, 5)
result = torch.cat([x1, x2], dim=-1)  # shape (2, 3, 9)
```

**Note**: PyTorch `cat()` uses `dim` parameter (same as ONNX `axis`), and always supports negative indexing.

---

## Differences from Other Operators

| Aspect | Concat | Unsqueeze | Squeeze |
|--------|--------|-----------|---------|
| **Operation** | Combines multiple tensors along an axis | Inserts size-1 dimensions | Removes size-1 dimensions |
| **Inputs** | Variadic (1 to 2,147,483,647) | Single input | Single input |
| **Axis Required** | v1: Optional (default=1), v4+: Required | Always required | Optional (can omit) |
| **Output Rank** | Same as input rank | `rank(input) + len(axes)` | `rank(input) - len(axes)` |
| **Shape Change** | Sum along axis, same elsewhere | Adds dimensions | Removes dimensions |

---

## References

- [ONNX Concat Operator Documentation](https://onnx.ai/onnx/operators/onnx__Concat.html)
- [NumPy concatenate Documentation](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)
- [PyTorch cat Documentation](https://pytorch.org/docs/stable/generated/torch.cat.html)

