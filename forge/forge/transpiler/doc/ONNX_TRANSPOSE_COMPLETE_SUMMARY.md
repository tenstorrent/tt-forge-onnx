# ONNX Transpose Complete Opset Version Summary

Based on the [official ONNX Transpose documentation](https://onnx.ai/onnx/operators/onnx__Transpose.html), this document provides a comprehensive summary of all opset versions.

## Overview

Transpose returns a transpose of the input tensor (similar to numpy.transpose). The optional attribute `perm` must be a permutation of the dimensions of the input tensor. Axis `i` of the output tensor corresponds to the axis `perm[i]` of the input tensor.

**Default Behavior**: If the attribute `perm` is omitted, its default value is `(n-1, ..., 0)`, where `n` is the rank of the input tensor. This effectively reverses all dimensions.

**Example**: 
- When `perm=(1, 0, 2)`, given an input tensor of shape `(1, 2, 3)`, the output shape will be `(2, 1, 3)`.
- When `perm=(1, 2, 0)`, given an input tensor of shape `(1, 2, 3)`, the output shape will be `(2, 3, 1)`.

---

## Version-by-Version Breakdown

### **Transpose v1** (since version 1)

**Key Characteristics:**
- **Perm**: Attribute (`perm` as INTS attribute)
  - A list of integers specifying the permutation of dimensions
  - By default, reverse the dimensions (if omitted: `(n-1, ..., 0)`)
  - Its length must be equal to the rank of the input
- **Inputs**: 
  - `data` (T): An input tensor
- **Outputs**:
  - `transposed` (T): Transposed output
- **Type Constraints**: 
  - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`

**Supported Types (v1):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Transpose v13** (since version 13)

**Key Characteristics:**
- **Perm**: Attribute (`perm` as INTS attribute)
  - A list of integers specifying the permutation of dimensions
  - By default, reverse the dimensions (if omitted: `(n-1, ..., 0)`)
  - **Note**: v13 explicitly states "Its length must be equal to the rank of the input" (more explicit than v1)
- **Inputs**: 
  - `data` (T): An input tensor
- **Outputs**:
  - `transposed` (T): Transposed output
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(bfloat16)`, `tensor(int4)`, `tensor(uint4)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v1:**
- **Type support expanded**: Added `bfloat16`, `int4`, `uint4`
- More explicit documentation about `perm` length requirement

**Supported Types (v13):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16` ⭐ (new)
- Integer (signed): `int4` ⭐ (new), `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4` ⭐ (new), `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Transpose v21** (since version 21)

**Key Characteristics:**
- **Perm**: Attribute (`perm` as INTS attribute)
  - A list of integers specifying the permutation of dimensions
  - By default, reverse the dimensions (if omitted: `(n-1, ..., 0)`)
  - Its length must be equal to the rank of the input
- **Inputs**: 
  - `data` (T): An input tensor
- **Outputs**:
  - `transposed` (T): Transposed output
- **Type Constraints**: 
  - **EXPANDED**: Adds support for multiple new float types
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)` ⭐, `tensor(float8e4m3fn)` ⭐, `tensor(float8e4m3fnuz)` ⭐, `tensor(float8e5m2)` ⭐, `tensor(float8e5m2fnuz)` ⭐, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v13:**
- **Type support expanded**: Added `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`

**Supported Types (v21):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1` ⭐ (new), `float8e4m3fn` ⭐ (new), `float8e4m3fnuz` ⭐ (new), `float8e5m2` ⭐ (new), `float8e5m2fnuz` ⭐ (new)
- Integer (signed): `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Transpose v23** (since version 23)

**Key Characteristics:**
- **Perm**: Attribute (`perm` as INTS attribute)
  - A list of integers specifying the permutation of dimensions
  - By default, reverse the dimensions (if omitted: `(n-1, ..., 0)`)
  - Its length must be equal to the rank of the input
- **Inputs**: 
  - `data` (T): An input tensor
- **Outputs**:
  - `transposed` (T): Transposed output
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(float8e8m0)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)` ⭐, `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v21:**
- **Type support expanded**: Added `float8e8m0`

**Supported Types (v23):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0` ⭐ (new)
- Integer (signed): `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Transpose v24** (since version 24)

**Key Characteristics:**
- **Perm**: Attribute (`perm` as INTS attribute)
  - A list of integers specifying the permutation of dimensions
  - By default, reverse the dimensions (if omitted: `(n-1, ..., 0)`)
  - Its length must be equal to the rank of the input
- **Inputs**: 
  - `data` (T): An input tensor
- **Outputs**:
  - `transposed` (T): Transposed output
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(int2)`, `tensor(uint2)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)`, `tensor(int16)`, `tensor(int2)` ⭐, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint2)` ⭐, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`

**Changes from v23:**
- **Type support expanded**: Added `int2`, `uint2`

**Supported Types (v24):**
- Boolean: `bool`
- Complex: `complex64`, `complex128`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0`
- Integer (signed): `int2` ⭐ (new), `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint2` ⭐ (new), `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Transpose v25** (since version 25)

**Key Characteristics:**
- **Perm**: Attribute (`perm` as INTS attribute)
  - A list of integers specifying the permutation of dimensions
  - By default, reverse the dimensions (if omitted: `(n-1, ..., 0)`)
  - Its length must be equal to the rank of the input
- **Inputs**: 
  - `data` (T): An input tensor
- **Outputs**:
  - `transposed` (T): Transposed output
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
| **v13** | `bfloat16`, `int4`, `uint4` | 18 |
| **v21** | `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz` | 23 |
| **v23** | `float8e8m0` | 24 |
| **v24** | `int2`, `uint2` | 26 |
| **v25** | None (same as v24) | 26 |

### Attribute Behavior

- **`perm` attribute**: Remains consistent across all versions
  - Always an attribute (never becomes an input tensor)
  - Optional - defaults to `(n-1, ..., 0)` (reverse all dimensions)
  - Must be a permutation of `[0, 1, ..., n-1]` where `n` is the rank
  - Length must equal the rank of the input tensor

### Key Behavioral Notes

1. **Default Permutation**: If `perm` is omitted, the default is `(n-1, ..., 0)`, which reverses all dimensions. For example:
   - Input shape `(2, 3, 4)` → Output shape `(4, 3, 2)`

2. **Permutation Examples**:
   - `perm=(1, 0, 2)` on shape `(1, 2, 3)` → `(2, 1, 3)` (swap first two dims)
   - `perm=(1, 2, 0)` on shape `(1, 2, 3)` → `(2, 3, 1)` (rotate dimensions)
   - `perm=(0, 1, 2)` on shape `(1, 2, 3)` → `(1, 2, 3)` (identity, no change)

3. **Rank Zero Tensors**: The operator supports tensors of any rank, including rank-0 (scalar) tensors. For a scalar, `perm` would be empty `()` and the output remains a scalar.

4. **Type Support**: The operator supports a wide range of types, expanding significantly from v1 to v25, with additions of:
   - Low-precision integers (`int2`, `int4`, `uint2`, `uint4`)
   - Low-precision floats (`bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0`)
   - Complex numbers (`complex64`, `complex128`)
   - Strings (`string`)

---

## Implementation Considerations

### For Converter Implementation

1. **Attribute Handling**: 
   - `perm` is always an attribute (never an input tensor)
   - If `perm` is not provided, compute default: `tuple(range(rank-1, -1, -1))` or `tuple(reversed(range(rank)))`

2. **Type Support**:
   - Ensure the converter handles all supported types for the target opset version
   - Type support is additive - newer versions support all types from previous versions

3. **Edge Cases**:
   - **Rank-0 (scalar)**: `perm=()` → output is scalar (no-op)
   - **Rank-1 (vector)**: `perm=(0,)` → output is same (no-op)
   - **Identity permutation**: `perm=(0, 1, 2, ...)` → can be optimized to Identity operator

4. **Optimization Opportunities**:
   - If `perm` results in no change (identity permutation), use `Identity` operator
   - If `perm` is the default reverse and input is 1D, use `Identity` operator
   - For 2D tensors, `perm=(1, 0)` is equivalent to matrix transpose

---

## Comparison with NumPy

The ONNX Transpose operator is similar to `numpy.transpose()`:

```python
import numpy as np

# ONNX: perm=(1, 0, 2)
# NumPy equivalent:
arr = np.array(...)  # shape (1, 2, 3)
result = np.transpose(arr, axes=(1, 0, 2))  # shape (2, 1, 3)

# ONNX: default perm (reverse)
# NumPy equivalent:
arr = np.array(...)  # shape (2, 3, 4)
result = np.transpose(arr)  # or np.transpose(arr, axes=(2, 1, 0))  # shape (4, 3, 2)
```

---

## References

- [ONNX Transpose Operator Documentation](https://onnx.ai/onnx/operators/onnx__Transpose.html)
- [NumPy transpose Documentation](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)

