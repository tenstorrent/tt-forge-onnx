# ONNX Flatten Complete Opset Version Summary

Based on the [official ONNX Flatten documentation](https://onnx.ai/onnx/operators/onnx__Flatten.html), this document provides a comprehensive summary of all opset versions.

## Overview

The **Flatten** operator flattens the input tensor into a 2D matrix. It takes a tensor of any rank and produces a 2D tensor by flattening dimensions up to (but not including) the specified `axis` into the outer dimension, and dimensions from `axis` onwards into the inner dimension.

**Key Behavior**: 
- Always produces a 2D output tensor
- Flattens dimensions before `axis` into the outer dimension
- Flattens dimensions from `axis` onwards into the inner dimension
- Default `axis=1` means first dimension stays, rest are flattened

**Example**: 
- Input: `[2, 3, 4, 5]` with `axis=1` → Output: `[2, 60]` (3×4×5 = 60)
- Input: `[2, 3, 4, 5]` with `axis=2` → Output: `[6, 20]` (2×3 = 6, 4×5 = 20)
- Input: `[2, 3, 4, 5]` with `axis=0` → Output: `[1, 120]` (all dimensions flattened)
- Input: `[2, 3, 4, 5]` with `axis=-1` → Output: `[120, 1]` (all but last flattened)

**Important**: 
- `axis` attribute behavior changed in v11: supports negative values (counting from the back)
- Type support expanded significantly across versions
- Always produces 2D output regardless of input rank

---

## Version-by-Version Breakdown

### **Flatten v1** (since version 1)

**Key Characteristics:**
- **Axis Range**: `[0, R]` where R is the rank of the input tensor
- **Negative Axis**: ❌ Not supported (only non-negative values)
- **Inputs**: 
  - `input` (T): A tensor of rank >= axis
- **Outputs**:
  - `output` (T): A 2D tensor with the contents of the input tensor
- **Type Constraints**: 
  - **LIMITED**: Only float types supported
  - `tensor(double)`, `tensor(float)`, `tensor(float16)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [0, R], where R is the rank of the input tensor. When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 … d_n)). |

**Examples (v1):**

```python
# Example 1: Default axis=1
# Input shape: (2, 3, 4, 5)
# axis=1: Flatten dimensions 1, 2, 3 → (2, 3×4×5) = (2, 60)
input = [[[[1, 2, 3, 4, 5], ...], ...], ...]  # shape: (2, 3, 4, 5)
output = [[1, 2, 3, ..., 120], [121, 122, ..., 240]]  # shape: (2, 60)

# Example 2: axis=0
# Input shape: (2, 3, 4, 5)
# axis=0: Flatten all dimensions → (1, 2×3×4×5) = (1, 120)
input = [[[[1, 2, 3, 4, 5], ...], ...], ...]  # shape: (2, 3, 4, 5)
output = [[1, 2, 3, ..., 120]]  # shape: (1, 120)

# Example 3: axis=2
# Input shape: (2, 3, 4, 5)
# axis=2: Flatten dimensions 0,1 → (6, 4×5) = (6, 20)
input = [[[[1, 2, 3, 4, 5], ...], ...], ...]  # shape: (2, 3, 4, 5)
output = [[1, 2, ..., 20], [21, 22, ..., 40], ..., [101, 102, ..., 120]]  # shape: (6, 20)
```

**Supported Types (v1):**
- Floating point: `float`, `double`, `float16`

**Limitations:**
- Only supports float tensor types
- Axis must be non-negative (no negative indexing)
- Limited type support

---

### **Flatten v9** (since version 9)

**Key Characteristics:**
- **Axis Range**: `[0, R]` where R is the rank of the input tensor (same as v1)
- **Negative Axis**: ❌ Not supported (only non-negative values)
- **Inputs**: 
  - `input` (T): A tensor of rank >= axis
- **Outputs**:
  - `output` (T): A 2D tensor with the contents of the input tensor
- **Type Constraints**: 
  - **EXPANDED**: Significantly expanded type support
  - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
  - `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`
  - `tensor(string)`
  - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [0, R], where R is the rank of the input tensor. When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 … d_n)). |

**Changes from v1:**
- ✅ **Type support expanded**: Added support for:
  - Boolean: `bool`
  - Complex: `complex128`, `complex64`
  - Integer (signed): `int8`, `int16`, `int32`, `int64`
  - Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
  - String: `string`
- Same axis behavior (non-negative only)

**Examples (v9):**

```python
# Example 1: Integer tensor
# Input shape: (2, 3, 4) with int32 dtype
# axis=1: Flatten dimensions 1, 2 → (2, 12)
input = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 
         [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]  # shape: (2, 3, 4)
output = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]  # shape: (2, 12)

# Example 2: Boolean tensor
# Input shape: (3, 2, 2) with bool dtype
# axis=1: Flatten dimensions 1, 2 → (3, 4)
input = [[[True, False], [True, True]], 
         [[False, False], [True, False]], 
         [[False, True], [False, False]]]  # shape: (3, 2, 2)
output = [[True, False, True, True],
          [False, False, True, False],
          [False, True, False, False]]  # shape: (3, 4)
```

**Supported Types (v9):**
- Boolean: `bool`
- Complex: `complex128`, `complex64`
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Flatten v11** (since version 11)

**Key Characteristics:**
- **Axis Range**: `[-r, r]` where r is the rank of the input tensor ⭐ **BREAKING CHANGE**
- **Negative Axis**: ✅ **Now supported** (counting dimensions from the back)
- **Inputs**: 
  - `input` (T): A tensor of rank >= axis
- **Outputs**:
  - `output` (T): A 2D tensor with the contents of the input tensor
- **Type Constraints**: 
  - **EXPANDED**: Added `tensor(bfloat16)`
  - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)`
  - `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`
  - `tensor(string)`
  - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of the input tensor. **Negative value means counting dimensions from the back.** When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 … d_n)). |

**Changes from v9:**
- ✅ **BREAKING CHANGE**: Axis range changed from `[0, R]` to `[-r, r]`
- ✅ **Negative axis support**: Can now use negative values to count from the back
- ✅ **Type support expanded**: Added `tensor(bfloat16)`

**Examples (v11):**

```python
# Example 1: Negative axis (NEW in v11)
# Input shape: (2, 3, 4, 5)
# axis=-1: Flatten all dimensions except the last → (2×3×4, 5) = (24, 5)
input = [[[[1, 2, 3, 4, 5], ...], ...], ...]  # shape: (2, 3, 4, 5)
output = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], ..., [116, 117, 118, 119, 120]]  # shape: (24, 5)

# Example 2: Negative axis=-2
# Input shape: (2, 3, 4, 5)
# axis=-2: Flatten dimensions 0, 1, 2 → (2×3×4, 5) = (24, 5) (same as axis=-1 for 4D tensor)
# For a 4D tensor: axis=-2 is equivalent to axis=2
input = [[[[1, 2, 3, 4, 5], ...], ...], ...]  # shape: (2, 3, 4, 5)
output = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], ..., [116, 117, 118, 119, 120]]  # shape: (24, 5)

# Example 3: axis=-3
# Input shape: (2, 3, 4, 5)
# axis=-3: Flatten dimensions 0, 1 → (2×3, 4×5) = (6, 20)
input = [[[[1, 2, 3, 4, 5], ...], ...], ...]  # shape: (2, 3, 4, 5)
output = [[1, 2, ..., 20], [21, 22, ..., 40], ..., [101, 102, ..., 120]]  # shape: (6, 20)

# Example 4: bfloat16 type (NEW in v11)
# Input shape: (2, 3, 4) with bfloat16 dtype
# axis=1: Flatten dimensions 1, 2 → (2, 12)
input = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], 
         [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]  # shape: (2, 3, 4), dtype: bfloat16
output = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
          [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]]  # shape: (2, 12), dtype: bfloat16
```

**Supported Types (v11):**
- Boolean: `bool`
- Complex: `complex128`, `complex64`
- Floating point: `float`, `double`, `float16`, `bfloat16` ⭐ (new)
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

**Why negative axis is useful:**
- More intuitive: `axis=-1` means "keep the last dimension"
- Consistent with NumPy/PyTorch negative indexing
- Easier to work with tensors of unknown rank

---

### **Flatten v13** (since version 13)

**Key Characteristics:**
- **Axis Range**: `[-r, r]` where r is the rank of the input tensor (same as v11)
- **Negative Axis**: ✅ Supported (same as v11)
- **Inputs**: 
  - `input` (T): A tensor of rank >= axis
- **Outputs**:
  - `output` (T): A 2D tensor with the contents of the input tensor
- **Type Constraints**: 
  - Same as v11:
    - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)`
    - `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`
    - `tensor(string)`
    - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 … d_n)). |

**Changes from v11:**
- No functional changes (same behavior)
- Documentation updates and clarifications

**Supported Types (v13):**
- Same as v11: Boolean, Complex, Floating point (including bfloat16), Integer (signed/unsigned), String

---

### **Flatten v21** (since version 21)

**Key Characteristics:**
- **Axis Range**: `[-r, r]` where r is the rank of the input tensor (same as v13)
- **Negative Axis**: ✅ Supported (same as v13)
- **Inputs**: 
  - `input` (T): A tensor of rank >= axis
- **Outputs**:
  - `output` (T): A 2D tensor with the contents of the input tensor
- **Type Constraints**: 
  - **EXPANDED**: Added new float types
  - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)`
  - **NEW**: `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)` ⭐
  - `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`
  - `tensor(string)`
  - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 … d_n)). |

**Changes from v13:**
- ✅ **Type support expanded**: Added new float types:
  - `tensor(float4e2m1)`: 4-bit floating point
  - `tensor(float8e4m3fn)`: 8-bit floating point (E4M3FN format)
  - `tensor(float8e4m3fnuz)`: 8-bit floating point (E4M3FNUZ format)
  - `tensor(float8e5m2)`: 8-bit floating point (E5M2 format)
  - `tensor(float8e5m2fnuz)`: 8-bit floating point (E5M2FNUZ format)

**Supported Types (v21):**
- Boolean: `bool`
- Complex: `complex128`, `complex64`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz` ⭐ (new)
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Flatten v23** (since version 23)

**Key Characteristics:**
- **Axis Range**: `[-r, r]` where r is the rank of the input tensor (same as v21)
- **Negative Axis**: ✅ Supported (same as v21)
- **Inputs**: 
  - `input` (T): A tensor of rank >= axis
- **Outputs**:
  - `output` (T): A 2D tensor with the contents of the input tensor
- **Type Constraints**: 
  - **EXPANDED**: Added more types
  - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)`
  - `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
  - **NEW**: `tensor(int4)` ⭐
  - `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`
  - `tensor(string)`
  - **NEW**: `tensor(uint4)` ⭐
  - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 … d_n)). |

**Changes from v21:**
- ✅ **Type support expanded**: Added:
  - `tensor(int4)`: 4-bit signed integer
  - `tensor(uint4)`: 4-bit unsigned integer

**Supported Types (v23):**
- Boolean: `bool`
- Complex: `complex128`, `complex64`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`
- Integer (signed): `int4` ⭐ (new), `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint4` ⭐ (new), `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

### **Flatten v24** (since version 24)

**Key Characteristics:**
- **Axis Range**: `[-r, r]` where r is the rank of the input tensor (same as v23)
- **Negative Axis**: ✅ Supported (same as v23)
- **Inputs**: 
  - `input` (T): A tensor of rank >= axis
- **Outputs**:
  - `output` (T): A 2D tensor with the contents of the input tensor
- **Type Constraints**: 
  - **EXPANDED**: Added more types
  - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)`
  - `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
  - `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`
  - `tensor(string)`
  - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 … d_n)). |

**Changes from v23:**
- No functional changes (same behavior)
- Type constraints updated to match IR version (IRv12)

**Supported Types (v24):**
- Same as v23: Boolean, Complex, Floating point (including new float types), Integer (signed/unsigned including int4/uint4), String

---

### **Flatten v25** (since version 25)

**Key Characteristics:**
- **Axis Range**: `[-r, r]` where r is the rank of the input tensor (same as v24)
- **Negative Axis**: ✅ Supported (same as v24)
- **Inputs**: 
  - `input` (T): A tensor of rank >= axis
- **Outputs**:
  - `output` (T): A 2D tensor with the contents of the input tensor
- **Type Constraints**: 
  - **EXPANDED**: Added more types
  - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)`
  - `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, **NEW**: `tensor(float8e8m0)` ⭐
  - **NEW**: `tensor(int2)` ⭐
  - `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`
  - `tensor(string)`
  - **NEW**: `tensor(uint2)` ⭐
  - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 … d_n)). |

**Changes from v24:**
- ✅ **Type support expanded**: Added:
  - `tensor(int2)`: 2-bit signed integer
  - `tensor(uint2)`: 2-bit unsigned integer
  - `tensor(float8e8m0)`: 8-bit floating point (E8M0 format)

**Supported Types (v25):**
- Boolean: `bool`
- Complex: `complex128`, `complex64`
- Floating point: `float`, `double`, `float16`, `bfloat16`, `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0` ⭐ (new)
- Integer (signed): `int2` ⭐ (new), `int4`, `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint2` ⭐ (new), `uint4`, `uint8`, `uint16`, `uint32`, `uint64`
- String: `string`

---

## Summary of Major Changes Across Versions

### Version Evolution Timeline

| Version | Key Changes |
|---------|-------------|
| **v1** | Initial version. Limited to float types (`float`, `double`, `float16`). Axis range: `[0, R]` (non-negative only). |
| **v9** | **Major type expansion**: Added support for bool, complex, integer (signed/unsigned), and string types. Axis range still `[0, R]`. |
| **v11** | **BREAKING CHANGE**: Axis range changed to `[-r, r]` (negative axis support). Added `bfloat16` type. |
| **v13** | No functional changes. Documentation updates. |
| **v21** | **New float types**: Added `float4e2m1`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`. |
| **v23** | **New integer types**: Added `int4` and `uint4`. |
| **v24** | No functional changes. Type constraints updated for IRv12. |
| **v25** | **New types**: Added `int2`, `uint2`, and `float8e8m0`. |

### Key Behavioral Changes

1. **Axis Range (v1 → v11)**:
   - **v1-v9**: `axis` must be in range `[0, R]` (non-negative only)
   - **v11+**: `axis` can be in range `[-r, r]` (negative values supported, counting from the back)
   - **Impact**: Code using negative axis values will fail in v1-v9 but work in v11+

2. **Type Support Evolution**:
   - **v1**: 3 types (float only)
   - **v9**: 15 types (added bool, complex, int, uint, string)
   - **v11**: 16 types (added bfloat16)
   - **v21**: 21 types (added new float formats)
   - **v23**: 23 types (added int4, uint4)
   - **v25**: 26 types (added int2, uint2, float8e8m0)

### Examples Demonstrating Version Differences

#### Example 1: Negative Axis (v11+ only)

```python
# Input: shape (2, 3, 4, 5)
# This works in v11+ but NOT in v1-v9

# v1-v9: ERROR - axis must be non-negative
# axis = -1  # ❌ Invalid in v1-v9

# v11+: SUCCESS - negative axis supported
# axis = -1  # ✅ Valid in v11+
# Result: shape (2×3×4, 5) = (24, 5)
```

#### Example 2: Type Support

```python
# v1: Only float types
input = tensor([[1.0, 2.0], [3.0, 4.0]], dtype=float32)  # ✅ Works
input = tensor([[1, 2], [3, 4]], dtype=int32)  # ❌ Fails in v1

# v9+: Integer types supported
input = tensor([[1, 2], [3, 4]], dtype=int32)  # ✅ Works in v9+
```

#### Example 3: Complex Flattening Scenarios

```python
# Scenario: 5D tensor (batch, channels, depth, height, width)
input_shape = (2, 3, 4, 5, 6)  # Total elements: 2×3×4×5×6 = 720

# axis=1 (default): Keep first dimension, flatten rest
# Output: (2, 3×4×5×6) = (2, 360)

# axis=2: Keep first 2 dimensions, flatten rest
# Output: (2×3, 4×5×6) = (6, 120)

# axis=-1 (v11+): Keep last dimension, flatten all before
# Output: (2×3×4×5, 6) = (120, 6)

# axis=-2 (v11+): Keep last 2 dimensions, flatten all before
# Output: (2×3×4, 5×6) = (24, 30)

# axis=0: Flatten everything
# Output: (1, 720)
```

---

## Comparison with NumPy

The ONNX Flatten operator is similar to NumPy's reshape operations:

```python
import numpy as np

# ONNX: Flatten with axis=1
# Input: shape (2, 3, 4, 5)
# Output: shape (2, 60)

# NumPy equivalent:
arr = np.random.randn(2, 3, 4, 5)
# Method 1: Using reshape
result = arr.reshape(2, -1)  # Flatten last 3 dims → (2, 60)

# Method 2: Using flatten with explicit calculation
outer_dim = arr.shape[0]  # 2
inner_dim = np.prod(arr.shape[1:])  # 3×4×5 = 60
result = arr.reshape(outer_dim, inner_dim)  # (2, 60)
```

**Key Differences**:
- ONNX Flatten always produces 2D output
- NumPy reshape can produce any shape
- ONNX Flatten uses `axis` to determine split point
- NumPy requires explicit shape calculation

---

## Comparison with PyTorch

```python
import torch

# ONNX: Flatten with axis=1
# Input: shape (2, 3, 4, 5)
# Output: shape (2, 60)

# PyTorch equivalent:
tensor = torch.randn(2, 3, 4, 5)
# Method 1: Using view
result = tensor.view(2, -1)  # Flatten last 3 dims → (2, 60)

# Method 2: Using flatten (PyTorch 0.4+)
result = tensor.flatten(start_dim=1)  # Flatten from dim 1 onwards → (2, 60)

# Method 3: Using reshape
result = tensor.reshape(2, -1)  # (2, 60)
```

**Key Differences**:
- ONNX Flatten always produces 2D output
- PyTorch `flatten()` can specify `start_dim` and `end_dim` for more control
- PyTorch `view()` and `reshape()` can produce any shape
- ONNX Flatten uses `axis` parameter (exclusive), PyTorch uses `start_dim` (inclusive)

---

## Implementation Notes

1. **Axis Validation**:
   - For v1-v9: Validate `axis` is in range `[0, R]`
   - For v11+: Validate `axis` is in range `[-r, r]`
   - Convert negative axis to positive: `axis = axis + rank` if `axis < 0`

2. **Output Shape Calculation**:
   ```python
   def calculate_flatten_output_shape(input_shape, axis):
       rank = len(input_shape)
       # Normalize axis (handle negative)
       if axis < 0:
           axis = axis + rank
       
       # Calculate outer dimension (product of dims [0:axis])
       outer_dim = 1
       for i in range(axis):
           outer_dim *= input_shape[i]
       
       # Calculate inner dimension (product of dims [axis:])
       inner_dim = 1
       for i in range(axis, rank):
           inner_dim *= input_shape[i]
       
       return (outer_dim, inner_dim)
   ```

3. **Type Constraints**:
   - Check input type against supported types for the opset version
   - Output type matches input type (no type conversion)

4. **Edge Cases**:
   - `axis=0`: All dimensions flattened → `(1, total_elements)`
   - `axis=rank`: Only first dimension kept → `(first_dim, 1)` if rank > 1
   - Scalar input (rank=0): Not supported (requires rank >= axis)

---

## References

- [ONNX Flatten Operator Documentation](https://onnx.ai/onnx/operators/onnx__Flatten.html)
- [NumPy Reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)
- [PyTorch Flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html)
- [PyTorch View](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)

