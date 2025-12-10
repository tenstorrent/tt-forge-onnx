# ONNX Add Complete Opset Version Summary

Based on the [official ONNX Add documentation](https://onnx.ai/onnx/operators/onnx__Add.html), this document provides a comprehensive summary of all opset versions.

## Overview

The **Add** operator performs element-wise binary addition between two tensors. It is one of the fundamental arithmetic operations in ONNX and supports broadcasting to handle tensors of different shapes.

**Key Behavior**: 
- Performs element-wise addition: `C = A + B`
- Supports multidirectional (NumPy-style) broadcasting (OPSET 7+)
- Limited broadcast support in earlier versions (OPSET 1-6)
- All inputs must have the same element type
- Output has the same element type as inputs

**Example**: 
- Input A: `[1, 2, 3]`, Input B: `[10, 20, 30]` → Output: `[11, 22, 33]`
- Input A: `[[1, 2], [3, 4]]`, Input B: `[10, 20]` → Output: `[[11, 22], [13, 24]]` (broadcasting)
- Input A: `[[1, 2], [3, 4]]`, Input B: `10` → Output: `[[11, 12], [13, 14]]` (scalar broadcasting)

**Important**: 
- Broadcasting behavior differs significantly between OPSET 1-6 and OPSET 7+
- OPSET 1-6: Requires `broadcast=1` attribute, limited broadcasting, uses `axis` attribute
- OPSET 7+: Multidirectional broadcasting always enabled, no attributes needed
- All input tensors must have the same element type (no type promotion)

---

## Version-by-Version Breakdown

### **Add v1** (since version 1)

**Key Characteristics:**
- **Broadcasting**: Limited broadcast support (requires `broadcast=1` attribute)
- **Broadcast Behavior**: Right-hand-side (B) is broadcasted to match left-hand-side (A)
- **Axis Attribute**: Optional `axis` attribute to specify broadcast dimensions
- **Legacy Attribute**: `consumed_inputs` (INTS): Legacy optimization attribute (deprecated)
- **Inputs**: 
  - `A` (T): First operand, should share the type with the second operand
  - `B` (T): Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size
- **Outputs**:
  - `C` (T): Result, has same dimensions and type as A
- **Type Constraints**: 
  - **LIMITED**: Only float types supported
  - `tensor(double)`, `tensor(float)`, `tensor(float16)`
- **Shape Inference**: ❌ No (shape inference: False)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `broadcast` | INT | ❌ | `0` | Pass 1 to enable broadcasting |
| `axis` | INT | ❌ | - | If set, defines the broadcast dimensions. See doc for details |
| `consumed_inputs` | INTS | ❌ | - | Legacy optimization attribute (deprecated) |

**Broadcasting Rules (v1):**
- Right-hand-side (B) can be:
  - Scalar tensor (empty shape `()`)
  - 1-element tensor (shape with all 1s, e.g., `(1, 1)`)
  - Contiguous subset of A's shape (suffix matching by default)
  - Contiguous subset starting at `axis` if specified
- 1-dim expansion doesn't work yet
- Examples of supported shapes (with `broadcast=1`):
  - `shape(A) = (2, 3, 4, 5)`, `shape(B) = ()` (scalar)
  - `shape(A) = (2, 3, 4, 5)`, `shape(B) = (1, 1)` (1-element tensor)
  - `shape(A) = (2, 3, 4, 5)`, `shape(B) = (5,)` (suffix match)
  - `shape(A) = (2, 3, 4, 5)`, `shape(B) = (4, 5)` (suffix match)
  - `shape(A) = (2, 3, 4, 5)`, `shape(B) = (3, 4)` with `axis=1` (axis-specified)
  - `shape(A) = (2, 3, 4, 5)`, `shape(B) = (2,)` with `axis=0` (axis-specified)

**Supported Types (v1):**
- Floating point: `float`, `double`, `float16`

**Limitations:**
- Only supports float tensor types
- No shape inference
- Limited broadcasting (requires explicit `broadcast=1`)
- Legacy `consumed_inputs` attribute (deprecated)
- 1-dim expansion doesn't work

---

### **Add v6** (since version 6)

**Key Characteristics:**
- **Broadcasting**: Limited broadcast support (requires `broadcast=1` attribute) - same as v1
- **Broadcast Behavior**: Right-hand-side (B) is broadcasted to match left-hand-side (A) - same as v1
- **Axis Attribute**: Optional `axis` attribute to specify broadcast dimensions - same as v1
- **Inputs**: 
  - `A` (T): First operand, should share the type with the second operand
  - `B` (T): Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size
- **Outputs**:
  - `C` (T): Result, has same dimensions and type as A
- **Type Constraints**: 
  - **EXPANDED**: Adds support for integer types
  - `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `broadcast` | INT | ❌ | `0` | Pass 1 to enable broadcasting |
| `axis` | INT | ❌ | - | If set, defines the broadcast dimensions. See doc for details |

**Changes from v1:**
- ✅ **Shape inference enabled**: Now supports shape inference
- ✅ **Extended type support**: Added integer types:
  - `tensor(int32)`, `tensor(int64)`
  - `tensor(uint32)`, `tensor(uint64)`
- ✅ **Removed legacy attribute**: `consumed_inputs` attribute removed (no longer needed)
- Same broadcasting limitations as v1

**Broadcasting Rules (v6):**
- Same as v1 (limited broadcasting, requires `broadcast=1`)

**Supported Types (v6):**
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int32`, `int64` ⭐ (new)
- Integer (unsigned): `uint32`, `uint64` ⭐ (new)

---

### **Add v7** (since version 7)

**Key Characteristics:**
- **Broadcasting**: Multidirectional (NumPy-style) broadcasting - **ALWAYS ENABLED**
- **Broadcast Behavior**: Full NumPy-style broadcasting with automatic dimension alignment
- **No Attributes**: All attributes removed (`broadcast`, `axis`, `consumed_inputs`)
- **Inputs**: 
  - `A` (T): First operand
  - `B` (T): Second operand
- **Outputs**:
  - `C` (T): Result, has same element type as two inputs
- **Type Constraints**: 
  - Same as v6: `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:** None (all removed)

**Changes from v6:**
- ✅ **BREAKING CHANGE**: Multidirectional broadcasting always enabled (no `broadcast` attribute needed)
- ✅ **BREAKING CHANGE**: Removed `broadcast` attribute (broadcasting is always on)
- ✅ **BREAKING CHANGE**: Removed `axis` attribute (automatic alignment from rightmost dimension)
- ✅ **Simplified API**: Cleaner operator definition with no attributes
- ✅ **More flexible**: Handles complex broadcasting scenarios automatically
- No type changes (same as v6)

**Broadcasting Rules (v7+):**
- Full NumPy-style broadcasting
- Automatic dimension alignment from right to left
- Dimensions of size 1 are automatically expanded
- Missing dimensions (on the left) are treated as size 1
- No need to specify `broadcast=1` or `axis`

**Supported Types (v7):**
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int32`, `int64`
- Integer (unsigned): `uint32`, `uint64`

**Why this is better:**
- More intuitive: Matches NumPy behavior
- More flexible: Handles complex broadcasting scenarios
- Cleaner code: No need to specify broadcast flags
- Automatic: Broadcasting rules applied automatically

---

### **Add v13** (since version 13)

**Key Characteristics:**
- **Broadcasting**: Multidirectional (NumPy-style) broadcasting - **ALWAYS ENABLED** (same as v7)
- **Broadcast Behavior**: Full NumPy-style broadcasting with automatic dimension alignment (same as v7)
- **No Attributes**: No attributes (same as v7)
- **Inputs**: 
  - `A` (T): First operand
  - `B` (T): Second operand
- **Outputs**:
  - `C` (T): Result, has same element type as two inputs
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(bfloat16)`
  - `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:** None (same as v7)

**Changes from v7:**
- ✅ **Type support expanded**: Added `tensor(bfloat16)`
- ✅ **Documentation updates**: Improved clarity
- No functional changes to broadcasting behavior

**Broadcasting Rules (v13):**
- Same as v7 (full NumPy-style broadcasting)

**Supported Types (v13):**
- Floating point: `float`, `double`, `float16`, `bfloat16` ⭐ (new)
- Integer (signed): `int32`, `int64`
- Integer (unsigned): `uint32`, `uint64`

---

### **Add v14** (since version 14)

**Key Characteristics:**
- **Broadcasting**: Multidirectional (NumPy-style) broadcasting - **ALWAYS ENABLED** (same as v7+)
- **Broadcast Behavior**: Full NumPy-style broadcasting with automatic dimension alignment (same as v7+)
- **No Attributes**: No attributes (same as v7+)
- **Inputs**: 
  - `A` (T): First operand
  - `B` (T): Second operand
- **Outputs**:
  - `C` (T): Result, has same element type as two inputs
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `uint8`, `int8`, `uint16`, and `int16`
  - Full list: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No
- **Support Level**: COMMON

**Attributes:** None (same as v7+)

**Changes from v13:**
- ✅ **Type support expanded**: Added `uint8`, `int8`, `uint16`, and `int16`
- ✅ **Documentation updates**: Further refinements
- No functional changes to broadcasting behavior

**Broadcasting Rules (v14):**
- Same as v7+ (full NumPy-style broadcasting)

**Supported Types (v14):**
- Floating point: `float`, `double`, `float16`, `bfloat16`
- Integer (signed): `int8` ⭐ (new), `int16` ⭐ (new), `int32`, `int64`
- Integer (unsigned): `uint8` ⭐ (new), `uint16` ⭐ (new), `uint32`, `uint64`

---

## Summary of Changes Across Versions

### Type Support Evolution

| Version | New Types Added | Total Types | Key Changes |
|---------|----------------|-------------|-------------|
| **v1** | Base float types | 3 | Initial version, float types only |
| **v6** | `int32`, `int64`, `uint32`, `uint64` | 7 | Added integer types, shape inference enabled |
| **v7** | None (same as v6) | 7 | **BREAKING**: Multidirectional broadcasting, removed attributes |
| **v13** | `bfloat16` | 8 | Added bfloat16 support |
| **v14** | `int8`, `int16`, `uint8`, `uint16` | 12 | Added 8-bit and 16-bit integer types |

### Broadcasting Evolution

| Version | Broadcasting | Attributes | Key Behavior |
|---------|--------------|------------|--------------|
| **v1** | Limited (requires `broadcast=1`) | `broadcast`, `axis`, `consumed_inputs` | Right-hand-side broadcasted to left-hand-side, suffix matching or axis-specified |
| **v6** | Limited (requires `broadcast=1`) | `broadcast`, `axis` | Same as v1, removed `consumed_inputs` |
| **v7+** | Multidirectional (always on) | None | Full NumPy-style broadcasting, automatic alignment |

### Attribute Evolution

| Version | `broadcast` | `axis` | `consumed_inputs` | Notes |
|--------|------------|--------|-------------------|-------|
| **v1** | ✅ Required (default: 0) | ✅ Optional | ✅ Present (legacy) | Must set `broadcast=1` to enable |
| **v6** | ✅ Required (default: 0) | ✅ Optional | ❌ Removed | Must set `broadcast=1` to enable |
| **v7+** | ❌ Removed | ❌ Removed | ❌ Removed | Broadcasting always enabled |

### Shape Inference Evolution

| Version | Shape Inference | Notes |
|---------|----------------|------|
| **v1** | ❌ No | No automatic shape inference |
| **v6+** | ✅ Yes | Shape inference enabled |

---

## Key Behavioral Notes

1. **Broadcasting Behavior**: 
   - **v1-v6**: Limited broadcasting, requires `broadcast=1` attribute
     - Right-hand-side (B) is broadcasted to match left-hand-side (A)
     - Uses suffix matching by default, or `axis` attribute for non-suffix alignment
     - 1-dim expansion doesn't work
   - **v7+**: Multidirectional (NumPy-style) broadcasting always enabled
     - Automatic dimension alignment from right to left
     - Dimensions of size 1 are automatically expanded
     - Missing dimensions treated as size 1

2. **Type Requirements**:
   - All inputs must have the same element type
   - Output has the same element type as inputs
   - No type promotion is performed

3. **Input Requirements**:
   - Always exactly 2 inputs: `A` and `B`
   - Both inputs must have compatible shapes (for broadcasting) or same shapes

4. **Output Characteristics**:
   - Output shape is determined by broadcasting rules
   - Output type matches input types

5. **Legacy Attributes**:
   - **v1 only**: `consumed_inputs` attribute (deprecated, removed in v6)

6. **Breaking Changes**:
   - **v7**: Major breaking change - removed `broadcast` and `axis` attributes
   - Models using v1-v6 with explicit `broadcast=1` will need to be updated for v7+
   - Broadcasting behavior is fundamentally different between v6 and v7

---

## Implementation Considerations

### For Converter Implementation

1. **Broadcasting Handling**:
   - **v1-v6**: 
     - Check for `broadcast` attribute (default: 0)
     - If `broadcast=1`, apply limited broadcasting rules
     - Use `axis` attribute if provided for non-suffix alignment
     - Right-hand-side (B) is broadcasted to match left-hand-side (A)
   - **v7+**: 
     - Broadcasting is always enabled (no attribute check needed)
     - Apply full NumPy-style broadcasting rules
     - Automatic dimension alignment from right to left

2. **Type Support**:
   - Ensure the converter handles all supported types for the target opset version
   - Type support is additive - newer versions support all types from previous versions
   - All inputs must have the same element type (no type promotion)

3. **Shape Inference**:
   - **v1**: No shape inference (must be provided explicitly)
   - **v6+**: Shape inference is supported
   - Calculate output shape based on broadcasting rules

4. **Input Validation**:
   - Verify that both inputs have the same element type
   - Verify that shapes are compatible for broadcasting (v7+) or match the broadcasting rules (v1-v6)
   - For v1-v6: If `broadcast=0`, verify that shapes match exactly

5. **Legacy Support**:
   - **v1**: Ignore `consumed_inputs` attribute (deprecated)
   - **v1-v6**: Handle `broadcast` and `axis` attributes appropriately

6. **Output Shape Calculation**:
   - **v1-v6**: Output shape matches input A's shape (B is broadcasted to match A)
   - **v7+**: Output shape is the element-wise maximum of input shapes after alignment

---

## Comparison with NumPy

The ONNX Add operator is similar to NumPy's addition with broadcasting:

```python
import numpy as np

# ONNX: Add with same shapes
# NumPy equivalent:
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
result = a + b  # [11, 22, 33]

# ONNX: Add with broadcasting (v7+)
# NumPy equivalent:
a = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
result = a + b  # [[11, 22], [13, 24]]

# ONNX: Add with scalar (v7+)
# NumPy equivalent:
a = np.array([[1, 2], [3, 4]])
b = 10
result = a + b  # [[11, 12], [13, 14]]
```

**Key Differences**:
- **v1-v6**: ONNX requires explicit `broadcast=1` attribute, NumPy always broadcasts
- **v7+**: ONNX behavior matches NumPy exactly (multidirectional broadcasting)
- **v1-v6**: ONNX has limited broadcasting (suffix matching or axis-specified), NumPy has full multidirectional broadcasting
- ONNX requires same element types (no type promotion), NumPy may promote types

**PyTorch Comparison**:
```python
import torch

# ONNX: Add with same shapes
# PyTorch equivalent:
a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30])
result = a + b  # [11, 22, 33]

# ONNX: Add with broadcasting (v7+)
# PyTorch equivalent:
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([10, 20])
result = a + b  # [[11, 22], [13, 24]]
```

**Note**: PyTorch addition always supports broadcasting (similar to ONNX v7+), and PyTorch may perform type promotion in some cases.

---

## Differences from Other Operators

| Aspect | Add | Mul | Sub |
|--------|-----|-----|-----|
| **Operation** | Addition (`A + B`) | Multiplication (`A * B`) | Subtraction (`A - B`) |
| **Inputs** | 2 inputs | 2 inputs | 2 inputs |
| **Broadcasting** | v1-v6: Limited, v7+: Multidirectional | v1-v6: Limited, v7+: Multidirectional | v1-v6: Limited, v7+: Multidirectional |
| **Attributes** | v1-v6: `broadcast`, `axis`, v7+: None | v1-v6: `broadcast`, `axis`, v7+: None | v1-v6: `broadcast`, `axis`, v7+: None |
| **Type Support** | Float + Integers (v6+), Extended (v14) | Float + Integers (v6+), Extended (v14) | Float + Integers (v6+), Extended (v14) |
| **Shape Inference** | v1: No, v6+: Yes | v1: No, v6+: Yes | v1: No, v6+: Yes |

**Note**: Add, Mul, and Sub operators have very similar evolution patterns across versions, with the main difference being the mathematical operation performed.

---

## References

- [ONNX Add Operator Documentation](https://onnx.ai/onnx/operators/onnx__Add.html)
- [ONNX Broadcasting Documentation](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [PyTorch Tensor Operations](https://pytorch.org/docs/stable/torch.html#math-operations)


