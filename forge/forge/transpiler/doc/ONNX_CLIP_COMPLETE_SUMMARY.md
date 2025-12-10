# ONNX Clip Complete Opset Version Summary

Based on the [official ONNX Clip documentation](https://onnx.ai/onnx/operators/onnx__Clip.html), this document provides a comprehensive summary of all opset versions.

## Overview

Clip operator limits the given input within an interval. The interval is specified by `min` and `max` values. Elements below `min` are replaced by `min`, and elements above `max` are replaced by `max`.

**Key Behavior**: 
- When `min` is greater than `max`, the clip operator sets all the `input` values to the value of `max` (v13+).
- This is equivalent to `Min(max, Max(input, min))`.
- Default values: `min` defaults to `numeric_limits::lowest()`, `max` defaults to `numeric_limits::max()`.

**Example**: 
- Input `[1, 2, 3, 4, 5]` with `min=2`, `max=4` → Output `[2, 2, 3, 4, 4]`
- Input `[1, 2, 3, 4, 5]` with `min=10`, `max=4` → Output `[4, 4, 4, 4, 4]` (v13+: all values set to max when min > max)
- Input `[1, 2, 3, 4, 5]` with `min=None`, `max=3` → Output `[1, 2, 3, 3, 3]` (min defaults to lowest)

**Important**: 
- `min` and `max` are optional (can be omitted)
- In v1-v6: `min` and `max` are attributes with default values
- In v11+: `min` and `max` are optional input tensors (scalars)
- All input tensors must have the same type

---

## Version-by-Version Breakdown

### **Clip v1** (since version 1)

**Key Characteristics:**
- **Min/Max**: Attributes (`min` and `max` as FLOAT attributes) - **OPTIONAL**
  - `min`: Minimum value, under which element is replaced by min
  - `max`: Maximum value, above which element is replaced by max
  - Defaults: `min` defaults to `numeric_limits::lowest()`, `max` defaults to `numeric_limits::max()`
- **Legacy Attribute**: 
  - `consumed_inputs` (INTS): Legacy optimization attribute (deprecated)
- **Inputs**: 
  - `input` (T): Input tensor whose elements to be clipped
- **Outputs**:
  - `output` (T): Output tensor with clipped input elements
- **Type Constraints**: 
  - **LIMITED**: Only float types supported
  - `tensor(double)`, `tensor(float)`, `tensor(float16)`
- **Shape Inference**: ❌ No (shape inference: False)
- **Function**: ❌ No

**Supported Types (v1):**
- Floating point: `float`, `double`, `float16`

**Limitations:**
- Only supports float tensor types
- No shape inference
- Legacy `consumed_inputs` attribute (deprecated)

---

### **Clip v6** (since version 6)

**Key Characteristics:**
- **Min/Max**: Attributes (`min` and `max` as FLOAT attributes) - **OPTIONAL**
  - `min`: Minimum value, under which element is replaced by min
    - **Default value**: `-3.402823e+38` (approximately `numeric_limits<float>::lowest()`)
  - `max`: Maximum value, above which element is replaced by max
    - **Default value**: `3.402823e+38` (approximately `numeric_limits<float>::max()`)
- **Inputs**: 
  - `input` (T): Input tensor whose elements to be clipped
- **Outputs**:
  - `output` (T): Output tensor with clipped input elements
- **Type Constraints**: 
  - Same as v1: `tensor(double)`, `tensor(float)`, `tensor(float16)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No

**Changes from v1:**
- **Removed**: `consumed_inputs` legacy attribute
- **Shape inference enabled**: Now supports shape inference
- **Explicit default values**: Default min/max values are now explicitly documented

**Supported Types (v6):**
- Floating point: `float`, `double`, `float16`

---

### **Clip v11** (since version 11)

**Key Characteristics:**
- **Min/Max**: Optional input tensors (not attributes) - **OPTIONAL**
  - `min` (optional, T): Minimum value, under which element is replaced by min
    - Must be a scalar (tensor of empty shape)
    - Defaults to `numeric_limits::lowest()` if not provided
  - `max` (optional, T): Maximum value, above which element is replaced by max
    - Must be a scalar (tensor of empty shape)
    - Defaults to `numeric_limits::max()` if not provided
- **Inputs**: 
  - Between 1 and 3 inputs
  - `input` (T): Input tensor whose elements to be clipped
  - `min` (optional, T): Minimum value scalar tensor
  - `max` (optional, T): Maximum value scalar tensor
- **Outputs**:
  - `output` (T): Output tensor with clipped input elements
- **Type Constraints**: 
  - Same as v6: `tensor(double)`, `tensor(float)`, `tensor(float16)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No

**Changes from v6:**
- **BREAKING CHANGE**: `min` and `max` are now **input tensors** instead of attributes
- **Dynamic min/max**: Min and max can now be provided at runtime as tensors
- **Scalar requirement**: Min and max input tensors must be scalars (empty shape)
- No type changes

**Supported Types (v11):**
- Floating point: `float`, `double`, `float16`

---

### **Clip v12** (since version 12)

**Key Characteristics:**
- **Min/Max**: Optional input tensors (not attributes) - **OPTIONAL**
  - Same as v11: `min` and `max` are optional input scalar tensors
- **Inputs**: 
  - Between 1 and 3 inputs
  - `input` (T): Input tensor whose elements to be clipped
  - `min` (optional, T): Minimum value scalar tensor
  - `max` (optional, T): Maximum value scalar tensor
- **Outputs**:
  - `output` (T): Output tensor with clipped input elements
- **Type Constraints**: 
  - **EXPANDED**: Adds support for integer types
  - Full list: `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ❌ No

**Changes from v11:**
- **Type support expanded**: Added `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`
- No functional changes to min/max handling

**Supported Types (v12):**
- Floating point: `float`, `double`, `float16`
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`

---

### **Clip v13** (since version 13)

**Key Characteristics:**
- **Min/Max**: Optional input tensors (not attributes) - **OPTIONAL**
  - Same as v12: `min` and `max` are optional input scalar tensors
- **Inputs**: 
  - Between 1 and 3 inputs
  - `input` (T): Input tensor whose elements to be clipped
  - `min` (optional, T): Minimum value scalar tensor
  - `max` (optional, T): Maximum value scalar tensor
- **Outputs**:
  - `output` (T): Output tensor with clipped input elements
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(bfloat16)`
  - Full list: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ Yes (shape inference: True)
- **Function**: ✅ Yes (function: True)

**Changes from v12:**
- **Type support expanded**: Added `bfloat16`
- **Function support**: Now supports function attribute (can be used in function definitions)
- **Behavior clarification**: Explicitly documents that when `min > max`, all values are set to `max`

**Supported Types (v13):**
- Floating point: `float`, `double`, `float16`, `bfloat16` ⭐ (new)
- Integer (signed): `int8`, `int16`, `int32`, `int64`
- Integer (unsigned): `uint8`, `uint16`, `uint32`, `uint64`

---

## Summary of Changes Across Versions

### Type Support Evolution

| Version | New Types Added | Total Types |
|---------|----------------|-------------|
| **v1** | Base float types | 3 |
| **v6** | None (same as v1) | 3 |
| **v11** | None (same as v6) | 3 |
| **v12** | `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64` | 11 |
| **v13** | `bfloat16` | 12 |

### Min/Max Parameter Evolution

| Version | Min/Max Format | Default Values | Shape Inference | Function |
|---------|----------------|----------------|-----------------|----------|
| **v1** | Attributes (FLOAT) | `numeric_limits::lowest()` / `numeric_limits::max()` | ❌ No | ❌ No |
| **v6** | Attributes (FLOAT) | `-3.402823e+38` / `3.402823e+38` | ✅ Yes | ❌ No |
| **v11-v13** | Input tensors (scalars) | `numeric_limits::lowest()` / `numeric_limits::max()` | ✅ Yes | v13: ✅ Yes |

### Key Behavioral Notes

1. **Min/Max Format**: 
   - **v1-v6**: `min` and `max` are attributes (FLOAT)
   - **v11+**: `min` and `max` are optional input tensors (must be scalars)

2. **Default Values**:
   - `min` defaults to `numeric_limits::lowest()` (most negative value for the type)
   - `max` defaults to `numeric_limits::max()` (largest positive value for the type)
   - In v6, explicit defaults are `-3.402823e+38` and `3.402823e+38` (float limits)

3. **Edge Case (v13+)**:
   - When `min > max`, the clip operator sets all input values to `max`
   - Example: `min=10`, `max=5` → all values become `5`

4. **Input Requirements**:
   - **v1-v6**: Single input tensor
   - **v11+**: Between 1 and 3 inputs (input, optional min, optional max)
   - Min and max input tensors must be scalars (empty shape)

5. **Type Consistency**:
   - All inputs (input, min, max) must have the same type
   - Output type matches input type

6. **Legacy Attribute**:
   - **v1 only**: `consumed_inputs` attribute (deprecated, removed in v6)

7. **Function Support**:
   - **v13+**: Supports function attribute (can be used in function definitions)

---

## Implementation Considerations

### For Converter Implementation

1. **Min/Max Handling**:
   - **v1-v6**: Extract `min` and `max` from attributes
     - Default to `numeric_limits::lowest()` and `numeric_limits::max()` if not provided
   - **v11+**: Extract `min` and `max` from input tensors
     - Check if `node_proto.input[1]` exists for `min`
     - Check if `node_proto.input[2]` exists for `max`
     - Both must be scalar tensors (empty shape)

2. **Type Support**:
   - Ensure the converter handles all supported types for the target opset version
   - Type support is additive - newer versions support all types from previous versions
   - All inputs (input, min, max) must have the same type

3. **Edge Cases**:
   - **v13+**: Handle case where `min > max` (set all values to `max`)
   - Handle missing min/max (use defaults)
   - Validate that min/max input tensors are scalars (v11+)

4. **Default Values**:
   - When min/max are not provided, use appropriate defaults based on dtype:
     - Float types: `-inf` / `+inf` or `numeric_limits::lowest()` / `numeric_limits::max()`
     - Integer types: `min_value` / `max_value` for the integer type

5. **Input Validation**:
   - **v11+**: Verify min/max input tensors are scalars (shape is empty or `()`)
   - Verify all inputs have the same type
   - Verify number of inputs is between 1 and 3 (v11+)

6. **Legacy Support**:
   - **v1**: Ignore `consumed_inputs` attribute (deprecated)

---

## Comparison with NumPy

The ONNX Clip operator is similar to `numpy.clip()`:

```python
import numpy as np

# ONNX: Clip with min=2, max=4
# NumPy equivalent:
arr = np.array([1, 2, 3, 4, 5])
result = np.clip(arr, 2, 4)  # [2, 2, 3, 4, 4]

# ONNX: Clip with only max=3 (min defaults to lowest)
# NumPy equivalent:
arr = np.array([1, 2, 3, 4, 5])
result = np.clip(arr, None, 3)  # [1, 2, 3, 3, 3]

# ONNX: Clip with only min=2 (max defaults to max)
# NumPy equivalent:
arr = np.array([1, 2, 3, 4, 5])
result = np.clip(arr, 2, None)  # [2, 2, 3, 4, 5]
```

**Key Differences**:
- NumPy `clip()` always supports min/max as parameters (not tensors)
- ONNX v1-v6 uses attributes, v11+ uses input tensors
- ONNX v13+ has special behavior when `min > max` (sets all to max)
- NumPy `clip()` doesn't have this special behavior

**PyTorch Comparison**:
```python
import torch

# ONNX: Clip with min=2, max=4
# PyTorch equivalent:
x = torch.tensor([1, 2, 3, 4, 5])
result = torch.clamp(x, min=2, max=4)  # [2, 2, 3, 4, 4]

# ONNX: Clip with only max=3
# PyTorch equivalent:
x = torch.tensor([1, 2, 3, 4, 5])
result = torch.clamp(x, max=3)  # [1, 2, 3, 3, 3]

# ONNX: Clip with only min=2
# PyTorch equivalent:
x = torch.tensor([1, 2, 3, 4, 5])
result = torch.clamp(x, min=2)  # [2, 2, 3, 4, 5]
```

**Note**: PyTorch `clamp()` uses `min` and `max` as keyword arguments, similar to ONNX v1-v6 attributes. PyTorch also doesn't have the special behavior when `min > max` (v13+).

---

## Differences from Other Operators

| Aspect | Clip | Concat | Unsqueeze |
|--------|------|--------|-----------|
| **Operation** | Clamps values to range | Combines tensors | Inserts dimensions |
| **Inputs** | 1-3 inputs (v11+) | Variadic (1 to 2,147,483,647) | Single input |
| **Min/Max** | Optional (attributes v1-v6, inputs v11+) | N/A | N/A |
| **Type Support** | Float + Integers (v12+) | All types (v4+) | All types (v1+) |
| **Default Values** | Min/max have defaults | Axis has default (v1 only) | Axes required |

---

## References

- [ONNX Clip Operator Documentation](https://onnx.ai/onnx/operators/onnx__Clip.html)
- [NumPy clip Documentation](https://numpy.org/doc/stable/reference/generated/numpy.clip.html)
- [PyTorch clamp Documentation](https://pytorch.org/docs/stable/generated/torch.clamp.html)

