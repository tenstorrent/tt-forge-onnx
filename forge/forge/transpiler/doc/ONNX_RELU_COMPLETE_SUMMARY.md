# ONNX Relu Operator - Complete Summary

## Overview

The **Relu** (Rectified Linear Unit) operator applies the rectified linear function elementwise to the input tensor. The function is defined as:

```
y = max(0, x)
```

This operator takes one input tensor and produces one output tensor where all negative values are set to zero, and all non-negative values remain unchanged.

## Version History

| Version | Since | Shape Inference | Function | Key Changes |
|---------|-------|----------------|----------|-------------|
| 1 | 1 | ❌ | ❌ | Initial version with `consumed_inputs` attribute |
| 6 | 6 | ✅ | ❌ | Added shape inference, removed `consumed_inputs` |
| 13 | 13 | ✅ | ❌ | Added `bfloat16` type support |
| 14 | 14 | ✅ | ✅ | Added integer types, converted to function |

---

## Relu - Version 1

**Since Version:** 1  
**Shape Inference:** ❌ False  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, `y = max(0, x)`, is applied to the tensor elementwise.

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `consumed_inputs` | INTS | ❌ | - | Legacy optimization attribute. This attribute is deprecated and should not be used. |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input tensor |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor with rectified linear function applied elementwise |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Shape Inference:** Not supported in v1
- **Legacy Attribute:** The `consumed_inputs` attribute is a legacy optimization hint and should be ignored in modern implementations
- **Type Support:** Limited to floating-point types only

---

## Relu - Version 6

**Since Version:** 6  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, `y = max(0, x)`, is applied to the tensor elementwise.

### Attributes

None

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input tensor |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor with rectified linear function applied elementwise |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Changes from v1

1. ✅ **Shape Inference Added:** The operator now supports shape inference
2. ❌ **Removed `consumed_inputs` Attribute:** The legacy optimization attribute was removed
3. 📊 **Type Support:** Unchanged (still 3 float types)

### Notes

- **Shape Inference:** Now supported, allowing automatic shape propagation
- **Cleaner API:** Removed deprecated `consumed_inputs` attribute

---

## Relu - Version 13

**Since Version:** 13  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, `y = max(0, x)`, is applied to the tensor elementwise.

### Attributes

None

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input tensor |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor with rectified linear function applied elementwise |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 4 types

**Description:** Constrain input and output types to float tensors.

### Changes from v6

1. ➕ **Added `bfloat16` Type:** Extended type support to include Brain Floating Point 16-bit format
2. 📊 **Type Count:** Increased from 3 to 4 types

### Notes

- **bfloat16 Support:** Added support for the bfloat16 floating-point format, commonly used in machine learning for better numerical stability compared to float16

---

## Relu - Version 14

**Since Version:** 14  
**Shape Inference:** ✅ True  
**Function:** ✅ True  
**Support Level:** COMMON

### Summary

Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, `y = max(0, x)`, is applied to the tensor elementwise.

### Function Body

The operator is now defined as a function with the following body:

```onnx
<domain: "", opset_import: ["" : 18]>
Relu (X) => (Y)
{
   Zero = Constant <value: tensor = float {0}> ()
   ZeroCast = CastLike (Zero, X)
   Y = Max (X, ZeroCast)
}
```

**Function Implementation:**
1. Creates a constant tensor with value `0.0` (float type)
2. Casts the zero constant to match the input tensor's type using `CastLike`
3. Applies `Max` operation between the input tensor and the zero tensor

### Attributes

None

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `X` | T | Input tensor |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor with rectified linear function applied elementwise |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(int16)`
- `tensor(int32)`
- `tensor(int64)`
- `tensor(int8)`

**Total:** 8 types

**Description:** Constrain input and output types to signed numeric tensors.

### Changes from v13

1. ✅ **Converted to Function:** The operator is now implemented as a function using `Constant`, `CastLike`, and `Max` operators
2. ➕ **Added Integer Types:** Extended type support to include signed integer types:
   - `int8`
   - `int16`
   - `int32`
   - `int64`
3. 📊 **Type Count:** Increased from 4 to 8 types
4. 🔄 **Function Body:** The operator behavior is now explicitly defined through a function body, making it easier to understand and potentially optimize

### Notes

- **Function Implementation:** The function body provides a clear, composable definition of Relu using standard ONNX operators
- **Integer Support:** Relu can now be applied to integer tensors, where negative values are clamped to 0
- **Type Flexibility:** The `CastLike` operation ensures the zero constant matches the input tensor's type automatically
- **Backward Compatibility:** The function implementation maintains the same mathematical behavior as previous versions

---

## Version Comparison Summary

### Type Support Evolution

| Version | Float Types | Integer Types | Total Types |
|---------|------------|---------------|-------------|
| v1 | double, float, float16 | - | 3 |
| v6 | double, float, float16 | - | 3 |
| v13 | bfloat16, double, float, float16 | - | 4 |
| v14 | bfloat16, double, float, float16 | int8, int16, int32, int64 | 8 |

### Feature Evolution

| Feature | v1 | v6 | v13 | v14 |
|---------|----|----|-----|-----|
| Shape Inference | ❌ | ✅ | ✅ | ✅ |
| Function Body | ❌ | ❌ | ❌ | ✅ |
| `consumed_inputs` Attribute | ✅ | ❌ | ❌ | ❌ |
| bfloat16 Support | ❌ | ❌ | ✅ | ✅ |
| Integer Types Support | ❌ | ❌ | ❌ | ✅ |

---

## Behavioral Notes

### Mathematical Definition

The Relu operator applies the following function elementwise:

```
y = max(0, x)
```

**Behavior:**
- If `x >= 0`: `y = x` (unchanged)
- If `x < 0`: `y = 0` (clamped to zero)

### Examples

**Example 1: Basic Float Relu**
```
Input:  X = [-1.0, 0.0, 2.5, -3.0, 4.0]
Output: Y = [ 0.0, 0.0, 2.5,  0.0, 4.0]
```

**Example 2: Integer Relu (v14+)**
```
Input:  X = [-5, 0, 10, -2, 7]
Output: Y = [ 0, 0, 10,  0, 7]
```

**Example 3: Multi-dimensional Tensor**
```
Input Shape:  (2, 3)
Input:        [[-1.0,  2.0, -0.5],
               [ 0.0, -3.0,  1.5]]
Output:       [[ 0.0,  2.0,  0.0],
               [ 0.0,  0.0,  1.5]]
```

### Edge Cases

1. **Zero Values:** Zero values remain zero (identity for zero)
2. **Negative Values:** All negative values are set to zero
3. **Positive Values:** All positive values remain unchanged
4. **Integer Types (v14+):** Integer negative values are clamped to 0 (integer zero)

---

## Implementation Considerations

### Function-Based Implementation (v14+)

In version 14, Relu is defined as a function using:
- `Constant`: Creates a zero tensor
- `CastLike`: Ensures type compatibility
- `Max`: Performs the elementwise maximum operation

**Advantages:**
- Clear, composable definition
- Easier to understand and verify
- Can be optimized by runtime implementations
- Type flexibility through `CastLike`

### Direct Implementation

For performance-critical scenarios, implementations may choose to:
- Use optimized Relu kernels instead of the function decomposition
- Leverage hardware-specific Relu instructions (e.g., SIMD operations)
- Apply in-place operations when possible

### Type Handling

- **Float Types:** Standard floating-point Relu operation
- **Integer Types (v14+):** Integer Relu where negative values become 0
- **Type Promotion:** The function body handles type matching automatically via `CastLike`

### Shape Inference

Since v6, the operator supports shape inference:
- Output shape always matches input shape
- No broadcasting or dimension changes occur

---

## Differences from Similar Operators

### Relu vs. Clip

| Aspect | Relu | Clip |
|--------|------|------|
| Lower Bound | Fixed at 0 | Configurable `min` |
| Upper Bound | None (unbounded) | Configurable `max` |
| Parameters | None (v6+) | `min` and `max` values |
| Use Case | Simple non-linearity | General value clamping |

**Relationship:** `Relu(X)` is equivalent to `Clip(X, min=0, max=None)` for float types.

### Relu vs. LeakyRelu

| Aspect | Relu | LeakyRelu |
|--------|------|-----------|
| Negative Slope | 0 (hard cutoff) | Configurable (typically small, e.g., 0.01) |
| Negative Values | Set to 0 | Multiplied by slope |
| Parameters | None | `alpha` (slope) |

---

## Common Use Cases

1. **Neural Network Activation:** Primary activation function in many deep learning models
2. **Non-linearity Introduction:** Adds non-linearity to linear transformations
3. **Sparsity Induction:** Creates sparse activations (many zeros)
4. **Gradient Flow:** Helps with gradient flow in deep networks (compared to sigmoid/tanh)
5. **Integer Quantization:** Integer Relu (v14+) useful for quantized models

---

## Testing Considerations

### Test Cases to Cover

1. **Basic Functionality:**
   - Positive values (unchanged)
   - Negative values (clamped to 0)
   - Zero values (remain zero)
   - Mixed positive/negative/zero

2. **Type Coverage:**
   - All supported float types (double, float, float16, bfloat16)
   - All supported integer types (v14+: int8, int16, int32, int64)

3. **Shape Coverage:**
   - Scalar tensors
   - 1D, 2D, 3D, and higher-dimensional tensors
   - Various tensor sizes

4. **Edge Cases:**
   - Very large positive values
   - Very large negative values
   - NaN values (should propagate or be handled)
   - Infinity values (should be handled appropriately)

5. **Version-Specific:**
   - v1: Test with/without `consumed_inputs` attribute (should be ignored)
   - v14: Verify function body implementation matches direct implementation

---

## References

- [ONNX Relu Operator Documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu)
- [ONNX Function Definition](https://github.com/onnx/onnx/blob/main/docs/Operators.md#functions)
- Related Operators: Clip, LeakyRelu, PRelu, ThresholdedRelu

---

**Document Version:** 1.0  
**Last Updated:** Based on ONNX opset versions 1, 6, 13, 14

