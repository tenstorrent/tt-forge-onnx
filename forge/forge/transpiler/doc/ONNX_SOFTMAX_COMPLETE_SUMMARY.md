# ONNX Softmax Operator - Complete Summary

## Overview

The **Softmax** operator computes the normalized exponential values for the given input tensor. It applies the softmax function along a specified axis, producing output values that sum to 1 along that axis (when `keepdims=1`).

The softmax function is defined as:

```
Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
```

This operator takes one input tensor and produces one output tensor with the same shape, containing the softmax values of the corresponding input.

## Version History

| Version | Since | Shape Inference | Function | Key Changes |
|---------|-------|----------------|----------|-------------|
| 1 | 1 | ✅ | ❌ | Initial version, axis defaults to 1, coerces to 2D |
| 11 | 11 | ✅ | ❌ | Same as v1, improved documentation |
| 13 | 13 | ✅ | ✅ | Converted to function, axis defaults to -1, added bfloat16 support |

---

## Softmax - Version 1

**Since Version:** 1  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

The operator computes the softmax (normalized exponential) values for each layer in the batch of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be coerced into one. For an arbitrary n-dimensional tensor input ∈ [a_0, a_1, …, a_{k-1}, a_k, …, a_{n-1}] and k is the axis provided, then input will be coerced into a 2-dimensional tensor with dimensions [a_0 * … * a_{k-1}, a_k * … * a_{n-1}]. For the default case where axis=1, this means the input tensor will be coerced into a 2D tensor of dimensions [a_0, a_1 * … * a_{n-1}], where a_0 is often the batch size. In this situation, we must have a_0 = N and a_1 * … * a_{n-1} = D. Each of these dimensions must be matched correctly, or else the operator will throw errors. The output tensor has the same shape and contains the softmax values of the corresponding input.

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely describes the batch_size |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `input` | T | The input tensor that's coerced into a 2D matrix of size (NxD) as described above |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output values with the same shape as input tensor (the original size without coercion) |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Shape Inference:** Supported, allowing automatic shape propagation
- **2D Coercion:** The input is conceptually coerced to 2D for computation, but the output maintains the original shape
- **Default Axis:** Defaults to axis=1, assuming the 0th dimension is the batch size
- **Type Support:** Limited to floating-point types only (double, float, float16)

---

## Softmax - Version 11

**Since Version:** 11  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

The operator computes the softmax (normalized exponential) values for each layer in the batch of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be coerced into one. For an arbitrary n-dimensional tensor input ∈ [a_0, a_1, …, a_{k-1}, a_k, …, a_{n-1}] and k is the axis provided, then input will be coerced into a 2-dimensional tensor with dimensions [a_0 * … * a_{k-1}, a_k * … * a_{n-1}]. For the default case where axis=1, this means the input tensor will be coerced into a 2D tensor of dimensions [a_0, a_1 * … * a_{n-1}], where a_0 is often the batch size. In this situation, we must have a_0 = N and a_1 * … * a_{n-1} = D. Each of these dimensions must be matched correctly, or else the operator will throw errors. The output tensor has the same shape and contains the softmax values of the corresponding input.

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `1` | Describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely describes the batch_size. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input). |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `input` | T | The input tensor that's coerced into a 2D matrix of size (NxD) as described above |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output values with the same shape as input tensor (the original size without coercion) |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Changes from v1

1. 📝 **Improved Documentation:** Added clarification about negative axis values and accepted range
2. 📊 **Type Support:** Unchanged (still 3 float types)
3. 🔄 **Behavior:** Unchanged from v1

### Notes

- **Negative Axis:** Supports negative axis values, counting from the back (e.g., -1 means the last dimension)
- **Axis Range:** Accepted range is [-r, r-1] where r = rank(input)
- **2D Coercion:** Same conceptual 2D coercion as v1

---

## Softmax - Version 13

**Since Version:** 13  
**Shape Inference:** ✅ True  
**Function:** ✅ True  
**Support Level:** COMMON

### Summary

The operator computes the normalized exponential values for the given input:

```
Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
```

The "axis" attribute indicates the dimension along which Softmax will be performed. The output tensor has the same shape and contains the Softmax values of the corresponding input.

### Function Body

The operator is now defined as a function with the following body:

```onnx
Softmax (input, axis = -1) => (output)
{
   exp_input = Exp (input)
   sum_exp = ReduceSum <axis = axis, keepdims = 1> (exp_input)
   output = Div (exp_input, sum_exp)
}
```

**Function Implementation:**
1. Applies `Exp` to the input tensor elementwise
2. Computes `ReduceSum` along the specified axis with `keepdims=1` to maintain dimensions
3. Divides the exponentiated input by the sum to normalize

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `axis` | INT | ❌ | `-1` | Describes the dimension Softmax will be performed on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input). |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `input` | T | The input tensor of rank >= axis |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output values with the same shape as the input tensor |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 4 types

**Description:** Constrain input and output types to float tensors.

### Changes from v11

1. ✅ **Converted to Function:** The operator is now implemented as a function using `Exp`, `ReduceSum`, and `Div` operators
2. 🔄 **Default Axis Changed:** Changed from `axis=1` to `axis=-1` (last dimension)
3. ➕ **Added `bfloat16` Type:** Extended type support to include Brain Floating Point 16-bit format
4. 📊 **Type Count:** Increased from 3 to 4 types
5. 🎯 **Simplified Semantics:** Removed the 2D coercion concept; now operates directly on the specified axis
6. 📝 **Clearer Definition:** The function body provides an explicit mathematical definition

### Notes

- **Function Implementation:** The function body provides a clear, composable definition of Softmax using standard ONNX operators
- **Default Axis:** Defaults to `-1` (last dimension), which is more intuitive for many use cases
- **bfloat16 Support:** Added support for the bfloat16 floating-point format, commonly used in machine learning
- **No 2D Coercion:** The operator now works directly on the specified axis without the conceptual 2D coercion
- **Type Flexibility:** The function implementation maintains type consistency automatically

---

## Version Comparison Summary

### Type Support Evolution

| Version | Float Types | Total Types |
|---------|------------|-------------|
| v1 | double, float, float16 | 3 |
| v11 | double, float, float16 | 3 |
| v13 | bfloat16, double, float, float16 | 4 |

### Feature Evolution

| Feature | v1 | v11 | v13 |
|---------|----|-----|-----|
| Shape Inference | ✅ | ✅ | ✅ |
| Function Body | ❌ | ❌ | ✅ |
| Default Axis | 1 | 1 | -1 |
| bfloat16 Support | ❌ | ❌ | ✅ |
| 2D Coercion Concept | ✅ | ✅ | ❌ |
| Negative Axis Support | ❓ | ✅ | ✅ |

### Attribute Changes

| Attribute | v1 | v11 | v13 |
|-----------|----|-----|-----|
| `axis` (default) | 1 | 1 | -1 |
| `axis` (range) | Not specified | [-r, r-1] | [-r, r-1] |
| `axis` (negative support) | Not specified | ✅ | ✅ |

---

## Behavioral Notes

### Mathematical Definition

The Softmax operator applies the following function along the specified axis:

```
Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

Where the summation is performed over all elements along the specified axis.

**Properties:**
- All output values are in the range (0, 1)
- The sum of outputs along the specified axis equals 1 (when `keepdims=1`)
- The output preserves the input shape
- The operation is numerically stable when implemented correctly (using max subtraction trick)

### Examples

**Example 1: Basic 1D Softmax (axis=-1)**
```
Input:  [1.0, 2.0, 3.0]
Exp:    [2.718, 7.389, 20.086]
Sum:    30.193
Output: [0.090, 0.245, 0.665]  (sums to 1.0)
```

**Example 2: 2D Tensor with axis=0**
```
Input Shape:  (3, 4)
Input:        [[1.0, 2.0, 3.0, 4.0],
               [2.0, 3.0, 4.0, 5.0],
               [1.5, 2.5, 3.5, 4.5]]

Axis=0: Softmax applied along rows (dimension 0)
Output: Each column's values sum to 1.0
```

**Example 3: 2D Tensor with axis=1 (v1/v11 default)**
```
Input Shape:  (2, 3)
Input:        [[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]]

Axis=1: Softmax applied along columns (dimension 1)
Output: Each row's values sum to 1.0
        [[0.090, 0.245, 0.665],
         [0.090, 0.245, 0.665]]
```

**Example 4: 3D Tensor with axis=-1 (v13 default)**
```
Input Shape:  (2, 3, 4)
Input:        [[[1.0, 2.0, 3.0, 4.0], ...], ...]

Axis=-1: Softmax applied along the last dimension
Output: Each [:, :, :] slice's last dimension values sum to 1.0
```

### Edge Cases

1. **All Equal Values:** If all values along the axis are equal, each output will be 1/n (where n is the size along that axis)
2. **Large Values:** Very large input values can cause numerical overflow in exp(); implementations should use the max-subtraction trick for stability
3. **Negative Values:** Negative values are valid; softmax handles them correctly
4. **Zero Values:** Zero values are valid inputs
5. **Extreme Differences:** When values differ greatly, the softmax of the larger value approaches 1.0, and others approach 0.0

### Numerical Stability

For numerical stability, implementations often use the max-subtraction trick:

```
Softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

This prevents overflow by subtracting the maximum value before exponentiation, which doesn't change the result but improves numerical stability.

---

## Implementation Considerations

### Function-Based Implementation (v13+)

In version 13, Softmax is defined as a function using:
- `Exp`: Computes elementwise exponential
- `ReduceSum`: Sums along the specified axis with `keepdims=1`
- `Div`: Divides the exponentiated values by the sum

**Advantages:**
- Clear, composable definition
- Easier to understand and verify
- Can be optimized by runtime implementations
- Explicit mathematical relationship

### Direct Implementation

For performance-critical scenarios, implementations may choose to:
- Use optimized Softmax kernels instead of the function decomposition
- Implement numerical stability tricks (max-subtraction)
- Leverage hardware-specific instructions (e.g., SIMD operations)
- Apply fused operations for better performance

### Axis Handling

- **v1/v11:** Default axis is 1, assuming batch dimension at 0
- **v13:** Default axis is -1 (last dimension), more intuitive for many cases
- **Negative Axis:** All versions (v11+) support negative axis values counting from the back
- **Axis Range:** Must be in [-r, r-1] where r = rank(input)

### Shape Inference

All versions support shape inference:
- Output shape always matches input shape exactly
- No broadcasting or dimension changes occur
- The axis dimension is preserved (with `keepdims=1` in v13)

### Type Handling

- **Float Types:** Standard floating-point softmax operation
- **bfloat16 (v13+):** Supports Brain Floating Point 16-bit format
- **Numerical Precision:** Different float types may yield slightly different results due to precision

---

## Differences from Similar Operators

### Softmax vs. LogSoftmax

| Aspect | Softmax | LogSoftmax |
|--------|---------|------------|
| Output Range | (0, 1) | (-∞, 0) |
| Output Sum | Sums to 1 | Log of sum to 1 |
| Use Case | Probabilities | Log probabilities (more stable) |
| Relationship | LogSoftmax(x) = log(Softmax(x)) | Softmax(x) = exp(LogSoftmax(x)) |

### Softmax vs. Sigmoid

| Aspect | Softmax | Sigmoid |
|--------|---------|---------|
| Input | Multi-element vector | Single element |
| Output | Normalized probabilities (sum to 1) | Individual probability (0 to 1) |
| Axis | Operates along specified axis | Elementwise operation |
| Use Case | Multi-class classification | Binary classification or elementwise |

**Relationship:** Sigmoid is essentially a 2-element Softmax.

---

## Common Use Cases

1. **Multi-class Classification:** Convert logits to probability distributions over classes
2. **Attention Mechanisms:** Normalize attention scores in transformer models
3. **Probability Normalization:** Convert arbitrary scores to valid probability distributions
4. **Neural Network Output Layers:** Final layer in classification networks
5. **Policy Networks:** In reinforcement learning, convert action scores to action probabilities

---

## Testing Considerations

### Test Cases to Cover

1. **Basic Functionality:**
   - 1D tensors with various axis values
   - 2D tensors with axis=0, axis=1, axis=-1
   - Higher-dimensional tensors
   - Verify output sums to 1 along the specified axis

2. **Type Coverage:**
   - All supported float types (double, float, float16, bfloat16)
   - Test numerical precision differences

3. **Axis Coverage:**
   - Positive axis values (0, 1, 2, ...)
   - Negative axis values (-1, -2, ...)
   - Default axis behavior (1 for v1/v11, -1 for v13)

4. **Edge Cases:**
   - All equal values along axis
   - Very large values (test numerical stability)
   - Very small values
   - Mixed positive and negative values
   - Zero values

5. **Shape Coverage:**
   - Scalar-like tensors (1D with single element)
   - Various tensor sizes and dimensions
   - Large tensors (performance testing)

6. **Version-Specific:**
   - v1/v11: Test 2D coercion behavior
   - v13: Verify function body implementation matches direct implementation
   - v13: Test default axis=-1 behavior

7. **Numerical Stability:**
   - Test with extreme values to ensure no overflow
   - Verify output sums are close to 1.0 (within numerical precision)
   - Test max-subtraction trick if implemented

---

## References

- [ONNX Softmax Operator Documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax)
- [ONNX Function Definition](https://github.com/onnx/onnx/blob/main/docs/Operators.md#functions)
- Related Operators: LogSoftmax, Sigmoid, Exp, ReduceSum

---

**Document Version:** 1.0  
**Last Updated:** Based on ONNX opset versions 1, 11, 13

