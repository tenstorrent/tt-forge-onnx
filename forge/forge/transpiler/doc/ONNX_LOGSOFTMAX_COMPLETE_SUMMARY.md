# ONNX LogSoftmax Operator - Complete Summary

## Overview

The **LogSoftmax** operator computes the logarithm of softmax values for the given input tensor. It applies the log-softmax function along a specified axis, producing output values that are the log of the softmax probabilities.

The log-softmax function is defined as:

```
LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
```

This operator takes one input tensor and produces one output tensor with the same shape, containing the log-softmax values of the corresponding input.

## Version History

| Version | Since | Shape Inference | Function | Key Changes |
|---------|-------|----------------|----------|-------------|
| 1 | 1 | ✅ | ❌ | Initial version, axis defaults to 1, coerces to 2D |
| 11 | 11 | ✅ | ❌ | Same as v1, improved documentation |
| 13 | 13 | ✅ | ✅ | Converted to function, axis defaults to -1, added bfloat16 support |

---

## LogSoftmax - Version 13

**Since Version:** 13  
**Shape Inference:** ✅ True  
**Function:** ✅ True  
**Support Level:** COMMON

### Summary

The operator computes the log of softmax values for the given input:

```
LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
```

The "axis" attribute indicates the dimension along which LogSoftmax will be performed. The output tensor has the same shape and contains the LogSoftmax values of the corresponding input.

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| axis | INT | No | -1 | Describes the dimension LogSoftmax will be performed on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input). |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| input | T | The input tensor of rank >= axis. |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| output | T | The output values with the same shape as the input tensor. |

### Type Constraints

**T** in ( tensor(bfloat16), tensor(double), tensor(float), tensor(float16) ):  
Constrain input and output types to float tensors.

### Function Body

```
< domain: "", opset_import: ["" : 18]>
LogSoftmax (X) => (Y) {
    Y = Log (Softmax (X))
}
```

### Changes from Previous Versions

- **v13 vs v11**: 
  - Converted to function body (Log(Softmax(X)))
  - Default axis changed from 1 to -1
  - Added bfloat16 type support

---

## LogSoftmax - Version 11

**Since Version:** 11  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

The operator computes the logsoftmax (log of softmax) values for each layer in the batch of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be coerced into one. For an arbitrary n-dimensional tensor input ∈ [a_0, a_1, …, a_{k-1}, a_k, …, a_{n-1}] and k is the axis provided, then input will be coerced into a 2-dimensional tensor with dimensions [a_0 * … * a_{k-1}, a_k * … * a_{n-1}]. For the default case where axis=1, this means the input tensor will be coerced into a 2D tensor of dimensions [a_0, a_1 * … * a_{n-1}], where a_0 is often the batch size. In this situation, we must have a_0 = N and a_1 * … * a_{n-1} = D. Each of these dimensions must be matched correctly, or else the operator will throw errors. The output tensor has the same shape and contains the logsoftmax values of the corresponding input.

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| axis | INT | No | 1 | Describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely describes the batch_size. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input). |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| input | T | The input tensor that's coerced into a 2D matrix of size (NxD) as described above. |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| output | T | The output values with the same shape as input tensor (the original size without coercion). |

### Type Constraints

**T** in ( tensor(double), tensor(float), tensor(float16) ):  
Constrain input and output types to float tensors.

### Changes from Previous Versions

- **v11 vs v1**: Improved documentation, no functional changes

---

## LogSoftmax - Version 1

**Since Version:** 1  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

The operator computes the logsoftmax (log of softmax) values for each layer in the batch of the given input. The input is a 2-D tensor (Tensor) of size (batch_size x input_feature_dimensions). The output tensor has the same shape and contains the logsoftmax values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be coerced into one. For an arbitrary n-dimensional tensor input ∈ [a_0, a_1, …, a_{k-1}, a_k, …, a_{n-1}] and k is the axis provided, then input will be coerced into a 2-dimensional tensor with dimensions [a_0 * … * a_{k-1}, a_k * … * a_{n-1}]. For the default case where axis=1, this means the input tensor will be coerced into a 2D tensor of dimensions [a_0, a_1 * … * a_{n-1}], where a_0 is often the batch size. In this situation, we must have a_0 = N and a_1 * … * a_{n-1} = D. Each of these dimensions must be matched correctly, or else the operator will throw errors.

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| axis | INT | No | 1 | Describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely describes the batch_size |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| input | T | The input tensor that's coerced into a 2D matrix of size (NxD) as described above. |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| output | T | The output values with the same shape as input tensor (the original size without coercion). |

### Type Constraints

**T** in ( tensor(double), tensor(float), tensor(float16) ):  
Constrain input and output types to float tensors.

---

## Behavioral Notes

1. **Numerical Stability**: LogSoftmax is more numerically stable than computing Log(Softmax(x)) separately, as it uses the log-sum-exp trick internally.

2. **Output Range**: The output values are in the range (-∞, 0], as they are logarithms of probabilities (which are in [0, 1]).

3. **Axis Coercion (v1-v12)**: For opset versions 1-12, the input is coerced to 2D along the specified axis. This behavior is simplified in v13+.

4. **Function Body (v13+)**: In opset 13+, LogSoftmax is defined as a function body: `Log(Softmax(X))`, making the relationship explicit.

## Implementation Considerations

1. **Default Axis**: 
   - v1-v12: Default axis is 1
   - v13+: Default axis is -1

2. **Type Support**:
   - v1-v12: Supports double, float, float16
   - v13+: Additionally supports bfloat16

3. **Function Body**: v13+ uses a function body, but the converter should still create a LogSoftmaxNode directly for efficiency.

## Differences from Similar Operators

- **LogSoftmax vs Softmax**: LogSoftmax computes log(softmax(x)), while Softmax computes softmax(x). LogSoftmax is more numerically stable for computing log probabilities.

- **LogSoftmax vs Log**: LogSoftmax is not simply Log(Softmax(x)) in terms of numerical stability - it uses the log-sum-exp trick.

## Common Use Cases

1. **Classification Losses**: Used in cross-entropy loss computation for classification tasks.
2. **Probability Logs**: When log probabilities are needed instead of probabilities.
3. **Numerical Stability**: Preferred over Log(Softmax(x)) for better numerical stability.

## Testing Considerations

1. Test with different axis values (0, 1, -1, -2, etc.)
2. Test with various input shapes (1D, 2D, 3D, 4D, 5D)
3. Test with different dtypes (float, double, float16, bfloat16 for v13+)
4. Test edge cases: all zeros, all equal values, extreme values
5. Verify output range: all values should be <= 0
6. Verify numerical stability with large input values
7. Test default axis behavior for different opset versions

