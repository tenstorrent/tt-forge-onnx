# ONNX Dropout Operator - Complete Summary

## Overview

The **Dropout** operator randomly sets some elements of the input tensor to zero during training to prevent overfitting. It is a regularization technique commonly used in neural networks. The operator supports both training and inference modes, with different behaviors in each mode.

The operator has evolved significantly across ONNX versions, transitioning from attribute-based control (`is_test`, `ratio`) to input-based control (`ratio` input, `training_mode` input) for more flexibility and dynamic behavior.

## Version History

| Version | Since | Shape Inference | Function | Key Changes |
|---------|-------|----------------|----------|-------------|
| 1 | 1 | ❌ | ❌ | Initial version, `is_test` and `ratio` attributes, `consumed_inputs` attribute |
| 6 | 6 | ✅ | ❌ | Removed `consumed_inputs`, shape inference enabled |
| 7 | 7 | ✅ | ❌ | Removed `is_test` attribute, uses test mode detection |
| 10 | 10 | ✅ | ❌ | Similar to v7, improved documentation |
| 12 | 12 | ✅ | ❌ | **Major change:** `ratio` and `training_mode` become optional inputs instead of attributes |
| 13 | 13 | ✅ | ❌ | Extended type support (bfloat16) |
| 22 | 22 | ✅ | ❌ | Extended type support (float8 types) |

---

## Dropout - Version 1

**Since Version:** 1  
**Shape Inference:** ❌ False  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Dropout takes one input data tensor and produces two tensor outputs: output (tensor) and mask (tensor). Depending on whether it is in test mode or not, the output Y will either be a random dropout, or a simple copy of the input. Note that the implementation does scaling in the training phase, so during testing nothing needs to be done.

**Example:** Basic dropout with 50% ratio in training mode.

```
Input data: [1, 3, 32, 32]  (batch=1, channels=3, height=32, width=32)
Attributes: ratio=0.5, is_test=0 (training mode)
Output: [1, 3, 32, 32]  (same shape, some elements zeroed)
Mask: [1, 3, 32, 32]  (boolean mask indicating dropped elements)
```

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `consumed_inputs` | INTS | ❌ | - | Legacy optimization attribute (deprecated) |
| `is_test` | INT | ❌ | `0` | If nonzero, run dropout in test mode where the output is simply Y = X |
| `ratio` | FLOAT | ❌ | `0.5` | The ratio of random dropout (probability of an element being zeroed) |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | The input data as Tensor |

**Input Count:** 1 input

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output tensor |
| `mask` | T (optional) | The output mask. If `is_test` is nonzero, this output is not filled |

**Output Count:** Between 1 and 2 outputs

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Training Mode (`is_test=0`):**
  - Randomly sets elements to zero based on `ratio`
  - Scales remaining elements by `1 / (1 - ratio)` to maintain expected value
  - Output formula: `output = scale * data * mask`, where `scale = 1 / (1 - ratio)`
  - Mask is generated and returned

- **Inference Mode (`is_test=1`):**
  - Output is a simple copy of input: `output = data`
  - Mask output is not filled (empty/undefined)

- **Shape Inference:** Not supported in v1
- **Legacy Attribute:** `consumed_inputs` is a legacy optimization attribute and should not be used in new models

### Example: Training Mode (v1)

```python
# ONNX Model (v1)
# Input data: [1, 3, 32, 32]
# Attributes: ratio=0.5, is_test=0
# Output: [1, 3, 32, 32] (scaled and masked)
# Mask: [1, 3, 32, 32] (boolean mask)

# PyTorch Equivalent
import torch
import torch.nn as nn

# During training
dropout = nn.Dropout(p=0.5)  # p = ratio
output = dropout(input_data)  # Randomly zeros 50% of elements, scales by 2.0
```

### Example: Inference Mode (v1)

```python
# ONNX Model (v1)
# Input data: [1, 3, 32, 32]
# Attributes: ratio=0.5, is_test=1
# Output: [1, 3, 32, 32] (copy of input)
# Mask: Not filled

# PyTorch Equivalent
dropout = nn.Dropout(p=0.5)
dropout.eval()  # Set to evaluation mode
output = dropout(input_data)  # Simple copy, no dropout applied
```

---

## Dropout - Version 6

**Since Version:** 6  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Dropout takes one input data tensor and produces two tensor outputs: output (tensor) and mask (tensor). Depending on whether it is in test mode or not, the output Y will either be a random dropout, or a simple copy of the input. Note that the implementation does scaling in the training phase, so during testing nothing needs to be done.

**Key Improvements:**
- ✅ Shape inference enabled
- ✅ Removed deprecated `consumed_inputs` attribute

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `is_test` | INT | ❌ | `0` | If nonzero, run dropout in test mode where the output is simply Y = X |
| `ratio` | FLOAT | ❌ | `0.5` | The ratio of random dropout (probability of an element being zeroed) |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | The input data as Tensor |

**Input Count:** 1 input

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output tensor |
| `mask` | T (optional) | The output mask. If `is_test` is nonzero, this output is not filled |

**Output Count:** Between 1 and 2 outputs

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Changes from v1

1. ✅ **Shape Inference:** Enabled automatic shape propagation
2. ✅ **Removed Legacy Attribute:** `consumed_inputs` attribute removed (was deprecated)

### Notes

- **Shape Inference:** Now supported, allowing automatic shape propagation through the graph
- **Behavior:** Functionally identical to v1, but with shape inference support
- **Training/Inference:** Same behavior as v1 using `is_test` attribute

### Example: Training Mode with Shape Inference (v6)

```python
# ONNX Model (v6)
# Input data: [1, 3, 32, 32]
# Attributes: ratio=0.5, is_test=0
# Output: [1, 3, 32, 32] (shape inferred automatically)
# Mask: [1, 3, 32, 32] (shape inferred automatically)

# PyTorch Equivalent
dropout = nn.Dropout(p=0.5)
output = dropout(input_data)  # Shape: [1, 3, 32, 32]
```

---

## Dropout - Version 7

**Since Version:** 7  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Dropout takes one input floating tensor and produces two tensor outputs: output (floating tensor) and mask (Tensor<bool>). Depending on whether it is in test mode or not, the output Y will either be a random dropout, or a simple copy of the input. Note that the implementation does scaling in the training phase, so during testing nothing needs to be done.

**Key Improvements:**
- ✅ Removed `is_test` attribute
- ✅ Uses test mode detection (typically from graph-level training flag)
- ✅ Mask type changed to `Tensor<bool>` (boolean) instead of same type as input

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ratio` | FLOAT | ❌ | `0.5` | The ratio of random dropout |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | The input data as Tensor |

**Input Count:** 1 input

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output tensor |
| `mask` | T1 (optional) | The output mask (boolean tensor) |

**Output Count:** Between 1 and 2 outputs

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

**T1** in:
- `tensor(bool)`

**Total:** 1 type

**Description:** Constrain output mask types to boolean tensors.

### Changes from v6

1. ✅ **Removed `is_test` Attribute:** No longer uses `is_test` attribute to control training/inference mode
2. ✅ **Test Mode Detection:** Relies on graph-level training flag or external context to determine mode
3. ✅ **Mask Type Change:** Mask output is now `Tensor<bool>` instead of same type as input

### Notes

- **Training Mode Detection:** The operator determines training vs inference mode from the graph context or external flags, not from an attribute
- **Mask Type:** Mask is now a boolean tensor (`Tensor<bool>`) indicating which elements were dropped (true = kept, false = dropped)
- **Behavior:** Training mode behavior is identical to v6, but inference mode is determined differently

### Example: Training Mode (v7)

```python
# ONNX Model (v7)
# Input data: [1, 3, 32, 32]
# Attributes: ratio=0.5
# Graph context: training=True
# Output: [1, 3, 32, 32] (scaled and masked)
# Mask: [1, 3, 32, 32] (boolean, true=kept, false=dropped)

# PyTorch Equivalent
dropout = nn.Dropout(p=0.5)
dropout.train()  # Set to training mode (graph-level flag)
output = dropout(input_data)
```

### Example: Inference Mode (v7)

```python
# ONNX Model (v7)
# Input data: [1, 3, 32, 32]
# Attributes: ratio=0.5
# Graph context: training=False
# Output: [1, 3, 32, 32] (copy of input)
# Mask: [1, 3, 32, 32] (all true, nothing dropped)

# PyTorch Equivalent
dropout = nn.Dropout(p=0.5)
dropout.eval()  # Set to evaluation mode (graph-level flag)
output = dropout(input_data)  # Simple copy
```

---

## Dropout - Version 10

**Since Version:** 10  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Dropout takes one input floating tensor and produces two tensor outputs: output (floating tensor) and mask (Tensor<bool>). Depending on whether it is in test mode or not, the output Y will either be a random dropout, or a simple copy of the input. Note that the implementation does scaling in the training phase, so during testing nothing needs to be done.

**Key Improvements:**
- ✅ Improved documentation
- ✅ Functionally similar to v7

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ratio` | FLOAT | ❌ | `0.5` | The ratio of random dropout |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | The input data as Tensor |

**Input Count:** 1 input

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output tensor |
| `mask` | T1 (optional) | The output mask (boolean tensor) |

**Output Count:** Between 1 and 2 outputs

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

**T1** in:
- `tensor(bool)`

**Total:** 1 type

**Description:** Constrain output mask types to boolean tensors.

### Changes from v7

1. 📝 **Documentation Update:** Improved documentation clarity
2. 🔄 **Functionality:** Functionally identical to v7

### Notes

- **Behavior:** Identical to v7
- **Training/Inference:** Uses graph-level training flag, not attributes
- **Mask Type:** Boolean tensor as in v7

---

## Dropout - Version 12

**Since Version:** 12  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs, output (floating-point tensor) and mask (optional Tensor<bool>). If training_mode is true then the output Y will be a random dropout; Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode, the user can simply not pass training_mode input or set it to false.

**Key Improvements:**
- ✅ **Major Change:** `ratio` becomes an optional input instead of attribute
- ✅ **Major Change:** `training_mode` becomes an optional input instead of relying on graph context
- ✅ More flexible and dynamic control of dropout behavior

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `seed` | INT | ❌ | - | Seed to the random generator, if not specified we will auto generate one |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | The input data as Tensor |
| `ratio` | T1 (optional) | The ratio of random dropout, with value in [0, 1). If this input was not set, or if it was set to 0, the output would be a simple copy of the input. If it's non-zero, output will be a random dropout of the scaled input, which is typically the case during training. It is an optional value, if not specified it will default to 0.5. |
| `training_mode` | T2 (optional) | If set to true then it indicates dropout is being used for training. It is an optional value hence unless specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode where nothing will be dropped from the input data and if mask is requested as output it will contain all ones. |

**Input Count:** Between 1 and 3 inputs

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output tensor |
| `mask` | T2 (optional) | The output mask (boolean tensor) |

**Output Count:** Between 1 and 2 outputs

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

**T1** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input 'ratio' types to float tensors.

**T2** in:
- `tensor(bool)`

**Total:** 1 type

**Description:** Constrain output 'mask' types and input 'training_mode' types to boolean tensors.

### Changes from v10

1. ✅ **`ratio` as Input:** `ratio` is now an optional input tensor instead of an attribute
   - Allows dynamic ratio values that can change per inference
   - Default value: 0.5 if not provided
   - If set to 0, output is a simple copy of input

2. ✅ **`training_mode` as Input:** `training_mode` is now an optional input tensor instead of graph context
   - Allows explicit control of training/inference mode per call
   - Default value: false (inference mode) if not provided
   - If false, ratio is ignored and operation mimics inference mode

3. ✅ **`seed` Attribute:** Added optional `seed` attribute for random number generator
   - Allows reproducible dropout patterns
   - If not specified, seed is auto-generated

4. ✅ **More Flexible:** Can now have different dropout ratios and training modes for different calls in the same graph

### Notes

- **Output Formula:** `output = scale * data * mask`, where `scale = 1 / (1 - ratio)`
- **Training Mode (`training_mode=True`):**
  - Randomly sets elements to zero based on `ratio`
  - Scales remaining elements by `1 / (1 - ratio)`
  - Mask indicates dropped elements (false = dropped, true = kept)

- **Inference Mode (`training_mode=False` or not provided):**
  - Output is a simple copy of input: `output = data`
  - Mask contains all ones (all elements kept)
  - `ratio` is ignored

- **Ratio Input:**
  - If not provided: defaults to 0.5
  - If set to 0: output is a simple copy (no dropout)
  - Must be in range [0, 1)

- **Dynamic Behavior:** Both `ratio` and `training_mode` can be different for each inference call

### Example: Training Mode with Dynamic Ratio (v12)

```python
# ONNX Model (v12)
# Input data: [1, 3, 32, 32]
# Input ratio: 0.3 (scalar tensor)
# Input training_mode: True (scalar tensor)
# Attributes: seed=42
# Output: [1, 3, 32, 32] (scaled and masked, 30% dropout)
# Mask: [1, 3, 32, 32] (boolean mask)

# PyTorch Equivalent
import torch

ratio_tensor = torch.tensor(0.3)
training_tensor = torch.tensor(True)

# Manual implementation
torch.manual_seed(42)
mask = (torch.rand_like(input_data) > ratio_tensor).float()
scale = 1.0 / (1.0 - ratio_tensor)
output = input_data * mask * scale
```

### Example: Inference Mode (v12)

```python
# ONNX Model (v12)
# Input data: [1, 3, 32, 32]
# Input training_mode: False (or not provided)
# Output: [1, 3, 32, 32] (copy of input)
# Mask: [1, 3, 32, 32] (all true)

# PyTorch Equivalent
output = input_data  # Simple copy
mask = torch.ones_like(input_data, dtype=torch.bool)
```

### Example: Ratio = 0 (v12)

```python
# ONNX Model (v12)
# Input data: [1, 3, 32, 32]
# Input ratio: 0.0 (scalar tensor)
# Input training_mode: True
# Output: [1, 3, 32, 32] (copy of input, no dropout)
# Mask: [1, 3, 32, 32] (all true)

# PyTorch Equivalent
output = input_data  # No dropout when ratio=0
mask = torch.ones_like(input_data, dtype=torch.bool)
```

---

## Dropout - Version 13

**Since Version:** 13  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs, output (floating-point tensor) and mask (optional Tensor<bool>). If training_mode is true then the output Y will be a random dropout; Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode, the user can simply not pass training_mode input or set it to false.

**Key Improvements:**
- ✅ Extended type support: Added `bfloat16` support

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `seed` | INT | ❌ | - | Seed to the random generator, if not specified we will auto generate one |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | The input data as Tensor |
| `ratio` | T1 (optional) | The ratio of random dropout, with value in [0, 1). If this input was not set, or if it was set to 0, the output would be a simple copy of the input. If it's non-zero, output will be a random dropout of the scaled input, which is typically the case during training. It is an optional value, if not specified it will default to 0.5. |
| `training_mode` | T2 (optional) | If set to true then it indicates dropout is being used for training. It is an optional value hence unless specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode where nothing will be dropped from the input data and if mask is requested as output it will contain all ones. |

**Input Count:** Between 1 and 3 inputs

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output tensor |
| `mask` | T2 (optional) | The output mask (boolean tensor) |

**Output Count:** Between 1 and 2 outputs

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 4 types

**Description:** Constrain input and output types to float tensors.

**T1** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input 'ratio' types to float tensors.

**T2** in:
- `tensor(bool)`

**Total:** 1 type

**Description:** Constrain output 'mask' types and input 'training_mode' types to boolean tensors.

### Changes from v12

1. ✅ **Extended Type Support:** Added `bfloat16` support for input/output tensor T
   - `bfloat16` (brain floating point 16-bit) is now supported
   - Useful for training on certain hardware accelerators

### Notes

- **Behavior:** Functionally identical to v12
- **Type Support:** Now supports `bfloat16` in addition to `double`, `float`, and `float16`
- **Ratio Input:** Still does not support `bfloat16` (only `double`, `float`, `float16`)

### Example: Using bfloat16 (v13)

```python
# ONNX Model (v13)
# Input data: [1, 3, 32, 32] (bfloat16)
# Input ratio: 0.5 (float scalar)
# Input training_mode: True (boolean scalar)
# Output: [1, 3, 32, 32] (bfloat16, scaled and masked)
# Mask: [1, 3, 32, 32] (boolean)

# PyTorch Equivalent
import torch

input_data_bf16 = input_data.to(torch.bfloat16)
ratio_tensor = torch.tensor(0.5, dtype=torch.float32)
training_tensor = torch.tensor(True)

# Manual implementation with bfloat16
torch.manual_seed(42)
mask = (torch.rand_like(input_data_bf16) > ratio_tensor).float()
scale = 1.0 / (1.0 - ratio_tensor)
output = (input_data_bf16 * mask * scale).to(torch.bfloat16)
```

---

## Dropout - Version 22

**Since Version:** 22  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs, output (floating-point tensor) and mask (optional Tensor<bool>). If training_mode is true then the output Y will be a random dropout; Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode, the user can simply not pass training_mode input or set it to false.

**Key Improvements:**
- ✅ Extended type support: Added float8 types (`float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`)
- ✅ Extended type support for ratio input: Added `bfloat16` and float8 types

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `seed` | INT | ❌ | - | Seed to the random generator, if not specified we will auto generate one |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `data` | T | The input data as Tensor |
| `ratio` | T1 (optional) | The ratio of random dropout, with value in [0, 1). If set to 0, the output would be a simple copy of the input. If it's non-zero, output will be a random dropout of the scaled input, which is typically the case during training. It is an optional value, if not specified it will default to 0.5. |
| `training_mode` | T2 (optional) | If set to true then it indicates dropout is being used for training. It is an optional value hence unless specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode where nothing will be dropped from the input data and if mask is requested as output it will contain all ones. |

**Input Count:** Between 1 and 3 inputs

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | T | The output tensor |
| `mask` | T2 (optional) | The output mask (boolean tensor) |

**Output Count:** Between 1 and 2 outputs

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(float8e4m3fn)`
- `tensor(float8e4m3fnuz)`
- `tensor(float8e5m2)`
- `tensor(float8e5m2fnuz)`

**Total:** 8 types

**Description:** Constrain input and output types to float tensors.

**T1** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(float8e4m3fn)`
- `tensor(float8e4m3fnuz)`
- `tensor(float8e5m2)`
- `tensor(float8e5m2fnuz)`

**Total:** 8 types

**Description:** Constrain input 'ratio' types to float tensors.

**T2** in:
- `tensor(bool)`

**Total:** 1 type

**Description:** Constrain output 'mask' types and input 'training_mode' types to boolean tensors.

### Changes from v13

1. ✅ **Extended Type Support for T:** Added float8 types:
   - `float8e4m3fn` (8-bit float, 4 exponent, 3 mantissa, finite numbers)
   - `float8e4m3fnuz` (8-bit float, 4 exponent, 3 mantissa, finite numbers, unsigned zero)
   - `float8e5m2` (8-bit float, 5 exponent, 2 mantissa)
   - `float8e5m2fnuz` (8-bit float, 5 exponent, 2 mantissa, unsigned zero)

2. ✅ **Extended Type Support for T1 (ratio):** Added `bfloat16` and float8 types:
   - Now supports all the same types as T (input/output)
   - Allows ratio to be specified in lower precision formats

### Notes

- **Behavior:** Functionally identical to v13
- **Type Support:** Now supports float8 types for both input/output and ratio
- **Float8 Types:** Useful for quantization and memory-efficient training/inference
- **Ratio Input:** Can now be specified in float8 or bfloat16 formats

### Example: Using float8 (v22)

```python
# ONNX Model (v22)
# Input data: [1, 3, 32, 32] (float8e4m3fn)
# Input ratio: 0.5 (float8e4m3fn scalar)
# Input training_mode: True (boolean scalar)
# Output: [1, 3, 32, 32] (float8e4m3fn, scaled and masked)
# Mask: [1, 3, 32, 32] (boolean)

# Note: PyTorch may not have direct float8 support, this is conceptual
# The ONNX model would handle the float8 conversion internally
```

---

## Summary of Changes Across Versions

### Type Support Evolution

| Version | New Types Added (T) | Total Types (T) | New Types Added (T1) | Total Types (T1) | Description |
|---------|---------------------|----------------|---------------------|------------------|-------------|
| **v1** | Base float types | 3 | N/A | N/A | `double`, `float`, `float16` |
| **v6** | - | 3 | N/A | N/A | Same as v1, shape inference enabled |
| **v7** | - | 3 | N/A | N/A | Same as v6, removed `is_test` |
| **v10** | - | 3 | N/A | N/A | Same as v7, improved docs |
| **v12** | - | 3 | Ratio as input | 3 | `ratio` becomes input, `training_mode` becomes input |
| **v13** | `bfloat16` | 4 | - | 3 | Added `bfloat16` support |
| **v22** | Float8 types | 8 | `bfloat16` + Float8 | 8 | Added float8 types for both T and T1 |

### Attribute Evolution

| Attribute | v1 | v6 | v7 | v10 | v12 | v13 | v22 | Notes |
|-----------|----|----|----|-----|-----|-----|-----|-------|
| `consumed_inputs` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Removed in v6 |
| `is_test` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | Removed in v7, replaced by graph context |
| `ratio` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Becomes input in v12 |
| `seed` | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | Added in v12 |

### Input Evolution

| Input | v1-v10 | v12+ | Notes |
|-------|--------|------|-------|
| `data` | ✅ Required | ✅ Required | Always required |
| `ratio` | ❌ (attribute) | ✅ Optional | Becomes optional input in v12 |
| `training_mode` | ❌ (graph context) | ✅ Optional | Becomes optional input in v12 |

### Output Evolution

| Output | v1-v6 | v7+ | Notes |
|--------|-------|-----|-------|
| `output` | ✅ T | ✅ T | Always same type as input |
| `mask` | ✅ T | ✅ T2 (bool) | Type changed from T to bool in v7 |

### Training/Inference Mode Control Evolution

| Version | Control Method | Default Mode | Notes |
|---------|----------------|--------------|-------|
| **v1-v6** | `is_test` attribute | Training (0) | Attribute-based control |
| **v7-v10** | Graph context | Inference | Relies on graph-level training flag |
| **v12+** | `training_mode` input | Inference (false) | Input-based control, more flexible |

### Key Behavioral Differences

1. **v1 vs v6:**
   - v6 adds shape inference
   - v6 removes `consumed_inputs` attribute

2. **v6 vs v7:**
   - v7 removes `is_test` attribute
   - v7 changes mask type from T to bool
   - v7 uses graph context for training mode

3. **v7 vs v12:**
   - v12 makes `ratio` an optional input (was attribute)
   - v12 makes `training_mode` an optional input (was graph context)
   - v12 adds `seed` attribute
   - v12 allows dynamic ratio and training mode per call

4. **v12 vs v13:**
   - v13 adds `bfloat16` support for input/output

5. **v13 vs v22:**
   - v22 adds float8 types for input/output
   - v22 adds `bfloat16` and float8 types for ratio input

---

## PyTorch Mapping

### Mapping Table

| ONNX | PyTorch | Notes |
|------|---------|-------|
| `data` | `input` | Input tensor |
| `ratio` (v1-v10 attribute, v12+ input) | `p` parameter | Dropout probability |
| `is_test` (v1-v6) | `training` mode | `is_test=0` → `training=True`, `is_test=1` → `training=False` |
| `training_mode` (v12+ input) | `training` mode | `training_mode=True` → `training=True`, `training_mode=False` → `training=False` |
| `seed` (v12+) | `torch.manual_seed()` | Random seed |
| `output` | `output` | Output tensor |
| `mask` | N/A | PyTorch doesn't expose mask directly |

### Key Differences

1. **Training Mode Control:**
   - **ONNX v1-v6:** Uses `is_test` attribute
   - **ONNX v7-v10:** Uses graph context
   - **ONNX v12+:** Uses `training_mode` input
   - **PyTorch:** Uses `.train()` / `.eval()` methods or `training` parameter

2. **Ratio/Rate:**
   - **ONNX:** `ratio` (probability of being zeroed)
   - **PyTorch:** `p` (probability of being zeroed)
   - **Same meaning:** Both represent probability of dropout

3. **Mask Output:**
   - **ONNX:** Can optionally return mask tensor
   - **PyTorch:** Does not expose mask in `nn.Dropout`, but can be obtained with manual implementation

4. **Scaling:**
   - **ONNX:** Always scales by `1 / (1 - ratio)` during training
   - **PyTorch:** `nn.Dropout` also scales by `1 / (1 - p)` during training

5. **Dynamic Ratio:**
   - **ONNX v12+:** Supports dynamic ratio via input tensor
   - **PyTorch:** `nn.Dropout` has fixed `p`, but `F.dropout()` can use dynamic values

---

## Examples

### Example 1: Basic Dropout Training (v1)

```python
# ONNX Model (v1)
# Input data: [1, 3, 32, 32]
# Attributes: ratio=0.5, is_test=0
# Output: [1, 3, 32, 32]
# Mask: [1, 3, 32, 32]

# PyTorch Equivalent
import torch.nn as nn

dropout = nn.Dropout(p=0.5)
dropout.train()  # is_test=0 means training mode
output = dropout(input_data)
```

### Example 2: Basic Dropout Inference (v1)

```python
# ONNX Model (v1)
# Input data: [1, 3, 32, 32]
# Attributes: ratio=0.5, is_test=1
# Output: [1, 3, 32, 32] (copy)
# Mask: Not filled

# PyTorch Equivalent
dropout = nn.Dropout(p=0.5)
dropout.eval()  # is_test=1 means inference mode
output = dropout(input_data)  # Simple copy
```

### Example 3: Dropout with Graph Context (v7)

```python
# ONNX Model (v7)
# Input data: [1, 3, 32, 32]
# Attributes: ratio=0.5
# Graph context: training=True
# Output: [1, 3, 32, 32]
# Mask: [1, 3, 32, 32] (boolean)

# PyTorch Equivalent
dropout = nn.Dropout(p=0.5)
dropout.train()  # Graph-level training flag
output = dropout(input_data)
```

### Example 4: Dynamic Ratio Dropout (v12)

```python
# ONNX Model (v12)
# Input data: [1, 3, 32, 32]
# Input ratio: 0.3 (scalar tensor, can change per call)
# Input training_mode: True (scalar tensor)
# Attributes: seed=42
# Output: [1, 3, 32, 32]
# Mask: [1, 3, 32, 32] (boolean)

# PyTorch Equivalent
import torch
import torch.nn.functional as F

ratio_tensor = torch.tensor(0.3)
torch.manual_seed(42)
output = F.dropout(input_data, p=ratio_tensor.item(), training=True)
```

### Example 5: Dropout with bfloat16 (v13)

```python
# ONNX Model (v13)
# Input data: [1, 3, 32, 32] (bfloat16)
# Input ratio: 0.5 (float scalar)
# Input training_mode: True (boolean scalar)
# Output: [1, 3, 32, 32] (bfloat16)
# Mask: [1, 3, 32, 32] (boolean)

# PyTorch Equivalent
import torch
import torch.nn as nn

input_bf16 = input_data.to(torch.bfloat16)
dropout = nn.Dropout(p=0.5)
dropout.train()
output = dropout(input_bf16)
```

### Example 6: Different Ratios Per Call (v12+)

```python
# ONNX Model (v12+)
# First call:
#   Input ratio: 0.5
#   Output: 50% dropout
# Second call:
#   Input ratio: 0.2
#   Output: 20% dropout

# PyTorch Equivalent
import torch.nn.functional as F

# First call
output1 = F.dropout(input_data, p=0.5, training=True)

# Second call with different ratio
output2 = F.dropout(input_data, p=0.2, training=True)
```

### Example 7: Ratio = 0 (No Dropout) (v12+)

```python
# ONNX Model (v12+)
# Input data: [1, 3, 32, 32]
# Input ratio: 0.0
# Input training_mode: True
# Output: [1, 3, 32, 32] (copy, no dropout)
# Mask: [1, 3, 32, 32] (all true)

# PyTorch Equivalent
output = input_data  # No dropout when p=0
```

### Example 8: Training Mode False (v12+)

```python
# ONNX Model (v12+)
# Input data: [1, 3, 32, 32]
# Input training_mode: False (or not provided)
# Output: [1, 3, 32, 32] (copy, ratio ignored)
# Mask: [1, 3, 32, 32] (all true)

# PyTorch Equivalent
dropout = nn.Dropout(p=0.5)
dropout.eval()  # training_mode=False
output = dropout(input_data)  # Simple copy, p is ignored
```

---

## Version Comparison Examples

### Comparison: v1 vs v12 - Training Mode

**v1 Approach:**
```python
# ONNX v1
# Attributes: ratio=0.5, is_test=0
# Uses attributes to control behavior

# PyTorch
dropout = nn.Dropout(p=0.5)
dropout.train()
output = dropout(input_data)
```

**v12 Approach:**
```python
# ONNX v12
# Input ratio: 0.5 (tensor)
# Input training_mode: True (tensor)
# Uses inputs to control behavior (more flexible)

# PyTorch
ratio_tensor = torch.tensor(0.5)
training_tensor = torch.tensor(True)
# Manual implementation or use F.dropout with dynamic p
output = F.dropout(input_data, p=ratio_tensor.item(), training=training_tensor.item())
```

**Key Difference:** v12 allows ratio and training_mode to be different for each inference call, while v1 requires them to be fixed at model creation time.

### Comparison: v7 vs v12 - Training Mode Detection

**v7 Approach:**
```python
# ONNX v7
# Attributes: ratio=0.5
# Graph context: training=True
# Relies on graph-level flag

# PyTorch
dropout = nn.Dropout(p=0.5)
dropout.train()  # Graph-level mode
output = dropout(input_data)
```

**v12 Approach:**
```python
# ONNX v12
# Input ratio: 0.5 (tensor)
# Input training_mode: True (tensor)
# Explicit per-call control

# PyTorch
ratio_tensor = torch.tensor(0.5)
training_tensor = torch.tensor(True)
output = F.dropout(input_data, p=ratio_tensor.item(), training=training_tensor.item())
```

**Key Difference:** v12 provides explicit per-call control, while v7 relies on graph-level context which may be less flexible.

### Comparison: v13 vs v22 - Type Support

**v13 Types:**
- Input/Output: `double`, `float`, `float16`, `bfloat16`
- Ratio: `double`, `float`, `float16`

**v22 Types:**
- Input/Output: `double`, `float`, `float16`, `bfloat16`, `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`
- Ratio: All same types as input/output

**Key Difference:** v22 adds float8 types for quantization and memory efficiency.

---

## Implementation Considerations

### Forge TIR Mapping

The Forge implementation maps ONNX Dropout to TIR `DropoutNode` with the following attributes:
- `p`: Dropout probability (from `ratio` attribute or input)
- `training`: Training mode flag (from `is_test`, graph context, or `training_mode` input)
- `seed`: Random seed (from `seed` attribute)

### Handling Different Versions

1. **v1-v6:** Extract `ratio` and `is_test` from attributes
2. **v7-v10:** Extract `ratio` from attribute, determine training mode from graph context
3. **v12+:** Extract `ratio` and `training_mode` from inputs (with defaults), extract `seed` from attribute

### Mask Output

- Forge typically only uses the `output` tensor, not the `mask`
- The mask can be optionally generated if needed for debugging or special use cases

---

## References

- [ONNX Dropout Operator Documentation](https://onnx.ai/onnx/operators/onnx__Dropout.html)
- [PyTorch Dropout Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [PyTorch Functional Dropout Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html)
