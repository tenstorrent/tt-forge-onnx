# ONNX Reshape Complete Opset Version Summary

Based on the [official ONNX Reshape documentation](https://onnx.ai/onnx/operators/onnx__Reshape.html), this document provides a comprehensive summary of all opset versions.

## Overview

Reshape the input tensor similar to `numpy.reshape`. The operator takes a data tensor and a shape specification, and outputs a reshaped tensor. The input tensor's shape and the output tensor's shape are required to have the same number of elements.

**Key Concepts:**
- **-1 dimension**: At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
- **0 dimension**: A dimension could be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor). This behavior changed with the introduction of `allowzero` attribute.
- **Empty shape**: Shape (second input) could be an empty shape, which means converting to a scalar.

---

## Version-by-Version Breakdown

### **Reshape v1** (since version 1)

**Key Characteristics:**
- **Shape**: Attribute (`shape` as INTS attribute)
  - New shape specified as attribute
  - List of integers defining the target shape
- **Consumed Inputs**: Attribute (`consumed_inputs` as INTS)
  - Legacy optimization attribute
- **Inputs**: 
  - `data` (T): Input tensor
- **Outputs**:
  - `reshaped` (T): Reshaped data
- **Type Constraints**: 
  - `tensor(double)`, `tensor(float)`, `tensor(float16)`
  - **Limited to float types only**
- **Shape Inference**: ❌ False (not supported)
- **Special Behavior**: 
  - Supports -1 for inferred dimension
  - Supports 0 to copy dimension from input
  - Supports empty shape for scalar conversion

**Summary**: Initial version with shape as attribute, limited to float types only, no shape inference.

---

### **Reshape v5** (since version 5)

**Key Characteristics:**
- **Shape**: **Input tensor** (`shape` as `tensor(int64)`)
  - **Major change**: Shape moved from attribute to input tensor
  - Specified shape for output
  - Must be a tensor of type int64
- **Inputs**: 
  - `data` (T): Input tensor
  - `shape` (tensor(int64)): Specified shape for output
- **Outputs**:
  - `reshaped` (T): Reshaped data
- **Type Constraints**: 
  - **Expanded significantly**:
    - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    - `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`
    - `tensor(string)`
    - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ True (supported)
- **Special Behavior**: 
  - Supports -1 for inferred dimension
  - Supports 0 to copy dimension from input
  - Supports empty shape for scalar conversion

**Changes from v1:**
- ✅ **Shape becomes input tensor** (major architectural change)
- ✅ **Type constraints expanded** (from 3 float types to 15+ types including bool, complex, int, uint, string)
- ✅ **Shape inference enabled**
- ❌ **Removed `consumed_inputs` attribute** (legacy)

**Summary**: Major version that moved shape from attribute to input tensor and significantly expanded type support.

---

### **Reshape v13** (since version 13)

**Key Characteristics:**
- **Shape**: Input tensor (`shape` as `tensor(int64)`)
- **Inputs**: 
  - `data` (T): Input tensor
  - `shape` (tensor(int64)): Specified shape for output
- **Outputs**:
  - `reshaped` (T): Reshaped data
- **Type Constraints**: 
  - Same as v5:
    - `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    - `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`
    - `tensor(string)`
    - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
  - **Added**: `tensor(bfloat16)`
- **Attributes**: None (no `allowzero` yet)
- **Shape Inference**: ✅ True
- **Special Behavior**: 
  - Supports -1 for inferred dimension
  - Supports 0 to copy dimension from input (default behavior)
  - Supports empty shape for scalar conversion

**Changes from v5:**
- ✅ **Added `bfloat16` type support**

**Summary**: Minor update adding bfloat16 type support.

---

### **Reshape v14** (since version 14)

**Key Characteristics:**
- **Shape**: Input tensor (`shape` as `tensor(int64)`)
- **Attributes**: 
  - `allowzero` (INT, default `0`):
    - **New attribute introduced**
    - By default (`allowzero=0`): When any value in the 'shape' input is equal to zero, the corresponding dimension value is copied from the input tensor dynamically
    - When `allowzero=1`: If any value in the 'shape' input is set to zero, the zero value is honored, similar to NumPy
    - **Constraint**: If `allowzero` is set, it is invalid for the specified shape to contain both a zero value and -1, as the value of the dimension corresponding to -1 cannot be determined uniquely
- **Inputs**: 
  - `data` (T): Input tensor
  - `shape` (tensor(int64)): Specified shape for output
- **Outputs**:
  - `reshaped` (T): Reshaped data
- **Type Constraints**: 
  - Same as v13: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ True
- **Special Behavior**: 
  - Supports -1 for inferred dimension
  - Supports 0 dimension with configurable behavior via `allowzero`:
    - `allowzero=0` (default): Copy dimension from input
    - `allowzero=1`: Explicitly set dimension to zero
  - Supports empty shape for scalar conversion
  - **Invalid**: Cannot have both 0 and -1 when `allowzero=1`

**Changes from v13:**
- ✅ **Introduced `allowzero` attribute** (major behavioral change)
- ✅ **Enhanced 0 dimension handling** (now configurable)

**Summary**: Introduced `allowzero` attribute to control behavior of 0 dimensions, allowing explicit zero dimensions (NumPy-like behavior).

---

### **Reshape v19** (since version 19)

**Key Characteristics:**
- **Shape**: Input tensor (`shape` as `tensor(int64)`)
- **Attributes**: 
  - `allowzero` (INT, default `0`): Same as v14
- **Inputs**: 
  - `data` (T): Input tensor
  - `shape` (tensor(int64)): Specified shape for output
- **Outputs**:
  - `reshaped` (T): Reshaped data
- **Type Constraints**: 
  - **Added new float8 types**:
    - `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
  - All types from v14 plus the new float8 types
- **Shape Inference**: ✅ True
- **Special Behavior**: Same as v14

**Changes from v14:**
- ✅ **Added float8 type support** (4 new float8 variants)

**Summary**: Added support for float8 data types (4 variants).

---

### **Reshape v21** (since version 21)

**Key Characteristics:**
- **Shape**: Input tensor (`shape` as `tensor(int64)`)
- **Attributes**: 
  - `allowzero` (INT, default `0`): Same as v14
- **Inputs**: 
  - `data` (T): Input tensor
  - `shape` (tensor(int64)): Specified shape for output
- **Outputs**:
  - `reshaped` (T): Reshaped data
- **Type Constraints**: 
  - **Added more float8 types**:
    - `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
  - All types from v19
- **Shape Inference**: ✅ True
- **Special Behavior**: Same as v14

**Changes from v19:**
- ✅ **Type constraints remain the same** (no new types added in this version)

**Summary**: No functional changes from v19, maintains same type support.

---

### **Reshape v23** (since version 23)

**Key Characteristics:**
- **Shape**: Input tensor (`shape` as `tensor(int64)`)
- **Attributes**: 
  - `allowzero` (INT, default `0`): Same as v14
- **Inputs**: 
  - `data` (T): Input tensor
  - `shape` (tensor(int64)): Specified shape for output
- **Outputs**:
  - `reshaped` (T): Reshaped data
- **Type Constraints**: 
  - **Added more float8 types**:
    - `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
  - All types from v21 plus `float4e2m1`
- **Shape Inference**: ✅ True
- **Special Behavior**: Same as v14

**Changes from v21:**
- ✅ **Added `float4e2m1` type support**

**Summary**: Added support for float4e2m1 type.

---

### **Reshape v24** (since version 24)

**Key Characteristics:**
- **Shape**: Input tensor (`shape` as `tensor(int64)`)
- **Attributes**: 
  - `allowzero` (INT, default `0`): Same as v14
- **Inputs**: 
  - `data` (T): Input tensor
  - `shape` (tensor(int64)): Specified shape for output
- **Outputs**:
  - `reshaped` (T): Reshaped data
- **Type Constraints**: 
  - Same as v23:
    - `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    - `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
    - `tensor(int16)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`
    - `tensor(string)`
    - `tensor(uint16)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
- **Shape Inference**: ✅ True
- **Special Behavior**: Same as v14

**Changes from v23:**
- ✅ **Added `int4` and `uint4` type support**

**Summary**: Added support for int4 and uint4 types.

---

### **Reshape v25** (since version 25)

**Key Characteristics:**
- **Shape**: Input tensor (`shape` as `tensor(int64)`)
- **Attributes**: 
  - `allowzero` (INT, default `0`): Same as v14
- **Inputs**: 
  - `data` (T): Input tensor
  - `shape` (tensor(int64)): Specified shape for output
- **Outputs**:
  - `reshaped` (T): Reshaped data
- **Type Constraints**: 
  - **Added more types**:
    - `tensor(float8e8m0)` (new)
    - `tensor(int2)`, `tensor(uint2)` (new)
  - All types from v24 plus the new types
- **Shape Inference**: ✅ True
- **Special Behavior**: Same as v14

**Changes from v24:**
- ✅ **Added `float8e8m0` type support**
- ✅ **Added `int2` and `uint2` type support**

**Summary**: Added support for float8e8m0, int2, and uint2 types.

---

## Summary Table

| Opset | Shape | Attributes | Key Changes | Type Count (approx) |
|-------|-------|------------|-------------|---------------------|
| **v1** | Attribute (INTS) | `consumed_inputs` (legacy) | Initial version | 3 (float only) |
| **v5** | **Input tensor** | None | Shape → input, expanded types | 15+ |
| **v13** | Input tensor | None | Added bfloat16 | 16+ |
| **v14** | Input tensor | **`allowzero`** | Introduced allowzero attribute | 16+ |
| **v19** | Input tensor | `allowzero` | Added float8 types (4 variants) | 20+ |
| **v21** | Input tensor | `allowzero` | No changes | 20+ |
| **v23** | Input tensor | `allowzero` | Added float4e2m1 | 21+ |
| **v24** | Input tensor | `allowzero` | Added int4, uint4 | 23+ |
| **v25** | Input tensor | `allowzero` | Added float8e8m0, int2, uint2 | 26+ |

---

## Key Behavioral Notes

### Dimension 0 Behavior

**Before opset 14 (v1-v13):**
- Dimension 0 always means "copy from input tensor"

**Opset 14+ (with `allowzero` attribute):**
- `allowzero=0` (default): Dimension 0 means "copy from input tensor" (backward compatible)
- `allowzero=1`: Dimension 0 means "explicitly set to zero" (NumPy-like behavior)
- **Invalid**: Cannot have both 0 and -1 when `allowzero=1`

### -1 Dimension (Inferred Dimension)

- Supported in all versions
- At most one dimension can be -1
- Value is inferred from the size of the tensor and remaining dimensions
- Cannot be combined with 0 when `allowzero=1` (opset 14+)

### Empty Shape (Scalar Conversion)

- Supported in all versions
- Empty shape means converting to a scalar (rank-zero tensor)

---

## Detailed Examples and Explanations

### Example 1: Basic Reshape (Opset 1)

**Input:**
```python
data = [[1, 2, 3, 4],
        [5, 6, 7, 8]]  # Shape: (2, 4) = 8 elements

# Opset 1: shape as attribute
shape = [4, 2]  # Attribute: shape=[4, 2]
```

**Output:**
```python
reshaped = [[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]]  # Shape: (4, 2) = 8 elements
```

**Explanation:** The 8 elements are rearranged from (2, 4) to (4, 2) layout.

---

### Example 2: Using -1 (Inferred Dimension)

**Input:**
```python
data = [[1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12]]  # Shape: (2, 6) = 12 elements

# Opset 1: shape as attribute
shape = [-1, 3]  # Attribute: shape=[-1, 3]
# -1 means: infer this dimension
# Calculation: total_elements = 12, other_dim = 3
# So: inferred_dim = 12 / 3 = 4
```

**Output:**
```python
reshaped = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]]  # Shape: (4, 3) = 12 elements
```

**Explanation:** 
- Total elements: 12
- Known dimension: 3
- Inferred dimension: 12 ÷ 3 = 4
- Result: (4, 3)

**Another -1 example:**
```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Shape: (12,) = 12 elements
shape = [2, -1, 2]  # Attribute: shape=[2, -1, 2]
# Calculation: total = 12, known dims = 2 * 2 = 4
# So: inferred_dim = 12 / 4 = 3
```

**Output:**
```python
reshaped = [[[1, 2],
             [3, 4],
             [5, 6]],
            [[7, 8],
             [9, 10],
             [11, 12]]]  # Shape: (2, 3, 2) = 12 elements
```

---

### Example 3: Using 0 (Copy from Input) - Opset 1

**Input:**
```python
data = [[1, 2, 3],
        [4, 5, 6]]  # Shape: (2, 3) = 6 elements

# Opset 1: shape as attribute
shape = [0, 6]  # Attribute: shape=[0, 6]
# 0 means: copy this dimension from input
# Input shape is (2, 3), so dimension 0 = 2
# So: shape becomes [2, 6] → but wait, 2*6=12 ≠ 6 elements!
# Actually, let's use a valid example:
```

**Valid Example with 0:**
```python
data = [[1, 2, 3],
        [4, 5, 6]]  # Shape: (2, 3) = 6 elements

shape = [0, 2]  # Attribute: shape=[0, 2]
# 0 means: copy dimension 0 from input = 2
# So: shape becomes [2, 2] → but 2*2=4 ≠ 6 elements!
# This is invalid. Let's try another:
```

**Correct Example with 0:**
```python
data = [[1, 2, 3],
        [4, 5, 6]]  # Shape: (2, 3) = 6 elements

shape = [2, 0]  # Attribute: shape=[2, 0]
# 0 means: copy dimension 1 from input = 3
# So: shape becomes [2, 3] → 2*3=6 ✓
```

**Output:**
```python
reshaped = [[1, 2, 3],
            [4, 5, 6]]  # Shape: (2, 3) = 6 elements (unchanged!)
```

**Another 0 Example:**
```python
data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # Shape: (2, 2, 2) = 8 elements

shape = [0, 4]  # Attribute: shape=[0, 4]
# 0 means: copy dimension 0 from input = 2
# So: shape becomes [2, 4] → 2*4=8 ✓
```

**Output:**
```python
reshaped = [[1, 2, 3, 4],
            [5, 6, 7, 8]]  # Shape: (2, 4) = 8 elements
```

**Key Point:** In opset 1-13, `0` always means "copy the corresponding dimension from the input tensor's shape at that position."

---

### Example 4: Difference Between -1 and 0

**Scenario:** Input shape is (2, 3, 4) = 24 elements, want output shape (6, 4)

**Using -1:**
```python
data = ...  # Shape: (2, 3, 4) = 24 elements
shape = [-1, 4]  # Attribute: shape=[-1, 4]
# -1 means: calculate this dimension
# Calculation: 24 / 4 = 6
# Result: (6, 4)
```

**Using 0:**
```python
data = ...  # Shape: (2, 3, 4) = 24 elements
shape = [0, 4]  # Attribute: shape=[0, 4]
# 0 means: copy dimension 0 from input
# Input dimension 0 = 2
# So: shape becomes [2, 4] → 2*4=8 ≠ 24 elements!
# This is INVALID!
```

**Correct Use of 0:**
```python
data = ...  # Shape: (2, 3, 4) = 24 elements
shape = [6, 0]  # Attribute: shape=[6, 0]
# 0 means: copy dimension 1 from input
# Input dimension 1 = 3
# So: shape becomes [6, 3] → 6*3=18 ≠ 24 elements!
# Still INVALID!

# To use 0 correctly:
shape = [0, 12]  # Attribute: shape=[0, 12]
# 0 means: copy dimension 0 from input = 2
# So: shape becomes [2, 12] → 2*12=24 ✓
```

**Summary of Differences:**

| Feature | -1 | 0 |
|--------|----|---|
| **Meaning** | Calculate/infer this dimension | Copy this dimension from input |
| **Calculation** | `total_elements / (product of other dims)` | `input_shape[position]` |
| **When to use** | When you know total elements but not this dimension | When you want to preserve a dimension from input |
| **Example** | `shape=[-1, 4]` with 24 elements → (6, 4) | `shape=[0, 4]` with input (2, 3, 4) → copies dim 0 = 2 → (2, 4) |

---

### Example 5: Empty Shape (Scalar Conversion)

**Input:**
```python
data = [[42]]  # Shape: (1, 1) = 1 element

# Opset 1: shape as attribute
shape = []  # Attribute: shape=[] (empty list)
```

**Output:**
```python
reshaped = 42  # Shape: () = scalar (rank-zero tensor)
```

**Explanation:** Empty shape converts any tensor with 1 element to a scalar (rank-zero tensor).

---

### Example 6: consumed_inputs Attribute (Legacy - Opset 1 only)

**What is `consumed_inputs`?**

`consumed_inputs` is a **legacy optimization attribute** from early ONNX versions. It was used to indicate which inputs could be modified in-place during optimization passes.

**Example:**
```python
# Opset 1 Reshape node
node = {
    "op_type": "Reshape",
    "inputs": ["data"],
    "outputs": ["reshaped"],
    "attributes": {
        "shape": [4, 2],
        "consumed_inputs": [0]  # Legacy attribute
    }
}
```

**Meaning:**
- `consumed_inputs = [0]` means: "Input 0 (data) can be consumed/modified in-place"
- This was a hint to optimizers that they could reuse the input tensor's memory for the output
- **Note:** This is a legacy attribute and was removed in opset 5+
- Modern ONNX doesn't use this; optimizers determine in-place operations automatically

**Why was it removed?**
- It was implementation-specific and not part of the mathematical operation
- Modern frameworks handle memory optimization automatically
- It made the specification more complex without clear benefits

**Important:** 
- **Opset 1 only**: `consumed_inputs` exists
- **Opset 5+**: `consumed_inputs` removed (no longer needed)

---

### Example 7: Complete Opset 1 Workflow

**ONNX Model (Opset 1):**
```python
import onnx
from onnx import helper, TensorProto

# Create a Reshape node
reshape_node = helper.make_node(
    'Reshape',
    inputs=['input_data'],
    outputs=['output_data'],
    shape=[-1, 8],  # Attribute: reshape to (inferred, 8)
    consumed_inputs=[0]  # Legacy: can modify input in-place
)

# Create graph
graph = helper.make_graph(
    [reshape_node],
    'reshape_example',
    [helper.make_tensor_value_info('input_data', TensorProto.FLOAT, [2, 3, 4])],
    [helper.make_tensor_value_info('output_data', TensorProto.FLOAT, [3, 8])]
)

# Create model
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 1)])
```

**Execution:**
```python
import numpy as np

# Input
input_data = np.random.randn(2, 3, 4).astype(np.float32)  # Shape: (2, 3, 4) = 24 elements

# Reshape with shape=[-1, 8]
# Calculation: 24 / 8 = 3
# Output shape: (3, 8)

output_data = input_data.reshape(3, 8)  # Shape: (3, 8) = 24 elements
```

**Key Points:**
1. Shape is an **attribute** (not an input) in opset 1
2. `consumed_inputs` is a legacy optimization hint
3. Only float types supported (float, double, float16)
4. -1 and 0 work as described above

---

## Understanding `allowzero` Attribute (Opset 14+)

The `allowzero` attribute was introduced in **Opset 14** to control how dimension `0` is interpreted in the shape specification.

### Default Behavior (allowzero=0)

**When `allowzero=0` (default):**
- Dimension `0` means: **"Copy the corresponding dimension from the input tensor"**
- This is the **backward-compatible** behavior (same as opset 1-13)
- The `0` is replaced by the actual dimension value from the input

### New Behavior (allowzero=1)

**When `allowzero=1`:**
- Dimension `0` means: **"Explicitly set this dimension to zero"** (NumPy-like behavior)
- The `0` is taken literally, not replaced
- This allows creating tensors with zero-sized dimensions

### Example 1: allowzero=0 (Default Behavior)

**Input:**
```python
data = [[1, 2, 3],
        [4, 5, 6]]  # Shape: (2, 3) = 6 elements
```

**Reshape with allowzero=0:**
```python
# Opset 14+: shape as input tensor
shape = [0, 2]  # Input tensor: [0, 2]
allowzero = 0  # Default

# Interpretation:
# - 0 means: copy dimension 0 from input = 2
# - 2 means: use 2
# Effective shape: [2, 2] → 2*2 = 4 elements
# But input has 6 elements! This is INVALID.
```

**Valid Example with allowzero=0:**
```python
data = [[1, 2, 3],
        [4, 5, 6]]  # Shape: (2, 3) = 6 elements

shape = [2, 0]  # Input tensor: [2, 0]
allowzero = 0  # Default

# Interpretation:
# - 2 means: use 2
# - 0 means: copy dimension 1 from input = 3
# Effective shape: [2, 3] → 2*3 = 6 elements ✓
```

**Output:**
```python
reshaped = [[1, 2, 3],
            [4, 5, 6]]  # Shape: (2, 3) = 6 elements (unchanged)
```

**Another Example with allowzero=0:**
```python
data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # Shape: (2, 2, 2) = 8 elements

shape = [0, 4]  # Input tensor: [0, 4]
allowzero = 0  # Default

# Interpretation:
# - 0 means: copy dimension 0 from input = 2
# - 4 means: use 4
# Effective shape: [2, 4] → 2*4 = 8 elements ✓
```

**Output:**
```python
reshaped = [[1, 2, 3, 4],
            [5, 6, 7, 8]]  # Shape: (2, 4) = 8 elements
```

---

### Example 2: allowzero=1 (NumPy-like Behavior)

**Input:**
```python
data = [[1, 2, 3],
        [4, 5, 6]]  # Shape: (2, 3) = 6 elements
```

**Reshape with allowzero=1:**
```python
shape = [0, 2]  # Input tensor: [0, 2]
allowzero = 1  # NumPy-like behavior

# Interpretation:
# - 0 means: explicitly set to 0 (NOT copy from input!)
# - 2 means: use 2
# Effective shape: [0, 2] → 0*2 = 0 elements
# This creates an empty tensor with shape (0, 2)
```

**Output:**
```python
reshaped = []  # Shape: (0, 2) = 0 elements (empty tensor)
# This is a valid tensor with 0 rows and 2 columns
```

**Another Example with allowzero=1:**
```python
data = [[1, 2, 3],
        [4, 5, 6]]  # Shape: (2, 3) = 6 elements

shape = [3, 0]  # Input tensor: [3, 0]
allowzero = 1

# Interpretation:
# - 3 means: use 3
# - 0 means: explicitly set to 0
# Effective shape: [3, 0] → 3*0 = 0 elements
# This creates an empty tensor with shape (3, 0)
```

**Output:**
```python
reshaped = []  # Shape: (3, 0) = 0 elements (empty tensor)
# This is a valid tensor with 3 rows and 0 columns
```

**Important Note:** When `allowzero=1` and shape contains `0`, the total number of elements in the output will be 0 (empty tensor), regardless of input size!

---

### Example 3: Side-by-Side Comparison

**Input:**
```python
data = [[1, 2, 3, 4],
        [5, 6, 7, 8]]  # Shape: (2, 4) = 8 elements
```

**Case A: allowzero=0 (Default)**
```python
shape = [0, 8]  # Input tensor: [0, 8]
allowzero = 0

# Interpretation:
# - 0 → copy dimension 0 from input = 2
# - 8 → use 8
# Effective shape: [2, 8] → 2*8 = 16 elements
# But input has 8 elements! This is INVALID.
```

**Case B: allowzero=1 (NumPy-like)**
```python
shape = [0, 8]  # Input tensor: [0, 8]
allowzero = 1

# Interpretation:
# - 0 → explicitly set to 0
# - 8 → use 8
# Effective shape: [0, 8] → 0*8 = 0 elements
# This creates an empty tensor (0, 8)
```

**Valid Example - allowzero=0:**
```python
data = [[1, 2, 3, 4],
        [5, 6, 7, 8]]  # Shape: (2, 4) = 8 elements

shape = [0, 4]  # Input tensor: [0, 4]
allowzero = 0

# Interpretation:
# - 0 → copy dimension 0 from input = 2
# - 4 → use 4
# Effective shape: [2, 4] → 2*4 = 8 elements ✓
```

**Output:**
```python
reshaped = [[1, 2, 3, 4],
            [5, 6, 7, 8]]  # Shape: (2, 4) = 8 elements (unchanged)
```

**Same Input - allowzero=1:**
```python
data = [[1, 2, 3, 4],
        [5, 6, 7, 8]]  # Shape: (2, 4) = 8 elements

shape = [0, 4]  # Input tensor: [0, 4]
allowzero = 1

# Interpretation:
# - 0 → explicitly set to 0
# - 4 → use 4
# Effective shape: [0, 4] → 0*4 = 0 elements
```

**Output:**
```python
reshaped = []  # Shape: (0, 4) = 0 elements (empty tensor)
```

---

### Example 4: Real-World Use Case - allowzero=1

**Scenario:** You want to create an empty tensor with a specific shape structure for later concatenation.

```python
# Input: Some data
data = [[1, 2, 3],
        [4, 5, 6]]  # Shape: (2, 3) = 6 elements

# Goal: Create an empty tensor with shape (0, 3) to match column structure
shape = [0, 3]  # Input tensor: [0, 3]
allowzero = 1

# Result:
reshaped = []  # Shape: (0, 3) = 0 elements
# This empty tensor can be used for concatenation:
# concat([reshaped, new_data], axis=0) where new_data has shape (N, 3)
```

**Why this is useful:**
- Allows creating placeholder tensors with zero-sized dimensions
- Useful for dynamic graph construction
- Matches NumPy's behavior: `np.zeros((0, 3))` creates shape (0, 3)

---

### Example 5: Invalid Combination - 0 and -1 with allowzero=1

**Important Constraint:** When `allowzero=1`, you **cannot** have both `0` and `-1` in the shape!

**Why?**
- `-1` needs to calculate the dimension: `total_elements / (product of other dims)`
- `0` with `allowzero=1` means the dimension is explicitly 0
- If both exist, the calculation becomes ambiguous: `total_elements / (0 * other_dims)` is undefined

**Invalid Example:**
```python
data = [[1, 2, 3],
        [4, 5, 6]]  # Shape: (2, 3) = 6 elements

shape = [-1, 0]  # Input tensor: [-1, 0]
allowzero = 1

# This is INVALID!
# - -1 needs to calculate: 6 / (0 * ?) = undefined
# - 0 means dimension is 0
# Cannot determine -1 when one dimension is 0!
```

**Valid Alternatives:**
```python
# Option 1: Use allowzero=0 (0 copies from input)
shape = [-1, 0]  # Input tensor: [-1, 0]
allowzero = 0
# - 0 → copy dimension 1 from input = 3
# - -1 → calculate: 6 / 3 = 2
# Effective shape: [2, 3] ✓

# Option 2: Don't use 0 with allowzero=1
shape = [-1, 3]  # Input tensor: [-1, 3]
allowzero = 1
# - 3 → use 3
# - -1 → calculate: 6 / 3 = 2
# Effective shape: [2, 3] ✓
```

---

### Example 6: Complete ONNX Model with allowzero

**ONNX Model (Opset 14+):**
```python
import onnx
from onnx import helper, TensorProto
import numpy as np

# Create shape input tensor
shape_tensor = np.array([0, 4], dtype=np.int64)

# Create Reshape node with allowzero=1
reshape_node = helper.make_node(
    'Reshape',
    inputs=['input_data', 'shape'],
    outputs=['output_data'],
    allowzero=1  # Attribute: allowzero=1
)

# Create graph
graph = helper.make_graph(
    [reshape_node],
    'reshape_allowzero_example',
    [
        helper.make_tensor_value_info('input_data', TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info('shape', TensorProto.INT64, [2])
    ],
    [helper.make_tensor_value_info('output_data', TensorProto.FLOAT, [0, 4])]
)

# Add shape as initializer
shape_initializer = helper.make_tensor(
    'shape',
    TensorProto.INT64,
    [2],
    shape_tensor.tobytes()
)

graph.initializer.append(shape_initializer)

# Create model
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
```

**Execution:**
```python
import numpy as np

# Input
input_data = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)  # Shape: (2, 3)

# With allowzero=1, shape=[0, 4] creates output shape (0, 4)
# This is an empty tensor with 0 rows and 4 columns
output_data = np.array([], dtype=np.float32).reshape(0, 4)  # Shape: (0, 4)
```

---

### Summary Table: allowzero Behavior

| allowzero | Dimension 0 Meaning | Example: shape=[0, 4] | Use Case |
|-----------|----------------------|----------------------|----------|
| **0** (default) | Copy from input | `0` → copy dim 0 from input<br>If input is (2, 3), becomes [2, 4] | Backward compatible, preserve dimensions |
| **1** | Explicitly zero | `0` → dimension is 0<br>Becomes [0, 4] (empty tensor) | Create empty tensors, NumPy compatibility |

**Key Takeaways:**
1. **allowzero=0** (default): `0` means "copy from input" - backward compatible with opset 1-13
2. **allowzero=1**: `0` means "explicitly zero" - creates empty tensors, matches NumPy behavior
3. **Cannot combine** `0` and `-1` when `allowzero=1` (ambiguous calculation)
4. **Empty tensors** are valid outputs when `allowzero=1` and shape contains `0`

### Type Support Evolution

- **v1**: Float types only (double, float, float16)
- **v5**: Major expansion (bool, complex, int, uint, string)
- **v13**: Added bfloat16
- **v14-v25**: Incremental additions of float8 variants, int2/int4, uint2/uint4

---

## Implementation Considerations

1. **Shape as Attribute vs Input**: 
   - Opset 1: Shape is an attribute (static)
   - Opset 5+: Shape is an input tensor (dynamic)

2. **Allowzero Attribute**:
   - Opset 1-13: Not available, 0 always copies from input
   - Opset 14+: Available, controls 0 dimension behavior

3. **Type Constraints**:
   - Early versions (v1) very limited
   - v5+ significantly expanded
   - v13+ incremental additions of specialized types

4. **Shape Inference**:
   - v1: Not supported
   - v5+: Supported

---

## References

- [ONNX Reshape Operator Documentation](https://onnx.ai/onnx/operators/onnx__Reshape.html)
- ONNX Specification Repository

