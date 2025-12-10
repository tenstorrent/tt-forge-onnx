# Squeeze Converter Implementation Examples

This document provides comprehensive examples of the Squeeze converter implementation, covering all cases including edge cases.

## Implementation Overview

The `SqueezeConverter` handles ONNX Squeeze operations across all opset versions:
- **v1-v10**: `axes` as attribute (non-negative indices only)
- **v11-v12**: `axes` as attribute (supports negative indices)
- **v13+**: `axes` as optional input tensor (supports negative indices)

### Key Features

1. **Negative Index Normalization**: All negative indices are normalized to positive indices
2. **Multiple Axes Support**: Creates multiple `SqueezeNode` operations for multiple axes (in reverse order to avoid index shifting)
3. **Automatic Size-1 Detection**: If `axes` is not provided, automatically finds and squeezes all dimensions with size 1
4. **Validation**: Validates that all specified axes have size 1 before squeezing
5. **Identity Optimization**: If no axes need to be squeezed, creates an `IdentityNode`

---

## Example Cases

### 1. Single Axis Squeeze

**Input**: Shape `(1, 3, 4)`, axes `[0]`  
**Output**: Shape `(3, 4)`

```python
# ONNX Model (v1-v11)
attrs = {'axes': [0]}

# TIR Graph
SqueezeNode(name="squeeze_axis_0", dim=0)
```

**Input**: Shape `(2, 1, 4)`, axes `[1]`  
**Output**: Shape `(2, 4)`

```python
# TIR Graph
SqueezeNode(name="squeeze_axis_1", dim=1)
```

---

### 2. Multiple Axes Squeeze

**Input**: Shape `(1, 2, 1, 3, 1)`, axes `[0, 2, 4]`  
**Output**: Shape `(2, 3)`

```python
# TIR Graph (squeezes in reverse order: 4, 2, 0)
SqueezeNode(name="squeeze_axis_4", dim=4)  # (1, 2, 1, 3)
  -> intermediate_0: (1, 2, 1, 3)
SqueezeNode(name="squeeze_axis_2", dim=2)  # (1, 2, 3)
  -> intermediate_1: (1, 2, 3)
SqueezeNode(name="squeeze_axis_0", dim=0)  # (2, 3)
  -> output_0: (2, 3)
```

**Note**: Axes are sorted in descending order to avoid index shifting after each squeeze.

---

### 3. Negative Indices (v11+)

**Input**: Shape `(1, 3, 1, 5)`, axes `[-1]` (v11+)  
**Output**: Shape `(1, 3, 5)`

```python
# Normalized: -1 -> 3 (for rank 4)
# TIR Graph
SqueezeNode(name="squeeze_axis_3", dim=3)  # Normalized from -1
```

**Input**: Shape `(1, 3, 1, 5)`, axes `[-4, -2]` (v11+)  
**Output**: Shape `(3, 5)`

```python
# Normalized: -4 -> 0, -2 -> 2
# TIR Graph (squeezes in reverse order: 2, 0)
SqueezeNode(name="squeeze_axis_2", dim=2)  # (1, 3, 5)
  -> intermediate_0: (1, 3, 5)
SqueezeNode(name="squeeze_axis_0", dim=0)  # (3, 5)
  -> output_0: (3, 5)
```

---

### 4. No Axes Specified (Squeeze All Size-1)

**Input**: Shape `(1, 3, 1, 5)`, axes `None`  
**Output**: Shape `(3, 5)` (all size-1 dims removed)

```python
# Automatically finds dims 0 and 2 (both have size 1)
# TIR Graph
SqueezeNode(name="squeeze_axis_2", dim=2)  # (1, 3, 5)
  -> intermediate_0: (1, 3, 5)
SqueezeNode(name="squeeze_axis_0", dim=0)  # (3, 5)
  -> output_0: (3, 5)
```

**Input**: Shape `(2, 3, 4)`, axes `None`  
**Output**: Shape `(2, 3, 4)` (no-op, no size-1 dims)

```python
# TIR Graph
IdentityNode(name="squeeze")  # No squeeze needed
```

---

### 5. Axes as Input Tensor (v13+)

**Input**: Shape `(1, 3, 1, 5)`, axes input tensor `[0, 2]` (v13+)  
**Output**: Shape `(3, 5)`

```python
# ONNX Model (v13+)
# Inputs: [data, axes]
# axes is a constant initializer: np.array([0, 2], dtype=np.int64)

# TIR Graph
SqueezeNode(name="squeeze_axis_2", dim=2)  # (1, 3, 5)
  -> intermediate_0: (1, 3, 5)
SqueezeNode(name="squeeze_axis_0", dim=0)  # (3, 5)
  -> output_0: (3, 5)
```

**Input**: Shape `(1, 3, 1, 5)`, no axes input (v13+)  
**Output**: Shape `(3, 5)` (squeezes all size-1 dims)

```python
# ONNX Model (v13+)
# Inputs: [data] (no axes input)

# TIR Graph (same as case 4)
SqueezeNode(name="squeeze_axis_2", dim=2)  # (1, 3, 5)
  -> intermediate_0: (1, 3, 5)
SqueezeNode(name="squeeze_axis_0", dim=0)  # (3, 5)
  -> output_0: (3, 5)
```

---

## Edge Cases

### 6. Scalar Output

**Input**: Shape `(1, 1, 1)`, axes `[0, 1, 2]`  
**Output**: Shape `()` (scalar)

```python
# TIR Graph
SqueezeNode(name="squeeze_axis_2", dim=2)  # (1, 1)
  -> intermediate_0: (1, 1)
SqueezeNode(name="squeeze_axis_1", dim=1)  # (1,)
  -> intermediate_1: (1,)
SqueezeNode(name="squeeze_axis_0", dim=0)  # ()
  -> output_0: ()
```

**Input**: Shape `(1, 1, 1)`, axes `None`  
**Output**: Shape `()` (scalar)

```python
# Same as above - automatically finds all size-1 dims
```

---

### 7. Single Element Tensor

**Input**: Shape `(1,)`, axes `[0]`  
**Output**: Shape `()` (scalar)

```python
# TIR Graph
SqueezeNode(name="squeeze_axis_0", dim=0)  # ()
  -> output_0: ()
```

**Input**: Shape `(1,)`, axes `None`  
**Output**: Shape `()` (scalar)

```python
# Same as above
```

---

### 8. All Ones (2D)

**Input**: Shape `(1, 1)`, axes `[0]`  
**Output**: Shape `(1,)`

```python
# TIR Graph
SqueezeNode(name="squeeze_axis_0", dim=0)  # (1,)
  -> output_0: (1,)
```

**Input**: Shape `(1, 1)`, axes `None`  
**Output**: Shape `()` (squeezes both dims)

```python
# TIR Graph
SqueezeNode(name="squeeze_axis_1", dim=1)  # (1,)
  -> intermediate_0: (1,)
SqueezeNode(name="squeeze_axis_0", dim=0)  # ()
  -> output_0: ()
```

---

### 9. High Dimensional

**Input**: Shape `(1, 2, 1, 3, 1, 4)`, axes `[0, 2, 4]`  
**Output**: Shape `(2, 3, 4)`

```python
# TIR Graph (squeezes in reverse order: 4, 2, 0)
SqueezeNode(name="squeeze_axis_4", dim=4)  # (1, 2, 1, 3, 4)
  -> intermediate_0: (1, 2, 1, 3, 4)
SqueezeNode(name="squeeze_axis_2", dim=2)  # (1, 2, 3, 4)
  -> intermediate_1: (1, 2, 3, 4)
SqueezeNode(name="squeeze_axis_0", dim=0)  # (2, 3, 4)
  -> output_0: (2, 3, 4)
```

---

### 10. No-Op Case

**Input**: Shape `(2, 3, 4)`, axes `None`  
**Output**: Shape `(2, 3, 4)` (no size-1 dims to squeeze)

```python
# TIR Graph
IdentityNode(name="squeeze")  # Optimization: no squeeze needed
```

**Input**: Shape `(2, 3, 4)`, axes `[]` (empty list)  
**Output**: Shape `(2, 3, 4)` (no axes to squeeze)

```python
# TIR Graph
IdentityNode(name="squeeze")  # Optimization: no squeeze needed
```

---

### 11. Duplicate Axes

**Input**: Shape `(1, 3, 1)`, axes `[0, 0, 2]` (duplicates)  
**Output**: Shape `(3,)`

```python
# Duplicates are removed: [0, 0, 2] -> [0, 2]
# TIR Graph
SqueezeNode(name="squeeze_axis_2", dim=2)  # (1, 3)
  -> intermediate_0: (1, 3)
SqueezeNode(name="squeeze_axis_0", dim=0)  # (3,)
  -> output_0: (3,)
```

---

### 12. Mixed Positive and Negative Indices (v11+)

**Input**: Shape `(1, 3, 1, 5)`, axes `[0, -2]` (v11+)  
**Output**: Shape `(3, 5)`

```python
# Normalized: 0 -> 0, -2 -> 2
# TIR Graph
SqueezeNode(name="squeeze_axis_2", dim=2)  # (1, 3, 5)
  -> intermediate_0: (1, 3, 5)
SqueezeNode(name="squeeze_axis_0", dim=0)  # (3, 5)
  -> output_0: (3, 5)
```

---

## Error Cases (Validation)

### 13. Invalid Axis (Out of Range)

**Input**: Shape `(1, 3, 4)`, axes `[5]`  
**Error**: `ValueError: axis 5 is out of range [0, 2]`

### 14. Invalid Axis (Size Not 1)

**Input**: Shape `(2, 3, 4)`, axes `[0]`  
**Error**: `ValueError: cannot squeeze axis 0 with size 2 (only dimensions of size 1 can be squeezed)`

### 15. Negative Index Out of Range (v11+)

**Input**: Shape `(1, 3, 4)`, axes `[-5]` (v11+)  
**Error**: `ValueError: axis -5 is out of range [0, 2]` (after normalization)

---

## Implementation Details

### Normalization Algorithm

```python
def _normalize_axes(axes, input_rank):
    """Normalize axes to positive integers."""
    if axes is None:
        return []
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    # Convert negative indices to positive
    return [idx + input_rank if idx < 0 else idx for idx in map(int, axes)]
```

**Examples**:
- Input rank 4, axes `[-1, 0, -2]` → `[3, 0, 2]`
- Input rank 3, axes `[0, 1, 2]` → `[0, 1, 2]` (no change)

### Multiple Axes Handling

When multiple axes need to be squeezed, they are processed in **descending order** to avoid index shifting:

```python
# Input shape: (1, 2, 1, 3, 1)
# Axes to squeeze: [0, 2, 4]

# Step 1: Squeeze axis 4 → (1, 2, 1, 3)
# Step 2: Squeeze axis 2 → (1, 2, 3)  (axis 2 is still valid)
# Step 3: Squeeze axis 0 → (2, 3)

# If we squeezed in ascending order [0, 2, 4]:
# Step 1: Squeeze axis 0 → (2, 1, 3, 1)  (shape changed!)
# Step 2: Squeeze axis 2 → (2, 1, 3)    (wrong axis - should be 1 now)
# Step 3: Squeeze axis 4 → ERROR (axis 4 doesn't exist anymore)
```

### Automatic Size-1 Detection

When `axes` is not provided, the converter automatically finds all dimensions with size 1:

```python
def _find_all_size_one_dims(input_shape):
    """Find all dimensions with size 1."""
    return [i for i, size in enumerate(input_shape) if size == 1]
```

**Examples**:
- `(1, 3, 1, 5)` → `[0, 2]`
- `(2, 3, 4)` → `[]` (no size-1 dims)
- `(1, 1, 1)` → `[0, 1, 2]`

---

## Test Coverage

The implementation includes comprehensive test cases covering:

1. ✅ Single axis squeeze (positive indices)
2. ✅ Multiple axes squeeze
3. ✅ Negative indices (v11+)
4. ✅ No axes specified (auto-detect)
5. ✅ Axes as input tensor (v13+)
6. ✅ Scalar output
7. ✅ Single element tensor
8. ✅ All ones cases
9. ✅ High dimensional tensors
10. ✅ No-op cases (Identity optimization)
11. ✅ Duplicate axes handling
12. ✅ Mixed positive/negative indices
13. ✅ All supported dtypes (FLOAT, DOUBLE, INT32, INT64, BOOL)
14. ✅ All opset versions (1, 11, 13, 21, 23, 24, 25)

---

## References

- [ONNX Squeeze Operator Documentation](https://onnx.ai/onnx/operators/onnx__Squeeze.html)
- [PyTorch squeeze Documentation](https://pytorch.org/docs/stable/generated/torch.squeeze.html)
- [ONNX_SQUEEZE_COMPLETE_SUMMARY.md](./ONNX_SQUEEZE_COMPLETE_SUMMARY.md)

