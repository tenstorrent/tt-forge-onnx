# ONNX ReduceMean Complete Opset Version Summary

Based on the [official ONNX ReduceMean documentation](https://onnx.ai/onnx/operators/onnx__ReduceMean.html), this document provides a comprehensive summary of all opset versions.

## Overview

ReduceMean computes the mean (average) of the input tensor's elements along the provided axes. The resulting tensor has the same rank as the input if `keepdims=1`, or has reduced dimensions pruned if `keepdims=0`.

---

## Version-by-Version Breakdown

### **ReduceMean v1** (since version 1)

**Key Characteristics:**
- **Axes**: Attribute (`axes` as INTS attribute)
  - List of integers along which to reduce
  - Default: reduce over all dimensions of the input tensor
  - **Note**: v1 doesn't specify accepted range in summary (unlike later versions)
- **Keepdims**: Attribute (default `1` = True)
- **Inputs**: 
  - `data` (T): Input tensor
- **Outputs**:
  - `reduced` (T): Reduced output tensor
- **Type Constraints**: 
  - `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
- **Special Behavior**: 
  - ✅ **Input tensors of rank zero are valid**
  - ⚠️ **Reduction over an empty set of values yields undefined** (unlike ReduceMax which yields -inf/min)

---

### **ReduceMean v11** (since version 11)

**Key Characteristics:**
- **Axes**: Attribute (`axes` as INTS attribute)
  - List of integers along which to reduce
  - Default: reduce over all dimensions
  - **Accepted range**: `[-r, r-1]` where r = rank(data) *(explicitly stated in v11+)*
- **Keepdims**: Attribute (default `1` = True)
- **Inputs**: 
  - `data` (T): Input tensor
- **Outputs**:
  - `reduced` (T): Reduced output tensor
- **Type Constraints**: 
  - Same as v1: `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
- **Special Behavior**: 
  - ⚠️ **Note**: v11 summary does NOT explicitly mention rank-zero tensors or empty set behavior (unlike v1)

**Changes from v1:**
- Explicitly documents the accepted range for axes: `[-r, r-1]`
- Removed explicit mention of rank-zero and empty set behavior in summary

---

### **ReduceMean v13** (since version 13)

**Key Characteristics:**
- **Axes**: **Still an attribute** (`axes` as INTS attribute)
  - List of integers along which to reduce
  - Default: reduce over all dimensions
  - **Accepted range**: `[-r, r-1]` where r = rank(data)
- **Keepdims**: Attribute (default `1` = True)
- **Inputs**: 
  - `data` (T): Input tensor
  - **Note**: No axes input - axes is still an attribute in v13
- **Outputs**:
  - `reduced` (T): Reduced output tensor
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(bfloat16)`
  - Full list: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
- **Special Behavior**: 
  - ✅ **Input tensors of rank zero are valid**
  - ⚠️ **Reduction over an empty set of values yields undefined**

**Changes from v11:**
- **Type support expanded**: Added `bfloat16`
- Restores explicit mention of rank-zero and empty set behavior in summary

---

### **ReduceMean v18** (since version 18) ⚠️ **MAJOR BREAKING CHANGE**

**Key Characteristics:**
- **Axes**: **Now an optional input tensor** (instead of attribute)
  - Second input: `axes` (optional, tensor of int64)
  - Can be provided as constant initializer
  - If not provided, behavior depends on `noop_with_empty_axes`
- **New Attribute**: `noop_with_empty_axes` (INT, default `0` = False)
  - If `True`: When axes is empty/not provided, operation is a **no-op (identity)**
  - If `False` (default): When axes is empty/not provided, **reduce over all axes**
  - Note: For composite reduction operators (like ReduceLogSum, ReduceSumSquare), the non-reduction steps still execute
- **Keepdims**: Attribute (default `1` = True)
- **Inputs**:
  - **Between 1 and 2 inputs**:
  - `data` (T): Input tensor
  - `axes` (optional, tensor(int64)): Optional input list of integers, along which to reduce
- **Outputs**:
  - `reduced` (T): Reduced output tensor
- **Type Constraints**: 
  - Same as v13: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
- **Special Behavior**: 
  - ✅ **Input tensors of rank zero are valid**
  - ⚠️ **Reduction over an empty set of values yields undefined**

**Changes from v13:**
- **BREAKING**: Axes is now an **optional input tensor** instead of attribute
- **NEW**: `noop_with_empty_axes` attribute introduced

---

## Summary Table

| Opset | Axes Format | noop_with_empty_axes | Type Support | Rank-0/Empty Set Mentioned | Empty Set Result |
|-------|-------------|---------------------|--------------|---------------------------|------------------|
| **1** | Attribute | ❌ Not supported | 8 types (no bfloat16) | ✅ Yes | ⚠️ Undefined |
| **11** | Attribute | ❌ Not supported | 8 types (no bfloat16) | ❌ No | ❌ Not mentioned |
| **13** | Attribute | ❌ Not supported | **9 types** (adds bfloat16) | ✅ Yes | ⚠️ Undefined |
| **18** | **Input tensor** | ✅ Supported | 9 types | ✅ Yes | ⚠️ Undefined |

---

## Key Observations

### 1. **Rank-Zero and Empty Set Behavior**
- **Explicitly documented in**: v1, v13, v18
- **Not mentioned in summary for**: v11
- **Empty set behavior**: Always **undefined** (unlike ReduceMax which yields -inf/min)

### 2. **Type Support Evolution**
- **v1-v11**: 8 types (no bfloat16)
- **v13+**: 9 types (adds bfloat16)
- **Note**: ReduceMean does NOT support int8, uint8, or bool (unlike ReduceMax)

### 3. **Axes Format Change**
- **v1-v13**: Axes as attribute
- **v18+**: Axes as optional input tensor (allows dynamic axes)

### 4. **noop_with_empty_axes**
- **Introduced in**: v18
- **Purpose**: Control behavior when axes is empty/not provided
- **Default**: `0` (False) = reduce all axes
- **When True**: Identity operation (no-op)
- **Note**: For composite reduction operators, non-reduction steps still execute

### 5. **Empty Set Reduction Behavior**
- **ReduceMean**: Always yields **undefined** (implementation-dependent)
- **ReduceMax**: Yields `-inf` (floats) or minimum value (ints)
- This is a key difference between ReduceMean and ReduceMax

---

## Comparison with ReduceMax

| Feature | ReduceMean | ReduceMax |
|---------|------------|-----------|
| **Axes as input tensor** | Opset 18+ | Opset 18+ |
| **noop_with_empty_axes** | Opset 18+ | Opset 18+ |
| **Empty set result** | ⚠️ Undefined | `-inf` (floats) or min (ints) |
| **Type support** | 9 types (no int8, uint8, bool) | 12 types (includes int8, uint8, bool in v20+) |
| **Rank-zero support** | ✅ Yes | ✅ Yes |

---

## Implementation Notes

### Current Codebase Implementation

The codebase in `forge/forge/transpiler/frontends/onnx/converters/reduction.py`:

1. **v1-v12**: Uses `_impl_v1()` method
   - Extracts `axes` from attributes
   - Converts to PyTorch `dim` format
   - Creates `ReduceMeanNode`

2. **v13**: Currently uses `_impl_v1()` method (correct - v13 still uses axes as attribute)
   - Same as v1-v12

3. **v18+**: Uses `_impl_v13()` method
   - Extracts `axes` from optional input tensor (second input)
   - Handles `noop_with_empty_axes` attribute
   - If `noop_with_empty_axes=True` and axes is empty/None → returns `IdentityNode`
   - Otherwise, falls back to `_impl_v1()` logic

### Important Considerations

1. **Backward Compatibility**: Opset 18+ uses axes as input tensor (not attribute)
2. **Empty Axes Behavior**: 
   - Opset 1-13: Empty axes = reduce all dimensions
   - Opset 18+: Controlled by `noop_with_empty_axes`
3. **Rank-Zero Tensors**: Should be supported across all versions (explicitly documented in v1, v13, v18)
4. **Empty Set Reduction**: Yields **undefined** (implementation-dependent, unlike ReduceMax)
5. **Type Limitations**: Does NOT support int8, uint8, or bool types

---

## Testing Recommendations

When testing ReduceMean across opset versions:

### Opset 1-12:
- ✅ Test with axes as attribute
- ✅ Test rank-zero (scalar) inputs
- ✅ Test empty tensor reductions (result is undefined - may vary by implementation)
- ✅ Test with different data types (no int8, uint8, bool)

### Opset 13:
- ✅ Test with axes as attribute (same as v1-v12)
- ✅ Test rank-zero (scalar) inputs
- ✅ Test empty tensor reductions
- ✅ Test with bfloat16 type (new in v13)

### Opset 18+:
- ✅ Test with axes as input tensor (constant initializer)
- ✅ Test with axes not provided
- ✅ Test with empty axes tensor + `noop_with_empty_axes=True` (should be identity)
- ✅ Test with empty axes tensor + `noop_with_empty_axes=False` (should reduce all)
- ✅ Test rank-zero inputs (behavior should be consistent)
- ✅ Test empty tensor reductions (result is undefined)

---

## Key Differences from ReduceMax

1. **Empty Set Behavior**: 
   - ReduceMean: **Undefined** (implementation-dependent)
   - ReduceMax: `-inf` (floats) or minimum value (ints)

2. **Type Support**:
   - ReduceMean: 9 types (no int8, uint8, bool)
   - ReduceMax: 12 types (includes int8, uint8, bool in v20+)

3. **Axes Format Change**:
   - Both: Axes becomes input tensor in opset 18
   - Both: `noop_with_empty_axes` introduced in opset 18

---

## References

- [ONNX ReduceMean Documentation](https://onnx.ai/onnx/operators/onnx__ReduceMean.html)
- ONNX Specification v1.21.0

