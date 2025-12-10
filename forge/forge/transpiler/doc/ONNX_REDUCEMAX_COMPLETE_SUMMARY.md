# ONNX ReduceMax Complete Opset Version Summary

Based on the [official ONNX ReduceMax documentation](https://onnx.ai/onnx/operators/onnx__ReduceMax.html), this document provides a comprehensive summary of all opset versions.

## Overview

ReduceMax computes the maximum of the input tensor's elements along the provided axes. The resulting tensor has the same rank as the input if `keepdims=1`, or has reduced dimensions pruned if `keepdims=0`.

---

## Version-by-Version Breakdown

### **ReduceMax v1** (since version 1)

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
  - ✅ **Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise**

---

### **ReduceMax v11** (since version 11)

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
  - ✅ **Input tensors of rank zero are valid**
  - ✅ **Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise**

**Changes from v1:**
- Explicitly documents the accepted range for axes: `[-r, r-1]`

---

### **ReduceMax v12** (since version 12)

**Key Characteristics:**
- **Axes**: Attribute (`axes` as INTS attribute)
  - List of integers along which to reduce
  - Default: reduce over all dimensions
  - **Accepted range**: `[-r, r-1]` where r = rank(data)
- **Keepdims**: Attribute (default `1` = True)
- **Inputs**: 
  - `data` (T): Input tensor
- **Outputs**:
  - `reduced` (T): Reduced output tensor
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(bfloat16)`, `tensor(int8)`, `tensor(uint8)`
  - Full list: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Special Behavior**: 
  - ⚠️ **Note**: v12 summary does NOT explicitly mention rank-zero tensors or empty set behavior (unlike v1 and v11)

**Changes from v11:**
- **Type support expanded**: Added `bfloat16`, `int8`, `uint8`
- Removed explicit mention of rank-zero and empty set behavior in summary (though behavior likely unchanged)

---

### **ReduceMax v13** (since version 13)

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
  - Same as v12: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Special Behavior**: 
  - ✅ **Input tensors of rank zero are valid**
  - ✅ **Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise**

**Changes from v12:**
- No functional changes - v13 is identical to v12
- Both use axes as attribute

---

### **ReduceMax v18** (since version 18) ⚠️ **MAJOR BREAKING CHANGE**

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
  - Same as v12/v13: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Special Behavior**: 
  - ✅ **Input tensors of rank zero are valid**
  - ✅ **Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise**

**Changes from v13:**
- **BREAKING**: Axes is now an **optional input tensor** instead of attribute
- **NEW**: `noop_with_empty_axes` attribute introduced
- Restores explicit mention of rank-zero and empty set behavior in summary

---

### **ReduceMax v20** (since version 20)

**Key Characteristics:**
- **Axes**: Optional input tensor (same as v18)
- **noop_with_empty_axes**: Supported (same as v18)
- **Keepdims**: Attribute (default `1` = True)
- **Inputs**:
  - **Between 1 and 2 inputs**:
  - `data` (T): Input tensor
  - `axes` (optional, tensor(int64)): Axes to reduce along
- **Outputs**:
  - `reduced` (T): Reduced output tensor
- **Type Constraints**: 
  - **EXPANDED**: Adds support for `tensor(bool)`
  - Full list: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
- **Special Behavior**: 
  - ✅ **Input tensors of rank zero are valid**
  - ✅ **Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise**
  - ✅ **NEW**: If the input data type is Boolean, the comparison should consider `False < True`

**Changes from v18:**
- **Type support expanded**: Added `tensor(bool)` support
- **NEW**: Boolean comparison semantics documented (False < True)

---

## Summary Table

| Opset | Axes Format | noop_with_empty_axes | Type Support | Rank-0/Empty Set Mentioned |
|-------|-------------|---------------------|--------------|---------------------------|
| **1** | Attribute | ❌ Not supported | 8 types (no bfloat16, int8, uint8) | ✅ Yes |
| **11** | Attribute | ❌ Not supported | 8 types (no bfloat16, int8, uint8) | ✅ Yes |
| **12** | Attribute | ❌ Not supported | **11 types** (adds bfloat16, int8, uint8) | ❌ No |
| **13** | Attribute | ❌ Not supported | 11 types | ✅ Yes |
| **18** | **Input tensor** | ✅ Supported | 11 types | ✅ Yes |
| **20** | Input tensor | ✅ Supported | **12 types** (adds bool) | ✅ Yes |

---

## Key Observations

### 1. **Rank-Zero and Empty Set Behavior**
- **Explicitly documented in**: v1, v11
- **Not mentioned in summary for**: v12, v13, v18, v20
- **Likely behavior**: Still valid (behavior probably unchanged, just not mentioned in later version summaries)

### 2. **Type Support Evolution**
- **v1-v11**: 8 types (no bfloat16, int8, uint8)
- **v12-v18**: 11 types (adds bfloat16, int8, uint8)
- **v20+**: 12 types (adds bool)

### 3. **Axes Format Change**
- **v1-v13**: Axes as attribute
- **v18+**: Axes as optional input tensor (allows dynamic axes)

### 4. **noop_with_empty_axes**
- **Introduced in**: v18
- **Purpose**: Control behavior when axes is empty/not provided
- **Default**: `0` (False) = reduce all axes
- **When True**: Identity operation (no-op)
- **Note**: For composite reduction operators, non-reduction steps still execute

---

## Implementation Notes

### Current Codebase Implementation

The codebase in `forge/forge/transpiler/frontends/onnx/converters/reduction.py`:

1. **v1-v12**: Uses `_impl_v1()` method
   - Extracts `axes` from attributes
   - Converts to PyTorch `dim` format
   - Creates `ReduceMaxNode`

2. **v13**: Currently uses `_impl_v1()` method (correct - v13 still uses axes as attribute)
   - Same as v1-v12

3. **v18+**: Currently uses `_impl_v13()` method
   - ⚠️ **Note**: Method is named `_impl_v13` but should apply to opset 18+
   - Extracts `axes` from optional input tensor (second input)
   - Handles `noop_with_empty_axes` attribute
   - If `noop_with_empty_axes=True` and axes is empty/None → returns `IdentityNode`
   - Otherwise, falls back to `_impl_v1()` logic
   - **Recommendation**: Consider renaming `_impl_v13` to `_impl_v18` for clarity

### Important Considerations

1. **Backward Compatibility**: Opset 18+ uses axes as input tensor (not attribute)
2. **Empty Axes Behavior**: 
   - Opset 1-13: Empty axes = reduce all dimensions
   - Opset 18+: Controlled by `noop_with_empty_axes`
3. **Rank-Zero Tensors**: Should be supported across all versions (explicitly documented in v1, v11)
4. **Empty Set Reduction**: Should yield `-inf` (floats) or minimum value (ints) across all versions

---

## Testing Recommendations

When testing ReduceMax across opset versions:

### Opset 1-12:
- ✅ Test with axes as attribute
- ✅ Test rank-zero (scalar) inputs
- ✅ Test empty tensor reductions (should yield -inf or min value)
- ✅ Test with different data types

### Opset 13:
- ✅ Test with axes as attribute (same as v1-v12)
- ✅ Test rank-zero (scalar) inputs
- ✅ Test empty tensor reductions

### Opset 18+:
- ✅ Test with axes as input tensor (constant initializer)
- ✅ Test with axes not provided
- ✅ Test with empty axes tensor + `noop_with_empty_axes=True` (should be identity)
- ✅ Test with empty axes tensor + `noop_with_empty_axes=False` (should reduce all)
- ✅ Test rank-zero inputs (behavior should be consistent)
- ✅ Test empty tensor reductions

### Opset 20+:
- ✅ All tests from opset 18
- ✅ Test with Boolean type inputs (False < True semantics)

---

## References

- [ONNX ReduceMax Documentation](https://onnx.ai/onnx/operators/onnx__ReduceMax.html)
- ONNX Specification v1.21.0

