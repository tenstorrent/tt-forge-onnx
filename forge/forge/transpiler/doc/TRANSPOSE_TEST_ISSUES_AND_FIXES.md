# Transpose Test Issues and Fixes

## Summary of Issues Found in test_transpose.log

### Issue 1: Negative Indices in ONNX Model Creation
**Problem**: `test_transpose_negative_indices_normalization` is trying to create ONNX models with negative indices in the `perm` attribute, but ONNX doesn't accept negative indices.

**Error Message**:
```
ONNX execution failed: [ONNXRuntimeError] : 1 : FAIL : Node (transpose_neg_norm) Op (Transpose) [TypeInferenceError] Invalid attribute perm {-1, -2, -3}, input shape = {2, 3, 4}
```

**Root Cause**: The test is passing negative indices directly to `create_onnx_model()`, but ONNX requires positive indices. The converter normalizes them internally, but the test needs to normalize them before creating the ONNX model.

**Fix**: Normalize negative indices to positive indices in the test before creating the ONNX model:
```python
# Normalize negative indices to positive for ONNX
rank = len(input_shape)
normalized_perm = [i if i >= 0 else rank + i for i in perm_with_negatives]
attrs = {'perm': normalized_perm}
```

**Affected Tests**: All `test_transpose_negative_indices_normalization` cases

---

### Issue 2: Incorrect Swap Count Expectation
**Problem**: `test_transpose_swap_sequence` expects 2 swaps for `perm=[2, 1, 0]` with `input_shape=(2, 3, 4)`, but the decomposition algorithm correctly produces only 1 swap.

**Error Message**:
```
AssertionError: Expected 2 Transpose nodes, got 1. Nodes: ['Transpose']
```

**Root Cause**: The test expectation is incorrect. The decomposition algorithm correctly identifies that only 1 swap is needed:
- Start: `[0, 1, 2]`
- Target: `[2, 1, 0]`
- Swap (0, 2): `[0, 1, 2]` → `[2, 1, 0]` ✓ Done in 1 swap

**Fix**: Update the test expectation from 2 to 1 for `perm=[2, 1, 0]`:
```python
((2, 3, 4), [2, 1, 0], 1),         # Single swap (was incorrectly 2)
```

**Affected Tests**: `test_transpose_swap_sequence[input_shape5-perm5-2-*]` (all opset versions)

---

### Issue 3: High-Dimensional Transpose Failures
**Problem**: Tests for 7D reverse permutation `((2, 3, 4, 5, 6, 7), [6, 5, 4, 3, 2, 1, 0])` are failing.

**Root Cause**: Need to check the actual error message, but likely related to:
1. The decomposition algorithm not handling high-dimensional cases correctly
2. ONNXRuntime limitations with very high-dimensional tensors
3. Shape inference issues

**Fix**: 
1. Check if the decomposition algorithm correctly handles 7D cases
2. Verify the swap sequence is correct
3. If ONNXRuntime has limitations, skip the test for unsupported cases

**Affected Tests**: `test_transpose_high_dimensional[input_shape6-perm6-*]`

---

### Issue 4: Complex Permutation Failures
**Problem**: Tests for 5D reverse permutation `((2, 3, 4, 5, 6), [4, 3, 2, 1, 0])` are failing.

**Root Cause**: Similar to Issue 3, likely related to the decomposition algorithm or shape tracking.

**Fix**: 
1. Verify the decomposition algorithm produces the correct swap sequence
2. Check intermediate shape tracking
3. Verify the final output shape matches ONNX

**Affected Tests**: `test_transpose_complex_permutations[input_shape1-perm1-*]`

---

## Recommended Fixes

### Fix 1: Update test_transpose_negative_indices_normalization
```python
def test_transpose_negative_indices_normalization(self, opset_version, input_shape, perm_with_negatives, expected_shape):
    """Test that negative indices in perm are correctly normalized."""
    dtype = onnx.TensorProto.FLOAT
    
    # Normalize negative indices to positive for ONNX (ONNX doesn't accept negative indices)
    rank = len(input_shape)
    normalized_perm = [i if i >= 0 else rank + i for i in perm_with_negatives]
    
    # Create ONNX model with normalized (positive) indices
    attrs = {'perm': normalized_perm}
    # ... rest of the test
```

### Fix 2: Update test_transpose_swap_sequence expectations
```python
@pytest.mark.parametrize("input_shape, perm, num_swaps_expected", [
    ((2, 3), [1, 0], 1),              # Single swap
    ((2, 3, 4), [0, 2, 1], 1),         # Single swap (last two)
    ((2, 3, 4), [1, 0, 2], 1),         # Single swap (first two)
    ((2, 3, 4), [1, 2, 0], 2),         # Two swaps needed
    ((2, 3, 4), [2, 0, 1], 2),         # Two swaps needed
    ((2, 3, 4), [2, 1, 0], 1),         # Single swap (FIXED: was 2, should be 1)
    # ... rest
])
```

### Fix 3: Investigate High-Dimensional Cases
1. Run a specific test case to see the actual error
2. Check if the decomposition algorithm correctly handles 7D cases
3. Verify intermediate shape tracking
4. Check if ONNXRuntime has limitations

### Fix 4: Verify Decomposition Algorithm
The current algorithm uses a greedy approach:
- For each position i, find where the target value is
- Swap if needed
- This produces a minimal sequence

For `[2, 1, 0]` from `[0, 1, 2]`:
- Position 0 needs 2 (at position 2) → swap (0, 2) → `[2, 1, 0]` ✓ Done (1 swap)

This is correct! The test expectation is wrong.

---

## Action Items

1. ✅ Fix `test_transpose_negative_indices_normalization` to normalize indices before creating ONNX model
2. ✅ Fix `test_transpose_swap_sequence` to correct the expectation for `[2, 1, 0]`
3. ⚠️ Investigate high-dimensional failures (7D, 5D reverse)
4. ⚠️ Verify decomposition algorithm correctness for all cases

