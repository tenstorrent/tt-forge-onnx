# ONNX Operations with Multiple Outputs

This document lists all ONNX operations that return multiple outputs, based on analysis of the TVM ONNX frontend implementation.

## Summary

**Total: ~20 operations** return multiple outputs in ONNX.

---

## Operations by Category

### 1. **Split Operations** (1 op)
- **Split** - Returns N outputs (variable number based on split sizes)
  - Outputs: Multiple tensors split along specified axis
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:2604-2648`
  - Returns: `TupleWrapper` with N outputs (or single output if N=1)

### 2. **TopK Operations** (1 op)
- **TopK** - Returns 2 outputs
  - Outputs: `[values, indices]`
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:4148-4186`
  - Returns: `TupleWrapper` with 2 outputs

### 3. **Normalization Operations** (4 ops)

- **BatchNormalization** - Returns 5 outputs
  - Outputs: `[output, saved_mean, saved_var, saved_mean, saved_var]` (last two are placeholders)
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:749-763`
  - Returns: `TupleWrapper` with 5 outputs

- **LayerNormalization** - Returns 3 outputs
  - Outputs: `[normalized_output, mean, inv_stdev]`
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:1188-1215`
  - Returns: `TupleWrapper` with 3 outputs

- **EmbedLayerNormalization** - Returns 2 outputs
  - Outputs: `[normalized_output, mask_index]`
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:1218-1263`
  - Returns: `TupleWrapper` with 2 outputs

- **SkipLayerNormalization** - Returns 3 outputs
  - Outputs: `[output, placeholder, placeholder]`
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:1266-1296`
  - Returns: `TupleWrapper` with 3 outputs

### 4. **Attention Operations** (2 ops)

- **Attention** - Returns 2 outputs
  - Outputs: `[output, present]` (present key-value cache)
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:1505-1640`
  - Returns: `TupleWrapper` with 2 outputs

- **QAttention** - Returns 2 outputs
  - Outputs: `[output, present]` (quantized attention)
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:1643-1833`
  - Returns: `TupleWrapper` with 2 outputs

### 5. **RNN Operations** (3 ops)

- **RNN** - Returns 2 outputs
  - Outputs: `[output, hidden_state]`
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:3743-3753`
  - Returns: `TupleWrapper` with 2 outputs

- **GRU** - Returns 2 outputs
  - Outputs: `[output, hidden_state]`
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:3949-3965`
  - Returns: `TupleWrapper` with 2 outputs

- **LSTM** - Returns 3 outputs
  - Outputs: `[output, hidden_state, cell_state]`
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:3858-3872`
  - Returns: `TupleWrapper` with 3 outputs

### 6. **Control Flow Operations** (2 ops)

- **If** - Returns N outputs (variable, based on branch outputs)
  - Outputs: Same as then/else branch outputs
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:4532-4606`
  - Returns: `TupleWrapper` with N outputs (if N > 1)

- **Loop** - Returns N outputs (variable, based on scan outputs)
  - Outputs: Loop state outputs + scan outputs
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:4698-4819`
  - Returns: `TupleWrapper` with N outputs

- **Scan** - Returns N outputs (variable, based on body outputs)
  - Outputs: Scan outputs from body graph
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:4609-4513`
  - Returns: `TupleWrapper` with N outputs

### 7. **Sequence Operations** (1 op)

- **SplitToSequence** - Returns N outputs (variable, based on split)
  - Outputs: Sequence of split tensors
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:6749-6843`
  - Returns: `Tuple` or `TupleWrapper` with N outputs

### 8. **Unique Operations** (1 op)

- **Unique** - Returns 4 outputs
  - Outputs: `[unique_values, indices, inverse_indices, counts]`
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:5985-6015`
  - Returns: `TupleWrapper` with 4 outputs

### 9. **Loss Operations** (1 op)

- **NegativeLogLikelihoodLoss** - Returns 2 outputs (optional)
  - Outputs: `[loss, log_softmax]` (log_softmax is optional)
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:6213-6323`
  - Returns: `TupleWrapper` with 2 outputs (if log_softmax requested)

### 10. **Optional Operations** (1 op)

- **OptionalGetElement** - Returns 1 or 2 outputs (depending on optional type)
  - Outputs: Extracted element from optional
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:2334-2340`
  - Returns: Single output or `TupleGetItem`

### 11. **MaxPool** (1 op - conditional)

- **MaxPool** - Can return 2 outputs (if indices requested)
  - Outputs: `[output, indices]` (indices are optional)
  - Note: TVM implementation typically returns single output, but ONNX spec allows indices
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:1915-1919`
  - Returns: Single output (indices not typically returned in TVM)

### 12. **NonMaxSuppression** (1 op)

- **NonMaxSuppression** - Returns 1 output (but internally uses multi-output operations)
  - Outputs: Selected indices
  - Note: Internally uses operations that return multiple outputs, but final output is single
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:4972-5016`
  - Returns: Single output (but uses multi-output ops internally)

### 13. **Random Operations** (multiple ops - internal use)

- **RandomUniformLike** - Returns 2 outputs internally (key, values)
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:6167-6194`
  - Returns: Single output (extracts values from TupleWrapper)

- **Multinomial** - Returns 2 outputs internally (key, indices)
  - Implementation: `third_party/tvm/python/tvm/relay/frontend/onnx.py:6197-6210`
  - Returns: Single output (extracts indices from TupleWrapper)

---

## Complete List (Alphabetical)

1. **Attention** - 2 outputs
2. **BatchNormalization** - 5 outputs
3. **EmbedLayerNormalization** - 2 outputs
4. **GRU** - 2 outputs
5. **If** - N outputs (variable)
6. **LayerNormalization** - 3 outputs
7. **Loop** - N outputs (variable)
8. **LSTM** - 3 outputs
9. **NegativeLogLikelihoodLoss** - 2 outputs (optional)
10. **QAttention** - 2 outputs
11. **RNN** - 2 outputs
12. **Scan** - N outputs (variable)
13. **SkipLayerNormalization** - 3 outputs
14. **Split** - N outputs (variable)
15. **SplitToSequence** - N outputs (variable)
16. **TopK** - 2 outputs
17. **Unique** - 4 outputs

**Total: 17 operations** (excluding internal-only multi-output operations)

---

## Operations with Optional Multiple Outputs

Some operations can return multiple outputs depending on configuration:

- **Dropout** - Can return 2 outputs (output, mask) but TVM typically returns 1
- **MaxPool** - Can return 2 outputs (output, indices) but TVM typically returns 1

---

## Implementation Notes

### TVM Handling
- TVM uses `TupleWrapper` to handle multiple outputs
- The `_construct_nodes` method in `GraphProto` (lines 7337-7371) handles multi-output operations:
  ```python
  if outputs_num > 1:
      # Handle optional outputs
      valid_outputs = [False] * outputs_num
      for i, output in enumerate(node_output):
          if output != "":
              valid_outputs[i] = True
      # Extract only valid outputs
      outputs = [op[i] for i, valid in enumerate(valid_outputs) if valid]
  ```

### Forge Constraint
- Forge operations only support **single output per node**
- Our transpiler must split multi-output ONNX operations into multiple TIR nodes
- Each TIR node should produce a single output

### Recommended Approach
1. Use `_get_used_outputs()` to determine which outputs are actually used
2. Create separate TIR nodes for each used output
3. Name nodes appropriately (e.g., `Split_output_0`, `TopK_values`, `TopK_indices`)
4. Map each output to appropriate Forge operations (e.g., Split → Index operations)

---

## References

- TVM ONNX Frontend: `third_party/tvm/python/tvm/relay/frontend/onnx.py`
- ONNX Operator Documentation: https://onnx.ai/onnx/operators/
- TVM GraphProto Implementation: Lines 7303-7371

