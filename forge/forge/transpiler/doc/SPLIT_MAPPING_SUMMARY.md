# ONNX Split → TVM Relay → Forge Index Mapping Summary

This document summarizes how ONNX Split operations are mapped through the TVM Relay pipeline to Forge Index operations.

## Overview

The mapping follows this pipeline:
1. **ONNX Split** → **TVM Relay Split** (ONNX Frontend)
2. **TVM Relay Split** → **StridedSlice** (LowerSplitToStridedSlice Pass)
3. **StridedSlice** → **Forge Index** (tvm_to_python.py)

---

## Stage 1: ONNX Split → TVM Relay Split

**Location**: `third_party/tvm/python/tvm/relay/frontend/onnx.py` (lines 2604-2648)

### Implementation Details

**ONNX Split Converter Class**: `Split(OnnxOpConverter)`

#### Opset v1 (lines 2608-2623):
```python
@classmethod
def _impl_v1(cls, inputs, attr, params):
    splits = attr.get("split", None)
    if splits is not None and len(splits) > 1:
        # Convert split sizes to cumulative indices
        indices = []
        index = 0
        for i in splits[:-1]:  # All but last
            index += i
            indices.append(index)
    else:
        # Equal split - use num_outputs
        indices = attr["tvm_custom"]["num_outputs"]
    
    # Create TVM Relay split operation
    output = _op.split(inputs[0], indices, attr.get("axis", 0))
    
    # Unpack if single output
    if len(output) == 1:
        output = output[0]
    return output
```

#### Opset v13+ (lines 2626-2648):
- Similar logic but `splits` comes from `inputs[1]` (as a tensor) instead of attribute
- Requires constant tensor for splits (dynamic splits not supported)
- Converts tensor to numpy array, then computes cumulative indices

### Key Points:
- **Input**: ONNX Split node with `split` attribute (list of sizes) or equal splits
- **Output**: TVM Relay `split` operation returning a `TupleWrapper` with multiple outputs
- **Indices**: Cumulative split points (e.g., splits `[2, 3, 5]` → indices `[2, 5]`)
- **Axis**: Dimension along which to split (default: 0)

---

## Stage 2: TVM Relay Split → StridedSlice

**Location**: `forge/forge/tvm_calls/relay/op/forge_passes.py` (lines 933-969)

### Implementation Details

**Pass Class**: `LowerSplitToStridedSlice(DFPatternCallback)`

#### Pattern Matching:
```python
def __init__(self):
    super().__init__(rewrite_once=True, require_type=True)
    self.act = wildcard()
    self.split = is_op("split")(self.act)
    self.pattern = is_tuple_get_item(wildcard())  # Matches tuple_get_item(split(...))
```

#### Conversion Logic (lines 941-969):
```python
def callback(self, pre, post, node_map):
    split = post.tuple_value().op  # Get the split operation
    
    if not self.split.match(split):
        return post
    
    act = split.args[0]  # Input tensor
    act_shape = pre.tuple_value().op.args[0].checked_type.shape
    axis = split.attrs.axis
    if axis < 0:
        axis += len(act_shape)
    
    # Compute split indices
    if isinstance(split.attrs.indices_or_sections, tvm.tir.expr.IntImm):
        # Equal split case
        sections = int(split.attrs.indices_or_sections)
        total_length = int(act_shape[axis])
        ios = list(range(total_length // sections, total_length, total_length // sections))
    else:
        # Specified split sizes case
        ios = [int(dim) for dim in split.attrs.indices_or_sections]
    ios.append(act_shape[axis])  # Add final index (end of tensor)
    
    # Calculate begin/end for this specific output
    begin = 0 if post.index == 0 else ios[post.index - 1]
    end = ios[post.index]
    
    # Optimization: skip if slice covers entire dimension
    if end - begin == act_shape[axis]:
        return act
    
    # Create strided_slice operation
    sliced_act = tvm.relay.strided_slice(act, (begin,), (end,), axes=(axis,))
    return sliced_act
```

### Key Points:
- **Pattern**: Matches `tuple_get_item(split(...), index)` - each output accessed separately
- **Conversion**: Each `tuple_get_item` is converted to a `strided_slice` with calculated `begin`/`end`
- **Indices Calculation**: 
  - For equal splits: `ios = [dim//sections, 2*dim//sections, ...]`
  - For specified splits: Uses the cumulative indices from split operation
- **Optimization**: Skips strided_slice if it would slice the entire dimension
- **Pass Order**: Applied in `forge_passes.py` line 4884, after `LiftLinearSplit` and before `PopulateStridedSliceAxes`

---

## Stage 3: StridedSlice → Forge Index

**Location**: `forge/forge/tvm_to_python.py`

### Mapping Configuration

#### Op Name Mapping (line 1546):
```python
tvm_op_to_forge_op = {
    ...
    "strided_slice": "index",
    ...
}
```

#### Function Name Mapping (line 1588):
```python
forge_op_to_function_name = {
    ...
    "index": "forge.op.Index",
    ...
}
```

#### Argument Population Function (lines 918-964):
```python
def populate_index_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    strides = [int(strides) for strides in node["attrs"]["strides"][0]]
    begin = [int(begin) for begin in node["attrs"]["begin"][0]]
    end = [int(e) for e in node["attrs"]["end"][0]]
    
    assert len(strides) == 1 and len(begin) == 1 and len(end) == 1, \
        "Strided slice should be on a single axis"
    assert int(node["attrs"]["num_inputs"]) == 1
    
    assert len(list(node["attrs"]["axes"][0])) == 1, "Select can only have 1 axis"
    dim = int(node["attrs"]["axes"][0][0])
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]
    
    # Use negative indexing
    if dim >= 0:
        dim -= len(input_shape)
    
    # Handle TVM's max int32 sentinel for "to the end"
    if end[0] == (2**31 - 1):
        end[0] = node["attrs"]["shape"][0][0][dim]
    
    args = [
        ("dim", f"{dim}"),
        ("start", f"{begin[0]}"),
        ("stop", f"{end[0]}"),
        ("stride", f"{strides[0]}"),
    ]
    return args
```

#### Function Registration (line 1649):
```python
forge_op_to_populate_args = {
    ...
    "index": populate_index_args,
    ...
}
```

### Forge Index Operation

**Location**: `forge/forge/op/tm.py` (lines 88-151)

```python
def Index(name: str, operandA: Tensor, dim: int, start: int, stop: int = None, stride: int = 1) -> Tensor:
    """
    TM (Tensor Manipulation) operation for slicing tensors.
    
    Parameters:
    - dim: Dimension to slice
    - start: Starting slice index (inclusive)
    - stop: Stopping slice index (exclusive)
    - stride: Stride amount along that dimension
    """
    if dim < 0:
        dim += len(operandA.shape)
    
    if stop is None:
        stop = start + 1
    
    if stop < 0:
        stop += operandA.shape[dim]
    
    # Handle negative stride (limited support)
    if stride < 0:
        assert operandA.shape[dim] == 1, "Negative stride only for size 1 dims"
        stride = abs(stride)
    
    assert stride > 0
    assert start < operandA.shape[dim]
    assert stop <= operandA.shape[dim]
    assert stride <= operandA.shape[dim]
    
    return op(OpType.Index, name, operandA, dim=dim, start=start, stop=stop, stride=stride).get_tensor()
```

### Key Points:
- **Single Axis**: StridedSlice must operate on a single axis (enforced by assertion)
- **Attribute Extraction**: Extracts `dim`, `start`, `stop`, `stride` from TVM strided_slice attributes
- **Negative Indexing**: Converts positive dims to negative indexing (Forge convention)
- **Sentinel Handling**: Handles TVM's max int32 sentinel (2^31 - 1) as "to the end"
- **Final Output**: Generates `forge.op.tm.Index(name, operandA, dim=dim, start=start, stop=stop, stride=stride)`

---

## Complete Example Flow

### Example: ONNX Split with splits=[2, 3, 5] on axis=0

1. **ONNX Split**:
   - Input shape: `[10, 4, 4]`
   - Splits: `[2, 3, 5]`
   - Axis: `0`
   - Outputs: 3 tensors with shapes `[2, 4, 4]`, `[3, 4, 4]`, `[5, 4, 4]`

2. **TVM Relay Split**:
   - Converts splits to indices: `[2, 5]` (cumulative: 2, 2+3=5)
   - Creates: `split(input, indices=[2, 5], axis=0)`
   - Returns: `TupleWrapper` with 3 outputs

3. **LowerSplitToStridedSlice** (for each output):
   - **Output 0**: `begin=0`, `end=2` → `strided_slice(act, (0,), (2,), axes=(0,))`
   - **Output 1**: `begin=2`, `end=5` → `strided_slice(act, (2,), (5,), axes=(0,))`
   - **Output 2**: `begin=5`, `end=10` → `strided_slice(act, (5,), (10,), axes=(0,))`

4. **Forge Index** (for each strided_slice):
   - **Output 0**: `forge.op.tm.Index(name, operandA, dim=-3, start=0, stop=2, stride=1)`
   - **Output 1**: `forge.op.tm.Index(name, operandA, dim=-3, start=2, stop=5, stride=1)`
   - **Output 2**: `forge.op.tm.Index(name, operandA, dim=-3, start=5, stop=10, stride=1)`

---

## Implications for Our Transpiler

Based on this analysis, our current `SplitNode` implementation should:

1. **Match TVM's Approach**: Create separate TIR nodes for each used output (✅ Already implemented)
2. **Use Index Operation**: Instead of `forge.op.tm.Split`, we should map to `forge.op.tm.Index` (❌ Currently uses Split)
3. **Calculate Begin/End**: For each split output, calculate the begin/end indices based on split sizes
4. **Handle Equal Splits**: When split sizes are not specified, divide evenly

### Recommended Change

Update `SplitNode` to emit `forge.op.tm.Index` instead of `forge.op.tm.Split`, with:
- `dim`: The split axis
- `start`: Beginning index for this split (calculated from split_index and split_sizes)
- `stop`: Ending index for this split
- `stride`: Always 1 for split operations

This would align our transpiler with the TVM→Forge pipeline and ensure consistency.


