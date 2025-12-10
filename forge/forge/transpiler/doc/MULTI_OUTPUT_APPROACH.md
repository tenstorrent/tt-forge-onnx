# Multi-Output Support with Pattern-Based Decomposition

## Overview

This document describes the approach for handling multi-output ONNX operations in our transpiler. We support multiple outputs in TIR and use pattern-based decomposition (similar to TVM) to transform operations without direct Forge equivalents.

## Architecture

### 1. TIRNode Supports Multiple Outputs

**Revert Changes:**
- Change `TIRNode.__init__`: `output: str` → `outputs: List[str]`
- Update `emit()` to handle multiple outputs
- Update `TIRGraph.add_node()` to handle multiple outputs
- Update all operation nodes to use `outputs: List[str]`

**Key Points:**
- TIRNode can represent operations with multiple outputs (e.g., Split, TopK)
- This is more natural for representing ONNX operations
- Forge constraint (single output) is enforced during code generation via decomposition

### 2. Direct ONNX Split → SplitNode Mapping (PyTorch-like)

**Map ONNX Split to SplitNode (similar to torch.split):**
- ONNX Split with outputs `[out1, out2, out3]` → Create **one SplitNode**
- SplitNode represents the split operation (similar to `torch.split()` which returns a tuple)
- SplitNode has `outputs: List[str]` with all output names
- SplitNode has `forge_op_function_name = "UNKNOWN"` (no direct Forge equivalent)

**SplitNode Structure:**
```python
class SplitNode(TIRNode):
    """
    PyTorch-like Split operation.
    Similar to torch.split() which returns a tuple of tensors.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],  # Multiple outputs
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               split_sizes: List[int] = None,
               dim: int = 0) -> 'SplitNode':
        """
        Create a SplitNode representing a split operation.
        
        Args:
            outputs: List of all output names from the split
            output_tensors: Dict mapping all output names to TensorInfo
        """
        return SplitNode(
            name=name,
            op_type="Split",
            inputs=inputs,
            outputs=outputs,  # Multiple outputs
            input_tensors=input_tensors,
            output_tensors=output_tensors,  # All outputs
            attrs={
                'split_sizes': split_sizes,
                'dim': dim,
            },
            forge_op_function_name="UNKNOWN"  # Must be decomposed
        )
    
    def eval(self, input_tensors):
        """Evaluate split operation, returns dict with all outputs."""
        x = input_tensors[self.inputs[0]]
        split_sizes = self.attrs.get('split_sizes', None)
        dim = self.attrs.get('dim', 0)
        
        # Perform split (similar to torch.split)
        if split_sizes is not None:
            if isinstance(split_sizes, list):
                split_sizes = tuple(split_sizes)
            splits = torch.split(x, split_sizes, dim=dim)
        else:
            # Equal split
            dim_size = x.shape[dim] if dim < len(x.shape) else x.shape[0]
            num_outputs = len(self.outputs)
            split_size = dim_size // num_outputs
            splits = torch.split(x, split_size, dim=dim)
        
        # Return all outputs
        return {output_name: splits[i] for i, output_name in enumerate(self.outputs)}
```

**ONNX Converter:**
```python
def _convert_split(self, node_proto, input_tensors, output_tensors, attrs, node_index):
    """Convert ONNX Split to SplitNode (PyTorch-like)."""
    node_name = node_proto.name if node_proto.name else f"Split_{node_index}"
    
    split_sizes = attrs.get('split', None)
    axis = attrs.get('axis', 0)
    
    # Create one SplitNode representing the entire split operation
    split_node = SplitNode.create(
        name=node_name,
        inputs=list(node_proto.input),
        outputs=list(node_proto.output),  # All outputs
        input_tensors=input_tensors,
        output_tensors=output_tensors,  # All outputs
        split_sizes=split_sizes,
        dim=axis
    )
    
    return [split_node]  # Single node, but with multiple outputs
```

### 3. UNKNOWN Operations

**For operations without direct Forge equivalents:**
- Set `forge_op_function_name = "UNKNOWN"`
- Examples: `SplitNode`, `TopKNode` (initially)

**Emit Check:**
```python
def emit(self) -> Dict[str, Any]:
    """
    Returns a dictionary describing the operation for code generation.
    """
    if self.forge_op_function_name == "UNKNOWN":
        raise NotImplementedError(
            f"Operation {self.op_type} has no direct Forge equivalent. "
            f"It must be decomposed using pattern callbacks before code generation. "
            f"Please run pattern callbacks (e.g., LowerSplitToStridedSlice) to decompose "
            f"this operation into Forge-compatible operations."
        )
    
    # For multi-output nodes, emit the first output (or handle appropriately)
    # Note: After pattern callbacks, all nodes should be single-output
    return {
        "function_name": self.forge_op_function_name,
        "node_name": self.name,
        "output_name": self.outputs[0] if len(self.outputs) > 0 else None,
        "output_names": self.outputs,  # Include all outputs
        "input_names": self.inputs,
        "input_shapes": [info.shape if info.shape else [] for info in self.input_tensors.values()],
        "input_dtypes": [info.torch_dtype if info.torch_dtype else None for info in self.input_tensors.values()],
        "args": self.forge_attrs,
    }
```

### 4. TIRGraph Updates

**Update `add_node()` to handle multiple outputs:**
```python
def add_node(self, node: TIRNode):
    """Add a node to the graph and update topology maps."""
    self.nodes.append(node)
    
    # Handle multiple outputs
    for output_name in node.outputs:
        self.producer_map[output_name] = node.name
    
    for in_name in node.inputs:
        if in_name not in self.consumer_map:
            self.consumer_map[in_name] = []
        self.consumer_map[in_name].append(node.name)
```

### 5. Pattern Callback System (To Be Implemented Later)

**Similar to TVM's DFPatternCallback:**
- Create `TIRPatternCallback` base class
- Pattern matching on TIRGraph structure
- Transform UNKNOWN nodes into known Forge operations

**Pattern Matching Information:**
The pattern matcher will check:
1. **Node Type**: Class type (e.g., `SplitNode`, `StridedSliceNode`)
2. **Op Type**: `node.op_type` (e.g., "Split", "StridedSlice")
3. **Forge Op Function Name**: `node.forge_op_function_name == "UNKNOWN"`
4. **Attributes**: `node.attrs` (e.g., `attrs.get('dim')`, `attrs.get('split_sizes')`)
5. **Shapes**: `node.input_tensors[name].shape`, `node.output_tensors[name].shape`
6. **Dtypes**: `node.input_tensors[name].torch_dtype`, `node.output_tensors[name].torch_dtype`
7. **Input Tensor Info**: `node.input_tensors` - full TensorInfo objects
8. **Output Tensor Info**: `node.output_tensors` - full TensorInfo objects
9. **Graph Structure**: `graph.consumer_map`, `graph.producer_map`

**Structure (To Be Implemented):**
```python
class TIRPatternCallback:
    """Base class for TIR graph pattern callbacks."""
    
    def match(self, graph: TIRGraph, node: TIRNode) -> bool:
        """
        Check if node matches pattern.
        """
        raise NotImplementedError
    
    def callback(self, graph: TIRGraph, node: TIRNode) -> List[TIRNode]:
        """
        Transform matched node into new nodes.
        Returns list of replacement nodes.
        """
        raise NotImplementedError
```

**Example: LowerSplitToStridedSlice (To Be Implemented Later)**
- Matches `SplitNode` with `forge_op_function_name == "UNKNOWN"`
- Transforms into multiple `StridedSliceNode` (one per output)
- Each `StridedSliceNode` maps to `forge.op.tm.Index`

## Flow Diagram

```
ONNX Split (outputs: [out1, out2, out3])
    ↓
SplitNode (outputs: [out1, out2, out3], forge_op_function_name="UNKNOWN")
    ↓
Code Generation Attempt → Error: "Must decompose using pattern callbacks"
    ↓
[Future] Pattern Callback: LowerSplitToStridedSlice
    ↓
[Future] StridedSliceNode (output: out1, forge.op.tm.Index)
[Future] StridedSliceNode (output: out2, forge.op.tm.Index)
[Future] StridedSliceNode (output: out3, forge.op.tm.Index)
    ↓
[Future] Code Generation: forge.op.tm.Index(...)
```

## Implementation Steps (Current Phase)

### Phase 1: Revert to Multiple Outputs (Now)

1. **Revert TIRNode to support multiple outputs**
   - Change `output: str` → `outputs: List[str]`
   - Update `__init__` signature
   - Update `emit()` to handle multiple outputs

2. **Update SplitNode**
   - Change to `outputs: List[str]`
   - Set `forge_op_function_name = "UNKNOWN"`
   - Update `eval()` to return all outputs
   - Update `create()` method signature

3. **Update TIRGraph**
   - Update `add_node()` to handle multiple outputs
   - Update `producer_map` to map each output to producer node

4. **Update ONNX Engine**
   - Update `_convert_split()` to pass `outputs=list(node_proto.output)`
   - Update all converter methods to use `outputs` instead of `output`

5. **Update all operation nodes**
   - Change `output: str` → `outputs: List[str]`
   - Update `create()` methods
   - Update `eval()` methods to return `{outputs[0]: ...}` or handle multiple

6. **Update code generator**
   - Handle multiple outputs in graph structure
   - Check for UNKNOWN operations and raise error

7. **Add UNKNOWN check in emit()**
   - Check if `forge_op_function_name == "UNKNOWN"`
   - Raise descriptive error with instructions

### Phase 2: Pattern Callbacks (Later)

8. **Create TIRPatternCallback base class**
9. **Create StridedSliceNode**
10. **Implement LowerSplitToStridedSlice callback**
11. **Create pattern callback runner**
12. **Integrate pattern callbacks into code generation pipeline**

## Benefits

1. **Natural Representation**: TIRNode can represent ONNX operations as they are (multi-output)
2. **Clear Error Messages**: UNKNOWN operations fail with helpful error messages
3. **Flexible Decomposition**: Pattern callbacks (when implemented) allow flexible transformation
4. **Extensible**: Easy to add new decomposition patterns later
5. **Clear Separation**: ONNX → TIR (multi-output, some UNKNOWN) → Pattern Callbacks → TIR (single-output, all known) → Forge Code

## Current Status

- ✅ Design documented
- ⏳ Phase 1: Revert to multiple outputs (in progress)
- ⏸️ Phase 2: Pattern callbacks (to be implemented later)
