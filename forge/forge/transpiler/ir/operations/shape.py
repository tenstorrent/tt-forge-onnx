"""
Shape/Reshape operations: Flatten, Reshape, Transpose, Squeeze, Unsqueeze
"""
import torch
from typing import Dict, List, Union

from forge.transpiler.ir.nodes import TIRNode
from forge.transpiler.ir.types import TensorInfo



class ReshapeNode(TIRNode):
    """
    PyTorch-like Reshape operation.
    Takes one input tensor and shape as attribute (matching torch.reshape API).
    Supports ONNX Reshape features:
    - -1 for inferred dimension
    - Shape is already resolved in converter (no 0 or empty shapes)
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               shape: tuple) -> 'ReshapeNode':
        """
        Static factory method to create a ReshapeNode.
        
        Args:
            name: Node name
            inputs: List of input tensor names
            outputs: List of output tensor names
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            shape: Target shape tuple (already resolved, may contain -1 for inference)
        """
        return ReshapeNode(
            name=name,
            op_type="Reshape",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'shape': shape},
            forge_op_function_name="forge.op.tm.Reshape"
        )
    
    def eval(self, input_tensors):
        """
        Evaluate Reshape operation.
        
        Note: Shape should already be resolved in the converter.
        torch.reshape supports -1 for inferred dimension.
        """
        x = input_tensors[self.inputs[0]]
        shape = self.attrs.get('shape', None)
        
        if shape is None:
            # Fallback: use output shape from TensorInfo
            output_info = list(self.output_tensors.values())[0]
            if output_info and output_info.shape:
                shape = tuple(s if s is not None else x.shape[i] 
                            for i, s in enumerate(output_info.shape))
            else:
                shape = x.shape
        
        # Normal case: shape is already resolved (no 0, no empty shapes)
        # torch.reshape supports -1, but we should have resolved it in converter
        # However, if -1 somehow remains, torch.reshape will handle it
        try:
            result = torch.reshape(x, shape)
        except RuntimeError as e:
            raise ValueError(f"Reshape failed: {e}. Input shape: {x.shape}, Target shape: {shape}")
        
        return {self.outputs[0]: result}



class TransposeNode(TIRNode):
    """
    PyTorch-like Transpose operation that swaps two dimensions.
    For multi-dimensional transpositions, create multiple TransposeNode instances.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim0: int, dim1: int) -> 'TransposeNode':
        """Static factory method to create a TransposeNode."""
        return TransposeNode(
            name=name,
            op_type="Transpose",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim0': dim0, 'dim1': dim1},
            forge_op_function_name="forge.op.tm.Transpose"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        dim0 = self.attrs['dim0']
        dim1 = self.attrs['dim1']
        return {self.outputs[0]: torch.transpose(x, dim0, dim1)}



class SqueezeNode(TIRNode):
    """
    PyTorch-like Squeeze operation.
    Takes one input tensor and dim as attribute (matching torch.squeeze API).
    dim can be int or tuple/list of ints (torch.squeeze accepts both).
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: Union[int, tuple, list]) -> 'SqueezeNode':
        """Static factory method to create a SqueezeNode."""
        return SqueezeNode(
            name=name,
            op_type="Squeeze",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim},
            forge_op_function_name="forge.op.tm.Squeeze"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch attrs to Forge attrs. Forge Squeeze requires dim as int (not tuple)."""
        forge_attrs = {}
        if 'dim' in attrs:
            dim = attrs['dim']
            if isinstance(dim, (list, tuple)):
                # If multiple dims, use first one (Forge limitation)
                forge_attrs['dim'] = dim[0] if len(dim) > 0 else 0
            else:
                forge_attrs['dim'] = dim
        return forge_attrs

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        dim = self.attrs.get('dim', None)
        if dim is not None:
            # torch.squeeze accepts int or tuple of ints
            # Convert list to tuple for consistency
            if isinstance(dim, list):
                dim = tuple(dim)
            # torch.squeeze handles both int and tuple directly
            return {self.outputs[0]: torch.squeeze(x, dim=dim)}
        else:
            return {self.outputs[0]: torch.squeeze(x)}


class UnsqueezeNode(TIRNode):
    """
    PyTorch-like Unsqueeze operation.
    Takes one input tensor and dim as attribute (matching torch.unsqueeze API).
    Forge Unsqueeze requires dim as attribute (single int, required).
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: int) -> 'UnsqueezeNode':
        """Static factory method to create an UnsqueezeNode."""
        return UnsqueezeNode(
            name=name,
            op_type="Unsqueeze",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim},
            forge_op_function_name="forge.op.tm.Unsqueeze"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs. 
        Forge Unsqueeze requires dim as int (required, no default).
        
        Raises:
            ValueError: If dim is None or not an int
        """
        forge_attrs = {}
        if 'dim' not in attrs or attrs['dim'] is None:
            raise ValueError(
                f"UnsqueezeNode '{self.name}': 'dim' attribute is required and cannot be None"
            )
        
        dim = attrs['dim']
        if not isinstance(dim, int):
            raise ValueError(
                f"UnsqueezeNode '{self.name}': 'dim' must be an int, got {type(dim).__name__}"
            )
        
        forge_attrs['dim'] = dim
        return forge_attrs

    def eval(self, input_tensors):
        """
        Evaluate Unsqueeze operation.
        
        Raises:
            ValueError: If dim is None or not an int
        """
        x = input_tensors[self.inputs[0]]
        dim = self.attrs.get('dim', None)
        
        if dim is None:
            raise ValueError(
                f"UnsqueezeNode '{self.name}': 'dim' attribute is required and cannot be None"
            )
        
        if not isinstance(dim, int):
            raise ValueError(
                f"UnsqueezeNode '{self.name}': 'dim' must be an int, got {type(dim).__name__}"
            )
        
        return {self.outputs[0]: torch.unsqueeze(x, dim=dim)}


class SplitNode(TIRNode):
    """
    PyTorch-like Split operation.
    Similar to torch.split() which returns a tuple of tensors.
    Represents a split operation that produces multiple outputs.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               split_sizes: List[int] = None,
               dim: int = 0) -> 'SplitNode':
        """
        Static factory method to create a SplitNode.
        
        Args:
            name: Node name
            inputs: List of input tensor names (single input)
            outputs: List of output tensor names (multiple outputs)
            input_tensors: Input tensor metadata
            output_tensors: Output tensor metadata (all outputs)
            split_sizes: List of sizes for each split (e.g., [2, 3, 5])
            dim: Dimension along which to split
        """
        return SplitNode(
            name=name,
            op_type="Split",
            inputs=inputs,
            outputs=outputs,  # Multiple outputs
            input_tensors=input_tensors,
            output_tensors=output_tensors,  # All outputs
            attrs={'split_sizes': split_sizes, 'dim': dim},
            forge_op_function_name="UNKNOWN"  # Must be decomposed via pattern callbacks
        )

    def eval(self, input_tensors):
        """Evaluate split operation, returns dict with all outputs."""
        x = input_tensors[self.inputs[0]]
        split_sizes = self.attrs.get('split_sizes', None)
        dim = self.attrs.get('dim', 0)
        
        # Perform split (similar to torch.split)
        if split_sizes is not None:
            # Convert to tuple if list
            if isinstance(split_sizes, list):
                split_sizes = tuple(split_sizes)
            splits = torch.split(x, split_sizes, dim=dim)
        else:
            # Equal split - divide evenly
            dim_size = x.shape[dim] if dim < len(x.shape) else x.shape[0]
            num_outputs = len(self.outputs)
            split_size = dim_size // num_outputs
            splits = torch.split(x, split_size, dim=dim)
        
        # Return all outputs
        result = {}
        for i, output_name in enumerate(self.outputs):
            if i < len(splits):
                result[output_name] = splits[i]
            else:
                # Fallback: use last split if index out of range
                result[output_name] = splits[-1]
        return result

