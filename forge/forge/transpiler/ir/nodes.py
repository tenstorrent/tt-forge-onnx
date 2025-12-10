"""
Base node class and operation registry for the transpiler IR.

Framework-agnostic - used by all frontends.
"""
import torch
from typing import Dict, List, Any
from forge.transpiler.ir.types import TensorInfo


class TIRNode:
    """
    Base class for all Transpiler Intermediate Representation nodes.
    Represents a node in the intermediate representation between ML frameworks 
    (e.g., ONNX, PaddlePaddle) and Forge module graphs.
    Framework-agnostic - operations are common across all frontends.
    
    TIRNodes are created with PyTorch-compatible attributes (attrs).
    The only conversion pipeline is: attrs -> forge_attrs.
    Framework-specific conversions (e.g., ONNX -> PyTorch) happen in the frontend converter.
    """
    def __init__(self, 
                 name: str, 
                 op_type: str,
                 inputs: List[str], 
                 outputs: List[str],  # Multiple output names
                 input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo],
                 attrs: Dict[str, Any],  # PyTorch-compatible attributes
                 forge_op_function_name: str = None,  # Forge operation function name
                 src_layer: str = None):  # Source layer name from original framework (e.g., ONNX node name)
        
        self.name = name
        self.op_type = op_type
        self.inputs = inputs # List of input names
        self.outputs = outputs  # List of output names (supports multiple outputs) - sanitized for code generation
        self.original_outputs = outputs.copy()  # Store original output names for debug/comparison (will be updated during sanitization)
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.attrs = attrs  # PyTorch-compatible attributes
        self.src_layer = src_layer  # Source layer name for debugging/tracking
        
        # Single conversion pipeline: attrs -> forge_attrs
        self.forge_attrs = self.convert_attrs_to_forge_attrs(self.attrs)
        
        # Set forge operation function name
        if forge_op_function_name is None:
            # Default: convert op_type to forge.op format
            op_type_lower = self.op_type.lower()
            if op_type_lower.startswith('conv'):
                self.forge_op_function_name = f"forge.op.convolution.{self.op_type}"
            elif op_type_lower in ['reshape', 'transpose', 'squeeze', 'unsqueeze']:
                self.forge_op_function_name = f"forge.op.tm.{self.op_type}"
            elif op_type_lower == 'subtract':
                self.forge_op_function_name = "forge.op.Subtract"
            elif op_type_lower == 'multiply':
                self.forge_op_function_name = "forge.op.Multiply"
            elif op_type_lower == 'divide':
                self.forge_op_function_name = "forge.op.Divide"
            elif op_type_lower == 'matmul':
                self.forge_op_function_name = "forge.op.Matmul"
            elif op_type_lower == 'concatenate' or op_type_lower == 'concat':
                self.forge_op_function_name = "forge.op.Concatenate"
            elif op_type_lower == 'batchnorm' or op_type_lower == 'batchnormalization':
                self.forge_op_function_name = "forge.op.Batchnorm"
            else:
                self.forge_op_function_name = f"forge.op.{self.op_type}"
        else:
            self.forge_op_function_name = forge_op_function_name

    def convert_attrs_to_forge_attrs(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert PyTorch attributes to Forge-specific attributes for code generation.
        This is the only attribute conversion pipeline in TIRNode.
        """
        return attrs.copy()

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute the node operation using PyTorch."""
        raise NotImplementedError("Subclasses must implement eval")

    def emit(self) -> Dict[str, Any]:
        """
        Returns a dictionary describing the operation for code generation.
        Matches the Operation class structure with keys:
        - function_name: Forge operation function name
        - node_name: Name of the node
        - output_name: Name of the output tensor (first output for compatibility)
        - output_names: List of all output tensor names
        - input_names: List of input tensor names
        - input_shapes: List of input tensor shapes
        - input_dtypes: List of input tensor dtypes
        - args: Dictionary of operation arguments
        """
        # Check for UNKNOWN operations
        if self.forge_op_function_name == "UNKNOWN":
            raise NotImplementedError(
                f"Operation {self.op_type} (node: {self.name}) has no direct Forge equivalent. "
                f"It must be decomposed using pattern callbacks before code generation. "
                f"Please run pattern callbacks (e.g., LowerSplitToStridedSlice) to decompose "
                f"this operation into Forge-compatible operations."
            )
        
        return {
            "function_name": self.forge_op_function_name,
            "node_name": self.name,
            "output_name": self.outputs[0] if len(self.outputs) > 0 else None,
            "output_names": self.outputs,  # Include all outputs
            "input_names": self.inputs,
            "input_shapes": [info.shape if info.shape else [] for info in self.input_tensors.values()],
            "input_dtypes": [info.torch_dtype if info.torch_dtype else None for info in self.input_tensors.values()],
            "args": self.forge_attrs,
            "src_layer": self.src_layer,  # Source layer name from original framework
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}' op_type='{self.op_type}'>"

