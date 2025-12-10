"""
Activation operations: Relu, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyRelu, Dropout
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from forge.transpiler.ir.nodes import TIRNode
from forge.transpiler.ir.types import TensorInfo


class ReluNode(TIRNode):
    """
    PyTorch-like Relu operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'ReluNode':
        """Static factory method to create a ReluNode."""
        return ReluNode(
            name=name,
            op_type="Relu",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Relu"
        )
    
    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        return {self.outputs[0]: F.relu(x)}



class SigmoidNode(TIRNode):
    """
    PyTorch-like Sigmoid operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'SigmoidNode':
        """Static factory method to create a SigmoidNode."""
        return SigmoidNode(
            name=name,
            op_type="Sigmoid",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Sigmoid"
        )
    
    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        return {self.outputs[0]: torch.sigmoid(x)}



class TanhNode(TIRNode):
    """
    PyTorch-like Tanh operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'TanhNode':
        """Static factory method to create a TanhNode."""
        return TanhNode(
            name=name,
            op_type="Tanh",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Tanh"
        )
    
    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        return {self.outputs[0]: F.tanh(x)}



class SoftmaxNode(TIRNode):
    """
    PyTorch-like Softmax operation.
    
    Similar to torch.softmax, the dim parameter must be explicitly provided.
    If dim is None, an error will be raised.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: Optional[int] = None) -> 'SoftmaxNode':
        """
        Static factory method to create a SoftmaxNode.
        
        Args:
            name: Node name
            inputs: List of input tensor names
            outputs: List of output tensor names
            input_tensors: Dictionary of input tensor info
            output_tensors: Dictionary of output tensor info
            dim: Dimension along which to apply softmax (must be provided, cannot be None)
        
        Raises:
            ValueError: If dim is None
        """
        if dim is None:
            raise ValueError("SoftmaxNode requires 'dim' parameter to be specified (cannot be None)")
        return SoftmaxNode(
            name=name,
            op_type="Softmax",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim},
            forge_op_function_name="forge.op.Softmax"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch attrs to Forge attrs, adding stable=True for Forge Softmax."""
        forge_attrs = attrs.copy()
        if 'stable' not in forge_attrs:
            forge_attrs['stable'] = True  # Default stable=True for Forge
        return forge_attrs

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        dim = self.attrs.get('dim', None)
        if dim is None:
            raise ValueError("SoftmaxNode requires 'dim' attribute to be set (cannot be None)")
        return {self.outputs[0]: F.softmax(x, dim=dim)}



class LogSoftmaxNode(TIRNode):
    """
    PyTorch-like LogSoftmax operation.
    
    Similar to torch.log_softmax, the dim parameter must be explicitly provided.
    If dim is None, an error will be raised.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: Optional[int] = None) -> 'LogSoftmaxNode':
        """
        Static factory method to create a LogSoftmaxNode.
        
        Args:
            name: Node name
            inputs: List of input tensor names
            outputs: List of output tensor names
            input_tensors: Dictionary of input tensor info
            output_tensors: Dictionary of output tensor info
            dim: Dimension along which to apply log_softmax (must be provided, cannot be None)
        
        Raises:
            ValueError: If dim is None
        """
        if dim is None:
            raise ValueError("LogSoftmaxNode requires 'dim' parameter to be specified (cannot be None)")
        return LogSoftmaxNode(
            name=name,
            op_type="LogSoftmax",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim},
            forge_op_function_name="forge.op.LogSoftmax"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch attrs to Forge attrs, adding stable=True for Forge LogSoftmax."""
        forge_attrs = attrs.copy()
        if 'stable' not in forge_attrs:
            forge_attrs['stable'] = True  # Default stable=True for Forge
        return forge_attrs

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        dim = self.attrs.get('dim', None)
        if dim is None:
            raise ValueError("LogSoftmaxNode requires 'dim' attribute to be set (cannot be None)")
        return {self.outputs[0]: F.log_softmax(x, dim=dim)}



class LeakyReluNode(TIRNode):
    """
    PyTorch-like LeakyRelu operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               negative_slope: float = 0.01) -> 'LeakyReluNode':
        """Static factory method to create a LeakyReluNode."""
        return LeakyReluNode(
            name=name,
            op_type="LeakyRelu",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'negative_slope': negative_slope},
            forge_op_function_name="forge.op.LeakyRelu"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch negative_slope to Forge alpha."""
        return {'alpha': attrs.get('negative_slope', 0.01)}

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        negative_slope = self.attrs.get('negative_slope', 0.01)
        return {self.outputs[0]: F.leaky_relu(x, negative_slope=negative_slope)}


class DropoutNode(TIRNode):
    """
    PyTorch-like Dropout operation.
    
    Supports different ONNX versions:
    - v1-v6: Uses `is_test` attribute and `ratio` attribute
    - v7-v10: Uses `ratio` attribute, training mode from graph context
    - v12+: Uses `ratio` and `training_mode` inputs, `seed` attribute
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               p: float = 0.5,
               training: bool = True,
               seed: int = 0) -> 'DropoutNode':
        """
        Static factory method to create a DropoutNode.
        
        Args:
            name: Node name
            inputs: List of input tensor names (data, [ratio], [training_mode])
            outputs: List of output tensor names (output, [mask])
            input_tensors: Dictionary of input tensor info
            output_tensors: Dictionary of output tensor info
            p: Dropout probability (default: 0.5)
            training: Training mode flag (default: True)
            seed: Random seed (default: 0)
        """
        return DropoutNode(
            name=name,
            op_type="Dropout",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'p': p, 'training': training, 'seed': seed},
            forge_op_function_name="forge.op.Dropout"
        )
    
    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch attrs to Forge attrs (p, training, seed)."""
        return {
            'p': attrs.get('p', 0.5),
            'training': attrs.get('training', True),
            'seed': attrs.get('seed', 0)
        }
    
    def eval(self, input_tensors):
        """Evaluate Dropout operation."""
        x = input_tensors[self.inputs[0]]
        p = self.attrs.get('p', 0.5)
        training = self.attrs.get('training', True)
        seed = self.attrs.get('seed', 0)
        
        # Set seed for reproducibility (always set, even if seed=0, to match C++ behavior)
        torch.manual_seed(seed)
        
        # Apply dropout
        if training:
            # Use F.dropout which applies dropout and scales by 1/(1-p)
            output = F.dropout(x, p=p, training=True)
        else:
            # In inference mode, just return the input
            output = x
        
        return {self.outputs[0]: output}

