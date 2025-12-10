"""
Reduction operations: ReduceSum, ReduceMean, ReduceMax
"""
import torch
from typing import Dict, List, Optional, Union, Tuple

from forge.transpiler.ir.nodes import TIRNode
from forge.transpiler.ir.types import TensorInfo



class ReduceSumNode(TIRNode):
    """
    PyTorch-like ReduceSum operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: Union[int, Tuple[int, ...], None] = None,
               keepdim: bool = False) -> 'ReduceSumNode':
        """Static factory method to create a ReduceSumNode."""
        return ReduceSumNode(
            name=name,
            op_type="ReduceSum",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim, 'keepdim': keepdim},
            forge_op_function_name="forge.op.ReduceSum"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch attrs to Forge attrs. Forge ReduceSum takes dim as int (not tuple)."""
        forge_attrs = {}
        if 'dim' in attrs:
            dim = attrs['dim']
            # Forge ReduceSum takes dim as int (single dim)
            if isinstance(dim, (tuple, list)):
                forge_attrs["dim"] = dim[0] if len(dim) > 0 else 0
            else:
                forge_attrs["dim"] = dim
        if 'keepdim' in attrs:
            forge_attrs["keep_dim"] = attrs["keepdim"]
        return forge_attrs

    def eval(self, input_tensors):
        """
        Evaluate ReduceSum operation.
        Matches PyTorch semantics: torch.sum(x, dim=dim, keepdim=keepdim)
        PyTorch handles dim=None with keepdim=True correctly, returning shape with all dims as size 1.
        """
        x = input_tensors[self.inputs[0]]
        dim = self.attrs.get('dim', None)
        # Ensure keepdim is bool (might be int from ONNX)
        keepdim = bool(self.attrs.get('keepdim', False))
        return {self.outputs[0]: torch.sum(x, dim=dim, keepdim=keepdim)}



class ReduceMeanNode(TIRNode):
    """
    PyTorch-like ReduceMean operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: Union[int, Tuple[int, ...], None] = None,
               keepdim: bool = False) -> 'ReduceMeanNode':
        """Static factory method to create a ReduceMeanNode."""
        return ReduceMeanNode(
            name=name,
            op_type="ReduceMean",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim, 'keepdim': keepdim},
            forge_op_function_name="forge.op.ReduceAvg"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch attrs to Forge attrs. Forge ReduceAvg takes dim as int (not tuple)."""
        forge_attrs = {}
        if 'dim' in attrs:
            dim = attrs['dim']
            # Forge ReduceAvg takes dim as int (single dim)
            if isinstance(dim, (tuple, list)):
                forge_attrs["dim"] = dim[0] if len(dim) > 0 else 0
            else:
                forge_attrs["dim"] = dim
        if 'keepdim' in attrs:
            forge_attrs["keep_dim"] = attrs["keepdim"]
        return forge_attrs

    def eval(self, input_tensors):
        """
        Evaluate ReduceMean operation.
        Matches PyTorch semantics: torch.mean(x, dim=dim, keepdim=keepdim)
        PyTorch handles dim=None with keepdim=True correctly, returning shape with all dims as size 1.
        """
        x = input_tensors[self.inputs[0]]
        dim = self.attrs.get('dim', None)
        # Ensure keepdim is bool (might be int from ONNX)
        keepdim = bool(self.attrs.get('keepdim', False))
        return {self.outputs[0]: torch.mean(x, dim=dim, keepdim=keepdim)}



class ReduceMaxNode(TIRNode):
    """
    PyTorch-like ReduceMax operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: Union[int, Tuple[int, ...], None] = None,
               keepdim: bool = False) -> 'ReduceMaxNode':
        """Static factory method to create a ReduceMaxNode."""
        return ReduceMaxNode(
            name=name,
            op_type="ReduceMax",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim, 'keepdim': keepdim},
            forge_op_function_name="forge.op.ReduceMax"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch attrs to Forge attrs. Forge ReduceMax takes dim as int (not tuple)."""
        forge_attrs = {}
        if 'dim' in attrs:
            dim = attrs['dim']
            # Forge ReduceMax takes dim as int (single dim)
            if isinstance(dim, (tuple, list)):
                forge_attrs["dim"] = dim[0] if len(dim) > 0 else 0
            else:
                forge_attrs["dim"] = dim
        if 'keepdim' in attrs:
            forge_attrs["keep_dim"] = attrs["keepdim"]
        return forge_attrs

    def eval(self, input_tensors):
        """
        Evaluate ReduceMax operation.
        Uses torch.amax() which handles all cases:
        - dim=None: reduces over all dimensions
        - dim=int: reduces over single dimension
        - dim=tuple: reduces over multiple dimensions
        - keepdim: preserves reduced dimensions when True
        """
        x = input_tensors[self.inputs[0]]
        dim = self.attrs.get('dim', None)
        # Ensure keepdim is bool (might be int from ONNX)
        keepdim = bool(self.attrs.get('keepdim', False))
        return {self.outputs[0]: torch.amax(x, dim=dim, keepdim=keepdim)}

