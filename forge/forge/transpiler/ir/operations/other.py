"""
Other operations: Concat, Clip, Cast, Pad, Identity
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional

from forge.transpiler.ir.nodes import TIRNode
from forge.transpiler.ir.types import TensorInfo, onnx_dtype_to_torch_dtype



class ConcatNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: int) -> 'ConcatNode':
        """
        Static factory method to create a ConcatNode.
        
        Args:
            name: Node name
            inputs: List of input tensor names
            outputs: List of output tensor names
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            dim: Dimension to concatenate along (required, no default)
        """
        return ConcatNode(
            name=name,
            op_type="Concat",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim},
            forge_op_function_name="forge.op.Concatenate"
        )
    
    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch dim to Forge axis.
        
        Raises:
            ValueError: If 'dim' is None or not present
        """
        if 'dim' not in attrs:
            raise ValueError(
                f"ConcatNode '{self.name}': 'dim' attribute is required but not found in attrs"
            )
        dim = attrs['dim']
        if dim is None:
            raise ValueError(
                f"ConcatNode '{self.name}': 'dim' attribute cannot be None. "
                "ConcatNode is similar to torch.cat() which requires a 'dim' argument."
            )
        if not isinstance(dim, int):
            raise TypeError(
                f"ConcatNode '{self.name}': 'dim' must be an integer, got {type(dim).__name__}"
            )
        return {'axis': dim}
    
    def eval(self, input_tensors):
        """
        Evaluate Concat operation.
        
        Raises:
            ValueError: If 'dim' is None or not present
        """
        inputs = [input_tensors[inp] for inp in self.inputs]
        if 'dim' not in self.attrs:
            raise ValueError(
                f"ConcatNode '{self.name}': 'dim' attribute is required but not found"
            )
        dim = self.attrs['dim']
        if dim is None:
            raise ValueError(
                f"ConcatNode '{self.name}': 'dim' attribute cannot be None. "
                "ConcatNode is similar to torch.cat() which requires a 'dim' argument."
            )
        if not isinstance(dim, int):
            raise TypeError(
                f"ConcatNode '{self.name}': 'dim' must be an integer, got {type(dim).__name__}"
            )
        return {self.outputs[0]: torch.cat(inputs, dim=dim)}



class ClipNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               min_val: float = None,
               max_val: float = None) -> 'ClipNode':
        """Static factory method to create a ClipNode."""
        attrs = {}
        if min_val is not None:
            attrs['min'] = min_val
        if max_val is not None:
            attrs['max'] = max_val
        return ClipNode(
            name=name,
            op_type="Clip",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs=attrs,
            forge_op_function_name="forge.op.Clip"
        )

    def eval(self, input_tensors):
        """
        Evaluate Clip operation.
        
        Note: torch.clamp automatically handles the case where min > max
        by setting all elements to max.
        """
        x = input_tensors[self.inputs[0]]
        min_val = self.attrs.get('min', None)
        max_val = self.attrs.get('max', None)
        
        # torch.clamp handles min > max case automatically
        return {self.outputs[0]: torch.clamp(x, min=min_val, max=max_val)}



class CastNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dtype: Optional[torch.dtype] = None) -> 'CastNode':
        """Static factory method to create a CastNode."""
        attrs = {}
        if dtype is not None:
            attrs['dtype'] = dtype
        return CastNode(
            name=name,
            op_type="Cast",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs=attrs,
            forge_op_function_name="forge.op.Cast"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        to_dtype = self.attrs.get('dtype', x.dtype)
        return {self.outputs[0]: x.to(dtype=to_dtype)}



class PadNode(TIRNode):
    """
    PyTorch-like Pad operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               pad: Union[List[int], Tuple[int, ...]],
               mode: str = "constant",
               value: float = 0.0) -> 'PadNode':
        """Static factory method to create a PadNode."""
        return PadNode(
            name=name,
            op_type="Pad",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'pad': pad, 'mode': mode, 'value': value},
            forge_op_function_name="forge.op.misc.Pad"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        pad = self.attrs['pad']
        mode = self.attrs.get('mode', 'constant')
        value = self.attrs.get('value', 0.0)
        
        # Validate pad type and convert to tuple if needed
        if isinstance(pad, list):
            # Convert list to tuple if it's a list of integers
            if all(isinstance(p, int) for p in pad):
                pad = tuple(pad)
            else:
                raise TypeError(
                    f"PadNode '{self.name}': pad must be a tuple or list of integers, "
                    f"got list with non-integer elements: {pad}"
                )
        elif isinstance(pad, tuple):
            # Validate tuple contains only integers
            if not all(isinstance(p, int) for p in pad):
                raise TypeError(
                    f"PadNode '{self.name}': pad tuple must contain only integers, "
                    f"got: {pad}"
                )
        else:
            raise TypeError(
                f"PadNode '{self.name}': pad must be a tuple or list of integers, "
                f"got {type(pad).__name__}: {pad}"
            )
        
        # Validate mode
        supported_modes = {'constant', 'reflect', 'replicate', 'circular'}
        if mode not in supported_modes:
            raise ValueError(
                f"PadNode '{self.name}': unsupported padding mode '{mode}'. "
                f"Supported modes are: {supported_modes}"
            )
        
        # Debug: Print input shape, pad, mode, and other attributes before calling torch pad API
        print(f"[PadNode.eval] Node: {self.name}")
        print(f"  Input shape: {x.shape}")
        print(f"  Input dtype: {x.dtype}")
        print(f"  Pad: {pad}")
        print(f"  Mode: {mode}")
        print(f"  Value: {value}")
        print(f"  All attrs: {self.attrs}")
        print(f"  Input rank: {len(x.shape)}")
        
        return {self.outputs[0]: F.pad(x, pad, mode=mode, value=value)}



class IdentityNode(TIRNode):
    """
    Identity operation - returns input tensor unchanged.
    Maps to PyTorch's identity operation (just returns the tensor as-is).
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'IdentityNode':
        """Static factory method to create an IdentityNode."""
        return IdentityNode(
            name=name,
            op_type="Identity",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Identity"
        )

    def eval(self, input_tensors):
        """
        Evaluate Identity operation.
        Returns the input tensor unchanged (identity function).
        """
        x = input_tensors[self.inputs[0]]
        # Return the tensor as-is (identity operation)
        return {self.outputs[0]: x}


class FullNode(TIRNode):
    """
    Full operation - creates a tensor filled with a specified value.
    Maps to PyTorch's torch.full() operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               shape: Tuple,
               fill_value: float = 0.0,
               dtype: Optional[torch.dtype] = None) -> 'FullNode':
        """
        Static factory method to create a FullNode.
        
        Args:
            name: Node name
            inputs: List of input tensor names (can be empty for constant creation)
            outputs: List of output tensor names
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            shape: Shape of the tensor to create
            fill_value: Value to fill the tensor with (default: 0.0)
            dtype: Data type of the tensor (if None, inferred from output_tensors)
        """
        attrs = {'shape': shape, 'fill_value': fill_value}
        if dtype is not None:
            attrs['dtype'] = dtype
        
        return FullNode(
            name=name,
            op_type="Full",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs=attrs,
            forge_op_function_name="forge.op.Full"
        )

    def eval(self, input_tensors):
        """
        Evaluate Full operation.
        Creates a tensor filled with the specified value.
        """
        shape = self.attrs.get('shape', None)
        fill_value = self.attrs.get('fill_value', 0.0)
        dtype = self.attrs.get('dtype', None)
        
        if shape is None:
            # Fallback: use output shape from TensorInfo
            output_info = list(self.output_tensors.values())[0]
            if output_info and output_info.shape:
                shape = tuple(s if s is not None else 1 for s in output_info.shape)
            else:
                raise ValueError("FullNode requires shape attribute or output shape")
        
        # If dtype not specified, try to infer from output tensor info
        if dtype is None:
            output_info = list(self.output_tensors.values())[0]
            if output_info and hasattr(output_info, 'onnx_dtype'):
                from forge.transpiler.ir.types import onnx_dtype_to_torch_dtype
                dtype = onnx_dtype_to_torch_dtype(output_info.onnx_dtype)
            else:
                dtype = torch.float32  # Default
        
        # Create tensor filled with value
        result = torch.full(shape, fill_value, dtype=dtype)
        return {self.outputs[0]: result}

