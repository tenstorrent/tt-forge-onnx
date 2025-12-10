"""
Convolution operations: Conv1d, Conv2d, Conv3d
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple


from forge.transpiler.ir.nodes import TIRNode
from forge.transpiler.ir.types import TensorInfo


def _normalize_conv_attr(value: Union[int, Tuple[int, ...], List[int]], 
                         ndim: int, name: str, node_name: str) -> Tuple[int, ...]:
    """
    Normalize convolution attribute (stride/dilation) to tuple format.
    
    Args:
        value: int, tuple, or list
        ndim: Expected number of dimensions (1, 2, or 3)
        name: Attribute name for error messages
        node_name: Node name for error messages
        
    Returns:
        Tuple of length ndim
    """
    if isinstance(value, int):
        return (value,) * ndim
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return (1,) * ndim
        elif len(value) == 1:
            return (value[0],) * ndim
        elif len(value) >= ndim:
            return tuple(value[:ndim])
        else:
            # Pad with last value
            return tuple(value) + (value[-1],) * (ndim - len(value))
    else:
        raise ValueError(
            f"{node_name}: {name} must be int or tuple/list of {ndim} ints, "
            f"got {type(value).__name__}"
        )


class Conv1dNode(TIRNode):
    """PyTorch-like Conv1d operation."""
    
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               stride: Union[int, Tuple[int]] = 1,
               padding: Union[int, str, Tuple[int]] = 0,
               dilation: Union[int, Tuple[int]] = 1,
               groups: int = 1) -> 'Conv1dNode':
        """Static factory method to create a Conv1dNode."""
        return Conv1dNode(
            name=name,
            op_type="Conv1d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'groups': groups
            },
            forge_op_function_name="forge.op.convolution.Conv1d"
        )
    
    def eval(self, input_tensors):
        """Execute Conv1d operation using torch.nn.functional.conv1d."""
        x = input_tensors[self.inputs[0]]  # Input: (N, C_in, W)
        w = input_tensors[self.inputs[1]]   # Weight: (C_out, C_in/groups, kW)
        b = input_tensors[self.inputs[2]] if len(self.inputs) > 2 else None  # Bias: (C_out,) optional
        
        # Normalize attributes
        stride_tuple = _normalize_conv_attr(self.attrs.get('stride', 1), 1, 'stride', self.name)
        dilation_tuple = _normalize_conv_attr(self.attrs.get('dilation', 1), 1, 'dilation', self.name)
        stride_val = stride_tuple[0]
        dilation_val = dilation_tuple[0]
        padding = self.attrs.get('padding', 0)
        groups = self.attrs.get('groups', 1)
        
        # Call PyTorch conv1d
        output = F.conv1d(
            input=x,
            weight=w,
            bias=b,
            stride=stride_val,
            padding=padding,
            dilation=dilation_val,
            groups=groups
        )
        
        return {self.outputs[0]: output}


class Conv2dNode(TIRNode):
    """PyTorch-like Conv2d operation."""
    
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               stride: Union[int, Tuple[int, int]] = 1,
               padding: Union[int, str, Tuple[int, int]] = 0,
               dilation: Union[int, Tuple[int, int]] = 1,
               groups: int = 1) -> 'Conv2dNode':
        """Static factory method to create a Conv2dNode."""
        return Conv2dNode(
            name=name,
            op_type="Conv2d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'groups': groups
            },
            forge_op_function_name="forge.op.convolution.Conv2d"
        )

    def eval(self, input_tensors):
        """Execute Conv2d operation using torch.nn.functional.conv2d."""
        x = input_tensors[self.inputs[0]]  # Input: (N, C_in, H, W)
        w = input_tensors[self.inputs[1]]   # Weight: (C_out, C_in/groups, kH, kW)
        b = input_tensors[self.inputs[2]] if len(self.inputs) > 2 else None  # Bias: (C_out,) optional
        
        # Normalize attributes
        stride = _normalize_conv_attr(self.attrs.get('stride', 1), 2, 'stride', self.name)
        dilation = _normalize_conv_attr(self.attrs.get('dilation', 1), 2, 'dilation', self.name)
        padding = self.attrs.get('padding', 0)
        groups = self.attrs.get('groups', 1)
        
        # Call PyTorch conv2d
        output = F.conv2d(
            input=x,
            weight=w,
            bias=b,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        
        return {self.outputs[0]: output}


class Conv3dNode(TIRNode):
    """PyTorch-like Conv3d operation."""
    
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               stride: Union[int, Tuple[int, int, int]] = 1,
               padding: Union[int, str, Tuple[int, int, int], Tuple[int, int, int, int, int, int]] = 0,
               dilation: Union[int, Tuple[int, int, int]] = 1,
               groups: int = 1) -> 'Conv3dNode':
        """Static factory method to create a Conv3dNode."""
        return Conv3dNode(
            name=name,
            op_type="Conv3d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'groups': groups
            },
            forge_op_function_name="forge.op.convolution.Conv3d"
        )
    
    def eval(self, input_tensors):
        """Execute Conv3d operation using torch.nn.functional.conv3d."""
        x = input_tensors[self.inputs[0]]  # Input: (N, C_in, D, H, W)
        w = input_tensors[self.inputs[1]]   # Weight: (C_out, C_in/groups, kD, kH, kW)
        b = input_tensors[self.inputs[2]] if len(self.inputs) > 2 else None  # Bias: (C_out,) optional
        
        # Normalize attributes
        stride = _normalize_conv_attr(self.attrs.get('stride', 1), 3, 'stride', self.name)
        dilation = _normalize_conv_attr(self.attrs.get('dilation', 1), 3, 'dilation', self.name)
        padding = self.attrs.get('padding', 0)
        groups = self.attrs.get('groups', 1)
        
        # Validate padding format (asymmetric padding should be handled by PadNode)
        if isinstance(padding, (tuple, list)) and len(padding) == 6:
            raise ValueError(
                f"Conv3dNode '{self.name}': Asymmetric padding (tuple of 6) is not supported by F.conv3d. "
                f"This should have been handled by a PadNode in the converter. Got padding={padding}"
            )
        
        # Call PyTorch conv3d
        output = F.conv3d(
            input=x,
            weight=w,
            bias=b,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        
        return {self.outputs[0]: output}
