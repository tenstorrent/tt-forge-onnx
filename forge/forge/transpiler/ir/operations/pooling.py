"""
Pooling operations: MaxPool1d, MaxPool2d, MaxPool3d, AveragePool1d, AveragePool2d, AveragePool3d, GlobalAveragePool
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple

from forge.transpiler.ir.nodes import TIRNode
from forge.transpiler.ir.types import TensorInfo


class MaxPool1dNode(TIRNode):
    """
    PyTorch-like MaxPool1d operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               kernel_size: int,
               stride: int = None,
               padding: Union[int, Tuple[int, int]] = 0,
               dilation: int = 1,
               ceil_mode: bool = False) -> 'MaxPool1dNode':
        """Static factory method to create a MaxPool1dNode."""
        if stride is None:
            stride = kernel_size
        
        return MaxPool1dNode(
            name=name,
            op_type="MaxPool1d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'ceil_mode': ceil_mode
            },
            forge_op_function_name="forge.op.MaxPool1d"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        kernel_size = self.attrs['kernel_size']
        stride = self.attrs.get('stride', kernel_size)
        padding = self.attrs.get('padding', 0)
        dilation = self.attrs.get('dilation', 1)
        ceil_mode = self.attrs.get('ceil_mode', False)
        return {self.outputs[0]: F.max_pool1d(x, kernel_size, stride, padding, dilation, ceil_mode)}


class MaxPool2dNode(TIRNode):
    """
    PyTorch-like MaxPool2d operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               kernel_size: Union[int, Tuple[int, int]],
               stride: Union[int, Tuple[int, int]] = None,
               padding: Union[int, Tuple[int, int]] = 0,
               dilation: Union[int, Tuple[int, int]] = 1,
               ceil_mode: bool = False) -> 'MaxPool2dNode':
        """Static factory method to create a MaxPool2dNode."""
        if stride is None:
            stride = kernel_size
        
        return MaxPool2dNode(
            name=name,
            op_type="MaxPool2d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'ceil_mode': ceil_mode
            },
            forge_op_function_name="forge.op.MaxPool2d"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        kernel_size = self.attrs['kernel_size']
        stride = self.attrs.get('stride', kernel_size)
        padding = self.attrs.get('padding', 0)
        dilation = self.attrs.get('dilation', 1)
        ceil_mode = self.attrs.get('ceil_mode', False)
        return {self.outputs[0]: F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)}


class MaxPool3dNode(TIRNode):
    """
    PyTorch-like MaxPool3d operation.
    Note: Forge does not have MaxPool3d operation - only MaxPool1d and MaxPool2d are supported.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               kernel_size: Union[int, Tuple[int, int, int]],
               stride: Union[int, Tuple[int, int, int]] = None,
               padding: Union[int, Tuple[int, int, int]] = 0,
               dilation: Union[int, Tuple[int, int, int]] = 1,
               ceil_mode: bool = False) -> 'MaxPool3dNode':
        """Static factory method to create a MaxPool3dNode."""
        if stride is None:
            stride = kernel_size
        
        return MaxPool3dNode(
            name=name,
            op_type="MaxPool3d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'ceil_mode': ceil_mode
            },
            forge_op_function_name="UNKNOWN"  # MaxPool3d not supported in Forge
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        kernel_size = self.attrs['kernel_size']
        stride = self.attrs.get('stride', kernel_size)
        padding = self.attrs.get('padding', 0)
        dilation = self.attrs.get('dilation', 1)
        ceil_mode = self.attrs.get('ceil_mode', False)
        return {self.outputs[0]: F.max_pool3d(x, kernel_size, stride, padding, dilation, ceil_mode)}


class AveragePool1dNode(TIRNode):
    """
    PyTorch-like AveragePool1d operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               kernel_size: int,
               stride: int = None,
               padding: Union[int, Tuple[int, int]] = 0,
               ceil_mode: bool = False,
               count_include_pad: bool = True) -> 'AveragePool1dNode':
        """Static factory method to create an AveragePool1dNode."""
        if stride is None:
            stride = kernel_size
        
        return AveragePool1dNode(
            name=name,
            op_type="AveragePool1d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'ceil_mode': ceil_mode,
                'count_include_pad': count_include_pad
            },
            forge_op_function_name="forge.op.AvgPool1d"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        kernel_size = self.attrs['kernel_size']
        stride = self.attrs.get('stride', kernel_size)
        padding = self.attrs.get('padding', 0)
        ceil_mode = self.attrs.get('ceil_mode', False)
        count_include_pad = self.attrs.get('count_include_pad', True)
        return {self.outputs[0]: F.avg_pool1d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}


class AveragePool2dNode(TIRNode):
    """
    PyTorch-like AveragePool2d operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               kernel_size: Union[int, Tuple[int, int]],
               stride: Union[int, Tuple[int, int]] = None,
               padding: Union[int, Tuple[int, int]] = 0,
               ceil_mode: bool = False,
               count_include_pad: bool = True) -> 'AveragePool2dNode':
        """Static factory method to create an AveragePool2dNode."""
        if stride is None:
            stride = kernel_size
        
        return AveragePool2dNode(
            name=name,
            op_type="AveragePool2d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'ceil_mode': ceil_mode,
                'count_include_pad': count_include_pad
            },
            forge_op_function_name="forge.op.AvgPool2d"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        kernel_size = self.attrs['kernel_size']
        stride = self.attrs.get('stride', kernel_size)
        padding = self.attrs.get('padding', 0)
        ceil_mode = self.attrs.get('ceil_mode', False)
        count_include_pad = self.attrs.get('count_include_pad', True)
        return {self.outputs[0]: F.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}


class AveragePool3dNode(TIRNode):
    """
    PyTorch-like AveragePool3d operation.
    Note: Forge does not have AveragePool3d operation - only AvgPool1d and AvgPool2d are supported.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               kernel_size: Union[int, Tuple[int, int, int]],
               stride: Union[int, Tuple[int, int, int]] = None,
               padding: Union[int, Tuple[int, int, int]] = 0,
               ceil_mode: bool = False,
               count_include_pad: bool = True) -> 'AveragePool3dNode':
        """Static factory method to create an AveragePool3dNode."""
        if stride is None:
            stride = kernel_size
        
        return AveragePool3dNode(
            name=name,
            op_type="AveragePool3d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'ceil_mode': ceil_mode,
                'count_include_pad': count_include_pad
            },
            forge_op_function_name="UNKNOWN"  # AvgPool3d not supported in Forge
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        kernel_size = self.attrs['kernel_size']
        stride = self.attrs.get('stride', kernel_size)
        padding = self.attrs.get('padding', 0)
        ceil_mode = self.attrs.get('ceil_mode', False)
        count_include_pad = self.attrs.get('count_include_pad', True)
        return {self.outputs[0]: F.avg_pool3d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}


class GlobalAveragePoolNode(TIRNode):
    """
    PyTorch-like GlobalAveragePool operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'GlobalAveragePoolNode':
        """Static factory method to create a GlobalAveragePoolNode."""
        return GlobalAveragePoolNode(
            name=name,
            op_type="GlobalAveragePool",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.ReduceAvg"  # GlobalAveragePool can be represented as ReduceAvg
        )
    
    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        # Global average pool over spatial dimensions (H, W)
        return {self.outputs[0]: F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)}
