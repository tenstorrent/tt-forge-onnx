# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pooling operations: MaxPool1d, MaxPool2d, MaxPool3d, AveragePool1d, AveragePool2d, AveragePool3d, GlobalAveragePool
"""
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Union, Tuple

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo


class MaxPool1dNode(TIRNode):
    """
    1D max pooling operation node.

    Applies a 1D max pooling over an input signal composed of several input planes.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        kernel_size: int,
        stride: int = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: int = 1,
        ceil_mode: bool = False,
    ) -> "MaxPool1dNode":
        """
        Create a MaxPool1dNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
            kernel_size: Size of the sliding window
            stride: Stride of the window (default: kernel_size)
            padding: Padding added to both sides (default: 0)
            dilation: Spacing between kernel elements (default: 1)
            ceil_mode: If True, use ceil instead of floor for output size (default: False)

        Returns:
            MaxPool1dNode instance
        """
        if stride is None:
            stride = kernel_size

        return MaxPool1dNode(
            name=name,
            op_type="MaxPool1d",
            inputs=inputs,
            outputs=outputs,
            attrs={
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "ceil_mode": ceil_mode,
            },
            forge_op_name="MaxPool1d",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate MaxPool1d operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        kernel_size = self.attrs["kernel_size"]
        stride = self.attrs.get("stride", kernel_size)
        padding = self.attrs.get("padding", 0)
        dilation = self.attrs.get("dilation", 1)
        ceil_mode = self.attrs.get("ceil_mode", False)
        return {self.output_names[0]: F.max_pool1d(x, kernel_size, stride, padding, dilation, ceil_mode)}


class MaxPool2dNode(TIRNode):
    """
    2D max pooling operation node.

    Applies a 2D max pooling over an input signal composed of several input planes.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        ceil_mode: bool = False,
    ) -> "MaxPool2dNode":
        """
        Create a MaxPool2dNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
            kernel_size: Size of the sliding window
            stride: Stride of the window (default: kernel_size)
            padding: Padding added to both sides (default: 0)
            dilation: Spacing between kernel elements (default: 1)
            ceil_mode: If True, use ceil instead of floor for output size (default: False)

        Returns:
            MaxPool2dNode instance
        """
        if stride is None:
            stride = kernel_size

        return MaxPool2dNode(
            name=name,
            op_type="MaxPool2d",
            inputs=inputs,
            outputs=outputs,
            attrs={
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "ceil_mode": ceil_mode,
            },
            forge_op_name="MaxPool2d",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate MaxPool2d operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        kernel_size = self.attrs["kernel_size"]
        stride = self.attrs.get("stride", kernel_size)
        padding = self.attrs.get("padding", 0)
        dilation = self.attrs.get("dilation", 1)
        ceil_mode = self.attrs.get("ceil_mode", False)
        return {self.output_names[0]: F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)}


class MaxPool3dNode(TIRNode):
    """
    3D max pooling operation node.

    Applies a 3D max pooling over an input signal composed of several input planes.
    Note: MaxPool3d is not available in Forge (only MaxPool1d and MaxPool2d exist).
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        ceil_mode: bool = False,
    ) -> "MaxPool3dNode":
        """
        Create a MaxPool3dNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
            kernel_size: Size of the sliding window
            stride: Stride of the window (default: kernel_size)
            padding: Padding added to both sides (default: 0)
            dilation: Spacing between kernel elements (default: 1)
            ceil_mode: If True, use ceil instead of floor for output size (default: False)

        Returns:
            MaxPool3dNode instance
        """
        if stride is None:
            stride = kernel_size

        return MaxPool3dNode(
            name=name,
            op_type="MaxPool3d",
            inputs=inputs,
            outputs=outputs,
            attrs={
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "ceil_mode": ceil_mode,
            },
            forge_op_name=None,  # MaxPool3d not available in Forge (only MaxPool1d, MaxPool2d exist)
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate MaxPool3d operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        kernel_size = self.attrs["kernel_size"]
        stride = self.attrs.get("stride", kernel_size)
        padding = self.attrs.get("padding", 0)
        dilation = self.attrs.get("dilation", 1)
        ceil_mode = self.attrs.get("ceil_mode", False)
        return {self.output_names[0]: F.max_pool3d(x, kernel_size, stride, padding, dilation, ceil_mode)}


class AveragePool1dNode(TIRNode):
    """
    1D average pooling operation node.

    Applies a 1D average pooling over an input signal composed of several input planes.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        kernel_size: int,
        stride: int = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> "AveragePool1dNode":
        """
        Create an AveragePool1dNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
            kernel_size: Size of the sliding window
            stride: Stride of the window (default: kernel_size)
            padding: Padding added to both sides (default: 0)
            ceil_mode: If True, use ceil instead of floor for output size (default: False)
            count_include_pad: If True, include zero-padding in average calculation (default: True)

        Returns:
            AveragePool1dNode instance
        """
        if stride is None:
            stride = kernel_size

        return AveragePool1dNode(
            name=name,
            op_type="AveragePool1d",
            inputs=inputs,
            outputs=outputs,
            attrs={
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "ceil_mode": ceil_mode,
                "count_include_pad": count_include_pad,
            },
            forge_op_name="AvgPool1d",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate AveragePool1d operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        kernel_size = self.attrs["kernel_size"]
        stride = self.attrs.get("stride", kernel_size)
        padding = self.attrs.get("padding", 0)
        ceil_mode = self.attrs.get("ceil_mode", False)
        count_include_pad = self.attrs.get("count_include_pad", True)
        return {self.output_names[0]: F.avg_pool1d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}


class AveragePool2dNode(TIRNode):
    """
    2D average pooling operation node.

    Applies a 2D average pooling over an input signal composed of several input planes.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> "AveragePool2dNode":
        """
        Create an AveragePool2dNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
            kernel_size: Size of the sliding window
            stride: Stride of the window (default: kernel_size)
            padding: Padding added to both sides (default: 0)
            ceil_mode: If True, use ceil instead of floor for output size (default: False)
            count_include_pad: If True, include zero-padding in average calculation (default: True)

        Returns:
            AveragePool2dNode instance
        """
        if stride is None:
            stride = kernel_size

        return AveragePool2dNode(
            name=name,
            op_type="AveragePool2d",
            inputs=inputs,
            outputs=outputs,
            attrs={
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "ceil_mode": ceil_mode,
                "count_include_pad": count_include_pad,
            },
            forge_op_name="AvgPool2d",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate AveragePool2d operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        kernel_size = self.attrs["kernel_size"]
        stride = self.attrs.get("stride", kernel_size)
        padding = self.attrs.get("padding", 0)
        ceil_mode = self.attrs.get("ceil_mode", False)
        count_include_pad = self.attrs.get("count_include_pad", True)
        return {self.output_names[0]: F.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}


class AveragePool3dNode(TIRNode):
    """
    3D average pooling operation node.

    Applies a 3D average pooling over an input signal composed of several input planes.
    Note: AveragePool3d is not available in Forge (only AvgPool1d and AvgPool2d exist).
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> "AveragePool3dNode":
        """
        Create an AveragePool3dNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
            kernel_size: Size of the sliding window
            stride: Stride of the window (default: kernel_size)
            padding: Padding added to both sides (default: 0)
            ceil_mode: If True, use ceil instead of floor for output size (default: False)
            count_include_pad: If True, include zero-padding in average calculation (default: True)

        Returns:
            AveragePool3dNode instance
        """
        if stride is None:
            stride = kernel_size

        return AveragePool3dNode(
            name=name,
            op_type="AveragePool3d",
            inputs=inputs,
            outputs=outputs,
            attrs={
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "ceil_mode": ceil_mode,
                "count_include_pad": count_include_pad,
            },
            forge_op_name=None,  # AvgPool3d not available in Forge (only AvgPool1d, AvgPool2d exist)
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate AveragePool3d operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        kernel_size = self.attrs["kernel_size"]
        stride = self.attrs.get("stride", kernel_size)
        padding = self.attrs.get("padding", 0)
        ceil_mode = self.attrs.get("ceil_mode", False)
        count_include_pad = self.attrs.get("count_include_pad", True)
        return {self.output_names[0]: F.avg_pool3d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}
