# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Comparison operations: Equal, Greater, Less, GreaterOrEqual, LessOrEqual

All comparison operations support NumPy-style broadcasting via PyTorch and
return boolean tensors.  Shape inference is provided by BinaryBroadcastShape.
"""
import torch
from typing import Dict
from collections import OrderedDict

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.utils.binary_ops import validate_binary_inputs_pytorch_style
from forge.transpiler.operations.shape_mixins import BinaryBroadcastShape


class EqualNode(BinaryBroadcastShape, TIRNode):
    """
    Element-wise equality operation node.

    Computes output = (input1 == input2) element-wise with broadcasting.
    Maps to ``torch.eq`` and returns a boolean tensor of the broadcasted shape.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
    ) -> "EqualNode":
        """
        Create an EqualNode.

        Args:
            name: Node name.
            inputs: OrderedDict mapping input names to TensorInfo (two inputs required).
            outputs: OrderedDict mapping output names to TensorInfo.

        Returns:
            EqualNode instance.
        """
        return EqualNode(
            name=name,
            op_type="Equal",
            inputs=inputs,
            outputs=outputs,
            attrs={},
            forge_op_name="Equal",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate the equality comparison using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors.

        Returns:
            Dictionary mapping output name to result boolean tensor.

        Raises:
            ValueError: If dtypes are incompatible or shapes cannot be broadcast.
        """
        if len(self.input_names) < 2:
            raise ValueError(f"EqualNode '{self.name}' requires 2 inputs, got {len(self.input_names)}.")
        a = input_tensors[self.input_names[0]]
        b = input_tensors[self.input_names[1]]
        validate_binary_inputs_pytorch_style(
            a.shape,
            b.shape,
            a.dtype,
            b.dtype,
            self.op_type,
            self.input_names[0],
            self.input_names[1],
            operation_category="comparison",
        )
        return {self.output_names[0]: torch.eq(a, b)}


class GreaterNode(BinaryBroadcastShape, TIRNode):
    """
    Element-wise greater-than operation node.

    Computes output = (input1 > input2) element-wise with broadcasting.
    Maps to ``torch.gt`` and returns a boolean tensor of the broadcasted shape.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
    ) -> "GreaterNode":
        """
        Create a GreaterNode.

        Args:
            name: Node name.
            inputs: OrderedDict mapping input names to TensorInfo (two inputs required).
            outputs: OrderedDict mapping output names to TensorInfo.

        Returns:
            GreaterNode instance.
        """
        return GreaterNode(
            name=name,
            op_type="Greater",
            inputs=inputs,
            outputs=outputs,
            attrs={},
            forge_op_name="Greater",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate the greater-than comparison using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors.

        Returns:
            Dictionary mapping output name to result boolean tensor.

        Raises:
            ValueError: If dtypes are incompatible or shapes cannot be broadcast.
        """
        if len(self.input_names) < 2:
            raise ValueError(f"GreaterNode '{self.name}' requires 2 inputs, got {len(self.input_names)}.")
        a = input_tensors[self.input_names[0]]
        b = input_tensors[self.input_names[1]]
        validate_binary_inputs_pytorch_style(
            a.shape,
            b.shape,
            a.dtype,
            b.dtype,
            self.op_type,
            self.input_names[0],
            self.input_names[1],
            operation_category="comparison",
        )
        return {self.output_names[0]: torch.gt(a, b)}


class LessNode(BinaryBroadcastShape, TIRNode):
    """
    Element-wise less-than operation node.

    Computes output = (input1 < input2) element-wise with broadcasting.
    Maps to ``torch.lt`` and returns a boolean tensor of the broadcasted shape.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
    ) -> "LessNode":
        """
        Create a LessNode.

        Args:
            name: Node name.
            inputs: OrderedDict mapping input names to TensorInfo (two inputs required).
            outputs: OrderedDict mapping output names to TensorInfo.

        Returns:
            LessNode instance.
        """
        return LessNode(
            name=name,
            op_type="Less",
            inputs=inputs,
            outputs=outputs,
            attrs={},
            forge_op_name="Less",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate the less-than comparison using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors.

        Returns:
            Dictionary mapping output name to result boolean tensor.

        Raises:
            ValueError: If dtypes are incompatible or shapes cannot be broadcast.
        """
        if len(self.input_names) < 2:
            raise ValueError(f"LessNode '{self.name}' requires 2 inputs, got {len(self.input_names)}.")
        a = input_tensors[self.input_names[0]]
        b = input_tensors[self.input_names[1]]
        validate_binary_inputs_pytorch_style(
            a.shape,
            b.shape,
            a.dtype,
            b.dtype,
            self.op_type,
            self.input_names[0],
            self.input_names[1],
            operation_category="comparison",
        )
        return {self.output_names[0]: torch.lt(a, b)}


class GreaterOrEqualNode(BinaryBroadcastShape, TIRNode):
    """
    Element-wise greater-than-or-equal operation node.

    Computes output = (input1 >= input2) element-wise with broadcasting.
    Maps to ``torch.ge`` and returns a boolean tensor of the broadcasted shape.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
    ) -> "GreaterOrEqualNode":
        """
        Create a GreaterOrEqualNode.

        Args:
            name: Node name.
            inputs: OrderedDict mapping input names to TensorInfo (two inputs required).
            outputs: OrderedDict mapping output names to TensorInfo.

        Returns:
            GreaterOrEqualNode instance.
        """
        return GreaterOrEqualNode(
            name=name,
            op_type="GreaterOrEqual",
            inputs=inputs,
            outputs=outputs,
            attrs={},
            forge_op_name="GreaterEqual",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate the greater-than-or-equal comparison using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors.

        Returns:
            Dictionary mapping output name to result boolean tensor.

        Raises:
            ValueError: If dtypes are incompatible or shapes cannot be broadcast.
        """
        if len(self.input_names) < 2:
            raise ValueError(f"GreaterOrEqualNode '{self.name}' requires 2 inputs, got {len(self.input_names)}.")
        a = input_tensors[self.input_names[0]]
        b = input_tensors[self.input_names[1]]
        validate_binary_inputs_pytorch_style(
            a.shape,
            b.shape,
            a.dtype,
            b.dtype,
            self.op_type,
            self.input_names[0],
            self.input_names[1],
            operation_category="comparison",
        )
        return {self.output_names[0]: torch.ge(a, b)}


class LessOrEqualNode(BinaryBroadcastShape, TIRNode):
    """
    Element-wise less-than-or-equal operation node.

    Computes output = (input1 <= input2) element-wise with broadcasting.
    Maps to ``torch.le`` and returns a boolean tensor of the broadcasted shape.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
    ) -> "LessOrEqualNode":
        """
        Create a LessOrEqualNode.

        Args:
            name: Node name.
            inputs: OrderedDict mapping input names to TensorInfo (two inputs required).
            outputs: OrderedDict mapping output names to TensorInfo.

        Returns:
            LessOrEqualNode instance.
        """
        return LessOrEqualNode(
            name=name,
            op_type="LessOrEqual",
            inputs=inputs,
            outputs=outputs,
            attrs={},
            forge_op_name="LessEqual",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate the less-than-or-equal comparison using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors.

        Returns:
            Dictionary mapping output name to result boolean tensor.

        Raises:
            ValueError: If dtypes are incompatible or shapes cannot be broadcast.
        """
        if len(self.input_names) < 2:
            raise ValueError(f"LessOrEqualNode '{self.name}' requires 2 inputs, got {len(self.input_names)}.")
        a = input_tensors[self.input_names[0]]
        b = input_tensors[self.input_names[1]]
        validate_binary_inputs_pytorch_style(
            a.shape,
            b.shape,
            a.dtype,
            b.dtype,
            self.op_type,
            self.input_names[0],
            self.input_names[1],
            operation_category="comparison",
        )
        return {self.output_names[0]: torch.le(a, b)}
