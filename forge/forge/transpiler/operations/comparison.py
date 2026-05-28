# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Comparison and logical operations.

Comparison: Equal, Greater, Less, GreaterOrEqual, LessOrEqual
Logical:    LogicalNot (unary), LogicalAnd (binary)

All operations support NumPy-style broadcasting via PyTorch.
Comparison and logical operations return boolean tensors.
Shape inference is provided by BinaryBroadcastShape / ElementwiseUnaryShape.
"""
import torch
from typing import Dict
from collections import OrderedDict

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.utils.binary_ops import validate_binary_inputs_pytorch_style
from forge.transpiler.operations.shape_mixins import BinaryBroadcastShape, ElementwiseUnaryShape


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


# ---------------------------------------------------------------------------
# Logical operations
# ---------------------------------------------------------------------------


class LogicalNotNode(ElementwiseUnaryShape, TIRNode):
    """
    Element-wise logical NOT operation node.

    Computes ``Y = NOT X`` element-wise.  Both input and output are
    ``tensor(bool)``.  Maps to ``torch.logical_not``.

    ONNX spec: available since opset 1.  No attributes.  Type constraint T is
    ``tensor(bool)`` only.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
    ) -> "LogicalNotNode":
        """
        Create a LogicalNotNode.

        Args:
            name:    Node name.
            inputs:  OrderedDict with one entry — X (the boolean tensor).
            outputs: OrderedDict mapping the output name to TensorInfo.

        Returns:
            LogicalNotNode instance.
        """
        return LogicalNotNode(
            name=name,
            op_type="LogicalNot",
            inputs=inputs,
            outputs=outputs,
            attrs={},
            forge_op_name="LogicalNot",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate logical NOT using PyTorch.

        Args:
            input_tensors: Mapping from input name to tensor.
                The single input must be a ``torch.bool`` tensor.

        Returns:
            Mapping from output name to result ``torch.bool`` tensor.

        Raises:
            ValueError: If fewer than 1 input is provided or the input is not
                a boolean tensor.
        """
        if len(self.input_names) < 1:
            raise ValueError(f"LogicalNotNode '{self.name}' requires 1 input, " f"got {len(self.input_names)}.")
        x = input_tensors[self.input_names[0]]
        if x.dtype != torch.bool:
            raise ValueError(
                f"LogicalNotNode '{self.name}': input must be a boolean tensor "
                f"(torch.bool), got {x.dtype}.  "
                f"ONNX Not is constrained to tensor(bool)."
            )
        return {self.output_names[0]: torch.logical_not(x)}


class LogicalAndNode(BinaryBroadcastShape, TIRNode):
    """
    Element-wise logical AND operation node.

    Computes ``C = A AND B`` element-wise with NumPy-style broadcasting.
    Both inputs and the output are ``tensor(bool)``.  Maps to
    ``torch.logical_and``.

    ONNX spec:
    - Opset 1-6: broadcasting is opt-in via ``broadcast`` / ``axis`` attributes.
    - Opset 7+:  multidirectional (NumPy-style) broadcasting, attributes removed.
    Type constraint T is ``tensor(bool)`` for both inputs.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
    ) -> "LogicalAndNode":
        """
        Create a LogicalAndNode.

        Args:
            name:    Node name.
            inputs:  OrderedDict with two entries — A and B (boolean tensors).
            outputs: OrderedDict mapping the output name to TensorInfo.

        Returns:
            LogicalAndNode instance.
        """
        return LogicalAndNode(
            name=name,
            op_type="LogicalAnd",
            inputs=inputs,
            outputs=outputs,
            attrs={},
            forge_op_name="LogicalAnd",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate logical AND using PyTorch.

        Args:
            input_tensors: Mapping from input name to tensor.
                Both inputs must be ``torch.bool`` tensors and their shapes
                must be broadcast-compatible.

        Returns:
            Mapping from output name to result ``torch.bool`` tensor.

        Raises:
            ValueError: If fewer than 2 inputs are provided, inputs are not
                boolean tensors, or shapes are incompatible for broadcasting.
        """
        if len(self.input_names) < 2:
            raise ValueError(f"LogicalAndNode '{self.name}' requires 2 inputs, " f"got {len(self.input_names)}.")
        a = input_tensors[self.input_names[0]]
        b = input_tensors[self.input_names[1]]

        for tensor, name in ((a, self.input_names[0]), (b, self.input_names[1])):
            if tensor.dtype != torch.bool:
                raise ValueError(
                    f"LogicalAndNode '{self.name}': input '{name}' must be a boolean "
                    f"tensor (torch.bool), got {tensor.dtype}.  "
                    f"ONNX And is constrained to tensor(bool)."
                )

        validate_binary_inputs_pytorch_style(
            a.shape,
            b.shape,
            a.dtype,
            b.dtype,
            self.op_type,
            self.input_names[0],
            self.input_names[1],
            operation_category="logical",
        )
        return {self.output_names[0]: torch.logical_and(a, b)}
