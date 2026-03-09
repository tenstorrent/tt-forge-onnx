# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Arithmetic operations: Add, Sub, Mul, Div, Pow, MatMul, Gemm
"""
import torch
from typing import Dict
from collections import OrderedDict

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.utils.binary_ops import validate_binary_inputs_pytorch_style
from forge.transpiler.operations.shape_mixins import (
    BinaryBroadcastShape,
    MatMulShape,
)


class PowNode(BinaryBroadcastShape, TIRNode):
    """
    Element-wise power operation: Z = X ^ Y.

    Both the base (X) and the exponent (Y) are tensor inputs, matching
    ``forge.op.Power`` (binary) and ``ttir.pow(%lhs, %rhs)`` which both
    require two tensor operands.

    **Dtype handling (ONNX v12+ heterogeneous types):**
    From opset 12, X has type T and Y has type T1 — they may differ.
    Output Z always has the same type as X (type T).
    When Y's dtype differs from X's dtype, :class:`PowConverter` inserts a
    :class:`~forge.transpiler.operations.other.CastNode` *before* this node
    in the TIR graph so that both inputs share the same dtype by the time
    ``eval()`` is called.  No runtime coercion is performed here.

    Shape inference follows NumPy multidirectional broadcasting rules via
    :class:`BinaryBroadcastShape`.

    Maps to ``forge.op.Power(name, X, Y)`` → ``OpType::Power`` → string
    ``"power"`` → MLIR ``ttir.pow(%lhs, %rhs)``.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
    ) -> "PowNode":
        """
        Create a PowNode.

        Args:
            name: Node name.
            inputs: OrderedDict with two entries — X (base) and Y (exponent).
                Y may be a constant initializer or a runtime activation.
            outputs: OrderedDict mapping output name to TensorInfo.

        Returns:
            PowNode instance.
        """
        return PowNode(
            name=name,
            op_type="Pow",
            inputs=inputs,
            outputs=outputs,
            attrs={},
            forge_op_name="Power",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate the power operation using PyTorch.

        Both X and Y are expected to have the same dtype when this node is
        reached.  If Y originally had a different type (ONNX v12+ heterogeneous
        T / T1), the :class:`PowConverter` inserts an explicit
        :class:`~forge.transpiler.operations.other.CastNode` for Y *before*
        this node so that the dtype contract is already satisfied here.

        Args:
            input_tensors: Mapping from input name to tensor.
                Must contain entries for both X (base) and Y (exponent).

        Returns:
            Mapping from output name to result tensor.

        Raises:
            ValueError: If fewer than 2 tensor inputs are provided.
        """
        if len(self.input_names) < 2:
            raise ValueError(
                f"PowNode '{self.name}' requires 2 inputs (base X and exponent Y) "
                f"but only has {len(self.input_names)}: {self.input_names}."
            )
        x = input_tensors[self.input_names[0]]
        y = input_tensors[self.input_names[1]]
        return {self.output_names[0]: torch.pow(x, y)}


class AddNode(BinaryBroadcastShape, TIRNode):
    """
    Addition operation node.

    Performs element-wise addition: output = input1 + input2
    Supports broadcasting automatically via PyTorch.
    """

    @staticmethod
    def create(name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo]) -> "AddNode":
        """
        Create an AddNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo

        Returns:
            AddNode instance
        """
        return AddNode(name=name, op_type="Add", inputs=inputs, outputs=outputs, attrs={}, forge_op_name="Add")

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate addition operation using PyTorch.

        Performs element-wise addition with broadcasting support.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If dtypes don't match or shapes are incompatible for broadcasting
        """
        if len(self.input_names) < 2:
            raise ValueError(
                f"AddNode '{self.name}' requires 2 inputs but only has {len(self.input_names)}. "
                f"Input names: {self.input_names}. "
                f"Inputs dict keys: {list(self.inputs.keys())}. "
                f"Input tensors provided: {list(input_tensors.keys())}"
            )
        if len(input_tensors) < 2:
            raise ValueError(
                f"AddNode '{self.name}' received {len(input_tensors)} input tensor(s) but requires 2. "
                f"Expected input names: {self.input_names}. "
                f"Received input names: {list(input_tensors.keys())}"
            )
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
            operation_category="arithmetic",
        )

        return {self.output_names[0]: torch.add(a, b)}


class SubNode(BinaryBroadcastShape, TIRNode):
    """
    Subtraction operation node.

    Performs element-wise subtraction: output = input1 - input2
    Supports broadcasting automatically via PyTorch.
    """

    @staticmethod
    def create(name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo]) -> "SubNode":
        """
        Create a SubNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo

        Returns:
            SubNode instance
        """
        return SubNode(name=name, op_type="Sub", inputs=inputs, outputs=outputs, attrs={}, forge_op_name="Subtract")

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate subtraction operation using PyTorch.

        Performs element-wise subtraction with broadcasting support.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If dtypes don't match or shapes are incompatible for broadcasting
        """
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
            operation_category="arithmetic",
        )

        return {self.output_names[0]: torch.sub(a, b)}


class MulNode(BinaryBroadcastShape, TIRNode):
    """
    Multiplication operation node using PyTorch API.

    Performs element-wise multiplication: output = input1 * input2
    Supports broadcasting automatically via PyTorch.
    """

    @staticmethod
    def create(name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo]) -> "MulNode":
        """Static factory method to create a MulNode."""
        return MulNode(name=name, op_type="Mul", inputs=inputs, outputs=outputs, attrs={}, forge_op_name="Multiply")

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate multiplication operation using PyTorch.

        Performs element-wise multiplication with broadcasting support.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If dtypes don't match or shapes are incompatible for broadcasting
        """
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
            operation_category="arithmetic",
        )

        return {self.output_names[0]: torch.mul(a, b)}


class DivNode(BinaryBroadcastShape, TIRNode):
    """
    Division operation node.

    Performs element-wise division: output = input1 / input2
    Supports broadcasting automatically via PyTorch.
    Note: Division by zero behavior follows PyTorch semantics (inf or NaN).
    """

    @staticmethod
    def create(name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo]) -> "DivNode":
        """
        Create a DivNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo

        Returns:
            DivNode instance
        """
        return DivNode(name=name, op_type="Div", inputs=inputs, outputs=outputs, attrs={}, forge_op_name="Divide")

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate division operation using PyTorch.

        Performs element-wise division with broadcasting support.
        Uses floor division for integer types to match ONNX semantics,
        true division for floating point types.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If dtypes don't match or shapes are incompatible for broadcasting
        """
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
            operation_category="arithmetic",
        )

        is_integer_type = not a.dtype.is_floating_point

        if is_integer_type:
            return {self.output_names[0]: torch.div(a, b, rounding_mode="floor")}
        else:
            return {self.output_names[0]: torch.div(a, b)}


class MatMulNode(MatMulShape, TIRNode):
    """
    Matrix multiplication operation node using PyTorch API.

        Performs matrix multiplication: output = input1 @ input2
        Supports batched matrix multiplication via PyTorch.
    """

    @staticmethod
    def create(name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo]) -> "MatMulNode":
        """
        Create a MatMulNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo

        Returns:
            MatMulNode instance
        """
        return MatMulNode(name=name, op_type="MatMul", inputs=inputs, outputs=outputs, attrs={}, forge_op_name="Matmul")

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate matrix multiplication operation using PyTorch.

        Performs matrix multiplication with support for batched operations.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        a = input_tensors[self.input_names[0]]
        b = input_tensors[self.input_names[1]]
        return {self.output_names[0]: torch.matmul(a, b)}
