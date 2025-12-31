# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Arithmetic operations: Add, Sub, Mul, Div, MatMul, Gemm
"""
import torch
from typing import Dict
from collections import OrderedDict

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo


def validate_broadcasting_pytorch_style(
    shape_a: torch.Size,
    shape_b: torch.Size,
    dtype_a: torch.dtype,
    dtype_b: torch.dtype,
    op_name: str,
    tensor_a_name: str,
    tensor_b_name: str,
) -> None:
    """
    Validate broadcasting and dtype compatibility following PyTorch style.

    This function validates:
    1. Dtype equality: Both tensors must have the same dtype (PyTorch requirement)
    2. Broadcasting compatibility: Shapes must be compatible for broadcasting

    Broadcasting rules (PyTorch/NumPy-style):
    - Shapes are compared from right to left
    - Two dimensions are compatible if:
      * They are equal, OR
      * One of them is 1, OR
      * One of them doesn't exist (missing dimension)

    Args:
        shape_a: Shape of first tensor
        shape_b: Shape of second tensor
        dtype_a: Dtype of first tensor
        dtype_b: Dtype of second tensor
        op_name: Name of the operation (for error messages)
        tensor_a_name: Name of first tensor (for error messages)
        tensor_b_name: Name of second tensor (for error messages)

    Raises:
        ValueError: If dtypes don't match or shapes are not compatible for broadcasting
    """
    # 1. Validate dtype equality (PyTorch requirement)
    if dtype_a != dtype_b:
        raise ValueError(
            f"Type mismatch in {op_name}: "
            f"Input tensors must have the same dtype. "
            f"{tensor_a_name} has dtype {dtype_a}, "
            f"{tensor_b_name} has dtype {dtype_b}. "
            f"PyTorch arithmetic operations require matching dtypes."
        )

    # 2. If shapes are equal, no broadcasting needed
    if shape_a == shape_b:
        return

    # 3. Validate broadcasting compatibility
    # Convert to tuples for easier manipulation
    shape_a_list = list(shape_a)
    shape_b_list = list(shape_b)

    # Pad shorter shape with 1s on the left (missing dimensions treated as 1)
    max_len = max(len(shape_a_list), len(shape_b_list))
    shape_a_padded = [1] * (max_len - len(shape_a_list)) + shape_a_list
    shape_b_padded = [1] * (max_len - len(shape_b_list)) + shape_b_list

    # Check compatibility from right to left
    incompatible_dims = []
    for i in range(max_len - 1, -1, -1):
        dim_a = shape_a_padded[i]
        dim_b = shape_b_padded[i]

        # Dimensions are compatible if:
        # 1. They are equal, OR
        # 2. One of them is 1
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            incompatible_dims.append((i, dim_a, dim_b))

    if incompatible_dims:
        dim_info = ", ".join([f"dim {d[0]}: {d[1]} vs {d[2]}" for d in incompatible_dims])
        raise ValueError(
            f"Broadcasting error in {op_name}: "
            f"Shapes {shape_a} ({tensor_a_name}) and {shape_b} ({tensor_b_name}) "
            f"are not compatible for broadcasting. "
            f"Incompatible dimensions: {dim_info}. "
            f"Two dimensions are compatible if they are equal OR one is 1."
        )


class AddNode(TIRNode):
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
        a = input_tensors[self.input_names[0]]
        b = input_tensors[self.input_names[1]]

        validate_broadcasting_pytorch_style(
            a.shape, b.shape, a.dtype, b.dtype, self.op_type, self.input_names[0], self.input_names[1]
        )

        return {self.output_names[0]: torch.add(a, b)}


class SubNode(TIRNode):
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

        validate_broadcasting_pytorch_style(
            a.shape, b.shape, a.dtype, b.dtype, self.op_type, self.input_names[0], self.input_names[1]
        )

        return {self.output_names[0]: torch.sub(a, b)}


class MulNode(TIRNode):
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

        validate_broadcasting_pytorch_style(
            a.shape, b.shape, a.dtype, b.dtype, self.op_type, self.input_names[0], self.input_names[1]
        )

        return {self.output_names[0]: torch.mul(a, b)}


class DivNode(TIRNode):
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

        validate_broadcasting_pytorch_style(
            a.shape, b.shape, a.dtype, b.dtype, self.op_type, self.input_names[0], self.input_names[1]
        )

        is_integer_type = not a.dtype.is_floating_point

        if is_integer_type:
            return {self.output_names[0]: torch.div(a, b, rounding_mode="floor")}
        else:
            return {self.output_names[0]: torch.div(a, b)}


class MatMulNode(TIRNode):
    """
    <<<<<<< Current (Your changes)
        Matrix multiplication operation node.
    =======
        Matrix multiplication operation node using PyTorch API.
    >>>>>>> Incoming (Background Agent changes)

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
