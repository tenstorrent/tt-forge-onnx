# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Activation operations: Relu, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyRelu, Dropout
"""
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Optional

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo


class ReluNode(TIRNode):
    """
    PyTorch-like Relu operation.
    """

    @staticmethod
    def create(name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo]) -> "ReluNode":
        """Static factory method to create a ReluNode."""
        return ReluNode(name=name, op_type="Relu", inputs=inputs, outputs=outputs, attrs={}, forge_op_name="Relu")

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate ReLU operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        return {self.output_names[0]: F.relu(x)}


class SigmoidNode(TIRNode):
    """
    PyTorch-like Sigmoid operation.
    """

    @staticmethod
    def create(name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo]) -> "SigmoidNode":
        """Static factory method to create a SigmoidNode."""
        return SigmoidNode(
            name=name, op_type="Sigmoid", inputs=inputs, outputs=outputs, attrs={}, forge_op_name="Sigmoid"
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Sigmoid operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        return {self.output_names[0]: torch.sigmoid(x)}


class TanhNode(TIRNode):
    """
    PyTorch-like Tanh operation.
    """

    @staticmethod
    def create(name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo]) -> "TanhNode":
        """Static factory method to create a TanhNode."""
        return TanhNode(name=name, op_type="Tanh", inputs=inputs, outputs=outputs, attrs={}, forge_op_name="Tanh")

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Tanh operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        return {self.output_names[0]: F.tanh(x)}


class SoftmaxNode(TIRNode):
    """
    PyTorch-like Softmax operation.

    Similar to torch.softmax, the dim parameter must be explicitly provided.
    If dim is None, an error will be raised.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        dim: Optional[int] = None,
    ) -> "SoftmaxNode":
        """
        Static factory method to create a SoftmaxNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
            dim: Dimension along which to apply softmax (must be provided, cannot be None)

        Raises:
            ValueError: If dim is None
        """
        if dim is None:
            raise ValueError("SoftmaxNode requires 'dim' parameter to be specified (cannot be None)")
        return SoftmaxNode(
            name=name, op_type="Softmax", inputs=inputs, outputs=outputs, attrs={"dim": dim}, forge_op_name="Softmax"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Adds stable=True for Forge Softmax operation (default behavior).

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes
        """
        forge_attrs = attrs.copy()
        if "stable" not in forge_attrs:
            forge_attrs["stable"] = True
        return forge_attrs

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Softmax operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If dim attribute is None
        """
        x = input_tensors[self.input_names[0]]
        dim = self.attrs.get("dim", None)
        if dim is None:
            raise ValueError("SoftmaxNode requires 'dim' attribute to be set (cannot be None)")
        return {self.output_names[0]: F.softmax(x, dim=dim)}


class LogSoftmaxNode(TIRNode):
    """
    PyTorch-like LogSoftmax operation.

    Similar to torch.log_softmax, the dim parameter must be explicitly provided.
    If dim is None, an error will be raised.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        dim: Optional[int] = None,
    ) -> "LogSoftmaxNode":
        """
        Static factory method to create a LogSoftmaxNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
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
            attrs={"dim": dim},
            forge_op_name="LogSoftmax",
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Adds stable=True for Forge LogSoftmax operation (default behavior).

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes
        """
        forge_attrs = attrs.copy()
        if "stable" not in forge_attrs:
            forge_attrs["stable"] = True
        return forge_attrs

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate LogSoftmax operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If dim attribute is None
        """
        x = input_tensors[self.input_names[0]]
        dim = self.attrs.get("dim", None)
        if dim is None:
            raise ValueError("LogSoftmaxNode requires 'dim' attribute to be set (cannot be None)")
        return {self.output_names[0]: F.log_softmax(x, dim=dim)}


class LeakyReluNode(TIRNode):
    """
    PyTorch-like LeakyRelu operation.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        negative_slope: float = 0.01,
    ) -> "LeakyReluNode":
        """Static factory method to create a LeakyReluNode."""
        return LeakyReluNode(
            name=name,
            op_type="LeakyRelu",
            inputs=inputs,
            outputs=outputs,
            attrs={"negative_slope": negative_slope},
            forge_op_name="LeakyRelu",
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Maps PyTorch 'negative_slope' to Forge 'alpha'.

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes
        """
        return {"alpha": attrs.get("negative_slope", 0.01)}

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate LeakyReLU operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        negative_slope = self.attrs.get("negative_slope", 0.01)
        return {self.output_names[0]: F.leaky_relu(x, negative_slope=negative_slope)}


class DropoutNode(TIRNode):
    """
    PyTorch-like Dropout operation.

    Supports different ONNX versions:
    - v1-v6: Uses `is_test` attribute and `ratio` attribute
    - v7-v10: Uses `ratio` attribute, training mode from graph context
    - v12+: Uses `ratio` and `training_mode` inputs, `seed` attribute
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        p: float = 0.5,
        training: bool = True,
        seed: int = 0,
    ) -> "DropoutNode":
        """
        Static factory method to create a DropoutNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo (data, [ratio], [training_mode])
            outputs: OrderedDict mapping output names to TensorInfo (output, [mask])
            p: Dropout probability (default: 0.5)
            training: Training mode flag (default: True)
            seed: Random seed (default: 0)
        """
        return DropoutNode(
            name=name,
            op_type="Dropout",
            inputs=inputs,
            outputs=outputs,
            attrs={"p": p, "training": training, "seed": seed},
            forge_op_name="Dropout",
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes (p, training, seed)
        """
        return {"p": attrs.get("p", 0.5), "training": attrs.get("training", True), "seed": attrs.get("seed", 0)}

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Dropout operation using PyTorch.

        In training mode, applies dropout with scaling. In inference mode, returns input unchanged.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        p = self.attrs.get("p", 0.5)
        training = self.attrs.get("training", True)
        seed = self.attrs.get("seed", 0)

        torch.manual_seed(seed)

        if training:
            output = F.dropout(x, p=p, training=True)
        else:
            output = x

        return {self.output_names[0]: output}
