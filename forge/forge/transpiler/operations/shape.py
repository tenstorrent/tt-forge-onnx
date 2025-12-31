# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shape/Reshape operations: Flatten, Reshape, Transpose, Squeeze, Unsqueeze
"""
import torch
from collections import OrderedDict
from typing import Dict, List, Union

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo


class ReshapeNode(TIRNode):
    """
    PyTorch-like Reshape operation.
    Takes one input tensor and shape as attribute (matching torch.reshape API).
    Supports ONNX Reshape features:
    - -1 for inferred dimension
    - Shape is already resolved in converter (no 0 or empty shapes)
    """

    @staticmethod
    def create(
        name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo], shape: tuple
    ) -> "ReshapeNode":
        """
        Static factory method to create a ReshapeNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
            shape: Target shape tuple (already resolved, may contain -1 for inference)
        """
        return ReshapeNode(
            name=name,
            op_type="Reshape",
            inputs=inputs,
            outputs=outputs,
            attrs={"shape": shape},
            forge_op_name="Reshape",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Reshape operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If reshape operation fails
        """
        x = input_tensors[self.input_names[0]]
        shape = self.attrs.get("shape", None)

        if shape is None:
            output_info = list(self.outputs.values())[0]
            if output_info and output_info.shape:
                shape = tuple(s if s is not None else x.shape[i] for i, s in enumerate(output_info.shape))
            else:
                shape = x.shape

        try:
            result = torch.reshape(x, shape)
        except RuntimeError as e:
            raise ValueError(f"Reshape failed: {e}. Input shape: {x.shape}, Target shape: {shape}")

        return {self.output_names[0]: result}


class TransposeNode(TIRNode):
    """
    PyTorch-like Transpose operation that swaps two dimensions.
    For multi-dimensional transpositions, create multiple TransposeNode instances.
    """

    @staticmethod
    def create(
        name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo], dim0: int, dim1: int
    ) -> "TransposeNode":
        """Static factory method to create a TransposeNode."""
        return TransposeNode(
            name=name,
            op_type="Transpose",
            inputs=inputs,
            outputs=outputs,
            attrs={"dim0": dim0, "dim1": dim1},
            forge_op_name="Transpose",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Transpose operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        dim0 = self.attrs["dim0"]
        dim1 = self.attrs["dim1"]
        return {self.output_names[0]: torch.transpose(x, dim0, dim1)}


class SqueezeNode(TIRNode):
    """
    PyTorch-like Squeeze operation.
    Takes one input tensor and dim as attribute (matching torch.squeeze API).
    dim can be int or tuple/list of ints (torch.squeeze accepts both).
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        dim: Union[int, tuple, list],
    ) -> "SqueezeNode":
        """Static factory method to create a SqueezeNode."""
        return SqueezeNode(
            name=name, op_type="Squeeze", inputs=inputs, outputs=outputs, attrs={"dim": dim}, forge_op_name="Squeeze"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Forge Squeeze requires dim as int (single dimension), not tuple.
        If multiple dims provided, uses first one.

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes
        """
        forge_attrs = {}
        if "dim" in attrs:
            dim = attrs["dim"]
            if isinstance(dim, (list, tuple)):
                forge_attrs["dim"] = dim[0] if len(dim) > 0 else 0
            else:
                forge_attrs["dim"] = dim
        return forge_attrs

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Squeeze operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        dim = self.attrs.get("dim", None)
        if dim is not None:
            if isinstance(dim, list):
                dim = tuple(dim)
            return {self.output_names[0]: torch.squeeze(x, dim=dim)}
        else:
            return {self.output_names[0]: torch.squeeze(x)}


class UnsqueezeNode(TIRNode):
    """
    PyTorch-like Unsqueeze operation.
    Takes one input tensor and dim as attribute (matching torch.unsqueeze API).
    Forge Unsqueeze requires dim as attribute (single int, required).
    """

    @staticmethod
    def create(
        name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo], dim: int
    ) -> "UnsqueezeNode":
        """Static factory method to create an UnsqueezeNode."""
        return UnsqueezeNode(
            name=name,
            op_type="Unsqueeze",
            inputs=inputs,
            outputs=outputs,
            attrs={"dim": dim},
            forge_op_name="Unsqueeze",
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Forge Unsqueeze requires dim as int (required, no default).

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes

        Raises:
            ValueError: If dim is None or not an int
        """
        forge_attrs = {}
        if "dim" not in attrs or attrs["dim"] is None:
            raise ValueError(f"UnsqueezeNode '{self.name}': 'dim' attribute is required and cannot be None")

        dim = attrs["dim"]
        if not isinstance(dim, int):
            raise ValueError(f"UnsqueezeNode '{self.name}': 'dim' must be an int, got {type(dim).__name__}")

        forge_attrs["dim"] = dim
        return forge_attrs

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Unsqueeze operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If dim is None or not an int
        """
        x = input_tensors[self.input_names[0]]
        dim = self.attrs.get("dim", None)

        if dim is None:
            raise ValueError(f"UnsqueezeNode '{self.name}': 'dim' attribute is required and cannot be None")

        if not isinstance(dim, int):
            raise ValueError(f"UnsqueezeNode '{self.name}': 'dim' must be an int, got {type(dim).__name__}")

        return {self.output_names[0]: torch.unsqueeze(x, dim=dim)}


class SplitNode(TIRNode):
    """
    PyTorch-like Split operation.

    Similar to torch.split() which returns a tuple of tensors.
    Represents a split operation that produces multiple outputs.
    Note: Split is not available in Forge and must be decomposed before code generation.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        split_sizes: List[int] = None,
        dim: int = 0,
    ) -> "SplitNode":
        """
        Static factory method to create a SplitNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo (single input)
            outputs: OrderedDict mapping output names to TensorInfo (multiple outputs)
            split_sizes: List of sizes for each split (e.g., [2, 3, 5])
            dim: Dimension along which to split
        """
        return SplitNode(
            name=name,
            op_type="Split",
            inputs=inputs,
            outputs=outputs,  # Multiple outputs
            attrs={"split_sizes": split_sizes, "dim": dim},
            forge_op_name=None,  # Split not available in Forge (must be decomposed)
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Split operation using PyTorch.

        Returns dictionary with all output tensors.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output names to result tensors
        """
        x = input_tensors[self.input_names[0]]
        split_sizes = self.attrs.get("split_sizes", None)
        dim = self.attrs.get("dim", 0)

        if split_sizes is not None:
            if isinstance(split_sizes, list):
                split_sizes = tuple(split_sizes)
            splits = torch.split(x, split_sizes, dim=dim)
        else:
            dim_size = x.shape[dim] if dim < len(x.shape) else x.shape[0]
            num_outputs = len(self.outputs)
            split_size = dim_size // num_outputs
            splits = torch.split(x, split_size, dim=dim)

        result = {}
        for i, output_name in enumerate(self.output_names):
            if i < len(splits):
                result[output_name] = splits[i]
            else:
                result[output_name] = splits[-1]
        return result
