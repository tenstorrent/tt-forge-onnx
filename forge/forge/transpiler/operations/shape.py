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
from forge.transpiler.operations.shape_mixins import (
    ReshapeShape,
    TransposeShape,
    SqueezeShape,
    UnsqueezeShape,
    BroadcastToShape,
    SplitShape,
)


class ReshapeNode(ReshapeShape, TIRNode):
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


class TransposeNode(TransposeShape, TIRNode):
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


class SqueezeNode(SqueezeShape, TIRNode):
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


class UnsqueezeNode(UnsqueezeShape, TIRNode):
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


class BroadcastNode(BroadcastToShape, TIRNode):
    """
    PyTorch-like Broadcast operation.

    Similar to torch.broadcast_to(input, shape), but constrained to single-axis broadcasting.
    This matches Forge's Broadcast op which only supports broadcasting along one dimension.

    The node validates that only one dimension differs between input and output shapes.
    The dimension and size are computed automatically from the shape difference.
    """

    @staticmethod
    def _normalize_shapes(input_shape: tuple, output_shape: tuple) -> tuple:
        """
        Normalize shapes to same rank by right-aligning (pad shorter with 1s on left).

        Returns:
            Tuple of (normalized_input_shape, normalized_output_shape, dim_offset)
            dim_offset: Offset to apply to dimension index when input was padded
        """
        input_dims = len(input_shape)
        output_dims = len(output_shape)

        if input_dims < output_dims:
            return (1,) * (output_dims - input_dims) + input_shape, output_shape, output_dims - input_dims
        elif output_dims < input_dims:
            return input_shape, (1,) * (input_dims - output_dims) + output_shape, 0
        else:
            return input_shape, output_shape, 0

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        output_shape: tuple,
    ) -> "BroadcastNode":
        """
        Static factory method to create a BroadcastNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo (single input)
            outputs: OrderedDict mapping output names to TensorInfo (single output)
            output_shape: Target output shape tuple

        Returns:
            BroadcastNode instance

        Raises:
            ValueError: If input and output shapes differ in more than one dimension,
                       or if shapes are incompatible for broadcasting
        """
        if len(inputs) != 1:
            raise ValueError(f"BroadcastNode '{name}': Expected exactly 1 input, got {len(inputs)}")

        if len(outputs) != 1:
            raise ValueError(f"BroadcastNode '{name}': Expected exactly 1 output, got {len(outputs)}")

        input_info = list(inputs.values())[0]
        input_shape = input_info.shape if input_info.shape else None

        if input_shape is None:
            raise ValueError(f"BroadcastNode '{name}': Cannot determine input shape")

        # Normalize shapes to same rank
        input_shape_normalized, output_shape_normalized, dim_offset = BroadcastNode._normalize_shapes(
            input_shape, output_shape
        )

        # Validate broadcasting compatibility and find differing dimension
        differing_dim = None
        for i, (in_dim, out_dim) in enumerate(zip(input_shape_normalized, output_shape_normalized)):
            # Check compatibility: dimensions must be equal, or one must be 1
            if in_dim != out_dim:
                if in_dim != 1 and out_dim != 1:
                    raise ValueError(
                        f"BroadcastNode '{name}': Incompatible dimensions at index {i}: "
                        f"input={in_dim}, output={out_dim}. "
                        f"For broadcasting, one dimension must be 1 or both must be equal."
                    )
                if differing_dim is not None:
                    raise ValueError(
                        f"BroadcastNode '{name}': Multiple dimensions differ between input and output shapes. "
                        f"BroadcastNode only supports single-axis broadcasting. "
                        f"Input shape: {input_shape}, Output shape: {output_shape}. "
                        f"Consider decomposing into multiple BroadcastNode instances."
                    )
                differing_dim = i

        # Compute and store Forge attributes during creation to avoid recomputation
        attrs = {"output_shape": output_shape}
        if differing_dim is not None:
            # Store computed dim and size for Forge conversion
            attrs["dim"] = differing_dim - dim_offset  # Adjust for padding offset
            attrs["size"] = output_shape_normalized[differing_dim]

        return BroadcastNode(
            name=name,
            op_type="Broadcast",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            forge_op_name="Broadcast",
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Uses pre-computed dim and size from create() to avoid recomputation.
        forge.op.Broadcast signature: Broadcast(name, operandA, dim, shape)
        where 'shape' is the output length of the broadcast dimension.

        Args:
            attrs: Dictionary containing output_shape, dim, and size (computed in create())

        Returns:
            Dictionary of Forge-specific attributes with dim and size

        Raises:
            ValueError: If dim or size are missing (should not happen if create() was used)
        """
        # Use pre-computed values from create()
        # forge.op.Broadcast signature: Broadcast(name, operandA, dim, shape)
        if "dim" in attrs and "size" in attrs:
            return {"dim": attrs["dim"], "shape": attrs["size"]}

        # Fallback: recompute if not available (shouldn't happen in normal flow)
        output_shape = attrs.get("output_shape")
        if output_shape is None:
            output_info = list(self.outputs.values())[0]
            if output_info and output_info.shape:
                output_shape = tuple(output_info.shape)
            else:
                raise ValueError(f"BroadcastNode '{self.name}': Cannot determine output_shape")

        input_info = list(self.inputs.values())[0]
        input_shape = input_info.shape if input_info.shape else None

        if input_shape is None:
            raise ValueError(f"BroadcastNode '{self.name}': Cannot determine input shape")

        # Recompute (fallback path)
        input_shape_normalized, output_shape_normalized, dim_offset = BroadcastNode._normalize_shapes(
            input_shape, output_shape
        )

        differing_dim = None
        for i, (in_dim, out_dim) in enumerate(zip(input_shape_normalized, output_shape_normalized)):
            if in_dim != out_dim:
                if differing_dim is not None:
                    raise ValueError(
                        f"BroadcastNode '{self.name}': Multiple dimensions differ. "
                        f"This should have been caught in create()."
                    )
                differing_dim = i

        if differing_dim is None:
            raise ValueError(
                f"BroadcastNode '{self.name}': No dimension differs. "
                f"Input and output shapes are identical. Use IdentityNode instead."
            )

        return {
            "dim": differing_dim - dim_offset,
            "shape": output_shape_normalized[differing_dim],
        }

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Broadcast operation using PyTorch.

        Uses torch.broadcast_to to perform the broadcasting.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If broadcasting fails
        """
        x = input_tensors[self.input_names[0]]
        output_shape = self.attrs.get("output_shape")

        # output_shape should always be set from create(), but fallback to output info if needed
        if output_shape is None:
            output_info = list(self.outputs.values())[0]
            if output_info and output_info.shape:
                output_shape = tuple(output_info.shape)
            else:
                raise ValueError(f"BroadcastNode '{self.name}': Cannot determine output_shape")

        try:
            result = torch.broadcast_to(x, output_shape)
        except RuntimeError as e:
            raise ValueError(
                f"BroadcastNode '{self.name}': Broadcasting failed: {e}. "
                f"Input shape: {x.shape}, Target shape: {output_shape}"
            )

        return {self.output_names[0]: result}


class SplitNode(SplitShape, TIRNode):
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
