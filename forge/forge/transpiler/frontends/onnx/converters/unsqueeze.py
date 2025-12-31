# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Unsqueeze operation converter with opset version support.

This module provides the converter for ONNX Unsqueeze operations, which add
dimensions of size 1 to tensors at specified positions. The converter handles
multiple opset versions with different attribute/input patterns.

Key features:
- Supports opset v1-v12 (axes as attribute) and v13+ (axes as input)
- Axes refer to output tensor dimensions (not input dimensions)
- Validates that axes are unique and within valid range
- Handles negative axis indices
"""
from typing import List, Dict, Any
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.shape import UnsqueezeNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.validation import validate_constant_input
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


class UnsqueezeConverter(OnnxOpConverter):
    """Converter for ONNX Unsqueeze operation with opset version support."""

    @classmethod
    def _normalize_axes(cls, axes: Any, output_rank: int) -> List[int]:
        """
        Normalize axes to positive integers, handling negative indices.

        For Unsqueeze, axes refer to OUTPUT tensor dimensions, not input dimensions.
        Output rank = input rank + len(axes)

        Args:
            axes: List of axis indices (can be negative)
            output_rank: Rank of the output tensor

        Returns:
            List of normalized positive axis indices
        """
        if axes is None:
            return []
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        # Normalize negative indices based on output rank
        return [idx + output_rank if idx < 0 else idx for idx in map(int, axes)]

    @classmethod
    def _get_and_validate_axes(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        graph_proto=None,
        opset_version: int = 1,
    ) -> List[int]:
        """
        Extract, normalize, and validate axes for Unsqueeze operation.

        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dictionary
            attrs: Node attributes
            graph_proto: ONNX graph proto (for accessing initializers)
            opset_version: Opset version

        Returns:
            Normalized list of axis indices

        Raises:
            ValueError: If axes are invalid (missing, duplicates, out of range)
        """
        input_info = list(input_tensors.values())[0]
        input_rank = len(input_info.shape) if input_info.shape else None

        if input_rank is None:
            raise ValueError(f"Cannot determine input rank for Unsqueeze node '{node_proto.name}'")

        # Extract axes from attribute (v1-v12) or input tensor (v13+)
        axes = None
        if opset_version >= 13 and len(node_proto.input) > 1:
            is_valid, axes, error_msg = validate_constant_input(node_proto, input_index=1, graph_proto=graph_proto)
            if not is_valid:
                # Fallback to attribute for backward compatibility
                axes = attrs.get("axes", None)
            elif axes is None:
                # Input exists but is None - check attribute
                axes = attrs.get("axes", None)
        else:
            # v1-v12: axes is attribute
            axes = attrs.get("axes", None)

        # Axes is required for Unsqueeze (unlike Squeeze)
        if axes is None:
            raise ValueError(
                f"Unsqueeze node '{node_proto.name or node_proto.op_type}' requires 'axes' parameter. "
                f"It cannot be omitted."
            )

        # Convert to list if single value
        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        # Calculate output rank: output_rank = input_rank + len(axes)
        output_rank = input_rank + len(axes)

        # Validate original axes are within range [-output_rank, output_rank-1] BEFORE normalization
        for axis in axes:
            if not (-output_rank <= axis < output_rank):
                raise ValueError(
                    f"Unsqueeze node '{node_proto.name or node_proto.op_type}': "
                    f"axis {axis} is out of range [-{output_rank}, {output_rank-1}] "
                    f"(output rank is {output_rank})"
                )

        # Normalize axes (handle negative indices)
        normalized_axes = cls._normalize_axes(axes, output_rank)

        # Verify normalization produced valid positive indices
        for i, (orig_axis, norm_axis) in enumerate(zip(axes, normalized_axes)):
            if not (0 <= norm_axis < output_rank):
                raise ValueError(
                    f"Unsqueeze node '{node_proto.name or node_proto.op_type}': "
                    f"axis {orig_axis} normalized to {norm_axis} is out of range [0, {output_rank-1}] "
                    f"(output rank is {output_rank})"
                )

        # Check for duplicates
        if len(normalized_axes) != len(set(normalized_axes)):
            duplicates = [ax for ax in normalized_axes if normalized_axes.count(ax) > 1]
            raise ValueError(
                f"Unsqueeze node '{node_proto.name or node_proto.op_type}': "
                f"axes contains duplicate values: {set(duplicates)}"
            )

        # Sort axes in ascending order for consistent processing
        # This ensures we insert dimensions from left to right, avoiding index shifting issues
        return sorted(normalized_axes)

    @classmethod
    def _process_unsqueeze(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset_version: int = 1,
    ) -> List:
        """
        Common processing logic for all opset versions.
        Creates multiple UnsqueezeNode instances (one per axis) since UnsqueezeNode only accepts single dim.
        """
        from forge.transpiler.operations.other import IdentityNode

        input_info = list(input_tensors.values())[0]
        input_shape = input_info.shape if input_info.shape else None

        if input_shape is None:
            raise ValueError(f"Cannot determine input shape for Unsqueeze node '{node_proto.name}'")

        node_name = node_proto.name or f"Unsqueeze_{node_index}"

        # Get and validate axes
        axes = cls._get_and_validate_axes(node_proto, input_tensors, attrs, graph_proto, opset_version)

        # Build OrderedDict for inputs and outputs
        # v13+ uses only data input, v1-v12 uses all inputs
        input_names = [node_proto.input[0]] if opset_version >= 13 else None
        input_dict, output_dict = build_input_output_dicts(
            node_proto, input_tensors, output_tensors, input_names=input_names
        )

        # If no axes to unsqueeze, return Identity
        if not axes:
            return [IdentityNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]

        # Prepare inputs (v13+ uses only data input, axes is embedded)
        # For v13+, axes is an input but we only need the data input for UnsqueezeNode
        # input_dict already contains all inputs, so we can use it directly
        # The data input is always the first input
        current_input_dict = OrderedDict()
        data_input_name = node_proto.input[0]
        if data_input_name in input_dict:
            current_input_dict[data_input_name] = input_dict[data_input_name]
        else:
            # Fallback: try to get from input_tensors directly
            if data_input_name in input_tensors:
                current_input_dict[data_input_name] = input_tensors[data_input_name]
            else:
                raise ValueError(
                    f"Cannot find TensorInfo for input '{data_input_name}'. "
                    f"Available in input_dict: {list(input_dict.keys())}, "
                    f"Available in input_tensors: {list(input_tensors.keys())}"
                )

        # Create UnsqueezeNode for each axis
        nodes = []
        current_shape = list(input_shape)
        onnx_dtype = getattr(input_info, "onnx_dtype", None)

        for axis_idx, axis in enumerate(axes):
            is_last = axis_idx == len(axes) - 1

            if is_last:
                node_outputs = [node_proto.output[0]]
                node_output_tensors = output_tensors.copy()
            else:
                # Create intermediate output
                intermediate_name = f"{node_name}_intermediate_{axis_idx}"
                node_outputs = [intermediate_name]

                # Calculate intermediate shape: insert dimension of size 1 at position 'axis'
                intermediate_shape = list(current_shape)
                intermediate_shape.insert(axis, 1)
                intermediate_shape = tuple(intermediate_shape)

                node_output_tensors = {
                    intermediate_name: TensorInfo(
                        name=intermediate_name, shape=intermediate_shape, onnx_dtype=onnx_dtype
                    )
                }

            # Build OrderedDict for this UnsqueezeNode
            # Pass current_input_dict as input_tensors - it contains the actual input (original or intermediate)
            # For intermediate nodes, only pass the data input name (not axes) to avoid lookup errors
            data_input_name = list(current_input_dict.keys())[0] if current_input_dict else node_proto.input[0]
            node_input_dict, node_output_dict = build_input_output_dicts(
                node_proto,
                current_input_dict,
                node_output_tensors,
                input_names=[data_input_name],  # Only include data input, not axes
                output_names=node_outputs,
            )

            # Create UnsqueezeNode with single dim (int, not list/tuple)
            nodes.append(
                UnsqueezeNode.create(
                    name=f"{node_name}_axis_{axis_idx}",
                    inputs=node_input_dict,
                    outputs=node_output_dict,
                    dim=axis,  # Single int, matching torch.unsqueeze API
                )
            )

            # Update for next iteration
            if not is_last:
                current_input_dict = node_output_dict.copy()
                # Update shape: insert dimension at position 'axis'
                current_shape.insert(axis, 1)

        return nodes

    @classmethod
    def convert(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset: int = 1,
    ) -> List:
        """
        Unsqueeze converter - single method handles all versions using opset parameter.

        - Opset v1-v10: axes as attribute (non-negative integers only)
        - Opset v11-v12: axes as attribute (supports negative indices)
        - Opset v13+: axes as input tensor (second input)
        """
        return cls._process_unsqueeze(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset)
