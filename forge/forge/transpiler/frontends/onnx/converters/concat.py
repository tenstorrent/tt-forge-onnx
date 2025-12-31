# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Concat operation converter with opset version support.

This module provides the converter for ONNX Concat operations, which concatenate
multiple tensors along a specified axis. The converter handles opset version
differences in axis attribute defaults and validates input compatibility.

Key features:
- Supports concatenation along any axis
- Handles opset version differences in axis default (v1: default=1, v4+: default=0)
- Validates that all inputs have compatible shapes (same rank, same dims except axis)
- Optimizes single-input case to IdentityNode
"""
from typing import List, Dict, Any
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.other import ConcatNode, IdentityNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


class ConcatConverter(OnnxOpConverter):
    """Converter for ONNX Concat operation with opset version support."""

    @classmethod
    def _normalize_axis(cls, axis: int, rank: int) -> int:
        """
        Normalize axis to positive integer, handling negative indices.

        Args:
            axis: Axis index (can be negative)
            rank: Rank of the input tensors

        Returns:
            Normalized positive axis index
        """
        if axis < 0:
            return axis + rank
        return axis

    @classmethod
    def _validate_and_normalize_axis(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        opset_version: int,
    ) -> int:
        """
        Extract, validate, and normalize the axis attribute.

        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dict
            attrs: Node attributes
            opset_version: Opset version

        Returns:
            Normalized axis (positive integer)
        """
        # Get axis from attributes
        axis = attrs.get("axis", None)

        # v1: axis is optional, defaults to 1
        if opset_version == 1:
            if axis is None:
                axis = 1
        # v4+: axis is required
        else:
            if axis is None:
                raise ValueError(
                    f"Concat node '{node_proto.name or node_proto.op_type}': "
                    f"'axis' attribute is required for opset version {opset_version}"
                )

        # Convert to int
        axis = int(axis)

        # Get input rank from first input tensor
        if not input_tensors:
            raise ValueError(f"Concat node '{node_proto.name or node_proto.op_type}': " "No input tensors provided")

        first_input_info = list(input_tensors.values())[0]
        input_rank = len(first_input_info.shape) if first_input_info.shape else None

        if input_rank is None:
            raise ValueError(f"Concat node '{node_proto.name or node_proto.op_type}': " "Cannot determine input rank")

        # Validate axis range BEFORE normalization
        # v1-v3: only non-negative indices [0, rank-1]
        if opset_version < 4:
            if axis < 0:
                raise ValueError(
                    f"Concat node '{node_proto.name or node_proto.op_type}': "
                    f"Negative axis {axis} not supported in opset version {opset_version}. "
                    f"Valid range is [0, {input_rank-1}]"
                )
            if not (0 <= axis < input_rank):
                raise ValueError(
                    f"Concat node '{node_proto.name or node_proto.op_type}': "
                    f"axis {axis} is out of range [0, {input_rank-1}] "
                    f"(input rank is {input_rank})"
                )
        # v4+: supports negative indices [-rank, rank-1]
        else:
            if not (-input_rank <= axis < input_rank):
                raise ValueError(
                    f"Concat node '{node_proto.name or node_proto.op_type}': "
                    f"axis {axis} is out of range [-{input_rank}, {input_rank-1}] "
                    f"(input rank is {input_rank})"
                )

        # Normalize negative indices (v11+ supports negative, but we normalize for all)
        normalized_axis = cls._normalize_axis(axis, input_rank)

        # Verify normalized axis is valid
        if not (0 <= normalized_axis < input_rank):
            raise ValueError(
                f"Concat node '{node_proto.name or node_proto.op_type}': "
                f"axis {axis} normalized to {normalized_axis} is out of range [0, {input_rank-1}] "
                f"(input rank is {input_rank})"
            )

        return normalized_axis

    @classmethod
    def _validate_inputs(cls, node_proto: NodeProto, input_tensors: OrderedDict[str, TensorInfo], axis: int) -> None:
        """
        Validate that all input tensors have compatible shapes for concatenation.

        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dict
            axis: Normalized axis to concatenate along
        """
        if len(input_tensors) < 1:
            raise ValueError(
                f"Concat node '{node_proto.name or node_proto.op_type}': " "At least one input tensor is required"
            )

        # Get shapes and ranks
        input_infos = list(input_tensors.values())
        first_shape = input_infos[0].shape
        first_rank = len(first_shape) if first_shape else None

        if first_rank is None:
            raise ValueError(
                f"Concat node '{node_proto.name or node_proto.op_type}': "
                "Cannot determine shape of first input tensor"
            )

        # Validate all inputs have same rank
        for i, input_info in enumerate(input_infos[1:], start=1):
            input_shape = input_info.shape
            input_rank = len(input_shape) if input_shape else None

            if input_rank is None:
                raise ValueError(
                    f"Concat node '{node_proto.name or node_proto.op_type}': "
                    f"Cannot determine shape of input tensor {i}"
                )

            if input_rank != first_rank:
                raise ValueError(
                    f"Concat node '{node_proto.name or node_proto.op_type}': "
                    f"All input tensors must have the same rank. "
                    f"Input 0 has rank {first_rank}, input {i} has rank {input_rank}"
                )

        # Validate all inputs have same shape except along concatenation axis
        for i, input_info in enumerate(input_infos[1:], start=1):
            input_shape = input_info.shape
            if input_shape is None:
                continue  # Skip if shape is unknown

            for dim_idx in range(first_rank):
                if dim_idx == axis:
                    continue  # Skip concatenation axis

                if dim_idx < len(input_shape) and dim_idx < len(first_shape):
                    if first_shape[dim_idx] is not None and input_shape[dim_idx] is not None:
                        if first_shape[dim_idx] != input_shape[dim_idx]:
                            raise ValueError(
                                f"Concat node '{node_proto.name or node_proto.op_type}': "
                                f"All input tensors must have the same shape in all dimensions "
                                f"except the concatenation axis (axis={axis}). "
                                f"Dimension {dim_idx}: input 0 has size {first_shape[dim_idx]}, "
                                f"input {i} has size {input_shape[dim_idx]}"
                            )

    @classmethod
    def _process_concat(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        opset_version: int,
    ) -> List:
        """
        Common processing logic for all opset versions.

        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            attrs: Node attributes
            node_index: Node index
            opset_version: Opset version

        Returns:
            List of TIR nodes (ConcatNode or IdentityNode)
        """
        node_name = node_proto.name or f"Concat_{node_index}"

        # Validate and normalize axis
        axis = cls._validate_and_normalize_axis(node_proto, input_tensors, attrs, opset_version)

        # Validate input compatibility
        cls._validate_inputs(node_proto, input_tensors, axis)

        # Handle single input case (no-op, return Identity)
        if len(node_proto.input) == 1:
            identity_input_dict, identity_output_dict = build_input_output_dicts(
                node_proto, input_tensors, output_tensors
            )

            return [IdentityNode.create(name=node_name, inputs=identity_input_dict, outputs=identity_output_dict)]

        # Build OrderedDict for inputs and outputs
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        # Create ConcatNode with validated axis
        return [
            ConcatNode.create(
                name=node_name, inputs=input_dict, outputs=output_dict, dim=axis  # Must be int, no default value
            )
        ]

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
        Concat converter - single method handles all versions using opset parameter.

        - Opset v1-v3: axis as attribute (optional, defaults to 1), only non-negative indices
        - Opset v4-v10: axis as attribute (required), only non-negative indices
        - Opset v11+: axis as attribute (required), supports negative indices
        """
        return cls._process_concat(node_proto, input_tensors, output_tensors, attrs, node_index, opset)
