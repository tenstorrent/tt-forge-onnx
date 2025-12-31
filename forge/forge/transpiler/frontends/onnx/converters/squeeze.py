# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Squeeze operation converter with opset version support.

This module provides the converter for ONNX Squeeze operations, which remove
dimensions of size 1 from tensors. The converter handles multiple opset versions
with different attribute/input patterns.

Key features:
- Supports opset v1-v12 (axes as attribute) and v13+ (axes as input)
- Can auto-detect all size-1 dimensions if axes not provided
- Optimizes no-op squeezes to IdentityNode
- Handles negative axis indices
"""
from typing import List, Dict, Any, Optional
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.shape import SqueezeNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.validation import validate_constant_input
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


class SqueezeConverter(OnnxOpConverter):
    """Converter for ONNX Squeeze operation with opset version support."""

    @classmethod
    def _normalize_axes(cls, axes: Any, input_rank: int) -> List[int]:
        """Normalize axes to positive integers, handling negative indices."""
        if axes is None:
            return []
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        return [idx + input_rank if idx < 0 else idx for idx in map(int, axes)]

    @classmethod
    def _get_axes_to_squeeze(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        graph_proto=None,
        opset_version: int = 1,
    ) -> Optional[List[int]]:
        """Extract and normalize axes from attribute (v1-v11) or input tensor (v13+).

        Returns:
            - None: axes not provided, should auto-detect all size-1 dims
            - []: axes explicitly empty, should do no squeeze (Identity)
            - List[int]: normalized axes to squeeze
        """
        input_info = list(input_tensors.values())[0]
        input_rank = len(input_info.shape) if input_info.shape else None

        if input_rank is None:
            raise ValueError(f"Cannot determine input rank for Squeeze node '{node_proto.name}'")

        if opset_version >= 13 and len(node_proto.input) > 1:
            is_valid, axes, _ = validate_constant_input(node_proto, input_index=1, graph_proto=graph_proto)
            if is_valid:
                if axes is not None:
                    # Normalize and return (empty list means no squeeze)
                    return cls._normalize_axes(axes, input_rank)
                # axes input exists but is None -> not provided, auto-detect
                return None

        axes = attrs.get("axes")
        if axes is not None:
            # Normalize and return (empty list means no squeeze)
            return cls._normalize_axes(axes, input_rank)
        # axes not provided -> auto-detect all size-1 dims
        return None

    @classmethod
    def _process_squeeze(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset_version: int = 1,
    ) -> List:
        """Common processing logic for all opset versions."""
        from forge.transpiler.operations.other import IdentityNode

        input_info = list(input_tensors.values())[0]
        input_shape = input_info.shape
        if input_shape is None:
            raise ValueError(f"Cannot determine input shape for Squeeze node '{node_proto.name}'")

        node_name = node_proto.name or f"Squeeze_{node_index}"

        # Get and process axes
        axes = cls._get_axes_to_squeeze(node_proto, input_tensors, attrs, graph_proto, opset_version)
        # None means auto-detect all size-1 dims, [] means no squeeze (Identity)
        if axes is None:
            axes = [i for i, size in enumerate(input_shape) if size == 1]

        if axes:
            # Validate and deduplicate
            for axis in axes:
                if not (0 <= axis < len(input_shape) and input_shape[axis] == 1):
                    raise ValueError(
                        f"Squeeze node '{node_name}': cannot squeeze axis {axis} "
                        f"(size={input_shape[axis] if 0 <= axis < len(input_shape) else 'out of range'})"
                    )
            axes = sorted(set(axes), reverse=True)

        # Build OrderedDict for inputs and outputs
        # v13+ uses only data input, v1-v12 uses all inputs
        input_names = [node_proto.input[0]] if opset_version >= 13 else None
        input_dict, output_dict = build_input_output_dicts(
            node_proto, input_tensors, output_tensors, input_names=input_names
        )

        # Create Identity if no squeeze needed, otherwise SqueezeNode
        if not axes:
            return [IdentityNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]

        dim_value = tuple(axes) if len(axes) > 1 else axes[0]
        return [SqueezeNode.create(name=node_name, inputs=input_dict, outputs=output_dict, dim=dim_value)]

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
        Squeeze converter - single method handles all versions using opset parameter.

        - Opset v1-v11: axes as attribute
        - Opset v13+: axes as optional input tensor
        """
        return cls._process_squeeze(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset)
