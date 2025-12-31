# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Split operation converter with opset version support.

This module provides the converter for ONNX Split operations, which split a
tensor into multiple tensors along a specified axis. The converter handles
multiple opset versions with different attribute/input patterns.

Key features:
- Supports opset v1-v12 (split as attribute) and v13+ (split as input)
- Split sizes can be provided as a list or inferred from output count
- Validates that split sizes sum to input dimension size
- Creates multiple output tensors as specified
"""
from typing import List, Dict, Any
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.shape import SplitNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.validation import validate_constant_input
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


class SplitConverter(OnnxOpConverter):
    """Converter for ONNX Split operation with opset version support."""

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
        Split converter with opset-based split extraction.

        - Opset v1-v12: split as attribute
        - Opset v13+: split as input tensor (second input, optional)
        """
        node_name = node_proto.name if node_proto.name else f"Split_{node_index}"
        axis = attrs.get("axis", 0)  # Dimension along which to split

        if opset < 13:
            # v1-v12: split as attribute
            split_sizes = attrs.get("split", None)  # ONNX attribute name
            inputs = list(node_proto.input)
        else:
            # v13+: split as input tensor (second input, optional)
            is_valid, split_sizes, error_msg = validate_constant_input(
                node_proto, input_index=1, graph_proto=graph_proto
            )

            # Convert split_sizes to list if it's a tuple or scalar
            if split_sizes is not None:
                if isinstance(split_sizes, (list, tuple)):
                    split_sizes = [int(x) for x in split_sizes]
                else:
                    split_sizes = [int(split_sizes)]

            # v13+ uses only data input, split is embedded
            inputs = [node_proto.input[0]]

        # Build OrderedDict for inputs and outputs
        split_input_dict, split_output_dict = build_input_output_dicts(
            node_proto, input_tensors, output_tensors, input_names=inputs
        )

        # Create one SplitNode representing the entire split operation
        split_node = SplitNode.create(
            name=node_name, inputs=split_input_dict, outputs=split_output_dict, split_sizes=split_sizes, dim=axis
        )

        return [split_node]
