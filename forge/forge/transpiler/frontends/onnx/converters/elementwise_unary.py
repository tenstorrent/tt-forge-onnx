# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Element-wise Unary operation converters.

This module provides converters for ONNX element-wise unary operations:
- Erf: Error function operation
- Future unary operations can be added here (e.g., Exp, Log, Abs, etc.)

Key features:
- Handles opset version differences where applicable
- Validates input/output shapes (output always matches input for unary ops)
- Unified converter for all element-wise unary operations
"""
from typing import List, Dict, Any
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.activation import ErfNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


class UnaryOpConverter(OnnxOpConverter):
    """
    Unified converter for element-wise unary operations: Erf, and future operations like Exp, Log, Abs, etc.

    This converter handles all element-wise unary operations using a single implementation,
    while maintaining separate operator nodes for each operation type.

    Unary operations use PyTorch API (torch.erf, torch.exp, torch.log, torch.abs, etc.).

    All operations preserve input shape and dtype:
    - Output shape always matches input shape exactly
    - Output dtype always matches input dtype
    - No broadcasting or dimension changes occur
    """

    # Mapping from ONNX op type to corresponding node class
    _OP_NODE_MAP = {
        "Erf": ErfNode,
        # Add more unary operations here as they are implemented
        # "Exp": ExpNode,
        # "Log": LogNode,
        # "Abs": AbsNode,
    }

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
        tir_graph=None,
    ) -> List:
        """
        Convert element-wise unary operations (Erf, and future operations).

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (typically empty for unary ops)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer
            opset: Opset version (default: 1)

        Returns:
            List containing a single node instance (ErfNode, ExpNode, LogNode, etc.)

        Raises:
            ValueError: If operation is unsupported or inputs are invalid
        """
        op_type = node_proto.op_type

        # Get the appropriate node class for this operation
        node_class = cls._OP_NODE_MAP.get(op_type)
        if node_class is None:
            raise ValueError(
                f"Unsupported unary operation: {op_type}. " f"Supported operations: {list(cls._OP_NODE_MAP.keys())}"
            )

        # Validate input count (unary ops take exactly one input)
        if len(input_tensors) != 1:
            raise ValueError(
                f"{op_type} node '{node_proto.name or f'{op_type}_{node_index}'}': "
                f"Expected exactly 1 input, got {len(input_tensors)}"
            )

        # Validate output count (unary ops produce exactly one output)
        if len(output_tensors) != 1:
            raise ValueError(
                f"{op_type} node '{node_proto.name or f'{op_type}_{node_index}'}': "
                f"Expected exactly 1 output, got {len(output_tensors)}"
            )

        # For unary operations, output shape and dtype always match input
        input_name = list(input_tensors.keys())[0]
        input_tensor = input_tensors[input_name]
        output_name = list(output_tensors.keys())[0]

        # Ensure output tensor info matches input (shape and dtype)
        output_tensors[output_name] = TensorInfo(
            name=output_name,
            shape=input_tensor.shape,
            onnx_dtype=input_tensor.onnx_dtype,
        )

        # Generate node name if not provided
        node_name = node_proto.name if node_proto.name else f"{op_type}_{node_index}"

        # Build OrderedDict for inputs and outputs
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        # Create and return the appropriate node
        # The node will use PyTorch operations (torch.erf, torch.exp, etc.)
        return [node_class.create(name=node_name, inputs=input_dict, outputs=output_dict)]
