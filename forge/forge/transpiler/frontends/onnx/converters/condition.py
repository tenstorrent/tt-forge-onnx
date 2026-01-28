# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Conditional operation converters.

This module provides converters for ONNX conditional operations:
- Where: Element-wise conditional selection (condition ? X : Y)

Key features:
- Supports multidirectional (NumPy-style) broadcasting
- Validates type compatibility (X and Y must have same dtype)
- Validates condition is boolean type
- Handles shape inference for broadcasted outputs
"""
from typing import List, Dict, Any
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.other import WhereNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts
from forge.transpiler.utils.binary_ops import compute_broadcasted_shape_multi


def _validate_where_inputs(
    node_name: str,
    condition_info: TensorInfo,
    x_info: TensorInfo,
    y_info: TensorInfo,
    opset: int,
) -> None:
    """
    Validate inputs for Where operation.

    Validates:
    1. Condition must be boolean type
    2. X and Y must have the same dtype
    3. All three inputs must be compatible for broadcasting

    Args:
        node_name: Name of the node (for error messages)
        condition_info: TensorInfo for condition input
        x_info: TensorInfo for X input
        y_info: TensorInfo for Y input
        opset: Opset version (for error messages, unused but kept for consistency)

    Raises:
        ValueError: If validation fails
    """
    # Check condition is boolean
    condition_dtype = condition_info.onnx_dtype
    if condition_dtype != 9:  # TensorProto.BOOL = 9
        raise ValueError(
            f"Where node '{node_name}': condition input must be boolean type (TensorProto.BOOL=9), "
            f"got dtype {condition_dtype}"
        )

    # Check X and Y have the same dtype
    if x_info.onnx_dtype != y_info.onnx_dtype:
        raise ValueError(
            f"Where node '{node_name}': X and Y inputs must have the same dtype. "
            f"X has dtype {x_info.onnx_dtype}, Y has dtype {y_info.onnx_dtype}"
        )

    # Validate broadcasting compatibility by computing output shape
    # If shapes are incompatible, compute_broadcasted_shape_multi returns None
    output_shape = compute_broadcasted_shape_multi(condition_info.shape, x_info.shape, y_info.shape)
    if output_shape is None:
        raise ValueError(
            f"Where node '{node_name}': Input shapes are not compatible for broadcasting. "
            f"condition: {condition_info.shape}, X: {x_info.shape}, Y: {y_info.shape}"
        )


class WhereConverter(OnnxOpConverter):
    """
    Converter for ONNX Where operation.

    Converts ONNX Where to TIR WhereNode. Where performs element-wise conditional selection:
    output = condition ? X : Y

    Supports multidirectional (NumPy-style) broadcasting for all three inputs.
    """

    @classmethod
    def convert(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset: int = 9,
        tir_graph=None,
    ) -> List:
        """
        Convert ONNX Where operation to TIR WhereNode.

        Where opset v9+: No version differences in converter logic (v16 adds bfloat16 support,
        but this is handled by ONNX shape inference, not the converter).

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo
            output_tensors: Dictionary mapping output names to TensorInfo
            attrs: Extracted attributes (Where has no attributes)
            node_index: Index of node in graph
            graph_proto: ONNX graph protocol buffer (unused)
            opset: Opset version (unused, no version differences in converter logic)

        Returns:
            List containing a single WhereNode

        Raises:
            ValueError: If inputs are invalid or incompatible for broadcasting
        """
        node_name = node_proto.name if node_proto.name else f"Where_{node_index}"

        # Validate input count
        if len(node_proto.input) != 3:
            raise ValueError(
                f"Where node '{node_name}': Expected 3 inputs (condition, X, Y), " f"got {len(node_proto.input)}"
            )

        # Get input tensor infos
        condition_input = node_proto.input[0]
        x_input = node_proto.input[1]
        y_input = node_proto.input[2]

        if condition_input not in input_tensors:
            raise ValueError(f"Where node '{node_name}': Condition input '{condition_input}' not found")
        if x_input not in input_tensors:
            raise ValueError(f"Where node '{node_name}': X input '{x_input}' not found")
        if y_input not in input_tensors:
            raise ValueError(f"Where node '{node_name}': Y input '{y_input}' not found")

        condition_info = input_tensors[condition_input]
        x_info = input_tensors[x_input]
        y_info = input_tensors[y_input]

        # Validate inputs
        _validate_where_inputs(node_name, condition_info, x_info, y_info, opset)

        # Compute output shape if not already set
        output_shape = compute_broadcasted_shape_multi(condition_info.shape, x_info.shape, y_info.shape)
        if output_shape is not None and len(output_tensors) > 0:
            output_name = list(output_tensors.keys())[0]
            output_tensors[output_name] = TensorInfo(
                name=output_name,
                shape=output_shape,
                onnx_dtype=x_info.onnx_dtype,  # Output dtype matches X and Y
            )

        # Build input/output dicts
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        # Create and return WhereNode
        return [WhereNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]
