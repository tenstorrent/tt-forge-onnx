# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Arithmetic operation converters.

This module provides converters for ONNX arithmetic operations:
- Add, Sub, Mul, Div: Element-wise binary operations with broadcasting support
- MatMul: Matrix multiplication operation

Key features:
- Handles opset version differences in broadcasting behavior (v1-6 vs v7+)
- Validates shape compatibility based on opset version
- Supports PyTorch-style multidirectional broadcasting (opset 7+)
- Handles limited broadcasting with axis attribute (opset 1-6)
"""
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.arithmetic import AddNode, SubNode, MulNode, DivNode, MatMulNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


def _are_shapes_equal(shape_a: Tuple, shape_b: Tuple) -> bool:
    """
    Check if two shapes are exactly equal.

    Args:
        shape_a: First shape tuple
        shape_b: Second shape tuple

    Returns:
        True if shapes are identical, False otherwise
    """
    return shape_a == shape_b


def _are_shapes_compatible_for_broadcasting(shape_a: Tuple, shape_b: Tuple) -> bool:
    """
    Check if two shapes are compatible for NumPy-style broadcasting (OPSET 7+).

    Two shapes are compatible for multidirectional broadcasting if:
    - They are equal, OR
    - For each dimension (aligned from right), they are equal OR one is 1 OR one is missing

    This implements PyTorch/NumPy-style broadcasting where dimensions are aligned
    from the right, and missing dimensions are treated as 1.

    Args:
        shape_a: First shape tuple
        shape_b: Second shape tuple

    Returns:
        True if shapes are compatible for broadcasting, False otherwise
    """
    if shape_a == shape_b:
        return True

    # Align shapes from the right (broadcasting aligns trailing dimensions)
    len_a, len_b = len(shape_a), len(shape_b)
    max_len = max(len_a, len_b)

    # Check compatibility dimension by dimension, starting from the right
    for i in range(max_len):
        # Get dimensions from right to left (trailing dimensions first)
        dim_a = shape_a[-(i + 1)] if i < len_a else 1
        dim_b = shape_b[-(i + 1)] if i < len_b else 1

        # Dimensions are compatible if:
        # - They are equal, OR
        # - One of them is 1 (can be broadcast)
        # Missing dimensions (when one shape is shorter) are treated as 1
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False

    return True


def _validate_limited_broadcasting(shape_a: Tuple, shape_b: Tuple, axis: Optional[int], op_type: str) -> None:
    """
    Validate broadcasting for OPSET 1-6 (limited broadcasting with axis attribute).

    In OPSET 1-6, broadcasting is limited:
    - B's shape must match a contiguous subset of A's shape
    - If axis is specified, B aligns starting at that axis
    - If axis is not specified, suffix matching is used (B aligns from right)

    Args:
        shape_a: Shape of tensor A (left operand)
        shape_b: Shape of tensor B (right operand)
        axis: Optional axis attribute (None means suffix matching)
        op_type: Operation type for error messages

    Raises:
        ValueError: If shapes are not compatible for limited broadcasting
    """
    shape_a_list = list(shape_a)
    shape_b_list = list(shape_b)

    if axis is None:
        # Suffix matching: B must match the suffix of A (aligned from right)
        # This is the default behavior when axis is not specified
        if len(shape_b_list) > len(shape_a_list):
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET 1-6, suffix matching): "
                f"Shape B {shape_b} has more dimensions than shape A {shape_a}. "
                f"B must match the suffix of A."
            )

        # Check compatibility from right to left (suffix matching)
        for i in range(len(shape_b_list)):
            dim_a = shape_a_list[-(i + 1)]
            dim_b = shape_b_list[-(i + 1)]

            if dim_a != dim_b:
                raise ValueError(
                    f"Broadcasting error in {op_type} (OPSET 1-6, suffix matching): "
                    f"Shapes {shape_a} and {shape_b} are not compatible. "
                    f"Dimension mismatch at position {len(shape_a_list) - i - 1}: {dim_a} vs {dim_b}"
                )
    else:
        # Axis-specified matching: B aligns starting at the specified axis in A
        # This allows B to be placed at a specific position in A's shape
        if axis < 0 or axis >= len(shape_a_list):
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET 1-6, axis={axis}): "
                f"Axis {axis} is out of range for shape A {shape_a} "
                f"(valid range: 0 to {len(shape_a_list) - 1})"
            )

        # B's dimensions must fit within A's dimensions starting at axis
        if len(shape_b_list) > len(shape_a_list) - axis:
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET 1-6, axis={axis}): "
                f"Shape B {shape_b} has {len(shape_b_list)} dimensions, but only "
                f"{len(shape_a_list) - axis} dimensions available starting at axis {axis} "
                f"in shape A {shape_a}"
            )

        # Check dimension compatibility starting at the specified axis
        for i in range(len(shape_b_list)):
            dim_a = shape_a_list[axis + i]
            dim_b = shape_b_list[i]

            if dim_a != dim_b:
                raise ValueError(
                    f"Broadcasting error in {op_type} (OPSET 1-6, axis={axis}): "
                    f"Shapes {shape_a} and {shape_b} are not compatible. "
                    f"Dimension mismatch at axis {axis + i}: {dim_a} vs {dim_b}"
                )


def _validate_broadcast_attributes(
    op_type: str, attrs: Dict[str, Any], input_tensors: OrderedDict[str, TensorInfo], opset: int
) -> None:
    """
    Validate broadcast and axis attributes based on opset version.

    This function only validates - it does not return processed attributes.
    Raises ValueError if validation fails.

    Args:
        op_type: Operation type (Add, Sub, Mul, Div)
        attrs: Extracted attributes dictionary
        input_tensors: Dictionary of input tensor information
        opset: Opset version

    Raises:
        ValueError: If shapes are incompatible for broadcasting
    """
    # Extract broadcast and axis attributes
    broadcast = attrs.get("broadcast", 0)
    axis = attrs.get("axis", None)

    # Get input shapes and validate
    if len(input_tensors) < 2:
        raise ValueError(f"{op_type} node: Expected 2 inputs, got {len(input_tensors)}")

    input_names = list(input_tensors.keys())
    tensor_a = input_tensors[input_names[0]]
    tensor_b = input_tensors[input_names[1]]

    shape_a = tensor_a.shape
    shape_b = tensor_b.shape

    if shape_a is None or shape_b is None:
        logger.warning(
            f"{op_type} node: Cannot validate broadcasting - one or both shapes are unknown. "
            f"Shape A: {shape_a}, Shape B: {shape_b}"
        )
        return  # Skip validation if shapes are unknown

    shapes_match = _are_shapes_equal(shape_a, shape_b)
    shapes_compatible_multidir = _are_shapes_compatible_for_broadcasting(shape_a, shape_b)

    # Handle broadcasting validation based on opset version
    if opset <= 6:
        # OPSET 1-6: Limited broadcasting, requires explicit broadcast=1 attribute
        # Broadcasting is opt-in, not automatic
        if not shapes_match:
            if broadcast == 0:
                raise ValueError(
                    f"Broadcasting error in {op_type} (OPSET {opset}): "
                    f"Shapes {shape_a} and {shape_b} don't match and broadcast=0. "
                    f"Set broadcast=1 to enable broadcasting."
                )
            else:
                # broadcast=1 is set, validate limited broadcasting rules
                # Limited broadcasting: B must match a contiguous subset of A's shape
                _validate_limited_broadcasting(shape_a, shape_b, axis, op_type)
                if axis is not None:
                    logger.debug(f"{op_type} node: Using axis={axis} for broadcasting (OPSET {opset})")

    else:
        # OPSET 7+: Multidirectional broadcasting always enabled (NumPy/PyTorch style)
        # The broadcast and axis attributes were removed in opset 7
        if broadcast != 0 or axis is not None:
            logger.warning(
                f"{op_type} node: 'broadcast' and 'axis' attributes are not supported "
                f"in OPSET {opset} (removed in OPSET 7+). These attributes will be ignored. "
                f"Multidirectional broadcasting is always enabled."
            )

        # Validate shapes are compatible for multidirectional broadcasting
        # In multidirectional broadcasting, dimensions align from the right
        # and are compatible if equal or one is 1
        if not shapes_match and not shapes_compatible_multidir:
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET {opset}): "
                f"Shapes {shape_a} and {shape_b} are not compatible for "
                f"multidirectional broadcasting. "
                f"Two dimensions are compatible if they are equal OR one is 1."
            )


class BinaryArithmeticConverter(OnnxOpConverter):
    """
    Unified converter for binary arithmetic operations: Add, Sub, Mul, Div.

    This converter handles all four operations (Add, Sub, Mul, Div) using a single
    implementation, while maintaining separate operator nodes for each operation type.
    The operator nodes use PyTorch API (torch.add, torch.sub, torch.mul, torch.div).

    Broadcasting Handling:
    - OPSET 1-6: Validates `broadcast=1` attribute and handles `axis` attribute
    - OPSET 7+: Multidirectional broadcasting always enabled (attributes ignored)
    - PyTorch operations handle broadcasting automatically
    """

    # Mapping from ONNX op type to corresponding node class
    _OP_NODE_MAP = {
        "Add": AddNode,
        "Sub": SubNode,
        "Mul": MulNode,
        "Div": DivNode,
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
    ) -> List:
        """
        Convert binary arithmetic operations (Add, Sub, Mul, Div).

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (may include broadcast, axis)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer
            opset: Opset version (default: 1)

        Returns:
            List containing a single node instance (AddNode, SubNode, MulNode, or DivNode)
        """
        op_type = node_proto.op_type

        # Get the appropriate node class for this operation
        node_class = cls._OP_NODE_MAP.get(op_type)
        if node_class is None:
            raise ValueError(
                f"Unsupported binary arithmetic operation: {op_type}. "
                f"Supported operations: {list(cls._OP_NODE_MAP.keys())}"
            )

        # Validate broadcast/axis attributes based on opset version
        # This ensures shapes are compatible and raises errors if not
        # Note: PyTorch operations handle broadcasting automatically, but we validate
        # to catch errors early and provide better error messages
        _validate_broadcast_attributes(op_type=op_type, attrs=attrs, input_tensors=input_tensors, opset=opset)

        # Generate node name if not provided
        node_name = node_proto.name if node_proto.name else f"{op_type}_{node_index}"

        # Build OrderedDict for inputs and outputs
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        # Create and return the appropriate node
        # The node will use PyTorch operations (torch.add, torch.sub, etc.)
        # which handle broadcasting automatically
        return [node_class.create(name=node_name, inputs=input_dict, outputs=output_dict)]
