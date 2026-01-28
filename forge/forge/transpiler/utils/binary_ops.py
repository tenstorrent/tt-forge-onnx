# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for binary operations and broadcasting.

This is the single canonical module for all broadcasting and binary-operation
validation utilities used across the transpiler.  It covers:

- NumPy/PyTorch-style shape broadcasting (compatibility checks + shape computation)
- OPSET 1-6 limited broadcasting validation (axis attribute)
- OPSET 7+ multidirectional broadcasting validation
- Dtype compatibility checking for arithmetic and comparison ops
"""
import torch
from typing import Tuple, Optional, Dict, Any
from collections import OrderedDict
from loguru import logger

from forge.transpiler.core.types import TensorInfo


def validate_binary_inputs_pytorch_style(
    shape_a: torch.Size,
    shape_b: torch.Size,
    dtype_a: torch.dtype,
    dtype_b: torch.dtype,
    op_name: str,
    tensor_a_name: str,
    tensor_b_name: str,
    operation_category: str = "operation",
) -> None:
    """
    Validate inputs for binary operations (arithmetic or comparison) following PyTorch style.

    This function validates:
    1. Dtype equality: Both tensors must have the same dtype (PyTorch requirement)
    2. Broadcasting compatibility: Shapes must be compatible for broadcasting

    Broadcasting rules (PyTorch/NumPy-style):
    - Shapes are compared from right to left
    - Two dimensions are compatible if:
      * They are equal, OR
      * One of them is 1, OR
      * One of them doesn't exist (missing dimension)

    Args:
        shape_a: Shape of first tensor
        shape_b: Shape of second tensor
        dtype_a: Dtype of first tensor
        dtype_b: Dtype of second tensor
        op_name: Name of the operation (for error messages)
        tensor_a_name: Name of first tensor (for error messages)
        tensor_b_name: Name of second tensor (for error messages)
        operation_category: Category of operation ("arithmetic" or "comparison") for error messages

    Raises:
        ValueError: If dtypes don't match or shapes are not compatible for broadcasting
    """
    # 1. Validate dtype equality (PyTorch requirement)
    if dtype_a != dtype_b:
        raise ValueError(
            f"Type mismatch in {op_name}: "
            f"Input tensors must have the same dtype. "
            f"{tensor_a_name} has dtype {dtype_a}, "
            f"{tensor_b_name} has dtype {dtype_b}. "
            f"PyTorch {operation_category} operations require matching dtypes."
        )

    # 2. If shapes are equal, no broadcasting needed
    if shape_a == shape_b:
        return

    # 3. Validate broadcasting compatibility
    shape_a_list = list(shape_a)
    shape_b_list = list(shape_b)

    # Pad shorter shape with 1s on the left (missing dimensions treated as 1)
    max_len = max(len(shape_a_list), len(shape_b_list))
    shape_a_padded = [1] * (max_len - len(shape_a_list)) + shape_a_list
    shape_b_padded = [1] * (max_len - len(shape_b_list)) + shape_b_list

    # Check compatibility from right to left
    incompatible_dims = []
    for i in range(max_len - 1, -1, -1):
        dim_a = shape_a_padded[i]
        dim_b = shape_b_padded[i]

        # Dimensions are compatible if:
        # 1. They are equal, OR
        # 2. One of them is 1
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            incompatible_dims.append((i, dim_a, dim_b))

    if incompatible_dims:
        dim_info = ", ".join([f"dim {d[0]}: {d[1]} vs {d[2]}" for d in incompatible_dims])
        raise ValueError(
            f"Broadcasting error in {op_name}: "
            f"Shapes {shape_a} ({tensor_a_name}) and {shape_b} ({tensor_b_name}) "
            f"are not compatible for broadcasting. "
            f"Incompatible dimensions: {dim_info}. "
            f"Two dimensions are compatible if they are equal OR one is 1."
        )


def are_shapes_equal(shape_a: Tuple, shape_b: Tuple) -> bool:
    """
    Check if two shapes are exactly equal.

    Args:
        shape_a: First shape tuple
        shape_b: Second shape tuple

    Returns:
        True if shapes are identical, False otherwise
    """
    return shape_a == shape_b


def _check_no_unknown_dims(shape: Tuple, context: str = "") -> None:
    """Raise ValueError if shape contains unknown dimensions (None, str, or int < 0)."""
    import numpy as np

    if shape is None:
        raise ValueError(
            f"Shape is None. Unknown dimensions are not supported.{f' Context: {context}' if context else ''}"
        )
    for idx, dim in enumerate(shape):
        if dim is None or isinstance(dim, str):
            raise ValueError(
                f"Unknown dimension at index {idx}: {dim}. All dimensions must be known integers. "
                f"Unknown dimensions are not supported." + (f" Context: {context}" if context else "")
            )
        if isinstance(dim, (int, np.integer)) and dim < 0:
            raise ValueError(
                f"Unknown/dynamic dimension at index {idx}: {dim}. "
                f"Negative dimensions must be resolved before use." + (f" Context: {context}" if context else "")
            )


def are_shapes_compatible_for_broadcasting(shape_a: Tuple, shape_b: Tuple) -> bool:
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

    Raises:
        ValueError: If shapes contain unknown dimensions (None, str, negative int)
    """
    if shape_a is None or shape_b is None:
        return False

    _check_no_unknown_dims(shape_a, "are_shapes_compatible_for_broadcasting")
    _check_no_unknown_dims(shape_b, "are_shapes_compatible_for_broadcasting")

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


def validate_limited_broadcasting(shape_a: Tuple, shape_b: Tuple, axis: Optional[int], op_type: str) -> None:
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


def validate_broadcast_attributes(
    op_type: str, attrs: Dict[str, Any], input_tensors: OrderedDict[str, TensorInfo], opset: int
) -> None:
    """
    Validate broadcast and axis attributes based on opset version.

    This function validates broadcasting compatibility for binary operations
    (both arithmetic and comparison) based on the opset version.

    This function only validates - it does not return processed attributes.
    Raises ValueError if validation fails.

    Args:
        op_type: Operation type (Add, Sub, Mul, Div, Equal, Greater, Less, etc.)
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

    shapes_match = are_shapes_equal(shape_a, shape_b)
    shapes_compatible_multidir = are_shapes_compatible_for_broadcasting(shape_a, shape_b)

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
                validate_limited_broadcasting(shape_a, shape_b, axis, op_type)
                if axis is not None:
                    logger.trace(f"{op_type} node: Using axis={axis} for broadcasting (OPSET {opset})")

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


# ---------------------------------------------------------------------------
# Shape computation
# ---------------------------------------------------------------------------


def compute_broadcasted_shape(shape_a: Tuple, shape_b: Tuple) -> Optional[Tuple]:
    """
    Compute the output shape produced by broadcasting two input shapes.

    Follows NumPy/PyTorch multidirectional broadcasting rules:

    1. Shapes are compared from right to left (trailing dimensions first).
    2. Two dimensions are compatible if they are equal, or one of them is 1,
       or one of them is missing (treated as 1).
    3. The output dimension is the maximum of the two compatible dimensions.

    Args:
        shape_a: First operand shape tuple.
        shape_b: Second operand shape tuple.

    Returns:
        Broadcasted shape tuple, or ``None`` if shapes are incompatible or either
        shape is ``None``.
    """
    if shape_a is None or shape_b is None:
        return None

    _check_no_unknown_dims(shape_a, "compute_broadcasted_shape")
    _check_no_unknown_dims(shape_b, "compute_broadcasted_shape")

    if shape_a == shape_b:
        return shape_a

    shape_a_list = list(shape_a)
    shape_b_list = list(shape_b)

    max_len = max(len(shape_a_list), len(shape_b_list))
    shape_a_padded = [1] * (max_len - len(shape_a_list)) + shape_a_list
    shape_b_padded = [1] * (max_len - len(shape_b_list)) + shape_b_list

    broadcasted = []
    for dim_a, dim_b in zip(shape_a_padded, shape_b_padded):
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return None
        broadcasted.append(max(dim_a, dim_b))

    return tuple(broadcasted)


def compute_broadcasted_shape_multi(*shapes: Tuple) -> Optional[Tuple]:
    """
    Compute the broadcasted output shape for two or more input shapes.

    Applies :func:`compute_broadcasted_shape` iteratively, left to right.

    Args:
        *shapes: Two or more shape tuples to broadcast together.

    Returns:
        Broadcasted shape tuple, or ``None`` if any pair is incompatible or any
        shape is ``None``.
    """
    if not shapes:
        return None
    if len(shapes) == 1:
        return shapes[0]

    result = shapes[0]
    for shape in shapes[1:]:
        result = compute_broadcasted_shape(result, shape)
        if result is None:
            return None
    return result
