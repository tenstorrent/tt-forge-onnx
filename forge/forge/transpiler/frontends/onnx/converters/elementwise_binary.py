# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Element-wise Binary operation converters.

This module provides converters for ONNX element-wise binary operations:
- Arithmetic: Add, Sub, Mul, Div, Pow - Element-wise binary operations with broadcasting support
- Comparison: Equal, Greater, Less, GreaterOrEqual, LessOrEqual - Element-wise comparison operations returning boolean tensors
- MatMul: Matrix multiplication operation

Key features:
- Handles opset version differences in broadcasting behavior (v1-6 vs v7+)
- Validates shape compatibility based on opset version
- Supports PyTorch-style multidirectional broadcasting (opset 7+)
- Handles limited broadcasting with axis attribute (opset 1-6)
- Unified converter for both arithmetic and comparison operations
"""
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.arithmetic import AddNode, SubNode, MulNode, DivNode, PowNode, MatMulNode
from forge.transpiler.operations.comparison import (
    EqualNode,
    GreaterNode,
    LessNode,
    GreaterOrEqualNode,
    LessOrEqualNode,
    LogicalAndNode,
)
from forge.transpiler.operations.other import CastNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts
from forge.transpiler.frontends.onnx.utils.shape_finder import validate_no_unknown_dimensions
from forge.transpiler.utils.binary_ops import (
    compute_broadcasted_shape,
    validate_broadcast_attributes,
    are_shapes_compatible_for_broadcasting,
)


class BinaryOpConverter(OnnxOpConverter):
    """
    Unified converter for element-wise binary operations: Arithmetic (Add, Sub, Mul, Div) and Comparison (Equal, Greater, Less, GreaterOrEqual, LessOrEqual).

    This converter handles all element-wise binary operations using a single implementation,
    while maintaining separate operator nodes for each operation type.

    Arithmetic operations use PyTorch API (torch.add, torch.sub, torch.mul, torch.div).
    Comparison operations use PyTorch API (torch.eq, torch.gt, torch.lt, torch.ge, torch.le, torch.ne).

    All operations support broadcasting:
    - OPSET 1-6: Validates `broadcast=1` attribute and handles `axis` attribute
    - OPSET 7+: Multidirectional broadcasting always enabled (attributes ignored)
    - PyTorch operations handle broadcasting automatically

    Output dtype handling:
    - Arithmetic operations: Output dtype matches input dtypes
    - Comparison operations: Output dtype is always bool (TensorProto.BOOL = 9)
    """

    # Mapping from ONNX op type to corresponding node class
    # Arithmetic operations
    _ARITHMETIC_NODE_MAP = {
        "Add": AddNode,
        "Sub": SubNode,
        "Mul": MulNode,
        "Div": DivNode,
    }

    # Comparison operations (numeric inputs → bool output)
    _COMPARISON_NODE_MAP = {
        "Equal": EqualNode,
        "Greater": GreaterNode,
        "Less": LessNode,
        "GreaterOrEqual": GreaterOrEqualNode,
        "LessOrEqual": LessOrEqualNode,
    }

    # Logical binary operations (bool inputs → bool output)
    _LOGICAL_NODE_MAP = {
        "And": LogicalAndNode,
    }

    # Combined mapping for all binary operations
    _OP_NODE_MAP = {**_ARITHMETIC_NODE_MAP, **_COMPARISON_NODE_MAP, **_LOGICAL_NODE_MAP}

    # Operations that always return a boolean output tensor
    _COMPARISON_OPS = set(_COMPARISON_NODE_MAP.keys()) | set(_LOGICAL_NODE_MAP.keys())

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
        Convert element-wise binary operations (Arithmetic: Add, Sub, Mul, Div or Comparison: Equal, Greater, Less, GreaterOrEqual, LessOrEqual).

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (may include broadcast, axis)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer
            opset: Opset version (default: 1)

        Returns:
            List containing a single node instance (AddNode, SubNode, MulNode, DivNode, EqualNode, GreaterNode, LessNode, GreaterOrEqualNode, or LessOrEqualNode)

        Raises:
            ValueError: If operation is unsupported or inputs are invalid
        """
        op_type = node_proto.op_type

        # Get the appropriate node class for this operation
        node_class = cls._OP_NODE_MAP.get(op_type)
        if node_class is None:
            raise ValueError(
                f"Unsupported binary operation: {op_type}. " f"Supported operations: {list(cls._OP_NODE_MAP.keys())}"
            )

        # Validate broadcast/axis attributes based on opset version
        # This ensures shapes are compatible and raises errors if not
        # Note: PyTorch operations handle broadcasting automatically, but we validate
        # to catch errors early and provide better error messages
        validate_broadcast_attributes(op_type=op_type, attrs=attrs, input_tensors=input_tensors, opset=opset)

        # Generate node name if not provided
        node_name = node_proto.name if node_proto.name else f"{op_type}_{node_index}"

        # Compute output shape and set output dtype if not already set
        if len(input_tensors) >= 2:
            input_names = list(input_tensors.keys())
            tensor_a = input_tensors[input_names[0]]
            tensor_b = input_tensors[input_names[1]]

            # Determine output dtype based on operation type
            if op_type in cls._COMPARISON_OPS:
                # Comparison operations always return boolean tensor
                output_dtype = 9  # TensorProto.BOOL
            else:
                # Arithmetic operations: output dtype matches input dtypes (validated to match)
                output_dtype = tensor_a.onnx_dtype

            # Compute output shape for opset 7+ (multidirectional broadcasting)
            # For opset < 7, shape should already be set by ONNX shape inference
            if opset >= 7:
                output_shape = compute_broadcasted_shape(tensor_a.shape, tensor_b.shape)
                if output_shape is not None and len(output_tensors) > 0:
                    output_name = list(output_tensors.keys())[0]
                    output_tensors[output_name] = TensorInfo(
                        name=output_name,
                        shape=output_shape,
                        onnx_dtype=output_dtype,
                    )
            elif len(output_tensors) > 0:
                # For opset < 7, ensure output dtype is correct (shape already set by ONNX)
                output_name = list(output_tensors.keys())[0]
                existing_output = output_tensors.get(output_name)
                if existing_output is not None:
                    # Update dtype if needed, preserve existing shape
                    output_tensors[output_name] = TensorInfo(
                        name=output_name,
                        shape=existing_output.shape,
                        onnx_dtype=output_dtype,
                    )

        # Build OrderedDict for inputs and outputs
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        # Create and return the appropriate node
        # The node will use PyTorch operations (torch.add, torch.sub, torch.eq, etc.)
        # which handle broadcasting automatically
        return [node_class.create(name=node_name, inputs=input_dict, outputs=output_dict)]


def _validate_matmul_shapes(
    node_name: str, shape_a: Optional[Tuple], shape_b: Optional[Tuple], opset: int
) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple]]:
    """
    Validate shapes for MatMul operation and compute output shape.

    MatMul performs matrix multiplication: Y = A @ B
    - For 2D: A [M, K] @ B [K, N] -> Y [M, N]
    - For N-dimensional: A [..., M, K] @ B [..., K, N] -> Y [..., M, N]
    - Batch dimensions are broadcastable

    Args:
        node_name: Name of the MatMul node (for error messages)
        shape_a: Shape of input tensor A
        shape_b: Shape of input tensor B
        opset: ONNX opset version

    Returns:
        Tuple of (shape_a, shape_b, output_shape) after validation

    Raises:
        ValueError: If shapes are incompatible for matrix multiplication
    """
    if shape_a is None or shape_b is None:
        raise ValueError(
            f"MatMul node {node_name}: Shapes must be known. "
            f"Shape A: {shape_a}, Shape B: {shape_b}. Unknown dimensions are not supported."
        )
    validate_no_unknown_dimensions(shape_a, f"MatMul {node_name}")
    validate_no_unknown_dimensions(shape_b, f"MatMul {node_name}")

    # Validate minimum dimensions
    if len(shape_a) < 1:
        raise ValueError(f"MatMul node {node_name}: Input A must have at least 1 dimension, got shape {shape_a}")
    if len(shape_b) < 1:
        raise ValueError(f"MatMul node {node_name}: Input B must have at least 1 dimension, got shape {shape_b}")

    # Handle 1D inputs: treat as 2D with leading dimension 1
    # ONNX MatMul requires at least 2D, but PyTorch matmul handles 1D
    # We'll validate as if they're 2D for consistency with ONNX spec
    if len(shape_a) == 1:
        # 1D A: [K] -> treat as [1, K] for validation
        shape_a_2d = (1,) + shape_a
        logger.trace(f"MatMul node {node_name}: Treating 1D input A {shape_a} as 2D {shape_a_2d}")
    else:
        shape_a_2d = shape_a

    if len(shape_b) == 1:
        # 1D B: [K] -> treat as [K, 1] for validation
        shape_b_2d = shape_b + (1,)
        logger.trace(f"MatMul node {node_name}: Treating 1D input B {shape_b} as 2D {shape_b_2d}")
    else:
        shape_b_2d = shape_b

    # Now both shapes are at least 2D
    # Validate matrix multiplication compatibility: A.shape[-1] == B.shape[-2]
    if shape_a_2d[-1] != shape_b_2d[-2]:
        raise ValueError(
            f"MatMul node {node_name}: Incompatible shapes for matrix multiplication. "
            f"A.shape[-1] ({shape_a_2d[-1]}) must equal B.shape[-2] ({shape_b_2d[-2]}). "
            f"A shape: {shape_a} (treated as {shape_a_2d}), B shape: {shape_b} (treated as {shape_b_2d})"
        )

    # Determine if this is batched (N-dimensional) or standard (2D) matrix multiplication
    is_batched = len(shape_a_2d) > 2 or len(shape_b_2d) > 2

    if is_batched:
        # N-dimensional case: validate batch dimensions are broadcastable
        batch_dims_a = shape_a_2d[:-2]
        batch_dims_b = shape_b_2d[:-2]

        # Compute broadcasted batch dimensions
        # Align from right and compute broadcasted shape
        max_batch_len = max(len(batch_dims_a), len(batch_dims_b))
        broadcasted_batch_dims = []

        for i in range(max_batch_len):
            # Get dimensions from right to left
            idx_a = len(batch_dims_a) - max_batch_len + i if i < len(batch_dims_a) else None
            idx_b = len(batch_dims_b) - max_batch_len + i if i < len(batch_dims_b) else None

            dim_a = batch_dims_a[idx_a] if idx_a is not None and idx_a >= 0 else 1
            dim_b = batch_dims_b[idx_b] if idx_b is not None and idx_b >= 0 else 1

            # Broadcasted dimension is max of the two (or 1 if one is missing)
            # Both dimensions are integers now
            if dim_a == dim_b:
                broadcasted_batch_dims.append(dim_a)
            elif dim_a == 1:
                broadcasted_batch_dims.append(dim_b)
            elif dim_b == 1:
                broadcasted_batch_dims.append(dim_a)
            else:
                # This should have been caught by broadcasting check, but validate anyway
                raise ValueError(
                    f"MatMul node {node_name}: Batch dimensions are not broadcastable. "
                    f"A batch dims: {batch_dims_a}, B batch dims: {batch_dims_b}. "
                    f"At position {i}: {dim_a} vs {dim_b} (both must be equal or one must be 1)"
                )

        # Validate batch dimensions are compatible for broadcasting
        if not are_shapes_compatible_for_broadcasting(batch_dims_a, batch_dims_b):
            raise ValueError(
                f"MatMul node {node_name}: Batch dimensions are not broadcastable. "
                f"A batch dims: {batch_dims_a}, B batch dims: {batch_dims_b}. "
                f"Batch dimensions must be compatible for NumPy/PyTorch-style broadcasting "
                f"(aligned from right, compatible if equal or one is 1)"
            )

        # Compute output shape: [broadcasted_batch_dims..., M, N]
        # M = A.shape[-2], N = B.shape[-1]
        M = shape_a_2d[-2]
        N = shape_b_2d[-1]
        output_shape = tuple(broadcasted_batch_dims) + (M, N)

        logger.trace(
            f"MatMul node {node_name}: Batched matrix multiplication. "
            f"A: {shape_a} -> {shape_a_2d}, B: {shape_b} -> {shape_b_2d}, "
            f"Batch dims: {batch_dims_a} x {batch_dims_b} -> {broadcasted_batch_dims}, "
            f"Output: {output_shape}"
        )
    else:
        # Standard 2D matrix multiplication
        M = shape_a_2d[-2]
        N = shape_b_2d[-1]
        output_shape = (M, N)

        logger.trace(
            f"MatMul node {node_name}: Standard 2D matrix multiplication. "
            f"A: {shape_a} -> {shape_a_2d} [M={M}, K={shape_a_2d[-1]}], "
            f"B: {shape_b} -> {shape_b_2d} [K={shape_b_2d[-2]}, N={N}], "
            f"Output: {output_shape} [M={M}, N={N}]"
        )

    return shape_a, shape_b, output_shape


class MatMulConverter(OnnxOpConverter):
    """
    Converter for ONNX MatMul (Matrix Multiplication) operation.

    MatMul performs matrix product that behaves like numpy.matmul.
    It computes Y = A @ B where A and B are N-dimensional matrices,
    with the last two dimensions being treated as matrices and all
    preceding dimensions as batch dimensions.

    Key features:
    - No attributes: MatMul has no configurable attributes
    - Standard 2D: Handles standard matrix multiplication [M, K] @ [K, N] -> [M, N]
    - N-dimensional: Handles batched matrix multiplication [..., M, K] @ [..., K, N] -> [..., M, N]
    - Broadcasting: Automatically handles broadcasting for batch dimensions
    - Shape validation: Comprehensive validation of matrix dimensions and batch broadcasting
    - All opset versions: Behavior is consistent across all opset versions (1, 9, 13)
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
        opset: int = 1,
        tir_graph=None,
    ) -> List:
        """
        Convert ONNX MatMul operation to MatMulNode.

        Supports both standard 2D matrix multiplication and N-dimensional batched
        matrix multiplication with automatic broadcasting of batch dimensions.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (MatMul has no attributes, but kept for consistency)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer
            opset: Opset version (default: 1)

        Returns:
            List containing a single MatMulNode instance

        Raises:
            ValueError: If inputs are invalid or shapes are incompatible
        """
        # Validate inputs
        if len(node_proto.input) != 2:
            raise ValueError(
                f"MatMul node {node_proto.name or f'MatMul_{node_index}'}: "
                f"Expected 2 inputs, got {len(node_proto.input)}"
            )

        # Get input tensor info
        input_a_name = node_proto.input[0]
        input_b_name = node_proto.input[1]

        if input_a_name not in input_tensors:
            raise ValueError(
                f"MatMul node {node_proto.name or f'MatMul_{node_index}'}: "
                f"Input A '{input_a_name}' not found in input_tensors"
            )
        if input_b_name not in input_tensors:
            raise ValueError(
                f"MatMul node {node_proto.name or f'MatMul_{node_index}'}: "
                f"Input B '{input_b_name}' not found in input_tensors"
            )

        tensor_a = input_tensors[input_a_name]
        tensor_b = input_tensors[input_b_name]

        # Validate shapes and compute output shape
        node_name = node_proto.name or f"MatMul_{node_index}"
        _, _, output_shape = _validate_matmul_shapes(node_name, tensor_a.shape, tensor_b.shape, opset)

        # Update output tensor shape if we computed it
        if output_shape is not None and len(output_tensors) > 0:
            output_name = list(output_tensors.keys())[0]
            # Update the output tensor info with computed shape
            output_tensors[output_name] = TensorInfo(
                name=output_name,
                shape=output_shape,
                onnx_dtype=tensor_a.onnx_dtype,  # Output dtype matches input dtype
            )

        # Build OrderedDict for inputs and outputs
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        # Create and return MatMulNode
        # MatMulNode uses torch.matmul which handles:
        # - Standard 2D matrix multiplication
        # - N-dimensional batched matrix multiplication
        # - Broadcasting of batch dimensions automatically
        return [MatMulNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]


class PowConverter(OnnxOpConverter):
    """
    Converter for ONNX Pow operation: Z = X ^ Y.

    Both X (base) and Y (exponent) are wired as tensor inputs to
    :class:`PowNode`, mapping to ``forge.op.Power`` (binary) and
    ``ttir.pow(%lhs, %rhs)``.

    Opset version differences
    -------------------------
    * **v1-v6**: Broadcasting is *opt-in* via ``broadcast=1`` attribute.
      The ``axis`` attribute specifies where to align Y within X.
      X and Y must share the same float type.
    * **v7-v11**: Multidirectional (NumPy-style) broadcasting always enabled.
      ``broadcast`` and ``axis`` attributes were removed.
      X and Y still share the same type.
    * **v12+**: Heterogeneous type constraints — X has type T, Y has type T1
      (may differ).  Output Z always has type T (same as X).  Y is cast to
      X's dtype via an explicit :class:`~forge.transpiler.operations.other.CastNode`
      inserted *before* the :class:`PowNode` in the TIR graph.
    * **v13, v15**: Broadened T and T1 type sets; behaviour is otherwise
      identical to v12.
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
        opset: int = 1,
        tir_graph=None,
    ) -> List:
        """
        Convert an ONNX Pow node to a TIR :class:`PowNode`.

        Steps performed (mirroring :class:`BinaryOpConverter`):

        1. Validate the number of inputs.
        2. Validate broadcasting compatibility using
           :func:`validate_broadcast_attributes` (handles v1-6 *broadcast* /
           *axis* attributes and v7+ multidirectional rules).
        3. Compute the output shape with
           :func:`compute_broadcasted_shape` (v7+) or preserve the ONNX
           shape-inferred shape (v1-6).
        4. Set output dtype to X's type (enforces T == Z for all opsets).
        5. **Cast injection**: if Y's dtype differs from X's dtype, insert a
           :class:`~forge.transpiler.operations.other.CastNode` (Y → X dtype)
           *before* the :class:`PowNode` in the TIR graph so that both inputs
           are type-matched when ``PowNode.eval()`` calls ``torch.pow``.  A
           ``TRACE`` log is emitted for expected opset-12+ heterogeneous types;
           a ``WARNING`` is emitted for older opsets where spec requires
           homogeneous types.
        6. Build input/output dicts and create :class:`PowNode`.

        Args:
            node_proto: ONNX node protocol buffer.
            input_tensors: Dictionary mapping input names to TensorInfo.
            output_tensors: Dictionary mapping output names to TensorInfo.
            attrs: Extracted node attributes (``broadcast``, ``axis`` for v1-6).
            node_index: Position of this node in the graph.
            graph_proto: ONNX graph proto (unused; kept for API consistency).
            opset: Opset version — drives broadcasting and type-check behaviour.
            tir_graph: Partially-built TIR graph (unused; kept for API consistency).

        Returns:
            List containing a :class:`PowNode`, optionally preceded by a
            :class:`~forge.transpiler.operations.other.CastNode` when Y's dtype
            differs from X's dtype.

        Raises:
            ValueError: If fewer than 2 inputs are provided, or shapes are
                incompatible for broadcasting under the active opset rules.
        """
        node_name = node_proto.name if node_proto.name else f"Pow_{node_index}"

        if len(node_proto.input) < 2:
            raise ValueError(f"Pow node '{node_name}': expected 2 inputs (X, Y), " f"got {len(node_proto.input)}.")

        x_name = node_proto.input[0]
        y_name = node_proto.input[1]
        tensor_x = input_tensors.get(x_name)
        tensor_y = input_tensors.get(y_name)

        # ── Step 1: Validate broadcasting (opset-aware) ───────────────────────
        validate_broadcast_attributes(
            op_type="Pow",
            attrs=attrs,
            input_tensors=input_tensors,
            opset=opset,
        )

        x_onnx_dtype = tensor_x.onnx_dtype if tensor_x is not None else None
        y_onnx_dtype = tensor_y.onnx_dtype if tensor_y is not None else None

        # ── Step 2: Compute output shape and dtype ────────────────────────────
        output_dtype = x_onnx_dtype

        if tensor_x is not None and tensor_y is not None:
            shape_x = tensor_x.shape
            shape_y = tensor_y.shape

            if opset >= 7:
                output_shape = compute_broadcasted_shape(shape_x, shape_y)
            else:
                output_shape = list(output_tensors.values())[0].shape if output_tensors else shape_x

            if output_shape is not None and output_tensors:
                out_name = list(output_tensors.keys())[0]
                output_tensors[out_name] = TensorInfo(
                    name=out_name,
                    shape=output_shape,
                    onnx_dtype=output_dtype,
                )
        elif output_dtype is not None and output_tensors:
            out_name = list(output_tensors.keys())[0]
            existing = output_tensors[out_name]
            output_tensors[out_name] = TensorInfo(
                name=out_name,
                shape=existing.shape,
                onnx_dtype=output_dtype,
            )

        # ── Step 3: Inject CastNode for Y when dtypes differ ─────────────────
        nodes = []
        pow_y_name = y_name

        if x_onnx_dtype is not None and y_onnx_dtype is not None and x_onnx_dtype != y_onnx_dtype:
            from forge.transpiler.core.types import onnx_dtype_to_torch_dtype

            level = "trace" if opset >= 12 else "warning"
            getattr(logger, level)(
                f"Pow node '{node_name}' (opset {opset}): Y dtype={y_onnx_dtype} (T1) differs "
                f"from X dtype={x_onnx_dtype} (T). Inserting CastNode Y → X's dtype."
            )

            cast_name = f"{node_name}_cast_y"
            cast_out_name = f"{cast_name}_output"
            x_torch_dtype = onnx_dtype_to_torch_dtype(x_onnx_dtype)

            cast_out_info = TensorInfo(
                name=cast_out_name,
                shape=tensor_y.shape,
                onnx_dtype=x_onnx_dtype,
            )
            cast_input_dict, cast_output_dict = build_input_output_dicts(
                node_proto,
                input_tensors,
                {cast_out_name: cast_out_info},
                input_names=[y_name],
                output_names=[cast_out_name],
            )
            nodes.append(
                CastNode.create(
                    name=cast_name,
                    inputs=cast_input_dict,
                    outputs=cast_output_dict,
                    dtype=x_torch_dtype,
                )
            )

            input_tensors = OrderedDict(input_tensors)
            input_tensors[cast_out_name] = cast_out_info
            pow_y_name = cast_out_name

        # ── Step 4: Build PowNode ─────────────────────────────────────────────
        input_dict, output_dict = build_input_output_dicts(
            node_proto,
            input_tensors,
            output_tensors,
            input_names=[x_name, pow_y_name],
        )
        nodes.append(PowNode.create(name=node_name, inputs=input_dict, outputs=output_dict))
        return nodes
