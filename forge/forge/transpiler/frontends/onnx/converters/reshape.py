# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Reshape operation converter with opset version support.

This module provides the converter for ONNX Reshape operations, which reshape
tensors to new shapes. The converter handles multiple opset versions with
different attribute/input patterns and optimizes special cases.

Key features:
- Supports opset v1-v4 (shape as attribute) and v5+ (shape as input)
- Handles shape values with -1 (inferred dimension)
- Optimizes no-op reshapes to IdentityNode
- Validates shape compatibility (total elements must match)
"""
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict
from onnx import NodeProto
import torch
import numpy as np
from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.operations.shape import ReshapeNode
from forge.transpiler.operations.other import FullNode, IdentityNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.validation import validate_constant_input, handle_validation_error
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts
from forge.transpiler.frontends.onnx.utils.shape_finder import validate_no_unknown_dimensions
from forge.transpiler.frontends.onnx.utils.constant_value_extractor import resolve_constant_tensor_value


class ReshapeConverter(OnnxOpConverter):
    """Converter for ONNX Reshape operation with opset version support."""

    @classmethod
    def _normalize_shape_value(cls, shape_value: Any) -> Tuple:
        """
        Normalize shape value to tuple of integers.

        Args:
            shape_value: Shape value (can be int, list, tuple, numpy array, etc.)

        Returns:
            Tuple of integers (or empty tuple for None)
        """
        if shape_value is None:
            return ()

        # Handle scalar integers
        if isinstance(shape_value, (int, np.integer)):
            return (int(shape_value),)

        # Handle sequences (tuple, list) - most common case
        if isinstance(shape_value, (tuple, list)):
            return tuple(int(x) for x in shape_value)

        # Handle numpy arrays
        if isinstance(shape_value, np.ndarray):
            return tuple(int(x) for x in shape_value.flatten())

        # Handle other iterables (but not strings)
        if hasattr(shape_value, "__iter__") and not isinstance(shape_value, (str, bytes)):
            try:
                return tuple(int(x) for x in shape_value)
            except (TypeError, ValueError):
                # Fallback: try to convert the whole value to int (scalar case)
                pass

        # Fallback: try to convert scalar value to int
        try:
            return (int(shape_value),)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot normalize shape value {shape_value} (type: {type(shape_value)}) " f"to tuple of integers: {e}"
            )

    @classmethod
    def _resolve_shape(cls, shape: Tuple, input_shape: Tuple, allowzero: int = 0) -> Tuple:
        """
        Resolve shape by converting -1 and 0 to actual dimension values.

        torch.reshape supports -1 but NOT 0. So we need to resolve:
        - -1: Infer from total elements and other dimensions
        - 0 with allowzero=0: Copy from input shape
        - 0 with allowzero=1: Keep as 0 (will be handled by creating Constant node)

        Args:
            shape: Target shape tuple (may contain -1, 0)
            input_shape: Input tensor shape
            allowzero: 0 = copy from input, 1 = explicit zero

        Returns:
            Resolved shape tuple (may still contain 0 if allowzero=1)
        """
        # Validate shape is a list or tuple
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"Shape must be a list or tuple, got {type(shape).__name__}: {shape}")

        shape = list(shape)
        input_shape = tuple(input_shape) if input_shape else ()
        validate_no_unknown_dimensions(input_shape, "Reshape _resolve_shape input")

        # Validate that shape doesn't contain both -1 and 0
        has_neg_one = -1 in shape
        contains_zero = 0 in shape
        if has_neg_one and contains_zero:
            raise ValueError(
                f"Shape cannot contain both -1 (inferred dimension) and 0 (copy/explicit zero). " f"Shape: {shape}"
            )

        # Calculate total elements from input (already validated to have no unknown dims)
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim

        # Handle 0 dimensions based on allowzero
        has_zero_kept = False  # Track if we kept a 0 with allowzero=1

        for i, s in enumerate(shape):
            if s == 0:
                if allowzero == 1:
                    # Keep as 0 (will create Constant node for empty tensor)
                    has_zero_kept = True
                else:
                    # Copy from input (default behavior, backward compatible)
                    if i < len(input_shape):
                        input_dim = input_shape[i]
                        shape[i] = input_dim
                    else:
                        raise ValueError(f"Cannot copy dimension {i} from input shape {input_shape}")

        # Handle -1 (inferred dimension) - only if no zeros with allowzero=1
        if -1 in shape and not (has_zero_kept and allowzero == 1):
            # Check for multiple -1 values (invalid)
            neg_one_indices = [i for i, s in enumerate(shape) if s == -1]
            if len(neg_one_indices) > 1:
                raise ValueError(f"Cannot infer dimension: shape contains multiple -1 values. " f"Shape: {shape}")

            inferred_idx = neg_one_indices[0]
            # Calculate product of known dimensions
            known_product = 1
            for i, s in enumerate(shape):
                if i != inferred_idx:
                    if isinstance(s, str) or s is None:
                        raise ValueError(
                            f"Cannot infer -1: shape contains unknown dimension at index {i}: {s}. "
                            f"Unknown dimensions are not supported."
                        )
                    known_product *= s if s > 0 else 1

            if known_product == 0:
                raise ValueError(f"Cannot infer dimension when product of other dimensions is 0. " f"Shape: {shape}")
            # Calculate inferred dimension
            inferred_dim = total_elements // known_product
            if total_elements % known_product != 0:
                raise ValueError(
                    f"Cannot reshape tensor of size {total_elements} into shape {shape} "
                    f"(product of known dims: {known_product})"
                )
            shape[inferred_idx] = inferred_dim

        # Validate that resolved shape matches total elements (if no -1 or 0 with allowzero=1)
        if -1 not in shape and not (has_zero_kept and allowzero == 1):
            resolved_product = 1
            for s in shape:
                if isinstance(s, str) or s is None:
                    raise ValueError(f"Shape contains unknown dimension: {s}. Unknown dimensions are not supported.")
                resolved_product *= s if s > 0 else 1

            if resolved_product != total_elements:
                raise ValueError(
                    f"Cannot reshape tensor of size {total_elements} into shape {shape} "
                    f"(product: {resolved_product})"
                )

        return tuple(shape)

    @classmethod
    def _convert_reshape_impl(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        shape: Tuple,
        allowzero: int,
        node_index: int,
    ) -> List:
        """
        Common implementation for Reshape conversion across all opset versions.

        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            shape: Normalized shape tuple (already extracted and normalized)
            allowzero: allowzero value (0 or 1)
            node_index: Node index for naming

        Returns:
            List of TIR nodes (ReshapeNode or FullNode)
        """
        node_name = node_proto.name if node_proto.name else f"Reshape_{node_index}"
        data_input = node_proto.input[0]

        # Get input info
        input_info = input_tensors.get(data_input)
        if input_info is None:
            error_msg = f"Reshape {node_name}: input tensor {data_input} not found"
            handle_validation_error(node_proto, error_msg, strict=True)

        input_shape = input_info.shape if input_info.shape else ()
        input_dtype = input_info.onnx_dtype if hasattr(input_info, "onnx_dtype") else None

        # Build OrderedDict for inputs and outputs
        input_dict, output_dict = build_input_output_dicts(
            node_proto, input_tensors, output_tensors, input_names=[data_input]
        )

        # Handle empty shape () or len(shape) == 0: create ReshapeNode with shape (-1) to flatten
        if shape == () or (isinstance(shape, tuple) and len(shape) == 0):
            return [ReshapeNode.create(name=node_name, inputs=input_dict, outputs=output_dict, shape=(-1,))]

        # Resolve -1 and 0 in shape
        resolved_shape = cls._resolve_shape(shape, input_shape, allowzero)

        # Check if resolved shape contains 0 with allowzero=1 (empty tensor)
        # Create FullNode for empty tensor instead of ReshapeNode
        if 0 in resolved_shape and allowzero == 1:
            torch_dtype = onnx_dtype_to_torch_dtype(input_dtype) if input_dtype else torch.float32
            # FullNode has no inputs
            empty_input_dict = OrderedDict()
            return [
                FullNode.create(
                    name=node_name,
                    inputs=empty_input_dict,
                    outputs=output_dict,
                    shape=resolved_shape,
                    fill_value=0.0,
                    dtype=torch_dtype,
                )
            ]

        # Optimization: If input shape and resolved shape are the same, use Identity
        if input_shape == resolved_shape:
            return [IdentityNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]

        # Normal case: create ReshapeNode
        return [ReshapeNode.create(name=node_name, inputs=input_dict, outputs=output_dict, shape=resolved_shape)]

    @classmethod
    def _extract_shape_from_input(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        graph_proto,
        tir_graph=None,
    ) -> Tuple[Tuple, Optional[str]]:
        """
        Extract and normalize the shape tensor for opset >= 5.

        Two strategies are tried in order, mirroring ExpandConverter:

        1. **Constant subgraph evaluation** via ``resolve_constant_tensor_value``:
           Traces the shape tensor backward through the TIR graph and evaluates
           any constant-only subgraph (e.g. a ``Concat`` of ``Shape``/``Gather``/
           ``Unsqueeze`` outputs).  This handles the common GPT-2 / transformer
           pattern where the target shape is assembled at model-compilation time
           from static dimension values.

        2. **Direct lookup** via ``validate_constant_input``:
           Checks initializers and inline ``Constant`` nodes directly.  Sufficient
           when the shape tensor is a plain weight or a single ``Constant`` node
           output.

        If both strategies fail the shape tensor depends on runtime values and
        the conversion is aborted with a descriptive error.

        Returns:
            ``(normalized_shape, None)`` on success, or ``(None, error_message)``
            on failure.
        """
        if len(node_proto.input) < 2:
            return None, "Reshape requires 2 inputs (data, shape)"

        shape_input_name = node_proto.input[1]
        node_name = node_proto.name or shape_input_name

        # Strategy 1: constant subgraph evaluation (handles Concat/Shape/Gather chains)
        if tir_graph is not None and graph_proto is not None:
            resolved, shape_values, _ = resolve_constant_tensor_value(shape_input_name, tir_graph, graph_proto)
            if resolved and shape_values is not None:
                return cls._normalize_shape_value(shape_values), None

        # Strategy 2: direct initializer / Constant-node lookup
        is_valid, shape_value, error_msg = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto, tir_graph=tir_graph
        )
        if is_valid and shape_value is not None:
            return cls._normalize_shape_value(shape_value), None

        # Both strategies failed — shape is a runtime-dependent tensor
        return (
            None,
            error_msg
            or (
                f"Reshape (node: {node_name}): Node '{node_name}' requires constant input "
                f"'{shape_input_name}' but it was not found in any compile-time constant store. "
                f"Tried: (1) constant subgraph evaluation, "
                f"(2) direct initializer/Constant-node lookup. "
                f"Dynamic inputs are not supported."
            ),
        )

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
        Reshape converter with opset-based shape extraction and allowzero handling.

        - Opset v1-v4: shape as attribute, allowzero=0 (not supported)
        - Opset v5-v13: shape as input tensor, allowzero=0 (default)
        - Opset v14+: shape as input tensor, allowzero attribute introduced
        """
        node_name = node_proto.name if node_proto.name else f"Reshape_{node_index}"

        if opset < 5:
            # v1-v4: shape as attribute
            shape = attrs.get("shape", None)
            if shape is None:
                error_msg = f"Reshape {node_name} (opset < 5) requires 'shape' attribute"
                handle_validation_error(node_proto, error_msg, strict=True)
            shape = cls._normalize_shape_value(shape)
            allowzero = 0
        else:
            # v5+: shape as input tensor
            shape, error_msg = cls._extract_shape_from_input(
                node_proto, input_tensors, output_tensors, graph_proto, tir_graph=tir_graph
            )
            if shape is None:
                handle_validation_error(node_proto, error_msg, strict=True)

            if opset < 14:
                # v5-v13: allowzero defaults to 0
                allowzero = 0
            else:
                # v14+: allowzero attribute
                allowzero = int(attrs.get("allowzero", 0))

        # Common conversion logic
        return cls._convert_reshape_impl(node_proto, input_tensors, output_tensors, shape, allowzero, node_index)
