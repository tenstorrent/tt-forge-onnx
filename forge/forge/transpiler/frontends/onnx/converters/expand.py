# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Expand operation converter.

This module provides the converter for ONNX Expand operations, which broadcast
an input tensor to a target shape. The converter decomposes Expand into a sequence
of Unsqueeze and Broadcast operations, since Forge's Broadcast only supports
single-axis broadcasting.

Key features:
- Supports opset v8 and v13
- Decomposes multi-axis broadcasting into single-axis Broadcast operations
- Handles rank alignment using Unsqueeze operations
- Validates broadcasting compatibility
"""
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.shape import UnsqueezeNode, BroadcastNode
from forge.transpiler.operations.other import IdentityNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.validation import validate_constant_input
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts
from forge.transpiler.frontends.onnx.utils.constant_value_extractor import (
    resolve_constant_tensor_value,
)
from forge.transpiler.frontends.onnx.utils.shape_finder import validate_no_unknown_dimensions


class ExpandConverter(OnnxOpConverter):
    """Converter for ONNX Expand operation with decomposition into Unsqueeze and Broadcast nodes."""

    @classmethod
    def _extract_shape_input(
        cls,
        node_proto: NodeProto,
        graph_proto=None,
        tir_graph=None,
        output_tensors=None,
        input_tensors=None,
    ) -> Tuple[bool, List[int], Optional[str]]:
        """
        Extract shape values from the second input (shape tensor).

        Two strategies are tried in order:

        1. Constant subgraph evaluation via resolve_constant_tensor_value:
           Works when all sources in the backward trace are constants/parameters
           (including computed_constants from ONNX Constant nodes).
        2. Direct lookup in initializers / Constant nodes via validate_constant_input.

        If both strategies fail the shape input is a runtime-dependent tensor.
        The caller should use the OnnxRuntime concrete-shape pre-pass
        (pass module_inputs to transpile()) to resolve this before conversion.

        Args:
            node_proto: ONNX node proto
            graph_proto: ONNX graph proto (for accessing initializers)
            tir_graph: TIRGraph (for accessing constants and inline shape resolution)
            output_tensors: Unused (kept for API compatibility)
            input_tensors: Unused (kept for API compatibility)

        Returns:
            Tuple of (is_valid, shape_values, error_message)
        """
        if len(node_proto.input) < 2:
            return False, None, f"Expand requires 2 inputs, got {len(node_proto.input)}"

        shape_input_name = node_proto.input[1]

        # Strategy 1: constant subgraph evaluation (no model input dependencies)
        if tir_graph is not None and graph_proto is not None:
            resolved, shape_values, _ = resolve_constant_tensor_value(shape_input_name, tir_graph, graph_proto)
            if resolved and shape_values is not None:
                return True, shape_values, None

        # Strategy 2: direct initializer / Constant-node lookup
        is_valid, shape_value, _ = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto, input_name=shape_input_name, tir_graph=tir_graph
        )
        if is_valid and shape_value is not None:
            if isinstance(shape_value, (list, tuple)):
                try:
                    return True, [int(x) for x in shape_value], None
                except (ValueError, TypeError) as e:
                    return False, None, f"Shape values must be integers: {e}"
            elif isinstance(shape_value, (int, float)):
                return True, [int(shape_value)], None
            else:
                return False, None, f"Unexpected shape value type: {type(shape_value)}"

        return (
            False,
            None,
            (
                f"Expand node '{node_proto.name}': could not resolve shape input "
                f"'{shape_input_name}'. Tried: (1) constant subgraph evaluation, "
                f"(2) direct initializer/Constant-node lookup. "
                f"The shape input depends on runtime values and cannot be determined "
                f"at compile time. Use the OnnxRuntime concrete-shape pre-pass "
                f"(pass module_inputs to transpile()) to resolve this."
            ),
        )

    @classmethod
    def _compute_broadcast_shape(
        cls, input_shape: Tuple[int, ...], target_shape: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """
        Compute output shape from input and target shapes using NumPy-style broadcasting rules.

        Args:
            input_shape: Input tensor shape
            target_shape: Target shape from Expand's shape input

        Returns:
            Tuple of (output_shape, normalized_input_shape, normalized_target_shape)
            - output_shape: Final broadcasted output shape
            - normalized_input_shape: Input shape padded to match rank
            - normalized_target_shape: Target shape padded to match rank

        Raises:
            ValueError: If shapes are incompatible for broadcasting
        """
        validate_no_unknown_dimensions(input_shape, "Expand _compute_broadcast_shape input")
        validate_no_unknown_dimensions(target_shape, "Expand _compute_broadcast_shape target")

        input_dims = len(input_shape)
        target_dims = len(target_shape)

        # Right-align dimensions by padding shorter shape with 1s on the left
        if input_dims < target_dims:
            input_shape_padded = (1,) * (target_dims - input_dims) + input_shape
            target_shape_padded = target_shape
        elif target_dims < input_dims:
            input_shape_padded = input_shape
            target_shape_padded = (1,) * (input_dims - target_dims) + target_shape
        else:
            input_shape_padded = input_shape
            target_shape_padded = target_shape

        # Compute output shape and validate compatibility
        output_shape = []
        for i in range(len(input_shape_padded)):
            in_dim = input_shape_padded[i]
            target_dim = target_shape_padded[i]

            # Validate compatibility: dimensions must be equal, or one must be 1
            # (both dimensions are integers now)
            if in_dim != target_dim and in_dim != 1 and target_dim != 1:
                raise ValueError(
                    f"Incompatible dimensions at index {i}: input={in_dim}, target={target_dim}. "
                    f"For broadcasting, one dimension must be 1 or both must be equal."
                )

            # Output is maximum of the two (NumPy broadcasting rule)
            # Both dimensions are integers at this point
            output_shape.append(max(in_dim, target_dim))

        return tuple(output_shape), input_shape_padded, target_shape_padded

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
        Convert ONNX Expand operation to a sequence of Unsqueeze and Broadcast nodes.

        The decomposition follows the pattern from TVM and Forge passes:
        1. Extract target shape from second input
        2. Normalize shapes (right-align with 1s)
        3. Compute output shape using maximum logic
        4. Add Unsqueeze operations if input rank < target rank
        5. Add Broadcast operations for each dimension that needs expansion

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (typically empty for Expand)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer
            opset: Opset version (8 or 13)

        Returns:
            List of TIR nodes (UnsqueezeNode and BroadcastNode instances)

        Raises:
            ValueError: If inputs are invalid or shapes are incompatible
        """
        # Validate inputs
        if len(node_proto.input) < 2:
            raise ValueError(
                f"Expand node '{node_proto.name or f'Expand_{node_index}'}': "
                f"Expected 2 inputs, got {len(node_proto.input)}"
            )

        if len(node_proto.output) != 1:
            raise ValueError(
                f"Expand node '{node_proto.name or f'Expand_{node_index}'}': "
                f"Expected 1 output, got {len(node_proto.output)}"
            )

        # Extract input tensor info
        input_name = node_proto.input[0]
        if input_name not in input_tensors:
            raise ValueError(
                f"Expand node '{node_proto.name or f'Expand_{node_index}'}': "
                f"Input '{input_name}' not found in input_tensors"
            )

        input_info = input_tensors[input_name]
        input_shape = input_info.shape if input_info.shape else None

        if input_shape is None:
            raise ValueError(
                f"Expand node '{node_proto.name or f'Expand_{node_index}'}': " f"Cannot determine input shape"
            )

        # Extract target shape from second input (four strategies — see _extract_shape_input)
        is_valid, target_shape_list, error_msg = cls._extract_shape_input(
            node_proto,
            graph_proto,
            tir_graph=tir_graph,
            output_tensors=output_tensors,
            input_tensors=input_tensors,
        )
        if not is_valid:
            raise ValueError(f"Expand node '{node_proto.name or f'Expand_{node_index}'}': {error_msg}")

        target_shape = tuple(target_shape_list)

        # Compute broadcast output shape
        try:
            output_shape, input_shape_normalized, _ = cls._compute_broadcast_shape(input_shape, target_shape)
        except ValueError as e:
            raise ValueError(f"Expand node '{node_proto.name or f'Expand_{node_index}'}': {e}")

        # If output shape matches input shape (no broadcasting needed), return Identity.
        # Identity takes only the data input — drop the shape tensor (second input).
        if output_shape == input_shape:
            node_name = node_proto.name or f"Expand_{node_index}"
            data_input_dict, output_dict = build_input_output_dicts(
                node_proto,
                input_tensors,
                output_tensors,
                input_names=[input_name],  # data only, not the shape tensor
                output_names=[node_proto.output[0]],
            )
            return [IdentityNode.create(name=node_name, inputs=data_input_dict, outputs=output_dict)]

        # Generate node name
        node_name = node_proto.name or f"Expand_{node_index}"

        # Build initial input/output dicts
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        # Pre-compute which dimensions need broadcasting (before Unsqueeze loop)
        # This helps determine if last Unsqueeze should use final output name
        broadcast_dims = [
            i for i, (in_dim, out_dim) in enumerate(zip(input_shape_normalized, output_shape)) if in_dim != out_dim
        ]
        has_broadcast_ops = len(broadcast_dims) > 0

        # Phase 1: Add Unsqueeze operations if input rank < target rank
        nodes = []
        current_input_dict = input_dict.copy()
        current_shape = list(input_shape)
        onnx_dtype = getattr(input_info, "onnx_dtype", None)

        input_rank = len(input_shape)
        target_rank = len(target_shape)
        num_unsqueezes = max(0, target_rank - input_rank)

        for i in range(num_unsqueezes):
            is_last_unsqueeze = i == num_unsqueezes - 1

            # If this is the last Unsqueeze AND there are no broadcast operations, use final output name
            if is_last_unsqueeze and not has_broadcast_ops:
                node_outputs = [node_proto.output[0]]
                node_output_tensors = output_tensors.copy()
            else:
                intermediate_name = f"{node_name}_unsqueeze_{i}"
                node_outputs = [intermediate_name]
                # Calculate intermediate shape: insert dimension of size 1 at position 0
                intermediate_shape = tuple([1] + current_shape)
                node_output_tensors = {
                    intermediate_name: TensorInfo(
                        name=intermediate_name, shape=intermediate_shape, onnx_dtype=onnx_dtype
                    )
                }

            # Build OrderedDict for this UnsqueezeNode
            data_input_name = list(current_input_dict.keys())[0]
            node_input_dict, node_output_dict = build_input_output_dicts(
                node_proto,
                current_input_dict,
                node_output_tensors,
                input_names=[data_input_name],
                output_names=node_outputs,
            )

            # Create UnsqueezeNode at position 0 (leftmost)
            nodes.append(
                UnsqueezeNode.create(
                    name=f"{node_name}_unsqueeze_{i}",
                    inputs=node_input_dict,
                    outputs=node_output_dict,
                    dim=0,  # Always unsqueeze at position 0 (leftmost)
                )
            )

            # Update for next iteration
            current_input_dict = node_output_dict.copy()
            if is_last_unsqueeze and not has_broadcast_ops:
                # Final shape reached - no more operations needed
                current_shape = list(output_shape)
            else:
                # Calculate intermediate shape: insert dimension of size 1 at position 0
                intermediate_shape = tuple([1] + current_shape)
                current_shape = list(intermediate_shape)

        # Phase 2: Add Broadcast operations for each dimension that needs expansion
        # Process dimensions from left to right (matching Forge passes pattern)
        # Note: broadcast_dims was already computed above

        for idx, i in enumerate(broadcast_dims):
            is_last_broadcast = idx == len(broadcast_dims) - 1

            if is_last_broadcast:
                # Last broadcast - use final output name
                node_outputs = [node_proto.output[0]]
                node_output_tensors = output_tensors.copy()
                one_axis_target_shape = tuple(output_shape)
            else:
                # Intermediate broadcast
                intermediate_name = f"{node_name}_broadcast_{i}"
                node_outputs = [intermediate_name]

                # Calculate intermediate shape: build shape up to this dimension
                # More efficient: use tuple slicing instead of list conversion
                one_axis_target_shape = tuple(output_shape[: i + 1]) + tuple(current_shape[i + 1 :])

                node_output_tensors = {
                    intermediate_name: TensorInfo(
                        name=intermediate_name, shape=one_axis_target_shape, onnx_dtype=onnx_dtype
                    )
                }

            # Build OrderedDict for this BroadcastNode
            data_input_name = list(current_input_dict.keys())[0]
            node_input_dict, node_output_dict = build_input_output_dicts(
                node_proto,
                current_input_dict,
                node_output_tensors,
                input_names=[data_input_name],
                output_names=node_outputs,
            )

            # Create BroadcastNode with target shape that only changes one dimension
            nodes.append(
                BroadcastNode.create(
                    name=f"{node_name}_broadcast_{i}",
                    inputs=node_input_dict,
                    outputs=node_output_dict,
                    output_shape=one_axis_target_shape,
                )
            )

            # Update for next iteration
            current_input_dict = node_output_dict.copy()
            current_shape = list(one_axis_target_shape)

        return nodes
