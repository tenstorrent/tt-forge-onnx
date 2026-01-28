# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Slice operation converter.

Converts ONNX Slice operations to TIR nodes. Slice is decomposed into a sequence
of IndexNode operations, one per axis, since Forge's Index operation only supports
single-axis slicing.

Supports opset versions 1, 10, 11, and 13.
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from onnx import NodeProto
import onnx
from loguru import logger

from forge.transpiler.core.types import TensorInfo
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts
from forge.transpiler.frontends.onnx.utils.validation import (
    validate_constant_input,
    ConverterValidationError,
)
from forge.transpiler.operations.indexing import IndexNode


class SliceConverter(OnnxOpConverter):
    """
    Converter for ONNX Slice operation.

    Supports opset versions 1, 10, 11, and 13.

    Conversion strategy:
    - Decompose multi-axis slicing into a sequence of IndexNode operations
    - Each IndexNode handles one axis
    - Process axes in order (innermost to outermost for better performance)
    - Preprocess: normalize axes, starts, ends, steps
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
        Convert ONNX Slice operation to TIR nodes.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (for v1) or empty dict (for v10+)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer (for constant extraction)
            opset: Opset version (1, 10, 11, or 13)

        Returns:
            List of IndexNode TIRNodes (one per axis)

        Raises:
            ConverterValidationError: If inputs are invalid or parameters cannot be extracted
        """
        node_name = node_proto.name or f"Slice_{node_index}"

        try:
            # Validate inputs
            cls._validate_inputs(node_proto, input_tensors, opset)

            # Extract slice parameters based on opset version
            starts, ends, axes, steps = cls._extract_slice_params(
                node_proto, attrs, graph_proto, opset, tir_graph=tir_graph
            )

            # Process slice operation
            return cls._process_slice(
                node_proto,
                input_tensors,
                output_tensors,
                starts,
                ends,
                axes,
                steps,
                node_name,
            )
        except (ConverterValidationError, ValueError) as e:
            logger.error(f"Slice node '{node_name}': {e}")
            raise

    @classmethod
    def _validate_inputs(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        opset: int,
    ) -> None:
        """
        Validate that required inputs are present.

        Raises:
            ConverterValidationError: If inputs are invalid
        """
        if opset == 1:
            # v1: Only data input required
            if len(node_proto.input) < 1:
                raise ConverterValidationError("Slice (opset v1) requires at least 1 input (data)")
            data_name = node_proto.input[0]
            if data_name not in input_tensors:
                raise ConverterValidationError(f"Slice data input '{data_name}' not found")
        else:
            # v10+: Requires data, starts, ends inputs
            if len(node_proto.input) < 3:
                raise ConverterValidationError(
                    f"Slice (opset v{opset}) requires at least 3 inputs (data, starts, ends)"
                )
            data_name = node_proto.input[0]
            if data_name not in input_tensors:
                raise ConverterValidationError(f"Slice data input '{data_name}' not found")

    @classmethod
    def _extract_slice_params(
        cls,
        node_proto: NodeProto,
        attrs: Dict[str, Any],
        graph_proto,
        opset: int,
        tir_graph=None,
    ) -> Tuple[List[int], List[int], Optional[List[int]], Optional[List[int]]]:
        """
        Extract slice parameters based on opset version.

        Args:
            node_proto: ONNX node proto
            attrs: Extracted attributes (for v1)
            graph_proto: Graph proto for constant extraction (for v10+)
            opset: Opset version

        Returns:
            Tuple of (starts, ends, axes, steps) as lists

        Raises:
            ConverterValidationError: If parameters cannot be extracted
        """
        node_name = node_proto.name or "Slice"

        if opset == 1:
            # v1: Extract from attributes
            starts = attrs.get("starts", [])
            ends = attrs.get("ends", [])
            axes = attrs.get("axes", None)
            steps = attrs.get("steps", [1] * len(starts))

            # Convert to lists of integers
            starts = cls._normalize_int_list(starts, "starts")
            ends = cls._normalize_int_list(ends, "ends")
            if axes is not None:
                axes = cls._normalize_int_list(axes, "axes")

            # Validate lengths match
            if len(starts) != len(ends):
                raise ConverterValidationError(
                    f"Slice node '{node_name}' (opset v1): "
                    f"starts and ends must have same length. "
                    f"Got starts={len(starts)}, ends={len(ends)}"
                )

        else:
            # v10+: Extract from inputs using validate_constant_input
            # Extract starts (required, input index 1)
            is_valid, starts, error_msg = validate_constant_input(
                node_proto, input_index=1, graph_proto=graph_proto, tir_graph=tir_graph
            )
            if not is_valid or starts is None:
                raise ConverterValidationError(
                    f"Slice node '{node_name}' requires constant 'starts' input. {error_msg or ''}"
                )
            starts = cls._normalize_int_list(starts, "starts")

            # Extract ends (required, input index 2)
            is_valid, ends, error_msg = validate_constant_input(
                node_proto, input_index=2, graph_proto=graph_proto, tir_graph=tir_graph
            )
            if not is_valid or ends is None:
                raise ConverterValidationError(
                    f"Slice node '{node_name}' requires constant 'ends' input. {error_msg or ''}"
                )
            ends = cls._normalize_int_list(ends, "ends")

            # Validate lengths match
            if len(starts) != len(ends):
                raise ConverterValidationError(
                    f"Slice node '{node_name}': starts and ends must have same length. "
                    f"Got starts={len(starts)}, ends={len(ends)}"
                )

            # Extract axes and steps (both optional)
            # Input order: [data, starts, ends, axes?, steps?]
            # If there are 5 inputs, index 3 is axes and index 4 is steps
            # If there are 4 inputs, index 3 could be axes OR steps (check input name)
            axes = None
            steps = None

            # Check input names to distinguish axes from steps when ambiguous
            input_names = list(node_proto.input)
            has_axes_by_name = len(input_names) > 3 and "axes" in [n.lower() for n in input_names[3:]]
            has_steps_by_name = any("steps" in n.lower() for n in input_names[3:])

            if len(node_proto.input) > 3:
                # Try to extract from index 3
                is_valid, value_at_3, error_msg = validate_constant_input(
                    node_proto, input_index=3, graph_proto=graph_proto, tir_graph=tir_graph
                )

                if is_valid and value_at_3 is not None:
                    value_at_3 = cls._normalize_int_list(value_at_3, "input[3]")

                    # Determine if index 3 is axes or steps
                    if len(node_proto.input) > 4:
                        # 5 inputs: index 3 is axes, index 4 is steps
                        axes = value_at_3
                        if len(axes) != len(starts):
                            raise ConverterValidationError(
                                f"Slice node '{node_name}': axes length ({len(axes)}) "
                                f"must match starts length ({len(starts)})"
                            )
                        # Extract steps from index 4
                        is_valid, steps, error_msg = validate_constant_input(
                            node_proto, input_index=4, graph_proto=graph_proto, tir_graph=tir_graph
                        )
                        if is_valid and steps is not None:
                            steps = cls._normalize_int_list(steps, "steps")
                            if len(steps) != len(starts):
                                raise ConverterValidationError(
                                    f"Slice node '{node_name}': steps length ({len(steps)}) "
                                    f"must match starts length ({len(starts)})"
                                )
                    else:
                        # 4 inputs: index 3 could be axes or steps
                        # Use input name to determine
                        input_3_name = input_names[3].lower() if len(input_names) > 3 else ""
                        if "axes" in input_3_name or (not has_steps_by_name and not "steps" in input_3_name):
                            # Likely axes
                            axes = value_at_3
                            if len(axes) != len(starts):
                                raise ConverterValidationError(
                                    f"Slice node '{node_name}': axes length ({len(axes)}) "
                                    f"must match starts length ({len(starts)})"
                                )
                        else:
                            # Likely steps
                            steps = value_at_3
                            if len(steps) != len(starts):
                                raise ConverterValidationError(
                                    f"Slice node '{node_name}': steps length ({len(steps)}) "
                                    f"must match starts length ({len(starts)})"
                                )

            # Default steps to 1 if not provided
            if steps is None:
                steps = [1] * len(starts)

        return starts, ends, axes, steps

    @classmethod
    def _normalize_int_list(cls, value: Any, param_name: str) -> List[int]:
        """
        Normalize a value to a list of integers.

        Args:
            value: Value to normalize (can be int, list, tuple, numpy array, etc.)
            param_name: Parameter name for error messages

        Returns:
            List of integers

        Raises:
            ConverterValidationError: If value cannot be converted to list of integers
        """
        try:
            if isinstance(value, (list, tuple)):
                return [int(x) for x in value]
            elif hasattr(value, "__iter__") and not isinstance(value, str):
                return [int(x) for x in value]
            else:
                return [int(value)]
        except (ValueError, TypeError) as e:
            raise ConverterValidationError(f"Slice parameter '{param_name}' contains non-integer values: {e}")

    @classmethod
    def _process_slice(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        starts: List[int],
        ends: List[int],
        axes: Optional[List[int]],
        steps: List[int],
        node_name: str,
    ) -> List:
        """
        Process slice operation by creating IndexNode for each axis.

        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            starts: List of start indices
            ends: List of end indices
            axes: Optional list of axes (defaults to [0, 1, ..., len(starts)-1])
            steps: List of step sizes
            node_name: Node name

        Returns:
            List of IndexNode TIRNodes
        """
        # Get data tensor
        data_name = node_proto.input[0]
        data_tensor = input_tensors[data_name]
        data_rank = len(data_tensor.shape) if data_tensor.shape else 0

        # Default axes if not provided
        if axes is None:
            axes = list(range(len(starts)))

        # Normalize negative axes
        axes = [ax + data_rank if ax < 0 else ax for ax in axes]

        # Validate axes
        for ax in axes:
            if ax < 0 or ax >= data_rank:
                raise ConverterValidationError(
                    f"Slice node '{node_name}': Invalid axis {ax} for tensor of rank {data_rank}"
                )

        # Validate steps (cannot be 0)
        for i, step in enumerate(steps):
            if step == 0:
                raise ConverterValidationError(f"Slice node '{node_name}': Step cannot be 0 for axis {axes[i]}")

        # Process each axis: create IndexNode chain
        nodes = []
        current_outputs = OrderedDict()
        current_outputs[data_name] = data_tensor
        current_data_name = data_name

        # Process axes in order (innermost first for better performance)
        for i, axis in enumerate(axes):
            start = starts[i]
            end = ends[i]
            step = steps[i]

            # Normalize and clamp indices
            start, end = cls._normalize_and_clamp_indices(start, end, step, axis, data_tensor, node_name)

            # Create IndexNode for this axis
            index_node_name = f"{node_name}_axis_{axis}"

            # Build input/output dicts
            if i == len(axes) - 1:
                # Last axis: use final output_tensors
                input_dict, output_dict = build_input_output_dicts(
                    node_proto,
                    current_outputs,
                    output_tensors,
                    input_names=[current_data_name],
                )
            else:
                # Intermediate axis: create intermediate output
                intermediate_output_name = f"{index_node_name}_output"
                intermediate_shape = cls._compute_output_shape(
                    current_outputs[current_data_name].shape, axis, start, end, step
                )

                intermediate_output = TensorInfo(
                    name=intermediate_output_name,
                    shape=intermediate_shape,
                    onnx_dtype=data_tensor.onnx_dtype,
                )
                current_outputs[intermediate_output_name] = intermediate_output

                input_dict, output_dict = build_input_output_dicts(
                    node_proto,
                    current_outputs,
                    {intermediate_output_name: intermediate_output},
                    input_names=[current_data_name],
                    output_names=[intermediate_output_name],
                )

            # Create IndexNode
            index_node = IndexNode.create(
                name=index_node_name,
                inputs=input_dict,
                outputs=output_dict,
                axis=axis,
                start=start,
                stop=end,
                stride=step,
            )
            nodes.append(index_node)

            # Update for next iteration
            if i < len(axes) - 1:
                current_data_name = list(output_dict.keys())[0]
                # Update data_tensor shape for next iteration
                if intermediate_shape:
                    current_outputs[current_data_name] = TensorInfo(
                        name=current_data_name,
                        shape=intermediate_shape,
                        onnx_dtype=data_tensor.onnx_dtype,
                    )

        return nodes

    @classmethod
    def _normalize_and_clamp_indices(
        cls,
        start: int,
        end: int,
        step: int,
        axis: int,
        data_tensor: TensorInfo,
        node_name: str,
    ) -> Tuple[int, int]:
        """
        Normalize negative indices and clamp to valid ranges.

        Args:
            start: Start index
            end: End index
            step: Step size
            axis: Axis index
            data_tensor: Data tensor info
            node_name: Node name for error messages

        Returns:
            Tuple of (normalized_start, normalized_end)
        """
        if not data_tensor.shape:
            # Unknown shape - cannot normalize/clamp
            return start, end

        axis_size = data_tensor.shape[axis]

        # Normalize negative indices
        if start < 0:
            start = axis_size + start
        if end < 0:
            end = axis_size + end

        # Clamp based on step direction
        if step > 0:
            # Positive step: clamp to [0, axis_size]
            start = max(0, min(start, axis_size))
            end = max(0, min(end, axis_size))
        else:
            # Negative step: clamp to [0, axis_size-1] for start, [-1, axis_size-1] for end
            start = max(0, min(start, axis_size - 1))
            end = max(-1, min(end, axis_size - 1))

        return start, end

    @classmethod
    def _compute_output_shape(
        cls,
        input_shape: Optional[Tuple],
        axis: int,
        start: int,
        end: int,
        step: int,
    ) -> Optional[Tuple]:
        """
        Compute output shape after slicing along one axis.

        Args:
            input_shape: Input tensor shape
            axis: Axis being sliced
            start: Start index
            end: End index
            step: Step size

        Returns:
            Output shape tuple or None if input_shape is None
        """
        if input_shape is None:
            return None

        output_shape = list(input_shape)

        # Compute output size along this axis
        if step > 0:
            output_size = max(0, (end - start + step - 1) // step)
        else:
            output_size = max(0, (start - end + abs(step) - 1) // abs(step))

        output_shape[axis] = output_size
        return tuple(output_shape)
