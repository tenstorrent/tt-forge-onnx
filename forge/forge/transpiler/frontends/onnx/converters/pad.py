# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Pad operation converter with opset version support.

This module provides the converter for ONNX Pad operations, which add padding to
tensors along specified dimensions. The converter handles multiple opset versions
with different attribute/input patterns and converts ONNX padding modes to PyTorch
compatible formats.

Key features:
- Supports opset v1-v18 (attributes) and v11+ (inputs)
- Handles multiple padding modes: constant, reflect, edge (replicate), wrap (circular)
- Converts ONNX padding format to PyTorch format
- Validates padding values for non-constant modes
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from onnx import NodeProto
from loguru import logger
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.other import PadNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.validation import (
    validate_constant_input,
    handle_validation_error,
    ValidationError,
)
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


def onnx_mode_to_pytorch(onnx_mode: str) -> str:
    """
    Convert ONNX Pad mode to PyTorch F.pad mode.

    ONNX and PyTorch use different names for the same padding behaviors:
    - ONNX 'edge' = PyTorch 'replicate' (replicates edge values)
    - ONNX 'wrap' = PyTorch 'circular' (wraps around, periodic)
    - ONNX 'constant' = PyTorch 'constant' (same name)
    - ONNX 'reflect' = PyTorch 'reflect' (same name)

    Args:
        onnx_mode: ONNX padding mode ('constant', 'reflect', 'edge', 'wrap')

    Returns:
        PyTorch padding mode ('constant', 'reflect', 'replicate', 'circular')

    Raises:
        ValueError: If onnx_mode is not supported
    """
    mode_mapping = {
        "constant": "constant",
        "reflect": "reflect",
        "edge": "replicate",  # ONNX edge = PyTorch replicate
        "wrap": "circular",  # ONNX wrap = PyTorch circular
    }

    if onnx_mode not in mode_mapping:
        raise ValueError(f"Unsupported ONNX Pad mode: {onnx_mode}. " f"Supported modes: {list(mode_mapping.keys())}")

    return mode_mapping[onnx_mode]


def validate_pytorch_nonconstant_padding(
    pads: List[int], axes: Optional[List[int]], input_rank: int, mode: str
) -> None:
    """
    Validate that padding configuration is compatible with PyTorch's constraints
    for non-constant padding modes.

    PyTorch F.pad constraints for non-constant modes (reflect, replicate, circular):
    - 2D tensors: Can pad only the last 1 dimension
    - 3D tensors: Can pad the last 1-2 dimensions
    - 4D tensors: Can pad the last 2-3 dimensions (Pad2d or Pad3d style)
    - 5D tensors: Can pad only the last 3 dimensions

    Reference: PyTorch documentation states:
    "Circular, replicate and reflection padding are implemented for padding the last 3
    dimensions of a 4D or 5D input tensor, the last 2 dimensions of a 3D or 4D input
    tensor, or the last dimension of a 2D or 3D input tensor."

    Note: For 4D tensors, reflect/replicate/circular modes can pad:
    - Last 2 dimensions (dims 2, 3) - corresponds to Pad2d classes (ReflectionPad2d, etc.)
    - Last 3 dimensions (dims 1, 2, 3) - corresponds to Pad3d classes (ReflectionPad3d, etc.)

    Args:
        pads: ONNX pads list [begin_0, begin_1, ..., end_0, end_1, ...]
        axes: Optional list of axes being padded (for selective padding, opset 18+)
        input_rank: Input tensor rank
        mode: PyTorch padding mode ('constant', 'reflect', 'replicate', 'circular')

    Raises:
        ValueError: If padding configuration violates PyTorch constraints
    """
    # Constant mode has no constraints
    if mode == "constant":
        return

    # Determine which dimensions are being padded
    num_padded_dims = len(pads) // 2
    begins = pads[:num_padded_dims]
    ends = pads[num_padded_dims:]

    # Find which dimensions have non-zero padding
    padded_dims = []
    if axes is not None:
        # Selective padding: check which axes are being padded
        for i, axis in enumerate(axes):
            normalized_axis = axis if axis >= 0 else input_rank + axis
            if begins[i] != 0 or ends[i] != 0:
                padded_dims.append(normalized_axis)
    else:
        # Full padding: check all dimensions
        for i in range(num_padded_dims):
            if begins[i] != 0 or ends[i] != 0:
                padded_dims.append(i)

    if not padded_dims:
        # No padding, nothing to validate
        return

    # Check if padded dimensions are the last N dimensions
    # PyTorch constraints (from official documentation):
    # - 2D: last 1 dim (Pad1d)
    # - 3D: last 1-2 dims (Pad1d or Pad2d)
    # - 4D: last 2-3 dims (Pad2d or Pad3d)
    # - 5D: last 3 dims (Pad3d)
    max_padded_dims = {
        2: 1,  # 2D: last 1 dim
        3: 2,  # 3D: last 1-2 dims
        4: 3,  # 4D: last 2-3 dims (can use Pad2d or Pad3d)
        5: 3,  # 5D: last 3 dims
    }

    # Minimum number of dimensions that can be padded (for validation)
    min_padded_dims = {
        2: 1,  # 2D: must pad last 1 dim
        3: 1,  # 3D: can pad last 1-2 dims
        4: 2,  # 4D: can pad last 2-3 dims (minimum is 2)
        5: 3,  # 5D: must pad last 3 dims
    }

    if input_rank not in max_padded_dims:
        raise ValueError(f"PyTorch F.pad with {mode} mode only supports 2D-5D tensors. " f"Got {input_rank}D tensor.")

    max_allowed = max_padded_dims[input_rank]
    min_allowed = min_padded_dims[input_rank]

    # Check if we're padding more dimensions than allowed
    if len(padded_dims) > max_allowed:
        raise ValueError(
            f"PyTorch F.pad with {mode} mode can pad at most the last {max_allowed} "
            f"dimension(s) of a {input_rank}D tensor, but {len(padded_dims)} dimension(s) "
            f"are being padded: {padded_dims}. "
            f"Padded dimensions must be the last N dimensions: "
            f"{list(range(input_rank - max_allowed, input_rank))}"
        )

    # Check if we're padding fewer dimensions than the minimum required
    # For 4D: must pad at least 2 dims (last 2 or last 3)
    # For 5D: must pad exactly 3 dims (last 3)
    if len(padded_dims) < min_allowed:
        raise ValueError(
            f"PyTorch F.pad with {mode} mode requires padding at least the last {min_allowed} "
            f"dimension(s) of a {input_rank}D tensor, but only {len(padded_dims)} dimension(s) "
            f"are being padded: {padded_dims}. "
            f"For {input_rank}D tensors, you must pad the last {min_allowed} to {max_allowed} dimensions."
        )

    # Check if padded dimensions are the last N dimensions
    expected_dims = list(range(input_rank - len(padded_dims), input_rank))
    if padded_dims != expected_dims:
        raise ValueError(
            f"PyTorch F.pad with {mode} mode can only pad the last {len(padded_dims)} "
            f"dimension(s) of a {input_rank}D tensor, but dimensions {padded_dims} are being padded. "
            f"Expected dimensions: {expected_dims}. "
            f"This is a PyTorch limitation - non-constant padding modes can only pad "
            f"the trailing dimensions, not arbitrary dimensions."
        )


def convert_onnx_pads_to_pytorch(
    onnx_pads: List[int], axes: Optional[List[int]] = None, input_rank: int = None, mode: str = "constant"
) -> Tuple[int, ...]:
    """
    Convert ONNX pads format to PyTorch pad format.

    ONNX format: [axis0_begin, axis1_begin, ..., axis0_end, axis1_end, ...]
    PyTorch format: (axis(n-1)_begin, axis(n-1)_end, ..., axis0_begin, axis0_end)

    Args:
        onnx_pads: ONNX pads list [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        axes: Optional list of axes being padded (for selective padding, opset 18+)
        input_rank: Input tensor rank (used when axes is None)
        mode: PyTorch padding mode for validation ('constant', 'reflect', 'replicate', 'circular')

    Returns:
        PyTorch pad tuple: (last_dim_begin, last_dim_end, ..., first_dim_begin, first_dim_end)

    Raises:
        ValueError: If pads format is invalid, axes are out of range, or PyTorch constraints violated
    """
    if not onnx_pads:
        return tuple()

    # Validate pads length is even
    if len(onnx_pads) % 2 != 0:
        raise ValueError(
            f"Invalid pads format: length must be even (got {len(onnx_pads)}). "
            f"Expected format: [begin_0, begin_1, ..., end_0, end_1, ...]"
        )

    num_axes = len(onnx_pads) // 2
    begins = onnx_pads[:num_axes]
    ends = onnx_pads[num_axes:]

    # If axes is provided, we need to create full padding for all dimensions
    if axes is not None and input_rank is not None:
        # Validate axes count matches pads
        if len(axes) != num_axes:
            raise ValueError(
                f"Number of axes ({len(axes)}) must match number of padded dimensions "
                f"({num_axes}). Pads format: [begin_0, ..., begin_{num_axes-1}, "
                f"end_0, ..., end_{num_axes-1}]"
            )

        # Validate input rank
        if input_rank <= 0:
            raise ValueError(f"Invalid input_rank: {input_rank}. Must be positive.")

        # Create full padding list for all axes
        full_pads = [0] * (2 * input_rank)
        for i, axis in enumerate(axes):
            # Normalize axis to positive
            if axis < 0:
                axis = input_rank + axis

            # Validate axis is within range
            if axis < 0 or axis >= input_rank:
                raise ValueError(
                    f"Invalid axis value: {axes[i]}. After normalization: {axis}. "
                    f"Must be in range [{-input_rank}, {input_rank - 1}]"
                )

            full_pads[2 * axis] = begins[i]  # begin
            full_pads[2 * axis + 1] = ends[i]  # end

        begins = [full_pads[2 * i] for i in range(input_rank)]
        ends = [full_pads[2 * i + 1] for i in range(input_rank)]
        num_axes = input_rank

    # Validate PyTorch constraints for non-constant modes
    # Create a pads list in ONNX format for validation
    validation_pads = []
    if axes is not None:
        # For selective padding, use the original pads
        validation_pads = onnx_pads
    else:
        # For full padding, reconstruct from begins/ends
        validation_pads = begins + ends

    validate_pytorch_nonconstant_padding(validation_pads, axes, input_rank, mode)

    # For non-constant modes, PyTorch only accepts padding for the trailing dimensions
    # We need to strip leading zero padding and only include the dimensions that are actually being padded
    if mode != "constant":
        if axes is None:
            # For full padding: Find the first dimension with non-zero padding
            first_padded_dim = None
            for i in range(num_axes):
                if begins[i] != 0 or ends[i] != 0:
                    first_padded_dim = i
                    break

            if first_padded_dim is not None:
                # Only include padding from first_padded_dim onwards (trailing dimensions)
                # PyTorch expects padding only for the dimensions being padded
                begins = begins[first_padded_dim:]
                ends = ends[first_padded_dim:]
                num_axes = len(begins)
        else:
            # For selective padding with axes: We need to extract only the trailing padded dimensions
            # Find which dimensions are actually being padded (from the full_pads we created)
            # and extract only the trailing ones
            padded_dims = []
            for i in range(input_rank):
                if begins[i] != 0 or ends[i] != 0:
                    padded_dims.append(i)

            if padded_dims:
                # Find the first padded dimension
                first_padded_dim = padded_dims[0]
                # Only include padding from first_padded_dim onwards (trailing dimensions)
                begins = begins[first_padded_dim:]
                ends = ends[first_padded_dim:]
                num_axes = len(begins)

    # PyTorch pads in reverse order (last dimension first)
    pytorch_pad = []
    for i in range(num_axes - 1, -1, -1):
        pytorch_pad.extend([begins[i], ends[i]])

    return tuple(pytorch_pad)


class PadConverter(OnnxOpConverter):
    """Converter for ONNX Pad operation with opset version support."""

    @classmethod
    def _validate_inputs(cls, node_proto: NodeProto, input_tensors: OrderedDict[str, TensorInfo]) -> None:
        """Validate that required inputs exist."""
        if not node_proto.input:
            raise ValidationError(
                f"Pad node '{node_proto.name or 'unknown'}' has no inputs. " f"At least one input (data) is required."
            )

        data_input = node_proto.input[0]
        if data_input not in input_tensors:
            raise ValidationError(
                f"Pad node '{node_proto.name or 'unknown'}' input '{data_input}' " f"not found in input_tensors."
            )

        tensor_info = input_tensors[data_input]
        if tensor_info.shape is None or len(tensor_info.shape) == 0:
            raise ValidationError(
                f"Pad node '{node_proto.name or 'unknown'}' input '{data_input}' "
                f"has invalid shape: {tensor_info.shape}"
            )

    @classmethod
    def _extract_and_validate_pads(
        cls, node_proto: NodeProto, attrs: Dict[str, Any], graph_proto, input_index: int = 1
    ) -> List[int]:
        """
        Extract and validate pads from input or attribute.

        Returns:
            List of pad values as integers

        Raises:
            ValidationError: If pads cannot be extracted or are invalid
        """
        # Try to extract from input first
        is_valid, pads, error_msg = validate_constant_input(
            node_proto, input_index=input_index, graph_proto=graph_proto
        )

        if is_valid and pads is not None:
            # Convert to list of integers
            try:
                if isinstance(pads, (list, tuple)):
                    pads = [int(x) for x in pads]
                elif hasattr(pads, "__iter__") and not isinstance(pads, str):
                    pads = [int(x) for x in pads]
                else:
                    pads = [int(pads)]
            except (ValueError, TypeError) as e:
                raise ValidationError(
                    f"Pad node '{node_proto.name or 'unknown'}' pads input contains " f"non-integer values: {e}"
                )
        else:
            # Fallback to attribute
            pads = attrs.get("pads", attrs.get("paddings", []))
            if not pads:
                raise ValidationError(
                    f"Pad node '{node_proto.name or 'unknown'}' requires 'pads' input "
                    f"or attribute, but neither was found. {error_msg or ''}"
                )
            # Convert to list of integers
            try:
                if isinstance(pads, (list, tuple)):
                    pads = [int(x) for x in pads]
                elif hasattr(pads, "__iter__") and not isinstance(pads, str):
                    pads = [int(x) for x in pads]
                else:
                    pads = [int(pads)]
            except (ValueError, TypeError) as e:
                raise ValidationError(
                    f"Pad node '{node_proto.name or 'unknown'}' pads attribute contains " f"non-integer values: {e}"
                )

        # Validate pads format
        if len(pads) % 2 != 0:
            raise ValidationError(
                f"Pad node '{node_proto.name or 'unknown'}' pads must have even length "
                f"(got {len(pads)}). Format: [begin_0, begin_1, ..., end_0, end_1, ...]"
            )

        if len(pads) == 0:
            raise ValidationError(f"Pad node '{node_proto.name or 'unknown'}' pads cannot be empty.")

        return pads

    @classmethod
    def _extract_constant_value(
        cls, node_proto: NodeProto, attrs: Dict[str, Any], graph_proto, input_index: int = 2, default: float = 0.0
    ) -> float:
        """Extract constant_value from input or attribute."""
        if len(node_proto.input) > input_index:
            is_valid, constant_value, _ = validate_constant_input(
                node_proto, input_index=input_index, graph_proto=graph_proto
            )
            if is_valid and constant_value is not None:
                try:
                    return float(constant_value)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Pad node '{node_proto.name or 'unknown'}' constant_value "
                        f"cannot be converted to float: {e}. Using default {default}."
                    )
                    return default

        # Fallback to attribute
        value = attrs.get("value", default)
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Pad node '{node_proto.name or 'unknown'}' value attribute "
                f"cannot be converted to float: {e}. Using default {default}."
            )
            return default

    @classmethod
    def _extract_axes(cls, node_proto: NodeProto, graph_proto, input_index: int = 3) -> Optional[List[int]]:
        """Extract axes from input (opset 18+)."""
        if len(node_proto.input) <= input_index:
            return None

        is_valid, axes_value, _ = validate_constant_input(node_proto, input_index=input_index, graph_proto=graph_proto)

        if not is_valid or axes_value is None:
            return None

        try:
            if isinstance(axes_value, (list, tuple)):
                axes = [int(x) for x in axes_value]
            elif hasattr(axes_value, "__iter__") and not isinstance(axes_value, str):
                axes = [int(x) for x in axes_value]
            else:
                axes = [int(axes_value)]
            return axes
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"Pad node '{node_proto.name or 'unknown'}' axes input contains " f"non-integer values: {e}"
            )

    @classmethod
    def _validate_mode_for_opset(cls, mode: str, opset_version: int, node_name: str) -> None:
        """Validate that the mode is supported for the given opset version."""
        if mode == "wrap" and opset_version < 19:
            raise ValidationError(
                f"Pad node '{node_name}' uses 'wrap' mode which is only available "
                f"in opset 19+, but model uses opset {opset_version}."
            )

    @classmethod
    def _get_input_rank(cls, input_tensors: OrderedDict[str, TensorInfo], node_proto: NodeProto) -> int:
        """Get input rank from tensor info only. Throws error if rank cannot be determined."""
        if not input_tensors:
            raise ValidationError(
                f"Pad node '{node_proto.name or 'unknown'}' cannot determine input rank: "
                f"input_tensors is empty or None."
            )

        data_input = node_proto.input[0]
        if data_input not in input_tensors:
            raise ValidationError(
                f"Pad node '{node_proto.name or 'unknown'}' cannot determine input rank: "
                f"data input '{data_input}' not found in input_tensors. "
                f"Available inputs: {list(input_tensors.keys())}"
            )

        shape = input_tensors[data_input].shape
        if shape is None:
            raise ValidationError(
                f"Pad node '{node_proto.name or 'unknown'}' cannot determine input rank: "
                f"shape is None for input '{data_input}'."
            )

        return len(shape)

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
        Pad converter with opset-based dispatch.

        - Opset v1: pads, mode, and value as attributes
        - Opset v2-v17: pads and constant_value as inputs, mode as attribute
        - Opset v18+: Adds axes input for selective axis padding
        """
        node_name = node_proto.name if node_proto.name else f"Pad_{node_index}"

        try:
            # Validate inputs
            cls._validate_inputs(node_proto, input_tensors)

            if opset == 1:
                # v1: pads, mode, and value as attributes
                pads = attrs.get("paddings", [])
                if not pads:
                    raise ValidationError(f"Pad node '{node_name}' (opset v1) requires 'paddings' attribute.")

                # Convert to list of integers
                try:
                    if isinstance(pads, (list, tuple)):
                        pads = [int(x) for x in pads]
                    elif hasattr(pads, "__iter__") and not isinstance(pads, str):
                        pads = [int(x) for x in pads]
                    else:
                        pads = [int(pads)]
                except (ValueError, TypeError) as e:
                    raise ValidationError(f"Pad node '{node_name}' paddings attribute contains non-integer values: {e}")

                # Validate pads format
                if len(pads) % 2 != 0:
                    raise ValidationError(f"Pad node '{node_name}' paddings must have even length (got {len(pads)}).")

                # Extract mode and value
                onnx_mode = attrs.get("mode", "constant")
                value = attrs.get("value", 0.0)
                axes = None
            elif opset < 18:
                # v2-v17: pads and constant_value as inputs, mode as attribute
                pads = cls._extract_and_validate_pads(node_proto, attrs, graph_proto, input_index=1)
                value = cls._extract_constant_value(node_proto, attrs, graph_proto, input_index=2)
                onnx_mode = attrs.get("mode", "constant")
                axes = None
            else:
                # v18+: Adds axes input for selective axis padding
                pads = cls._extract_and_validate_pads(node_proto, attrs, graph_proto, input_index=1)
                onnx_mode = attrs.get("mode", "constant")
                mode = onnx_mode_to_pytorch(onnx_mode)

                # Extract constant_value and axes
                value = cls._extract_constant_value(node_proto, attrs, graph_proto, input_index=2)
                if onnx_mode != "constant":
                    value = 0.0  # Not used for non-constant modes

                axes = cls._extract_axes(node_proto, graph_proto, input_index=3)

            # Validate mode
            cls._validate_mode_for_opset(onnx_mode, opset, node_name)
            mode = onnx_mode_to_pytorch(onnx_mode)

            # Get input rank
            input_rank = cls._get_input_rank(input_tensors, node_proto)

            # Validate axes if provided (v18+)
            if axes is not None:
                if len(axes) != len(set(axes)):
                    raise ValidationError(f"Pad node '{node_name}' axes input contains duplicate values: {axes}")
                for axis in axes:
                    normalized = axis if axis >= 0 else input_rank + axis
                    if normalized < 0 or normalized >= input_rank:
                        raise ValidationError(
                            f"Pad node '{node_name}' axis {axis} is out of range. "
                            f"Input rank: {input_rank}, valid range: [{-input_rank}, {input_rank - 1}]"
                        )

            # Convert ONNX pads format to PyTorch format
            pytorch_pads = convert_onnx_pads_to_pytorch(pads, axes=axes, input_rank=input_rank, mode=mode)

            # Build OrderedDict for inputs and outputs
            pad_input_dict, pad_output_dict = build_input_output_dicts(
                node_proto, input_tensors, output_tensors, input_names=[node_proto.input[0]]
            )

            return [
                PadNode.create(
                    name=node_name,
                    inputs=pad_input_dict,  # Only data input, other inputs are embedded
                    outputs=pad_output_dict,
                    pad=pytorch_pads,
                    mode=mode,
                    value=value,
                )
            ]
        except (ValidationError, ValueError) as e:
            handle_validation_error(node_proto, str(e), strict=True)
            return []
