# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Conv operation converter with opset version support.

This module provides the converter for ONNX Conv (convolution) operations, supporting
1D, 2D, and 3D convolutions. The converter handles opset version differences and
converts ONNX convolution attributes to PyTorch-compatible format.

Key features:
- Supports Conv1d, Conv2d, Conv3d based on input dimensions
- Handles auto_pad modes (SAME_UPPER, SAME_LOWER, VALID)
- Converts ONNX padding format to PyTorch format
- Supports groups (grouped convolutions)
- Handles dilation and stride attributes

Based on onnx2pytorch's convert_layer() approach.
"""
from typing import List, Dict, Any, Union, Tuple
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.conv import Conv1dNode, Conv2dNode, Conv3dNode
from forge.transpiler.operations.other import PadNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


class ConvConverter(OnnxOpConverter):
    """
    Converter for ONNX Conv operation with opset version support.

    Supports opset versions: 1, 11, 22
    """

    @staticmethod
    def _compute_autopad_padding(
        input_size: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        mode: str = "SAME_UPPER",
        opset_version: int = 11,
    ) -> Tuple[int, int]:
        """
        Compute padding values for Conv auto_pad modes.

        Note: Conv operations always use ceil behavior for auto_pad modes.

        Args:
            input_size: Input size in this dimension
            kernel_size: Kernel size in this dimension
            stride: Stride in this dimension
            dilation: Dilation in this dimension
            mode: "SAME_UPPER", "SAME_LOWER", or "VALID"
            opset_version: ONNX opset version (11 for v1-v18, 19 for v19+)

        Returns:
            Tuple of (pad_before, pad_after) for this dimension
        """
        if mode == "VALID":
            return (0, 0)

        # Calculate effective kernel size with dilation
        effective_kernel = (kernel_size - 1) * dilation + 1

        # Conv always uses ceil(input / stride) for auto_pad modes
        output_size = (input_size + stride - 1) // stride  # ceil(input / stride)

        # Total padding needed to achieve the desired output size
        total_pad = max(0, (output_size - 1) * stride + effective_kernel - input_size)

        if mode == "SAME_UPPER":
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
        else:  # SAME_LOWER
            pad_after = total_pad // 2
            pad_before = total_pad - pad_after

        return (pad_before, pad_after)

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
        Conv converter - single method handles all versions using opset parameter.

        - Opset v1-v10: groups as attribute (default 1)
        - Opset v11+: groups as attribute (same as v1, but with better validation)
        - Opset v22+: Extended type support (functionally same as v11)
        """
        # Determine opset_version for internal implementation
        if opset < 11:
            opset_version = 11  # Use v11 logic for v1-v10
        elif opset < 22:
            opset_version = 11
        else:
            opset_version = 19  # Use v19 logic for v22+
        return cls._convert_conv_impl(
            node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset_version=opset_version
        )

    @staticmethod
    def _normalize_to_tuple(
        value: Union[int, List[int], Tuple[int, ...]], ndim: int, default: int = 1
    ) -> Tuple[int, ...]:
        """
        Normalize stride/dilation to tuple format for given number of dimensions.

        Handles various input formats:
        - int: Broadcast to all dimensions
        - Single-element list/tuple: Broadcast to all dimensions
        - Multi-element list/tuple: Use first ndim elements, or pad with last element

        Args:
            value: Input value (int, list, or tuple)
            ndim: Number of dimensions (1, 2, or 3 for Conv1d/2d/3d)
            default: Default value if input is invalid

        Returns:
            Tuple of length ndim with normalized values
        """
        if isinstance(value, int):
            # Single value: broadcast to all dimensions
            return (value,) * ndim
        elif isinstance(value, (list, tuple)):
            if len(value) == 1:
                # Single-element list: broadcast to all dimensions
                return (value[0],) * ndim
            elif len(value) >= ndim:
                # Enough elements: take first ndim
                return tuple(value[:ndim])
            else:
                # Not enough elements: pad with last element
                return tuple(value) + (value[-1],) * (ndim - len(value))
        else:
            # Invalid input: use default
            return (default,) * ndim

    @staticmethod
    def _convert_onnx_pads_to_pytorch(onnx_pads: List[int], ndim: int) -> Union[int, Tuple[int, ...]]:
        """
        Convert ONNX pads format to PyTorch padding format.

        ONNX format: [pad_dim0_begin, pad_dim1_begin, ..., pad_dim0_end, pad_dim1_end, ...]
        PyTorch format:
        - int: Same padding on all sides (symmetric)
        - (int, int): (H, W) for 2D, (W,) for 1D
        - (int, int, int): (D, H, W) for 3D
        - (int, int, int, int, int, int): Asymmetric padding (W_beg, W_end, H_beg, H_end, D_beg, D_end)

        Args:
            onnx_pads: ONNX padding list [begin_dim0, begin_dim1, ..., end_dim0, end_dim1, ...]
            ndim: Number of spatial dimensions (1, 2, or 3)

        Returns:
            PyTorch-compatible padding format (int or tuple)

        Raises:
            ValueError: If padding list length doesn't match 2 * ndim
        """
        if not onnx_pads or len(onnx_pads) != 2 * ndim:
            return 0

        if ndim == 1:
            # Conv1d: [W_begin, W_end]
            pad_W_begin, pad_W_end = onnx_pads
            # Symmetric: return int, asymmetric: return tuple
            return pad_W_begin if pad_W_begin == pad_W_end else (pad_W_begin, pad_W_end)

        elif ndim == 2:
            # Conv2d: [H_begin, W_begin, H_end, W_end]
            pad_H_begin, pad_W_begin, pad_H_end, pad_W_end = onnx_pads
            # All equal: return single int
            if pad_H_begin == pad_H_end == pad_W_begin == pad_W_end:
                return pad_H_begin
            # H and W symmetric separately: return (H, W)
            if pad_H_begin == pad_H_end and pad_W_begin == pad_W_end:
                return (pad_H_begin, pad_W_begin)
            # Asymmetric: PyTorch format (W_beg, W_end, H_beg, H_end)
            return (pad_W_begin, pad_W_end, pad_H_begin, pad_H_end)

        elif ndim == 3:
            # Conv3d: [D_begin, H_begin, W_begin, D_end, H_end, W_end]
            pad_D_begin, pad_H_begin, pad_W_begin, pad_D_end, pad_H_end, pad_W_end = onnx_pads
            # All equal: return single int
            if pad_D_begin == pad_D_end == pad_H_begin == pad_H_end == pad_W_begin == pad_W_end:
                return pad_D_begin
            # D, H, W symmetric separately: return (D, H, W)
            if pad_D_begin == pad_D_end and pad_H_begin == pad_H_end and pad_W_begin == pad_W_end:
                return (pad_D_begin, pad_H_begin, pad_W_begin)
            # Asymmetric: PyTorch format (W_beg, W_end, H_beg, H_end, D_beg, D_end)
            return (pad_W_begin, pad_W_end, pad_H_begin, pad_H_end, pad_D_begin, pad_D_end)

        else:
            raise ValueError(f"Unsupported number of dimensions for padding conversion: {ndim}")

    @classmethod
    def _create_pad_node(
        cls,
        node_proto: NodeProto,
        node_name: str,
        input_name: str,
        input_tensor: TensorInfo,
        pad_list: List[int],
        kernel_dims: int,
    ) -> Tuple[PadNode, str, Dict[str, TensorInfo]]:
        """
        Create a PadNode for the given padding.

        Returns:
            Tuple of (pad_node, pad_output_name, pad_output_tensors)
        """
        pad_output = f"{node_name}_padded"
        pad_output_tensors = {pad_output: input_tensor}

        # Build OrderedDict for Pad node
        pad_input_dict, pad_output_dict = build_input_output_dicts(
            node_proto,
            {input_name: input_tensor},
            pad_output_tensors,
            input_names=[input_name],
            output_names=[pad_output],
        )

        pad_node = PadNode.create(
            name=f"{node_name}_pad",
            inputs=pad_input_dict,
            outputs=pad_output_dict,
            pad=tuple(pad_list),
            mode="constant",
            value=0.0,
        )

        return pad_node, pad_output, pad_output_tensors

    @classmethod
    def _handle_auto_pad(
        cls,
        node_proto: NodeProto,
        node_name: str,
        input_name: str,
        input_tensor: TensorInfo,
        input_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, ...],
        stride_tuple: Tuple[int, ...],
        dilation_tuple: Tuple[int, ...],
        auto_pad: str,
        kernel_dims: int,
        opset_version: int = 11,
    ) -> Tuple[PadNode, str, Dict[str, TensorInfo]]:
        """
        Handle auto_pad by creating a PadNode with computed padding.

        Args:
            opset_version: ONNX opset version (11 for v1-v18, 19 for v19+)
                          Used to determine the correct auto_pad formula.
        """
        spatial_dims = input_shape[2:]  # Skip batch and channel

        # Compute padding for each spatial dimension
        # Conv always uses ceil behavior for auto_pad modes
        pads = []
        for in_size, k_size, s, d in zip(spatial_dims, kernel_shape, stride_tuple, dilation_tuple):
            pad_before, pad_after = cls._compute_autopad_padding(in_size, k_size, s, d, auto_pad, opset_version)
            pads.extend([pad_before, pad_after])

        # Convert to PyTorch F.pad format (reverse order: last dim first)
        # For 1D: [pad_W_begin, pad_W_end] -> [padLeft, padRight]
        # For 2D: [pad_H_begin, pad_W_begin, pad_H_end, pad_W_end] -> [padLeft, padRight, padTop, padBottom]
        # For 3D: [pad_D_begin, pad_H_begin, pad_W_begin, pad_D_end, pad_H_end, pad_W_end] ->
        #         [padLeft, padRight, padTop, padBottom, padFront, padBack]
        pad_list = []
        for i in range(kernel_dims - 1, -1, -1):
            idx_begin = i * 2
            idx_end = idx_begin + 1
            pad_list.extend([pads[idx_begin], pads[idx_end]])

        return cls._create_pad_node(node_proto, node_name, input_name, input_tensor, pad_list, kernel_dims)

    @classmethod
    def _handle_asymmetric_padding(
        cls,
        node_proto: NodeProto,
        node_name: str,
        input_name: str,
        input_tensor: TensorInfo,
        conv_padding: Union[Tuple, List],
        kernel_dims: int,
    ) -> Tuple[PadNode, str, Dict[str, TensorInfo]]:
        """Handle asymmetric padding by creating a PadNode."""
        # PyTorch F.pad format is already correct from _convert_onnx_pads_to_pytorch
        pad_list = list(conv_padding)
        return cls._create_pad_node(node_proto, node_name, input_name, input_tensor, pad_list, kernel_dims)

    @classmethod
    def _convert_conv_impl(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset_version: int = 11,
    ) -> List:
        """
        Common implementation for Conv conversion. Supports Conv1d, Conv2d, and Conv3d.

        Args:
            opset_version: ONNX opset version (11 for v1-v18, 19 for v19+)
                          Used to determine the correct auto_pad formula.
        """
        nodes = []
        node_name = node_proto.name if node_proto.name else f"Conv_{node_index}"

        # Get kernel shape to determine conv dimension
        kernel_shape = attrs.get("kernel_shape", None)
        if kernel_shape is None:
            weight_name = node_proto.input[1] if len(node_proto.input) > 1 else None
            if weight_name and weight_name in input_tensors:
                weight_shape = input_tensors[weight_name].shape
                if weight_shape and len(weight_shape) >= 2:
                    kernel_shape = weight_shape[2:]
                else:
                    raise ValueError(f"Cannot infer kernel_shape for Conv {node_name} from weight shape {weight_shape}")
            else:
                raise ValueError(f"Cannot infer kernel_shape for Conv {node_name}: weight tensor not found")

        # Normalize kernel_shape to tuple
        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape,)
        elif isinstance(kernel_shape, (list, tuple)):
            kernel_shape = tuple(kernel_shape)
        else:
            raise ValueError(f"Invalid kernel_shape type for Conv {node_name}: {type(kernel_shape)}")

        kernel_dims = len(kernel_shape)
        if kernel_dims not in (1, 2, 3):
            raise ValueError(
                f"Unsupported Conv dimension: {kernel_dims}D (kernel_shape={kernel_shape}). "
                f"Only Conv1d (1D kernel), Conv2d (2D kernel), and Conv3d (3D kernel) are supported."
            )

        # Initialize conv inputs and padding
        conv_inputs = list(node_proto.input)
        conv_input_tensors = input_tensors.copy()
        conv_padding = 0

        # Handle AUTO_PAD
        auto_pad = attrs.get("auto_pad", "NOTSET")
        if auto_pad in ("SAME_UPPER", "SAME_LOWER", "VALID"):
            input_shape = input_tensors[conv_inputs[0]].shape
            if input_shape is None:
                raise ValueError(f"Cannot compute auto_pad for Conv {node_name} with unknown input shape")

            stride_tuple = cls._normalize_to_tuple(attrs.get("strides", 1), kernel_dims, default=1)
            dilation_tuple = cls._normalize_to_tuple(attrs.get("dilations", 1), kernel_dims, default=1)

            pad_node, pad_output, pad_output_tensors = cls._handle_auto_pad(
                node_proto,
                node_name,
                conv_inputs[0],
                input_tensors[conv_inputs[0]],
                input_shape,
                kernel_shape,
                stride_tuple,
                dilation_tuple,
                auto_pad,
                kernel_dims,
                opset_version,
            )
            nodes.append(pad_node)

            # Update conv inputs to use padded output
            conv_inputs = [pad_output]
            conv_input_tensors = {pad_output: pad_output_tensors[pad_output]}

            # Add weight and bias inputs
            for i in range(1, len(node_proto.input)):
                input_name = node_proto.input[i]
                if input_name in input_tensors:
                    conv_inputs.append(input_name)
                    conv_input_tensors[input_name] = input_tensors[input_name]

            conv_padding = 0

        # Handle explicit padding
        else:  # auto_pad == 'NOTSET'
            onnx_pads = attrs.get("pads", [0] * (2 * kernel_dims))
            if isinstance(onnx_pads, (list, tuple)) and len(onnx_pads) == 2 * kernel_dims:
                conv_padding = cls._convert_onnx_pads_to_pytorch(list(onnx_pads), kernel_dims)

                # Check if asymmetric padding needs PadNode (Conv1d: tuple of 2, Conv3d: tuple of 6)
                needs_pad_node = (
                    kernel_dims == 1 and isinstance(conv_padding, (tuple, list)) and len(conv_padding) == 2
                ) or (kernel_dims == 3 and isinstance(conv_padding, (tuple, list)) and len(conv_padding) == 6)

                if needs_pad_node:
                    pad_node, pad_output, pad_output_tensors = cls._handle_asymmetric_padding(
                        node_proto, node_name, conv_inputs[0], input_tensors[conv_inputs[0]], conv_padding, kernel_dims
                    )
                    nodes.append(pad_node)

                    # Update conv inputs to use padded output
                    conv_inputs = [pad_output]
                    conv_input_tensors = {pad_output: pad_output_tensors[pad_output]}

                    # Add weight and bias inputs
                    for i in range(1, len(node_proto.input)):
                        input_name = node_proto.input[i]
                        if input_name in input_tensors:
                            conv_inputs.append(input_name)
                            conv_input_tensors[input_name] = input_tensors[input_name]

                    conv_padding = 0

        # Extract and normalize attributes
        stride = cls._normalize_to_tuple(attrs.get("strides", 1), kernel_dims, default=1)
        dilation = cls._normalize_to_tuple(attrs.get("dilations", 1), kernel_dims, default=1)
        groups = attrs.get("group", 1)

        # Build OrderedDict for inputs and outputs
        conv_input_dict, conv_output_dict = build_input_output_dicts(
            node_proto, conv_input_tensors, output_tensors, input_names=conv_inputs
        )

        # Create appropriate Conv node
        if kernel_dims == 1:
            conv_node = Conv1dNode.create(
                name=node_name,
                inputs=conv_input_dict,
                outputs=conv_output_dict,
                stride=stride[0] if len(stride) == 1 else stride,
                padding=conv_padding,
                dilation=dilation[0] if len(dilation) == 1 else dilation,
                groups=groups,
            )
        elif kernel_dims == 2:
            conv_node = Conv2dNode.create(
                name=node_name,
                inputs=conv_input_dict,
                outputs=conv_output_dict,
                stride=stride,
                padding=conv_padding,
                dilation=dilation,
                groups=groups,
            )
        else:  # kernel_dims == 3
            conv_node = Conv3dNode.create(
                name=node_name,
                inputs=conv_input_dict,
                outputs=conv_output_dict,
                stride=stride,
                padding=conv_padding,
                dilation=dilation,
                groups=groups,
            )

        nodes.append(conv_node)
        return nodes
