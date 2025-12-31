# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Reduction operation converters.

This module provides converters for ONNX reduction operations:
- ReduceSum: Sum elements along specified axes
- ReduceMean: Average elements along specified axes
- ReduceMax: Maximum elements along specified axes

Key features:
- Handles opset version differences in keepdims default (v1: default=1, v13+: default=1)
- Supports axes as attribute (v1-v12) or input (v13+)
- Normalizes axes to PyTorch dim format (handles negative indices, removes duplicates)
- Converts to appropriate TIR reduction nodes
"""
from typing import List, Dict, Any, Union, Tuple, Optional, Type, Sequence
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.reduction import ReduceSumNode, ReduceMeanNode, ReduceMaxNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.validation import validate_constant_input
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


def convert_axes_to_dim(
    axes: Optional[Union[int, Sequence[int]]], rank: Optional[int] = None
) -> Optional[Union[int, Tuple[int, ...]]]:
    """
    Convert ONNX axes attribute to PyTorch dim format.
    Normalizes negative indices and removes duplicates.

    Args:
        axes: ONNX axes (can be None, int, list, or tuple)
        rank: Optional rank of the input tensor for normalizing negative indices

    Returns:
        PyTorch dim format: None, int, or tuple of ints (with duplicates removed)
    """
    if axes is None:
        return None

    # Convert to list for processing
    if isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    else:
        axes_list = [axes]

    if len(axes_list) == 0:
        return None

    # Normalize negative indices and remove duplicates
    seen = set()
    unique_axes = []
    for axis in axes_list:
        # Normalize negative index if rank is provided
        if rank is not None:
            normalized_axis = axis if axis >= 0 else rank + axis
        else:
            normalized_axis = axis

        # Only add if not already seen
        if normalized_axis not in seen:
            unique_axes.append(normalized_axis)
            seen.add(normalized_axis)
    axes_list = unique_axes

    if len(axes_list) == 0:
        return None
    elif len(axes_list) == 1:
        return axes_list[0]  # Single int for PyTorch
    else:
        return tuple(axes_list)  # Tuple for multiple dims


def extract_keepdims(attrs: Dict[str, Any]) -> bool:
    """
    Extract and convert keepdims attribute from ONNX format to bool.
    ONNX keepdims can be int (0/1) or bool, default is 1 (True).

    Args:
        attrs: ONNX node attributes

    Returns:
        Boolean keepdim value
    """
    return bool(attrs.get("keepdims", 1))


class BaseReduceConverter(OnnxOpConverter):
    """Base converter for ONNX reduction operations."""

    # Subclasses should override these
    NODE_CLASS: Optional[Type] = None
    OP_NAME: str = ""

    @classmethod
    def _create_reduce_node(
        cls,
        node_proto: NodeProto,
        node_name: str,
        data_input: str,
        output_name: str,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        dim: Optional[Union[int, Tuple[int, ...]]],
        keepdim: bool,
    ) -> List:
        """
        Create a reduction node using the appropriate node class.

        Args:
            node_name: Name for the node
            data_input: Input tensor name
            output_name: Output tensor name
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            dim: Dimension(s) to reduce (None, int, or tuple)
            keepdim: Whether to keep reduced dimensions

        Returns:
            List containing the created node
        """
        if cls.NODE_CLASS is None:
            raise NotImplementedError(f"{cls.__name__} must set NODE_CLASS")

        # Build OrderedDict for inputs and outputs
        input_dict, output_dict = build_input_output_dicts(
            node_proto, input_tensors, output_tensors, input_names=[data_input]
        )

        node = cls.NODE_CLASS.create(name=node_name, inputs=input_dict, outputs=output_dict, dim=dim, keepdim=keepdim)
        return [node]

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
        Base reduction converter - handles axes extraction and noop_with_empty_axes based on opset.

        - Opset v1-v12: axes as attribute
        - Opset v13+ (ReduceSum) or v18+ (ReduceMean/ReduceMax): axes as optional input tensor, noop_with_empty_axes attribute
        """
        if cls.OP_NAME == "":
            raise NotImplementedError(f"{cls.__name__} must set OP_NAME")

        node_name = node_proto.name if node_proto.name else f"{cls.OP_NAME}_{node_index}"

        # Extract axes based on opset and operation type
        # ReduceSum: v13+ uses input tensor
        # ReduceMean/ReduceMax: v18+ uses input tensor
        if (opset >= 13 and cls.OP_NAME == "ReduceSum") or (opset >= 18 and cls.OP_NAME in ["ReduceMean", "ReduceMax"]):
            # Extract noop_with_empty_axes attribute (default is 0/false)
            noop_with_empty_axes = bool(attrs.get("noop_with_empty_axes", 0))
            axes = cls._handle_axes_input_tensor(node_proto, attrs, graph_proto)

            # Handle noop_with_empty_axes: if True and axes is empty/None, return identity
            if noop_with_empty_axes and axes is None:
                from forge.transpiler.operations.other import IdentityNode

                data_input = node_proto.input[0]
                input_info = input_tensors[data_input]

                # Build OrderedDict for inputs and outputs
                input_dict, output_dict = build_input_output_dicts(
                    node_proto, input_tensors, output_tensors, input_names=[data_input]
                )

                return [IdentityNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]
        else:
            # v1-v12: axes as attribute
            axes = attrs.get("axes", None)

        keepdims = extract_keepdims(attrs)

        data_input = node_proto.input[0]
        input_info = input_tensors[data_input]
        # Get rank from input shape for normalizing negative indices
        rank = len(input_info.shape) if input_info.shape else None
        dim = convert_axes_to_dim(axes, rank=rank)

        # PyTorch reduction ops handle dim=None with keepdim=True correctly,
        # returning the expected shape (all dims as size 1), so no Reshape needed
        return cls._create_reduce_node(
            node_proto=node_proto,
            node_name=node_name,
            data_input=data_input,
            output_name=node_proto.output[0],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=dim,
            keepdim=keepdims,
        )

    @classmethod
    def _handle_axes_input_tensor(
        cls, node_proto: NodeProto, attrs: Dict[str, Any], graph_proto=None
    ) -> Optional[List[int]]:
        """
        Helper method to extract axes from optional input tensor (for opset 13+).

        Returns:
            List of axes (normalized) or None if not provided/empty
        """
        # Validate and extract axes from constant input (second input, optional)
        is_valid, axes, error_msg = validate_constant_input(node_proto, input_index=1, graph_proto=graph_proto)

        # Convert to list if it's a scalar or array
        if axes is not None:
            if isinstance(axes, (list, tuple)):
                axes = list(int(x) for x in axes)
            elif hasattr(axes, "__iter__") and not isinstance(axes, str):
                axes = list(int(x) for x in axes)
            else:
                axes = [int(axes)]

            # Empty axes list
            if len(axes) == 0:
                axes = None

        # Fallback to attribute if not provided as input (for backward compatibility)
        if axes is None:
            axes = attrs.get("axes", None)
            if axes is not None and isinstance(axes, (list, tuple)) and len(axes) == 0:
                axes = None

        return axes


class ReduceSumConverter(BaseReduceConverter):
    """Converter for ONNX ReduceSum operation."""

    NODE_CLASS = ReduceSumNode
    OP_NAME = "ReduceSum"


class ReduceMeanConverter(BaseReduceConverter):
    """Converter for ONNX ReduceMean operation."""

    NODE_CLASS = ReduceMeanNode
    OP_NAME = "ReduceMean"

    @classmethod
    def _create_reduce_node(
        cls,
        node_proto: NodeProto,
        node_name: str,
        data_input: str,
        output_name: str,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        dim: Optional[Union[int, Tuple[int, ...]]],
        keepdim: bool,
    ) -> List:
        """
        Create a ReduceMean node with Cast nodes for integer types.

        torch.mean() requires floating point types. For integer inputs,
        we insert Cast nodes before (to float) and after (back to original dtype).
        """
        from forge.transpiler.operations.other import CastNode
        from forge.transpiler.core.types import onnx_dtype_to_torch_dtype
        import onnx

        input_info = input_tensors[data_input]
        input_dtype = input_info.onnx_dtype

        # Check if input is integer type
        is_integer = input_dtype in [
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
            onnx.TensorProto.INT8,
            onnx.TensorProto.UINT8,
            onnx.TensorProto.UINT32,
            onnx.TensorProto.UINT64,
        ]

        nodes = []
        current_input = data_input
        current_input_tensors = input_tensors

        # If integer type, insert Cast to float before ReduceMean
        if is_integer:
            cast_to_float_name = f"{node_name}_cast_to_float"
            cast_to_float_output = f"{cast_to_float_name}_output"
            # Use onnx_dtype_to_torch_dtype to get float32 dtype
            float_dtype = onnx_dtype_to_torch_dtype(onnx.TensorProto.FLOAT)

            # Create intermediate tensor info for float cast output
            float_output_info = TensorInfo(
                name=cast_to_float_output, shape=input_info.shape, onnx_dtype=onnx.TensorProto.FLOAT
            )
            cast_to_float_tensors = {cast_to_float_output: float_output_info}

            # Build OrderedDict for Cast node (using intermediate tensors)
            cast_input_dict, cast_output_dict = build_input_output_dicts(
                node_proto,
                current_input_tensors,
                cast_to_float_tensors,
                input_names=[current_input],
                output_names=[cast_to_float_output],
            )

            cast_to_float_node = CastNode.create(
                name=cast_to_float_name, inputs=cast_input_dict, outputs=cast_output_dict, dtype=float_dtype
            )
            nodes.append(cast_to_float_node)

            # Update for ReduceMean node
            current_input = cast_to_float_output
            current_input_tensors = {cast_to_float_output: float_output_info}

        # Create ReduceMean node
        reduce_output_name = output_name if not is_integer else f"{node_name}_float_output"
        reduce_output_tensors = (
            output_tensors
            if not is_integer
            else {
                reduce_output_name: TensorInfo(
                    name=reduce_output_name, shape=output_tensors[output_name].shape, onnx_dtype=onnx.TensorProto.FLOAT
                )
            }
        )

        # Build OrderedDict for ReduceMean node (using intermediate tensors)
        reduce_input_dict, reduce_output_dict = build_input_output_dicts(
            node_proto,
            current_input_tensors,
            reduce_output_tensors,
            input_names=[current_input],
            output_names=[reduce_output_name],
        )

        reduce_node = cls.NODE_CLASS.create(
            name=node_name, inputs=reduce_input_dict, outputs=reduce_output_dict, dim=dim, keepdim=keepdim
        )
        nodes.append(reduce_node)

        # If integer type, insert Cast back to original dtype after ReduceMean
        if is_integer:
            cast_back_name = f"{node_name}_cast_back"
            reduce_output = f"{node_name}_float_output"
            original_torch_dtype = onnx_dtype_to_torch_dtype(input_dtype)

            # Build OrderedDict for Cast back node
            cast_back_input_tensors = OrderedDict()
            cast_back_input_tensors[reduce_output] = TensorInfo(
                name=reduce_output, shape=output_tensors[output_name].shape, onnx_dtype=onnx.TensorProto.FLOAT
            )

            cast_back_input_dict, cast_back_output_dict = build_input_output_dicts(
                node_proto,
                cast_back_input_tensors,
                output_tensors,
                input_names=[reduce_output],
                output_names=[output_name],
            )

            cast_back_node = CastNode.create(
                name=cast_back_name,
                inputs=cast_back_input_dict,
                outputs=cast_back_output_dict,
                dtype=original_torch_dtype,
            )
            nodes.append(cast_back_node)

        return nodes


class ReduceMaxConverter(BaseReduceConverter):
    """Converter for ONNX ReduceMax operation."""

    NODE_CLASS = ReduceMaxNode
    OP_NAME = "ReduceMax"
