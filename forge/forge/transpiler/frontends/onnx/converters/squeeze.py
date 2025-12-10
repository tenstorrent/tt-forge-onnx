"""
ONNX Squeeze operation converter with opset version support.
"""
from typing import List, Dict, Any, Optional
from onnx import NodeProto
from forge.transpiler.ir.types import TensorInfo
from forge.transpiler.ir.operations.shape import SqueezeNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.validation import validate_constant_input


class SqueezeConverter(OnnxOpConverter):
    """Converter for ONNX Squeeze operation with opset version support."""
    
    @classmethod
    def _normalize_axes(cls, axes: Any, input_rank: int) -> List[int]:
        """Normalize axes to positive integers, handling negative indices."""
        if axes is None:
            return []
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        return [idx + input_rank if idx < 0 else idx for idx in map(int, axes)]
    
    @classmethod
    def _get_axes_to_squeeze(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                             attrs: Dict[str, Any], graph_proto=None, opset_version: int = 1) -> Optional[List[int]]:
        """Extract and normalize axes from attribute (v1-v11) or input tensor (v13+).
        
        Returns:
            - None: axes not provided, should auto-detect all size-1 dims
            - []: axes explicitly empty, should do no squeeze (Identity)
            - List[int]: normalized axes to squeeze
        """
        input_info = list(input_tensors.values())[0]
        input_rank = len(input_info.shape) if input_info.shape else None
        
        if input_rank is None:
            raise ValueError(f"Cannot determine input rank for Squeeze node '{node_proto.name}'")
        
        if opset_version >= 13 and len(node_proto.input) > 1:
            is_valid, axes, _ = validate_constant_input(node_proto, input_index=1, graph_proto=graph_proto)
            if is_valid:
                if axes is not None:
                    # Normalize and return (empty list means no squeeze)
                    return cls._normalize_axes(axes, input_rank)
                # axes input exists but is None -> not provided, auto-detect
                return None
        
        axes = attrs.get('axes')
        if axes is not None:
            # Normalize and return (empty list means no squeeze)
            return cls._normalize_axes(axes, input_rank)
        # axes not provided -> auto-detect all size-1 dims
        return None
    
    @classmethod
    def _process_squeeze(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                        output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                        node_index: int, graph_proto=None, opset_version: int = 1) -> List:
        """Common processing logic for all opset versions."""
        from forge.transpiler.ir.operations.other import IdentityNode
        
        input_info = list(input_tensors.values())[0]
        input_shape = input_info.shape
        if input_shape is None:
            raise ValueError(f"Cannot determine input shape for Squeeze node '{node_proto.name}'")
        
        node_name = node_proto.name or f"Squeeze_{node_index}"
        
        # Get and process axes
        axes = cls._get_axes_to_squeeze(node_proto, input_tensors, attrs, graph_proto, opset_version)
        # None means auto-detect all size-1 dims, [] means no squeeze (Identity)
        if axes is None:
            axes = [i for i, size in enumerate(input_shape) if size == 1]
        
        if axes:
            # Validate and deduplicate
            for axis in axes:
                if not (0 <= axis < len(input_shape) and input_shape[axis] == 1):
                    raise ValueError(
                        f"Squeeze node '{node_name}': cannot squeeze axis {axis} "
                        f"(size={input_shape[axis] if 0 <= axis < len(input_shape) else 'out of range'})"
                    )
            axes = sorted(set(axes), reverse=True)
        
        # Prepare inputs (v13+ uses only data input)
        if opset_version >= 13:
            data_input = node_proto.input[0]
            inputs = [data_input]
            input_tensors_dict = {data_input: input_tensors[data_input]}
        else:
            inputs = list(node_proto.input)
            input_tensors_dict = input_tensors
        
        # Create Identity if no squeeze needed, otherwise SqueezeNode
        if not axes:
            return [IdentityNode.create(
                name=node_name,
                inputs=inputs,
                outputs=[node_proto.output[0]],
                input_tensors=input_tensors_dict,
                output_tensors=output_tensors
            )]
        
        dim_value = tuple(axes) if len(axes) > 1 else axes[0]
        return [SqueezeNode.create(
            name=node_name,
            inputs=inputs,
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors_dict,
            output_tensors=output_tensors,
            dim=dim_value
        )]
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Squeeze opset v1-v11: axes as attribute."""
        return cls._process_squeeze(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, 1)
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """Squeeze opset v11: Same as v1 but supports negative indices."""
        return cls._impl_v1(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _impl_v13(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """Squeeze opset v13+: axes as optional input tensor."""
        return cls._process_squeeze(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, 13)
    
    # All versions v13+ use the same implementation
    _impl_v21 = _impl_v23 = _impl_v24 = _impl_v25 = _impl_v13

