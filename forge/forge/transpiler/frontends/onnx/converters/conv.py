"""
ONNX Conv operation converter with opset version support.
Based on onnx2pytorch's convert_layer() approach.
"""
from typing import List, Dict, Any, Union, Tuple
from onnx import NodeProto
from forge.transpiler.ir.types import TensorInfo
from forge.transpiler.ir.operations.conv import Conv1dNode, Conv2dNode, Conv3dNode
from forge.transpiler.ir.operations.other import PadNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from loguru import logger


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
        opset_version: int = 11
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
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Conv opset v1-v10: groups as attribute (default 1)."""
        return cls._convert_conv_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset_version=11)
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """Conv opset v11+: groups as attribute (same as v1, but with better validation)."""
        return cls._convert_conv_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset_version=11)
    
    @classmethod
    def _impl_v22(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """Conv opset v22+: Extended type support. Functionally same as v11 for our purposes."""
        return cls._convert_conv_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset_version=19)
    
    @staticmethod
    def _normalize_to_tuple(value: Union[int, List[int], Tuple[int, ...]], 
                           ndim: int, default: int = 1) -> Tuple[int, ...]:
        """Normalize stride/dilation to tuple format for given number of dimensions."""
        if isinstance(value, int):
            return (value,) * ndim
        elif isinstance(value, (list, tuple)):
            if len(value) == 1:
                return (value[0],) * ndim
            elif len(value) >= ndim:
                return tuple(value[:ndim])
            else:
                return tuple(value) + (value[-1],) * (ndim - len(value))
        else:
            return (default,) * ndim
    
    @staticmethod
    def _convert_onnx_pads_to_pytorch(onnx_pads: List[int], ndim: int) -> Union[int, Tuple[int, ...]]:
        """
        Convert ONNX pads format to PyTorch padding format.
        
        ONNX format: [pad_dim0_begin, pad_dim1_begin, ..., pad_dim0_end, pad_dim1_end, ...]
        PyTorch format: int, (int,), (int, int), (int, int, int), or (int, int, int, int, int, int)
        """
        if not onnx_pads or len(onnx_pads) != 2 * ndim:
            return 0
        
        if ndim == 1:
            pad_W_begin, pad_W_end = onnx_pads
            return pad_W_begin if pad_W_begin == pad_W_end else (pad_W_begin, pad_W_end)
        
        elif ndim == 2:
            pad_H_begin, pad_W_begin, pad_H_end, pad_W_end = onnx_pads
            if pad_H_begin == pad_H_end == pad_W_begin == pad_W_end:
                return pad_H_begin
            if pad_H_begin == pad_H_end and pad_W_begin == pad_W_end:
                return (pad_H_begin, pad_W_begin)
            # Asymmetric: (padLeft, padRight, padTop, padBottom)
            return (pad_W_begin, pad_W_end, pad_H_begin, pad_H_end)
        
        elif ndim == 3:
            pad_D_begin, pad_H_begin, pad_W_begin, pad_D_end, pad_H_end, pad_W_end = onnx_pads
            if pad_D_begin == pad_D_end == pad_H_begin == pad_H_end == pad_W_begin == pad_W_end:
                return pad_D_begin
            if pad_D_begin == pad_D_end and pad_H_begin == pad_H_end and pad_W_begin == pad_W_end:
                return (pad_D_begin, pad_H_begin, pad_W_begin)
            # Asymmetric: (padLeft, padRight, padTop, padBottom, padFront, padBack)
            return (pad_W_begin, pad_W_end, pad_H_begin, pad_H_end, pad_D_begin, pad_D_end)
        
        else:
            raise ValueError(f"Unsupported number of dimensions for padding conversion: {ndim}")
    
    @classmethod
    def _create_pad_node(cls, node_name: str, input_name: str, input_tensor: TensorInfo,
                        pad_list: List[int], kernel_dims: int) -> Tuple[PadNode, str, Dict[str, TensorInfo]]:
        """
        Create a PadNode for the given padding.
        
        Returns:
            Tuple of (pad_node, pad_output_name, pad_output_tensors)
        """
        pad_output = f"{node_name}_padded"
        pad_output_tensors = {pad_output: input_tensor}
        
        pad_node = PadNode.create(
            name=f"{node_name}_pad",
            inputs=[input_name],
            outputs=[pad_output],
            input_tensors={input_name: input_tensor},
            output_tensors=pad_output_tensors,
            pad=tuple(pad_list),
            mode='constant',
            value=0.0
        )
        
        return pad_node, pad_output, pad_output_tensors
    
    @classmethod
    def _handle_auto_pad(cls, node_name: str, input_name: str, input_tensor: TensorInfo,
                         input_shape: Tuple[int, ...], kernel_shape: Tuple[int, ...],
                         stride_tuple: Tuple[int, ...], dilation_tuple: Tuple[int, ...],
                         auto_pad: str, kernel_dims: int, opset_version: int = 11) -> Tuple[PadNode, str, Dict[str, TensorInfo]]:
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
        
        return cls._create_pad_node(node_name, input_name, input_tensor, pad_list, kernel_dims)
    
    @classmethod
    def _handle_asymmetric_padding(cls, node_name: str, input_name: str, input_tensor: TensorInfo,
                                   conv_padding: Union[Tuple, List], kernel_dims: int) -> Tuple[PadNode, str, Dict[str, TensorInfo]]:
        """Handle asymmetric padding by creating a PadNode."""
        # PyTorch F.pad format is already correct from _convert_onnx_pads_to_pytorch
        pad_list = list(conv_padding)
        return cls._create_pad_node(node_name, input_name, input_tensor, pad_list, kernel_dims)
    
    @classmethod
    def _convert_conv_impl(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                          output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                          node_index: int, graph_proto=None, opset_version: int = 11) -> List:
        """
        Common implementation for Conv conversion. Supports Conv1d, Conv2d, and Conv3d.
        
        Args:
            opset_version: ONNX opset version (11 for v1-v18, 19 for v19+)
                          Used to determine the correct auto_pad formula.
        """
        nodes = []
        node_name = node_proto.name if node_proto.name else f"Conv_{node_index}"
        
        # Get kernel shape to determine conv dimension
        kernel_shape = attrs.get('kernel_shape', None)
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
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
            input_shape = input_tensors[conv_inputs[0]].shape
            if input_shape is None:
                raise ValueError(f"Cannot compute auto_pad for Conv {node_name} with unknown input shape")
            
            stride_tuple = cls._normalize_to_tuple(attrs.get('strides', 1), kernel_dims, default=1)
            dilation_tuple = cls._normalize_to_tuple(attrs.get('dilations', 1), kernel_dims, default=1)
            
            pad_node, pad_output, pad_output_tensors = cls._handle_auto_pad(
                node_name, conv_inputs[0], input_tensors[conv_inputs[0]],
                input_shape, kernel_shape, stride_tuple, dilation_tuple, auto_pad, kernel_dims, opset_version
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
            onnx_pads = attrs.get('pads', [0] * (2 * kernel_dims))
            if isinstance(onnx_pads, (list, tuple)) and len(onnx_pads) == 2 * kernel_dims:
                conv_padding = cls._convert_onnx_pads_to_pytorch(list(onnx_pads), kernel_dims)
                
                # Check if asymmetric padding needs PadNode (Conv1d: tuple of 2, Conv3d: tuple of 6)
                needs_pad_node = (
                    (kernel_dims == 1 and isinstance(conv_padding, (tuple, list)) and len(conv_padding) == 2) or
                    (kernel_dims == 3 and isinstance(conv_padding, (tuple, list)) and len(conv_padding) == 6)
                )
                
                if needs_pad_node:
                    pad_node, pad_output, pad_output_tensors = cls._handle_asymmetric_padding(
                        node_name, conv_inputs[0], input_tensors[conv_inputs[0]], conv_padding, kernel_dims
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
        stride = cls._normalize_to_tuple(attrs.get('strides', 1), kernel_dims, default=1)
        dilation = cls._normalize_to_tuple(attrs.get('dilations', 1), kernel_dims, default=1)
        groups = attrs.get('group', 1)
        
        # Create appropriate Conv node
        conv_output_tensors = output_tensors.copy()
        if kernel_dims == 1:
            conv_node = Conv1dNode.create(
                name=node_name,
                inputs=conv_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=conv_input_tensors,
                output_tensors=conv_output_tensors,
                stride=stride[0] if len(stride) == 1 else stride,
                padding=conv_padding,
                dilation=dilation[0] if len(dilation) == 1 else dilation,
                groups=groups
            )
        elif kernel_dims == 2:
            conv_node = Conv2dNode.create(
                name=node_name,
                inputs=conv_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=conv_input_tensors,
                output_tensors=conv_output_tensors,
                stride=stride,
                padding=conv_padding,
                dilation=dilation,
                groups=groups
            )
        else:  # kernel_dims == 3
            conv_node = Conv3dNode.create(
                name=node_name,
                inputs=conv_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=conv_input_tensors,
                output_tensors=conv_output_tensors,
                stride=stride,
                padding=conv_padding,
                dilation=dilation,
                groups=groups
            )
        
        nodes.append(conv_node)
        return nodes
