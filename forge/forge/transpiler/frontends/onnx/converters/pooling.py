"""
ONNX Pooling operation converters (MaxPool, AveragePool, GlobalAveragePool).
"""
from typing import List, Dict, Any, Union, Tuple
from onnx import NodeProto
from loguru import logger
from forge.transpiler.ir.types import TensorInfo
from forge.transpiler.ir.operations.pooling import (
    MaxPool1dNode, MaxPool2dNode, MaxPool3dNode,
    AveragePool1dNode, AveragePool2dNode, AveragePool3dNode,
    GlobalAveragePoolNode
)
from forge.transpiler.ir.operations.other import PadNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter


class MaxPoolConverter(OnnxOpConverter):
    """
    Converter for ONNX MaxPool operation.
    
    Supports opset versions: 1, 8, 10, 11, 12, 22
    Supports dimensions: MaxPool1d, MaxPool2d, MaxPool3d
    
    Version differences:
    - v1: Basic attributes (auto_pad, kernel_shape, pads, strides), single output
    - v8: Added optional Indices output, added storage_order attribute
    - v10: Added ceil_mode attribute, added dilations attribute
    - v11: Improved formulas with explicit ceil_mode handling for auto_pad
    - v12: Extended type support (int8, uint8)
    - v22: Extended type support (bfloat16), auto_pad respects ceil_mode
    
    Key differences from ONNX to PyTorch:
    - dilations: PyTorch MaxPool DOES support dilation (unlike AvgPool)
    - Indices output: ONNX v8+ optional output, PyTorch has return_indices parameter
    - storage_order: ONNX v8+ attribute for Indices computation (not used if Indices not requested)
    """
    
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
    def _compute_autopad_padding(
        input_size: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        mode: str = "SAME_UPPER",
        ceil_mode: bool = False,
        opset_version: int = 11
    ) -> Tuple[int, int]:
        """
        Compute padding values for MaxPool auto_pad modes.
        
        According to ONNX spec:
        - v1-v11: auto_pad always uses ceil behavior, regardless of ceil_mode
        - v22: auto_pad respects ceil_mode (uses floor formulas when ceil_mode=0)
        
        Args:
            input_size: Input size in this dimension
            kernel_size: Kernel size in this dimension
            stride: Stride in this dimension
            dilation: Dilation in this dimension
            mode: "SAME_UPPER", "SAME_LOWER", or "VALID"
            ceil_mode: ceil_mode attribute value
            opset_version: ONNX opset version
        
        Returns:
            Tuple of (pad_before, pad_after) for this dimension
        """
        if mode == "VALID":
            return (0, 0)
        
        # Calculate effective kernel size with dilation
        effective_kernel = (kernel_size - 1) * dilation + 1
        
        # Calculate output size based on opset_version and ceil_mode
        # According to ONNX spec (https://onnx.ai/onnx/operators/onnx__MaxPool.html):
        # - v1-v11: SAME always uses ceil(input/stride) regardless of ceil_mode
        # - v22: SAME respects ceil_mode:
        #   * ceil_mode=True: ceil(input/stride)
        #   * ceil_mode=False: floor((input-1)/stride) + 1
        # ONNX Runtime follows the spec for SAME padding.
        if mode in ("SAME_UPPER", "SAME_LOWER"):
            if opset_version >= 22:
                # v22: SAME respects ceil_mode
                if ceil_mode:
                    # ceil(input / stride)
                    output_size = (input_size + stride - 1) // stride
                else:
                    # floor((input - 1) / stride) + 1
                    output_size = ((input_size - 1) // stride) + 1
            else:
                # v1-v11: SAME always uses ceil(input/stride) regardless of ceil_mode
                output_size = (input_size + stride - 1) // stride  # ceil(input / stride)
        elif opset_version >= 22:
            # v22: VALID mode respects ceil_mode
            if ceil_mode:
                # ceil(input / stride)
                output_size = (input_size + stride - 1) // stride
            else:
                # floor((input - 1) / stride) + 1
                output_size = ((input_size - 1) // stride) + 1
        else:
            # v1-v11: VALID mode always uses ceil behavior
            output_size = (input_size + stride - 1) // stride  # ceil(input / stride)
        
        # Total padding needed to achieve the desired output size
        # Formula: total_pad = (output_size - 1) * stride + effective_kernel - input_size
        total_pad = max(0, (output_size - 1) * stride + effective_kernel - input_size)
        
        if mode == "SAME_UPPER":
            # More padding on the right/bottom (end of dimension)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
        else:  # SAME_LOWER
            # More padding on the left/top (beginning of dimension)
            pad_after = total_pad // 2
            pad_before = total_pad - pad_after
        
        return (pad_before, pad_after)
    
    @staticmethod
    def _convert_onnx_pads_to_pytorch(onnx_pads: List[int], ndim: int) -> Union[int, Tuple[int, ...]]:
        """
        Convert ONNX pads format to PyTorch padding format.
        
        ONNX format: [pad_dim0_begin, pad_dim1_begin, ..., pad_dim0_end, pad_dim1_end, ...]
        PyTorch format: int, (int,), (int, int), or (int, int, int, int, int, int)
        
        For 2D: ONNX [padH_begin, padW_begin, padH_end, padW_end] 
               -> PyTorch (padW_begin, padW_end, padH_begin, padH_end)
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
            # Asymmetric: PyTorch format is (padLeft, padRight, padTop, padBottom)
            # which is (padW_begin, padW_end, padH_begin, padH_end)
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
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        MaxPool opset v1-v7: Basic attributes, no ceil_mode, no dilations.
        """
        # Use common implementation with opset_version=1 (no ceil_mode, no dilations)
        return cls._convert_maxpool_impl(node_proto, input_tensors, output_tensors, attrs, 
                                         node_index, graph_proto, opset_version=1)
    
    @classmethod
    def _impl_v8(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        MaxPool opset v8-v9: Added Indices output and storage_order.
        Note: ceil_mode was NOT introduced in v8, it was introduced in v10.
        """
        # Use common implementation with opset_version=8 (no ceil_mode, no dilations)
        return cls._convert_maxpool_impl(node_proto, input_tensors, output_tensors, attrs, 
                                         node_index, graph_proto, opset_version=8)
    
    @classmethod
    def _impl_v10(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        MaxPool opset v10: Added ceil_mode and dilations attributes.
        """
        # Use common implementation with opset_version=10 (has ceil_mode, has dilations)
        return cls._convert_maxpool_impl(node_proto, input_tensors, output_tensors, attrs, 
                                         node_index, graph_proto, opset_version=10)
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        MaxPool opset v11-v21: Improved formulas with explicit ceil_mode handling.
        Auto_pad always uses ceil behavior regardless of ceil_mode.
        """
        # Use common implementation with opset_version=11 (has ceil_mode, has dilations, auto_pad always ceil)
        return cls._convert_maxpool_impl(node_proto, input_tensors, output_tensors, attrs, 
                                         node_index, graph_proto, opset_version=11)
    
    @classmethod
    def _impl_v22(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        MaxPool opset v22+: auto_pad respects ceil_mode, added bfloat16 support.
        """
        # Use common implementation with opset_version=22 (auto_pad respects ceil_mode)
        return cls._convert_maxpool_impl(node_proto, input_tensors, output_tensors, attrs, 
                                         node_index, graph_proto, opset_version=22)
    
    @classmethod
    def _convert_maxpool_impl(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                              output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                              node_index: int, graph_proto=None, opset_version: int = 11) -> List:
        """
        Common implementation for MaxPool conversion across all opset versions.
        
        Args:
            opset_version: ONNX opset version (1, 8, 10, 11, 22)
                          Used to determine the correct auto_pad formula and attribute availability.
        """
        nodes = []
        node_name = node_proto.name if node_proto.name else f"MaxPool_{node_index}"
        
        # Get kernel shape to determine pool dimension
        # kernel_shape is a REQUIRED attribute in ONNX MaxPool
        kernel_shape = attrs.get('kernel_shape')
        if kernel_shape is None:
            raise ValueError(
                f"MaxPool {node_name}: 'kernel_shape' is a required attribute but was not provided. "
                f"Please specify kernel_shape in the ONNX model."
            )
        
        # Normalize kernel_shape to tuple and determine dimensions
        if isinstance(kernel_shape, int):
            kernel_dims = 1
            kernel_shape_tuple = (kernel_shape,)
            kernel_size = kernel_shape
        elif isinstance(kernel_shape, (list, tuple)):
            kernel_dims = len(kernel_shape)
            kernel_shape_tuple = tuple(kernel_shape) if isinstance(kernel_shape, list) else kernel_shape
            if kernel_dims == 1:
                kernel_size = kernel_shape_tuple[0]
            else:
                kernel_size = kernel_shape_tuple
        else:
            raise ValueError(f"Invalid kernel_shape type for MaxPool {node_name}: {type(kernel_shape)}")
        
        # Extract and normalize attributes
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        strides = attrs.get('strides', 1)  # ONNX default is 1
        strides_tuple = cls._normalize_to_tuple(strides, kernel_dims, default=1)
        
        # Extract dilations (available from v10+)
        dilations = attrs.get('dilations')
        if dilations is not None:
            dilations_tuple = cls._normalize_to_tuple(dilations, kernel_dims, default=1)
        else:
            # Default dilation is 1
            dilations_tuple = (1,) * kernel_dims
        
        # Extract ceil_mode (available from v10+)
        # For v1 and v8, ceil_mode is not available, so default to False
        if opset_version >= 10:
            ceil_mode = bool(attrs.get('ceil_mode', 0))
        else:
            ceil_mode = False  # v1 and v8 don't have ceil_mode
        
        # Handle AUTO_PAD by creating PadNode if needed
        pool_inputs = list(node_proto.input)
        pool_input_tensors = input_tensors.copy()
        
        if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
            input_shape = input_tensors[pool_inputs[0]].shape
            if input_shape is None:
                logger.warning(f"Cannot compute auto_pad for MaxPool {node_name} with unknown input shape")
                auto_pad = 'NOTSET'
            elif auto_pad == 'VALID':
                # VALID means no padding - skip PadNode creation
                auto_pad = 'NOTSET'
            else:
                # Compute padding for each spatial dimension
                pads = []
                spatial_dims = input_shape[2:]  # Skip batch and channel
                for in_size, k_size, s, d in zip(spatial_dims, kernel_shape_tuple, strides_tuple, dilations_tuple):
                    pad_before, pad_after = cls._compute_autopad_padding(
                        in_size, k_size, s, d, auto_pad, ceil_mode, opset_version
                    )
                    pads.extend([pad_before, pad_after])
                
                # Convert to PyTorch F.pad format (reverse order: last dim first)
                pad_list = []
                for i in range(kernel_dims - 1, -1, -1):
                    pad_list.extend([pads[i * 2], pads[i * 2 + 1]])
                
                pad_name = f"{node_name}_pad"
                pad_output = f"{node_name}_padded"
                
                # Calculate output shape after padding
                input_info = input_tensors[pool_inputs[0]]
                padded_shape = list(input_shape)
                # Update spatial dimensions (skip batch and channel dims)
                for i in range(kernel_dims):
                    dim_idx = 2 + i  # Skip batch (0) and channel (1)
                    padded_shape[dim_idx] = padded_shape[dim_idx] + pads[i * 2] + pads[i * 2 + 1]
                
                # Create output tensor with updated shape
                pad_output_tensors = {
                    pad_output: TensorInfo(
                        name=pad_output,
                        shape=tuple(padded_shape),
                        onnx_dtype=getattr(input_info, 'onnx_dtype', None)
                    )
                }
                
                # For MaxPool, padding should use a very negative value so that padded values
                # don't affect the max operation. ONNX MaxPool excludes padded values from max.
                # PyTorch's F.pad supports float('-inf'), but we use a very large negative value
                # as a fallback for compatibility.
                try:
                    pad_value = float('-inf')
                except (ValueError, OverflowError):
                    pad_value = -1e10
                
                pad_node = PadNode.create(
                    name=pad_name,
                    inputs=[pool_inputs[0]],
                    outputs=[pad_output],
                    input_tensors={pool_inputs[0]: input_tensors[pool_inputs[0]]},
                    output_tensors=pad_output_tensors,
                    pad=tuple(pad_list),
                    mode='constant',
                    value=pad_value
                )
                nodes.append(pad_node)
                
                pool_inputs = [pad_output]
                pool_input_tensors = {pad_output: pad_output_tensors[pad_output]}
        
        # Extract and convert padding
        onnx_pads = attrs.get('pads')
        if onnx_pads and len(onnx_pads) > 0:
            # Check for asymmetric padding (not supported by PyTorch MaxPool)
            if kernel_dims == 1:
                pad_W_begin, pad_W_end = onnx_pads
                if pad_W_begin != pad_W_end:
                    raise ValueError(
                        f"MaxPool {node_name}: Asymmetric padding is not supported. "
                        f"Got pads={onnx_pads}. PyTorch MaxPool1d only supports symmetric padding. "
                        f"Use auto_pad='SAME_UPPER' or 'SAME_LOWER' for asymmetric padding behavior."
                    )
            elif kernel_dims == 2:
                pad_H_begin, pad_W_begin, pad_H_end, pad_W_end = onnx_pads
                if pad_H_begin != pad_H_end or pad_W_begin != pad_W_end:
                    raise ValueError(
                        f"MaxPool {node_name}: Asymmetric padding is not supported. "
                        f"Got pads={onnx_pads}. PyTorch MaxPool2d only supports symmetric padding "
                        f"(int or (int, int)). Use auto_pad='SAME_UPPER' or 'SAME_LOWER' for "
                        f"asymmetric padding behavior."
                    )
            elif kernel_dims == 3:
                pad_D_begin, pad_H_begin, pad_W_begin, pad_D_end, pad_H_end, pad_W_end = onnx_pads
                if pad_D_begin != pad_D_end or pad_H_begin != pad_H_end or pad_W_begin != pad_W_end:
                    raise ValueError(
                        f"MaxPool {node_name}: Asymmetric padding is not supported. "
                        f"Got pads={onnx_pads}. PyTorch MaxPool3d only supports symmetric padding "
                        f"(int or (int, int, int)). Use auto_pad='SAME_UPPER' or 'SAME_LOWER' for "
                        f"asymmetric padding behavior."
                    )
            
            padding = cls._convert_onnx_pads_to_pytorch(onnx_pads, kernel_dims)
        elif auto_pad != 'NOTSET':
            padding = 0  # Padding already handled by PadNode
        else:
            padding = 0  # Default no padding
        
        # Normalize stride and dilation for node creation
        stride_for_node = strides_tuple[0] if kernel_dims == 1 else strides_tuple
        dilation_for_node = dilations_tuple[0] if kernel_dims == 1 else dilations_tuple
        
        # Create appropriate MaxPool node based on dimension
        if kernel_dims == 1:
            pool_node = MaxPool1dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride_for_node,
                padding=padding,
                dilation=dilation_for_node,
                ceil_mode=ceil_mode
            )
        elif kernel_dims == 2:
            pool_node = MaxPool2dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride_for_node,
                padding=padding,
                dilation=dilation_for_node,
                ceil_mode=ceil_mode
            )
        elif kernel_dims == 3:
            pool_node = MaxPool3dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride_for_node,
                padding=padding,
                dilation=dilation_for_node,
                ceil_mode=ceil_mode
            )
        else:
            raise ValueError(f"Unsupported MaxPool dimension: {kernel_dims}")
        
        nodes.append(pool_node)
        return nodes


class AveragePoolConverter(OnnxOpConverter):
    """
    Converter for ONNX AveragePool operation.
    
    Supports opset versions: 1, 11, 19
    Supports dimensions: AveragePool1d, AveragePool2d, AveragePool3d
    
    Version differences:
    - v1: No ceil_mode, no count_include_pad, no dilations
    - v7: count_include_pad added
    - v10: ceil_mode added
    - v11-v18: Auto_pad always uses ceil(input/stride), regardless of ceil_mode
      (ONNX Runtime behavior, differs from spec)
    - v19+: dilations added, auto_pad respects ceil_mode
      (floor((input-1)/stride)+1 when ceil_mode=False)
    
    Key differences from ONNX to PyTorch:
    - count_include_pad: ONNX default is 0 (exclude), PyTorch default is True (include)
    - dilations: PyTorch AvgPool does NOT support dilation (error if != 1)
    """
    
    @staticmethod
    def _compute_autopad_padding(
        input_size: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        mode: str = "SAME_UPPER",
        ceil_mode: bool = False,
        opset_version: int = 11
    ) -> Tuple[int, int]:
        """
        Compute padding values for AveragePool auto_pad modes.
        
        Note: ONNX Runtime behavior for auto_pad modes:
        - Opset 1-18: Always uses ceil(input / stride) for auto_pad, regardless of ceil_mode
          (This differs from the ONNX spec, but we align with ONNX Runtime for compatibility)
        - Opset 19+: Respects ceil_mode:
          - ceil_mode=True: ceil(input / stride)
          - ceil_mode=False: floor((input - 1) / stride) + 1
        
        Args:
            input_size: Input size in this dimension
            kernel_size: Kernel size in this dimension
            stride: Stride in this dimension
            dilation: Dilation in this dimension
            mode: "SAME_UPPER", "SAME_LOWER", or "VALID"
            ceil_mode: ceil_mode attribute value (used for opset 19+)
            opset_version: ONNX opset version (11 for v1-v18, 19 for v19+)
        
        Returns:
            Tuple of (pad_before, pad_after) for this dimension
        """
        if mode == "VALID":
            return (0, 0)
        
        # Calculate effective kernel size with dilation
        effective_kernel = (kernel_size - 1) * dilation + 1
        
        # Calculate output size based on opset_version
        # ONNX Runtime behavior: opset 1-18 always uses ceil for auto_pad, opset 19+ respects ceil_mode
        if opset_version >= 19:
            # Opset 19+: Respect ceil_mode
            if ceil_mode:
                # ceil(input / stride)
                output_size = (input_size + stride - 1) // stride
            else:
                # floor((input - 1) / stride) + 1
                output_size = ((input_size - 1) // stride) + 1
        else:
            # Opset 1-18: Always use ceil behavior for auto_pad, regardless of ceil_mode
            # This matches ONNX Runtime's actual behavior (even though spec says otherwise)
            output_size = (input_size + stride - 1) // stride  # ceil(input / stride)
        
        # Total padding needed to achieve the desired output size
        # Formula: total_pad = (output_size - 1) * stride + effective_kernel - input_size
        total_pad = max(0, (output_size - 1) * stride + effective_kernel - input_size)
        
        if mode == "SAME_UPPER":
            # More padding on the right/bottom (end of dimension)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
        else:  # SAME_LOWER
            # More padding on the left/top (beginning of dimension)
            pad_after = total_pad // 2
            pad_before = total_pad - pad_after
        
        return (pad_before, pad_after)
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        AveragePool opset v1-v10: Basic attributes.
        - v1: No ceil_mode, no count_include_pad
        - v7: count_include_pad added
        - v10: ceil_mode added
        For auto_pad: Always uses ceil(input/stride) regardless of ceil_mode.
        """
        # Determine actual opset version for proper behavior
        # For v1-v6: opset_version=1 (no ceil_mode, no count_include_pad)
        # For v7-v9: opset_version=7 (has count_include_pad, no ceil_mode)
        # For v10: opset_version=10 (has ceil_mode, has count_include_pad)
        # Since we can't distinguish, use 10 as default (most permissive)
        return cls._convert_avgpool_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset_version=10)
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        AveragePool opset v11-v18: Improved auto_pad formulas.
        - Has ceil_mode and count_include_pad
        - Auto_pad respects ceil_mode: floor(input/stride) when ceil_mode=False
        """
        return cls._convert_avgpool_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset_version=11)
    
    @classmethod
    def _impl_v19(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        AveragePool opset v19+: Added dilations attribute.
        - Has ceil_mode, count_include_pad, and dilations
        - Auto_pad respects ceil_mode: floor((input-1)/stride)+1 when ceil_mode=False
        """
        return cls._convert_avgpool_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset_version=19)
    
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
    def _convert_avgpool_impl(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                              output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                              node_index: int, graph_proto=None, opset_version: int = 19) -> List:
        """
        Common implementation for AveragePool conversion across all opset versions.
        
        Args:
            opset_version: ONNX opset version (11 for v1-v18, 19 for v19+)
                          Used to determine the correct auto_pad formula.
        """
        nodes = []
        node_name = node_proto.name if node_proto.name else f"AveragePool_{node_index}"
        
        # Get kernel shape to determine pool dimension
        # kernel_shape is a REQUIRED attribute in ONNX AveragePool
        kernel_shape = attrs.get('kernel_shape')
        if kernel_shape is None:
            raise ValueError(
                f"AveragePool {node_name}: 'kernel_shape' is a required attribute but was not provided. "
                f"Please specify kernel_shape in the ONNX model."
            )
        
        # Normalize kernel_shape to tuple and determine dimensions
        if isinstance(kernel_shape, int):
            kernel_dims = 1
            kernel_shape_tuple = (kernel_shape,)
            kernel_size = kernel_shape
        elif isinstance(kernel_shape, (list, tuple)):
            kernel_dims = len(kernel_shape)
            kernel_shape_tuple = tuple(kernel_shape) if isinstance(kernel_shape, list) else kernel_shape
            if kernel_dims == 1:
                kernel_size = kernel_shape_tuple[0]
            else:
                kernel_size = kernel_shape_tuple
        else:
            raise ValueError(f"Invalid kernel_shape type for AveragePool {node_name}: {type(kernel_shape)}")
        
        # Check for dilation (available in v19+, but PyTorch doesn't support it)
        dilations = attrs.get('dilations')
        if dilations is not None:
            dilations_tuple = cls._normalize_to_tuple(dilations, kernel_dims, default=1)
            # Check if any dilation != 1 (PyTorch AvgPool doesn't support dilation)
            if any(d != 1 for d in dilations_tuple):
                raise ValueError(
                    f"AveragePool {node_name}: PyTorch AvgPool does not support dilation > 1. "
                    f"Got dilations={dilations_tuple}. ONNX models with dilation > 1 are not supported."
                )
            # Use normalized dilations for auto_pad calculation
            dilation_for_autopad = dilations_tuple
        else:
            # Default dilation is 1
            dilation_for_autopad = (1,) * kernel_dims
        
        # Extract and normalize attributes
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        strides = attrs.get('strides', 1)  # ONNX default is 1
        strides_tuple = cls._normalize_to_tuple(strides, kernel_dims, default=1)
        
        # Extract ceil_mode early (needed for auto_pad calculation)
        # ceil_mode: ONNX default is 0 (False), available from v7+
        ceil_mode = bool(attrs.get('ceil_mode', 0))
        
        # Handle AUTO_PAD by creating PadNode if needed
        pool_inputs = list(node_proto.input)
        pool_input_tensors = input_tensors.copy()
        
        if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
            input_shape = input_tensors[pool_inputs[0]].shape
            if input_shape is None:
                logger.warning(f"Cannot compute auto_pad for AveragePool {node_name} with unknown input shape")
                auto_pad = 'NOTSET'
            elif auto_pad == 'VALID':
                # VALID means no padding - skip PadNode creation
                auto_pad = 'NOTSET'
            else:
                # Compute padding for each spatial dimension
                # The _compute_autopad_padding method uses ceil_mode and opset_version to determine
                # the correct output size formula according to ONNX spec
                pads = []
                spatial_dims = input_shape[2:]  # Skip batch and channel
                for in_size, k_size, s, d in zip(spatial_dims, kernel_shape_tuple, strides_tuple, dilation_for_autopad):
                    pad_before, pad_after = cls._compute_autopad_padding(in_size, k_size, s, d, auto_pad, ceil_mode, opset_version)
                    pads.extend([pad_before, pad_after])
                
                # Convert to PyTorch F.pad format (reverse order: last dim first)
                pad_list = []
                for i in range(kernel_dims - 1, -1, -1):
                    pad_list.extend([pads[i * 2], pads[i * 2 + 1]])
                
                pad_name = f"{node_name}_pad"
                pad_output = f"{node_name}_padded"
                
                # Calculate output shape after padding
                # For 2D: (N, C, H, W) with pad (pad_h_before, pad_h_after, pad_w_before, pad_w_after)
                # Output: (N, C, H + pad_h_before + pad_h_after, W + pad_w_before + pad_w_after)
                input_info = input_tensors[pool_inputs[0]]
                padded_shape = list(input_shape)
                # Update spatial dimensions (skip batch and channel dims)
                for i in range(kernel_dims):
                    dim_idx = 2 + i  # Skip batch (0) and channel (1)
                    padded_shape[dim_idx] = padded_shape[dim_idx] + pads[i * 2] + pads[i * 2 + 1]
                
                # Create output tensor with updated shape
                pad_output_tensors = {
                    pad_output: TensorInfo(
                        name=pad_output,
                        shape=tuple(padded_shape),
                        onnx_dtype=getattr(input_info, 'onnx_dtype', None)
                    )
                }
                
                pad_node = PadNode.create(
                    name=pad_name,
                    inputs=[pool_inputs[0]],
                    outputs=[pad_output],
                    input_tensors={pool_inputs[0]: input_tensors[pool_inputs[0]]},
                    output_tensors=pad_output_tensors,
                    pad=tuple(pad_list),
                    mode='constant',
                    value=0.0
                )
                nodes.append(pad_node)
                
                pool_inputs = [pad_output]
                pool_input_tensors = {pad_output: pad_output_tensors[pad_output]}
        
        # Extract and convert padding
        onnx_pads = attrs.get('pads')
        if onnx_pads and len(onnx_pads) > 0:
            # Check for asymmetric padding (not supported by PyTorch AveragePool)
            if kernel_dims == 1:
                pad_W_begin, pad_W_end = onnx_pads
                if pad_W_begin != pad_W_end:
                    raise ValueError(
                        f"AveragePool {node_name}: Asymmetric padding is not supported. "
                        f"Got pads={onnx_pads}. PyTorch AvgPool1d only supports symmetric padding. "
                        f"Use auto_pad='SAME_UPPER' or 'SAME_LOWER' for asymmetric padding behavior."
                    )
            elif kernel_dims == 2:
                pad_H_begin, pad_W_begin, pad_H_end, pad_W_end = onnx_pads
                if pad_H_begin != pad_H_end or pad_W_begin != pad_W_end:
                    raise ValueError(
                        f"AveragePool {node_name}: Asymmetric padding is not supported. "
                        f"Got pads={onnx_pads}. PyTorch AvgPool2d only supports symmetric padding "
                        f"(int or (int, int)). Use auto_pad='SAME_UPPER' or 'SAME_LOWER' for "
                        f"asymmetric padding behavior."
                    )
            elif kernel_dims == 3:
                pad_D_begin, pad_H_begin, pad_W_begin, pad_D_end, pad_H_end, pad_W_end = onnx_pads
                if pad_D_begin != pad_D_end or pad_H_begin != pad_H_end or pad_W_begin != pad_W_end:
                    raise ValueError(
                        f"AveragePool {node_name}: Asymmetric padding is not supported. "
                        f"Got pads={onnx_pads}. PyTorch AvgPool3d only supports symmetric padding "
                        f"(int or (int, int, int)). Use auto_pad='SAME_UPPER' or 'SAME_LOWER' for "
                        f"asymmetric padding behavior."
                    )
            
            padding = cls._convert_onnx_pads_to_pytorch(onnx_pads, kernel_dims)
        elif auto_pad != 'NOTSET':
            padding = 0  # Padding already handled by PadNode
        else:
            padding = 0  # Default no padding
        
        # Extract other attributes with correct defaults
        # ceil_mode already extracted above (needed for auto_pad calculation)
        
        # count_include_pad: ONNX default is 0 (False), PyTorch default is True
        count_include_pad = bool(attrs.get('count_include_pad', 0))
        
        # Normalize stride for node creation
        stride_for_node = strides_tuple[0] if kernel_dims == 1 else strides_tuple
        
        # Create appropriate AveragePool node based on dimension
        if kernel_dims == 1:
            # Normalize padding for 1D (can be int or tuple of 2)
            p = padding if isinstance(padding, (int, tuple)) else 0
            
            pool_node = AveragePool1dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride_for_node,
                padding=p,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad
            )
        elif kernel_dims == 2:
            pool_node = AveragePool2dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride_for_node,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad
            )
        elif kernel_dims == 3:
            pool_node = AveragePool3dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride_for_node,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad
            )
        else:
            raise ValueError(f"Unsupported AveragePool dimension: {kernel_dims}")
        
        nodes.append(pool_node)
        return nodes


class GlobalAveragePoolConverter(OnnxOpConverter):
    """Converter for ONNX GlobalAveragePool operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """GlobalAveragePool opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"GlobalAveragePool_{node_index}"
        return [GlobalAveragePoolNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]

