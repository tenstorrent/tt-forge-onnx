"""
ONNX Reshape operation converter with opset version support.
"""
from typing import List, Dict, Any, Tuple, Optional
from onnx import NodeProto
import torch
import numpy as np
from forge.transpiler.ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.ir.operations.shape import ReshapeNode
from forge.transpiler.ir.operations.other import FullNode, IdentityNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.validation import validate_constant_input, handle_validation_error


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
        if hasattr(shape_value, '__iter__') and not isinstance(shape_value, (str, bytes)):
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
                f"Cannot normalize shape value {shape_value} (type: {type(shape_value)}) "
                f"to tuple of integers: {e}"
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
            raise TypeError(
                f"Shape must be a list or tuple, got {type(shape).__name__}: {shape}"
            )
        
        shape = list(shape)
        input_shape = tuple(input_shape) if input_shape else ()
        
        # Validate that shape doesn't contain both -1 and 0
        has_neg_one = -1 in shape
        contains_zero = 0 in shape
        if has_neg_one and contains_zero:
            raise ValueError(
                f"Shape cannot contain both -1 (inferred dimension) and 0 (copy/explicit zero). "
                f"Shape: {shape}"
            )
        
        # Calculate total elements from input
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim if dim is not None else 1
        
        # Handle 0 dimensions based on allowzero
        has_zero_kept = False  # Track if we kept a 0 with allowzero=1
        
        for i, s in enumerate(shape):
            if s == 0:
                if allowzero == 1:
                    # Keep as 0 (will create Constant node for empty tensor)
                    has_zero_kept = True
                else:
                    # Copy from input (default behavior, backward compatible)
                    if i < len(input_shape) and input_shape[i] is not None:
                        shape[i] = input_shape[i]
                    else:
                        raise ValueError(
                            f"Cannot copy dimension {i} from input shape {input_shape}"
                        )
        
        # Handle -1 (inferred dimension) - only if no zeros with allowzero=1
        if -1 in shape and not (has_zero_kept and allowzero == 1):
            # Check for multiple -1 values (invalid)
            neg_one_indices = [i for i, s in enumerate(shape) if s == -1]
            if len(neg_one_indices) > 1:
                raise ValueError(
                    f"Cannot infer dimension: shape contains multiple -1 values. "
                    f"Shape: {shape}"
                )
            
            inferred_idx = neg_one_indices[0]
            # Calculate product of known dimensions
            known_product = 1
            for i, s in enumerate(shape):
                if i != inferred_idx:
                    known_product *= s if s > 0 else 1
            
            if known_product == 0:
                raise ValueError(
                    f"Cannot infer dimension when product of other dimensions is 0. "
                    f"Shape: {shape}"
                )
            
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
                resolved_product *= s if s > 0 else 1
            
            if resolved_product != total_elements:
                raise ValueError(
                    f"Cannot reshape tensor of size {total_elements} into shape {shape} "
                    f"(product: {resolved_product})"
                )
        
        return tuple(shape)
    
    @classmethod
    def _convert_reshape_impl(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                               output_tensors: Dict[str, TensorInfo], shape: Tuple,
                               allowzero: int, node_index: int) -> List:
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
        output_name = node_proto.output[0]
        data_input = node_proto.input[0]
        
        # Get input info
        input_info = input_tensors.get(data_input)
        if input_info is None:
            error_msg = f"Reshape {node_name}: input tensor {data_input} not found"
            handle_validation_error(node_proto, error_msg, strict=True)
            return []
        
        input_shape = input_info.shape if input_info.shape else ()
        input_dtype = input_info.onnx_dtype if hasattr(input_info, 'onnx_dtype') else None
        
        # Handle empty shape () or len(shape) == 0: create ReshapeNode with shape (-1) to flatten
        if shape == () or (isinstance(shape, tuple) and len(shape) == 0):
            return [ReshapeNode.create(
                name=node_name,
                inputs=[data_input],
                outputs=[output_name],
                input_tensors={data_input: input_info},
                output_tensors=output_tensors,
                shape=(-1,)
            )]
        
        # Resolve -1 and 0 in shape
        resolved_shape = cls._resolve_shape(shape, input_shape, allowzero)
        
        # Check if resolved shape contains 0 with allowzero=1 (empty tensor)
        # Create FullNode for empty tensor instead of ReshapeNode
        if 0 in resolved_shape and allowzero == 1:
            torch_dtype = onnx_dtype_to_torch_dtype(input_dtype) if input_dtype else torch.float32
            return [FullNode.create(
                name=node_name,
                inputs=[],  # No inputs for constant creation
                outputs=[output_name],
                input_tensors={},
                output_tensors=output_tensors,
                shape=resolved_shape,
                fill_value=0.0,
                dtype=torch_dtype
            )]
        
        # Optimization: If input shape and resolved shape are the same, use Identity
        if input_shape == resolved_shape:
            return [IdentityNode.create(
                name=node_name,
                inputs=[data_input],
                outputs=[output_name],
                input_tensors={data_input: input_info},
                output_tensors=output_tensors
            )]
        
        # Normal case: create ReshapeNode
        return [ReshapeNode.create(
            name=node_name,
            inputs=[data_input],
            outputs=[output_name],
            input_tensors={data_input: input_info},
            output_tensors=output_tensors,
            shape=resolved_shape
        )]
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Reshape opset v1-v4: shape as attribute.
        """
        node_name = node_proto.name if node_proto.name else f"Reshape_{node_index}"
        
        # Extract shape from attribute
        shape = attrs.get('shape', None)
        if shape is None:
            error_msg = f"Reshape {node_name} (opset < 5) requires 'shape' attribute"
            handle_validation_error(node_proto, error_msg, strict=True)
            return []
        
        shape = cls._normalize_shape_value(shape)
        
        # Opset 1-4: allowzero not supported, defaults to 0 (copy from input)
        allowzero = 0
        
        return cls._convert_reshape_impl(node_proto, input_tensors, output_tensors, 
                                         shape, allowzero, node_index)
    
    @classmethod
    def _extract_shape_from_input(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                                   output_tensors: Dict[str, TensorInfo], graph_proto) -> Tuple[Tuple, Optional[str]]:
        """
        Extract and normalize shape from input tensor (for opset >= 5).
        
        Returns:
            Tuple of (normalized_shape, error_message)
            If error_message is not None, extraction failed
        """
        # Validate and extract shape from constant input (second input)
        is_valid, shape_value, error_msg = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
        
        if not is_valid:
            # If shape input is optional and not provided, try to get from output shape
            if shape_value is None and len(node_proto.output) > 0:
                output_info = output_tensors.get(node_proto.output[0])
                if output_info and output_info.shape:
                    shape_value = output_info.shape
                else:
                    return None, error_msg or "Shape input required"
            else:
                return None, error_msg or "Shape input required"
        
        # Normalize shape_value to tuple
        shape = cls._normalize_shape_value(shape_value)
        return shape, None
    
    @classmethod
    def _impl_v5(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Reshape opset v5-v13: shape as input tensor (second input).
        No allowzero attribute (introduced in opset 14).
        """
        # Extract shape from input tensor
        shape, error_msg = cls._extract_shape_from_input(node_proto, input_tensors, output_tensors, graph_proto)
        if shape is None:
            handle_validation_error(node_proto, error_msg, strict=True)
            return []
        
        # Opset 5-13: allowzero defaults to 0 (copy from input)
        allowzero = 0
        
        return cls._convert_reshape_impl(node_proto, input_tensors, output_tensors, 
                                         shape, allowzero, node_index)
    
    @classmethod
    def _impl_v14(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Reshape opset v14+: shape as input tensor, allowzero attribute introduced.
        
        allowzero (default 0):
        - 0: Dimension 0 means copy from input (backward compatible)
        - 1: Dimension 0 means explicitly zero (NumPy-like)
        """
        # Extract shape from input tensor
        shape, error_msg = cls._extract_shape_from_input(node_proto, input_tensors, output_tensors, graph_proto)
        if shape is None:
            handle_validation_error(node_proto, error_msg, strict=True)
            return []
        
        # Extract allowzero attribute (default is 0)
        allowzero = int(attrs.get('allowzero', 0))
        
        return cls._convert_reshape_impl(node_proto, input_tensors, output_tensors, 
                                         shape, allowzero, node_index)

