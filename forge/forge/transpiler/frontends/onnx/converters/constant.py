"""
ONNX Constant operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
import onnx
from forge.transpiler.ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from loguru import logger


class ConstantConverter(OnnxOpConverter):
    """Converter for ONNX Constant operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Constant opset v1-v8: value stored in 'value' attribute as TensorProto.
        """
        return cls._convert_constant_impl(node_proto, attrs, node_index)
    
    @classmethod
    def _impl_v9(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Constant opset v9+: value can be stored in 'value' attribute (TensorProto) 
        or 'sparse_value' attribute (SparseTensorProto).
        """
        return cls._convert_constant_impl(node_proto, attrs, node_index)
    
    @classmethod
    def _impl_v12(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Constant opset v12+: value can be stored in:
        - 'value' attribute (TensorProto)
        - 'sparse_value' attribute (SparseTensorProto)
        - 'value_int' attribute (int)
        - 'value_float' attribute (float)
        - 'value_strings' attribute (list of strings)
        - 'value_ints' attribute (list of ints)
        - 'value_floats' attribute (list of floats)
        """
        return cls._convert_constant_impl(node_proto, attrs, node_index)
    
    @classmethod
    def _convert_constant_impl(cls, node_proto: NodeProto, attrs: Dict[str, Any],
                                node_index: int) -> List:
        """
        Common implementation for Constant conversion.
        Extracts constant value from attributes and returns empty list
        (constant will be stored in graph.constants by the engine).
        """
        node_name = node_proto.name if node_proto.name else f"Constant_{node_index}"
        
        # Extract constant value from attributes
        # Priority: value (TensorProto) > sparse_value > value_int/value_float/value_strings/etc.
        constant_value = None
        
        # Try 'value' attribute (TensorProto) - most common
        # Note: extract_attributes may have already converted it to numpy array
        if 'value' in attrs and attrs['value'] is not None:
            try:
                from onnx import numpy_helper
                value_attr = attrs['value']
                
                # Check if it's already a numpy array (from extract_attributes)
                if isinstance(value_attr, (list, tuple)) or hasattr(value_attr, 'shape'):
                    # Already converted to numpy array
                    import numpy as np
                    if isinstance(value_attr, np.ndarray):
                        constant_value = value_attr
                    else:
                        constant_value = np.array(value_attr)
                elif hasattr(value_attr, 'data_type'):
                    # It's a TensorProto - convert to numpy
                    np_array = numpy_helper.to_array(value_attr)
                    constant_value = np_array
                else:
                    # Try to get the raw attribute from node_proto
                    for attr in node_proto.attribute:
                        if attr.name == 'value' and attr.type == onnx.AttributeProto.TENSOR:
                            np_array = numpy_helper.to_array(attr.t)
                            constant_value = np_array
                            break
            except Exception as e:
                logger.warning(f"Failed to extract value from 'value' attribute in Constant {node_name}: {e}")
        
        # Try 'sparse_value' attribute (SparseTensorProto) - opset >= 9
        if constant_value is None and 'sparse_value' in attrs and attrs['sparse_value'] is not None:
            try:
                from onnx import numpy_helper
                sparse_value_attr = attrs['sparse_value']
                # Convert sparse tensor to dense (simplified - full sparse support would be more complex)
                logger.warning(f"Constant {node_name} uses sparse_value, converting to dense (may be inefficient)")
                # For now, we'll try to extract the values
                if hasattr(sparse_value_attr, 'values'):
                    np_array = numpy_helper.to_array(sparse_value_attr.values)
                    constant_value = np_array
            except Exception as e:
                logger.warning(f"Failed to extract value from 'sparse_value' attribute in Constant {node_name}: {e}")
        
        # Try scalar value attributes (opset >= 12)
        if constant_value is None:
            if 'value_int' in attrs and attrs['value_int'] is not None:
                constant_value = attrs['value_int']
            elif 'value_float' in attrs and attrs['value_float'] is not None:
                constant_value = attrs['value_float']
            elif 'value_strings' in attrs and attrs['value_strings'] is not None:
                # String constants - convert to tensor of indices or handle specially
                logger.warning(f"Constant {node_name} uses value_strings, which may not be fully supported")
                constant_value = attrs['value_strings']
            elif 'value_ints' in attrs and attrs['value_ints'] is not None:
                import numpy as np
                constant_value = np.array(attrs['value_ints'], dtype=np.int64)
            elif 'value_floats' in attrs and attrs['value_floats'] is not None:
                import numpy as np
                constant_value = np.array(attrs['value_floats'], dtype=np.float32)
        
        if constant_value is None:
            raise ValueError(
                f"Constant {node_name} has no valid value attribute. "
                f"Expected one of: value, sparse_value, value_int, value_float, value_strings, value_ints, value_floats"
            )
        
        # Convert to torch tensor
        import torch
        import numpy as np
        
        if isinstance(constant_value, np.ndarray):
            # Get dtype from the array or infer from ONNX attribute
            onnx_dtype = None
            
            # Try to get dtype from raw node_proto attributes (before extract_attributes conversion)
            for attr in node_proto.attribute:
                if attr.name == 'value' and attr.type == onnx.AttributeProto.TENSOR:
                    onnx_dtype = attr.t.data_type
                    break
                elif attr.name == 'sparse_value' and attr.type == onnx.AttributeProto.SPARSE_TENSOR:
                    if hasattr(attr.sparse_tensor, 'values') and hasattr(attr.sparse_tensor.values, 'data_type'):
                        onnx_dtype = attr.sparse_tensor.values.data_type
                        break
            
            if onnx_dtype is not None:
                torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)
            else:
                # Infer from numpy array dtype
                torch_dtype = onnx_dtype_to_torch_dtype(constant_value.dtype)
            
            torch_tensor = torch.from_numpy(constant_value).to(torch_dtype)
        elif isinstance(constant_value, (int, float)):
            # Scalar value
            if isinstance(constant_value, int):
                torch_tensor = torch.tensor(constant_value, dtype=torch.int64)
            else:
                torch_tensor = torch.tensor(constant_value, dtype=torch.float32)
        elif isinstance(constant_value, list):
            # List of values
            np_array = np.array(constant_value)
            torch_dtype = onnx_dtype_to_torch_dtype(np_array.dtype)
            torch_tensor = torch.from_numpy(np_array).to(torch_dtype)
        else:
            raise ValueError(
                f"Constant {node_name} has unsupported value type: {type(constant_value)}"
            )
        
        # Store constant value in a special attribute that the engine will use
        # We return an empty list because Constant nodes don't create TIR nodes
        # Instead, the engine will store the constant in tir_graph.constants
        # We attach the constant value to the node_proto for the engine to extract
        if not hasattr(node_proto, '_forge_constant_value'):
            node_proto._forge_constant_value = torch_tensor
        
        # Return empty list - no TIR node needed, constant will be stored by engine
        return []

