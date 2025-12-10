"""
Simple versioned converters for operations without significant version differences.
These are wrapper converters that delegate to the engine's converter methods.
"""
from typing import List, Dict, Any, Callable
from onnx import NodeProto
from forge.transpiler.ir.types import TensorInfo
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter


class SimpleConverter(OnnxOpConverter):
    """
    Base class for simple converters that don't have version differences.
    Wraps a converter function.
    """
    
    def __init__(self, converter_func: Callable):
        self.converter_func = converter_func
    
    @classmethod
    def create(cls, converter_func: Callable):
        """Create a simple converter wrapper."""
        class Wrapper(cls):
            @classmethod
            def _impl_v1(cls, node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto=None):
                return converter_func(node_proto, input_tensors, output_tensors, attrs, node_index)
        return Wrapper


def create_simple_versioned_converter(converter_func: Callable) -> type:
    """
    Create a versioned converter for operations without version differences.
    All versions delegate to the same converter function.
    """
    class SimpleVersionedConverter(OnnxOpConverter):
        @classmethod
        def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                     output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                     node_index: int, graph_proto=None) -> List:
            """All versions use the same implementation."""
            return converter_func(node_proto, input_tensors, output_tensors, attrs, node_index)
    
    return SimpleVersionedConverter

