"""
ONNX-specific utility functions.
"""
import onnx
import torch
from onnx import ModelProto
from typing import List, Tuple


def remove_initializers_from_input(model: ModelProto) -> ModelProto:
    """
    Removes initializer names from the graph's input list. 
    """
    graph_inputs = model.graph.input
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    
    # Filter out inputs that are also initializers. Modifies list in place.
    for i in range(len(graph_inputs) - 1, -1, -1):
        if graph_inputs[i].name in initializer_names:
            del graph_inputs[i]

    return model


def get_inputs_names(onnx_graph) -> List[str]:
    """Get list of input names (excluding initializers)."""
    param_names = {x.name for x in onnx_graph.initializer}
    input_names = [x.name for x in onnx_graph.input if x.name not in param_names]
    return input_names


def get_outputs_names(onnx_graph) -> List[str]:
    """Get list of output names."""
    return [x.name for x in onnx_graph.output]


def torch_dtype_to_onnx_dtype(torch_dtype: torch.dtype) -> int:
    """
    Convert torch.dtype to ONNX TensorProto dtype.
    
    Args:
        torch_dtype: PyTorch dtype (e.g., torch.float32, torch.int64)
        
    Returns:
        ONNX TensorProto dtype constant (e.g., onnx.TensorProto.FLOAT)
        
    Example:
        >>> torch_dtype_to_onnx_dtype(torch.float32)
        1  # onnx.TensorProto.FLOAT
        >>> torch_dtype_to_onnx_dtype(torch.int64)
        7  # onnx.TensorProto.INT64
    """
    dtype_map = {
        torch.float32: onnx.TensorProto.FLOAT,
        torch.float64: onnx.TensorProto.DOUBLE,
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.int32: onnx.TensorProto.INT32,
        torch.int64: onnx.TensorProto.INT64,
        torch.uint8: onnx.TensorProto.UINT8,
        torch.int8: onnx.TensorProto.INT8,
        torch.bool: onnx.TensorProto.BOOL,
    }
    return dtype_map.get(torch_dtype, onnx.TensorProto.FLOAT)

