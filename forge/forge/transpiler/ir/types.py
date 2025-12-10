"""
Type definitions for the transpiler IR.
"""
import onnx
from onnx import TensorProto
import torch
from typing import Optional, Tuple


def onnx_dtype_to_torch_dtype(onnx_dtype: int) -> torch.dtype:
    """Converts an onnx.TensorProto.DataType to a torch.dtype."""
    dtype_map = {
        int(TensorProto.FLOAT): torch.float32,
        int(TensorProto.UINT8): torch.uint8,
        int(TensorProto.INT8): torch.int8,
        int(TensorProto.INT16): torch.int16,
        int(TensorProto.INT32): torch.int32,
        int(TensorProto.INT64): torch.int64,
        int(TensorProto.BOOL): torch.bool,
        int(TensorProto.FLOAT16): torch.float16,
        int(TensorProto.DOUBLE): torch.float64,
        int(TensorProto.COMPLEX64): torch.complex64,
        int(TensorProto.COMPLEX128): torch.complex128,
        # We ignore complex types, strings, and non-standard types for simplicity
    }
    return dtype_map.get(onnx_dtype, torch.float32)


class TensorInfo:
    """
    Tensor metadata for the IR.
    Stores shape, ONNX dtype, and the derived PyTorch dtype.
    Framework-agnostic - works for all frontends.
    """
    def __init__(self, 
                 name: str, 
                 shape: Optional[Tuple], 
                 onnx_dtype: int):
        self.name = name
        self.shape = shape
        self.onnx_dtype = onnx_dtype
        self.torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)

    def __repr__(self):
        return f"TensorInfo(name='{self.name}', shape={self.shape}, dtype={self.torch_dtype})"

