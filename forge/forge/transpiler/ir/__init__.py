"""
Intermediate Representation (IR) module.

Contains common IR types and operations shared across all frontends.
"""
from forge.transpiler.ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.ir.nodes import TIRNode

# Import operations
from forge.transpiler.ir.operations import *

__all__ = [
    'TensorInfo',
    'onnx_dtype_to_torch_dtype',
    'TIRNode',
]

