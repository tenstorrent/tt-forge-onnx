"""
Forge Transpiler Package

A multi-frontend transpiler for converting ML framework models to Forge intermediate representation.
Supports ONNX, with PaddlePaddle and TensorFlow coming soon.
"""

# Configure logging using loguru
from loguru import logger

# Import operations to register them (must be imported first)
from forge.transpiler.ir.operations import *

# Public API - IR (common across all frontends)
from forge.transpiler.ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.ir.nodes import TIRNode

# Public API - Core
from forge.transpiler.core.graph import TIRGraph

# Public API - Code Generation
from forge.transpiler.codegen.transpiler_generator import TranspilerCodeGenerator
from forge.transpiler.codegen.transpiler_to_forge import generate_forge_module_from_transpiler

# Public API - ONNX Frontend
from forge.transpiler.frontends.onnx import ONNXToForgeTranspiler
from forge.transpiler.frontends.onnx.converters import (
    extract_attributes,
    extract_attr_value,
    remove_initializers_from_input,
    get_inputs_names,
    get_outputs_names,
)
from forge.transpiler.frontends.onnx.debug import debug_node_output, get_activation_value

# Public API - Common Utils
from forge.transpiler.utils import (
    is_constant,
    is_symmetric_padding,
    extract_padding_for_conv,
    get_selection,
)

__all__ = [
    # Types (IR)
    'TensorInfo',
    'onnx_dtype_to_torch_dtype',
    # Nodes (IR)
    'TIRNode',
    # Graph (Core)
    'TIRGraph',
    # Code Generation
    'TranspilerCodeGenerator',
    'generate_forge_module_from_transpiler',
    # ONNX Frontend
    'ONNXToForgeTranspiler',
    # ONNX Converters
    'extract_attributes',
    'extract_attr_value',
    'remove_initializers_from_input',
    'get_inputs_names',
    'get_outputs_names',
    # ONNX Debug
    'debug_node_output',
    'get_activation_value',
    # Common Utils
    'is_constant',
    'is_symmetric_padding',
    'extract_padding_for_conv',
    'get_selection',
]
