# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Forge Transpiler Package

A multi-frontend transpiler for converting ML framework models to Forge intermediate representation.
Supports ONNX, with PaddlePaddle coming soon.
"""

# Import operations to register them (must be imported first)
from forge.transpiler.operations import *

# Public API - Core
from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.graph import TIRGraph

# Public API - Code Generation
from forge.transpiler.codegen.transpiler_generator import TranspilerCodeGenerator
from forge.transpiler.codegen.transpiler_to_forge import generate_forge_module_from_transpiler

# Public API - ONNX Frontend
from forge.transpiler.frontends.onnx import ONNXToForgeTranspiler, UnsupportedOperationError, ONNXModelValidationError
from forge.transpiler.frontends.onnx.utils import (
    extract_attributes,
    extract_attr_value,
    remove_initializers_from_input,
    get_inputs_names,
    get_outputs_names,
)
from forge.transpiler.frontends.onnx.debug import debug_node_output, get_activation_value


__all__ = [
    # Types (IR)
    "TensorInfo",
    "onnx_dtype_to_torch_dtype",
    # Nodes (IR)
    "TIRNode",
    # Graph (Core)
    "TIRGraph",
    # Code Generation
    "TranspilerCodeGenerator",
    "generate_forge_module_from_transpiler",
    # ONNX Frontend
    "ONNXToForgeTranspiler",
    "UnsupportedOperationError",
    "ONNXModelValidationError",
    # ONNX Converters
    "extract_attributes",
    "extract_attr_value",
    "remove_initializers_from_input",
    "get_inputs_names",
    "get_outputs_names",
    # ONNX Debug
    "debug_node_output",
    "get_activation_value",
]
