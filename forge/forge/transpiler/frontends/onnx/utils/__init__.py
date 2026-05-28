# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX frontend utilities.

This package contains utility modules for the ONNX frontend, including:
- Naming utilities for node name sanitization
- Attribute extraction utilities
- Converter helper functions
- Validation utilities

Transpiler-level exceptions (``TranspilerError``, ``ConversionError``,
``UnsupportedOperationError``, ``ONNXModelValidationError``, etc.) are defined
in ``forge.transpiler.utils.exceptions`` and re-exported here for convenience.
"""
from forge.transpiler.frontends.onnx.utils.naming import sanitize_name, ensure_unique_name
from forge.transpiler.utils.exceptions import UnsupportedOperationError, ONNXModelValidationError
from forge.transpiler.frontends.onnx.utils.attributes import (
    extract_attributes,
    extract_attr_value,
    AttributeParsingError,
)
from forge.transpiler.frontends.onnx.utils.onnx_graph import (
    remove_initializers_from_input,
    get_inputs_names,
    get_outputs_names,
    torch_dtype_to_onnx_dtype,
)
from forge.transpiler.frontends.onnx.utils.io_builder import (
    build_input_output_dicts,
)
from forge.transpiler.frontends.onnx.utils.validation import (
    ConverterValidationError,
    validate_attributes,
    validate_constant_input,
    handle_validation_error,
)
from forge.transpiler.frontends.onnx.utils.onnx_printer import (
    print_onnx_model,
)
from forge.transpiler.utils.graph_printer import print_tir_graph

__all__ = [
    # Naming
    "sanitize_name",
    "ensure_unique_name",
    # Exceptions
    "UnsupportedOperationError",
    "ONNXModelValidationError",
    "ConverterValidationError",
    # Attributes
    "extract_attributes",
    "extract_attr_value",
    "AttributeParsingError",
    # Converter utilities
    "remove_initializers_from_input",
    "get_inputs_names",
    "get_outputs_names",
    "torch_dtype_to_onnx_dtype",
    "build_input_output_dicts",
    # Validation
    "validate_attributes",
    "validate_constant_input",
    "handle_validation_error",
    # Debug utilities
    "print_onnx_model",
    "print_tir_graph",
]
