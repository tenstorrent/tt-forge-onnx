# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Common utilities shared across all frontends.

Framework-agnostic utilities that work with TIRGraph and other core components.
Framework-specific utilities are located in their respective frontend utils
directories (e.g., frontends/onnx/utils/).
"""
from forge.transpiler.utils.graph_printer import print_tir_graph
from forge.transpiler.utils.binary_ops import (
    validate_binary_inputs_pytorch_style,
    are_shapes_equal,
    are_shapes_compatible_for_broadcasting,
    validate_limited_broadcasting,
    validate_broadcast_attributes,
    compute_broadcasted_shape,
    compute_broadcasted_shape_multi,
)
from forge.transpiler.utils.exceptions import (
    TranspilerError,
    ConversionError,
    ValidationError,
    DebugValidationError,
    UnsupportedOperationError,
    ONNXModelValidationError,
)

__all__ = [
    # Graph utilities
    "print_tir_graph",
    # Broadcasting / binary-op utilities
    "validate_binary_inputs_pytorch_style",
    "are_shapes_equal",
    "are_shapes_compatible_for_broadcasting",
    "validate_limited_broadcasting",
    "validate_broadcast_attributes",
    "compute_broadcasted_shape",
    "compute_broadcasted_shape_multi",
    # Exceptions
    "TranspilerError",
    "ConversionError",
    "ValidationError",
    "DebugValidationError",
    "UnsupportedOperationError",
    "ONNXModelValidationError",
]
