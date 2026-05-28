# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX operation converters.

This package contains all operation-specific converters for converting ONNX operations
to TIR nodes. Each converter handles the conversion from ONNX NodeProto to TIRNode(s)
or ConstantResult, with support for different ONNX opset versions.

Converters follow a consistent pattern:
- Inherit from OnnxOpConverter base class
- Implement convert() classmethod with opset version support
- Return ConverterResult (either List[TIRNode] or ConstantResult)
- Handle opset-specific differences using the opset parameter

For utilities and helper functions, see frontends.onnx.utils.
"""
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.converter_result import (
    ConverterResult,
    ConstantResult,
    is_constant_result,
    is_tir_nodes_result,
)

__all__ = [
    "OnnxOpConverter",
    "ConverterResult",
    "ConstantResult",
    "is_constant_result",
    "is_tir_nodes_result",
]
