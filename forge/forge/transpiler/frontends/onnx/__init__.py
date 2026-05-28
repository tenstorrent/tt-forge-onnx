# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX frontend for transpiler.
"""
# Register ONNX op-type → ShapeEvalMeta mappings with the core registry.
# This import must come before any TIR node for an ONNX op is instantiated.
from forge.transpiler.frontends.onnx.operations import op_shape_meta as _op_shape_meta  # noqa: F401

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.utils.exceptions import UnsupportedOperationError, ONNXModelValidationError

__all__ = ["ONNXToForgeTranspiler", "UnsupportedOperationError", "ONNXModelValidationError"]
