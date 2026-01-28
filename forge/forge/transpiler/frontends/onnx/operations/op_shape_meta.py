# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX frontend op-shape metadata registration.

This module owns the mapping from TIR op-type names (as used in the ONNX
frontend) to ShapeEvalMeta instances and registers it with the core registry
at import time.

Shape dependency classes
------------------------
SHAPE_ONLY          – output shape is determined entirely by input shapes.
                      Fake-execution is safe; shape rules are available.
VALUE_OF_SHAPE_INPUT – output shape depends on the *value* of a special
                      shape-input tensor (e.g. Reshape's second input).
VALUE_DEPENDENT     – output shape depends on runtime input data values;
                      this is the conservative default for unknown ops.

To add a new ONNX-frontend op type, append it to the appropriate section
below and re-import the module (it is safe to call register_shape_eval_meta
more than once).
"""

from forge.transpiler.core.shape_eval import (
    SHAPE_ONLY,
    VALUE_OF_SHAPE_INPUT,
    register_shape_eval_meta,
)

# Keys are TIR op_type strings set by the ONNX frontend converters.
_ONNX_OP_SHAPE_EVAL_META = {
    # ── Arithmetic / element-wise ─────────────────────────────────────────
    "Add": SHAPE_ONLY,
    "Sub": SHAPE_ONLY,
    "Mul": SHAPE_ONLY,
    "Div": SHAPE_ONLY,
    "Equal": SHAPE_ONLY,
    "Greater": SHAPE_ONLY,
    "Less": SHAPE_ONLY,
    "GreaterOrEqual": SHAPE_ONLY,
    "LessOrEqual": SHAPE_ONLY,
    # ── Linear / matrix ──────────────────────────────────────────────────
    "MatMul": SHAPE_ONLY,
    # ── Indexing / lookup ─────────────────────────────────────────────────
    "Embedding": SHAPE_ONLY,
    "IndexSelect": SHAPE_ONLY,
    "Index": SHAPE_ONLY,
    # ── Shape-preserving transforms ───────────────────────────────────────
    "Transpose": SHAPE_ONLY,
    "Cast": SHAPE_ONLY,
    "Clip": SHAPE_ONLY,
    "Identity": SHAPE_ONLY,
    "Concat": SHAPE_ONLY,
    "Squeeze": SHAPE_ONLY,
    "Unsqueeze": SHAPE_ONLY,
    # ── Reduction ops ─────────────────────────────────────────────────────
    "ReduceSum": SHAPE_ONLY,
    "ReduceMean": SHAPE_ONLY,
    "ReduceMax": SHAPE_ONLY,
    # ── Activation functions ──────────────────────────────────────────────
    "Relu": SHAPE_ONLY,
    "Sigmoid": SHAPE_ONLY,
    "Tanh": SHAPE_ONLY,
    "Sqrt": SHAPE_ONLY,
    "Erf": SHAPE_ONLY,
    "LeakyRelu": SHAPE_ONLY,
    "Softmax": SHAPE_ONLY,
    "LogSoftmax": SHAPE_ONLY,
    "Dropout": SHAPE_ONLY,
    # ── Other shape-safe ops ──────────────────────────────────────────────
    "Pad": SHAPE_ONLY,
    "Split": SHAPE_ONLY,
    "Where": SHAPE_ONLY,
    "Broadcast": SHAPE_ONLY,
    "Conv1d": SHAPE_ONLY,
    "Conv2d": SHAPE_ONLY,
    "Conv3d": SHAPE_ONLY,
    "MaxPool1d": SHAPE_ONLY,
    "MaxPool2d": SHAPE_ONLY,
    "MaxPool3d": SHAPE_ONLY,
    "AveragePool1d": SHAPE_ONLY,
    "AveragePool2d": SHAPE_ONLY,
    "AveragePool3d": SHAPE_ONLY,
    "Reciprocal": SHAPE_ONLY,
    "Pow": SHAPE_ONLY,
    "LayerNorm": SHAPE_ONLY,
    "Shape": SHAPE_ONLY,
    "Constant": SHAPE_ONLY,
    "Full": SHAPE_ONLY,
    # ── Shape-control ops (output shape depends on a shape-input value) ───
    "Reshape": VALUE_OF_SHAPE_INPUT,
    "Expand": VALUE_OF_SHAPE_INPUT,
    "Slice": VALUE_OF_SHAPE_INPUT,
    "ConstantOfShape": VALUE_OF_SHAPE_INPUT,
}

# Register with the core registry — executed once when this module is imported.
register_shape_eval_meta(_ONNX_OP_SHAPE_EVAL_META)
