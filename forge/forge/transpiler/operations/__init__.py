# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Operations package for TIR nodes.
Imports all operations to register them.
"""
# Import all operations to register them
from forge.transpiler.operations.arithmetic import (
    AddNode,
    SubNode,
    MulNode,
    DivNode,
    MatMulNode,
)
from forge.transpiler.operations.activation import (
    ReluNode,
    SigmoidNode,
    TanhNode,
    SoftmaxNode,
    LogSoftmaxNode,
    LeakyReluNode,
    DropoutNode,
)
from forge.transpiler.operations.conv import Conv1dNode, Conv2dNode, Conv3dNode
from forge.transpiler.operations.pooling import (
    MaxPool1dNode,
    MaxPool2dNode,
    MaxPool3dNode,
    AveragePool1dNode,
    AveragePool2dNode,
    AveragePool3dNode,
)
from forge.transpiler.operations.shape import (
    ReshapeNode,
    TransposeNode,
    SqueezeNode,
    UnsqueezeNode,
)
from forge.transpiler.operations.reduction import (
    ReduceSumNode,
    ReduceMeanNode,
    ReduceMaxNode,
)
from forge.transpiler.operations.other import (
    ConcatNode,
    ClipNode,
    CastNode,
    IdentityNode,
    FullNode,
)

__all__ = [
    # Arithmetic
    "AddNode",
    "SubNode",
    "MulNode",
    "DivNode",
    "MatMulNode",
    # Convolution
    "Conv1dNode",
    "Conv2dNode",
    "Conv3dNode",
    # Activation
    "ReluNode",
    "SigmoidNode",
    "TanhNode",
    "SoftmaxNode",
    "LogSoftmaxNode",
    "LeakyReluNode",
    "DropoutNode",
    # Pooling
    "MaxPool1dNode",
    "MaxPool2dNode",
    "MaxPool3dNode",
    "AveragePool1dNode",
    "AveragePool2dNode",
    "AveragePool3dNode",
    # Shape
    "ReshapeNode",
    "TransposeNode",
    "SqueezeNode",
    "UnsqueezeNode",
    # Reduction
    "ReduceSumNode",
    "ReduceMeanNode",
    "ReduceMaxNode",
    # Other
    "ConcatNode",
    "ClipNode",
    "CastNode",
    "IdentityNode",
    "FullNode",
]
