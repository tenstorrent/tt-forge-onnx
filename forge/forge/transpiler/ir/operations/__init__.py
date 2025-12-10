"""
Operations package for TIR nodes.
Imports all operations to register them.
"""
# Import all operations to register them
from forge.transpiler.ir.operations.arithmetic import (
    AddNode,
    SubNode,
    MulNode,
    DivNode,
    MatMulNode,
)
from forge.transpiler.ir.operations.activation import (
    ReluNode,
    SigmoidNode,
    TanhNode,
    SoftmaxNode,
    LogSoftmaxNode,
    LeakyReluNode,
    DropoutNode,
)
from forge.transpiler.ir.operations.conv import (
    Conv1dNode,
    Conv2dNode,
    Conv3dNode
)
from forge.transpiler.ir.operations.pooling import (
    MaxPool1dNode,
    MaxPool2dNode,
    MaxPool3dNode,
    AveragePool1dNode,
    AveragePool2dNode,
    AveragePool3dNode,
    GlobalAveragePoolNode,
)
from forge.transpiler.ir.operations.normalization import (
    BatchNormalizationNode,
)
from forge.transpiler.ir.operations.shape import (
    ReshapeNode,
    TransposeNode,
    SqueezeNode,
    UnsqueezeNode,
)
from forge.transpiler.ir.operations.reduction import (
    ReduceSumNode,
    ReduceMeanNode,
    ReduceMaxNode,
)
from forge.transpiler.ir.operations.other import (
    ConcatNode,
    ClipNode,
    CastNode,
    IdentityNode,
    FullNode,
)

__all__ = [
    # Arithmetic
    'AddNode',
    'SubNode',
    'MulNode',
    'DivNode',
    'MatMulNode',
    # Convolution
    'Conv1dNode',
    'Conv2dNode',
    'Conv3dNode',
    # Activation
    'ReluNode',
    'SigmoidNode',
    'TanhNode',
    'SoftmaxNode',
    'LogSoftmaxNode',
    'LeakyReluNode',
    'DropoutNode',
    # Pooling
    'MaxPool1dNode',
    'MaxPool2dNode',
    'MaxPool3dNode',
    'AveragePool1dNode',
    'AveragePool2dNode',
    'AveragePool3dNode',
    'GlobalAveragePoolNode',
    # Normalization
    'BatchNormalizationNode',
    # Shape
    'ReshapeNode',
    'TransposeNode',
    'SqueezeNode',
    'UnsqueezeNode',
    # Reduction
    'ReduceSumNode',
    'ReduceMeanNode',
    'ReduceMaxNode',
    # Other
    'ConcatNode',
    'ClipNode',
    'CastNode',
    'IdentityNode',
    'FullNode',
]

