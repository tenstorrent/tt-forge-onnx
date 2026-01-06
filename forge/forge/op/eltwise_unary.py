# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union
import torch

from forge._C import DataFormat
from forge._C.ops import OpType
from ..tensor import Tensor, pytorch_dtype_to_forge_dataformat
from .common import ForgeOp as op


def Abs(name: str, operandA: Tensor) -> Tensor:
    """
    Computes the elementwise absolute value of the input tensor.

    The Abs operation returns the magnitude of each element without regard
    to its sign. For real numbers, it returns the non-negative value.
    This operation is idempotent: abs(abs(x)) = abs(x).

    Parameters
    ----------
    name : str
        Name identifier for this operation in the computation graph.
        Use empty string to auto-generate.

    operandA : Tensor
        Input tensor of any shape. All elements will have absolute
        value computed independently.

    Returns
    -------
    Tensor
        Output tensor with same shape as input. Each element is the
        absolute value of the corresponding input element.

    Mathematical Definition
    -----------------------
    abs(x) = |x| = { x if x >= 0, -x if x < 0 }

    See Also
    --------
    forge.op.Relu : ReLU activation (sets negatives to zero)
    forge.op.Sigmoid : Sigmoid activation function
    """

    return op(OpType.Abs, name, operandA).get_tensor()


def Cast(name: str, operandA: Tensor, dtype: Union[torch.dtype, DataFormat]) -> Tensor:
    """
    Cast

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dtype: Union[torch.dtype, DataFormat]
        Specify Torch datatype / Forge DataFormat to convert operandA

    Returns
    -------
    Tensor
        Forge tensor
    """
    dtype = pytorch_dtype_to_forge_dataformat(dtype)
    return op(OpType.Cast, name, operandA, dtype=dtype).get_tensor(out_df=dtype)


def Exp(name: str, operandA: Tensor) -> Tensor:

    """
    Exponent operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Exp, name, operandA).get_tensor()


def Log(name: str, operandA: Tensor) -> Tensor:

    """
    Log operation: natural logarithm of the elements of `operandA`
        yi = log_e(xi) for all xi in operandA tensor

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Log, name, operandA).get_tensor()


def Pow(name: str, operandA: Tensor, exponent: Union[int, float]) -> Tensor:

    """
    Pow operation: `operandA` to the power of `exponent`
        yi = pow(xi, exponent) for all xi in operandA tensor

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Pow, name, operandA, exponent=exponent).get_tensor()


def Identity(name: str, operandA: Tensor, unsqueeze: str = None, unsqueeze_dim: int = None) -> Tensor:

    """
    Identity operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    unsqueeze: str
        If set, the operation returns a new tensor with a dimension of size one inserted at the specified position.

    unsqueeze_dim: int
        The index at where singleton dimenion can be inserted

    Returns
    -------
    Tensor
        Forge tensor
    """

    if unsqueeze == None and unsqueeze_dim == None:
        return op(OpType.Nop, name, operandA).get_tensor()
    else:
        return op(OpType.Nop, name, operandA, unsqueeze=unsqueeze, unsqueeze_dim=unsqueeze_dim).get_tensor()


def Reciprocal(name: str, operandA: Tensor) -> Tensor:

    """
    Reciprocal operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Reciprocal, name, operandA).get_tensor()


def Sqrt(name: str, operandA: Tensor) -> Tensor:

    """
    Square root.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Sqrt, name, operandA).get_tensor()


def Relu(name: str, operandA: Tensor) -> Tensor:
    """
    Applies the Rectified Linear Unit (ReLU) activation function elementwise.

    ReLU sets all negative values to zero while keeping positive values
    unchanged. This introduces non-linearity to neural networks and is one
    of the most commonly used activation functions due to its simplicity
    and effectiveness.

    Parameters
    ----------
    name : str
        Name identifier for this operation in the computation graph.
        Use empty string to auto-generate.

    operandA : Tensor
        Input tensor of any shape. The ReLU function is applied
        independently to each element.

    Returns
    -------
    Tensor
        Output tensor with same shape as input. Each element is
        max(0, x) where x is the corresponding input element.

    Mathematical Definition
    -----------------------
    relu(x) = max(0, x) = { x if x > 0, 0 if x <= 0 }

    See Also
    --------
    forge.op.LeakyRelu : Leaky ReLU with non-zero negative slope
    forge.op.Gelu : Gaussian Error Linear Unit
    forge.op.Sigmoid : Sigmoid activation function
    """

    return op(OpType.Relu, name, operandA).get_tensor()


def LeakyRelu(name: str, operandA: Tensor, alpha: float) -> Tensor:

    """
    Leaky ReLU

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    alpha: float
        Controls the angle of the negative slope

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.LeakyRelu, name, operandA, parameter=alpha).get_tensor()


def Gelu(name: str, operandA: Tensor, approximate="none") -> Tensor:

    """
    GeLU

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    approximate: str
        The gelu approximation algorithm to use: 'none' | 'tanh'.
        Default: 'none'

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Gelu, name, operandA, approximate=approximate).get_tensor()


def Sigmoid(name: str, operandA: Tensor) -> Tensor:
    """
    Sigmoid

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Sigmoid, name, operandA).get_tensor()


def Clip(name: str, operandA: Tensor, min: float, max: float) -> Tensor:
    """
    Clips tensor values between min and max

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    min: float
        Minimum value

    max: float
        Maximum value

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Clip, name, operandA, min=min, max=max).get_tensor()


def Sine(name: str, operandA: Tensor) -> Tensor:
    """
    Elementwise sine

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Sine, name, operandA).get_tensor()


def Atan(name: str, operandA: Tensor) -> Tensor:
    """
    Elementwise arctangent (atan)

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Atan, name, operandA).get_tensor()


def Cosine(name: str, operandA: Tensor) -> Tensor:
    """
    Elementwise cosine

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Cosine, name, operandA).get_tensor()


def Tanh(name: str, operandA: Tensor) -> Tensor:

    """
    Tanh operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Tanh, name, operandA).get_tensor()


def LogicalNot(name: str, operandA: Tensor) -> Tensor:

    """
    Logical not operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.LogicalNot, name, operandA).get_tensor()


def Erf(name: str, operandA: Tensor) -> Tensor:
    """
    Error function (erf)

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.Erf, name, operandA).get_tensor()
