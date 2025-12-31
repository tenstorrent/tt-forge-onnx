# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unified result type for all ONNX converter methods.

This module provides a consistent, type-safe interface for converter results, supporting
both TIR nodes (normal operations) and constants (special case). Converters can return
either a list of TIR nodes for normal operations or a ConstantResult for constant values.

Why two result types:
- Normal operations produce TIR nodes that are added to the graph
- Constant operations produce values that are stored directly in graph.constants
- This separation allows the engine to handle constants specially without checking types
"""
from dataclasses import dataclass
from typing import List, Union, TYPE_CHECKING, TypeGuard, Callable, TypeVar, overload
import torch

if TYPE_CHECKING:
    from forge.transpiler.core.node import TIRNode

# Type variable for generic result handling
T = TypeVar("T")


@dataclass(frozen=True)
class ConstantResult:
    """
    Result containing a constant value from ConstantConverter.

    This is a special result type that indicates the converter produced
    a constant value rather than TIR nodes.

    Attributes:
        value: The constant tensor value
        output_name: The output name for this constant
    """

    value: torch.Tensor
    output_name: str

    def __post_init__(self):
        """Validate attributes after initialization."""
        if not isinstance(self.value, torch.Tensor):
            raise TypeError(f"ConstantResult.value must be torch.Tensor, got {type(self.value)}")
        if not isinstance(self.output_name, str):
            raise TypeError(f"ConstantResult.output_name must be str, got {type(self.output_name)}")

    def __repr__(self):
        return (
            f"ConstantResult(output_name='{self.output_name}', " f"shape={self.value.shape}, dtype={self.value.dtype})"
        )


# Type alias for converter results
# Converters can return either a list of TIR nodes (normal) or ConstantResult (constants)
ConverterResult = Union[List["TIRNode"], ConstantResult]


def is_constant_result(result: ConverterResult) -> TypeGuard[ConstantResult]:
    """
    Type guard to check if result is a ConstantResult.

    This enables proper type narrowing in type checkers (mypy, pyright).

    Args:
        result: Converter result to check

    Returns:
        True if result is ConstantResult, False if it's a list of TIR nodes
    """
    return isinstance(result, ConstantResult)


def is_tir_nodes_result(result: ConverterResult) -> TypeGuard[List["TIRNode"]]:
    """
    Type guard to check if result is a list of TIR nodes.

    Args:
        result: Converter result to check

    Returns:
        True if result is a list of TIR nodes, False if it's ConstantResult
    """
    return not isinstance(result, ConstantResult)


@overload
def match_result(
    result: ConstantResult, on_constant: Callable[[ConstantResult], T], on_nodes: Callable[[List["TIRNode"]], T]
) -> T:
    ...


@overload
def match_result(
    result: List["TIRNode"], on_constant: Callable[[ConstantResult], T], on_nodes: Callable[[List["TIRNode"]], T]
) -> T:
    ...


def match_result(
    result: ConverterResult, on_constant: Callable[[ConstantResult], T], on_nodes: Callable[[List["TIRNode"]], T]
) -> T:
    """
    Pattern matching for converter results (functional approach).

    This provides a clean, functional way to handle different result types
    without isinstance checks scattered throughout the code.

    Args:
        result: Converter result to match
        on_constant: Handler function for ConstantResult
        on_nodes: Handler function for List[TIRNode]

    Returns:
        Result of the appropriate handler function

    Example:
        >>> result = ConstantResult(value=tensor, output_name="const")
        >>> match_result(
        ...     result,
        ...     on_constant=lambda c: f"Constant: {c.output_name}",
        ...     on_nodes=lambda nodes: f"Nodes: {len(nodes)}"
        ... )
        'Constant: const'
    """
    if is_constant_result(result):
        return on_constant(result)
    else:
        return on_nodes(result)


def get_tir_nodes(result: ConverterResult) -> List["TIRNode"]:
    """
    Extract TIR nodes from converter result.

    Args:
        result: Converter result

    Returns:
        List of TIR nodes (empty list for constants)
    """
    if is_constant_result(result):
        return []
    return result


def get_constant_info(result: ConverterResult) -> tuple[torch.Tensor, str] | None:
    """
    Extract constant value and output name from converter result.

    Args:
        result: Converter result

    Returns:
        Tuple of (value, output_name) if constant, None otherwise
    """
    if is_constant_result(result):
        return (result.value, result.output_name)
    return None


def unwrap_constant(result: ConverterResult) -> ConstantResult:
    """
    Unwrap a ConstantResult, raising an error if it's not a constant.

    Args:
        result: Converter result

    Returns:
        ConstantResult instance

    Raises:
        TypeError: If result is not a ConstantResult
    """
    if not is_constant_result(result):
        raise TypeError(f"Expected ConstantResult, got {type(result)}")
    return result


def unwrap_tir_nodes(result: ConverterResult) -> List["TIRNode"]:
    """
    Unwrap a list of TIR nodes, raising an error if it's a constant.

    Args:
        result: Converter result

    Returns:
        List of TIR nodes

    Raises:
        TypeError: If result is not a list of TIR nodes
    """
    if is_constant_result(result):
        raise TypeError(f"Expected List[TIRNode], got ConstantResult")
    return result
