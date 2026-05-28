# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Input validation and graceful handling utilities for ONNX converters.
"""
from typing import Dict, List, Any, Optional, Tuple
from onnx import NodeProto
from loguru import logger


class ConverterValidationError(Exception):
    """
    Raised for validation errors that occur inside ONNX converter functions.

    This is intentionally distinct from ``forge.transpiler.utils.exceptions.ValidationError``
    (which represents transpiler-level validation failures) to make the origin
    of each error unambiguous.
    """


def validate_attributes(
    node_proto: NodeProto,
    attrs: Dict[str, Any],
    required_attrs: Optional[List[str]] = None,
    attr_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate node attributes.

    Args:
        node_proto: ONNX node proto
        attrs: Extracted attributes dictionary
        required_attrs: List of required attribute names
        attr_ranges: Dictionary mapping attribute names to (min, max) value ranges

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required attributes
    if required_attrs:
        for attr_name in required_attrs:
            if attr_name not in attrs or attrs[attr_name] is None:
                return False, (
                    f"Node {node_proto.name or node_proto.op_type} requires attribute '{attr_name}' "
                    f"but it was not found or is None"
                )

    # Check attribute value ranges
    if attr_ranges:
        for attr_name, (min_val, max_val) in attr_ranges.items():
            if attr_name in attrs and attrs[attr_name] is not None:
                attr_val = attrs[attr_name]
                if isinstance(attr_val, (int, float)):
                    if attr_val < min_val or attr_val > max_val:
                        return False, (
                            f"Node {node_proto.name or node_proto.op_type} attribute '{attr_name}' "
                            f"has value {attr_val}, but must be in range [{min_val}, {max_val}]"
                        )

    return True, None


def validate_constant_input(
    node_proto: NodeProto,
    input_index: int,
    graph_proto,
    input_name: str = None,
    tir_graph=None,
    to_python: bool = True,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Validate and extract a constant value from an op's input tensor.

    Lookup order:
        1. TIR graph stores (params, constants, computed_constants) — covers
           ONNX initializers already loaded and Constant/ConstantOfShape outputs
           created during transpilation.
        2. Raw ONNX graph initializers — fallback for cases where the TIR graph
           has not yet been populated (e.g. early validation passes).

    Args:
        node_proto:   ONNX node proto (used for error messages).
        input_index:  Index of the input to validate.
        graph_proto:  ONNX graph proto (may be None; used for initializer fallback).
        input_name:   Explicit input tensor name; defaults to
                      node_proto.input[input_index].
        tir_graph:    Partially-built TIR graph; searched first when provided.
        to_python:    If ``True`` (default) the returned ``constant_value`` is a
                      Python scalar or list.  If ``False`` it is a
                      ``numpy.ndarray``, avoiding an extra Python→numpy
                      conversion round-trip in callers that need an array
                      directly (e.g. ``GatherConverter``).

    Returns:
        (is_valid, constant_value, error_message)
        is_valid        – True when a constant value was found, or when the
                          input is optional and was not provided.
        constant_value  – Python scalar or list (``to_python=True``) or
                          ``numpy.ndarray`` (``to_python=False``); None when
                          optional or on failure.
        error_message   – Human-readable reason on failure, None on success.
    """
    if input_index >= len(node_proto.input):
        return True, None, None  # optional input not provided

    input_name = input_name or node_proto.input[input_index]
    node_id = node_proto.name or node_proto.op_type

    # ── Step 1: TIR graph stores ──────────────────────────────────────────────
    # NOTE: explicit is-None checks are required here — Python's `or` operator
    # evaluates truthiness, which raises RuntimeError for multi-element tensors
    # (numpy arrays or torch.Tensors with more than one element).
    if tir_graph is not None:
        tensor = tir_graph.params.get(input_name)
        if tensor is None:
            tensor = tir_graph.constants.get(input_name)
        if tensor is None and hasattr(tir_graph, "computed_constants"):
            tensor = tir_graph.computed_constants.get(input_name)
        if tensor is not None:
            return True, _tensor_to_python(tensor, to_python=to_python), None

    # ── Step 2: raw ONNX initializers ────────────────────────────────────────
    if graph_proto is None:
        return (
            False,
            None,
            (f"Node '{node_id}' requires constant input '{input_name}' " f"but graph_proto is not available."),
        )

    for init in graph_proto.initializer:
        if init.name == input_name:
            try:
                from onnx import numpy_helper

                return True, _tensor_to_python(numpy_helper.to_array(init), to_python=to_python), None
            except Exception as exc:
                return False, None, (f"Node '{node_id}': failed to read initializer '{input_name}': {exc}")

    return (
        False,
        None,
        (
            f"Node '{node_id}' requires constant input '{input_name}' "
            f"but it was not found in any compile-time constant store. "
            f"Dynamic inputs are not supported."
        ),
    )


def _tensor_to_python(tensor, to_python: bool = True) -> Any:
    """
    Convert a numpy array or torch.Tensor to a Python value or keep as numpy.

    Args:
        tensor:    Input value — ``torch.Tensor``, ``numpy.ndarray``, or an
                   already-converted Python scalar / list.
        to_python: If ``True`` (default) convert to a Python scalar (for
                   single-element arrays) or a Python list (for multi-element
                   arrays).  If ``False`` return the intermediate
                   ``numpy.ndarray`` directly, avoiding a potentially expensive
                   round-trip back to numpy when the caller needs an array.

    Returns:
        * ``to_python=True``  → Python ``int`` / ``float`` (scalar) or ``list``
        * ``to_python=False`` → ``numpy.ndarray``
        * Passthrough for values that are already plain Python objects.
    """
    import numpy as np
    import torch as _torch

    if isinstance(tensor, _torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        return tensor  # already a Python scalar/list

    if not to_python:
        return arr
    return arr.item() if arr.size == 1 else arr.tolist()


def handle_validation_error(node_proto: NodeProto, error_msg: str, strict: bool = False) -> bool:
    """
    Handle validation errors gracefully.

    Args:
        node_proto: ONNX node proto
        error_msg: Error message
        strict: If True, raise exception. If False, log warning and return False.

    Returns:
        True if error was handled gracefully, False if should skip this node
    """
    if strict:
        raise ConverterValidationError(f"{node_proto.op_type} (node: {node_proto.name}): {error_msg}")
    else:
        logger.warning(f"{node_proto.op_type} (node: {node_proto.name}): {error_msg}")
        return False
