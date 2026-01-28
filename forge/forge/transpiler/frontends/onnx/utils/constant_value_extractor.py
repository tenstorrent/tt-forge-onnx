# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
constant_value_extractor.py – extract integer values from constant tensors.

Public API
----------
    resolve_constant_tensor_value(tensor_name, tir_graph, graph_proto)

Purpose
-------
Some ONNX ops receive their configuration as an input tensor rather than an
attribute.  For example, the Reshape op receives the target shape as a second
input tensor.  At model-compilation time we need those values as Python
integers – not just the tensor's shape.

This module traces backward from the target tensor, verifies that all sources
are constants/parameters (no runtime inputs), and then evaluates the subgraph
to extract the concrete integer values.

Shared graph utilities (trace_backward, run_subgraph, …) live in
subgraph_utils.py and are also used by shape_finder.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from loguru import logger

from forge.transpiler.core.graph import TIRGraph
from forge.transpiler.frontends.onnx.utils.subgraph_utils import (
    trace_backward,
    run_subgraph,
    topo_sort,
)

__all__ = ["resolve_constant_tensor_value"]


def resolve_constant_tensor_value(
    tensor_name: str,
    tir_graph: TIRGraph,
    graph_proto,
) -> Tuple[bool, Optional[List[int]], Optional[str]]:
    """
    Extract the integer values of *tensor_name* by evaluating its source subgraph.

    Args:
        tensor_name : Name of the tensor whose values are needed.
        tir_graph   : Partially-built TIR graph.
        graph_proto : ONNX graph proto (for initialiser lookups).

    Returns:
        (success, values, error_message)

        success       – True when extraction succeeded.
        values        – Flat list of ints, e.g. [1, 128, 768], or None.
        error_message – Human-readable failure reason, or None on success.
    """
    if tir_graph is None or graph_proto is None:
        return False, None, "tir_graph and graph_proto are required"

    # Trace backward to find all source constants and the computation path
    source_tensors, required_inputs, trace_path, source_type = trace_backward(tensor_name, tir_graph, graph_proto)

    # If any source is a runtime model input, the value is not available
    if required_inputs:
        return (
            False,
            None,
            f"Tensor '{tensor_name}' depends on runtime model input(s) "
            f"{list(required_inputs)}. Values are not available at compile time.",
        )

    if not source_tensors:
        return (
            False,
            None,
            f"Tensor '{tensor_name}' could not be traced to any constants or parameters.",
        )

    # If the tensor is itself a constant, read it directly (no subgraph needed)
    if tensor_name in source_tensors:
        result = _to_int_list(tensor_name, source_tensors[tensor_name])
        if result[0]:
            logger.trace(f"  const-value '{tensor_name}' = {result[1]}  [direct constant]")
        return result

    # Otherwise evaluate the subgraph to compute the value
    if not trace_path:
        return False, None, f"No computation path found for '{tensor_name}'."

    sorted_nodes = topo_sort(trace_path)
    result = run_subgraph(source_tensors, trace_path, tensor_name, {}, tir_graph, "value_eval")
    if result is None:
        return (
            False,
            None,
            f"Subgraph evaluation for '{tensor_name}' did not produce a result.",
        )

    result_vals = _to_int_list(tensor_name, result)
    if result_vals[0]:
        _chain = " → ".join(n.op_type for n in sorted_nodes)
        logger.trace(f"  const-value '{tensor_name}' = {result_vals[1]}" f"  [subgraph: {_chain}]")
    return result_vals


# ---------------------------------------------------------------------------
# Helper unique to constant_value_extractor
# ---------------------------------------------------------------------------


def _to_int_list(
    tensor_name: str,
    tensor: torch.Tensor,
) -> Tuple[bool, Optional[List[int]], Optional[str]]:
    """Flatten *tensor* to a list of Python ints."""
    try:
        values = [int(x) for x in tensor.detach().cpu().numpy().flatten().tolist()]
        return True, values, None
    except Exception as exc:
        return False, None, f"Failed to convert '{tensor_name}' to int list: {exc}"
