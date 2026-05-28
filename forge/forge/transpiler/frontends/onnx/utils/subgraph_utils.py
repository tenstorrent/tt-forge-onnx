# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
subgraph_utils.py – shared graph-tracing and subgraph-execution helpers.

Internal module used by shape_finder.py and constant_value_extractor.py.

Functions
---------
    trace_backward       – BFS backward trace to collect sources and compute path
    run_subgraph         – build a mini TIRGraph and execute it
    topo_sort            – topological sort of TIRNodes (Kahn's algorithm)
    copy_node            – shallow-copy a TIRNode for use in a subgraph
    get_producer         – find the producer node name for a tensor
    lookup_constant      – look up a constant/parameter tensor value
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Dict, List, Optional, Set, Tuple

import torch
from loguru import logger
from onnx import numpy_helper

from forge.transpiler.core.graph import TIRGraph
from forge.transpiler.core.node import TIRNode


# ---------------------------------------------------------------------------
# Backward trace
# ---------------------------------------------------------------------------


def trace_backward(
    tensor_name: str,
    tir_graph: TIRGraph,
    graph_proto,
) -> Tuple[Dict[str, torch.Tensor], Set[str], List[TIRNode], str]:
    """
    BFS backward through the TIR graph starting from *tensor_name*.

    Every leaf tensor encountered is classified as one of:
    - compile-time constant  → added to *source_tensors* with its value
    - runtime model input    → added to *required_inputs*
                               A tensor is a runtime input ONLY when its name
                               appears in tir_graph.inputs or
                               tir_graph.original_inputs.  Unresolved leaves
                               that are not graph inputs are logged as warnings
                               but not treated as runtime inputs (this avoids
                               falsely blocking constant-subgraph evaluation).

    Returns:
        source_tensors  – {name: tensor} for all compile-time constants/params
        required_inputs – set of names that are genuine runtime model inputs
        trace_path      – TIRNode objects along the computation path
                          (discovery order; callers should topo-sort before use)
        source_type     – 'constant' | 'input' | 'mixed' | 'unknown'
    """
    # Build a fast lookup set for graph inputs (both sanitised and original names)
    graph_input_names: Set[str] = set(tir_graph.inputs) | set(tir_graph.original_inputs)

    source_tensors: Dict[str, torch.Tensor] = {}
    required_inputs: Set[str] = set()
    trace_path: List[TIRNode] = []
    visited: Set[str] = set()

    queue = deque([tensor_name])

    while queue:
        name = queue.popleft()
        if name in visited:
            continue
        visited.add(name)

        producer_name = get_producer(name, tir_graph)

        if producer_name is None:
            # Leaf node: classify as constant, model input, or unknown
            value = lookup_constant(name, tir_graph, graph_proto)
            if value is not None:
                source_tensors[name] = value
            elif name in graph_input_names:
                required_inputs.add(name)
            else:
                # The tensor is neither a known constant nor a registered graph
                # input.  This can happen for tensors that are produced by a node
                # not yet added to the TIR graph (e.g. a later ONNX op in the
                # topo-sorted list).  Log and skip rather than misclassify.
                logger.trace(f"  leaf '{name}': not a constant/graph-input, skipping")
            continue

        if producer_name == "constant":
            # Synthetic marker set by some converters directly in tir_graph.constants
            value = tir_graph.constants.get(name)
            if value is not None:
                source_tensors[name] = value
            continue

        node = tir_graph.get_node_by_name(producer_name)
        if node is None:
            continue

        if node not in trace_path:
            trace_path.append(node)

        for inp in node.inputs:
            if inp not in visited:
                queue.append(inp)

    # Classify overall source type
    if required_inputs and source_tensors:
        source_type = "mixed"
    elif required_inputs:
        source_type = "input"
    elif source_tensors:
        source_type = "constant"
    else:
        source_type = "unknown"

    return source_tensors, required_inputs, trace_path, source_type


# ---------------------------------------------------------------------------
# Subgraph execution
# ---------------------------------------------------------------------------


def run_subgraph(
    source_tensors: Dict[str, torch.Tensor],
    trace_path: List[TIRNode],
    target_tensor: str,
    extra_inputs: Dict[str, torch.Tensor],
    tir_graph: TIRGraph,
    label: str = "shape_eval",
) -> Optional[torch.Tensor]:
    """
    Build a temporary TIRGraph from *source_tensors* + *trace_path*, run it,
    and return the output tensor for *target_tensor*.

    Args:
        source_tensors – constants/parameters to seed the subgraph
        trace_path     – nodes to add to the subgraph
        target_tensor  – name of the tensor to extract from the outputs
        extra_inputs   – additional input tensors (e.g. fake model inputs)
        tir_graph      – the original graph (for name-sanitisation lookup)
        label          – prefix used for the temporary graph name (for debugging)

    Returns:
        The output torch.Tensor, or None if anything fails.
    """
    try:
        sub = TIRGraph(name=f"{label}_{target_tensor}", framework=tir_graph.framework, log_execution=False)

        for name, value in source_tensors.items():
            sub.constants[name] = value

        for node in topo_sort(trace_path):
            sub.add_node(copy_node(node))

        output_key = tir_graph.original_to_sanitized.get(target_tensor, target_tensor)
        sub.outputs = [output_key]

        # Log the compact subgraph structure (op-chain only, no per-node execution)
        if trace_path:
            _sorted = topo_sort(trace_path)
            _chain = " → ".join(n.op_type for n in _sorted)
            logger.trace(f"  [{label}] {len(_sorted)} node(s): {_chain}")

        outputs = sub.run(inputs=extra_inputs, enable_gc=False)
        return outputs.get(target_tensor) or outputs.get(output_key)

    except Exception as exc:
        logger.trace(f"  [{label}] failed for '{target_tensor}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------


def topo_sort(nodes: List[TIRNode]) -> List[TIRNode]:
    """
    Topological sort of *nodes* using Kahn's algorithm.

    Nodes whose inputs are not produced by other nodes in the list are treated
    as roots (in-degree 0) and scheduled first.
    """
    name_to_node = {n.name: n for n in nodes}
    in_degree = {n.name: 0 for n in nodes}
    dependents: Dict[str, List[str]] = {n.name: [] for n in nodes}

    for node in nodes:
        for inp in node.inputs:
            for other in nodes:
                if inp in other.outputs:
                    dependents[other.name].append(node.name)
                    in_degree[node.name] += 1
                    break

    queue = deque(name for name, deg in in_degree.items() if deg == 0)
    sorted_nodes: List[TIRNode] = []

    while queue:
        name = queue.popleft()
        sorted_nodes.append(name_to_node[name])
        for dep in dependents[name]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(sorted_nodes) != len(nodes):
        logger.warning(
            f"Topo-sort incomplete ({len(sorted_nodes)}/{len(nodes)}). " "Possible cycle or missing dependency."
        )

    return sorted_nodes


def copy_node(node: TIRNode) -> TIRNode:
    """Shallow-copy a TIRNode for use in a temporary subgraph."""
    return node.__class__(
        name=node.name,
        op_type=node.op_type,
        inputs=OrderedDict(node.inputs),
        outputs=OrderedDict(node.outputs),
        attrs=node.attrs.copy(),
        forge_op_name=node.forge_op_name,
        src_layer=node.src_layer,
    )


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_producer(tensor_name: str, tir_graph: TIRGraph) -> Optional[str]:
    """
    Return the name of the TIR node that produces *tensor_name*, or None if
    the tensor is a leaf (constant, parameter, or model input).

    Handles both original and sanitised tensor names.
    """
    key = tensor_name
    if key not in tir_graph.producer_map and tir_graph.original_to_sanitized:
        key = tir_graph.original_to_sanitized.get(tensor_name, tensor_name)
    return tir_graph.producer_map.get(key)


def lookup_constant(
    tensor_name: str,
    tir_graph: TIRGraph,
    graph_proto,
) -> Optional[torch.Tensor]:
    """
    Return the tensor value for *tensor_name* if it is a constant or model
    parameter, otherwise return None.

    Search order:
        1. tir_graph.constants          – non-trainable values from ONNX initialisers
        2. tir_graph.params             – trainable weights
        3. tir_graph.computed_constants – produced during transpilation
                                          (ONNX Constant ops, ConstantOfShape, etc.);
                                          not in model.graph.initializer
        4. ONNX graph initialisers      – raw fallback when TIR stores not yet populated
    """
    if tensor_name in tir_graph.constants:
        return tir_graph.constants[tensor_name]
    if tensor_name in tir_graph.params:
        return tir_graph.params[tensor_name]
    if hasattr(tir_graph, "computed_constants") and tensor_name in tir_graph.computed_constants:
        return tir_graph.computed_constants[tensor_name]
    if graph_proto is not None:
        for init in graph_proto.initializer:
            if init.name == tensor_name:
                return torch.from_numpy(numpy_helper.to_array(init))
    return None
