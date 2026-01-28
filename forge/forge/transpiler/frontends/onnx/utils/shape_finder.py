# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
shape_finder.py – resolve unknown tensor dimensions during ONNX → TIR conversion.

Public API
----------
    UnknownDimensionError           – raised when a shape cannot be resolved
    validate_no_unknown_dimensions  – assert a shape is fully concrete
    resolve_unknown_shapes          – resolve all unknown dims in a node's inputs

How resolution works (three steps, tried in order)
----------------------------------------------------
  Step 1 – Shape rules
      Call node.infer_output_shapes() on every node in the backward trace.
      Pure Python formulas, no tensor execution needed.

  Step 2 – Constant subgraph execution
      Build a mini TIR graph with only constants/parameters and run it.
      Skipped when any runtime model input is in the trace.

  Step 3 – Fake-input execution
      Same as step 2 but model inputs are replaced with deterministic dummy
      tensors (zeros/ones). Skipped when any node in the trace has
      VALUE_DEPENDENT shape metadata (fake values would give wrong shapes).

Shared graph utilities (trace_backward, run_subgraph, topo_sort, …) live in
subgraph_utils.py and are also used by constant_value_extractor.py.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger
from onnx import NodeProto

from forge.transpiler.utils.exceptions import ConversionError
from forge.transpiler.core.graph import TIRGraph
from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.shape_eval import ShapeDependency
from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.frontends.onnx.utils.subgraph_utils import (
    trace_backward,
    run_subgraph,
    topo_sort,
    copy_node,
)

__all__ = [
    "UnknownDimensionError",
    "validate_no_unknown_dimensions",
    "resolve_unknown_shapes",
    "resolve_output_shapes",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class UnknownDimensionError(ConversionError):
    """Raised when an unknown tensor dimension cannot be resolved at compile time."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_no_unknown_dimensions(shape: Optional[Tuple], context: str = "") -> None:
    """
    Raise UnknownDimensionError if *shape* contains any unknown dimension.

    An unknown dimension is: None, a symbolic string, or a negative integer.
    Call this before any operation that requires fully-concrete shapes.
    """
    suffix = f"  Context: {context}" if context else ""
    if shape is None:
        raise UnknownDimensionError("", "", f"Shape is None.{suffix}")
    for i, dim in enumerate(shape):
        if dim is None or isinstance(dim, str):
            raise UnknownDimensionError(
                "",
                "",
                f"Unknown dimension at axis {i}: {dim!r}. " f"All dimensions must be concrete integers.{suffix}",
            )
        if isinstance(dim, (int, np.integer)) and dim < 0:
            raise UnknownDimensionError(
                "",
                "",
                f"Negative (dynamic) dimension at axis {i}: {dim}. " f"Must be resolved before use.{suffix}",
            )


def resolve_unknown_shapes(
    node_proto: NodeProto,
    input_tensors: "OrderedDict[str, TensorInfo]",
    tir_graph: TIRGraph,
    graph_proto,
) -> "OrderedDict[str, TensorInfo]":
    """
    Resolve unknown dimensions in every tensor in *input_tensors*.

    Called automatically by the base converter before every op-specific
    converter so that converters can assume shapes are always concrete.

    Returns a new OrderedDict where every TensorInfo has a concrete shape.
    Raises UnknownDimensionError for any tensor that cannot be resolved.
    """
    node_id = node_proto.name or f"{node_proto.op_type}"
    unknown_tensors = [name for name, info in input_tensors.items() if _has_unknown(info.shape)]

    result: OrderedDict[str, TensorInfo] = OrderedDict()

    if not unknown_tensors:
        return {name: info for name, info in input_tensors.items()}

    # Show what we're about to resolve, with current (unknown) shape and dtype
    _pre_lines = [
        f"    '{n}': shape={input_tensors[n].shape}"
        + (f"  dtype={input_tensors[n].torch_dtype}" if input_tensors[n].torch_dtype else "")
        for n in unknown_tensors
    ]
    logger.trace(f"  [{node_id}] Resolving {len(unknown_tensors)} unknown shape(s):\n" + "\n".join(_pre_lines))

    resolved_lines = []
    for name, info in input_tensors.items():
        if not _has_unknown(info.shape):
            result[name] = info  # already concrete – nothing to do
            continue

        shape, technique = _resolve_one_tensor(name, input_tensors, tir_graph, graph_proto, node_proto)
        resolved_lines.append(f"    '{name}': {info.shape} -> {shape}  [via {technique}]")
        result[name] = TensorInfo(info.name, shape, info.onnx_dtype)

    if resolved_lines:
        logger.trace(f"  [{node_id}] Resolved {len(resolved_lines)} unknown shape(s):\n" + "\n".join(resolved_lines))

    return result


# ---------------------------------------------------------------------------
# Single-tensor resolver
# ---------------------------------------------------------------------------


def _recover_shape_from_graph_proto(tensor_name: str, current_shape: Optional[Tuple], graph_proto) -> Optional[Tuple]:
    """
    Try to recover concrete dimensions from the original graph_proto input declaration.

    ONNX shape inference sometimes converts zero-sized dimensions (0) to unknown (None).
    This step recovers those zero-valued dimensions by reading the original
    ``ValueInfoProto`` declarations in ``graph_proto.input``.

    Returns a concrete shape tuple if recovery succeeds, or None otherwise.
    """
    if graph_proto is None or current_shape is None:
        return None

    for input_vi in graph_proto.input:
        if input_vi.name != tensor_name:
            continue
        if not input_vi.type.tensor_type.HasField("shape"):
            return None

        original_dims = input_vi.type.tensor_type.shape.dim
        if len(original_dims) != len(current_shape):
            return None

        recovered = []
        for orig_dim, inferred_dim in zip(original_dims, current_shape):
            if inferred_dim is not None:
                # Already concrete — keep as-is
                recovered.append(inferred_dim)
            elif orig_dim.WhichOneof("value") == "dim_value":
                # Original declaration has a concrete value (including 0)
                recovered.append(orig_dim.dim_value)
            else:
                # Truly unknown (e.g., symbolic dim_param or unset) — cannot recover
                recovered.append(None)

        # Only return if we improved the shape (fewer Nones than before)
        if recovered.count(None) < current_shape.count(None):
            if None not in recovered:
                return tuple(recovered)

    return None


def _resolve_one_tensor(
    tensor_name: str,
    input_tensors: "OrderedDict[str, TensorInfo]",
    tir_graph: TIRGraph,
    graph_proto,
    node_proto: NodeProto,
) -> Tuple:
    """
    Try the three resolution strategies in order and return the first
    concrete shape found.  Raises UnknownDimensionError if all fail.
    """
    current_shape = input_tensors[tensor_name].shape

    # Step 0: recover zero-sized dimensions from the graph_proto input declarations.
    # ONNX shape inference may convert (0,) → (None,) for zero-sized tensors.
    # We recover them before the backward trace to avoid spurious UnknownDimensionErrors.
    shape = _recover_shape_from_graph_proto(tensor_name, current_shape, graph_proto)
    if shape is not None:
        return shape, "graph-proto-recovery"

    # Trace backward to discover all sources and the computation path
    src_tensors, req_inputs, trace_path, src_type = trace_backward(tensor_name, tir_graph, graph_proto)
    trace = {
        "source_tensors": src_tensors,
        "required_inputs": req_inputs,
        "trace_path": trace_path,
        "source_type": src_type,
    }

    # Step 1: shape rules (no execution)
    shape = _try_shape_rules(tensor_name, trace, input_tensors, tir_graph)
    if shape is not None:
        return shape, "shape-rules"

    # Step 2: constant-only subgraph execution
    shape = _try_constant_subgraph(tensor_name, trace, tir_graph)
    if shape is not None:
        return shape, "const-subgraph"

    # Step 3: fake-input subgraph execution (only when safe)
    shape = _try_fake_execution(tensor_name, trace, input_tensors, tir_graph, node_proto)
    if shape is not None:
        return shape, "fake-execution"

    _raise_error(tensor_name, current_shape, trace, node_proto)


# ---------------------------------------------------------------------------
# Step 1 – shape rules (no execution)
# ---------------------------------------------------------------------------


def _try_shape_rules(
    tensor_name: str,
    trace: dict,
    input_tensors: "OrderedDict[str, TensorInfo]",
    tir_graph: TIRGraph,
) -> Optional[Tuple]:
    """
    Propagate shapes forward through the trace using node.infer_output_shapes().
    Each node computes its output shapes from its input shapes via a formula.
    No tensors are executed.
    """
    trace_path: List[TIRNode] = trace["trace_path"]
    if not trace_path:
        return None

    # Seed the shape map from all known sources
    shape_map: Dict[str, Tuple] = {}

    for n, info in input_tensors.items():
        if info.shape is not None:
            shape_map[n] = tuple(info.shape)

    for n, value in trace["source_tensors"].items():
        shape_map[n] = tuple(value.shape)

    for node in trace_path:
        for n, info in {**node.inputs, **node.outputs}.items():
            if info.shape is not None:
                shape_map.setdefault(n, tuple(info.shape))

    # Propagate forward in topological order
    for node in topo_sort(trace_path):
        inferred = node.infer_output_shapes(shape_map)
        if inferred:
            for out_name, out_shape in inferred.items():
                if out_shape is not None:
                    shape_map[out_name] = tuple(out_shape)

    # The target may appear under its original or sanitised name
    sanitized = tir_graph.original_to_sanitized.get(tensor_name, tensor_name)
    result = shape_map.get(tensor_name) or shape_map.get(sanitized)
    if result is None:
        return None
    result_tuple = tuple(result)
    # Only accept the result if every dimension is a concrete integer.
    # If propagation merely echoed back the original unknown shape (e.g. 'unk__0'
    # stays unknown because it comes from a dynamic model input), return None so
    # that Steps 2 and 3 get a chance to resolve it via subgraph execution.
    if _has_unknown(result_tuple):
        return None
    return result_tuple


# ---------------------------------------------------------------------------
# Step 2 – constant-only subgraph execution
# ---------------------------------------------------------------------------


def _try_constant_subgraph(
    tensor_name: str,
    trace: dict,
    tir_graph: TIRGraph,
) -> Optional[Tuple]:
    """
    Build and run a mini TIR graph using only constants/parameters.
    Returns the output shape, or None if any runtime input is involved.
    """
    if trace["source_type"] in {"input", "mixed"}:
        return None  # runtime inputs present – cannot evaluate without them

    source_tensors = trace["source_tensors"]
    if not source_tensors:
        return None

    # If the tensor is itself a direct constant, just read its shape
    if tensor_name in source_tensors:
        return tuple(source_tensors[tensor_name].shape)

    trace_path = trace["trace_path"]
    if not trace_path:
        return None

    result = run_subgraph(source_tensors, trace_path, tensor_name, {}, tir_graph, "shape_const")
    return tuple(result.shape) if result is not None else None


# ---------------------------------------------------------------------------
# Step 3 – fake-input subgraph execution
# ---------------------------------------------------------------------------


def _try_fake_execution(
    tensor_name: str,
    trace: dict,
    input_tensors: "OrderedDict[str, TensorInfo]",
    tir_graph: TIRGraph,
    node_proto: NodeProto,
) -> Optional[Tuple]:
    """
    Same as step 2 but replace model inputs with deterministic dummy tensors.
    Skipped when any node in the trace is VALUE_DEPENDENT, because fake input
    values would cause ops like Reshape to compute incorrect output shapes.
    """
    trace_path: List[TIRNode] = trace["trace_path"]
    required_inputs: Set[str] = trace["required_inputs"]

    if not trace_path:
        return None

    # Safety guard: skip if any node's output shape depends on input *values*
    for node in trace_path:
        meta = getattr(node, "shape_eval_meta", None)
        if meta is not None and meta.dependency == ShapeDependency.VALUE_DEPENDENT:
            return None

    # Create a deterministic fake tensor for every required model input
    fake_inputs: Dict[str, torch.Tensor] = {}
    for inp_name in required_inputs:
        info = _find_tensor_info(inp_name, trace_path, input_tensors)
        if info is None:
            return None
        try:
            fake_inputs[inp_name] = _make_fake_tensor(info, inp_name, node_proto)
        except UnknownDimensionError:
            return None

    result = run_subgraph(trace["source_tensors"], trace_path, tensor_name, fake_inputs, tir_graph, "shape_fake")
    return tuple(result.shape) if result is not None else None


# ---------------------------------------------------------------------------
# Helpers unique to shape_finder
# ---------------------------------------------------------------------------


def _find_tensor_info(
    tensor_name: str,
    trace_path: List[TIRNode],
    input_tensors: "OrderedDict[str, TensorInfo]",
) -> Optional[TensorInfo]:
    """Look up TensorInfo for *tensor_name* in the converter inputs or trace nodes."""
    if tensor_name in input_tensors:
        return input_tensors[tensor_name]
    for node in trace_path:
        if tensor_name in node.inputs:
            return node.inputs[tensor_name]
        if tensor_name in node.outputs:
            return node.outputs[tensor_name]
    return None


def _make_fake_tensor(
    info: TensorInfo,
    name: str,
    node_proto: NodeProto,
) -> torch.Tensor:
    """
    Create a deterministic dummy tensor from TensorInfo metadata.

    Unknown dimensions (None, string symbols, negative ints) are replaced with 1.
    This allows fake-input execution to produce concrete output shapes even when
    the model was exported with dynamic axes (e.g. dynamic batch size).
    The caller only uses the *shape* of the resulting tensor, never its values,
    so the substituted size does not affect correctness.

    bools → zeros, floats/complex → ones, integers → zeros.
    """
    if info.shape is None:
        raise UnknownDimensionError(
            node_proto.op_type,
            node_proto.name or "",
            f"Cannot create fake tensor for '{name}': shape is None.",
        )
    # Replace any unknown dimension (symbolic string, None, negative int) with 1.
    concrete_shape = tuple(d if (isinstance(d, (int, np.integer)) and d >= 0) else 1 for d in info.shape)
    if any(d != orig for d, orig in zip(concrete_shape, info.shape) if orig is not None):
        logger.trace(f"  fake tensor '{name}': {info.shape} -> {concrete_shape}")
    dtype = info.torch_dtype or onnx_dtype_to_torch_dtype(info.onnx_dtype)
    if dtype == torch.bool:
        return torch.zeros(concrete_shape, dtype=dtype)
    if dtype.is_floating_point or dtype.is_complex:
        return torch.ones(concrete_shape, dtype=dtype)
    return torch.zeros(concrete_shape, dtype=dtype)


def _has_unknown(shape: Optional[Tuple]) -> bool:
    """Return True if *shape* is None or contains any unknown dimension."""
    if shape is None:
        return True
    for dim in shape:
        if dim is None or isinstance(dim, str):
            return True
        if isinstance(dim, (int, np.integer)) and dim < 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


def _raise_error(
    tensor_name: str,
    current_shape: Optional[Tuple],
    trace: dict,
    node_proto: NodeProto,
) -> None:
    node_id = node_proto.name or f"{node_proto.op_type}_{id(node_proto)}"

    lines = [
        "",
        "=" * 70,
        "UNRESOLVABLE UNKNOWN DIMENSION",
        "=" * 70,
        f"Node  : {node_id} ({node_proto.op_type})",
        f"Tensor: {tensor_name}",
        f"Shape : {current_shape}",
        "",
        f"Source type    : {trace['source_type']}",
        f"Source tensors : {list(trace['source_tensors'])}",
        f"Model inputs   : {list(trace['required_inputs'])}",
    ]

    if trace["trace_path"]:
        path = " → ".join(n.name for n in trace["trace_path"])
        lines.append(f"Trace path     : {path} → {tensor_name}")

    lines += [""]

    if trace["source_type"] in {"input", "mixed"}:
        lines += [
            "The shape depends on a runtime model input.",
            "All shapes must be concrete at compile time.",
            "",
            "Fix: export the model with fixed (non-dynamic) input shapes,",
            "     or run ONNX shape inference before conversion.",
        ]
    else:
        lines += [
            "All three resolution strategies failed.",
            "This may indicate an unsupported op in the computation path.",
            "Please file an issue with the model and this error.",
        ]

    lines += ["", "=" * 70, ""]

    raise UnknownDimensionError(node_proto.op_type, node_id, "\n".join(lines))


# ---------------------------------------------------------------------------
# Forward output-shape resolution (symmetric counterpart to resolve_unknown_shapes)
# ---------------------------------------------------------------------------


def _seed_output_shape_map(
    input_tensors: "OrderedDict[str, TensorInfo]",
    tir_graph: "TIRGraph",
) -> Dict[str, Tuple]:
    """
    Build the initial *shape_map* from all fully-resolved concrete sources:
    * Already-resolved input tensors (post ``resolve_unknown_shapes``).
    * Constants, computed constants, and params resident in *tir_graph*.

    This is the forward counterpart of the backward shape-map seeding done
    inside ``resolve_unknown_shapes``.
    """
    shape_map: Dict[str, Tuple] = {}
    for name, info in input_tensors.items():
        if info.shape is not None and not _has_unknown(info.shape):
            shape_map[name] = tuple(info.shape)
    for store in (tir_graph.constants, tir_graph.computed_constants, tir_graph.params):
        for name, tensor in store.items():
            if tensor is not None:
                shape_map[name] = tuple(tensor.shape)
    return shape_map


def _try_forward_shape_rules(
    tir_node: "TIRNode",
    unknown_outputs: List[str],
    shape_map: Dict[str, Tuple],
) -> Dict[str, Tuple]:
    """
    Step 1 – formula-based inference via ``node.infer_output_shapes()``.

    Returns a dict of ``{output_name: resolved_shape}`` for every unknown
    output that could be resolved without tensor execution.  Returns an
    empty dict on failure or partial success (only successfully resolved
    names are included).
    """
    inferred = tir_node.infer_output_shapes(shape_map) or {}
    return {
        oname: tuple(oshape)
        for oname in unknown_outputs
        if (oshape := inferred.get(oname)) is not None and not _has_unknown(oshape)
    }


def _build_fake_inputs_for_node(
    tir_node: "TIRNode",
    shape_map: Dict[str, Tuple],
    tir_graph: "TIRGraph",
) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, "torch.Tensor"], bool]:
    """
    Build ``(source_tensors, fake_inputs, can_run)`` for Step 2 execution.

    *source_tensors* : real constant/param tensors that go into ``sub.constants``.
    *fake_inputs*    : synthetic activation tensors built from concrete shapes.
    *can_run*        : False when any activation input has no known concrete shape.
    """
    source_tensors: Dict[str, "torch.Tensor"] = {}
    fake_inputs: Dict[str, "torch.Tensor"] = {}

    for inp_name, inp_info in tir_node.inputs.items():
        const_val = (
            tir_graph.constants.get(inp_name)
            or tir_graph.computed_constants.get(inp_name)
            or tir_graph.params.get(inp_name)
        )
        if const_val is not None:
            source_tensors[inp_name] = const_val
            continue

        inp_shape = shape_map.get(inp_name)
        if inp_shape is None:
            return {}, {}, False

        try:
            concrete = tuple(d if (isinstance(d, (int, np.integer)) and d > 0) else 1 for d in inp_shape)
            dtype = inp_info.torch_dtype or torch.float32
            if dtype == torch.bool:
                fake_inputs[inp_name] = torch.zeros(concrete, dtype=dtype)
            elif dtype.is_floating_point:
                fake_inputs[inp_name] = torch.ones(concrete, dtype=dtype)
            else:
                fake_inputs[inp_name] = torch.zeros(concrete, dtype=dtype)
        except Exception:
            return {}, {}, False

    return source_tensors, fake_inputs, True


def _run_node_for_shapes(
    tir_node: "TIRNode",
    source_tensors: Dict[str, "torch.Tensor"],
    fake_inputs: Dict[str, "torch.Tensor"],
    tir_graph: "TIRGraph",
) -> Dict[str, Tuple]:
    """
    Step 2 – run the TIR node in a minimal sub-graph and return
    ``{output_name: shape}`` for all outputs produced.

    All outputs are captured in a single execution pass (efficient).
    Returns an empty dict on any failure (best-effort, never raises).
    """
    try:
        sub = TIRGraph(
            name=f"output_shape_{tir_node.name}",
            framework=tir_graph.framework,
            log_execution=False,
        )
        for name, value in source_tensors.items():
            sub.constants[name] = value
        sub.add_node(copy_node(tir_node))
        sub.outputs = list(tir_node.outputs.keys())
        outputs = sub.run(inputs=fake_inputs, enable_gc=False)
        return {
            oname: tuple(otensor.shape)
            for oname, otensor in (outputs or {}).items()
            if otensor is not None and hasattr(otensor, "shape")
        }
    except Exception as exc:
        logger.trace(f"  [output_shape] fake-execution failed for '{tir_node.name}'" f" ({tir_node.op_type}): {exc}")
        return {}


def _apply_resolved_shapes(
    resolved: Dict[str, Tuple],
    unknown_outputs: List[str],
    tir_node: "TIRNode",
    output_tensors: "OrderedDict[str, TensorInfo]",
    shape_map: Dict[str, Tuple],
    resolved_lines: List[str],
    strategy: str,
) -> List[str]:
    """
    Mutate *tir_node.outputs* and *output_tensors* in-place for every name in
    *resolved*, propagate concrete shapes into *shape_map*, and append a
    human-readable log line to *resolved_lines*.

    Returns the list of output names that were *not* resolved (still unknown).
    """
    still_unknown = list(unknown_outputs)
    for oname, new_shape in resolved.items():
        if oname not in unknown_outputs or _has_unknown(new_shape):
            continue
        old_info = tir_node.outputs[oname]
        tir_node.outputs[oname] = TensorInfo(oname, new_shape, old_info.onnx_dtype)
        shape_map[oname] = new_shape
        if oname in output_tensors and _has_unknown(output_tensors[oname].shape):
            old_out = output_tensors[oname]
            output_tensors[oname] = TensorInfo(old_out.name, new_shape, old_out.onnx_dtype)
            resolved_lines.append(f"    '{oname}': {old_out.shape} → {new_shape}  [{strategy}]")
        if oname in still_unknown:
            still_unknown.remove(oname)
    return still_unknown


def resolve_output_shapes(
    node_proto: NodeProto,
    tir_nodes: List["TIRNode"],
    input_tensors: "OrderedDict[str, TensorInfo]",
    output_tensors: "OrderedDict[str, TensorInfo]",
    tir_graph: "TIRGraph",
) -> None:
    """
    Resolve unknown dimensions in *output_tensors* using forward propagation.

    Called **after** the converter returns TIR nodes but **before** they are
    sanitized and added to the graph.  At this point:

    * TIR node names still use original ONNX names.
    * *input_tensors* are fully concrete (post ``resolve_unknown_shapes``).
    * *output_tensors* may carry symbolic shapes from ``value_info_map``.

    Resolution strategies (applied in order for each node with unknown outputs):

    Step 1 — Shape rules
        ``node.infer_output_shapes(shape_map)`` seeded with concrete inputs.
        Formula-based, zero tensor execution.

    Step 2 — Fake-input subgraph execution
        Build fake activation tensors from concrete shapes, run the node as a
        mini sub-graph, read shapes from the produced tensors.
        Skipped for ``VALUE_DEPENDENT`` nodes (fake values would yield wrong
        shapes, e.g. Reshape reads the *values* of its shape input).

    Mutates *output_tensors* **and** each TIR node's ``outputs`` dict in-place
    so that downstream ONNX nodes and the ``value_info_map`` propagation both
    see concrete shapes.
    """
    if not any(_has_unknown(info.shape) for info in output_tensors.values()):
        return  # Fast path — nothing symbolic to resolve
    if not tir_nodes:
        return

    node_id = node_proto.name or node_proto.op_type
    logger.trace(f"[{node_id}] Resolving unknown output shapes ...")

    shape_map = _seed_output_shape_map(input_tensors, tir_graph)
    resolved_lines: List[str] = []

    for tir_node in topo_sort(tir_nodes):
        # Constant/Full nodes already have concrete shapes — propagate forward.
        if tir_node.op_type in ("Full", "Constant"):
            for oname, oinfo in tir_node.outputs.items():
                if oinfo.shape and not _has_unknown(oinfo.shape):
                    shape_map[oname] = tuple(oinfo.shape)
            continue

        # Propagate already-known outputs and collect unknowns.
        unknown_out = []
        for oname, oinfo in tir_node.outputs.items():
            if oinfo.shape and not _has_unknown(oinfo.shape):
                shape_map[oname] = tuple(oinfo.shape)
            else:
                unknown_out.append(oname)

        if not unknown_out:
            continue  # All outputs concrete — nothing to do.

        # ── Step 1: formula-based shape rules (no tensor execution) ──────────
        step1 = _try_forward_shape_rules(tir_node, unknown_out, shape_map)
        if step1:
            unknown_out = _apply_resolved_shapes(
                step1, unknown_out, tir_node, output_tensors, shape_map, resolved_lines, "shape-rules"
            )

        if not unknown_out:
            continue  # All resolved by Step 1.

        # ── Step 2: fake-input subgraph execution ────────────────────────────
        # Skip VALUE_DEPENDENT nodes — fake tensor values would produce
        # incorrect shapes for ops whose output shape depends on input values
        # (e.g. Reshape reads shape-input values, not just shapes).
        meta = getattr(tir_node, "shape_eval_meta", None)
        if meta is not None and meta.dependency == ShapeDependency.VALUE_DEPENDENT:
            logger.trace(
                f"  [{node_id}/{tir_node.op_type}] Skipping fake-execution"
                f" (VALUE_DEPENDENT) — {unknown_out} remain unresolved"
            )
            continue

        source_tensors_, fake_inputs_, can_run = _build_fake_inputs_for_node(tir_node, shape_map, tir_graph)
        if not can_run:
            logger.trace(
                f"  [{node_id}/{tir_node.op_type}] Cannot build fake inputs" f" — {unknown_out} remain unresolved"
            )
            continue

        step2 = _run_node_for_shapes(tir_node, source_tensors_, fake_inputs_, tir_graph)
        if step2:
            _apply_resolved_shapes(
                step2, unknown_out, tir_node, output_tensors, shape_map, resolved_lines, "fake-execution"
            )

    if resolved_lines:
        logger.trace(f"[{node_id}] Resolved {len(resolved_lines)} output shape(s):\n" + "\n".join(resolved_lines))
