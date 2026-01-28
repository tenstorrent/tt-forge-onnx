# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared trace-level formatter for the ONNX → TIR transpiler.

``TranspilerLogger`` is the single place that owns all multi-line log
formatting for:

  * ONNX → TIR node conversion  (used by ``engine.py``)
  * TIRGraph execution & debug validation  (used by ``graph.py``)

Keeping formatting isolated here means the core engine and graph code stay
free of f-string clutter and are easier to read and maintain.
"""
from loguru import logger


class TranspilerLogger:
    """
    Static helper class for formatting and emitting TRACE/INFO-level log
    messages during ONNX → TIR transpilation and graph execution.

    All public methods are either:
    * **format_*** — pure formatters that return a string.
    * **build_*** — helpers that build lists of detail lines.
    * **emit_*** — thin wrappers that call ``logger.trace`` / ``logger.info``.

    Nothing here mutates transpiler state.
    """

    # ------------------------------------------------------------------
    # Shared separators
    # ------------------------------------------------------------------

    _SEP = "=" * 68  # heavy boundary — ONNX node open/close
    _SEP2 = "-" * 50  # lighter divider — between ONNX section and TIR section
    _vSEP = "=" * 68  # VALIDATE block boundary
    _vSEP2 = "-" * 50  # VALIDATE inner divider

    # ══════════════════════════════════════════════════════════════════
    # Section A — ONNX → TIR conversion  (engine.py)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _fmt_attr_val(v) -> str:
        """Truncate large attribute values (e.g. numpy arrays) for readability."""
        s = str(v)
        return s[:100] + "  ...(truncated)" if len(s) > 100 else s

    @staticmethod
    def _conv_tensor_line(idx: int, name: str, tensor_info) -> str:
        """Numbered tensor line used in the ONNX node header."""
        shape = tuple(tensor_info.shape) if tensor_info and tensor_info.shape else "?"
        dtype = tensor_info.torch_dtype if tensor_info else "?"
        return f"    {idx + 1}) '{name}'  shape={shape}  dtype={dtype}"

    @staticmethod
    def _tir_tensor_line(idx: int, name: str, tensor_info) -> str:
        """Numbered TIR tensor line (indented deeper than ONNX lines)."""
        shape = tuple(tensor_info.shape) if tensor_info and tensor_info.shape else "?"
        dtype = tensor_info.torch_dtype if tensor_info else "?"
        return f"       {idx + 1}) '{name}'  shape={shape}  dtype={dtype}"

    @classmethod
    def format_onnx_node_section(
        cls, node_proto, input_tensors, output_tensors, attrs: dict, i: int, total_nodes: int
    ) -> str:
        """
        Build the header block for a single ONNX node being converted.

        Shows progress counter, op-type, node name, numbered ONNX
        inputs/outputs with shape + dtype, and node attributes (truncated).

        Args:
            node_proto:      ONNX NodeProto being converted.
            input_tensors:   ``OrderedDict[name, TensorInfo]`` for inputs.
            output_tensors:  ``OrderedDict[name, TensorInfo]`` for outputs.
            attrs:           Extracted attribute dict (name → value).
            i:               0-based node index in the graph.
            total_nodes:     Total ONNX nodes in the graph.

        Returns:
            Formatted multi-line string (no trailing newline).
        """
        op_type = node_proto.op_type
        node_name = node_proto.name or f"{op_type}_{i}"

        in_lines = [cls._conv_tensor_line(idx, n, input_tensors.get(n)) for idx, n in enumerate(node_proto.input)]
        out_lines = [cls._conv_tensor_line(idx, n, output_tensors.get(n)) for idx, n in enumerate(node_proto.output)]
        attr_lines = [f"    {k} = {cls._fmt_attr_val(v)}" for k, v in attrs.items()]

        return (
            f"[{i + 1}/{total_nodes}] Converting ONNX {op_type} -> TIR  |  node: '{node_name}'\n"
            f"{cls._SEP}\n"
            f"  Inputs  ({len(in_lines)}):\n"
            + ("\n".join(in_lines) if in_lines else "    (none)")
            + f"\n  Outputs ({len(out_lines)}):\n"
            + ("\n".join(out_lines) if out_lines else "    (none)")
            + (
                f"\n  Attributes ({len(attr_lines)}):\n" + "\n".join(attr_lines)
                if attr_lines
                else "\n  Attributes: (none)"
            )
        )

    @staticmethod
    def format_constant_result(original_output_name: str, clean_name: str, value) -> str:
        """
        Format log entry for a ConstantResult (entire ONNX node output is a
        compile-time constant, e.g. ``Constant``, ``ConstantOfShape``).
        """
        return (
            f"  -> CONSTANT  output='{original_output_name}'\n"
            f"     sanitized : '{clean_name}'\n"
            f"     shape     : {tuple(value.shape)}\n"
            f"     dtype     : {value.dtype}"
        )

    @staticmethod
    def format_mapped_summary(op_type: str, tir_nodes: list) -> str:
        """One-line summary: how many TIR nodes an ONNX op was mapped to."""
        n_ops = sum(1 for n in tir_nodes if n.op_type != "Full")
        n_consts = sum(1 for n in tir_nodes if n.op_type == "Full")
        detail = f"  [{n_ops} op(s) + {n_consts} computed const(s)]" if n_consts else f"  [{n_ops} op(s)]"
        return f"  ONNX '{op_type}' -> {len(tir_nodes)} TIR node(s){detail}"

    @staticmethod
    def format_full_const(tir_node, original_output_name: str, sanitized_output_name: str, constant_value) -> str:
        """Format log entry for a FullNode promoted to a computed constant."""
        return (
            f"  -> FULL->CONST '{tir_node.name}'\n"
            f"     orig : '{original_output_name}'\n"
            f"     san  : '{sanitized_output_name}'\n"
            f"     shape: {tuple(constant_value.shape)}  dtype: {constant_value.dtype}"
        )

    @classmethod
    def format_tir_node_detail(cls, tir_node, idx: int) -> str:
        """
        Full detail block for a single live TIR node (inputs, outputs, attributes).

        Args:
            tir_node: TIRNode to describe.
            idx:      1-based display index among TIR nodes produced for this ONNX op.
        """
        in_lines = [cls._tir_tensor_line(j, n, ti) for j, (n, ti) in enumerate(tir_node.inputs.items())]
        out_lines = [cls._tir_tensor_line(j, n, ti) for j, (n, ti) in enumerate(tir_node.outputs.items())]
        attr_lines = [f"       {k} = {v}" for k, v in tir_node.attrs.items()] if tir_node.attrs else []

        return (
            f"  [{idx}] TIR {tir_node.op_type}: '{tir_node.name}'\n"
            + f"     Inputs  ({len(in_lines)}):\n"
            + ("\n".join(in_lines) if in_lines else "       (none)")
            + f"\n     Outputs ({len(out_lines)}):\n"
            + ("\n".join(out_lines) if out_lines else "       (none)")
            + (f"\n     Attributes:\n" + "\n".join(attr_lines) if attr_lines else "\n     Attributes: (none)")
        )

    @classmethod
    def emit_node_trace(cls, onnx_section: str, log_lines: list) -> None:
        """
        Assemble and emit one ``logger.trace`` call for a converted ONNX node.

        Combines the ONNX header section with result lines into one cohesive block.
        """
        logger.trace(onnx_section + f"\n{cls._SEP2}\n" + "\n".join(log_lines) + f"\n{cls._SEP}")

    # ══════════════════════════════════════════════════════════════════
    # Section B — TIRGraph execution  (graph.py)
    # ══════════════════════════════════════════════════════════════════

    @classmethod
    def emit_graph_header(cls, name: str, total_nodes: int, debug_mode: bool, op_types: list) -> None:
        """
        Emit an INFO-level message at the start of graph execution.

        Shows graph name, node count, debug flag, and the set of unique
        op-types that will be executed.
        """
        logger.info(
            f"Executing Graph '{name}'\n"
            f"  Nodes    : {total_nodes}  |  debug={debug_mode}\n"
            f"  Op types ({len(op_types)}): {op_types}"
        )

    @staticmethod
    def build_node_input_lines(node_input_names, tensor_memory: dict) -> list:
        """
        Build pre-evaluation input description lines for a TIR node.

        Args:
            node_input_names: Iterable of input tensor names for this node.
            tensor_memory:    Current execution tensor store.

        Returns:
            List of formatted strings (one per input).
        """
        lines = []
        for j, n in enumerate(node_input_names):
            t = tensor_memory.get(n)
            if t is not None:
                shape = tuple(t.shape) if hasattr(t, "shape") else "?"
                dtype = getattr(t, "dtype", "?")
                lines.append(f"      {j + 1}) '{n}'  shape={shape}  dtype={dtype}")
            else:
                lines.append(f"      {j + 1}) '{n}'  [MISSING]")
        return lines

    @staticmethod
    def build_node_output_lines(outputs: dict) -> list:
        """
        Build post-evaluation output description lines for a TIR node.

        Args:
            outputs: Dict of {name: tensor} returned by ``node.eval()``.

        Returns:
            List of formatted strings (one per output).
        """
        lines = []
        for j, (k, v) in enumerate(outputs.items()):
            shape = tuple(v.shape) if hasattr(v, "shape") else "?"
            dtype = getattr(v, "dtype", "?")
            lines.append(f"      {j + 1}) '{k}'  shape={shape}  dtype={dtype}")
        return lines

    @staticmethod
    def format_node_exec_trace(exec_idx: int, total: int, node, in_lines: list, out_lines: list) -> str:
        """
        Full execution trace for one TIR node (used in non-debug mode).

        Shows inputs with shape/dtype and computed outputs.
        """
        return (
            f"  [{exec_idx + 1}/{total}] {node.op_type} '{node.name}'\n"
            + f"    Inputs  ({len(in_lines)}):\n"
            + ("\n".join(in_lines) if in_lines else "      (none)")
            + f"\n    Result  ({len(out_lines)}):\n"
            + ("\n".join(out_lines) if out_lines else "      (none)")
        )

    @staticmethod
    def format_node_exec_compact(exec_idx: int, total: int, node) -> str:
        """
        Compact execution header for one TIR node (used in debug mode).

        Full details are shown in the subsequent VALIDATE block, so only
        the counter + op-type + name are needed here.
        """
        return f"  [{exec_idx + 1}/{total}] {node.op_type} '{node.name}'"

    # ------------------------------------------------------------------
    # Debug-validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_tensor_fmt(name: str, tensor_memory: dict, original_to_sanitized: dict) -> str:
        """
        Look up a tensor (by original or sanitized name) and return a
        ``shape=... dtype=...`` string for use in VALIDATE blocks.
        """
        san = original_to_sanitized.get(name, name)
        t = tensor_memory.get(san)
        if t is None:
            t = tensor_memory.get(name)
        if t is not None and hasattr(t, "shape"):
            return f"  shape={tuple(t.shape)}  dtype={getattr(t, 'dtype', '?')}"
        return "  shape=?  dtype=?"

    @classmethod
    def build_onnx_validate_io_lines(cls, onnx_node_proto, tensor_memory: dict, original_to_sanitized: dict) -> tuple:
        """
        Build numbered ONNX input/output lines (with live shape/dtype) for the
        VALIDATE block.

        Args:
            onnx_node_proto:       ONNX NodeProto for the frontend node.
            tensor_memory:         Current execution tensor store.
            original_to_sanitized: Name mapping from the TIRGraph.

        Returns:
            (in_lines, out_lines) — two lists of formatted strings.
        """
        in_lines, out_lines = [], []
        if onnx_node_proto is None:
            return in_lines, out_lines
        for j, n in enumerate(getattr(onnx_node_proto, "input", [])):
            fmt = cls._resolve_tensor_fmt(n, tensor_memory, original_to_sanitized)
            in_lines.append(f"      {j + 1}) '{n}'{fmt}")
        for j, n in enumerate(getattr(onnx_node_proto, "output", [])):
            fmt = cls._resolve_tensor_fmt(n, tensor_memory, original_to_sanitized)
            out_lines.append(f"      {j + 1}) '{n}'{fmt}")
        return in_lines, out_lines

    @staticmethod
    def build_tir_validate_detail_lines(tir_node_names: list, get_node_fn, tensor_memory: dict) -> list:
        """
        Build TIR node detail lines (op-type, inputs, outputs with live
        shape/dtype) for the VALIDATE block.

        Args:
            tir_node_names: Ordered list of TIR node names to describe.
            get_node_fn:    Callable ``(name) -> TIRNode | None``
                            (typically ``tir_graph.get_node_by_name``).
            tensor_memory:  Current execution tensor store.

        Returns:
            List of formatted strings.
        """
        lines = []
        for idx, tir_node_name in enumerate(tir_node_names):
            tir_node = get_node_fn(tir_node_name)
            tir_op = tir_node.op_type if tir_node else "?"
            lines.append(f"    [{idx + 1}] {tir_op} '{tir_node_name}'")
            if tir_node:
                in_names = list(tir_node.inputs.keys()) if hasattr(tir_node.inputs, "keys") else list(tir_node.inputs)
                lines.append(f"       Inputs  ({len(in_names)}):")
                for j, inp in enumerate(in_names):
                    t = tensor_memory.get(inp)
                    sh = tuple(t.shape) if t is not None and hasattr(t, "shape") else "?"
                    dt = getattr(t, "dtype", "?") if t is not None else "?"
                    lines.append(f"         {j + 1}) '{inp}'  shape={sh}  dtype={dt}")
                out_names = (
                    list(tir_node.outputs.keys()) if hasattr(tir_node.outputs, "keys") else list(tir_node.outputs)
                )
                lines.append(f"       Outputs ({len(out_names)}):")
                for j, outp in enumerate(out_names):
                    t = tensor_memory.get(outp)
                    sh = tuple(t.shape) if t is not None and hasattr(t, "shape") else "?"
                    dt = getattr(t, "dtype", "?") if t is not None else "?"
                    lines.append(f"         {j + 1}) '{outp}'  shape={sh}  dtype={dt}")
        return lines

    @classmethod
    def emit_validate_block(
        cls,
        onnx_op: str,
        frontend_node_name: str,
        np_in_lines: list,
        np_out_lines: list,
        tir_nodes_for_frontend: list,
        tir_detail_lines: list,
    ) -> None:
        """
        Emit the full VALIDATE trace block for one frontend (ONNX) node.

        Shows both the ONNX node's inputs/outputs (with live shapes) and
        the mapped TIR node(s) with their inputs/outputs.
        """
        logger.trace(
            f"[VALIDATE] ONNX '{onnx_op}' | node: '{frontend_node_name}'\n"
            f"{cls._vSEP}\n"
            f"  ONNX inputs  ({len(np_in_lines)}):\n"
            + ("\n".join(np_in_lines) if np_in_lines else "    (none)")
            + f"\n  ONNX outputs ({len(np_out_lines)}):\n"
            + ("\n".join(np_out_lines) if np_out_lines else "    (none)")
            + f"\n{cls._vSEP2}\n"
            + f"  TIR node(s) ({len(tir_nodes_for_frontend)}):\n"
            + ("\n".join(tir_detail_lines) if tir_detail_lines else "    (none)")
            + f"\n{cls._vSEP2}"
        )

    @classmethod
    def emit_validate_ok(cls, frontend_node_name: str, n_tir_nodes: int) -> None:
        """Emit the [OK] trace line after a successful debug validation."""
        logger.trace(f"  [OK] '{frontend_node_name}' validated against " f"{n_tir_nodes} TIR node(s)\n{cls._vSEP}")

    @staticmethod
    def emit_validate_summary(n_validated: int, framework: str) -> None:
        """Emit the end-of-graph debug validation summary."""
        logger.trace(
            f"Debug validation summary: Validated {n_validated} "
            f"{framework.upper()} node(s) against their corresponding TIR nodes"
        )

    @staticmethod
    def emit_gc_freed(freed_names: list) -> None:
        """Emit a TRACE line listing tensors freed by GC in this iteration."""
        logger.trace(f"  GC freed: {', '.join(repr(n) for n in freed_names)}")
