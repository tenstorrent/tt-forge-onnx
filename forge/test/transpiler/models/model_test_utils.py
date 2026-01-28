# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for model-level transpiler tests.

This module eliminates the boilerplate that was duplicated across every model
test (BERT, ResNet50, MNIST, …).  Each helper covers one well-defined phase of
the transpilation pipeline so that individual test functions stay concise and
focused on model-specific details.

Public API
----------
export_to_onnx          – export a PyTorch model to a temporary ONNX file
                          and return the loaded ``onnx.ModelProto``.
run_tir_transpilation   – transpile an ONNX model to TIRGraph and print a
                          structured summary.
run_onnx_comparison     – compare TIRGraph outputs against ONNX Runtime and
                          print a human-readable diff report.
run_forge_module_gen    – generate a Forge module from an ONNX model via the
                          transpiler codegen pipeline.
"""
import contextlib
import os
import tempfile
from typing import Dict, List

import onnx
import torch

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import compare_tir_with_onnx, print_tir_graph


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def export_to_onnx(
    pytorch_model: torch.nn.Module,
    test_input,
    input_names: List[str],
    output_names: List[str],
    opset_version: int = 17,
    do_constant_folding: bool = True,
):
    """
    Context manager: export a PyTorch model to a temporary ONNX file.

    Handles file creation and cleanup automatically.  The body of the
    ``with`` block receives the loaded and schema-checked ``onnx.ModelProto``.

    Args:
        pytorch_model: PyTorch ``nn.Module`` in eval mode.
        test_input: Example input tensor (or tuple of tensors) used for tracing.
        input_names: ONNX graph input names, in the same order as ``test_input``.
        output_names: ONNX graph output names.
        opset_version: Target ONNX opset version.
        do_constant_folding: Whether to apply constant folding during export.

    Yields:
        ``onnx.ModelProto`` — the loaded and validated ONNX model.

    Raises:
        AssertionError: If ``onnx.checker.check_model`` reports an invalid model.
    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx_path = tmp.name

    try:
        torch.onnx.export(
            pytorch_model,
            test_input,
            onnx_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            do_constant_folding=do_constant_folding,
        )

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        opset = onnx_model.opset_import[0].version
        n_nodes = len(onnx_model.graph.node)
        print(f"\n✓ ONNX export complete: opset={opset}, nodes={n_nodes}")

        yield onnx_model

    finally:
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)


# ---------------------------------------------------------------------------
# TIRGraph transpilation
# ---------------------------------------------------------------------------


def run_tir_transpilation(onnx_model: "onnx.ModelProto", title: str, debug: bool = True):
    """
    Transpile an ONNX model to a TIRGraph and print a structured summary.

    Args:
        onnx_model: Loaded ONNX model proto.
        title: Human-readable title used in the graph structure printout.
        debug: If ``True``, enable debug mode (ONNX Runtime cross-check during
               transpilation).

    Returns:
        The resulting ``TIRGraph`` instance.
    """
    transpiler = ONNXToForgeTranspiler(validate_model=True, debug=debug)
    tir_graph = transpiler.transpile(onnx_model)

    op_type_counts: Dict[str, int] = {}
    for node in tir_graph.nodes:
        op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1

    print(
        f"\n✓ TIRGraph: {len(tir_graph.nodes)} nodes, "
        f"{len(tir_graph.params)} params, "
        f"{len(tir_graph.constants)} constants"
    )
    print("  Node counts:", ", ".join(f"{k}:{v}" for k, v in sorted(op_type_counts.items())))

    print_tir_graph(tir_graph, title=title, detailed=True)

    return tir_graph


# ---------------------------------------------------------------------------
# ONNX Runtime comparison
# ---------------------------------------------------------------------------


def run_onnx_comparison(
    tir_graph,
    onnx_model: "onnx.ModelProto",
    input_data: Dict[str, "numpy.ndarray"],
) -> Dict:
    """
    Compare TIRGraph outputs against ONNX Runtime and print a diff report.

    Args:
        tir_graph: Transpiled ``TIRGraph``.
        onnx_model: Reference ONNX model (used to drive ONNX Runtime).
        input_data: Dict mapping ONNX input names to NumPy arrays.

    Returns:
        The raw comparison dict returned by ``compare_tir_with_onnx``, with
        keys ``"errors"``, ``"matches"``, and ``"diffs"``.
    """
    comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)

    print("\n[ONNX Runtime Comparison]")
    if comparison["errors"]:
        print(f"  Errors ({len(comparison['errors'])}):")
        for err in comparison["errors"]:
            print(f"    - {err}")
    else:
        print("  ✓ No errors")

    if all(comparison["matches"].values()):
        print("  ✓ All outputs match")
    else:
        print("  ⚠ Output mismatches:")
        for output_name, matched in comparison["matches"].items():
            if not matched:
                print(f"    - {output_name}")
                diff = comparison.get("diffs", {}).get(output_name)
                if diff:
                    print(
                        f"      Max diff: {diff.get('max_diff', 'N/A')}, " f"Mean diff: {diff.get('mean_diff', 'N/A')}"
                    )

    return comparison


# ---------------------------------------------------------------------------
# Forge module generation
# ---------------------------------------------------------------------------


def run_forge_module_gen(
    onnx_model: "onnx.ModelProto",
    test_input: torch.Tensor,
    graph_name: str,
    resolve_dynamic_shapes: bool = False,
) -> None:
    """
    Generate a Forge module from an ONNX model via the transpiler codegen pipeline.

    Args:
        onnx_model: Loaded ONNX model proto.
        test_input: Representative input tensor passed to the code generator.
        graph_name: Identifier used for the generated Forge graph / module.
        resolve_dynamic_shapes: Whether to resolve dynamic shapes during transpilation.
    """
    from forge.config import CompilerConfig
    from forge.module import OnnxModule
    from forge.transpiler.codegen.transpiler_to_forge import generate_forge_module_from_transpiler
    from forge.verify.config import DeprecatedVerifyConfig

    onnx_module = OnnxModule(graph_name, onnx_model)

    compiler_cfg = CompilerConfig(
        compile_transpiler_to_python=True,
        transpiler_enable_debug=True,
        transpiler_resolve_dynamic_shapes=resolve_dynamic_shapes,
    )

    verify_cfg = DeprecatedVerifyConfig()
    verify_cfg.verify_forge_codegen_vs_framework = True
    verify_cfg.verify_transpiler_graph = True

    generate_forge_module_from_transpiler(
        framework_mod=onnx_module,
        module_inputs=[test_input],
        compiler_cfg=compiler_cfg,
        graph_name=graph_name,
        verify_cfg=verify_cfg,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 80


def print_section(title: str) -> None:
    """Print a visually distinct section header."""
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")
