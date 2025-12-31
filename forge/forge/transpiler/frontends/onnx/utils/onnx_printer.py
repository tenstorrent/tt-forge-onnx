# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Debug and visualization utilities for ONNX frontend.

These utilities are useful for debugging and inspecting ONNX models.
ONNX-specific utilities are here, while framework-agnostic TIR graph utilities
are in forge.transpiler.utils.graph_printer.
"""
import onnx
from loguru import logger


def print_onnx_model(onnx_model: onnx.ModelProto, title: str = "ONNX Model"):
    """
    Print ONNX model using ONNX's built-in printer.

    Args:
        onnx_model: ONNX ModelProto to print
        title: Optional title for the output
    """
    try:
        import onnx.printer

        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}")
        print(onnx.printer.to_text(onnx_model))
        print(f"{'='*80}\n")
    except ImportError:
        logger.warning("onnx.printer not available, falling back to string representation")
        print(f"\n{title}:")
        print(str(onnx_model))
    except Exception as e:
        logger.warning(f"Failed to print ONNX model: {e}")
        print(f"\n{title}:")
        print(f"Model: {onnx_model.graph.name}")
        print(f"Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"Outputs: {[out.name for out in onnx_model.graph.output]}")
        print(f"Nodes: {[node.name for node in onnx_model.graph.node]}")
