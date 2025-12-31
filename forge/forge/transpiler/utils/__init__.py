# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Common utilities shared across all frontends.

Framework-agnostic utilities that work with TIRGraph and other core components.
Framework-specific utilities are located in their respective frontend utils
directories (e.g., frontends/onnx/utils/).
"""
from forge.transpiler.utils.graph_printer import print_tir_graph

__all__ = [
    "print_tir_graph",
]
