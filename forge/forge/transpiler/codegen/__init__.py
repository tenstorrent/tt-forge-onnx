# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Code generation module for converting TIRGraph to Python code.

This module provides utilities for generating executable Python Forge module code
from TIRGraph representations. It includes the TranspilerCodeGenerator class and
the main entry point function for transpiler-based Forge module generation.
"""
from forge.transpiler.codegen.transpiler_generator import TranspilerCodeGenerator
from forge.transpiler.codegen.transpiler_to_forge import generate_forge_module_from_transpiler

__all__ = ["TranspilerCodeGenerator", "generate_forge_module_from_transpiler"]
