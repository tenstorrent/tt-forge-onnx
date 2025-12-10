"""
Code generation module for converting TIRGraph to Python code.
"""
# TranspilerCodeGenerator is the main code generator
from forge.transpiler.codegen.transpiler_generator import TranspilerCodeGenerator
from forge.transpiler.codegen.transpiler_to_forge import generate_forge_module_from_transpiler

__all__ = ['TranspilerCodeGenerator', 'generate_forge_module_from_transpiler']

