# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Transpiler Graph to Forge Module conversion.

This module provides the main entry point for transpiler-based Forge module generation.
It handles the complete pipeline from framework models to executable Forge modules,
including transpilation, code generation, dynamic module loading, and verification.
"""
import os
import sys
import importlib.util
from typing import List, Tuple
import torch
from loguru import logger

from forge.transpiler.core.graph import TIRGraph
from forge.transpiler.codegen.transpiler_generator import TranspilerCodeGenerator
import forge
from forge.module import ForgeModule, OnnxModule
from forge.tensor import to_forge_tensors
from forge.verify.config import _get_global_verify_config


def generate_forge_module_from_transpiler(
    framework_mod,
    module_inputs: List[torch.Tensor],
    compiler_cfg=None,
    graph_name: str = None,
    verify_cfg=None,
) -> Tuple[List[ForgeModule], List[forge.Tensor]]:
    """
    Generate Forge modules from framework model using transpiler.

    This function handles the complete transpiler path:
    1. Convert framework model to TIRGraph
    2. Compare framework and TIR graph outputs (if verification enabled)
    3. Generate Python code from TIRGraph
    4. File I/O for generated Python modules
    5. Dynamic module import and instantiation
    6. Parameter loading from framework models
    7. Verification (if enabled)

    Args:
        framework_mod: Original framework module (OnnxModule)
        module_inputs: Original model inputs
        compiler_cfg: Compiler configuration
        graph_name: Name for the generated module
        verify_cfg: Verification configuration

    Returns:
        Tuple of (forge_modules, forge_inputs)
        - forge_modules: List of ForgeModule instances (typically one)
        - forge_inputs: List of Forge Tensor inputs (ready for forward() calls)

    Raises:
        ValueError: If TIRGraph is invalid or code generation fails
        ImportError: If generated module cannot be imported
        AssertionError: If parameter loading fails
    """
    from forge.config import CompilerConfig

    # Step 1: Initialize configurations with defaults if not provided
    # CompilerConfig contains transpiler settings (debug mode, etc.)
    if compiler_cfg is None:
        compiler_cfg = CompilerConfig()
    # Verification config controls output comparison and validation
    if verify_cfg is None:
        verify_cfg = _get_global_verify_config()

    # Step 2: Validate framework support
    # Currently only ONNX models are supported via the transpiler
    if not isinstance(framework_mod, OnnxModule):
        raise NotImplementedError(f"Transpiler currently only supports ONNX models. " f"Got: {type(framework_mod)}")

    # Step 3: Set graph name (use module name if not specified)
    if graph_name is None:
        graph_name = framework_mod.name

    # Step 4: Convert framework model to TIRGraph using ONNX transpiler
    # The transpiler converts ONNX operations to TIR nodes and builds the graph
    from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler

    # Create transpiler with validation enabled and debug mode from config
    # Debug mode enables ONNX Runtime comparison for validation
    transpiler = ONNXToForgeTranspiler(validate_model=True, debug=compiler_cfg.transpiler_enable_debug)
    # Perform conversion: ONNX ModelProto -> TIRGraph
    tir_graph = transpiler.transpile(framework_mod.module)

    # Set graph name to match module name for consistency
    tir_graph.name = framework_mod.name

    # Step 5: Run framework model to get reference outputs (if verification enabled)
    # These outputs will be compared against generated Forge module outputs later
    framework_outputs = None
    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        # Execute framework model on CPU to get reference outputs
        framework_outputs = framework_mod.cpu_eval_forward(*module_inputs)

    # Step 6: Verify TIR graph outputs match framework outputs (if enabled)
    # This validates that the transpilation step correctly converted the model
    if verify_cfg.verify_transpiler_graph:
        _verify_transpiler_graph_outputs(framework_mod, tir_graph, module_inputs, verify_cfg)

    # Step 7: Generate Python code from TIRGraph
    # Convert module name to PascalCase for class name (e.g., "my_model" -> "MyModel")
    module_name = graph_name
    class_name = _to_pascal_case(module_name)

    # Determine whether to delete intermediate tensors after use
    # If op-level comparison is enabled, keep intermediates for debugging (may cause OOM)
    # Otherwise, delete intermediates to save memory
    delete_inputs = not verify_cfg.enable_op_level_comparision
    if not delete_inputs:
        logger.warning("Preserving Intermediate tensor values in ForgeModule forward may cause out-of-memory issues")

    # Create code generator with TIR graph and configuration
    # The generator will produce Python code implementing the model as a ForgeModule class
    code_generator = TranspilerCodeGenerator(
        tir_graph=tir_graph,
        class_name=class_name,
        delete_inputs=delete_inputs,
    )
    # Generate Python source code string
    python_code = code_generator.generate()

    # Step 8: Write generated Python code to file
    # Create output directory if it doesn't exist
    module_directory = "generated_modules"
    os.makedirs(module_directory, exist_ok=True)
    filename = f"{module_name}.py"
    file_path = os.path.join(module_directory, filename)

    # Write Python code to file
    with open(file_path, "w") as f:
        f.write(python_code)

    logger.info(f"Generated Forge module code: {file_path}")

    # Step 9: Dynamically import and instantiate the generated module
    # Add current directory to Python path for imports
    sys.path.append(".")
    # Load the generated Python module from file
    module = import_from_path(module_name, file_path)
    # Get the ForgeModule class from the module
    TestClass = getattr(module, class_name)

    # Instantiate the ForgeModule with the module name
    forge_mod = TestClass(module_name)

    # Step 10: Load parameters from framework model into ForgeModule
    # This copies weights/biases from the ONNX model to the Forge module
    forge_mod.process_framework_parameters(framework_mod.module)

    # Step 11: Convert PyTorch input tensors to Forge Tensor format
    # Forge tensors are the format expected by ForgeModule.forward()
    forge_inputs = list(to_forge_tensors(tuple(module_inputs)))

    # Step 12: Verify generated Forge module outputs match framework outputs (if enabled)
    # This is the final verification step: compare Forge module execution with original framework
    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        if framework_outputs is None:
            logger.warning("Verification enabled but framework outputs not available")
        else:
            # Execute Forge module to get outputs
            from forge.tvm_to_python import get_forge_outputs

            forge_outputs = get_forge_outputs([forge_mod], ["TTDevice"], forge_inputs)
            # Compare Forge outputs with framework outputs
            from forge.tvm_to_python import verify_framework_vs_forge_codegen

            verify_framework_vs_forge_codegen(framework_outputs, forge_outputs, verify_cfg)
            logger.info("Verification passed: Forge module matches framework outputs")

    # Return Forge module and inputs ready for forward() calls
    return [forge_mod], forge_inputs


def _verify_transpiler_graph_outputs(
    framework_mod,
    tir_graph: TIRGraph,
    module_inputs: List[torch.Tensor],
    verify_cfg,
):
    """
    Compare framework model outputs with TIR graph outputs.

    This verifies that the TIR graph correctly represents the framework model
    by comparing outputs from both execution paths.

    Args:
        framework_mod: Framework module (OnnxModule)
        tir_graph: TIRGraph to verify
        module_inputs: Input tensors (PyTorch tensors)
        verify_cfg: Verification configuration

    Raises:
        ValueError: If TIR graph original_outputs not set
        KeyError: If output name not found in TIR graph outputs
        AssertionError: If output count mismatch or output values don't match
    """
    logger.info("Verifying transpiler graph outputs against framework model...")

    if isinstance(framework_mod, OnnxModule):
        # Step 1: Execute framework model to get reference outputs
        # These are the "golden" outputs that the TIR graph should match
        framework_outputs = framework_mod.cpu_eval_forward(*module_inputs)

        # Step 2: Prepare inputs for TIR graph execution
        # Map PyTorch input tensors to TIR graph input names
        # Skip parameters and constants (they're already in the graph)
        tir_inputs = {}
        for i, input_name in enumerate(tir_graph.inputs):
            # Skip if this "input" is actually a parameter or constant
            # (ONNX models may list initializers as inputs)
            if input_name in tir_graph.params or input_name in tir_graph.constants:
                continue
            # Map input by position (TIR graph inputs match framework inputs)
            if i < len(module_inputs):
                tir_inputs[input_name] = module_inputs[i]

        # Step 3: Execute TIR graph to get outputs
        # enable_gc=True enables garbage collection of intermediate activations
        # TIR graph expects torch tensors, inputs are already PyTorch tensors
        tir_outputs = tir_graph.run(tir_inputs, enable_gc=True)

        # Step 4: Normalize outputs to lists for comparison
        # Framework outputs may be single tensor or list/tuple
        framework_output_list = (
            framework_outputs if isinstance(framework_outputs, (list, tuple)) else [framework_outputs]
        )

        # Validate that TIR graph has original output names set
        # original_outputs maps sanitized names back to ONNX names for comparison
        if not tir_graph.original_outputs:
            raise ValueError(
                "TIR graph original_outputs not set. Cannot verify outputs. "
                f"Graph outputs (sanitized): {tir_graph.outputs}"
            )

        # Extract TIR outputs in the same order as framework outputs
        # Use original_outputs to get outputs by their ONNX names
        tir_output_list = []
        for out_name in tir_graph.original_outputs:
            if out_name not in tir_outputs:
                raise KeyError(
                    f"Output '{out_name}' not found in TIR graph outputs. "
                    f"Available outputs: {list(tir_outputs.keys())}, "
                    f"Graph outputs (original): {tir_graph.original_outputs}"
                )
            tir_output_list.append(tir_outputs[out_name])

        # Step 5: Validate output count matches
        # Both framework and TIR graph should produce the same number of outputs
        if len(framework_output_list) != len(tir_output_list):
            raise AssertionError(
                f"Output count mismatch: framework={len(framework_output_list)}, " f"TIR graph={len(tir_output_list)}"
            )

        # Step 6: Compare each output pair with tolerance settings
        # Uses dtype-specific tolerances from verify_cfg for numerical comparison
        for i, (golden, tir_out) in enumerate(zip(framework_output_list, tir_output_list)):
            from forge.verify.compare import compare_with_golden

            # Determine tensor dtype for tolerance lookup
            # verify_cfg.rtol and verify_cfg.atol are dictionaries keyed by dtype
            tensor_dtype = golden.dtype if hasattr(golden, "dtype") else torch.float32
            # Get tolerance values for this dtype, or fall back to float32 defaults
            rtol = verify_cfg.rtol.get(tensor_dtype, verify_cfg.rtol.get(torch.float32, 1e-05))
            atol = verify_cfg.atol.get(tensor_dtype, verify_cfg.atol.get(torch.float32, 1e-08))
            # Pearson correlation coefficient threshold (default 0.99)
            pcc = verify_cfg.pcc if verify_cfg.pcc is not None else 0.99

            # Handle None values (use defaults)
            if rtol is None:
                rtol = 1e-05
            if atol is None:
                atol = 1e-08

            # Compare outputs with specified tolerances
            # Raises AssertionError if outputs don't match within tolerance
            assert compare_with_golden(
                golden, tir_out, pcc=pcc, rtol=rtol, atol=atol
            ), f"Output {i} mismatch: golden={golden}, tir_out={tir_out}"

        logger.info("Transpiler graph verification passed: TIR graph matches framework outputs")


def _to_pascal_case(name: str) -> str:
    """
    Convert name to PascalCase.

    Converts underscore-separated names to PascalCase by capitalizing each word.
    This is used to generate Python class names from module names.

    Examples:
        "my_module" -> "MyModule"
        "resnet50" -> "Resnet50"
        "transformer_model" -> "TransformerModel"

    Args:
        name: Name to convert (e.g., "my_module" -> "MyModule")

    Returns:
        PascalCase string suitable for use as a Python class name
    """
    # Split on underscores, capitalize each word, then join without separators
    return "".join(word.capitalize() for word in name.split("_"))


def import_from_path(module_name: str, file_path: str):
    """
    Dynamically import a module from a file path.

    Uses importlib to load a Python module from a file system path and
    register it in sys.modules for subsequent imports. This allows importing
    generated modules without requiring them to be in Python's import path.

    The process:
    1. Create module spec from file location
    2. Create module object from spec
    3. Register module in sys.modules (allows import statements to find it)
    4. Execute module code to populate module namespace

    Args:
        module_name: Name for the module (used in sys.modules registry)
        file_path: Path to the Python file to import (absolute or relative)

    Returns:
        Imported module object with all classes, functions, and variables defined

    Raises:
        AssertionError: If module cannot be loaded (spec is None, file not found or invalid)
    """
    # Create module specification from file location
    # This tells Python how to load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Could not load module {module_name} from {file_path}"

    # Create module object from spec (empty namespace at this point)
    module = importlib.util.module_from_spec(spec)

    # Register module in sys.modules before execution
    # This allows the module to import itself or be imported by other code
    sys.modules[module_name] = module

    # Execute module code to populate module namespace
    # This runs all top-level code in the file (imports, class definitions, etc.)
    spec.loader.exec_module(module)

    return module
