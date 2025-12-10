"""
Transpiler Graph to Forge Module conversion.
Main entry point for transpiler-based Forge module generation.
"""
import os
import sys
import importlib.util
from typing import List, Tuple, Union, Dict, Any
import torch
import numpy as np
from loguru import logger

from forge.transpiler.core.graph import TIRGraph
from forge.transpiler.codegen.transpiler_generator import TranspilerCodeGenerator
import forge
from forge.module import ForgeModule, OnnxModule
from forge.tensor import to_pt_tensors, to_forge_tensors
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
    
    # Initialize configurations
    if compiler_cfg is None:
        compiler_cfg = CompilerConfig()
    if verify_cfg is None:
        verify_cfg = _get_global_verify_config()
    
    # Validate framework support
    if not isinstance(framework_mod, OnnxModule):
        raise NotImplementedError(
            f"Transpiler currently only supports ONNX models. "
            f"Got: {type(framework_mod)}"
        )
    
    if graph_name is None:
        graph_name = framework_mod.name
    
    # Convert framework model to TIRGraph
    from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
    
    transpiler = ONNXToForgeTranspiler(
        validate_model=True, 
        debug=compiler_cfg.transpiler_enable_debug
    )
    tir_graph = transpiler.transpile(framework_mod.module)
    
    # Set TIR graph name to module name
    tir_graph.name = framework_mod.name
    
    # Get framework outputs for verification (if needed)
    framework_outputs = None
    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        framework_outputs = framework_mod.cpu_eval_forward(*module_inputs)
    
    # Compare framework and TIR graph outputs (if verification enabled)
    if verify_cfg.verify_transpiler_graph:
        _verify_transpiler_graph_outputs(
            framework_mod,
            tir_graph,
            module_inputs,
            verify_cfg
        )
    
    # Generate Python code from TIRGraph
    module_name = graph_name
    class_name = _to_pascal_case(module_name)
    
    delete_inputs = not verify_cfg.enable_op_level_comparision
    if not delete_inputs:
        logger.warning(
            "Preserving Intermediate tensor values in ForgeModule forward may cause out-of-memory issues"
        )

    # Generate Python code string using TranspilerCodeGenerator
    code_generator = TranspilerCodeGenerator(
        tir_graph=tir_graph,
        class_name=class_name,
        delete_inputs=delete_inputs,
    )
    python_code = code_generator.generate()
    
    # Write Python code to file
    module_directory = "generated_modules"
    os.makedirs(module_directory, exist_ok=True)
    filename = f"{module_name}.py"
    file_path = os.path.join(module_directory, filename)
    
    with open(file_path, "w") as f:
        f.write(python_code)
    
    logger.info(f"Generated Forge module code: {file_path}")
    
    # Dynamically import and instantiate module
    sys.path.append(".")
    module = import_from_path(module_name, file_path)
    TestClass = getattr(module, class_name)
    
    # Create ForgeModule instance
    forge_mod = TestClass(module_name)
    
    # Load parameters from framework model
    forge_mod.process_framework_parameters(framework_mod.module)
    
    forge_inputs = list(to_forge_tensors(tuple(module_inputs)))
    
    # Verify generated module (if enabled)
    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        if framework_outputs is None:
            logger.warning("Verification enabled but framework outputs not available")
        else:
            from forge.tvm_to_python import get_forge_outputs
            # Use TTDevice for transpiler path
            forge_outputs = get_forge_outputs([forge_mod], ["TTDevice"], forge_inputs)
            from forge.tvm_to_python import verify_framework_vs_forge_codegen
            verify_framework_vs_forge_codegen(
                framework_outputs, 
                forge_outputs, 
                verify_cfg
            )
            logger.info("Verification passed: Forge module matches framework outputs")
    
    return [forge_mod], forge_inputs


def _verify_transpiler_graph_outputs(
    framework_mod,
    tir_graph: TIRGraph,
    module_inputs: List[torch.Tensor],
    verify_cfg,
):
    """
    Compare framework model outputs with TIR graph outputs.
    
    This verifies that the TIR graph correctly represents the framework model.
    
    Args:
        framework_mod: Framework module (OnnxModule)
        tir_graph: TIRGraph to verify
        module_inputs: Input tensors
        verify_cfg: Verification configuration (DeprecatedVerifyConfig)
    """
    logger.info("Verifying transpiler graph outputs against framework model...")
    

    if isinstance(framework_mod, OnnxModule):
        
        # Run framework model (inputs are already PyTorch tensors)
        framework_outputs = framework_mod.cpu_eval_forward(*module_inputs)
        
        # Prepare inputs for TIR graph (convert to dict)
        # TIR graph expects torch tensors, inputs are already PyTorch tensors
        tir_inputs = {}
        for i, input_name in enumerate(tir_graph.inputs):
            # Skip parameters and constants
            if input_name in tir_graph.params or input_name in tir_graph.constants:
                continue
            # Map inputs by position
            if i < len(module_inputs):
                tir_inputs[input_name] = module_inputs[i]
        
        # Run TIR graph
        tir_outputs = tir_graph.run(tir_inputs, enable_gc=True)
        
        # Compare outputs
        framework_output_list = framework_outputs if isinstance(framework_outputs, (list, tuple)) else [framework_outputs]
        # Use original_outputs since tir_graph.run() returns outputs with original names as keys
        if not tir_graph.original_outputs:
            raise ValueError(
                "TIR graph original_outputs not set. Cannot verify outputs. "
                f"Graph outputs (sanitized): {tir_graph.outputs}"
            )
        tir_output_list = []
        for out_name in tir_graph.original_outputs:
            if out_name not in tir_outputs:
                raise KeyError(
                    f"Output '{out_name}' not found in TIR graph outputs. "
                    f"Available outputs: {list(tir_outputs.keys())}, "
                    f"Graph outputs (original): {tir_graph.original_outputs}"
                )
            tir_output_list.append(tir_outputs[out_name])
        
        if len(framework_output_list) != len(tir_output_list):
            raise AssertionError(
                f"Output count mismatch: framework={len(framework_output_list)}, "
                f"TIR graph={len(tir_output_list)}"
            )

        
        for i, (golden, tir_out) in enumerate(zip(framework_output_list, tir_output_list)):
            from forge.verify.compare import compare_with_golden
            
            # Extract rtol, atol, pcc from verify_cfg
            # verify_cfg.rtol and verify_cfg.atol are dictionaries keyed by dtype
            # Get the appropriate value based on the tensor dtype
            tensor_dtype = golden.dtype if hasattr(golden, 'dtype') else torch.float32
            rtol = verify_cfg.rtol.get(tensor_dtype, verify_cfg.rtol.get(torch.float32, 1e-05))
            atol = verify_cfg.atol.get(tensor_dtype, verify_cfg.atol.get(torch.float32, 1e-08))
            pcc = verify_cfg.pcc if verify_cfg.pcc is not None else 0.99
            
            # Handle None values (use defaults)
            if rtol is None:
                rtol = 1e-05
            if atol is None:
                atol = 1e-08
            
            assert compare_with_golden(
                golden,
                tir_out,
                pcc=pcc,
                rtol=rtol,
                atol=atol
            ), f"Output {i} mismatch: golden={golden}, tir_out={tir_out}"
        
        logger.info("Transpiler graph verification passed: TIR graph matches framework outputs")



def _to_pascal_case(name: str) -> str:
    """Convert name to PascalCase."""
    return ''.join(word.capitalize() for word in name.split('_'))


def import_from_path(module_name: str, file_path: str):
    """Dynamically import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Could not load module {module_name} from {file_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


