"""
Test utilities for creating ONNX models and comparing results.
"""
import onnx
import onnx.helper
import onnx.numpy_helper
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from loguru import logger


def create_onnx_model(
    op_type: str,
    input_shapes: List[Tuple],
    input_dtypes: List[int],
    output_shapes: List[Tuple],
    output_dtypes: List[int],
    attrs: Dict[str, Any] = None,
    opset_version: int = 11,
    node_name: str = None,
    input_names: List[str] = None,
    output_names: List[str] = None,
    initializers: Dict[str, np.ndarray] = None
) -> onnx.ModelProto:
    """
    Create a simple ONNX model with a single operation.
    
    Args:
        op_type: ONNX operation type (e.g., "ReduceSum", "Add")
        input_shapes: List of input tensor shapes
        input_dtypes: List of ONNX dtype enums (e.g., onnx.TensorProto.FLOAT)
        output_shapes: List of output tensor shapes
        output_dtypes: List of ONNX dtype enums
        attrs: Operation attributes dictionary
        opset_version: ONNX opset version
        node_name: Name for the operation node
        input_names: Names for input tensors (default: ["input_0", "input_1", ...])
        output_names: Names for output tensors (default: ["output_0", "output_1", ...])
        initializers: Dictionary of initializer tensors (for constants)
        
    Returns:
        ONNX ModelProto
    """
    if attrs is None:
        attrs = {}
    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(input_shapes))]
    if output_names is None:
        output_names = [f"output_{i}" for i in range(len(output_shapes))]
    if initializers is None:
        initializers = {}
    if node_name is None:
        node_name = f"{op_type.lower()}_node"
    
    # Create input value infos
    # Note: In ONNX, if an input is an initializer, it should NOT be in input_value_infos
    # because initializers are already typed. However, the node still references them by name.
    # For ONNX Runtime compatibility, we skip creating value_info for initializers.
    input_value_infos = []
    for i, (shape, dtype, name) in enumerate(zip(input_shapes, input_dtypes, input_names)):
        # Skip if this input is an initializer - initializers are already typed
        if name not in initializers:
            input_value_infos.append(
                onnx.helper.make_tensor_value_info(name, dtype, list(shape))
            )
    
    # Create output value infos
    output_value_infos = []
    for shape, dtype, name in zip(output_shapes, output_dtypes, output_names):
        output_value_infos.append(
            onnx.helper.make_tensor_value_info(name, dtype, list(shape))
        )
    
    # Create initializer tensors
    initializer_list = []
    for name, array in initializers.items():
        initializer_list.append(onnx.numpy_helper.from_array(array, name=name))
    
    # Create the operation node with attributes
    # make_node accepts attributes as keyword arguments directly
    node_kwargs = {'name': node_name}
    for key, value in attrs.items():
        if isinstance(value, bool):
            # Convert bool to int for ONNX
            node_kwargs[key] = int(value)
        elif isinstance(value, (list, tuple)):
            # For lists, pass as-is (ONNX will handle conversion)
            node_kwargs[key] = list(value)
        else:
            # Pass other types as-is
            node_kwargs[key] = value
    
    node = onnx.helper.make_node(
        op_type,
        input_names,
        output_names,
        **node_kwargs
    )
    
    # Create graph
    graph = onnx.helper.make_graph(
        [node],
        f"test_{op_type.lower()}_graph",
        input_value_infos,
        output_value_infos,
        initializer_list
    )
    
    # Create model
    model = onnx.helper.make_model(
        graph,
        producer_name="forge-test",
        opset_imports=[onnx.helper.make_opsetid("", opset_version)]
    )
    
    # Validate model
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        logger.warning(f"Model validation warning: {e}")
    
    return model


def compare_tir_with_onnx(
    tir_graph,
    onnx_model: onnx.ModelProto,
    input_data: Dict[str, np.ndarray],
    rtol: float = 1e-3,
    atol: float = 1e-4
) -> Dict[str, Any]:
    """
    Compare TIR graph execution with ONNX model execution.
    
    Args:
        tir_graph: TIRGraph instance
        onnx_model: Original ONNX model
        input_data: Dictionary of input tensor data (numpy arrays)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        Dictionary with comparison results
    """
    import onnxruntime as ort
    
    results = {
        'tir_outputs': {},
        'onnx_outputs': {},
        'matches': {},
        'errors': []
    }
    
    # Run TIR graph
    try:
        input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
        tir_outputs = tir_graph.run(input_dict)
        results['tir_outputs'] = {
            name: output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else np.array(output)
            for name, output in tir_outputs.items()
        }
    except Exception as e:
        results['errors'].append(f"TIR execution failed: {e}")
        return results
    
    # Run ONNX model
    try:
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        onnx_inputs = {name: data for name, data in input_data.items()}
        onnx_outputs = sess.run(None, onnx_inputs)
        
        # Map outputs by name
        output_names = [output.name for output in onnx_model.graph.output]
        for i, output_name in enumerate(output_names):
            if i < len(onnx_outputs):
                results['onnx_outputs'][output_name] = onnx_outputs[i]
    except Exception as e:
        results['errors'].append(f"ONNX execution failed: {e}")
        return results
    
    # Compare outputs
    for output_name in onnx_model.graph.output:
        output_name = output_name.name
        if output_name in results['tir_outputs'] and output_name in results['onnx_outputs']:
            tir_out = results['tir_outputs'][output_name]
            onnx_out = results['onnx_outputs'][output_name]
            
            if tir_out.shape != onnx_out.shape:
                results['matches'][output_name] = False
                results['errors'].append(
                    f"Shape mismatch for {output_name}: TIR {tir_out.shape} vs ONNX {onnx_out.shape}"
                )
            else:
                matches = np.allclose(tir_out, onnx_out, rtol=rtol, atol=atol)
                results['matches'][output_name] = matches
                if not matches:
                    max_diff = np.abs(tir_out - onnx_out).max()
                    mean_diff = np.abs(tir_out - onnx_out).mean()
                    results['errors'].append(
                        f"Value mismatch for {output_name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
                    )
        else:
            results['errors'].append(f"Output {output_name} missing in TIR or ONNX outputs")
    
    return results


def verify_tir_graph_structure(
    tir_graph,
    onnx_model: onnx.ModelProto,
    expected_op_types: List[str] = None
) -> Dict[str, Any]:
    """
    Verify that the TIR graph structure matches the ONNX model.
    
    Args:
        tir_graph: TIRGraph instance
        onnx_model: Original ONNX model
        expected_op_types: List of expected TIR operation types
        
    Returns:
        Dictionary with verification results
    """
    results = {
        'node_count_match': len(tir_graph.nodes) == len(onnx_model.graph.node),
        'input_count_match': len(tir_graph.inputs) == len([inp for inp in onnx_model.graph.input if inp.name not in [init.name for init in onnx_model.graph.initializer]]),
        'output_count_match': len(tir_graph.outputs) == len(onnx_model.graph.output),
        'node_types': [node.op_type for node in tir_graph.nodes],
        'expected_op_types': expected_op_types,
        'warnings': []
    }
    
    if expected_op_types:
        actual_types = [node.op_type for node in tir_graph.nodes]
        if set(actual_types) != set(expected_op_types):
            results['warnings'].append(
                f"Operation type mismatch: expected {expected_op_types}, got {actual_types}"
            )
    
    return results


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


def print_tir_graph(tir_graph, title: str = "TIR Graph", detailed: bool = True):
    """
    Print TIR graph structure in a readable format.
    
    Args:
        tir_graph: TIRGraph instance to print
        title: Optional title for the output
        detailed: If True, print detailed node information including attributes
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # Graph metadata
    print(f"Graph Name: {tir_graph.name}")
    print(f"Debug Mode: {tir_graph.debug_mode}")
    print(f"Number of Nodes: {len(tir_graph.nodes)}")
    print()
    
    # Inputs
    print("Inputs:")
    for inp in tir_graph.inputs:
        print(f"  - {inp}")
    print()
    
    # Outputs
    print("Outputs:")
    for out in tir_graph.outputs:
        print(f"  - {out}")
    print()
    
    # Parameters and Constants
    if tir_graph.params:
        print("Parameters (Trainable):")
        for name, tensor in tir_graph.params.items():
            print(f"  - {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        print()
    
    if tir_graph.constants:
        print("Constants (Non-trainable):")
        for name, tensor in tir_graph.constants.items():
            print(f"  - {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        print()
    
    # Nodes in topological order
    print("Nodes (Topological Order):")
    print("-" * 80)
    sorted_nodes = tir_graph.get_topological_sort()
    
    for i, node in enumerate(sorted_nodes, 1):
        print(f"\n[{i}] {node.name} ({node.op_type})")
        print(f"    Inputs:  {node.inputs}")
        print(f"    Outputs: {node.outputs}")
        
        if detailed:
            # Print input tensor info
            if hasattr(node, 'input_tensors') and node.input_tensors:
                print("    Input Tensor Info:")
                for inp_name, tensor_info in node.input_tensors.items():
                    shape_str = tuple(tensor_info.shape) if tensor_info.shape else "unknown"
                    print(f"      {inp_name}: shape={shape_str}, dtype={tensor_info.torch_dtype}")
            
            # Print output tensor info
            if hasattr(node, 'output_tensors') and node.output_tensors:
                print("    Output Tensor Info:")
                for out_name, tensor_info in node.output_tensors.items():
                    shape_str = tuple(tensor_info.shape) if tensor_info.shape else "unknown"
                    print(f"      {out_name}: shape={shape_str}, dtype={tensor_info.torch_dtype}")
            
            # Print attributes
            if hasattr(node, 'attrs') and node.attrs:
                print("    Attributes:")
                for key, value in node.attrs.items():
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    print(f"      {key} = {value_str}")
            
            # Print Forge operation name
            if hasattr(node, 'forge_op_function_name') and node.forge_op_function_name:
                print(f"    Forge Op: {node.forge_op_function_name}")
    
    print(f"\n{'='*80}\n")

