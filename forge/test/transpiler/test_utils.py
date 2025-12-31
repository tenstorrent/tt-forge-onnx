# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test utilities for creating ONNX models and comparing results.
"""
import onnx
import onnx.helper
import onnx.numpy_helper
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from loguru import logger

from forge.transpiler.utils.graph_printer import print_tir_graph


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
    initializers: Dict[str, np.ndarray] = None,
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
            input_value_infos.append(onnx.helper.make_tensor_value_info(name, dtype, list(shape)))

    # Create output value infos
    output_value_infos = []
    for shape, dtype, name in zip(output_shapes, output_dtypes, output_names):
        output_value_infos.append(onnx.helper.make_tensor_value_info(name, dtype, list(shape)))

    # Create initializer tensors
    initializer_list = []
    for name, array in initializers.items():
        initializer_list.append(onnx.numpy_helper.from_array(array, name=name))

    # Create the operation node with attributes
    # make_node accepts attributes as keyword arguments directly
    node_kwargs = {"name": node_name}
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

    node = onnx.helper.make_node(op_type, input_names, output_names, **node_kwargs)

    # Create graph
    graph = onnx.helper.make_graph(
        [node], f"test_{op_type.lower()}_graph", input_value_infos, output_value_infos, initializer_list
    )

    # Create model
    model = onnx.helper.make_model(
        graph, producer_name="forge-test", opset_imports=[onnx.helper.make_opsetid("", opset_version)]
    )

    # Validate model
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        logger.warning(f"Model validation warning: {e}")

    return model


def compare_tir_with_onnx(
    tir_graph, onnx_model: onnx.ModelProto, input_data: Dict[str, np.ndarray], rtol: float = 1e-3, atol: float = 1e-4
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

    results = {"tir_outputs": {}, "onnx_outputs": {}, "matches": {}, "errors": []}

    # Run TIR graph
    try:
        input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
        tir_outputs = tir_graph.run(input_dict)
        results["tir_outputs"] = {
            name: output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else np.array(output)
            for name, output in tir_outputs.items()
        }
    except Exception as e:
        results["errors"].append(f"TIR execution failed: {e}")
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
                results["onnx_outputs"][output_name] = onnx_outputs[i]
    except Exception as e:
        results["errors"].append(f"ONNX execution failed: {e}")
        return results

    # Compare outputs
    for output_name in onnx_model.graph.output:
        output_name = output_name.name
        if output_name in results["tir_outputs"] and output_name in results["onnx_outputs"]:
            tir_out = results["tir_outputs"][output_name]
            onnx_out = results["onnx_outputs"][output_name]

            if tir_out.shape != onnx_out.shape:
                results["matches"][output_name] = False
                results["errors"].append(
                    f"Shape mismatch for {output_name}: TIR {tir_out.shape} vs ONNX {onnx_out.shape}"
                )
            else:
                matches = np.allclose(tir_out, onnx_out, rtol=rtol, atol=atol)
                results["matches"][output_name] = matches
                if not matches:
                    max_diff = np.abs(tir_out - onnx_out).max()
                    mean_diff = np.abs(tir_out - onnx_out).mean()
                    results["errors"].append(
                        f"Value mismatch for {output_name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
                    )
        else:
            results["errors"].append(f"Output {output_name} missing in TIR or ONNX outputs")

    return results


def verify_tir_graph_structure(
    tir_graph, onnx_model: onnx.ModelProto, expected_op_types: List[str] = None
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
        "node_count_match": len(tir_graph.nodes) == len(onnx_model.graph.node),
        "input_count_match": len(tir_graph.inputs)
        == len(
            [
                inp
                for inp in onnx_model.graph.input
                if inp.name not in [init.name for init in onnx_model.graph.initializer]
            ]
        ),
        "output_count_match": len(tir_graph.outputs) == len(onnx_model.graph.output),
        "node_types": [node.op_type for node in tir_graph.nodes],
        "expected_op_types": expected_op_types,
        "warnings": [],
    }

    if expected_op_types:
        actual_types = [node.op_type for node in tir_graph.nodes]
        if set(actual_types) != set(expected_op_types):
            results["warnings"].append(f"Operation type mismatch: expected {expected_op_types}, got {actual_types}")

    return results
