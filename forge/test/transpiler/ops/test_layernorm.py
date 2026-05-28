# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX LayerNormalization operation (opset v17).
Tests single output, multiple outputs, different axis values, epsilon, shapes, and dtypes.
"""
import pytest
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


def _create_layernorm_model(
    opset_version: int,
    input_shape: tuple,
    axis: int = -1,
    epsilon: float = 1e-5,
    num_outputs: int = 1,
    has_bias: bool = True,
    dtype: int = onnx.TensorProto.FLOAT,
    node_name: str = "layernorm_test",
) -> onnx.ModelProto:
    """
    Create an ONNX LayerNormalization model.

    Args:
        opset_version: ONNX opset version (must be >= 17)
        input_shape: Input tensor shape
        axis: First normalization dimension (default: -1)
        epsilon: Epsilon value for numerical stability (default: 1e-5)
        num_outputs: Number of outputs (1=Y only, 2=Y+Mean, 3=Y+Mean+InvStdDev)
        has_bias: Whether to include bias input
        dtype: Input/output dtype
        node_name: Name for the LayerNormalization node

    Returns:
        ONNX ModelProto
    """
    if opset_version < 17:
        raise ValueError(f"LayerNormalization requires opset >= 17, got {opset_version}")

    # Determine normalized shape (shape of dimensions to normalize over)
    rank = len(input_shape)
    if axis < 0:
        axis = rank + axis
    normalized_shape = tuple(input_shape[axis:])

    # Map ONNX dtype to numpy dtype for Scale and Bias
    dtype_map = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.DOUBLE: np.float64,
    }
    np_dtype = dtype_map.get(dtype, np.float32)

    # Create scale tensor (must match normalized_shape and dtype)
    scale_shape = normalized_shape
    scale_array = np.ones(scale_shape, dtype=np_dtype)

    # Create bias tensor (optional, must match normalized_shape and dtype)
    bias_array = None
    if has_bias:
        bias_array = np.zeros(scale_shape, dtype=np_dtype)

    # Prepare inputs
    input_names = ["X", "Scale"]
    input_shapes = [input_shape, scale_shape]
    input_dtypes = [dtype, dtype]
    initializers = {"Scale": scale_array}

    if has_bias:
        input_names.append("Bias")
        input_shapes.append(scale_shape)
        input_dtypes.append(dtype)
        initializers["Bias"] = bias_array

    # Prepare outputs
    # Output Y has same shape as input
    # Output Mean and InvStdDev have shape with normalized axes reduced to 1
    mean_shape = list(input_shape)
    for i in range(axis, rank):
        mean_shape[i] = 1
    mean_shape = tuple(mean_shape)

    output_names = ["Y"]
    output_shapes = [input_shape]
    output_dtypes = [dtype]

    if num_outputs >= 2:
        output_names.append("Mean")
        output_shapes.append(mean_shape)
        output_dtypes.append(dtype)  # Mean uses same dtype as input

    if num_outputs >= 3:
        output_names.append("InvStdDev")
        output_shapes.append(mean_shape)
        output_dtypes.append(dtype)  # InvStdDev uses same dtype as input

    # Attributes
    attrs = {
        "axis": axis,
        "epsilon": epsilon,
    }

    return create_onnx_model(
        op_type="LayerNormalization",
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        attrs=attrs,
        opset_version=opset_version,
        node_name=node_name,
        input_names=input_names,
        output_names=output_names,
        initializers=initializers,
    )


@pytest.mark.transpiler
class TestLayerNormalization:
    """
    Comprehensive test cases for LayerNormalization operation.
    Tests single output, multiple outputs, different axis values, epsilon, shapes, and dtypes.
    """

    # ============================================================================
    # TEST CASES: SINGLE OUTPUT (Y only) - Should use LayerNormNode
    # ============================================================================

    @pytest.mark.parametrize("opset_version", [17])
    @pytest.mark.parametrize(
        "input_shape, axis",
        [
            # 2D inputs - test all valid axes
            ((3, 4), -2),  # Normalize over all dimensions (axis=0)
            ((3, 4), -1),  # Normalize over last dimension (axis=1)
            ((3, 4), 0),  # Normalize over all dimensions (positive)
            ((3, 4), 1),  # Normalize over last dimension (positive)
            ((5, 10), -2),  # Normalize over all dimensions
            ((5, 10), -1),  # Normalize over last dimension
            ((5, 10), 0),  # Normalize over all dimensions (positive)
            ((5, 10), 1),  # Normalize over last dimension (positive)
            # 3D inputs - test all valid axes
            ((2, 3, 4), -3),  # Normalize over all dimensions (axis=0)
            ((2, 3, 4), -2),  # Normalize over last 2 dimensions (axis=1)
            ((2, 3, 4), -1),  # Normalize over last dimension (axis=2)
            ((2, 3, 4), 0),  # Normalize over all dimensions (positive)
            ((2, 3, 4), 1),  # Normalize over last 2 dimensions (positive)
            ((2, 3, 4), 2),  # Normalize over last dimension (positive)
            # 4D inputs - test all valid axes
            ((2, 3, 4, 5), -4),  # Normalize over all dimensions (axis=0)
            ((2, 3, 4, 5), -3),  # Normalize over last 3 dimensions (axis=1)
            ((2, 3, 4, 5), -2),  # Normalize over last 2 dimensions (axis=2)
            ((2, 3, 4, 5), -1),  # Normalize over last dimension (axis=3)
            ((2, 3, 4, 5), 0),  # Normalize over all dimensions (positive)
            ((2, 3, 4, 5), 1),  # Normalize over last 3 dimensions (positive)
            ((2, 3, 4, 5), 2),  # Normalize over last 2 dimensions (positive)
            ((2, 3, 4, 5), 3),  # Normalize over last dimension (positive)
            # 5D inputs - test all valid axes
            ((1, 2, 3, 4, 5), -5),  # Normalize over all dimensions (axis=0)
            ((1, 2, 3, 4, 5), -4),  # Normalize over last 4 dimensions (axis=1)
            ((1, 2, 3, 4, 5), -3),  # Normalize over last 3 dimensions (axis=2)
            ((1, 2, 3, 4, 5), -2),  # Normalize over last 2 dimensions (axis=3)
            ((1, 2, 3, 4, 5), -1),  # Normalize over last dimension (axis=4)
            ((1, 2, 3, 4, 5), 0),  # Normalize over all dimensions (positive)
            ((1, 2, 3, 4, 5), 1),  # Normalize over last 4 dimensions (positive)
            ((1, 2, 3, 4, 5), 2),  # Normalize over last 3 dimensions (positive)
            ((1, 2, 3, 4, 5), 3),  # Normalize over last 2 dimensions (positive)
            ((1, 2, 3, 4, 5), 4),  # Normalize over last dimension (positive)
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
        ],
    )
    def test_layernorm_single_output(self, opset_version, input_shape, axis, dtype):
        """Test LayerNormalization with single output (Y only) - should use LayerNormNode."""
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # Create ONNX model
        onnx_model = _create_layernorm_model(
            opset_version=opset_version,
            input_shape=input_shape,
            axis=axis,
            num_outputs=1,  # Single output
            has_bias=True,
            dtype=dtype,
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have single LayerNormNode
        assert len(tir_graph.nodes) == 1, f"Expected 1 node for single output, got {len(tir_graph.nodes)}"

        layernorm_nodes = [n for n in tir_graph.nodes if n.op_type == "LayerNorm"]
        assert len(layernorm_nodes) == 1, (
            f"Expected 1 LayerNormNode, got {len(layernorm_nodes)}. " f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        )

        layernorm_node = layernorm_nodes[0]
        assert (
            len(layernorm_node.inputs) >= 2
        ), f"LayerNormNode should have at least 2 inputs (X, Scale), got {len(layernorm_node.inputs)}"
        assert (
            len(layernorm_node.outputs) == 1
        ), f"LayerNormNode should have 1 output for single output case, got {len(layernorm_node.outputs)}"

        # Verify attributes
        # Normalize axis to positive for comparison (converter normalizes negative axes)
        rank = len(input_shape)
        expected_axis = axis if axis >= 0 else rank + axis
        assert (
            layernorm_node.attrs.get("axis") == expected_axis
        ), f"Expected axis={expected_axis} (normalized from {axis}), got {layernorm_node.attrs.get('axis')}"
        assert (
            abs(layernorm_node.attrs.get("epsilon", 0) - 1e-5) < 1e-6
        ), f"Expected epsilon≈1e-5, got {layernorm_node.attrs.get('epsilon')}"

        # Test execution
        input_data = {
            "X": np.random.randn(*input_shape).astype(np_dtype),
        }

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-3, atol=1e-4)
        assert len(comparison.get("errors", [])) == 0, f"Comparison errors: {comparison.get('errors', [])}"
        assert comparison["matches"].get("Y", False), "Output Y does not match ONNX Runtime"

    # ============================================================================
    # TEST CASES: MULTIPLE OUTPUTS (Y, Mean, InvStdDev) - Should decompose
    # ============================================================================

    @pytest.mark.parametrize("opset_version", [17])
    @pytest.mark.parametrize(
        "input_shape, axis",
        [
            # 2D inputs - test all valid axes
            ((3, 4), -2),  # Normalize over all dimensions
            ((3, 4), -1),  # Normalize over last dimension
            ((3, 4), 0),  # Normalize over all dimensions (positive)
            ((3, 4), 1),  # Normalize over last dimension (positive)
            # 3D inputs - test all valid axes
            ((2, 3, 4), -3),  # Normalize over all dimensions
            ((2, 3, 4), -2),  # Normalize over last 2 dimensions
            ((2, 3, 4), -1),  # Normalize over last dimension
            ((2, 3, 4), 0),  # Normalize over all dimensions (positive)
            ((2, 3, 4), 1),  # Normalize over last 2 dimensions (positive)
            ((2, 3, 4), 2),  # Normalize over last dimension (positive)
            # 4D inputs - test all valid axes
            ((2, 3, 4, 5), -4),  # Normalize over all dimensions
            ((2, 3, 4, 5), -3),  # Normalize over last 3 dimensions
            ((2, 3, 4, 5), -2),  # Normalize over last 2 dimensions
            ((2, 3, 4, 5), -1),  # Normalize over last dimension
            ((2, 3, 4, 5), 0),  # Normalize over all dimensions (positive)
            ((2, 3, 4, 5), 1),  # Normalize over last 3 dimensions (positive)
            ((2, 3, 4, 5), 2),  # Normalize over last 2 dimensions (positive)
            ((2, 3, 4, 5), 3),  # Normalize over last dimension (positive)
        ],
    )
    @pytest.mark.parametrize(
        "num_outputs",
        [
            2,  # Y, Mean
            3,  # Y, Mean, InvStdDev
        ],
    )
    @pytest.mark.parametrize("dtype", [onnx.TensorProto.FLOAT])
    def test_layernorm_multiple_outputs(self, opset_version, input_shape, axis, num_outputs, dtype):
        """Test LayerNormalization with multiple outputs - should decompose into multiple TIR nodes."""
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # Create ONNX model
        onnx_model = _create_layernorm_model(
            opset_version=opset_version,
            input_shape=input_shape,
            axis=axis,
            num_outputs=num_outputs,
            has_bias=True,
            dtype=dtype,
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have multiple nodes (decomposition)
        assert len(tir_graph.nodes) > 1, f"Expected multiple nodes for decomposition, got {len(tir_graph.nodes)}"

        # Should NOT have LayerNormNode (decomposed instead)
        layernorm_nodes = [n for n in tir_graph.nodes if n.op_type == "LayerNorm"]
        assert len(layernorm_nodes) == 0, (
            f"Expected no LayerNormNode for multiple outputs (should decompose), "
            f"got {len(layernorm_nodes)}. Nodes: {[n.op_type for n in tir_graph.nodes]}"
        )

        # Should have ReduceMean, Sub, Mul, Add, Sqrt, Reciprocal nodes
        # Note: We use ReciprocalNode instead of Div for InvStdDev
        node_types = [n.op_type for n in tir_graph.nodes]
        assert "ReduceMean" in node_types, "Expected ReduceMean node in decomposition"
        assert "Sub" in node_types, "Expected Sub node in decomposition"
        assert "Mul" in node_types, "Expected Mul node in decomposition"
        assert "Add" in node_types, "Expected Add node in decomposition"
        assert "Sqrt" in node_types, "Expected Sqrt node in decomposition"
        # Check for Reciprocal (for InvStdDev) or Div (legacy)
        assert (
            "Reciprocal" in node_types or "Div" in node_types
        ), "Expected Reciprocal or Div node in decomposition for InvStdDev"

        # Test execution
        input_data = {
            "X": np.random.randn(*input_shape).astype(np_dtype),
        }

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-3, atol=1e-4)
        assert len(comparison.get("errors", [])) == 0, f"Comparison errors: {comparison.get('errors', [])}"

        # Verify all outputs match
        expected_outputs = ["Y"]
        if num_outputs >= 2:
            expected_outputs.append("Mean")
        if num_outputs >= 3:
            expected_outputs.append("InvStdDev")

        for output_name in expected_outputs:
            assert comparison["matches"].get(output_name, False), f"Output {output_name} does not match ONNX Runtime"

    # ============================================================================
    # TEST CASES: DIFFERENT EPSILON VALUES
    # ============================================================================

    @pytest.mark.parametrize("opset_version", [17])
    @pytest.mark.parametrize(
        "epsilon",
        [
            1e-5,  # Default
            1e-6,  # Smaller epsilon
            1e-4,  # Larger epsilon
            1e-3,  # Much larger epsilon
        ],
    )
    def test_layernorm_epsilon(self, opset_version, epsilon):
        """Test LayerNormalization with different epsilon values."""
        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT
        np_dtype = np.float32

        # Create ONNX model
        onnx_model = _create_layernorm_model(
            opset_version=opset_version,
            input_shape=input_shape,
            axis=-1,
            epsilon=epsilon,
            num_outputs=1,
            has_bias=True,
            dtype=dtype,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify epsilon attribute
        layernorm_node = [n for n in tir_graph.nodes if n.op_type == "LayerNorm"][0]
        assert (
            abs(layernorm_node.attrs.get("epsilon", 0) - epsilon) < 1e-6
        ), f"Expected epsilon={epsilon}, got {layernorm_node.attrs.get('epsilon')}"

        # Test execution
        input_data = {
            "X": np.random.randn(*input_shape).astype(np_dtype),
        }

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-3, atol=1e-4)
        assert len(comparison.get("errors", [])) == 0, f"Comparison errors: {comparison.get('errors', [])}"
        assert comparison["matches"].get("Y", False), "Output Y does not match ONNX Runtime"

    # ============================================================================
    # TEST CASES: WITHOUT BIAS
    # ============================================================================

    @pytest.mark.parametrize("opset_version", [17])
    @pytest.mark.parametrize(
        "input_shape, axis",
        [
            ((3, 4), -1),
            ((2, 3, 4), -1),
            ((2, 3, 4), -2),
        ],
    )
    def test_layernorm_no_bias(self, opset_version, input_shape, axis):
        """Test LayerNormalization without bias input."""
        dtype = onnx.TensorProto.FLOAT
        np_dtype = np.float32

        # Create ONNX model without bias
        onnx_model = _create_layernorm_model(
            opset_version=opset_version,
            input_shape=input_shape,
            axis=axis,
            num_outputs=1,
            has_bias=False,  # No bias
            dtype=dtype,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        layernorm_node = [n for n in tir_graph.nodes if n.op_type == "LayerNorm"][0]
        assert (
            len(layernorm_node.inputs) == 2
        ), f"Expected 2 inputs (X, Scale) without bias, got {len(layernorm_node.inputs)}"

        # Test execution
        input_data = {
            "X": np.random.randn(*input_shape).astype(np_dtype),
        }

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-3, atol=1e-4)
        assert len(comparison.get("errors", [])) == 0, f"Comparison errors: {comparison.get('errors', [])}"
        assert comparison["matches"].get("Y", False), "Output Y does not match ONNX Runtime"
