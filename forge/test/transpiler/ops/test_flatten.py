# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Flatten operation.
Tests all opset versions, dimensions (1D-5D), axis values (positive, negative, zero), and value comparison.
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
    verify_tir_graph_structure,
)


def _calculate_flatten_output_shape(input_shape, axis):
    """
    Calculate expected output shape for Flatten operation.

    Args:
        input_shape: Input tensor shape
        axis: Flatten axis (normalized to non-negative)

    Returns:
        Tuple of (outer_dim, inner_dim) for 2D output
    """
    rank = len(input_shape)

    # Normalize negative axis
    if axis < 0:
        axis = axis + rank

    # Calculate outer dimension: product of dimensions [0:axis)
    outer_dim = 1
    for i in range(axis):
        outer_dim *= input_shape[i]

    # Calculate inner dimension: product of dimensions [axis:]
    inner_dim = 1
    for i in range(axis, rank):
        inner_dim *= input_shape[i]

    return (outer_dim, inner_dim)


@pytest.mark.transpiler
class TestFlatten:
    """Comprehensive test cases for Flatten operation."""

    @pytest.mark.parametrize("opset_version", [1, 9, 11, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "input_shape",
        [
            # 1D shapes
            (10,),
            (1,),
            (100,),
            # 2D shapes
            (2, 3),
            (1, 5),
            (5, 1),
            (10, 20),
            # 3D shapes
            (2, 3, 4),
            (1, 1, 1),
            (1, 10, 5),
            (10, 1, 5),
            (5, 4, 3),
            # 4D shapes
            (2, 3, 4, 5),
            (1, 2, 3, 4),
            (2, 1, 1, 1),
            (1, 1, 1, 1),
            (3, 4, 5, 6),
            # 5D shapes
            (2, 3, 4, 5, 6),
            (1, 2, 3, 4, 5),
            (2, 1, 1, 1, 1),
            (1, 1, 1, 1, 1),
        ],
    )
    @pytest.mark.parametrize(
        "axis",
        [
            # Positive axis values
            0,
            1,
            2,
            3,
            4,
            # Zero (special case)
            0,
            # Negative axis values (will be skipped for opset < 11)
            -1,
            -2,
            -3,
            -4,
            -5,
        ],
    )
    def test_flatten_all_dimensions_and_axes(self, opset_version, input_shape, axis):
        """Test Flatten with all dimensions (1D-5D) and axis values."""
        rank = len(input_shape)

        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip negative axis for opset < 11
        if axis < 0 and opset_version < 11:
            pytest.skip(f"Negative axis={axis} only supported in opset 11+, got opset {opset_version}")

        # Skip axis values that are out of range
        if abs(axis) > rank:
            pytest.skip(f"Axis {axis} is out of range for input rank {rank}")

        # Normalize axis for validation
        normalized_axis = axis if axis >= 0 else axis + rank

        # Skip if normalized axis is out of range
        if normalized_axis > rank:
            pytest.skip(f"Normalized axis {normalized_axis} is out of range for input rank {rank}")

        # Calculate expected output shape
        expected_output_shape = _calculate_flatten_output_shape(input_shape, normalized_axis)

        # Create test data
        total_elements = np.prod(input_shape)
        input_data = np.arange(total_elements, dtype=np.float32).reshape(input_shape)

        # Create ONNX model
        attrs = {"axis": axis}
        onnx_model = create_onnx_model(
            op_type="Flatten",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=opset_version,
            node_name="flatten_test",
            input_names=["input_0"],
            output_names=["output_0"],
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify graph structure
        verification = verify_tir_graph_structure(
            tir_graph, onnx_model, expected_op_types=["Reshape"]  # Flatten maps to Reshape
        )
        assert verification["node_count_match"], "Node count should match"
        assert verification["input_count_match"], "Input count should match"
        assert verification["output_count_match"], "Output count should match"
        assert "Reshape" in verification["node_types"], "Should have Reshape node"

        # Compare with ONNX Runtime
        input_dict = {"input_0": input_data}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict)

        # Verify no errors
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify outputs match
        assert comparison["matches"][
            "output_0"
        ], f"Output mismatch for input_shape={input_shape}, axis={axis}, opset={opset_version}"

        # Verify output shape
        tir_output = comparison["tir_outputs"]["output_0"]
        assert (
            tir_output.shape == expected_output_shape
        ), f"Output shape mismatch: expected {expected_output_shape}, got {tir_output.shape}"

    @pytest.mark.parametrize("opset_version", [1, 9, 11, 13, 21, 23])
    @pytest.mark.parametrize(
        "input_shape, axis, expected_output_shape",
        [
            # 1D cases
            ((10,), 0, (1, 10)),
            ((10,), 1, (10, 1)),
            # 2D cases
            ((2, 3), 0, (1, 6)),
            ((2, 3), 1, (2, 3)),
            ((2, 3), 2, (6, 1)),
            # 3D cases
            ((2, 3, 4), 0, (1, 24)),
            ((2, 3, 4), 1, (2, 12)),
            ((2, 3, 4), 2, (6, 4)),
            ((2, 3, 4), 3, (24, 1)),
            # 4D cases
            ((2, 3, 4, 5), 0, (1, 120)),
            ((2, 3, 4, 5), 1, (2, 60)),
            ((2, 3, 4, 5), 2, (6, 20)),
            ((2, 3, 4, 5), 3, (24, 5)),
            ((2, 3, 4, 5), 4, (120, 1)),
            # 5D cases
            ((2, 3, 4, 5, 6), 0, (1, 720)),
            ((2, 3, 4, 5, 6), 1, (2, 360)),
            ((2, 3, 4, 5, 6), 2, (6, 120)),
            ((2, 3, 4, 5, 6), 3, (24, 30)),
            ((2, 3, 4, 5, 6), 4, (120, 6)),
            ((2, 3, 4, 5, 6), 5, (720, 1)),
        ],
    )
    def test_flatten_positive_axis(self, opset_version, input_shape, axis, expected_output_shape):
        """Test Flatten with positive axis values."""
        # Create test data
        total_elements = np.prod(input_shape)
        input_data = np.arange(total_elements, dtype=np.float32).reshape(input_shape)

        # Create ONNX model
        attrs = {"axis": axis}
        onnx_model = create_onnx_model(
            op_type="Flatten",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=opset_version,
            node_name="flatten_test",
            input_names=["input_0"],
            output_names=["output_0"],
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        input_dict = {"input_0": input_data}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"][
            "output_0"
        ], f"Output mismatch for input_shape={input_shape}, axis={axis}, opset={opset_version}"

        # Verify output shape
        tir_output = comparison["tir_outputs"]["output_0"]
        assert (
            tir_output.shape == expected_output_shape
        ), f"Output shape mismatch: expected {expected_output_shape}, got {tir_output.shape}"

    @pytest.mark.parametrize("opset_version", [11, 13, 21, 23])
    @pytest.mark.parametrize(
        "input_shape, axis, expected_output_shape",
        [
            # 2D cases with negative axis
            ((2, 3), -1, (2, 3)),  # axis=-1 → axis=1: [0:1)=2, [1:2)=3
            ((2, 3), -2, (1, 6)),  # axis=-2 → axis=0: [0:0)=1, [0:2)=6
            # 3D cases with negative axis
            ((2, 3, 4), -1, (6, 4)),  # axis=-1 → axis=2: [0:2)=6, [2:3)=4
            ((2, 3, 4), -2, (2, 12)),  # axis=-2 → axis=1: [0:1)=2, [1:3)=12
            ((2, 3, 4), -3, (1, 24)),  # axis=-3 → axis=0: [0:0)=1, [0:3)=24
            # 4D cases with negative axis
            ((2, 3, 4, 5), -1, (24, 5)),  # axis=-1 → axis=3: [0:3)=24, [3:4)=5
            ((2, 3, 4, 5), -2, (6, 20)),  # axis=-2 → axis=2: [0:2)=6, [2:4)=20
            ((2, 3, 4, 5), -3, (2, 60)),  # axis=-3 → axis=1: [0:1)=2, [1:4)=60
            ((2, 3, 4, 5), -4, (1, 120)),  # axis=-4 → axis=0: [0:0)=1, [0:4)=120
            # 5D cases with negative axis
            ((2, 3, 4, 5, 6), -1, (120, 6)),  # axis=-1 → axis=4: [0:4)=120, [4:5)=6
            ((2, 3, 4, 5, 6), -2, (24, 30)),  # axis=-2 → axis=3: [0:3)=24, [3:5)=30
            ((2, 3, 4, 5, 6), -3, (6, 120)),  # axis=-3 → axis=2: [0:2)=6, [2:5)=120
            ((2, 3, 4, 5, 6), -4, (2, 360)),  # axis=-4 → axis=1: [0:1)=2, [1:5)=360
            ((2, 3, 4, 5, 6), -5, (1, 720)),  # axis=-5 → axis=0: [0:0)=1, [0:5)=720
        ],
    )
    def test_flatten_negative_axis(self, opset_version, input_shape, axis, expected_output_shape):
        """Test Flatten with negative axis values (opset 11+ only)."""
        # Create test data
        total_elements = np.prod(input_shape)
        input_data = np.arange(total_elements, dtype=np.float32).reshape(input_shape)

        # Create ONNX model
        attrs = {"axis": axis}
        onnx_model = create_onnx_model(
            op_type="Flatten",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=opset_version,
            node_name="flatten_test",
            input_names=["input_0"],
            output_names=["output_0"],
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        input_dict = {"input_0": input_data}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"][
            "output_0"
        ], f"Output mismatch for input_shape={input_shape}, axis={axis}, opset={opset_version}"

        # Verify output shape
        tir_output = comparison["tir_outputs"]["output_0"]
        assert (
            tir_output.shape == expected_output_shape
        ), f"Output shape mismatch: expected {expected_output_shape}, got {tir_output.shape}"

    @pytest.mark.parametrize("opset_version", [1, 9, 11, 13, 21, 23])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (2, 3, 4),
            (1, 5, 10),
            (3, 4, 5, 6),
        ],
    )
    def test_flatten_default_axis(self, opset_version, input_shape):
        """Test Flatten with default axis=1 (no axis attribute)."""
        # Calculate expected output shape with axis=1
        expected_output_shape = _calculate_flatten_output_shape(input_shape, 1)

        # Create test data
        total_elements = np.prod(input_shape)
        input_data = np.arange(total_elements, dtype=np.float32).reshape(input_shape)

        # Create ONNX model without axis attribute (defaults to 1)
        attrs = {}
        onnx_model = create_onnx_model(
            op_type="Flatten",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=opset_version,
            node_name="flatten_test",
            input_names=["input_0"],
            output_names=["output_0"],
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        input_dict = {"input_0": input_data}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"][
            "output_0"
        ], f"Output mismatch for input_shape={input_shape}, default axis=1, opset={opset_version}"

    @pytest.mark.parametrize("opset_version", [1, 9, 11, 13, 21, 23])
    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (onnx.TensorProto.FLOAT, np.float32),
            (onnx.TensorProto.DOUBLE, np.float64),
            (onnx.TensorProto.INT32, np.int32),
            (onnx.TensorProto.INT64, np.int64),
        ],
    )
    def test_flatten_dtypes(self, opset_version, dtype, np_dtype):
        """Test Flatten with different data types."""
        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")

        input_shape = (2, 3, 4)
        axis = 1
        expected_output_shape = _calculate_flatten_output_shape(input_shape, axis)

        # Create test data
        total_elements = np.prod(input_shape)
        input_data = np.arange(total_elements, dtype=np_dtype).reshape(input_shape)

        # Create ONNX model
        attrs = {"axis": axis}
        onnx_model = create_onnx_model(
            op_type="Flatten",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_output_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="flatten_test",
            input_names=["input_0"],
            output_names=["output_0"],
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        input_dict = {"input_0": input_data}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], f"Output mismatch for dtype={dtype}, opset={opset_version}"

    @pytest.mark.parametrize("opset_version", [1, 9, 11, 13, 21, 23])
    def test_flatten_value_equality(self, opset_version):
        """Test that Flatten preserves all values correctly."""
        input_shape = (2, 3, 4, 5)
        axis = 2
        expected_output_shape = _calculate_flatten_output_shape(input_shape, axis)

        # Create test data with known values
        input_data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)

        # Create ONNX model
        attrs = {"axis": axis}
        onnx_model = create_onnx_model(
            op_type="Flatten",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=opset_version,
            node_name="flatten_test",
            input_names=["input_0"],
            output_names=["output_0"],
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        input_dict = {"input_0": input_data}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], f"Output mismatch for opset={opset_version}"

        # Verify values are preserved (flattened correctly)
        tir_output = comparison["tir_outputs"]["output_0"]
        onnx_output = comparison["onnx_outputs"]["output_0"]

        # Reshape both to 1D for comparison
        tir_flat = tir_output.flatten()
        onnx_flat = onnx_output.flatten()

        np.testing.assert_array_equal(
            tir_flat, onnx_flat, err_msg=f"Value mismatch: TIR output does not match ONNX output"
        )

        # Verify the flattened output matches the original input when reshaped back
        input_flat = input_data.flatten()
        np.testing.assert_array_equal(
            tir_flat, input_flat, err_msg=f"Value mismatch: Flattened output does not preserve input values"
        )

    def test_flatten_graph_structure(self):
        """Test that Flatten is correctly converted to ReshapeNode."""
        input_shape = (2, 3, 4)
        axis = 1
        expected_output_shape = _calculate_flatten_output_shape(input_shape, axis)

        # Create ONNX model
        attrs = {"axis": axis}
        onnx_model = create_onnx_model(
            op_type="Flatten",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=13,
            node_name="flatten_test",
            input_names=["input_0"],
            output_names=["output_0"],
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify graph structure
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"

        # tir_graph.nodes is a list, get the first (and only) node
        node = tir_graph.nodes[0]
        assert node.op_type == "Reshape", f"Expected Reshape node, got {node.op_type}. Flatten should map to Reshape."

        assert node.name == "flatten_test", f"Expected node name 'flatten_test', got {node.name}"

        # node.inputs and node.outputs are OrderedDicts, check keys
        assert list(node.inputs.keys()) == ["input_0"], f"Expected inputs ['input_0'], got {list(node.inputs.keys())}"

        # Check original output name (before sanitization)
        assert list(node.original_outputs) == [
            "output_0"
        ], f"Expected outputs ['output_0'], got {list(node.original_outputs)}"

        # Verify shape attribute
        assert "shape" in node.attrs, "Reshape node should have 'shape' attribute"
        assert (
            node.attrs["shape"] == expected_output_shape
        ), f"Expected shape {expected_output_shape}, got {node.attrs['shape']}"
