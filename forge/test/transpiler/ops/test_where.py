# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Where operation.
Tests different input shapes, dtypes, opset versions, broadcasting, and edge cases.
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.utils.binary_ops import compute_broadcasted_shape_multi
from forge.transpiler.utils.exceptions import ConversionError
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestWhere:
    """Comprehensive test cases for Where operation."""

    @pytest.mark.parametrize("opset_version", [9, 16])
    @pytest.mark.parametrize(
        "condition_shape, x_shape, y_shape",
        [
            # Same shapes
            ((2, 3), (2, 3), (2, 3)),
            ((1, 4), (1, 4), (1, 4)),
            ((3, 4, 5), (3, 4, 5), (3, 4, 5)),
            # Broadcasting: condition broadcasts
            ((1, 3), (2, 3), (2, 3)),
            ((2, 1), (2, 3), (2, 3)),
            # Broadcasting: X and Y broadcast
            ((2, 3), (1, 3), (2, 1)),
            ((2, 3), (2, 1), (1, 3)),
            # Scalar-like broadcasting
            ((1,), (2, 3), (2, 3)),
            ((2, 3), (1,), (2, 3)),
            # Higher dimensions
            ((1, 2, 3), (4, 2, 3), (4, 2, 3)),
            ((2, 3, 4, 5), (2, 3, 4, 5), (2, 3, 4, 5)),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
        ],
    )
    def test_where_basic(self, opset_version, condition_shape, x_shape, y_shape, dtype):
        """Test basic Where operations across opset versions with various shapes and dtypes."""
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
            onnx.TensorProto.BFLOAT16: np.float32,  # Use float32 as approximation
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # Compute expected output shape (broadcasted shape)
        output_shape = compute_broadcasted_shape_multi(condition_shape, x_shape, y_shape)
        assert (
            output_shape is not None
        ), f"Shapes should be compatible: condition={condition_shape}, x={x_shape}, y={y_shape}"

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, dtype, dtype],
            output_shapes=[output_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="where_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"

        where_nodes = [n for n in tir_graph.nodes if n.op_type == "Where"]
        assert len(where_nodes) == 1, (
            f"Expected 1 WhereNode, got {len(where_nodes)}. " f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        )

        where_node = where_nodes[0]
        assert where_node.forge_op_name == "Where", f"Expected forge_op_name='Where', got '{where_node.forge_op_name}'"
        assert len(where_node.input_names) == 3, f"Expected 3 inputs, got {len(where_node.input_names)}"
        assert len(where_node.output_names) == 1, f"Expected 1 output, got {len(where_node.output_names)}"

        # Create test inputs
        condition = np.random.choice([True, False], size=condition_shape).astype(np.bool_)
        x = np.random.randn(*x_shape).astype(np_dtype)
        y = np.random.randn(*y_shape).astype(np_dtype)

        input_data = {
            "input_0": condition,
            "input_1": x,
            "input_2": y,
        }

        # Compare with ONNX runtime
        # Use more lenient tolerance for FLOAT16 due to lower precision
        if dtype == onnx.TensorProto.FLOAT16:
            rtol_val, atol_val = 1e-3, 1e-3
        else:
            rtol_val, atol_val = 1e-6, 1e-6

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol_val, atol=atol_val)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, condition_shape={condition_shape}, "
            f"x_shape={x_shape}, y_shape={y_shape}, dtype={dtype}"
        )
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison['matches']}"

    @pytest.mark.parametrize("opset_version", [9, 16])
    def test_where_broadcasting(self, opset_version):
        """Test Where with various broadcasting scenarios."""
        # Test 1: Scalar condition
        condition_shape = (1,)
        x_shape = (2, 3)
        y_shape = (2, 3)
        output_shape = (2, 3)

        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT],
            output_shapes=[output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=opset_version,
            node_name="where_broadcast_scalar",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        condition = np.array([True], dtype=np.bool_)
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        y = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)

        input_data = {"input_0": condition, "input_1": x, "input_2": y}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Test 2: Condition broadcasts along first dimension
        condition_shape = (1, 3)
        x_shape = (2, 3)
        y_shape = (2, 3)
        output_shape = (2, 3)

        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT],
            output_shapes=[output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=opset_version,
            node_name="where_broadcast_dim",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        condition = np.array([[True, False, True]], dtype=np.bool_)
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        y = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)

        input_data = {"input_0": condition, "input_1": x, "input_2": y}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    @pytest.mark.parametrize("opset_version", [9, 16])
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
            # Skip BOOL dtype - ONNX Runtime does not support BOOL for X/Y inputs in Where operation
            # onnx.TensorProto.BOOL,
        ],
    )
    def test_where_dtypes(self, opset_version, dtype):
        """Test Where with different dtypes."""
        condition_shape = (2, 3)
        x_shape = (2, 3)
        y_shape = (2, 3)
        output_shape = (2, 3)

        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
            onnx.TensorProto.BOOL: np.bool_,
            onnx.TensorProto.BFLOAT16: np.float32,  # Use float32 as approximation
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, dtype, dtype],
            output_shapes=[output_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="where_dtype_test",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        condition = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
        if dtype == onnx.TensorProto.BOOL:
            x = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
            y = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)
        else:
            x = np.random.randint(1, 10, size=x_shape).astype(np_dtype)
            y = np.random.randint(10, 20, size=y_shape).astype(np_dtype)

        input_data = {"input_0": condition, "input_1": x, "input_2": y}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    @pytest.mark.parametrize("opset_version", [9, 16])
    def test_where_edge_cases(self, opset_version):
        """Test Where with edge cases."""
        # Test 1: All True condition
        condition_shape = (2, 2)
        x_shape = (2, 2)
        y_shape = (2, 2)
        output_shape = (2, 2)

        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT],
            output_shapes=[output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=opset_version,
            node_name="where_all_true",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        condition = np.ones(condition_shape, dtype=np.bool_)
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

        input_data = {"input_0": condition, "input_1": x, "input_2": y}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output matches X (all True condition)
        tir_output = comparison["tir_outputs"]["output_0"]
        np.testing.assert_allclose(tir_output, x, rtol=1e-6, atol=1e-6)

        # Test 2: All False condition
        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT],
            output_shapes=[output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=opset_version,
            node_name="where_all_false",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        condition = np.zeros(condition_shape, dtype=np.bool_)
        input_data = {"input_0": condition, "input_1": x, "input_2": y}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output matches Y (all False condition)
        tir_output = comparison["tir_outputs"]["output_0"]
        np.testing.assert_allclose(tir_output, y, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("opset_version", [9, 16])
    def test_where_high_dimensional(self, opset_version):
        """Test Where with high-dimensional tensors."""
        condition_shape = (2, 3, 4, 5)
        x_shape = (2, 3, 4, 5)
        y_shape = (2, 3, 4, 5)
        output_shape = (2, 3, 4, 5)

        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT],
            output_shapes=[output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=opset_version,
            node_name="where_high_dim",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        condition = np.random.choice([True, False], size=condition_shape).astype(np.bool_)
        x = np.random.randn(*x_shape).astype(np.float32)
        y = np.random.randn(*y_shape).astype(np.float32)

        input_data = {"input_0": condition, "input_1": x, "input_2": y}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    def test_where_validation_errors(self):
        """Test Where converter validation errors."""
        # Test 1: Wrong number of inputs
        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[(2, 3), (2, 3)],  # Only 2 inputs instead of 3
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT],
            output_shapes=[(2, 3)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=9,
            node_name="where_wrong_inputs",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=False)
        with pytest.raises(ConversionError, match="Expected 3 inputs"):
            transpiler.transpile(onnx_model)

        # Test 2: Condition not boolean
        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[(2, 3), (2, 3), (2, 3)],
            input_dtypes=[
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT,
            ],  # Condition is FLOAT, not BOOL
            output_shapes=[(2, 3)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=9,
            node_name="where_wrong_condition_type",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=False)
        with pytest.raises(ConversionError, match="condition input must be boolean"):
            transpiler.transpile(onnx_model)

        # Test 3: X and Y have different dtypes
        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[(2, 3), (2, 3), (2, 3)],
            input_dtypes=[
                onnx.TensorProto.BOOL,
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.INT32,
            ],  # X and Y have different dtypes
            output_shapes=[(2, 3)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=9,
            node_name="where_different_dtypes",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=False)
        with pytest.raises(ConversionError, match="X and Y inputs must have the same dtype"):
            transpiler.transpile(onnx_model)

        # Test 4: Incompatible shapes for broadcasting
        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[(2, 3), (5, 7), (2, 3)],  # X shape (5, 7) incompatible with condition (2, 3)
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT],
            output_shapes=[(2, 3)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=9,
            node_name="where_incompatible_shapes",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=False)
        with pytest.raises(ConversionError, match="not compatible for broadcasting"):
            transpiler.transpile(onnx_model)

    @pytest.mark.parametrize("opset_version", [9, 16])
    def test_where_scalar_broadcasting(self, opset_version):
        """Test Where with scalar (0D) inputs."""
        # Test scalar condition
        condition_shape = ()  # Scalar
        x_shape = (2, 3)
        y_shape = (2, 3)
        output_shape = (2, 3)

        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT],
            output_shapes=[output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=opset_version,
            node_name="where_scalar_condition",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        condition = np.array(True, dtype=np.bool_)  # Scalar
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        y = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)

        input_data = {"input_0": condition, "input_1": x, "input_2": y}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    @pytest.mark.parametrize("opset_version", [16])
    def test_where_bfloat16(self, opset_version):
        """Test Where with bfloat16 dtype (opset 16+).

        Note: ONNX Runtime does not support BFLOAT16 for Where operation,
        so this test only validates that the converter can handle BFLOAT16
        without comparing against ONNX Runtime.
        """
        condition_shape = (2, 3)
        x_shape = (2, 3)
        y_shape = (2, 3)
        output_shape = (2, 3)

        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16],
            output_shapes=[output_shape],
            output_dtypes=[onnx.TensorProto.BFLOAT16],
            attrs={},
            opset_version=opset_version,
            node_name="where_bfloat16",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - converter should handle BFLOAT16
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
        where_nodes = [n for n in tir_graph.nodes if n.op_type == "Where"]
        assert len(where_nodes) == 1, f"Expected 1 WhereNode, got {len(where_nodes)}"

        # Skip ONNX Runtime comparison since it doesn't support BFLOAT16 for Where
        pytest.skip("ONNX Runtime does not support BFLOAT16 for Where operation")

    @pytest.mark.parametrize("opset_version", [9, 16])
    def test_where_complex_broadcasting(self, opset_version):
        """Test Where with complex broadcasting scenarios."""
        # Test: condition (1, 1, 3), x (2, 1, 3), y (1, 4, 1) -> output (2, 4, 3)
        condition_shape = (1, 1, 3)
        x_shape = (2, 1, 3)
        y_shape = (1, 4, 1)
        output_shape = (2, 4, 3)

        onnx_model = create_onnx_model(
            op_type="Where",
            input_shapes=[condition_shape, x_shape, y_shape],
            input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT],
            output_shapes=[output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=opset_version,
            node_name="where_complex_broadcast",
        )

        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        condition = np.random.choice([True, False], size=condition_shape).astype(np.bool_)
        x = np.random.randn(*x_shape).astype(np.float32)
        y = np.random.randn(*y_shape).astype(np.float32)

        input_data = {"input_0": condition, "input_1": x, "input_2": y}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
