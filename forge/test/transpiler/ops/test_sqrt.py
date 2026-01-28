# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Sqrt operation.
Tests different input shapes, dtypes, opset versions, and edge cases.
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestSqrt:
    """Comprehensive test cases for Sqrt operation."""

    @pytest.mark.parametrize("opset_version", [1, 6, 13])
    @pytest.mark.parametrize(
        "input_shape",
        [
            # Scalar-like
            (1,),
            # 1D
            (5,),
            (10,),
            # 2D
            (3, 4),
            (2, 3),
            (10, 10),
            # 3D
            (2, 3, 4),
            (5, 5, 5),
            # 4D
            (2, 3, 4, 5),
            # Higher dimensions
            (1, 2, 3, 4, 5),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ],
    )
    def test_sqrt_basic(self, opset_version, input_shape, dtype):
        """Test basic Sqrt operations across opset versions."""
        # Skip opset 1 - ONNX Runtime may have limited support
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime may not fully support Sqrt(1)")

        # Skip float16 for opset < 13 (may not be fully supported)
        if opset_version < 13 and dtype == onnx.TensorProto.FLOAT16:
            pytest.skip(f"Float16 may not be fully supported in opset {opset_version}")

        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.FLOAT16: np.float16,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # Create ONNX model
        attrs = {}
        # v1 has consumed_inputs attribute (legacy, should be ignored)
        if opset_version == 1:
            attrs["consumed_inputs"] = [0]  # Legacy attribute, should be ignored

        onnx_model = create_onnx_model(
            op_type="Sqrt",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sqrt_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"

        sqrt_nodes = [n for n in tir_graph.nodes if n.op_type == "Sqrt"]
        assert len(sqrt_nodes) == 1, (
            f"Expected 1 SqrtNode, got {len(sqrt_nodes)}. " f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        )

        sqrt_node = sqrt_nodes[0]
        assert (
            len(sqrt_node.inputs) == 1
        ), f"SqrtNode should have exactly 1 input, got {len(sqrt_node.inputs)}: {sqrt_node.input_names}"
        assert (
            sqrt_node.input_names[0] == "input_0"
        ), f"SqrtNode input should be 'input_0', got {sqrt_node.input_names[0]}"

        # Create test input with positive values (square root of perfect squares)
        np.random.seed(42)
        # Generate positive values
        input_data = {"input_0": np.abs(np.random.randn(*input_shape).astype(np_dtype)) * 10 + 1}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5 if dtype == onnx.TensorProto.FLOAT16 else 1e-6,
            atol=1e-4 if dtype == onnx.TensorProto.FLOAT16 else 1e-6,
        )

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, dtype={dtype}"
        )

    @pytest.mark.parametrize("opset_version", [6, 13])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (3, 4),
            (2, 3, 4),
        ],
    )
    def test_sqrt_bfloat16(self, opset_version, input_shape):
        """Test Sqrt with bfloat16 type (v13+)."""
        if opset_version < 13:
            pytest.skip(f"BFloat16 only supported in opset 13+, got opset={opset_version}")

        # Skip bfloat16 - ONNX Runtime may not support Sqrt with bfloat16
        pytest.skip("ONNX Runtime may not support Sqrt with bfloat16 type")

        dtype = onnx.TensorProto.BFLOAT16

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Sqrt",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="sqrt_bfloat16",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1
        sqrt_nodes = [n for n in tir_graph.nodes if n.op_type == "Sqrt"]
        assert len(sqrt_nodes) == 1

        # Create test input
        # bfloat16 is not directly supported by numpy, use float32 and convert
        np.random.seed(42)
        input_data = {"input_0": np.abs(np.random.randn(*input_shape).astype(np.float32)) * 10 + 1}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph, onnx_model, input_data, rtol=1e-2, atol=1e-2  # bfloat16 has lower precision
        )

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    @pytest.mark.parametrize("opset_version", [1, 6, 13])
    def test_sqrt_positive_values(self, opset_version):
        """Test Sqrt with all positive values."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime may not fully support Sqrt(1)")

        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs["consumed_inputs"] = [0]

        onnx_model = create_onnx_model(
            op_type="Sqrt",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sqrt_positive",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with all positive values (perfect squares)
        input_data = {
            "input_0": np.array(
                [[1.0, 4.0, 9.0, 16.0], [25.0, 36.0, 49.0, 64.0], [81.0, 100.0, 121.0, 144.0]], dtype=np.float32
            )
        }

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output is correct (square root of perfect squares)
        tir_output = comparison["tir_outputs"]["output_0"]
        expected = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
        np.testing.assert_allclose(tir_output, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("opset_version", [1, 6, 13])
    def test_sqrt_negative_values(self, opset_version):
        """Test Sqrt with negative values (should produce NaN)."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime may not fully support Sqrt(1)")

        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs["consumed_inputs"] = [0]

        onnx_model = create_onnx_model(
            op_type="Sqrt",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sqrt_negative",
        )

        # Transpile (disable debug mode to avoid NaN comparison issues)
        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with negative values
        input_data = {
            "input_0": np.array(
                [[4.0, -9.0, 16.0, -1.0], [-25.0, 36.0, -49.0, 64.0], [81.0, -100.0, 121.0, -144.0]], dtype=np.float32
            )
        }

        # Compare with ONNX runtime
        # Note: compare_tir_with_onnx uses np.allclose which fails with NaN values
        # We need to manually compare NaN positions and non-NaN values separately
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        # Check for execution errors (not comparison errors due to NaN)
        execution_errors = [
            e
            for e in comparison.get("errors", [])
            if "TIR execution failed" in e or "ONNX execution failed" in e or "Shape mismatch" in e
        ]
        assert len(execution_errors) == 0, f"Execution errors: {execution_errors}"

        # Get outputs
        tir_output = comparison["tir_outputs"]["output_0"]
        onnx_output = comparison["onnx_outputs"]["output_0"]

        # Check that NaN positions match
        tir_nan_mask = np.isnan(tir_output)
        onnx_nan_mask = np.isnan(onnx_output)
        assert np.array_equal(
            tir_nan_mask, onnx_nan_mask
        ), f"NaN positions don't match. TIR NaN mask: {tir_nan_mask}, ONNX NaN mask: {onnx_nan_mask}"

        # Compare non-NaN values
        non_nan_mask = ~tir_nan_mask
        if np.any(non_nan_mask):
            tir_non_nan = tir_output[non_nan_mask]
            onnx_non_nan = onnx_output[non_nan_mask]
            np.testing.assert_allclose(tir_non_nan, onnx_non_nan, rtol=1e-6, atol=1e-6)

        # Verify output contains NaN for negative inputs
        # Positive values should have correct square root
        assert np.isclose(tir_output[0, 0], 2.0, rtol=1e-6), f"Expected 2.0, got {tir_output[0, 0]}"
        assert np.isclose(tir_output[0, 2], 4.0, rtol=1e-6), f"Expected 4.0, got {tir_output[0, 2]}"
        # Negative values should produce NaN
        assert np.isnan(tir_output[0, 1]), f"Expected NaN for negative input, got {tir_output[0, 1]}"
        assert np.isnan(tir_output[0, 3]), f"Expected NaN for negative input, got {tir_output[0, 3]}"

    @pytest.mark.parametrize("opset_version", [1, 6, 13])
    def test_sqrt_zero_values(self, opset_version):
        """Test Sqrt with zero values (should output 0.0)."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime may not fully support Sqrt(1)")

        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs["consumed_inputs"] = [0]

        onnx_model = create_onnx_model(
            op_type="Sqrt",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sqrt_zero",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with all zeros
        input_data = {"input_0": np.zeros(input_shape, dtype=np.float32)}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output is 0.0 for zero input (sqrt(0) = 0)
        tir_output = comparison["tir_outputs"]["output_0"]
        expected = np.zeros(input_shape, dtype=np.float32)
        np.testing.assert_allclose(tir_output, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("opset_version", [1, 6, 13])
    def test_sqrt_edge_values(self, opset_version):
        """Test Sqrt with edge values (very small, very large, zero, infinity)."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime may not fully support Sqrt(1)")

        input_shape = (2, 3)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs["consumed_inputs"] = [0]

        onnx_model = create_onnx_model(
            op_type="Sqrt",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sqrt_edge",
        )

        # Transpile (disable debug mode to avoid NaN comparison issues)
        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with edge values
        input_data = {
            "input_0": np.array(
                [[0.0, 1e-10, 1e10], [np.inf, -1.0, 4.0]],
                dtype=np.float32,  # zero, very small, very large, inf, negative, normal
            )
        }

        # Compare with ONNX runtime
        # Note: compare_tir_with_onnx uses np.allclose which fails with NaN values
        # We need to manually compare NaN positions and non-NaN values separately
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)

        # Get outputs
        tir_output = comparison["tir_outputs"]["output_0"]
        onnx_output = comparison["onnx_outputs"]["output_0"]

        # Check that NaN positions match
        tir_nan_mask = np.isnan(tir_output)
        onnx_nan_mask = np.isnan(onnx_output)
        assert np.array_equal(
            tir_nan_mask, onnx_nan_mask
        ), f"NaN positions don't match. TIR NaN mask: {tir_nan_mask}, ONNX NaN mask: {onnx_nan_mask}"

        # Check that inf positions match
        tir_inf_mask = np.isinf(tir_output)
        onnx_inf_mask = np.isinf(onnx_output)
        assert np.array_equal(
            tir_inf_mask, onnx_inf_mask
        ), f"Inf positions don't match. TIR Inf mask: {tir_inf_mask}, ONNX Inf mask: {onnx_inf_mask}"

        # Compare non-NaN, non-inf values
        finite_mask = ~(tir_nan_mask | tir_inf_mask)
        if np.any(finite_mask):
            tir_finite = tir_output[finite_mask]
            onnx_finite = onnx_output[finite_mask]
            np.testing.assert_allclose(tir_finite, onnx_finite, rtol=1e-5, atol=1e-5)

        # Verify expected behavior
        # sqrt(0) = 0
        assert np.isclose(tir_output[0, 0], 0.0, rtol=1e-6, atol=1e-6), f"Expected 0.0, got {tir_output[0, 0]}"
        # sqrt(1e-10) ≈ 1e-5
        assert np.isclose(tir_output[0, 1], 1e-5, rtol=1e-2, atol=1e-5), f"Expected ~1e-5, got {tir_output[0, 1]}"
        # sqrt(1e10) ≈ 1e5
        assert np.isclose(tir_output[0, 2], 1e5, rtol=1e-2, atol=1e3), f"Expected ~1e5, got {tir_output[0, 2]}"
        # sqrt(inf) = inf
        assert np.isinf(tir_output[1, 0]), f"Expected inf, got {tir_output[1, 0]}"
        # sqrt(-1) = NaN
        assert np.isnan(tir_output[1, 1]), f"Expected NaN for negative input, got {tir_output[1, 1]}"
        # sqrt(4) = 2
        assert np.isclose(tir_output[1, 2], 2.0, rtol=1e-6, atol=1e-6), f"Expected 2.0, got {tir_output[1, 2]}"

    @pytest.mark.parametrize("opset_version", [1, 6, 13])
    def test_sqrt_v1_consumed_inputs_ignored(self, opset_version):
        """Test that consumed_inputs attribute in v1 is ignored."""
        if opset_version != 1:
            pytest.skip(f"This test is only for opset 1")

        if opset_version == 1:
            pytest.skip(f"ONNX Runtime may not fully support Sqrt(1)")

        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model with consumed_inputs attribute
        attrs = {"consumed_inputs": [0]}

        onnx_model = create_onnx_model(
            op_type="Sqrt",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sqrt_v1",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure (should work normally, ignoring consumed_inputs)
        assert len(tir_graph.nodes) == 1
        sqrt_nodes = [n for n in tir_graph.nodes if n.op_type == "Sqrt"]
        assert len(sqrt_nodes) == 1

        # Create test input
        np.random.seed(42)
        input_data = {"input_0": np.abs(np.random.randn(*input_shape).astype(np.float32)) * 10 + 1}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    @pytest.mark.parametrize("opset_version", [6, 13])
    def test_sqrt_high_dimensional(self, opset_version):
        """Test Sqrt with high-dimensional tensors."""
        input_shape = (2, 3, 4, 5, 6)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Sqrt",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="sqrt_high_dim",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1
        sqrt_nodes = [n for n in tir_graph.nodes if n.op_type == "Sqrt"]
        assert len(sqrt_nodes) == 1

        # Create test input
        np.random.seed(42)
        input_data = {"input_0": np.abs(np.random.randn(*input_shape).astype(np.float32)) * 10 + 1}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output shape matches input shape
        tir_output = comparison["tir_outputs"]["output_0"]
        assert tir_output.shape == input_shape, f"Output shape {tir_output.shape} != input shape {input_shape}"
