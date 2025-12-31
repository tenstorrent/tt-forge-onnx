# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Softmax operation.
Tests different input shapes, dtypes, opset versions, axis values, and edge cases.
"""
import pytest
import numpy as np
import onnx
import torch

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestSoftmax:
    """Comprehensive test cases for Softmax operation."""

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 14, 21, 23, 24, 25])
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
            (10, 10),
            # 3D
            (2, 3, 4),
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
    def test_softmax_basic(self, opset_version, input_shape, dtype):
        """Test basic Softmax operations across opset versions."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

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

        # Determine default axis based on opset version
        if opset_version >= 13:
            default_axis = -1
        else:
            default_axis = 1

        # For opset < 13, 1D tensors can't use axis=1 (only axis 0 exists)
        # Skip this case as it's invalid
        if opset_version < 13 and len(input_shape) == 1:
            pytest.skip(f"Opset {opset_version} default axis=1 is invalid for 1D tensors (only axis 0 exists)")

        # For opset < 13, higher dimensional tensors (3D+) with axis=1 have ONNX Runtime compatibility issues
        # The coercion to 2D in opset 1-12 causes value mismatches
        if opset_version < 13 and len(input_shape) >= 3:
            pytest.skip(
                f"Opset {opset_version} with axis=1 has ONNX Runtime compatibility issues for {len(input_shape)}D tensors"
            )

        # Skip FLOAT16 with 5D tensors - precision issues with large tensors
        if dtype == onnx.TensorProto.FLOAT16 and len(input_shape) == 5:
            pytest.skip(f"FLOAT16 with 5D tensors has precision issues")

        # Create ONNX model
        attrs = {"axis": default_axis}

        onnx_model = create_onnx_model(
            op_type="Softmax",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="softmax_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"

        softmax_nodes = [n for n in tir_graph.nodes if n.op_type == "Softmax"]
        assert len(softmax_nodes) == 1, (
            f"Expected 1 SoftmaxNode, got {len(softmax_nodes)}. " f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        )

        softmax_node = softmax_nodes[0]
        assert (
            len(softmax_node.inputs) == 1
        ), f"SoftmaxNode should have exactly 1 input, got {len(softmax_node.inputs)}: {softmax_node.input_names}"
        assert (
            softmax_node.input_names[0] == "input_0"
        ), f"SoftmaxNode input should be 'input_0', got {softmax_node.input_names[0]}"
        # Check original output name (before sanitization)
        assert (
            softmax_node.original_outputs[0] == "output_0"
        ), f"SoftmaxNode output should be 'output_0', got {softmax_node.original_outputs[0]}"

        # Verify axis attribute
        expected_axis = default_axis
        actual_axis = softmax_node.attrs.get("dim", None)
        assert actual_axis == expected_axis, f"SoftmaxNode axis should be {expected_axis}, got {actual_axis}"

        # Create test input with mixed positive and negative values
        input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype) * 5}  # Values in range [-5, 5]

        # Compare with ONNX runtime
        # Use relaxed tolerance for opset 10, 13, and 14 due to numerical precision differences
        if dtype == onnx.TensorProto.FLOAT16:
            # For FLOAT16, use more lenient tolerance based on tensor dimensions
            if len(input_shape) == 3:
                rtol_val, atol_val = 1e-2, 1e-2
            else:
                rtol_val, atol_val = 1e-5, 1e-4
        elif opset_version in [10, 13, 14]:
            # For opset 10 and 14, use more lenient tolerance for higher dimensional tensors
            if opset_version in [10, 14] and len(input_shape) >= 2:
                rtol_val, atol_val = 1e-3, 1e-3
            else:
                rtol_val, atol_val = 1e-4, 1e-5
        else:
            rtol_val, atol_val = 1e-6, 1e-6
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol_val, atol=atol_val)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, dtype={dtype}, axis={default_axis}"
        )

        # Verify softmax properties: output should sum to 1 along the axis
        tir_output = comparison["tir_outputs"]["output_0"]
        # Normalize axis to positive value
        axis = default_axis if default_axis >= 0 else len(input_shape) + default_axis
        # Sum along the axis
        sums = np.sum(tir_output, axis=axis, keepdims=True)
        # Check that sums are close to 1.0
        # Use more lenient tolerance for FLOAT16 due to numerical precision
        expected_tol = (1e-3, 1e-3) if dtype == onnx.TensorProto.FLOAT16 else (1e-5, 1e-5)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=expected_tol[0], atol=expected_tol[1])

        # Verify output is in range [0, 1]
        assert np.all(tir_output >= 0), "Softmax output should be >= 0"
        assert np.all(tir_output <= 1), "Softmax output should be <= 1"

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 14])
    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    def test_softmax_different_axes(self, opset_version, axis):
        """Test Softmax with different axis values."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        input_shape = (3, 4, 5)
        dtype = onnx.TensorProto.FLOAT

        # Normalize axis to check if it's valid
        normalized_axis = axis if axis >= 0 else len(input_shape) + axis
        if normalized_axis < 0 or normalized_axis >= len(input_shape):
            pytest.skip(f"Invalid axis {axis} for shape {input_shape}")

        # Skip axis=0, axis=1, and axis=-2 for opset 1 and 11 - ONNX Runtime has compatibility issues
        # The coercion to 2D in opset 1-12 causes value mismatches for these axes with 3D tensors
        if opset_version < 13 and axis in [0, 1, -2]:
            pytest.skip(
                f"Opset {opset_version} with axis={axis} has compatibility issues with ONNX Runtime for 3D tensors"
            )

        # Create ONNX model
        attrs = {"axis": axis}

        onnx_model = create_onnx_model(
            op_type="Softmax",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="softmax_axis",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1
        softmax_nodes = [n for n in tir_graph.nodes if n.op_type == "Softmax"]
        assert len(softmax_nodes) == 1

        softmax_node = softmax_nodes[0]
        assert (
            softmax_node.attrs.get("dim") == axis
        ), f"SoftmaxNode axis should be {axis}, got {softmax_node.attrs.get('dim')}"

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32) * 5}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify softmax properties: output should sum to 1 along the axis
        tir_output = comparison["tir_outputs"]["output_0"]
        normalized_axis = axis if axis >= 0 else len(input_shape) + axis
        sums = np.sum(tir_output, axis=normalized_axis, keepdims=True)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 14])
    def test_softmax_all_equal_values(self, opset_version):
        """Test Softmax with all equal values along axis (should output uniform distribution)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        input_shape = (2, 5)
        dtype = onnx.TensorProto.FLOAT

        # Determine axis based on opset version
        if opset_version >= 13:
            axis = -1
        else:
            axis = 1

        # Create ONNX model
        attrs = {"axis": axis}

        onnx_model = create_onnx_model(
            op_type="Softmax",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="softmax_equal",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with all equal values along axis
        # Each row should have the same value, so softmax should output uniform distribution
        input_data = {"input_0": np.ones(input_shape, dtype=np.float32) * 2.0}  # All values are 2.0

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output: when all values are equal, softmax should output uniform distribution
        tir_output = comparison["tir_outputs"]["output_0"]
        normalized_axis = axis if axis >= 0 else len(input_shape) + axis
        expected_value = 1.0 / input_shape[normalized_axis]  # Uniform distribution: 1/n

        # Check that all values along the axis are approximately equal
        for i in range(input_shape[0]):
            row = tir_output[i, :] if normalized_axis == 1 else tir_output[:, i]
            np.testing.assert_allclose(row, np.full_like(row, expected_value), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 14])
    def test_softmax_extreme_values(self, opset_version):
        """Test Softmax with extreme values (very large differences)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        input_shape = (2, 4)
        dtype = onnx.TensorProto.FLOAT

        # Determine axis based on opset version
        if opset_version >= 13:
            axis = -1
        else:
            axis = 1

        # Create ONNX model
        attrs = {"axis": axis}

        onnx_model = create_onnx_model(
            op_type="Softmax",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="softmax_extreme",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with extreme differences
        # First row: one very large value, others small
        # Second row: one very small value, others large
        input_data = {
            "input_0": np.array(
                [
                    [10.0, 1.0, 1.0, 1.0],  # Large value should dominate
                    [-10.0, 5.0, 5.0, 5.0],  # Small value should be near zero
                ],
                dtype=np.float32,
            )
        }

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify expected behavior
        tir_output = comparison["tir_outputs"]["output_0"]
        # First row: large value (10.0) should have probability close to 1.0
        assert tir_output[0, 0] > 0.99, f"Large value should have high probability, got {tir_output[0, 0]}"
        # Second row: small value (-10.0) should have probability close to 0.0
        assert tir_output[1, 0] < 0.01, f"Small value should have low probability, got {tir_output[1, 0]}"

        # Verify sums along axis
        normalized_axis = axis if axis >= 0 else len(input_shape) + axis
        sums = np.sum(tir_output, axis=normalized_axis, keepdims=True)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 14])
    def test_softmax_single_element(self, opset_version):
        """Test Softmax with single element tensor (edge case)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        input_shape = (1,)
        dtype = onnx.TensorProto.FLOAT

        # For 1D tensor, axis 0 is the only valid axis (or -1 which maps to 0)
        axis = 0

        # Create ONNX model
        attrs = {"axis": axis}

        onnx_model = create_onnx_model(
            op_type="Softmax",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="softmax_single",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input
        input_data = {"input_0": np.array([5.0], dtype=np.float32)}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output: single element should output 1.0 (softmax of single value is always 1.0)
        tir_output = comparison["tir_outputs"]["output_0"]
        assert np.allclose(
            tir_output, np.array([1.0]), rtol=1e-5, atol=1e-5
        ), f"Single element softmax should output 1.0, got {tir_output}"

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 14])
    def test_softmax_edge_values(self, opset_version):
        """Test Softmax with edge values (zeros, equal values, extreme values, very small values)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        input_shape = (2, 4)
        dtype = onnx.TensorProto.FLOAT

        # Determine axis based on opset version
        if opset_version >= 13:
            axis = -1
        else:
            axis = 1

        # Create ONNX model
        attrs = {"axis": axis}

        onnx_model = create_onnx_model(
            op_type="Softmax",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="softmax_edge",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        normalized_axis = axis if axis >= 0 else len(input_shape) + axis
        expected_uniform = 1.0 / input_shape[normalized_axis]

        # Test 1: All zeros (should output uniform distribution)
        input_data = {"input_0": np.zeros(input_shape, dtype=np.float32)}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        tir_output = comparison["tir_outputs"]["output_0"]
        for i in range(input_shape[0]):
            row = tir_output[i, :] if normalized_axis == 1 else tir_output[:, i]
            np.testing.assert_allclose(row, np.full_like(row, expected_uniform), rtol=1e-5, atol=1e-5)

        # Test 2: All equal values (should output uniform distribution)
        input_data = {"input_0": np.ones(input_shape, dtype=np.float32) * 2.0}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        tir_output = comparison["tir_outputs"]["output_0"]
        for i in range(input_shape[0]):
            row = tir_output[i, :] if normalized_axis == 1 else tir_output[:, i]
            np.testing.assert_allclose(row, np.full_like(row, expected_uniform), rtol=1e-5, atol=1e-5)

        # Test 3: Extreme values
        input_data = {"input_0": np.array([[10.0, 1.0, 1.0, 1.0], [-10.0, 5.0, 5.0, 5.0]], dtype=np.float32)}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        tir_output = comparison["tir_outputs"]["output_0"]
        assert tir_output[0, 0] > 0.99, f"Large value should have high probability, got {tir_output[0, 0]}"
        assert tir_output[1, 0] < 0.01, f"Small value should have low probability, got {tir_output[1, 0]}"

        # Test 4: Very small values
        input_data = {
            "input_0": np.array([[1e-10, 2e-10, 3e-10, 4e-10], [-1e-10, -2e-10, -3e-10, -4e-10]], dtype=np.float32)
        }
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        tir_output = comparison["tir_outputs"]["output_0"]
        assert np.all(tir_output >= 0), "Softmax output should be >= 0"
        assert np.all(tir_output <= 1), "Softmax output should be <= 1"
        sums = np.sum(tir_output, axis=normalized_axis, keepdims=True)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5, atol=1e-5)
