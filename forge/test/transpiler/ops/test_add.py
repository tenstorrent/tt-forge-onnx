# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Add operations.
Tests all broadcasting cases, opset versions, dtypes, and edge cases.
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.core.exceptions import ConversionError
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
    verify_tir_graph_structure,
)


# ============================================================================
# HELPER METHODS FOR CREATING ADD MODELS
# ============================================================================


def _create_add_model(
    opset_version,
    input_shapes,
    input_dtypes=None,
    output_shape=None,
    output_dtype=None,
    attrs=None,
    node_name="add_node",
):
    """
    Helper to create Add ONNX model.

    Args:
        opset_version: ONNX opset version
        input_shapes: List of two input shapes [(shape_a), (shape_b)]
        input_dtypes: List of two input dtypes (default: FLOAT for both)
        output_shape: Output shape (default: inferred from inputs)
        output_dtype: Output dtype (default: same as inputs)
        attrs: Additional attributes (broadcast, axis for opset 1-6)
        node_name: Name for the Add node
    """
    if input_dtypes is None:
        input_dtypes = [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT]
    if output_dtype is None:
        output_dtype = input_dtypes[0]
    if attrs is None:
        attrs = {}
    if output_shape is None:
        # Infer output shape (for broadcasting, take max of each dimension)
        shape_a, shape_b = input_shapes[0], input_shapes[1]
        max_len = max(len(shape_a), len(shape_b))
        shape_a_padded = [1] * (max_len - len(shape_a)) + list(shape_a)
        shape_b_padded = [1] * (max_len - len(shape_b)) + list(shape_b)
        output_shape = tuple(max(a, b) for a, b in zip(shape_a_padded, shape_b_padded))

    return create_onnx_model(
        op_type="Add",
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=[output_shape],
        output_dtypes=[output_dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name=node_name,
    )


# ============================================================================
# TEST CASES: BASIC ADDITION (SAME SHAPES)
# ============================================================================


@pytest.mark.transpiler
class TestAddBasic:
    """Test basic addition with same shapes."""

    def test_add_1d_same_shape(self):
        """Test Add with 1D tensors of same shape."""
        opset = 13
        input_shapes = [(3,), (3,)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "input_1": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.array([11.0, 22.0, 33.0], dtype=np.float32)
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_2d_same_shape(self):
        """Test Add with 2D tensors of same shape."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.array([[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]], dtype=np.float32)
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_3d_same_shape(self):
        """Test Add with 3D tensors of same shape."""
        opset = 13
        input_shapes = [(2, 2, 3), (2, 2, 3)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((2, 2, 3), dtype=np.float32),
            "input_1": np.ones((2, 2, 3), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.ones((2, 2, 3), dtype=np.float32) * 3.0
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)


# ============================================================================
# TEST CASES: BROADCASTING (OPSET 7+)
# ============================================================================


@pytest.mark.transpiler
class TestAddBroadcasting:
    """Test broadcasting cases (OPSET 7+)."""

    def test_add_scalar_broadcasting(self):
        """Test Add with scalar broadcasting."""
        opset = 13
        input_shapes = [(2, 3), ()]  # 2D tensor + scalar

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array(10.0, dtype=np.float32),  # Scalar
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.array([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]], dtype=np.float32)
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_1d_broadcasting_suffix(self):
        """Test Add with 1D tensor broadcasting (suffix matching)."""
        opset = 13
        input_shapes = [(2, 3), (3,)]  # 2D tensor + 1D tensor

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_1d_broadcasting_dimension_1(self):
        """Test Add with 1D tensor broadcasting (dimension of size 1)."""
        opset = 13
        input_shapes = [(3, 4), (3, 1)]  # 2D tensor + 2D tensor with dim=1

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((3, 4), dtype=np.float32),
            "input_1": np.array([[10.0], [20.0], [30.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result: each row gets added the corresponding value
        expected = np.array(
            [[11.0, 11.0, 11.0, 11.0], [21.0, 21.0, 21.0, 21.0], [31.0, 31.0, 31.0, 31.0]], dtype=np.float32
        )
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_3d_broadcasting(self):
        """Test Add with 3D broadcasting."""
        opset = 13
        input_shapes = [(2, 2, 2), (2, 2)]  # 3D tensor + 2D tensor

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((2, 2, 2), dtype=np.float32),
            "input_1": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_4d_broadcasting(self):
        """Test Add with 4D broadcasting."""
        opset = 13
        input_shapes = [(2, 3, 4, 5), (3, 4, 5)]  # 4D tensor + 3D tensor

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((2, 3, 4, 5), dtype=np.float32),
            "input_1": np.ones((3, 4, 5), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.ones((2, 3, 4, 5), dtype=np.float32) * 3.0
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_multiple_dimension_1(self):
        """Test Add with multiple dimensions of size 1."""
        opset = 13
        input_shapes = [(5, 1, 4), (1, 3, 1)]  # Multiple dims of size 1

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((5, 1, 4), dtype=np.float32),
            "input_1": np.ones((1, 3, 1), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result shape
        assert comparison["tir_outputs"]["output_0"].shape == (5, 3, 4)

    def test_add_broadcasting_1d_to_2d_row(self):
        """Test Add with 1D broadcasting to 2D (row vector)."""
        opset = 13
        input_shapes = [(3, 4), (4,)]  # 2D + 1D (row)

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.ones((3, 4), dtype=np.float32),
            "input_1": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_broadcasting_1d_to_2d_column(self):
        """Test Add with 1D broadcasting to 2D (column vector)."""
        opset = 13
        input_shapes = [(3, 4), (3, 1)]  # 2D + column vector

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.ones((3, 4), dtype=np.float32),
            "input_1": np.array([[10.0], [20.0], [30.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_broadcasting_2d_to_3d(self):
        """Test Add with 2D broadcasting to 3D."""
        opset = 13
        input_shapes = [(2, 3, 4), (3, 4)]  # 3D + 2D

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.ones((2, 3, 4), dtype=np.float32),
            "input_1": np.ones((3, 4), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_broadcasting_2d_to_4d(self):
        """Test Add with 2D broadcasting to 4D."""
        opset = 13
        input_shapes = [(2, 3, 4, 5), (4, 5)]  # 4D + 2D

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.ones((2, 3, 4, 5), dtype=np.float32),
            "input_1": np.ones((4, 5), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_broadcasting_1d_to_3d(self):
        """Test Add with 1D broadcasting to 3D."""
        opset = 13
        input_shapes = [(2, 3, 4), (4,)]  # 3D + 1D

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.ones((2, 3, 4), dtype=np.float32),
            "input_1": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_broadcasting_scalar_to_3d(self):
        """Test Add with scalar broadcasting to 3D."""
        opset = 13
        input_shapes = [(2, 3, 4), ()]  # 3D + scalar

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {"input_0": np.ones((2, 3, 4), dtype=np.float32), "input_1": np.array(10.0, dtype=np.float32)}

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.ones((2, 3, 4), dtype=np.float32) * 11.0
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_broadcasting_scalar_to_4d(self):
        """Test Add with scalar broadcasting to 4D."""
        opset = 13
        input_shapes = [(2, 3, 4, 5), ()]  # 4D + scalar

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {"input_0": np.ones((2, 3, 4, 5), dtype=np.float32), "input_1": np.array(5.0, dtype=np.float32)}

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_broadcasting_complex_pattern(self):
        """Test Add with complex broadcasting pattern."""
        opset = 13
        input_shapes = [(1, 2, 1, 4), (2, 1, 1)]  # Complex pattern - compatible shapes

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.ones((1, 2, 1, 4), dtype=np.float32),
            "input_1": np.ones((2, 1, 1), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result shape
        assert comparison["tir_outputs"]["output_0"].shape == (1, 2, 1, 4)

    def test_add_broadcasting_5d(self):
        """Test Add with 5D broadcasting."""
        opset = 13
        input_shapes = [(2, 3, 4, 5, 6), (4, 5, 6)]  # 5D + 3D

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.ones((2, 3, 4, 5, 6), dtype=np.float32),
            "input_1": np.ones((4, 5, 6), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"


# ============================================================================
# TEST CASES: OPSET 1-6 (LIMITED BROADCASTING)
# ============================================================================


@pytest.mark.transpiler
class TestAddOpset1To6:
    """Test Add with OPSET 1-6 (limited broadcasting with attributes)."""

    @pytest.mark.skip(reason="ONNX Runtime does not support OPSET 1")
    def test_add_opset_1_no_broadcast_same_shape(self):
        """Test Add OPSET 1 with same shapes (no broadcast needed)."""
        opset = 1
        input_shapes = [(2, 3), (2, 3)]
        attrs = {"broadcast": 0}  # No broadcasting

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.skip(reason="ONNX Runtime does not support OPSET 1")
    def test_add_opset_1_broadcast_scalar(self):
        """Test Add OPSET 1 with scalar broadcasting."""
        opset = 1
        input_shapes = [(2, 3), ()]  # Scalar
        attrs = {"broadcast": 1}  # Enable broadcasting

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array(10.0, dtype=np.float32),  # Scalar
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.skip(reason="ONNX Runtime does not support OPSET 1")
    def test_add_opset_1_broadcast_suffix_match(self):
        """Test Add OPSET 1 with suffix matching (default)."""
        opset = 1
        input_shapes = [(2, 3, 4), (3, 4)]  # Suffix match
        attrs = {"broadcast": 1}  # Enable broadcasting

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((2, 3, 4), dtype=np.float32),
            "input_1": np.ones((3, 4), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.skip(
        reason="OPSET 1-6 axis-based broadcasting requires tensor reshaping for PyTorch compatibility - not yet implemented"
    )
    def test_add_opset_6_broadcast_with_axis(self):
        """Test Add OPSET 6 with axis attribute."""
        opset = 6
        input_shapes = [(2, 3, 4, 5), (3, 4)]  # Axis-specified match
        attrs = {"broadcast": 1, "axis": 1}  # Enable broadcasting, start at axis 1

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((2, 3, 4, 5), dtype=np.float32),
            "input_1": np.ones((3, 4), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.skip(
        reason="OPSET 1-6 axis-based broadcasting requires tensor reshaping for PyTorch compatibility - not yet implemented"
    )
    def test_add_opset_6_broadcast_axis_0(self):
        """Test Add OPSET 6 with axis=0."""
        opset = 6
        input_shapes = [(2, 3, 4), (2,)]  # Axis 0 match
        attrs = {"broadcast": 1, "axis": 0}  # Enable broadcasting, start at axis 0

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((2, 3, 4), dtype=np.float32),
            "input_1": np.array([10.0, 20.0], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.skip(reason="ONNX Runtime does not support OPSET 1")
    def test_add_opset_1_broadcast_axis_2(self):
        """Test Add OPSET 1 with axis=2."""
        opset = 1
        input_shapes = [(2, 3, 4, 5), (4, 5)]  # Axis 2 match
        attrs = {"broadcast": 1, "axis": 2}  # Enable broadcasting, start at axis 2

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.ones((2, 3, 4, 5), dtype=np.float32),
            "input_1": np.ones((4, 5), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.skip(reason="ONNX Runtime does not fully support OPSET 6 broadcasting in all cases")
    def test_add_opset_6_broadcast_suffix_various_shapes(self):
        """Test Add OPSET 6 with suffix matching for various shapes."""
        opset = 6
        test_cases = [
            ((2, 3, 4, 5), (5,)),  # 4D + 1D
            ((2, 3, 4, 5), (4, 5)),  # 4D + 2D
            ((2, 3, 4, 5), (3, 4, 5)),  # 4D + 3D
            ((10, 20, 30), (20, 30)),  # 3D + 2D
            ((10, 20, 30), (30,)),  # 3D + 1D
        ]

        for shape_a, shape_b in test_cases:
            input_shapes = [shape_a, shape_b]
            attrs = {"broadcast": 1}  # Suffix matching (no axis)

            model = _create_add_model(opset, input_shapes, attrs=attrs)
            transpiler = ONNXToForgeTranspiler(validate_model=True)
            tir_graph = transpiler.transpile(model)

            input_data = {
                "input_0": np.ones(shape_a, dtype=np.float32),
                "input_1": np.ones(shape_b, dtype=np.float32) * 2.0,
            }

            comparison = compare_tir_with_onnx(tir_graph, model, input_data)
            assert len(comparison["errors"]) == 0, f"Shape {shape_a} + {shape_b} failed: {comparison['errors']}"
            assert comparison["matches"]["output_0"], f"Shape {shape_a} + {shape_b} outputs should match"

    @pytest.mark.skip(
        reason="OPSET 1-6 axis-based broadcasting requires tensor reshaping for PyTorch compatibility - not yet implemented"
    )
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_add_opset_6_broadcast_various_axes(self, axis):
        """Test Add OPSET 6 with various axis values."""
        opset = 6

        # Create compatible shapes for each axis
        if axis == 0:
            input_shapes = [(5, 3, 4), (5,)]
        elif axis == 1:
            input_shapes = [(2, 3, 4, 5), (3, 4)]
        else:  # axis == 2
            input_shapes = [(2, 3, 4, 5), (4, 5)]

        attrs = {"broadcast": 1, "axis": axis}

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        shape_a, shape_b = input_shapes
        input_data = {
            "input_0": np.ones(shape_a, dtype=np.float32),
            "input_1": np.ones(shape_b, dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Axis {axis} failed: {comparison['errors']}"
        assert comparison["matches"]["output_0"], f"Axis {axis} outputs should match"


# ============================================================================
# TEST CASES: ALL SUPPORTED DTYPES
# ============================================================================


@pytest.mark.transpiler
class TestAddDtypes:
    """Test Add with all supported dtypes."""

    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (onnx.TensorProto.FLOAT, np.float32),
            (onnx.TensorProto.DOUBLE, np.float64),
            (onnx.TensorProto.INT32, np.int32),
            (onnx.TensorProto.INT64, np.int64),
        ],
    )
    def test_add_basic_dtypes(self, dtype, np_dtype):
        """Test Add with basic dtypes (float32, double, int32, int64)."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [dtype, dtype]

        model = _create_add_model(opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
            "input_1": np.array([[10, 20, 30], [40, 50, 60]], dtype=np_dtype),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (onnx.TensorProto.UINT8, np.uint8),
        ],
    )
    def test_add_unsigned_int_dtypes(self, dtype, np_dtype):
        """Test Add with unsigned integer dtypes (OPSET 14+)."""
        opset = 14
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [dtype, dtype]

        model = _create_add_model(opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
            "input_1": np.array([[10, 20, 30], [40, 50, 60]], dtype=np_dtype),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.skip(reason="PyTorch does not support uint16, uint32, uint64 for add operation")
    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (onnx.TensorProto.UINT16, np.uint16),
            (onnx.TensorProto.UINT32, np.uint32),
            (onnx.TensorProto.UINT64, np.uint64),
        ],
    )
    def test_add_unsigned_int_dtypes_unsupported(self, dtype, np_dtype):
        """Test Add with unsupported unsigned integer dtypes."""

    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (onnx.TensorProto.INT8, np.int8),
            (onnx.TensorProto.INT16, np.int16),
        ],
    )
    def test_add_small_int_dtypes(self, dtype, np_dtype):
        """Test Add with small integer dtypes (int8, int16) (OPSET 14+)."""
        opset = 14
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [dtype, dtype]

        model = _create_add_model(opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
            "input_1": np.array([[10, 20, 30], [40, 50, 60]], dtype=np_dtype),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.skip(reason="ONNX Runtime requires actual float16 input, not float32")
    def test_add_float16(self):
        """Test Add with float16."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT16]

        model = _create_add_model(opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Use float32 for numpy (float16 support may vary)
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data, rtol=1e-2, atol=1e-2)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    @pytest.mark.skip(reason="ONNX Runtime does not support bfloat16 for Add operation")
    def test_add_bfloat16(self):
        """Test Add with bfloat16 (OPSET 13+)."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16]

        model = _create_add_model(opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Use float32 for numpy (bfloat16 support may vary)
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data, rtol=1e-2, atol=1e-2)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"


# ============================================================================
# TEST CASES: ERROR CASES (SHOULD RAISE ERRORS)
# ============================================================================


@pytest.mark.transpiler
class TestAddErrors:
    """Test error cases that should raise exceptions."""

    def test_add_incompatible_shapes_opset_7(self):
        """Test Add with incompatible shapes in OPSET 7+ (should raise error)."""
        opset = 13
        input_shapes = [(2, 3), (2, 4)]  # Incompatible: 3 vs 4, neither is 1

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)

        # This should raise an error during transpilation
        with pytest.raises(ConversionError) as exc_info:
            tir_graph = transpiler.transpile(model)

        # Verify error message mentions broadcasting
        assert "broadcast" in str(exc_info.value).lower() or "compatible" in str(exc_info.value).lower()

    def test_add_opset_1_no_broadcast_incompatible(self):
        """Test Add OPSET 1 with incompatible shapes and broadcast=0 (should raise error)."""
        opset = 1
        input_shapes = [(2, 3), (2, 4)]  # Incompatible shapes
        attrs = {"broadcast": 0}  # No broadcasting

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)

        # This should raise an error during transpilation
        with pytest.raises(ConversionError) as exc_info:
            tir_graph = transpiler.transpile(model)

        # Verify error message mentions broadcast
        assert "broadcast" in str(exc_info.value).lower()

    def test_add_opset_1_broadcast_axis_invalid(self):
        """Test Add OPSET 1 with invalid axis (should raise error)."""
        opset = 1
        input_shapes = [(2, 3), (3, 4)]  # Shapes that don't match at axis 0
        attrs = {"broadcast": 1, "axis": 0}  # Invalid axis

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)

        # This should raise an error during transpilation
        with pytest.raises(ConversionError) as exc_info:
            tir_graph = transpiler.transpile(model)

        # Verify error message mentions axis or broadcasting
        assert "axis" in str(exc_info.value).lower() or "broadcast" in str(exc_info.value).lower()


# ============================================================================
# TEST CASES: EDGE CASES
# ============================================================================


@pytest.mark.transpiler
class TestAddEdgeCases:
    """Test edge cases for Add operation."""

    def test_add_zero_tensor(self):
        """Test Add with zero tensor."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.zeros((2, 3), dtype=np.float32),
            "input_1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_negative_values(self):
        """Test Add with negative values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.array([[9.0, 18.0, 27.0], [36.0, 45.0, 54.0]], dtype=np.float32)
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_large_shapes(self):
        """Test Add with large shapes."""
        opset = 13
        input_shapes = [(100, 50), (100, 50)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.random.randn(100, 50).astype(np.float32),
            "input_1": np.random.randn(100, 50).astype(np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_single_element_tensor(self):
        """Test Add with single element tensors."""
        opset = 13
        input_shapes = [(1,), (1,)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {"input_0": np.array([5.0], dtype=np.float32), "input_1": np.array([10.0], dtype=np.float32)}

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.array([15.0], dtype=np.float32)
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_all_dimensions_1(self):
        """Test Add with all dimensions of size 1."""
        opset = 13
        input_shapes = [(1, 1, 1), (1, 1, 1)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[[5.0]]], dtype=np.float32),
            "input_1": np.array([[[10.0]]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_very_large_first_dimension(self):
        """Test Add with very large first dimension."""
        opset = 13
        input_shapes = [(1000, 10), (1000, 10)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.random.randn(1000, 10).astype(np.float32),
            "input_1": np.random.randn(1000, 10).astype(np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_very_small_tensors(self):
        """Test Add with very small tensors (1x1)."""
        opset = 13
        input_shapes = [(1, 1), (1, 1)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {"input_0": np.array([[5.0]], dtype=np.float32), "input_1": np.array([[10.0]], dtype=np.float32)}

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_very_long_1d(self):
        """Test Add with very long 1D tensors."""
        opset = 13
        input_shapes = [(10000,), (10000,)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.arange(10000, dtype=np.float32),
            "input_1": np.arange(10000, dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_mixed_positive_negative(self):
        """Test Add with mixed positive and negative values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, -20.0, 30.0], [-40.0, 50.0, -60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.array([[9.0, -18.0, 27.0], [-36.0, 45.0, -54.0]], dtype=np.float32)
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], expected, rtol=1e-5)

    def test_add_very_small_values(self):
        """Test Add with very small floating point values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1e-10, 2e-10, 3e-10], [4e-10, 5e-10, 6e-10]], dtype=np.float32),
            "input_1": np.array([[1e-10, 2e-10, 3e-10], [4e-10, 5e-10, 6e-10]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    def test_add_very_large_values(self):
        """Test Add with very large floating point values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1e10, 2e10, 3e10], [4e10, 5e10, 6e10]], dtype=np.float32),
            "input_1": np.array([[1e10, 2e10, 3e10], [4e10, 5e10, 6e10]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize(
        "rank_a, rank_b",
        [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
        ],
    )
    def test_add_all_rank_combinations(self, rank_a, rank_b):
        """Test Add with all combinations of tensor ranks."""
        opset = 13

        # Create compatible shapes for broadcasting
        # For broadcasting to work, the smaller rank tensor should match the suffix of the larger one
        # OR both should have compatible dimensions (equal or one is 1)
        if rank_a >= rank_b:
            # A has more or equal dimensions - B should match suffix of A
            shape_a = tuple(range(3, 3 + rank_a))  # Start from 3 to avoid 1s
            shape_b = tuple(range(3, 3 + rank_b))
            # Make sure B matches suffix of A
            shape_b = shape_a[-rank_b:] if rank_b > 0 else shape_b
        else:
            # B has more dimensions - A should match suffix of B
            shape_b = tuple(range(3, 3 + rank_b))  # Start from 3 to avoid 1s
            shape_a = tuple(range(3, 3 + rank_a))
            # Make sure A matches suffix of B
            shape_a = shape_b[-rank_a:] if rank_a > 0 else shape_a

        input_shapes = [shape_a, shape_b]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.ones(shape_a, dtype=np.float32),
            "input_1": np.ones(shape_b, dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Rank {rank_a} + {rank_b} failed: {comparison['errors']}"
        assert comparison["matches"]["output_0"], f"Rank {rank_a} + {rank_b} outputs should match"


# ============================================================================
# TEST CASES: OPSET VERSION COMPARISON
# ============================================================================


@pytest.mark.transpiler
class TestAddOpsetVersions:
    """Test Add across different opset versions."""

    @pytest.mark.parametrize("opset", [7, 13, 14])
    def test_add_same_shape_all_opsets(self, opset):
        """Test Add with same shapes across all opset versions."""
        input_shapes = [(2, 3), (2, 3)]
        attrs = {}

        model = _create_add_model(opset, input_shapes, attrs=attrs)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"OPSET {opset} comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], f"OPSET {opset} outputs should match"

    @pytest.mark.parametrize("opset", [7, 13, 14])
    def test_add_broadcasting_opsets_7_plus(self, opset):
        """Test Add broadcasting in OPSET 7+."""
        input_shapes = [(2, 3), (3,)]  # Broadcasting case

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"OPSET {opset} comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], f"OPSET {opset} outputs should match"


# ============================================================================
# TEST CASES: GRAPH STRUCTURE VERIFICATION
# ============================================================================


@pytest.mark.transpiler
class TestAddGraphStructure:
    """Test Add graph structure and node creation."""

    def test_add_graph_structure(self):
        """Test that Add creates correct graph structure."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Verify graph structure
        verification = verify_tir_graph_structure(tir_graph, model, expected_op_types=["Add"])
        assert verification["node_count_match"], "Node count should match"
        assert verification["input_count_match"], "Input count should match"
        assert verification["output_count_match"], "Output count should match"
        assert "Add" in verification["node_types"], "Should have Add node"

    def test_add_node_attributes(self):
        """Test that Add node has correct attributes."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_add_model(opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Find Add node
        add_nodes = [node for node in tir_graph.nodes if node.op_type == "Add"]
        assert len(add_nodes) == 1, "Should have exactly one Add node"

        add_node = add_nodes[0]
        assert len(add_node.inputs) == 2, "Add node should have 2 inputs"
        assert len(add_node.outputs) == 1, "Add node should have 1 output"
        assert add_node.forge_op_function_name == "forge.op.Add", "Should have correct Forge op name"
