# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX comparison operations: Equal, Greater, Less, GreaterOrEqual, LessOrEqual.
Tests all broadcasting cases, opset versions, dtypes, and edge cases.
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.utils.exceptions import ConversionError
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
    verify_tir_graph_structure,
)


# ============================================================================
# HELPER METHODS FOR CREATING COMPARISON MODELS
# ============================================================================


def _create_comparison_model(
    op_type,
    opset_version,
    input_shapes,
    input_dtypes=None,
    output_shape=None,
    output_dtype=None,
    attrs=None,
    node_name=None,
):
    """
    Helper to create comparison ONNX model (Equal, Greater, Less, GreaterOrEqual, LessOrEqual).

    Args:
        op_type: Operation type ('Equal', 'Greater', 'Less', 'GreaterOrEqual', 'LessOrEqual')
        opset_version: ONNX opset version
        input_shapes: List of two input shapes [(shape_a), (shape_b)]
        input_dtypes: List of two input dtypes (default: FLOAT for both)
        output_shape: Output shape (default: inferred from inputs)
        output_dtype: Output dtype (default: BOOL for comparison ops)
        attrs: Additional attributes (broadcast, axis for opset 1-6)
        node_name: Name for the node (default: {op_type.lower()}_node)
    """
    if input_dtypes is None:
        input_dtypes = [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT]
    if output_dtype is None:
        # Comparison ops always output BOOL
        output_dtype = onnx.TensorProto.BOOL
    if attrs is None:
        attrs = {}
    if node_name is None:
        node_name = f"{op_type.lower()}_node"
    if output_shape is None:
        # Infer output shape (for broadcasting, take max of each dimension)
        shape_a, shape_b = input_shapes[0], input_shapes[1]
        max_len = max(len(shape_a), len(shape_b))
        shape_a_padded = [1] * (max_len - len(shape_a)) + list(shape_a)
        shape_b_padded = [1] * (max_len - len(shape_b)) + list(shape_b)
        output_shape = tuple(max(a, b) for a, b in zip(shape_a_padded, shape_b_padded))

    return create_onnx_model(
        op_type=op_type,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=[output_shape],
        output_dtypes=[output_dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name=node_name,
    )


# ============================================================================
# TEST CASES: BASIC OPERATIONS (SAME SHAPES)
# ============================================================================


@pytest.mark.transpiler
class TestComparisonBasic:
    """Test basic comparison operations with same shapes."""

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_1d_same_shape(self, op_type):
        """Test comparison operations with 1D tensors of same shape."""
        opset = 13
        input_shapes = [(3,), (3,)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "input_1": np.array([1.0, 5.0, 2.0], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result based on operation
        if op_type == "Equal":
            expected = np.array([True, False, False], dtype=bool)
        elif op_type == "Greater":
            expected = np.array([False, False, True], dtype=bool)
        elif op_type == "Less":
            expected = np.array([False, True, False], dtype=bool)
        elif op_type == "GreaterOrEqual":
            expected = np.array([True, False, True], dtype=bool)
        elif op_type == "LessOrEqual":
            expected = np.array([True, True, False], dtype=bool)

        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_2d_same_shape(self, op_type):
        """Test comparison operations with 2D tensors of same shape."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_3d_same_shape(self, op_type):
        """Test comparison operations with 3D tensors of same shape."""
        opset = 13
        input_shapes = [(2, 3, 4), (2, 3, 4)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((2, 3, 4), dtype=np.float32),
            "input_1": np.ones((2, 3, 4), dtype=np.float32) * 2.0,
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_equal_values(self, op_type):
        """Test comparison operations with equal input values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data - both inputs are the same
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result based on operation
        result = comparison["tir_outputs"]["output_0"]
        if op_type == "Equal":
            assert np.all(result), "All values should be True for Equal"
        elif op_type == "Greater":
            assert not np.any(result), "All values should be False for Greater"
        elif op_type == "Less":
            assert not np.any(result), "All values should be False for Less"
        elif op_type == "GreaterOrEqual":
            assert np.all(result), "All values should be True for GreaterOrEqual"
        elif op_type == "LessOrEqual":
            assert np.all(result), "All values should be True for LessOrEqual"


# ============================================================================
# TEST CASES: BROADCASTING (OPSET 7+)
# ============================================================================


@pytest.mark.transpiler
class TestComparisonBroadcasting:
    """Test comparison operations with broadcasting (OPSET 7+)."""

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_scalar_broadcasting(self, op_type):
        """Test comparison operations with scalar broadcasting."""
        opset = 13
        input_shapes = [(2, 3), ()]  # Scalar

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array(2.5, dtype=np.float32),  # Scalar
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_1d_broadcasting_suffix(self, op_type):
        """Test comparison operations with 1D broadcasting (suffix matching)."""
        opset = 13
        input_shapes = [(2, 3), (3,)]  # 2D + 1D (suffix match)

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_1d_broadcasting_dimension_1(self, op_type):
        """Test comparison operations with 1D broadcasting (dimension of size 1)."""
        opset = 13
        input_shapes = [(3, 4), (3, 1)]  # 2D tensor + 2D tensor with dim=1

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((3, 4), dtype=np.float32),
            "input_1": np.array([[2.0], [1.0], [3.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_3d_broadcasting(self, op_type):
        """Test comparison operations with 3D broadcasting."""
        opset = 13
        input_shapes = [(2, 2, 2), (2, 2)]  # 3D tensor + 2D tensor

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Test data
        input_data = {
            "input_0": np.ones((2, 2, 2), dtype=np.float32),
            "input_1": np.array([[2.0, 1.0], [0.5, 3.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_4d_broadcasting(self, op_type):
        """Test comparison operations with 4D broadcasting."""
        opset = 13
        input_shapes = [(2, 3, 4, 5), (3, 4, 5)]  # 4D tensor + 3D tensor

        model = _create_comparison_model(op_type, opset, input_shapes)
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

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_multiple_dimension_1(self, op_type):
        """Test comparison operations with multiple dimensions of size 1."""
        opset = 13
        input_shapes = [(5, 1, 4), (1, 3, 1)]  # Multiple dims of size 1

        model = _create_comparison_model(op_type, opset, input_shapes)
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


# ============================================================================
# TEST CASES: ALL SUPPORTED DTYPES
# ============================================================================


@pytest.mark.transpiler
class TestComparisonDtypes:
    """Test comparison operations with all supported dtypes."""

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (onnx.TensorProto.FLOAT, np.float32),
            (onnx.TensorProto.DOUBLE, np.float64),
            (onnx.TensorProto.INT32, np.int32),
            (onnx.TensorProto.INT64, np.int64),
        ],
    )
    def test_comparison_basic_dtypes(self, op_type, dtype, np_dtype):
        """Test comparison operations with basic dtypes (float32, double, int32, int64)."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [dtype, dtype]

        model = _create_comparison_model(op_type, opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
            "input_1": np.array([[1, 5, 2], [4, 3, 6]], dtype=np_dtype),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_unsigned_int_dtypes(self, op_type):
        """Test comparison operations with unsigned integer dtypes (OPSET 14+)."""
        opset = 14
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [onnx.TensorProto.UINT8, onnx.TensorProto.UINT8]

        model = _create_comparison_model(op_type, opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            "input_1": np.array([[1, 5, 2], [4, 3, 6]], dtype=np.uint8),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (onnx.TensorProto.INT8, np.int8),
            (onnx.TensorProto.INT16, np.int16),
        ],
    )
    def test_comparison_small_int_dtypes(self, op_type, dtype, np_dtype):
        """Test comparison operations with small integer dtypes (int8, int16) (OPSET 14+)."""
        opset = 14
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [dtype, dtype]

        model = _create_comparison_model(op_type, opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
            "input_1": np.array([[1, 5, 2], [4, 3, 6]], dtype=np_dtype),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"


# ============================================================================
# TEST CASES: ERROR CASES (SHOULD RAISE ERRORS)
# ============================================================================


@pytest.mark.transpiler
class TestComparisonErrors:
    """Test error cases that should raise exceptions."""

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_incompatible_shapes_opset_7(self, op_type):
        """Test comparison operations with incompatible shapes in OPSET 7+ (should raise error)."""
        opset = 13
        input_shapes = [(2, 3), (2, 4)]  # Incompatible: 3 vs 4, neither is 1

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)

        # This should raise an error during transpilation
        with pytest.raises(ConversionError) as exc_info:
            tir_graph = transpiler.transpile(model)

        # Verify error message mentions broadcasting
        assert "broadcast" in str(exc_info.value).lower() or "compatible" in str(exc_info.value).lower()


# ============================================================================
# TEST CASES: EDGE CASES
# ============================================================================


@pytest.mark.transpiler
class TestComparisonEdgeCases:
    """Test edge cases for comparison operations."""

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_zero_tensor(self, op_type):
        """Test comparison operations with zero tensor."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.zeros((2, 3), dtype=np.float32),
            "input_1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_negative_values(self, op_type):
        """Test comparison operations with negative values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32),
            "input_1": np.array([[1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_single_element_tensor(self, op_type):
        """Test comparison operations with single element tensors."""
        opset = 13
        input_shapes = [(1,), (1,)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {"input_0": np.array([5.0], dtype=np.float32), "input_1": np.array([10.0], dtype=np.float32)}

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_large_values(self, op_type):
        """Test comparison operations with large values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1e10, 2e10, 3e10], [4e10, 5e10, 6e10]], dtype=np.float32),
            "input_1": np.array([[1e10, 1e10, 3e10], [4e10, 5e10, 6e10]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_small_values(self, op_type):
        """Test comparison operations with very small values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1e-10, 2e-10, 3e-10], [4e-10, 5e-10, 6e-10]], dtype=np.float32),
            "input_1": np.array([[1e-10, 1e-10, 3e-10], [4e-10, 5e-10, 6e-10]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"


# ============================================================================
# TEST CASES: OPERATION-SPECIFIC TESTS
# ============================================================================


@pytest.mark.transpiler
class TestEqualSpecific:
    """Test Equal-specific cases."""

    def test_equal_all_true(self):
        """Test Equal with all values equal."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("Equal", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result (all True)
        expected = np.ones((2, 3), dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)

    def test_equal_all_false(self):
        """Test Equal with all values different."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("Equal", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result (all False)
        expected = np.zeros((2, 3), dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)


@pytest.mark.transpiler
class TestGreaterSpecific:
    """Test Greater-specific cases."""

    def test_greater_all_true(self):
        """Test Greater with all values in first input greater."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("Greater", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
            "input_1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result (all True)
        expected = np.ones((2, 3), dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)

    def test_greater_all_false(self):
        """Test Greater with all values in first input less or equal."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("Greater", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result (all False)
        expected = np.zeros((2, 3), dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)


@pytest.mark.transpiler
class TestLessSpecific:
    """Test Less-specific cases."""

    def test_less_all_true(self):
        """Test Less with all values in first input less."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("Less", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result (all True)
        expected = np.ones((2, 3), dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)

    def test_less_all_false(self):
        """Test Less with all values in first input greater or equal."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("Less", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
            "input_1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result (all False)
        expected = np.zeros((2, 3), dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)


@pytest.mark.transpiler
class TestGreaterOrEqualSpecific:
    """Test GreaterOrEqual-specific cases."""

    def test_greater_or_equal_all_true(self):
        """Test GreaterOrEqual with all values in first input greater or equal."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("GreaterOrEqual", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
            "input_1": np.array([[1.0, 20.0, 3.0], [40.0, 5.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result (all True)
        expected = np.ones((2, 3), dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)

    def test_greater_or_equal_mixed(self):
        """Test GreaterOrEqual with mixed results."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("GreaterOrEqual", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.array([[True, False, True], [True, True, True]], dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)


@pytest.mark.transpiler
class TestLessOrEqualSpecific:
    """Test LessOrEqual-specific cases."""

    def test_less_or_equal_all_true(self):
        """Test LessOrEqual with all values in first input less or equal."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("LessOrEqual", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[10.0, 2.0, 30.0], [40.0, 50.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result (all True)
        expected = np.ones((2, 3), dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)

    def test_less_or_equal_mixed(self):
        """Test LessOrEqual with mixed results."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model("LessOrEqual", opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        input_data = {
            "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "input_1": np.array([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], dtype=np.float32),
        }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], "Outputs should match"

        # Verify result
        expected = np.array([[True, True, False], [True, False, True]], dtype=bool)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)


# ============================================================================
# TEST CASES: OPSET VERSION COMPARISON
# ============================================================================


@pytest.mark.transpiler
class TestComparisonOpsetVersions:
    """Test comparison operations across different opset versions."""

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    @pytest.mark.parametrize("opset", [7, 12, 13, 16, 19])
    def test_comparison_same_shape_all_opsets(self, op_type, opset):
        """Test comparison operations with same shapes across all opset versions."""
        # Skip opset 7 for operators that don't support it
        if opset == 7 and op_type in ["Equal", "GreaterOrEqual", "LessOrEqual"]:
            pytest.skip(
                f"{op_type} not supported in opset 7 (Equal only supports int/bool, GreaterOrEqual/LessOrEqual don't exist)"
            )

        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # For opset 7 Equal, use int32 instead of float32
        if opset == 7 and op_type == "Equal":
            input_data = {
                "input_0": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
                "input_1": np.array([[1, 5, 2], [4, 3, 6]], dtype=np.int32),
            }
        else:
            input_data = {
                "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                "input_1": np.array([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], dtype=np.float32),
            }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"OPSET {opset} comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], f"OPSET {opset} outputs should match"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    @pytest.mark.parametrize("opset", [7, 12, 13, 16, 19])
    def test_comparison_broadcasting_opsets_7_plus(self, op_type, opset):
        """Test comparison broadcasting in OPSET 7+."""
        # Skip opset 7 for operators that don't support it
        if opset == 7 and op_type in ["Equal", "GreaterOrEqual", "LessOrEqual"]:
            pytest.skip(
                f"{op_type} not supported in opset 7 (Equal only supports int/bool, GreaterOrEqual/LessOrEqual don't exist)"
            )

        input_shapes = [(2, 3), (3,)]  # Broadcasting case

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # For opset 7 Equal, use int32 instead of float32
        if opset == 7 and op_type == "Equal":
            input_data = {
                "input_0": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
                "input_1": np.array([2, 3, 4], dtype=np.int32),
            }
        else:
            input_data = {
                "input_0": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                "input_1": np.array([2.0, 3.0, 4.0], dtype=np.float32),
            }

        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"OPSET {opset} comparison errors: {comparison['errors']}"
        assert comparison["matches"]["output_0"], f"OPSET {opset} outputs should match"


# ============================================================================
# TEST CASES: GRAPH STRUCTURE VERIFICATION
# ============================================================================


@pytest.mark.transpiler
class TestComparisonGraphStructure:
    """Test comparison graph structure and node creation."""

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_graph_structure(self, op_type):
        """Test that comparison operations create correct graph structure."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Verify graph structure
        verification = verify_tir_graph_structure(tir_graph, model, expected_op_types=[op_type])
        assert verification["node_count_match"], "Node count should match"
        assert verification["input_count_match"], "Input count should match"
        assert verification["output_count_match"], "Output count should match"
        assert op_type in verification["node_types"], f"Should have {op_type} node"

    @pytest.mark.parametrize("op_type", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"])
    def test_comparison_node_attributes(self, op_type):
        """Test that comparison nodes have correct attributes."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]

        model = _create_comparison_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        # Find comparison node
        comparison_nodes = [node for node in tir_graph.nodes if node.op_type == op_type]
        assert len(comparison_nodes) == 1, f"Should have exactly one {op_type} node"

        comparison_node = comparison_nodes[0]
        assert len(comparison_node.inputs) == 2, f"{op_type} node should have 2 inputs"
        assert len(comparison_node.outputs) == 1, f"{op_type} node should have 1 output"

        # Verify Forge op function names
        expected_forge_names = {
            "Equal": "forge.op.Equal",
            "Greater": "forge.op.Greater",
            "Less": "forge.op.Less",
            "GreaterOrEqual": "forge.op.GreaterEqual",
            "LessOrEqual": "forge.op.LessEqual",
        }
        assert (
            comparison_node.forge_op_function_name == expected_forge_names[op_type]
        ), f"Should have correct Forge op name for {op_type}"
