"""
Test cases for ONNX arithmetic operations: Add, Sub, Mul, Div.
Tests all broadcasting cases, opset versions, dtypes, and edge cases.
"""
import pytest
import numpy as np
import onnx
import torch
from loguru import logger

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.frontends.onnx.tests.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
    verify_tir_graph_structure,
    print_onnx_model,
    print_tir_graph
)


# ============================================================================
# HELPER METHODS FOR CREATING ARITHMETIC MODELS
# ============================================================================

def _create_arithmetic_model(op_type, opset_version, input_shapes, input_dtypes=None, 
                             output_shape=None, output_dtype=None,
                             attrs=None, node_name=None):
    """
    Helper to create arithmetic ONNX model (Add, Sub, Mul, Div).
    
    Args:
        op_type: Operation type ('Add', 'Sub', 'Mul', 'Div')
        opset_version: ONNX opset version
        input_shapes: List of two input shapes [(shape_a), (shape_b)]
        input_dtypes: List of two input dtypes (default: FLOAT for both)
        output_shape: Output shape (default: inferred from inputs)
        output_dtype: Output dtype (default: same as inputs)
        attrs: Additional attributes (broadcast, axis for opset 1-6)
        node_name: Name for the node (default: {op_type.lower()}_node)
    """
    if input_dtypes is None:
        input_dtypes = [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT]
    if output_dtype is None:
        output_dtype = input_dtypes[0]
    if attrs is None:
        attrs = {}
    if node_name is None:
        node_name = f'{op_type.lower()}_node'
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
        node_name=node_name
    )


# ============================================================================
# TEST CASES: BASIC OPERATIONS (SAME SHAPES)
# ============================================================================

class TestArithmeticBasic:
    """Test basic arithmetic operations with same shapes."""
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_1d_same_shape(self, op_type):
        """Test arithmetic operations with 1D tensors of same shape."""
        opset = 13
        input_shapes = [(3,), (3,)]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        input_data = {
            'input_0': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'input_1': np.array([10.0, 20.0, 30.0], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
        
        # Verify result based on operation
        if op_type == 'Add':
            expected = np.array([11.0, 22.0, 33.0], dtype=np.float32)
        elif op_type == 'Sub':
            expected = np.array([-9.0, -18.0, -27.0], dtype=np.float32)
        elif op_type == 'Mul':
            expected = np.array([10.0, 40.0, 90.0], dtype=np.float32)
        else:  # Div
            expected = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        
        np.testing.assert_allclose(comparison['tir_outputs']['output_0'], expected, rtol=1e-5)
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_2d_same_shape(self, op_type):
        """Test arithmetic operations with 2D tensors of same shape."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        input_data = {
            'input_0': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            'input_1': np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_3d_same_shape(self, op_type):
        """Test arithmetic operations with 3D tensors of same shape."""
        opset = 13
        input_shapes = [(2, 3, 4), (2, 3, 4)]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        input_data = {
            'input_0': np.ones((2, 3, 4), dtype=np.float32),
            'input_1': np.ones((2, 3, 4), dtype=np.float32) * 2.0
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"


# ============================================================================
# TEST CASES: BROADCASTING (OPSET 7+)
# ============================================================================

class TestArithmeticBroadcasting:
    """Test arithmetic operations with broadcasting (OPSET 7+)."""
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_scalar_broadcasting(self, op_type):
        """Test arithmetic operations with scalar broadcasting."""
        opset = 13
        input_shapes = [(2, 3), ()]  # Scalar
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        input_data = {
            'input_0': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            'input_1': np.array(10.0, dtype=np.float32)  # Scalar
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_1d_broadcasting_suffix(self, op_type):
        """Test arithmetic operations with 1D broadcasting (suffix matching)."""
        opset = 13
        input_shapes = [(2, 3), (3,)]  # 2D + 1D (suffix match)
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        input_data = {
            'input_0': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            'input_1': np.array([10.0, 20.0, 30.0], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_1d_broadcasting_dimension_1(self, op_type):
        """Test arithmetic operations with 1D broadcasting (dimension of size 1)."""
        opset = 13
        input_shapes = [(3, 4), (3, 1)]  # 2D tensor + 2D tensor with dim=1
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        input_data = {
            'input_0': np.ones((3, 4), dtype=np.float32),
            'input_1': np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_3d_broadcasting(self, op_type):
        """Test arithmetic operations with 3D broadcasting."""
        opset = 13
        input_shapes = [(2, 2, 2), (2, 2)]  # 3D tensor + 2D tensor
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        input_data = {
            'input_0': np.ones((2, 2, 2), dtype=np.float32),
            'input_1': np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_4d_broadcasting(self, op_type):
        """Test arithmetic operations with 4D broadcasting."""
        opset = 13
        input_shapes = [(2, 3, 4, 5), (3, 4, 5)]  # 4D tensor + 3D tensor
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        input_data = {
            'input_0': np.ones((2, 3, 4, 5), dtype=np.float32),
            'input_1': np.ones((3, 4, 5), dtype=np.float32) * 2.0
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_multiple_dimension_1(self, op_type):
        """Test arithmetic operations with multiple dimensions of size 1."""
        opset = 13
        input_shapes = [(5, 1, 4), (1, 3, 1)]  # Multiple dims of size 1
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        input_data = {
            'input_0': np.ones((5, 1, 4), dtype=np.float32),
            'input_1': np.ones((1, 3, 1), dtype=np.float32) * 2.0
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
        
        # Verify result shape
        assert comparison['tir_outputs']['output_0'].shape == (5, 3, 4)


# ============================================================================
# TEST CASES: ALL SUPPORTED DTYPES
# ============================================================================

class TestArithmeticDtypes:
    """Test arithmetic operations with all supported dtypes."""
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    @pytest.mark.parametrize("dtype, np_dtype", [
        (onnx.TensorProto.FLOAT, np.float32),
        (onnx.TensorProto.DOUBLE, np.float64),
        (onnx.TensorProto.INT32, np.int32),
        (onnx.TensorProto.INT64, np.int64),
    ])
    def test_arithmetic_basic_dtypes(self, op_type, dtype, np_dtype):
        """Test arithmetic operations with basic dtypes (float32, double, int32, int64)."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [dtype, dtype]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
            'input_1': np.array([[10, 20, 30], [40, 50, 60]], dtype=np_dtype)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_unsigned_int_dtypes(self, op_type):
        """Test arithmetic operations with unsigned integer dtypes (OPSET 14+)."""
        opset = 14
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [onnx.TensorProto.UINT8, onnx.TensorProto.UINT8]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            'input_1': np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    @pytest.mark.parametrize("dtype, np_dtype", [
        (onnx.TensorProto.INT8, np.int8),
        (onnx.TensorProto.INT16, np.int16),
    ])
    def test_arithmetic_small_int_dtypes(self, op_type, dtype, np_dtype):
        """Test arithmetic operations with small integer dtypes (int8, int16) (OPSET 14+)."""
        opset = 14
        input_shapes = [(2, 3), (2, 3)]
        input_dtypes = [dtype, dtype]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes, input_dtypes=input_dtypes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
            'input_1': np.array([[10, 20, 30], [40, 50, 60]], dtype=np_dtype)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"


# ============================================================================
# TEST CASES: ERROR CASES (SHOULD RAISE ERRORS)
# ============================================================================

class TestArithmeticErrors:
    """Test error cases that should raise exceptions."""
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_incompatible_shapes_opset_7(self, op_type):
        """Test arithmetic operations with incompatible shapes in OPSET 7+ (should raise error)."""
        opset = 13
        input_shapes = [(2, 3), (2, 4)]  # Incompatible: 3 vs 4, neither is 1
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        
        # This should raise an error during transpilation
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            tir_graph = transpiler.transpile(model)
        
        # Verify error message mentions broadcasting
        assert "broadcast" in str(exc_info.value).lower() or "compatible" in str(exc_info.value).lower()


# ============================================================================
# TEST CASES: EDGE CASES
# ============================================================================

class TestArithmeticEdgeCases:
    """Test edge cases for arithmetic operations."""
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_zero_tensor(self, op_type):
        """Test arithmetic operations with zero tensor."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.zeros((2, 3), dtype=np.float32),
            'input_1': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_negative_values(self, op_type):
        """Test arithmetic operations with negative values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32),
            'input_1': np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_single_element_tensor(self, op_type):
        """Test arithmetic operations with single element tensors."""
        opset = 13
        input_shapes = [(1,), (1,)]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([5.0], dtype=np.float32),
            'input_1': np.array([10.0], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"


# ============================================================================
# TEST CASES: OPERATION-SPECIFIC TESTS
# ============================================================================

class TestDivisionSpecific:
    """Test division-specific cases (division by zero, etc.)."""
    
    def test_div_division_by_zero(self):
        """Test division by zero (should produce inf or nan)."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model('Div', opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            'input_1': np.array([[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]], dtype=np.float32)  # Some zeros
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
        
        # Verify that division by zero produces inf or nan
        result = comparison['tir_outputs']['output_0']
        # Check that positions with zero divisor have inf or nan
        assert np.isinf(result[0, 1]) or np.isnan(result[0, 1]), "Division by zero should produce inf or nan"
        assert np.isinf(result[1, 1]) or np.isnan(result[1, 1]), "Division by zero should produce inf or nan"
    
    def test_div_small_values(self):
        """Test division with very small values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model('Div', opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[1e-10, 2e-10, 3e-10], [4e-10, 5e-10, 6e-10]], dtype=np.float32),
            'input_1': np.array([[1e-10, 2e-10, 3e-10], [4e-10, 5e-10, 6e-10]], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
        
        # Verify result (should be all 1.0)
        expected = np.ones((2, 3), dtype=np.float32)
        np.testing.assert_allclose(comparison['tir_outputs']['output_0'], expected, rtol=1e-5)


class TestSubtractionSpecific:
    """Test subtraction-specific cases."""
    
    def test_sub_result_negative(self):
        """Test subtraction that results in negative values."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model('Sub', opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            'input_1': np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
        
        # Verify result (all negative)
        expected = np.array([[-9.0, -18.0, -27.0], [-36.0, -45.0, -54.0]], dtype=np.float32)
        np.testing.assert_allclose(comparison['tir_outputs']['output_0'], expected, rtol=1e-5)


class TestMultiplicationSpecific:
    """Test multiplication-specific cases."""
    
    def test_mul_result_zero(self):
        """Test multiplication that results in zero."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model('Mul', opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            'input_1': np.array([[10.0, 0.0, 30.0], [40.0, 0.0, 60.0]], dtype=np.float32)  # Some zeros
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
        
        # Verify result
        expected = np.array([[10.0, 0.0, 90.0], [160.0, 0.0, 360.0]], dtype=np.float32)
        np.testing.assert_allclose(comparison['tir_outputs']['output_0'], expected, rtol=1e-5)


# ============================================================================
# TEST CASES: OPSET VERSION COMPARISON
# ============================================================================

class TestArithmeticOpsetVersions:
    """Test arithmetic operations across different opset versions."""
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    @pytest.mark.parametrize("opset", [7, 13, 14])
    def test_arithmetic_same_shape_all_opsets(self, op_type, opset):
        """Test arithmetic operations with same shapes across all opset versions."""
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            'input_1': np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"OPSET {opset} comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], f"OPSET {opset} outputs should match"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    @pytest.mark.parametrize("opset", [7, 13, 14])
    def test_arithmetic_broadcasting_opsets_7_plus(self, op_type, opset):
        """Test arithmetic broadcasting in OPSET 7+."""
        input_shapes = [(2, 3), (3,)]  # Broadcasting case
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            'input_1': np.array([10.0, 20.0, 30.0], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"OPSET {opset} comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], f"OPSET {opset} outputs should match"


# ============================================================================
# TEST CASES: GRAPH STRUCTURE VERIFICATION
# ============================================================================

class TestArithmeticGraphStructure:
    """Test arithmetic graph structure and node creation."""
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_graph_structure(self, op_type):
        """Test that arithmetic operations create correct graph structure."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Verify graph structure
        verification = verify_tir_graph_structure(tir_graph, model, expected_op_types=[op_type])
        assert verification['node_count_match'], "Node count should match"
        assert verification['input_count_match'], "Input count should match"
        assert verification['output_count_match'], "Output count should match"
        assert op_type in verification['node_types'], f"Should have {op_type} node"
    
    @pytest.mark.parametrize("op_type", ['Add', 'Sub', 'Mul', 'Div'])
    def test_arithmetic_node_attributes(self, op_type):
        """Test that arithmetic nodes have correct attributes."""
        opset = 13
        input_shapes = [(2, 3), (2, 3)]
        
        model = _create_arithmetic_model(op_type, opset, input_shapes)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Find arithmetic node
        arithmetic_nodes = [node for node in tir_graph.nodes if node.op_type == op_type]
        assert len(arithmetic_nodes) == 1, f"Should have exactly one {op_type} node"
        
        arithmetic_node = arithmetic_nodes[0]
        assert len(arithmetic_node.inputs) == 2, f"{op_type} node should have 2 inputs"
        assert len(arithmetic_node.outputs) == 1, f"{op_type} node should have 1 output"
        
        # Verify Forge op function names
        expected_forge_names = {
            'Add': 'forge.op.Add',
            'Sub': 'forge.op.Subtract',
            'Mul': 'forge.op.Multiply',
            'Div': 'forge.op.Divide'
        }
        assert arithmetic_node.forge_op_function_name == expected_forge_names[op_type], \
            f"Should have correct Forge op name for {op_type}"

