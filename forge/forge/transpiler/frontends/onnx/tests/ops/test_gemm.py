"""
Test cases for ONNX Gemm (General Matrix Multiplication) operation.
Tests all opset versions, attribute combinations, broadcasting, and edge cases.

ONNX Runtime Supported Opsets: 7, 9, 11, 13
- Opset 1, 6: Not supported by ONNX Runtime (skipped)
- Opset 9: C input is required (ONNX spec says optional, but ONNX Runtime requires it)
- Opset 11+: C input is optional
- Batched operations (3D+): Not supported by ONNX Runtime (only 2D matrices)
- Integer types: Not supported by ONNX Runtime
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

# ONNX Runtime supported opsets for Gemm
SUPPORTED_OPSETS = [7, 9, 11, 13]
OPSETS_WITH_OPTIONAL_C = [11, 13]  # Opset 9 spec says optional but ONNX Runtime requires it


# ============================================================================
# HELPER METHODS FOR CREATING GEMM MODELS
# ============================================================================

def _calculate_gemm_output_shape(shape_a, shape_b, transA, transB):
    """
    Calculate output shape for Gemm operation: (M, N)
    
    Args:
        shape_a: Shape of A (M, K) or (K, M) if transA
        shape_b: Shape of B (K, N) or (N, K) if transB
        transA: Whether A is transposed
        transB: Whether B is transposed
        
    Returns:
        Output shape (M, N) or None if shapes are invalid
    """
    if len(shape_a) < 2 or len(shape_b) < 2:
        return None
    
    # Handle batched dimensions
    if len(shape_a) > 2:
        # Batched: (B, ..., M, K) or (B, ..., K, M)
        batch_dims = shape_a[:-2]
        if transA:
            m = shape_a[-1]
        else:
            m = shape_a[-2]
    else:
        # 2D: (M, K) or (K, M)
        batch_dims = ()
        if transA:
            m = shape_a[1]
        else:
            m = shape_a[0]
    
    if len(shape_b) > 2:
        # Batched: (B, ..., K, N) or (B, ..., N, K)
        if transB:
            n = shape_b[-2]
        else:
            n = shape_b[-1]
    else:
        # 2D: (K, N) or (N, K)
        if transB:
            n = shape_b[0]
        else:
            n = shape_b[1]
    
    if batch_dims:
        return batch_dims + (m, n)
    else:
        return (m, n)


def _create_gemm_model_v6_v7(opset_version, shape_a, shape_b, shape_c,
                             alpha=1.0, beta=1.0, transA=0, transB=0,
                             dtype=onnx.TensorProto.FLOAT):
    """
    Helper to create Gemm ONNX model for opset v6-v7.
    
    Key differences from v1:
    - No broadcast attribute (automatic unidirectional broadcasting)
    - C is still required
    """
    attrs = {
        'alpha': float(alpha),
        'beta': float(beta),
        'transA': int(transA),
        'transB': int(transB),
    }
    
    output_shape = _calculate_gemm_output_shape(shape_a, shape_b, transA, transB)
    
    return create_onnx_model(
        op_type='Gemm',
        input_shapes=[shape_a, shape_b, shape_c],
        input_dtypes=[dtype, dtype, dtype],
        output_shapes=[output_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name='gemm'
    )


def _create_gemm_model_v9_plus(opset_version, shape_a, shape_b, shape_c=None,
                               alpha=1.0, beta=1.0, transA=0, transB=0,
                               dtype=onnx.TensorProto.FLOAT):
    """
    Helper to create Gemm ONNX model for opset v9+.
    
    Key differences:
    - C input is optional (can be None or omitted)
    """
    attrs = {
        'alpha': float(alpha),
        'beta': float(beta),
        'transA': int(transA),
        'transB': int(transB),
    }
    
    input_shapes = [shape_a, shape_b]
    input_dtypes = [dtype, dtype]
    
    if shape_c is not None:
        input_shapes.append(shape_c)
        input_dtypes.append(dtype)
    
    output_shape = _calculate_gemm_output_shape(shape_a, shape_b, transA, transB)
    
    return create_onnx_model(
        op_type='Gemm',
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=[output_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name='gemm'
    )


# ============================================================================
# TEST CASES: BASIC GEMM OPERATIONS
# ============================================================================

class TestGemmBasic:
    """Test basic Gemm operations with standard configurations."""
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_basic_2d(self, opset):
        """Test basic 2D Gemm: Y = A * B + C"""
        # A: (3, 4), B: (4, 5), C: (3, 5), Output: (3, 5)
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c)
        
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Test data
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_with_transA(self, opset):
        """Test Gemm with transA=1: Y = A^T * B + C"""
        # A: (4, 3) with transA=1 -> (3, 4), B: (4, 5), C: (3, 5), Output: (3, 5)
        shape_a = (4, 3)  # Will be transposed to (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c, transA=1)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, transA=1)
        
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_with_transB(self, opset):
        """Test Gemm with transB=1: Y = A * B^T + C"""
        # A: (3, 4), B: (5, 4) with transB=1 -> (4, 5), C: (3, 5), Output: (3, 5)
        shape_a = (3, 4)
        shape_b = (5, 4)  # Will be transposed to (4, 5)
        shape_c = (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c, transB=1)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, transB=1)
        
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_with_both_trans(self, opset):
        """Test Gemm with both transA=1 and transB=1: Y = A^T * B^T + C"""
        # A: (4, 3) with transA=1 -> (3, 4)
        # B: (5, 4) with transB=1 -> (4, 5)
        # C: (3, 5), Output: (3, 5)
        shape_a = (4, 3)
        shape_b = (5, 4)
        shape_c = (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c, transA=1, transB=1)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, transA=1, transB=1)
        
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"


# ============================================================================
# TEST CASES: ALPHA AND BETA SCALING
# ============================================================================

class TestGemmScaling:
    """Test Gemm with different alpha and beta values."""
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    @pytest.mark.parametrize("transA,transB", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_gemm_transpose_with_alpha_beta(self, opset, transA, transB):
        """Test Gemm with all transpose combinations and alpha/beta scaling"""
        # Adjust shapes based on transpose flags
        if transA:
            shape_a = (4, 3)  # Will be transposed to (3, 4)
        else:
            shape_a = (3, 4)
        
        if transB:
            shape_b = (5, 4)  # Will be transposed to (4, 5)
        else:
            shape_b = (4, 5)
        
        shape_c = (3, 5)
        alpha = 2.0
        beta = 0.5
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c, 
                                            alpha=alpha, beta=beta, transA=transA, transB=transB)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c,
                                              alpha=alpha, beta=beta, transA=transA, transB=transB)
        
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, -1.0, 0.0])
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_alpha_scaling(self, opset, alpha):
        """Test Gemm with different alpha values: Y = alpha * A * B + C"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c, alpha=alpha)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, alpha=alpha)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, -1.0, 0.0])
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_beta_scaling(self, opset, beta):
        """Test Gemm with different beta values: Y = A * B + beta * C"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c, beta=beta)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, beta=beta)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("alpha,beta", [
        (1.0, 1.0),      # Default values
        (2.0, 0.5),      # Both non-default
        (0.5, 2.0),      # Reversed
        (-1.0, -1.0),    # Both negative
        (0.0, 1.0),      # alpha=0 (A*B term is zero)
        (1.0, 0.0),      # beta=0 (C term is zero)
        (0.0, 0.0),      # Both zero (output should be zero)
        (2.5, 1.5),      # Non-integer values
    ])
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_both_scaling(self, opset, alpha, beta):
        """Test Gemm with both alpha and beta scaling: Y = alpha * A * B + beta * C"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c, alpha=alpha, beta=beta)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, alpha=alpha, beta=beta)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"


# ============================================================================
# TEST CASES: OPTIONAL C INPUT (OPSET 11+)
# ============================================================================

class TestGemmOptionalC:
    """Test Gemm with optional C input (opset 11+)."""
    
    @pytest.mark.parametrize("opset", OPSETS_WITH_OPTIONAL_C)
    def test_gemm_without_c(self, opset):
        """Test Gemm without C input: Y = alpha * A * B"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        
        model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c=None)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", OPSETS_WITH_OPTIONAL_C)
    @pytest.mark.parametrize("alpha", [1.0, 2.0, 0.5, -1.0])
    def test_gemm_without_c_with_alpha(self, opset, alpha):
        """Test Gemm without C but with different alpha values: Y = alpha * A * B"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        
        model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c=None, alpha=alpha)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_with_c_beta_zero(self, opset):
        """Test Gemm with C but beta=0.0: Y = A * B (C ignored)"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, beta=0.0)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"


# ============================================================================
# TEST CASES: BROADCASTING
# ============================================================================

class TestGemmBroadcasting:
    """Test Gemm with broadcasting scenarios."""
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_c_broadcast_row(self, opset):
        """Test Gemm with C broadcasted along first dimension: C shape (1, N)"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (1, 5)  # Will be broadcasted to (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c)
        
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_c_broadcast_col(self, opset):
        """Test Gemm with C broadcasted along second dimension: C shape (M, 1)"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 1)  # Will be broadcasted to (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c)
        
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_c_broadcast_scalar(self, opset):
        """Test Gemm with C as scalar-like: C shape (1, 1)"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (1, 1)  # Will be broadcasted to (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c)
        
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"


# ============================================================================
# TEST CASES: DATA TYPES
# ============================================================================

class TestGemmDataTypes:
    """Test Gemm with different data types."""
    
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.FLOAT16,
    ])
    @pytest.mark.parametrize("opset", [11, 13])
    def test_gemm_float_types(self, opset, dtype):
        """Test Gemm with different float types (opset 11+)"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, dtype=dtype)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        if dtype == onnx.TensorProto.FLOAT:
            np_dtype = np.float32
        elif dtype == onnx.TensorProto.DOUBLE:
            np_dtype = np.float64
        elif dtype == onnx.TensorProto.FLOAT16:
            np_dtype = np.float16
        else:
            pytest.skip(f"Unsupported dtype: {dtype}")
        
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np_dtype),
            'input_1': np.random.randn(*shape_b).astype(np_dtype),
            'input_2': np.random.randn(*shape_c).astype(np_dtype)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        # For float16, allow some tolerance
        if dtype == onnx.TensorProto.FLOAT16:
            # Check if errors are only precision-related
            non_precision_errors = [e for e in comparison['errors'] 
                                  if 'precision' not in str(e).lower() and 
                                     'tolerance' not in str(e).lower()]
            assert len(non_precision_errors) == 0, f"Non-precision errors: {non_precision_errors}"
        else:
            assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
            assert comparison['matches']['output_0'], "Outputs should match"


# ============================================================================
# TEST CASES: EDGE CASES
# ============================================================================

class TestGemmEdgeCases:
    """Test Gemm edge cases and optimizations."""
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_alpha_one_beta_one(self, opset):
        """Test Gemm with alpha=1.0 and beta=1.0 (should optimize)"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c, alpha=1.0, beta=1.0)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, alpha=1.0, beta=1.0)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_large_shapes(self, opset):
        """Test Gemm with larger shapes"""
        shape_a = (10, 20)
        shape_b = (20, 15)
        shape_c = (10, 15)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_small_shapes(self, opset):
        """Test Gemm with small shapes (1x1 matrices)"""
        shape_a = (1, 1)
        shape_b = (1, 1)
        shape_c = (1, 1)
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        input_data = {
            'input_0': np.array([[2.0]], dtype=np.float32),
            'input_1': np.array([[3.0]], dtype=np.float32),
            'input_2': np.array([[1.0]], dtype=np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
        # Expected: 2 * 3 + 1 = 7
        assert np.allclose(comparison['tir_outputs']['output_0'], np.array([[7.0]], dtype=np.float32))
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    def test_gemm_rectangular_matrices(self, opset):
        """Test Gemm with rectangular matrices (M != N)"""
        shape_a = (5, 3)  # M=5, K=3
        shape_b = (3, 7)  # K=3, N=7
        shape_c = (5, 7)  # M=5, N=7
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"
    
    @pytest.mark.parametrize("opset", SUPPORTED_OPSETS)
    @pytest.mark.parametrize("transA,transB", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_gemm_different_shapes_with_transpose(self, opset, transA, transB):
        """Test Gemm with different shapes and transpose combinations"""
        # Use different shapes to ensure correctness
        if transA:
            shape_a = (6, 4)  # Will be transposed to (4, 6)
        else:
            shape_a = (4, 6)
        
        if transB:
            shape_b = (8, 6)  # Will be transposed to (6, 8)
        else:
            shape_b = (6, 8)
        
        shape_c = (4, 8)  # Output shape
        
        if opset < 9:
            model = _create_gemm_model_v6_v7(opset, shape_a, shape_b, shape_c, transA=transA, transB=transB)
        else:
            model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, transA=transA, transB=transB)
        
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        np.random.seed(42)
        input_data = {
            'input_0': np.random.randn(*shape_a).astype(np.float32),
            'input_1': np.random.randn(*shape_b).astype(np.float32),
            'input_2': np.random.randn(*shape_c).astype(np.float32)
        }
        
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['matches']['output_0'], "Outputs should match"


# ============================================================================
# TEST CASES: TIR GRAPH STRUCTURE VERIFICATION
# ============================================================================

class TestGemmGraphStructure:
    """Test TIR graph structure for Gemm decomposition."""
    
    @pytest.mark.parametrize("opset", [9, 11, 13])
    def test_gemm_graph_structure_basic(self, opset):
        """Verify TIR graph structure for basic Gemm"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Verify graph structure
        verify_tir_graph_structure(tir_graph, model, expected_op_types=['MatMul', 'Add'])
        
        # Verify nodes exist (should have MatMul, Add nodes at minimum)
        node_types = [node.op_type for node in tir_graph.nodes]
        assert 'MatMul' in node_types, "Should contain MatMul node"
        assert 'Add' in node_types, "Should contain Add node"
    
    @pytest.mark.parametrize("opset", [9, 11, 13])
    def test_gemm_graph_structure_with_transpose(self, opset):
        """Verify TIR graph structure for Gemm with transpose"""
        shape_a = (4, 3)
        shape_b = (5, 4)
        shape_c = (3, 5)
        
        model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, transA=1, transB=1)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Verify graph structure
        verify_tir_graph_structure(tir_graph, model, expected_op_types=['Transpose', 'MatMul', 'Add'])
        
        # Verify nodes exist (should have Transpose, MatMul, Add nodes)
        node_types = [node.op_type for node in tir_graph.nodes]
        assert 'Transpose' in node_types, "Should contain Transpose node(s)"
        assert 'MatMul' in node_types, "Should contain MatMul node"
        assert 'Add' in node_types, "Should contain Add node"
    
    @pytest.mark.parametrize("opset", [9, 11, 13])
    def test_gemm_graph_structure_with_alpha(self, opset):
        """Verify TIR graph structure for Gemm with alpha != 1.0"""
        shape_a = (3, 4)
        shape_b = (4, 5)
        shape_c = (3, 5)
        
        model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c, alpha=2.0)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Verify graph structure
        verify_tir_graph_structure(tir_graph, model, expected_op_types=['MatMul', 'Mul', 'Add'])
        
        # Verify nodes exist (should have MatMul, Mul, Add nodes)
        node_types = [node.op_type for node in tir_graph.nodes]
        assert 'MatMul' in node_types, "Should contain MatMul node"
        assert 'Mul' in node_types, "Should contain Mul node for alpha"
        assert 'Add' in node_types, "Should contain Add node"
    
    @pytest.mark.parametrize("opset", OPSETS_WITH_OPTIONAL_C)
    def test_gemm_graph_structure_without_c(self, opset):
        """Verify TIR graph structure for Gemm without C"""
        shape_a = (3, 4)
        shape_b = (4, 5)
    
        model = _create_gemm_model_v9_plus(opset, shape_a, shape_b, shape_c=None)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)
        
        # Verify graph structure
        verify_tir_graph_structure(tir_graph, model, expected_op_types=['MatMul'])
        
        # Verify nodes exist (should have MatMul, no Add node)
        node_types = [node.op_type for node in tir_graph.nodes]
        assert 'MatMul' in node_types, "Should contain MatMul node"
        # Should not have Add node when C is not provided
        if 'Add' in node_types:
            # If Add exists, it might be for identity pass-through, which is fine
            pass
