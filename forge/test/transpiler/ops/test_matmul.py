# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX MatMul operation.
Tests different input shapes (1D to 5D), mixed dimensions, dtypes, opset versions, and edge cases.

MatMul performs matrix product that behaves like numpy.matmul:
- Standard 2D: [M, K] @ [K, N] -> [M, N]
- N-dimensional: [..., M, K] @ [..., K, N] -> [..., M, N]
- Broadcasting: Batch dimensions are automatically broadcasted

Decomposition: MatMul -> MatMulNode (uses torch.matmul)
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


# ============================================================================
# HELPER METHODS FOR CREATING MATMUL MODELS
# ============================================================================


def _create_matmul_model(
    opset_version,
    input_shape_a,
    input_shape_b,
    dtype=onnx.TensorProto.FLOAT,
):
    """
    Helper to create MatMul ONNX model.

    Args:
        opset_version: ONNX opset version (1, 9, 13)
        input_shape_a: Shape of input tensor A [..., M, K]
        input_shape_b: Shape of input tensor B [..., K, N]
        dtype: Input/output dtype

    Returns:
        ONNX ModelProto
    """
    # Validate shapes for matrix multiplication
    if len(input_shape_a) < 1 or len(input_shape_b) < 1:
        raise ValueError(f"MatMul requires at least 1 dimension for both inputs")

    # Handle 1D inputs (treat as 2D for shape calculation)
    shape_a_2d = input_shape_a if len(input_shape_a) >= 2 else (1,) + input_shape_a
    shape_b_2d = input_shape_b if len(input_shape_b) >= 2 else input_shape_b + (1,)

    # Validate matrix multiplication compatibility: A.shape[-1] == B.shape[-2]
    if shape_a_2d[-1] != shape_b_2d[-2]:
        raise ValueError(
            f"MatMul incompatible shapes: A.shape[-1] ({shape_a_2d[-1]}) must equal "
            f"B.shape[-2] ({shape_b_2d[-2]}). A: {input_shape_a}, B: {input_shape_b}"
        )

    # Compute output shape
    M = shape_a_2d[-2]
    N = shape_b_2d[-1]

    if len(shape_a_2d) > 2 or len(shape_b_2d) > 2:
        # Batched case: compute broadcasted batch dimensions
        batch_dims_a = shape_a_2d[:-2]
        batch_dims_b = shape_b_2d[:-2]

        # Compute broadcasted batch dimensions (align from right)
        max_batch_len = max(len(batch_dims_a), len(batch_dims_b))
        broadcasted_batch_dims = []

        for i in range(max_batch_len):
            idx_a = len(batch_dims_a) - max_batch_len + i if i < len(batch_dims_a) else None
            idx_b = len(batch_dims_b) - max_batch_len + i if i < len(batch_dims_b) else None

            dim_a = batch_dims_a[idx_a] if idx_a is not None and idx_a >= 0 else 1
            dim_b = batch_dims_b[idx_b] if idx_b is not None and idx_b >= 0 else 1

            # Broadcasted dimension
            if dim_a == dim_b:
                broadcasted_batch_dims.append(dim_a)
            elif dim_a == 1:
                broadcasted_batch_dims.append(dim_b)
            elif dim_b == 1:
                broadcasted_batch_dims.append(dim_a)
            else:
                raise ValueError(f"Incompatible batch dimensions: {dim_a} vs {dim_b}")

        output_shape = tuple(broadcasted_batch_dims) + (M, N)
    else:
        # Standard 2D case
        output_shape = (M, N)

    # MatMul has no attributes
    attrs = {}

    return create_onnx_model(
        op_type="MatMul",
        input_shapes=[input_shape_a, input_shape_b],
        input_dtypes=[dtype, dtype],
        output_shapes=[output_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name="matmul",
        input_names=["A", "B"],
        output_names=["Y"],
    )


# ============================================================================
# MATMUL TESTS - COMPREHENSIVE
# ============================================================================


@pytest.mark.transpiler
class TestMatMul:
    """Comprehensive test cases for MatMul operation."""

    # ========================================================================
    # BASIC TESTS - Standard 2D and N-Dimensional Matrix Multiplication
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 9, 13])
    @pytest.mark.parametrize(
        "shape_a, shape_b",
        [
            # Standard 2D cases
            ((3, 4), (4, 5)),  # [M, K] @ [K, N]
            ((1, 4), (4, 5)),  # Single row A
            ((3, 4), (4, 1)),  # Single column B
            ((1, 4), (4, 1)),  # Both single
            ((5, 3), (3, 7)),  # Different sizes
            ((10, 8), (8, 6)),  # Larger matrices
            # 3D batched (same batch size)
            ((2, 3, 4), (2, 4, 5)),  # [B, M, K] @ [B, K, N]
            ((4, 5, 6), (4, 6, 7)),  # Larger batch
            # 3D with broadcasting
            ((2, 3, 4), (4, 5)),  # [B, M, K] @ [K, N] -> [B, M, N]
            # 4D batched
            ((2, 3, 4, 5), (2, 3, 5, 6)),  # [B1, B2, M, K] @ [B1, B2, K, N]
            # 4D with broadcasting
            ((2, 3, 4, 5), (5, 6)),  # [B1, B2, M, K] @ [K, N] -> [B1, B2, M, N]
            ((2, 1, 4, 5), (2, 3, 5, 6)),  # Mixed broadcasting [2, 1, ...] @ [2, 3, ...] -> [2, 3, ...]
            # 5D batched
            ((2, 3, 4, 5, 6), (2, 3, 4, 6, 7)),  # [B1, B2, B3, M, K] @ [B1, B2, B3, K, N]
            # 5D with broadcasting
            ((2, 3, 4, 5, 6), (6, 7)),  # [B1, B2, B3, M, K] @ [K, N] -> [B1, B2, B3, M, N]
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
        ],
    )
    def test_matmul_basic(self, opset_version, shape_a, shape_b, dtype):
        """Test basic matrix multiplication (2D to 5D) with various dtypes."""
        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")

        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # Create test data
        np.random.seed(42)
        data_a = np.random.randn(*shape_a).astype(np_dtype)
        data_b = np.random.randn(*shape_b).astype(np_dtype)

        # Create ONNX model
        try:
            onnx_model = _create_matmul_model(opset_version, shape_a, shape_b, dtype)
        except ValueError as e:
            pytest.skip(f"Invalid shape combination: {e}")

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True, debug=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have exactly 1 MatMul node
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
        assert tir_graph.nodes[0].op_type == "MatMul", f"Expected MatMul node, got {tir_graph.nodes[0].op_type}"

        # Compare with ONNX Runtime
        input_dict = {"A": data_a, "B": data_b}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"][
            "Y"
        ], f"Output mismatch for shapes A={shape_a}, B={shape_b}, dtype={dtype}, opset={opset_version}"

    # ========================================================================
    # DTYPE TESTS - Different Data Types
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 9, 13])
    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (onnx.TensorProto.FLOAT, np.float32),
            (onnx.TensorProto.DOUBLE, np.float64),
            (onnx.TensorProto.FLOAT16, np.float16),
            (onnx.TensorProto.BFLOAT16, np.float32),  # Use float32 for bfloat16 representation
            (onnx.TensorProto.INT32, np.int32),
            (onnx.TensorProto.INT64, np.int64),
            (onnx.TensorProto.UINT32, np.uint32),
            (onnx.TensorProto.UINT64, np.uint64),
        ],
    )
    def test_matmul_dtypes(self, opset_version, dtype, np_dtype):
        """Test MatMul with different data types."""
        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")

        # Skip FLOAT16 with opset 1 - ONNX Runtime has Cast attribute type issues
        if dtype == onnx.TensorProto.FLOAT16 and opset_version == 1:
            pytest.skip(f"FLOAT16 with opset 1 not supported by ONNX Runtime (Cast attribute type mismatch)")

        # Skip bfloat16 for opset < 13
        if dtype == onnx.TensorProto.BFLOAT16 and opset_version < 13:
            pytest.skip(f"BFloat16 only supported in opset 13+, got opset={opset_version}")

        # Skip BFLOAT16 with opset 13 - ONNX Runtime doesn't support MatMul(13) with bfloat16
        if dtype == onnx.TensorProto.BFLOAT16 and opset_version == 13:
            pytest.skip(f"BFloat16 MatMul(13) not supported by ONNX Runtime")

        # Skip integer types for opset < 9
        if (
            dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64, onnx.TensorProto.UINT32, onnx.TensorProto.UINT64]
            and opset_version < 9
        ):
            pytest.skip(f"Integer types only supported in opset 9+, got opset={opset_version}")

        # Skip unsigned integer types - PyTorch doesn't support matmul for unsigned integers
        if dtype in [onnx.TensorProto.UINT32, onnx.TensorProto.UINT64]:
            pytest.skip(f"Unsigned integer types (UINT32/UINT64) not supported by PyTorch matmul")

        shape_a = (3, 4)
        shape_b = (4, 5)

        # Create test data
        np.random.seed(42)
        if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64, onnx.TensorProto.UINT32, onnx.TensorProto.UINT64]:
            # Use integer values for integer types
            data_a = np.random.randint(0, 10, size=shape_a, dtype=np_dtype)
            data_b = np.random.randint(0, 10, size=shape_b, dtype=np_dtype)
        else:
            data_a = np.random.randn(*shape_a).astype(np_dtype)
            data_b = np.random.randn(*shape_b).astype(np_dtype)

        # Create ONNX model
        onnx_model = _create_matmul_model(opset_version, shape_a, shape_b, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True, debug=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        input_dict = {"A": data_a, "B": data_b}
        # Use looser tolerance for float16
        rtol = 1e-3 if dtype == onnx.TensorProto.FLOAT16 else 1e-5
        atol = 1e-3 if dtype == onnx.TensorProto.FLOAT16 else 1e-6
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison["matches"]["Y"], f"Output mismatch for dtype={dtype}, opset={opset_version}"

    # ========================================================================
    # BROADCASTING TESTS - Mixed Dimensions and Broadcasting Scenarios
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 9, 13])
    @pytest.mark.parametrize(
        "shape_a, shape_b",
        [
            # 3D broadcasting
            ((2, 3, 4), (1, 4, 5)),  # Broadcast batch dim: [2, ...] @ [1, ...]
            ((1, 3, 4), (2, 4, 5)),  # Broadcast batch dim: [1, ...] @ [2, ...]
            # 4D broadcasting
            ((2, 1, 3, 4), (2, 3, 4, 5)),  # Broadcast middle dim: [2, 1, ...] @ [2, 3, ...]
            ((1, 1, 3, 4), (2, 3, 4, 5)),  # Broadcast multiple dims: [1, 1, ...] @ [2, 3, ...]
            ((2, 3, 4, 5), (1, 1, 5, 6)),  # Broadcast on B: [2, 3, ...] @ [1, 1, ...]
            # 5D broadcasting
            ((2, 1, 1, 3, 4), (2, 3, 4, 4, 5)),  # Broadcast multiple dims: [2, 1, 1, ...] @ [2, 3, 4, ...]
            ((1, 1, 1, 3, 4), (2, 3, 4, 4, 5)),  # Broadcast all batch dims: [1, 1, 1, ...] @ [2, 3, 4, ...]
        ],
    )
    def test_matmul_broadcasting(self, opset_version, shape_a, shape_b):
        """Test MatMul with broadcasting scenarios."""
        dtype = onnx.TensorProto.FLOAT
        np_dtype = np.float32

        # Create test data
        np.random.seed(42)
        data_a = np.random.randn(*shape_a).astype(np_dtype)
        data_b = np.random.randn(*shape_b).astype(np_dtype)

        # Create ONNX model
        try:
            onnx_model = _create_matmul_model(opset_version, shape_a, shape_b, dtype)
        except ValueError as e:
            pytest.skip(f"Invalid shape combination: {e}")

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True, debug=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
        assert tir_graph.nodes[0].op_type == "MatMul", f"Expected MatMul node, got {tir_graph.nodes[0].op_type}"

        # Compare with ONNX Runtime
        input_dict = {"A": data_a, "B": data_b}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors for A={shape_a}, B={shape_b}: {comparison['errors']}"
        assert comparison["matches"]["Y"], f"Output mismatch for A={shape_a}, B={shape_b}"

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 9, 13])
    @pytest.mark.parametrize(
        "shape_a, shape_b",
        [
            # 1D inputs (treated as 2D)
            ((4,), (4,)),  # [K] @ [K] -> scalar-like
            ((4,), (4, 1)),  # [K] @ [K, 1] -> [1]
            ((1, 4), (4,)),  # [1, K] @ [K] -> [1]
            # Very small matrices
            ((1, 1), (1, 1)),  # [1, 1] @ [1, 1] -> [1, 1]
            ((1, 2), (2, 1)),  # [1, 2] @ [2, 1] -> [1, 1]
            # Large batch with small matrices
            ((100, 1, 1), (100, 1, 1)),  # [100, 1, 1] @ [100, 1, 1] -> [100, 1, 1]
            ((100, 2, 3), (100, 3, 4)),  # [100, 2, 3] @ [100, 3, 4] -> [100, 2, 4]
        ],
    )
    def test_matmul_edge_cases(self, opset_version, shape_a, shape_b):
        """Test MatMul edge cases."""
        dtype = onnx.TensorProto.FLOAT
        np_dtype = np.float32

        # Create test data
        np.random.seed(42)
        data_a = np.random.randn(*shape_a).astype(np_dtype)
        data_b = np.random.randn(*shape_b).astype(np_dtype)

        # Create ONNX model
        try:
            onnx_model = _create_matmul_model(opset_version, shape_a, shape_b, dtype)
        except ValueError as e:
            pytest.skip(f"Invalid shape combination: {e}")

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True, debug=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        input_dict = {"A": data_a, "B": data_b}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_dict, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors for A={shape_a}, B={shape_b}: {comparison['errors']}"
        assert comparison["matches"]["Y"], f"Output mismatch for A={shape_a}, B={shape_b}"
