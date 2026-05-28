# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for the ONNX Trilu operation converter.

Trilu (opset 14+) extracts the upper or lower triangular part of a 2-D matrix
or a batch of 2-D matrices.  Because no native ``forge.op.Trilu`` exists, the
converter decomposes it into a static triangular mask (computed at transpile
time) multiplied element-wise with the input tensor:

    float32 input  → 1 MulNode  (+ mask in computed_constants)
    other dtype    → 1 CastNode + 1 MulNode (+ mask in computed_constants)

Covered scenarios
-----------------
Basic
  - Upper/lower triangular, k=0 (float32, default)
  - Square and non-square matrices (wider/taller)
  - Minimal [1×1] matrix
Diagonal offset (k)
  - Positive k (shift toward super-diagonal)
  - Negative k (shift toward sub-diagonal)
  - k so large that output is all-zeros
  - k so large that output equals input (all-kept)
  - k absent (defaults to 0) vs k=0 explicit — must produce identical results
Batched inputs
  - 3-D [B, N, M] and 4-D [B, T, N, M]
  - Non-square batch [B, N, M] with N ≠ M
Dtypes
  - float32  → no CastNode
  - float16  → CastNode inserted (Example 7 from docs)
  - float64  → CastNode inserted
  - int32    → CastNode inserted
  - int64    → CastNode inserted
k as constant initializer
  - k resolved via graph initializer (ONNX constant)
  - Parametrized over 0, ±1, ±2 for both upper and lower
Graph structure verification
  - Exactly 1 MulNode for float32
  - CastNode precedes MulNode for non-float32
  - Cast output is wired as input to MulNode (not mask directly)
  - Mask stored in computed_constants with correct shape and float32 dtype
  - Only 1 mask entry per Trilu node in computed_constants
  - forge_op_function_name of MulNode == "forge.op.Multiply"
  - MulNode has exactly 2 inputs
  - src_layer populated on every produced TIR node
  - verify_tir_graph_structure output_count_match
  - Stored mask values are correct (explicit byte-level check)
Numerical / input-value edge cases
  - All-zero input
  - All-ones input
  - Negative input values preserved or zeroed correctly
  - Output dtype matches input dtype exactly
  - Output shape equals input shape for 2-D / 3-D / 4-D inputs
  - Large [64×64] matrix (numerical sanity)
Explicit doc-example tests (mirroring docs/onnx_trilu.md)
  - Example 1: upper=1, k=0, 3×3 float32
  - Example 2: upper=1, k=2, 3×4 float32
  - Example 3: upper=1, k=-1, 3×3 float32
  - Example 4: lower=0, k=0, 3×3 float32
  - Example 5: lower=0, k=1, 3×3 float32
  - Example 6: batched [2, 3, 3] float32
  - Example 7: float16 → CastNode inserted
  - Example 8: no k input (default k=0), 2×3 float32
Error cases
  - opset < 14 → ConversionError
  - dynamic (runtime) k input → ConversionError
"""
import pytest
import numpy as np
import torch
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.utils.exceptions import ConversionError
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
    verify_tir_graph_structure,
)


# ============================================================================
# HELPER
# ============================================================================


def _create_trilu_model(
    input_shape,
    upper: int = 1,
    k_val: int = None,
    input_dtype: int = None,
    node_name: str = "trilu_node",
    opset_version: int = 14,
):
    """
    Create a single-node ONNX Trilu model.

    Args:
        input_shape:   Shape of the input tensor (rank >= 2).
        upper:         1 for upper triangular (default), 0 for lower.
        k_val:         Integer diagonal offset.  When provided, a scalar int64
                       constant initializer named ``"k"`` is embedded.  When
                       ``None`` the ``k`` input is omitted (ONNX default k=0).
        input_dtype:   ONNX dtype enum for the input (default: FLOAT/float32).
        node_name:     Name for the Trilu ONNX node.
        opset_version: ONNX opset version (default: 14).

    Returns:
        ONNX ModelProto with a single Trilu node.
    """
    if input_dtype is None:
        input_dtype = onnx.TensorProto.FLOAT

    input_names = ["input_0"]
    input_shapes = [input_shape]
    input_dtypes = [input_dtype]
    initializers = {}

    if k_val is not None:
        input_names.append("k")
        input_shapes.append(())
        input_dtypes.append(onnx.TensorProto.INT64)
        initializers["k"] = np.array(k_val, dtype=np.int64)

    return create_onnx_model(
        op_type="Trilu",
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=[input_shape],
        output_dtypes=[input_dtype],
        attrs={"upper": upper},
        opset_version=opset_version,
        node_name=node_name,
        input_names=input_names,
        output_names=["output_0"],
        initializers=initializers,
    )


def _run_tir(tir_graph, np_input: np.ndarray) -> np.ndarray:
    """Run the TIR graph with a single float32/int input and return output as numpy."""
    result = tir_graph.run({"input_0": torch.from_numpy(np_input)})
    return result[tir_graph.outputs[0]].detach().cpu().numpy()


# ============================================================================
# BASIC UPPER / LOWER TRIANGULAR TESTS (k=0, float32)
# ============================================================================


@pytest.mark.transpiler
class TestTriluBasic:
    """Basic upper/lower triangular extraction with k=0 and float32 inputs."""

    def test_trilu_upper_square(self):
        """Upper triangular of a 3×3 square matrix (k=0)."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["output_0"],
            np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]], dtype=np.float32),
        )

    def test_trilu_lower_square(self):
        """Lower triangular of a 3×3 square matrix (k=0)."""
        model = _create_trilu_model((3, 3), upper=0)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["output_0"],
            np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]], dtype=np.float32),
        )

    def test_trilu_upper_nonsquare_wider(self):
        """Upper triangular of a [2, 4] (fewer rows than cols) matrix."""
        model = _create_trilu_model((2, 4), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=0))

    def test_trilu_lower_nonsquare_taller(self):
        """Lower triangular of a [4, 2] (more rows than cols) matrix."""
        model = _create_trilu_model((4, 2), upper=0)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 9, dtype=np.float32).reshape(4, 2)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.tril(x, k=0))

    def test_trilu_upper_1x1(self):
        """Trilu on a [1, 1] matrix is a no-op (entire matrix retained)."""
        model = _create_trilu_model((1, 1), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[7.0]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], x)

    def test_trilu_lower_1x1(self):
        """Lower Trilu on a [1, 1] is also a no-op."""
        model = _create_trilu_model((1, 1), upper=0)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[42.0]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], x)


# ============================================================================
# DIAGONAL OFFSET (k) TESTS
# ============================================================================


@pytest.mark.transpiler
class TestTriluDiagonalOffset:
    """Test the k diagonal offset for upper and lower triangular."""

    def test_trilu_upper_k_positive(self):
        """upper=1, k=2: keep elements two or more positions right of diagonal."""
        model = _create_trilu_model((3, 4), upper=1, k_val=2)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=2))

    def test_trilu_upper_k_negative(self):
        """upper=1, k=-1: main diagonal and one sub-diagonal are also kept."""
        model = _create_trilu_model((3, 3), upper=1, k_val=-1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=-1))

    def test_trilu_lower_k_positive(self):
        """upper=0, k=1: main diagonal plus one super-diagonal are kept."""
        model = _create_trilu_model((3, 3), upper=0, k_val=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.tril(x, k=1))

    def test_trilu_lower_k_negative(self):
        """upper=0, k=-2: only elements two or more positions below diagonal."""
        model = _create_trilu_model((4, 4), upper=0, k_val=-2)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.tril(x, k=-2))

    def test_trilu_upper_k_large_positive_all_zeros(self):
        """upper=1, k > M-1: no element satisfies j >= i+k → output all zeros."""
        model = _create_trilu_model((3, 3), upper=1, k_val=4)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["output_0"],
            np.zeros((3, 3), dtype=np.float32),
        )

    def test_trilu_lower_k_large_negative_all_zeros(self):
        """upper=0, k < -(N-1): no element satisfies j <= i+k → output all zeros."""
        model = _create_trilu_model((3, 3), upper=0, k_val=-4)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["output_0"],
            np.zeros((3, 3), dtype=np.float32),
        )

    def test_trilu_lower_k_large_positive_all_kept(self):
        """upper=0, k >= M-1: every element satisfies j <= i+k → output == input."""
        N = 3
        model = _create_trilu_model((N, N), upper=0, k_val=N - 1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, N * N + 1, dtype=np.float32).reshape(N, N)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], x)

    def test_trilu_upper_k_large_negative_all_kept(self):
        """upper=1, k <= -(N-1): every element is kept → output == input."""
        N = 3
        model = _create_trilu_model((N, N), upper=1, k_val=-(N - 1))
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, N * N + 1, dtype=np.float32).reshape(N, N)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], x)

    def test_trilu_k_default_zero_equals_explicit_k0(self):
        """Absent k must produce the same result as explicit k=0."""
        model_k0 = _create_trilu_model((3, 3), upper=1, k_val=0)
        model_no_k = _create_trilu_model((3, 3), upper=1)

        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_k0 = transpiler.transpile(model_k0)
        tir_no_k = transpiler.transpile(model_no_k)

        x = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        xt = torch.from_numpy(x)

        out_k0 = tir_k0.run({"input_0": xt})[tir_k0.outputs[0]].numpy()
        out_no_k = tir_no_k.run({"input_0": xt})[tir_no_k.outputs[0]].numpy()

        np.testing.assert_array_equal(out_k0, out_no_k)


# ============================================================================
# BATCHED INPUT TESTS
# ============================================================================


@pytest.mark.transpiler
class TestTriluBatched:
    """Trilu on batched 3-D and 4-D tensors (mask broadcasts over batch dims)."""

    def test_trilu_3d_upper(self):
        """Upper triangular on a [2, 3, 3] batch — mask [3,3] broadcasts to [2,3,3]."""
        model = _create_trilu_model((2, 3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 19, dtype=np.float32).reshape(2, 3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=0))

    def test_trilu_3d_lower(self):
        """Lower triangular on a [2, 3, 3] batch."""
        model = _create_trilu_model((2, 3, 3), upper=0)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 19, dtype=np.float32).reshape(2, 3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.tril(x, k=0))

    def test_trilu_3d_upper_k1(self):
        """3-D batch, upper=1, k=1: sub-diagonal is also zeroed."""
        model = _create_trilu_model((2, 3, 3), upper=1, k_val=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 19, dtype=np.float32).reshape(2, 3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=1))

    def test_trilu_4d_upper(self):
        """Upper triangular on a [2, 2, 3, 3] 4-D batch."""
        model = _create_trilu_model((2, 2, 3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 37, dtype=np.float32).reshape(2, 2, 3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=0))

    def test_trilu_4d_lower_k_neg1(self):
        """Lower triangular on a [2, 2, 3, 3] 4-D batch with k=-1."""
        model = _create_trilu_model((2, 2, 3, 3), upper=0, k_val=-1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 37, dtype=np.float32).reshape(2, 2, 3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.tril(x, k=-1))

    def test_trilu_3d_nonsquare_batch(self):
        """3-D batch [2, 3, 4]: non-square matrices, mask shape [3, 4]."""
        model = _create_trilu_model((2, 3, 4), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=0))

    def test_trilu_batch_each_slice_independent(self):
        """Each 2-D slice of a batched input is masked independently and identically."""
        model = _create_trilu_model((3, 4, 4), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.random.default_rng(0).standard_normal((3, 4, 4)).astype(np.float32)
        result = _run_tir(tir_graph, x)

        for b in range(3):
            np.testing.assert_array_equal(result[b], np.triu(x[b], k=0))


# ============================================================================
# DTYPE TESTS
# ============================================================================


@pytest.mark.transpiler
class TestTriluDtypes:
    """
    Trilu across all supported numeric dtypes.

    float32 → no CastNode (mask already float32).
    All others → CastNode inserted to align mask dtype with input dtype.
    """

    @pytest.mark.parametrize(
        "onnx_dtype, np_dtype",
        [
            (onnx.TensorProto.FLOAT, np.float32),
            (onnx.TensorProto.FLOAT16, np.float16),
            (onnx.TensorProto.DOUBLE, np.float64),
            (onnx.TensorProto.INT64, np.int64),
        ],
    )
    def test_trilu_upper_dtypes(self, onnx_dtype, np_dtype):
        """upper=1, k=0 produces correct results for all supported dtypes."""
        model = _create_trilu_model((3, 3), upper=1, input_dtype=onnx_dtype)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 10, dtype=np_dtype).reshape(3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Dtype {np_dtype}: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=0))

    @pytest.mark.parametrize(
        "onnx_dtype, np_dtype",
        [
            (onnx.TensorProto.FLOAT, np.float32),
            (onnx.TensorProto.FLOAT16, np.float16),
            (onnx.TensorProto.DOUBLE, np.float64),
            (onnx.TensorProto.INT64, np.int64),
        ],
    )
    def test_trilu_lower_dtypes(self, onnx_dtype, np_dtype):
        """upper=0, k=0 produces correct results for all supported dtypes."""
        model = _create_trilu_model((3, 3), upper=0, input_dtype=onnx_dtype)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 10, dtype=np_dtype).reshape(3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Dtype {np_dtype}: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.tril(x, k=0))

    @pytest.mark.parametrize(
        "onnx_dtype, np_dtype",
        [
            (onnx.TensorProto.FLOAT, np.float32),
            (onnx.TensorProto.FLOAT16, np.float16),
            (onnx.TensorProto.DOUBLE, np.float64),
            (onnx.TensorProto.INT64, np.int64),
        ],
    )
    def test_trilu_output_dtype_matches_input_dtype(self, onnx_dtype, np_dtype):
        """Output tensor dtype must equal input tensor dtype (ONNX spec: output type T = input type T)."""
        model = _create_trilu_model((3, 3), upper=1, input_dtype=onnx_dtype)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 10, dtype=np_dtype).reshape(3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Dtype {np_dtype}: {comparison['errors']}"
        result = comparison["tir_outputs"]["output_0"]
        assert result.dtype == np_dtype, f"Output dtype {result.dtype} must match input dtype {np_dtype}"


# ============================================================================
# k AS CONSTANT INITIALIZER — PARAMETRIZED
# ============================================================================


@pytest.mark.transpiler
class TestTriluKConstant:
    """Test that k supplied as a constant initializer is resolved and applied correctly."""

    @pytest.mark.parametrize("k_val", [0, 1, 2, -1, -2])
    def test_trilu_upper_k_initializer(self, k_val):
        """upper=1 with constant k: output matches numpy reference."""
        model = _create_trilu_model((4, 4), upper=1, k_val=k_val)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"k={k_val}: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=k_val))

    @pytest.mark.parametrize("k_val", [0, 1, 2, -1, -2])
    def test_trilu_lower_k_initializer(self, k_val):
        """upper=0 with constant k: output matches numpy reference."""
        model = _create_trilu_model((4, 4), upper=0, k_val=k_val)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"k={k_val}: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.tril(x, k=k_val))


# ============================================================================
# GRAPH STRUCTURE TESTS
# ============================================================================


@pytest.mark.transpiler
class TestTriluGraphStructure:
    """Verify the TIR graph topology produced by TriluConverter."""

    def test_trilu_float32_produces_exactly_one_mul_no_cast(self):
        """float32 input → 1 MulNode, 0 CastNodes."""
        model = _create_trilu_model((3, 3), upper=1, input_dtype=onnx.TensorProto.FLOAT)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        node_types = [n.op_type for n in tir_graph.nodes]
        assert "Mul" in node_types, f"Expected a Mul node, got {node_types}"
        assert "Cast" not in node_types, f"Unexpected Cast for float32: {node_types}"
        assert node_types.count("Mul") == 1, f"Expected exactly 1 Mul, got {node_types}"

    @pytest.mark.parametrize(
        "onnx_dtype",
        [
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
        ],
    )
    def test_trilu_non_float32_produces_cast_before_mul(self, onnx_dtype):
        """Non-float32 input → CastNode immediately before MulNode."""
        model = _create_trilu_model((3, 3), upper=1, input_dtype=onnx_dtype)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        node_types = [n.op_type for n in tir_graph.nodes]
        assert "Cast" in node_types, f"Expected Cast for dtype {onnx_dtype}: {node_types}"
        assert "Mul" in node_types, f"Expected Mul node: {node_types}"

        cast_idx = node_types.index("Cast")
        mul_idx = node_types.index("Mul")
        assert cast_idx < mul_idx, "CastNode must precede MulNode"

    @pytest.mark.parametrize(
        "onnx_dtype",
        [
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
        ],
    )
    def test_trilu_cast_output_is_mul_second_input(self, onnx_dtype):
        """
        The CastNode output (cast mask) must be wired as the second input to MulNode —
        not the raw float32 mask constant.
        """
        model = _create_trilu_model((3, 3), upper=1, input_dtype=onnx_dtype)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        cast_nodes = [n for n in tir_graph.nodes if n.op_type == "Cast"]
        mul_nodes = [n for n in tir_graph.nodes if n.op_type == "Mul"]

        assert cast_nodes, "Expected a CastNode"
        assert mul_nodes, "Expected a MulNode"

        cast_output_name = list(cast_nodes[0].outputs.keys())[0]
        mul_input_names = list(mul_nodes[0].inputs.keys())

        assert cast_output_name in mul_input_names, (
            f"CastNode output '{cast_output_name}' must be an input to MulNode. " f"MulNode inputs: {mul_input_names}"
        )

    def test_trilu_mask_in_computed_constants_with_correct_shape(self):
        """Mask is stored in computed_constants with shape (N, M) and dtype float32."""
        model = _create_trilu_model((2, 4, 5), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        mask_keys = [k for k in tir_graph.computed_constants if "trilu_mask" in k]
        assert mask_keys, (
            f"No triangular mask in computed_constants. " f"Keys: {list(tir_graph.computed_constants.keys())}"
        )
        mask = tir_graph.computed_constants[mask_keys[0]]
        assert mask.shape == (4, 5), f"Mask shape should be (4,5), got {mask.shape}"
        assert mask.dtype == torch.float32, f"Mask dtype should be float32, got {mask.dtype}"

    def test_trilu_exactly_one_mask_per_node_in_computed_constants(self):
        """Each Trilu node stores exactly one triangular mask in computed_constants."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        mask_keys = [k for k in tir_graph.computed_constants if "trilu_mask" in k]
        assert len(mask_keys) == 1, f"Expected exactly 1 mask in computed_constants, got {len(mask_keys)}: {mask_keys}"

    def test_trilu_mul_forge_op_name_is_multiply(self):
        """MulNode produced by Trilu must map to forge.op.Multiply."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        mul_nodes = [n for n in tir_graph.nodes if n.op_type == "Mul"]
        assert mul_nodes, "No Mul node found"
        assert (
            mul_nodes[0].forge_op_function_name == "forge.op.Multiply"
        ), f"Expected forge.op.Multiply, got {mul_nodes[0].forge_op_function_name}"

    def test_trilu_mul_has_exactly_two_inputs(self):
        """MulNode must have exactly 2 inputs: input tensor and (possibly cast) mask."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        mul_nodes = [n for n in tir_graph.nodes if n.op_type == "Mul"]
        assert mul_nodes, "No Mul node found"
        assert len(mul_nodes[0].inputs) == 2, f"MulNode must have 2 inputs, got {list(mul_nodes[0].inputs.keys())}"

    def test_trilu_src_layer_populated_on_all_nodes(self):
        """Every TIR node produced by Trilu must have src_layer set."""
        model = _create_trilu_model((3, 3), upper=1, node_name="my_trilu")
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        for node in tir_graph.nodes:
            assert node.src_layer is not None, f"Node '{node.name}' ({node.op_type}) has src_layer=None"

    def test_trilu_output_count_matches_onnx(self):
        """verify_tir_graph_structure must report output_count_match=True."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        result = verify_tir_graph_structure(tir_graph, model, expected_op_types=["Mul"])
        assert result["output_count_match"], "Output count mismatch between TIR and ONNX"

    def test_trilu_upper_k0_mask_values(self):
        """Stored mask for upper=1, k=0, 3×3 must be the exact upper-triangular ones matrix."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        mask_key = next(k for k in tir_graph.computed_constants if "trilu_mask" in k)
        mask = tir_graph.computed_constants[mask_key].numpy()
        expected = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(mask, expected)

    def test_trilu_lower_k0_mask_values(self):
        """Stored mask for upper=0, k=0, 3×3 must be the exact lower-triangular ones matrix."""
        model = _create_trilu_model((3, 3), upper=0)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        mask_key = next(k for k in tir_graph.computed_constants if "trilu_mask" in k)
        mask = tir_graph.computed_constants[mask_key].numpy()
        expected = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(mask, expected)

    def test_trilu_upper_k2_mask_values_nonsquare(self):
        """Stored mask for upper=1, k=2, shape [3,4] — matches torch.triu result exactly."""
        model = _create_trilu_model((3, 4), upper=1, k_val=2)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        mask_key = next(k for k in tir_graph.computed_constants if "trilu_mask" in k)
        mask = tir_graph.computed_constants[mask_key].numpy()
        expected = np.triu(np.ones((3, 4), dtype=np.float32), k=2)
        np.testing.assert_array_equal(mask, expected)

    def test_trilu_mask_values_binary(self):
        """All mask values must be exactly 0.0 or 1.0 (no intermediate values)."""
        for upper in [0, 1]:
            for k in [-1, 0, 1]:
                model = _create_trilu_model((4, 4), upper=upper, k_val=k)
                tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)
                mask_key = next(k_ for k_ in tir_graph.computed_constants if "trilu_mask" in k_)
                mask = tir_graph.computed_constants[mask_key].numpy()
                unique_vals = np.unique(mask)
                assert set(unique_vals).issubset(
                    {0.0, 1.0}
                ), f"Mask for upper={upper}, k={k} contains non-binary values: {unique_vals}"


# ============================================================================
# NUMERICAL / INPUT-VALUE EDGE CASES
# ============================================================================


@pytest.mark.transpiler
class TestTriluEdgeCases:
    """Numerical correctness on special input values and shapes."""

    def test_trilu_all_zeros_input(self):
        """All-zero input → all-zero output regardless of upper/k."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.zeros((3, 3), dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], x)

    def test_trilu_all_ones_input(self):
        """All-ones input: output equals the mask itself."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.ones((3, 3), dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=0))

    def test_trilu_negative_values_zeroed_outside_triangle(self):
        """Negative values outside the triangle must be zeroed, not sign-flipped."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        expected = np.triu(x, k=0)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)
        # Explicitly assert elements that should be zero are exactly 0 (not -0 artefacts)
        result = comparison["tir_outputs"]["output_0"]
        assert result[1, 0] == 0.0, f"Below-diagonal element should be 0, got {result[1, 0]}"
        assert result[2, 0] == 0.0
        assert result[2, 1] == 0.0

    def test_trilu_negative_values_preserved_inside_triangle(self):
        """Negative values inside the triangle must be preserved exactly."""
        model = _create_trilu_model((3, 3), upper=0)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        expected = np.tril(x, k=0)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)
        result = comparison["tir_outputs"]["output_0"]
        # Main diagonal should be preserved exactly
        assert result[0, 0] == -1.0
        assert result[1, 1] == -5.0
        assert result[2, 2] == -9.0

    def test_trilu_mixed_positive_negative_input(self):
        """Mixed positive/negative values: inside triangle kept, outside zeroed."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], np.triu(x, k=0))

    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (2, 2, 4, 5)])
    def test_trilu_output_shape_equals_input_shape(self, shape):
        """Output shape must always match input shape exactly."""
        model = _create_trilu_model(shape, upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.ones(shape, dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Shape {shape}: {comparison['errors']}"
        assert (
            comparison["tir_outputs"]["output_0"].shape == shape
        ), f"Output shape {comparison['tir_outputs']['output_0'].shape} != {shape}"

    def test_trilu_large_matrix(self):
        """[64×64] float32 matrix — numerical correctness at larger scale."""
        model = _create_trilu_model((64, 64), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(42)
        x = rng.standard_normal((64, 64)).astype(np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], np.triu(x, k=0), rtol=1e-6)


# ============================================================================
# EXPLICIT DOC-EXAMPLE TESTS (mirroring docs/onnx_trilu.md)
# ============================================================================


@pytest.mark.transpiler
class TestTriluDocExamples:
    """
    Exact input/output pairs taken verbatim from docs/onnx_trilu.md.
    These are regression anchors — changes to them indicate a correctness regression.
    """

    def test_example_1_upper_k0_3x3(self):
        """Example 1: upper=1, k=0 (default), 3×3 float32."""
        model = _create_trilu_model((3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = _run_tir(tir_graph, x)
        expected = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_example_2_upper_k2_3x4(self):
        """Example 2: upper=1, k=2, 3×4 float32."""
        model = _create_trilu_model((3, 4), upper=1, k_val=2)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
        result = _run_tir(tir_graph, x)
        expected = np.array([[0, 0, 3, 4], [0, 0, 0, 8], [0, 0, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_example_3_upper_k_neg1_3x3(self):
        """Example 3: upper=1, k=-1, 3×3 float32 — diagonal and one sub-diagonal kept."""
        model = _create_trilu_model((3, 3), upper=1, k_val=-1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = _run_tir(tir_graph, x)
        expected = np.array([[1, 2, 3], [4, 5, 6], [0, 8, 9]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_example_4_lower_k0_3x3(self):
        """Example 4: upper=0, k=0 (default), 3×3 float32."""
        model = _create_trilu_model((3, 3), upper=0)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = _run_tir(tir_graph, x)
        expected = np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_example_5_lower_k1_3x3(self):
        """Example 5: upper=0, k=1, 3×3 float32 — lower tri + one super-diagonal."""
        model = _create_trilu_model((3, 3), upper=0, k_val=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = _run_tir(tir_graph, x)
        expected = np.array([[1, 2, 0], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_example_6_batched_3d_upper_k0(self):
        """Example 6: batched [2, 3, 3] float32, upper=1, k=0."""
        model = _create_trilu_model((2, 3, 3), upper=1)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]],
            dtype=np.float32,
        )
        result = _run_tir(tir_graph, x)
        expected = np.array(
            [[[1, 2, 3], [0, 5, 6], [0, 0, 9]], [[10, 11, 12], [0, 14, 15], [0, 0, 18]]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_example_7_float16_cast_node_inserted(self):
        """Example 7: float16 input → CastNode inserted, output is float16."""
        model = _create_trilu_model((3, 3), upper=1, input_dtype=onnx.TensorProto.FLOAT16)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        # Verify graph structure: CastNode precedes MulNode
        node_types = [n.op_type for n in tir_graph.nodes]
        assert "Cast" in node_types, "float16 input must trigger CastNode insertion"
        assert node_types.index("Cast") < node_types.index("Mul"), "CastNode must come before MulNode"

        # Verify numerical correctness via compare_tir_with_onnx
        x = np.arange(1, 10, dtype=np.float16).reshape(3, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        expected = np.triu(x, k=0)
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], expected)

        # Verify output dtype is float16 (not float32 from the mask)
        assert comparison["tir_outputs"]["output_0"].dtype == np.float16, "Output must preserve float16 dtype"

    def test_example_8_no_k_input_lower_2x3(self):
        """Example 8: no k input (default k=0), upper=0, 2×3 float32."""
        model = _create_trilu_model((2, 3), upper=0)
        tir_graph = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[5, 3, 2], [1, 4, 7]], dtype=np.float32)
        result = _run_tir(tir_graph, x)
        expected = np.array([[5, 0, 0], [1, 4, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


# ============================================================================
# ERROR CASE TESTS
# ============================================================================


@pytest.mark.transpiler
class TestTriluErrors:
    """Cases that must raise ConversionError during transpilation."""

    def test_trilu_opset_below_14_raises_conversion_error(self):
        """Trilu at opset < 14 must raise ConversionError with message mentioning opset 14."""
        model = _create_trilu_model((3, 3), upper=1, opset_version=13)
        transpiler = ONNXToForgeTranspiler(validate_model=False)

        with pytest.raises(ConversionError) as exc_info:
            transpiler.transpile(model)

        error_str = str(exc_info.value)
        assert (
            "14" in error_str or "opset" in error_str.lower()
        ), f"Expected error mentioning opset 14, got: {error_str}"

    def test_trilu_dynamic_k_raises_conversion_error(self):
        """
        k as a runtime graph input (no initializer) must raise ConversionError
        because the mask cannot be precomputed at transpile time.
        """
        model = create_onnx_model(
            op_type="Trilu",
            input_shapes=[(3, 3), ()],
            input_dtypes=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            output_shapes=[(3, 3)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={"upper": 1},
            opset_version=14,
            node_name="trilu_dynamic_k",
            input_names=["input_0", "k"],
            output_names=["output_0"],
            initializers={},
        )
        transpiler = ONNXToForgeTranspiler(validate_model=True)

        with pytest.raises(ConversionError) as exc_info:
            transpiler.transpile(model)

        error_str = str(exc_info.value).lower()
        assert (
            "k" in error_str or "constant" in error_str or "dynamic" in error_str
        ), f"Expected error about dynamic k, got: {exc_info.value}"

    def test_trilu_opset_13_error_message_contains_node_info(self):
        """ConversionError for opset < 14 must identify the failing node."""
        model = _create_trilu_model((3, 3), upper=1, opset_version=13, node_name="bad_trilu")
        transpiler = ONNXToForgeTranspiler(validate_model=False)

        with pytest.raises(ConversionError) as exc_info:
            transpiler.transpile(model)

        assert "Trilu" in str(exc_info.value) or "bad_trilu" in str(
            exc_info.value
        ), f"Error message should identify node: {exc_info.value}"
