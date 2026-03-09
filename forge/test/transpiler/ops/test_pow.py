# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Pow operation.

PowNode is a pure binary element-wise op: both X (base) and Y (exponent) are
tensor inputs.  Y may come from a compile-time constant initializer or a
runtime activation; in both cases it is wired as a second tensor operand.

Covers:
- Basic same-shape evaluation (1D, 2D, 3D, 4D)
- Broadcasting (scalar, row-vector, column-vector, higher-rank, multi dim-1)
- All supported dtypes (float32, float64, int32, int64)
- Opset v1-v6: broadcast / axis attributes (opt-in broadcasting)
- Opset v7+: multidirectional (NumPy-style) broadcasting, always on
- Opset v12+: heterogeneous types (X dtype T, Y dtype T1, output dtype T)
  with explicit CastNode injection in the TIR graph
- Y as a constant initializer (scalar or shaped tensor)
- Y as a runtime graph input
- Edge cases: exponent 0, 1, -1, large, fractional, base 1, single element
- Error cases: incompatible shapes
- Graph structure and node attribute verification
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
# HELPER
# ============================================================================


def _create_pow_model(
    opset_version: int,
    input_shapes,
    input_dtypes=None,
    output_shape=None,
    output_dtype=None,
    attrs=None,
    node_name: str = "pow_node",
    y_initializer: np.ndarray = None,
):
    """
    Create a single-node ONNX Pow model.

    Args:
        opset_version: ONNX opset version.
        input_shapes: List of two shapes: [x_shape, y_shape].
        input_dtypes: List of two ONNX dtypes (default: FLOAT for both).
        output_shape: Output shape (auto-computed via broadcasting if None).
        output_dtype: ONNX dtype for the output (default: same as X).
        attrs: Extra node attributes (e.g. ``{"broadcast": 1}`` for opset 1-6).
        node_name: Name for the Pow node.
        y_initializer: When provided, Y is embedded as a constant initializer
            (not a graph input).  The numpy array value is used directly.

    Returns:
        ONNX ModelProto with a single Pow node.
    """
    if input_dtypes is None:
        input_dtypes = [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT]
    if output_dtype is None:
        output_dtype = input_dtypes[0]
    if attrs is None:
        attrs = {}
    if output_shape is None:
        shape_x, shape_y = input_shapes[0], input_shapes[1]
        max_rank = max(len(shape_x), len(shape_y))
        xs = (1,) * (max_rank - len(shape_x)) + tuple(shape_x)
        ys = (1,) * (max_rank - len(shape_y)) + tuple(shape_y)
        output_shape = tuple(max(a, b) for a, b in zip(xs, ys))

    initializers = {}
    if y_initializer is not None:
        initializers["input_1"] = y_initializer

    return create_onnx_model(
        op_type="Pow",
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=[output_shape],
        output_dtypes=[output_dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name=node_name,
        input_names=["input_0", "input_1"],
        output_names=["output_0"],
        initializers=initializers,
    )


# ============================================================================
# BASIC SAME-SHAPE TESTS
# ============================================================================


@pytest.mark.transpiler
class TestPowBasic:
    """Test Pow with same-shape inputs across 1D–4D tensors."""

    def test_pow_1d_same_shape(self):
        """1-D tensors of same shape."""
        model = _create_pow_model(13, [(4,), (4,)])
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    def test_pow_2d_same_shape(self):
        """2-D tensors of same shape."""
        model = _create_pow_model(13, [(2, 3), (2, 3)])
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        y = np.array([[2.0, 1.0, 0.5], [0.0, 3.0, 2.0]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    def test_pow_3d_same_shape(self):
        """3-D tensors of same shape."""
        model = _create_pow_model(13, [(2, 3, 4), (2, 3, 4)])
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones((2, 3, 4), dtype=np.float32) * 2.0
        y = np.ones((2, 3, 4), dtype=np.float32) * 3.0
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    def test_pow_4d_same_shape(self):
        """4-D tensors of same shape."""
        model = _create_pow_model(13, [(2, 3, 4, 5), (2, 3, 4, 5)])
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones((2, 3, 4, 5), dtype=np.float32) * 2.0
        y = np.ones((2, 3, 4, 5), dtype=np.float32) * 2.0
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)


# ============================================================================
# BROADCASTING TESTS (OPSET 7+)
# ============================================================================


@pytest.mark.transpiler
class TestPowBroadcasting:
    """Pow broadcasting — NumPy-style multidirectional (opset 7+)."""

    def test_pow_scalar_y(self):
        """Scalar Y (shape ()) broadcasts over any X shape."""
        y_val = np.array(2.0, dtype=np.float32)
        model = _create_pow_model(13, [(2, 3), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**2, rtol=1e-5)

    def test_pow_1d_suffix_broadcast(self):
        """1-D Y (N,) broadcasts over last dim of 2-D X (M, N)."""
        model = _create_pow_model(13, [(3, 4), (4,)], output_shape=(3, 4))
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones((3, 4), dtype=np.float32) * 2.0
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    def test_pow_row_vector_broadcast(self):
        """Y is a row vector (1, N) broadcast over a matrix (M, N)."""
        model = _create_pow_model(13, [(3, 4), (1, 4)], output_shape=(3, 4))
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones((3, 4), dtype=np.float32) * 2.0
        y = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    def test_pow_column_vector_broadcast(self):
        """Y is a column vector (M, 1) broadcast over a matrix (M, N)."""
        model = _create_pow_model(13, [(3, 4), (3, 1)], output_shape=(3, 4))
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones((3, 4), dtype=np.float32) * 3.0
        y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    def test_pow_3d_vs_2d_broadcast(self):
        """3-D X vs 2-D Y: NumPy suffix-alignment broadcasting."""
        model = _create_pow_model(13, [(2, 3, 4), (3, 4)], output_shape=(2, 3, 4))
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones((2, 3, 4), dtype=np.float32) * 2.0
        y = np.ones((3, 4), dtype=np.float32) * 2.0
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    def test_pow_4d_vs_3d_broadcast(self):
        """4-D X vs 3-D Y: NumPy suffix-alignment broadcasting."""
        model = _create_pow_model(13, [(2, 3, 4, 5), (3, 4, 5)], output_shape=(2, 3, 4, 5))
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones((2, 3, 4, 5), dtype=np.float32) * 2.0
        y = np.ones((3, 4, 5), dtype=np.float32) * 3.0
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    def test_pow_multiple_dim1_broadcast(self):
        """Multiple size-1 dimensions in both X and Y."""
        model = _create_pow_model(13, [(5, 1, 4), (1, 3, 1)], output_shape=(5, 3, 4))
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones((5, 1, 4), dtype=np.float32) * 2.0
        y = np.ones((1, 3, 1), dtype=np.float32) * 2.0
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        assert comparison["tir_outputs"]["output_0"].shape == (5, 3, 4)
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)


# ============================================================================
# OPSET VERSION TESTS
# ============================================================================


@pytest.mark.transpiler
class TestPowOpsetVersions:
    """Test Pow across all supported opset versions."""

    @pytest.mark.parametrize("opset", [7, 9, 11, 12, 13, 15])
    def test_pow_same_shape_opsets(self, opset):
        """Same-shape Pow produces the same result across opsets 7–15."""
        model = _create_pow_model(opset, [(2, 3), (2, 3)])
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        y = np.full((2, 3), 2.0, dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Opset {opset}: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    @pytest.mark.parametrize("opset", [7, 13, 15])
    def test_pow_scalar_constant_all_opsets(self, opset):
        """Constant-initializer scalar Y is consistent across opsets."""
        y_val = np.array(2.0, dtype=np.float32)
        model = _create_pow_model(opset, [(2, 3), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Opset {opset}: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**2, rtol=1e-5)

    def test_pow_opset1_same_shape(self):
        """Opset 1 with same shapes — no broadcast attribute needed."""
        model = _create_pow_model(1, [(2, 3), (2, 3)])
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        y = np.full((2, 3), 2.0, dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)

    def test_pow_opset6_broadcast_scalar_y(self):
        """Opset 6 with broadcast=1: scalar constant Y broadcasts over X."""
        y_val = np.array(2.0, dtype=np.float32)
        model = _create_pow_model(
            6,
            [(2, 3), ()],
            attrs={"broadcast": 1},
            y_initializer=y_val,
        )
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**2, rtol=1e-5)


# ============================================================================
# DTYPE TESTS
# ============================================================================


@pytest.mark.transpiler
class TestPowDtypes:
    """Test Pow with all supported numeric dtypes (homogeneous X and Y)."""

    @pytest.mark.parametrize(
        "onnx_dtype, np_dtype",
        [
            (onnx.TensorProto.FLOAT, np.float32),
            (onnx.TensorProto.DOUBLE, np.float64),
            (onnx.TensorProto.INT32, np.int32),
            (onnx.TensorProto.INT64, np.int64),
        ],
    )
    def test_pow_basic_dtypes(self, onnx_dtype, np_dtype):
        """Pow with same dtype for X and Y across float32, float64, int32, int64."""
        exponent = np.array(2, dtype=np_dtype)
        model = _create_pow_model(
            13,
            [(3,), ()],
            input_dtypes=[onnx_dtype, onnx_dtype],
            y_initializer=exponent,
        )
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([1, 2, 3], dtype=np_dtype)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors for {np_dtype}: {comparison['errors']}"
        np.testing.assert_array_equal(comparison["tir_outputs"]["output_0"], x**2)


# ============================================================================
# HETEROGENEOUS DTYPE TESTS (OPSET 12+)
# ============================================================================


@pytest.mark.transpiler
class TestPowHeterogeneousDtypes:
    """
    Opset v12+: X has type T, Y has type T1 (may differ).
    Output Z has the same type as X (T).

    When X and Y have different dtypes, ``PowConverter`` injects an explicit
    ``CastNode`` (Y → X dtype) *before* the ``PowNode`` in the TIR graph so
    that both operands are type-matched when ``PowNode.eval()`` runs.
    """

    @pytest.mark.parametrize(
        "x_onnx_dtype, x_np_dtype, y_onnx_dtype, y_np_dtype",
        [
            (onnx.TensorProto.FLOAT, np.float32, onnx.TensorProto.INT32, np.int32),
            (onnx.TensorProto.FLOAT, np.float32, onnx.TensorProto.INT64, np.int64),
            (onnx.TensorProto.FLOAT, np.float32, onnx.TensorProto.DOUBLE, np.float64),
        ],
    )
    def test_pow_heterogeneous_types_v12(self, x_onnx_dtype, x_np_dtype, y_onnx_dtype, y_np_dtype):
        """Opset 12: output dtype must match X (T), not Y (T1)."""
        y_val = np.array(2, dtype=y_np_dtype)
        model = _create_pow_model(
            12,
            [(3,), ()],
            input_dtypes=[x_onnx_dtype, y_onnx_dtype],
            output_dtype=x_onnx_dtype,
            y_initializer=y_val,
        )
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([1.0, 2.0, 3.0], dtype=x_np_dtype)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"], f"Errors: {comparison['errors']}"
        result = comparison["tir_outputs"]["output_0"]
        assert (
            result.dtype == x_np_dtype
        ), f"Output dtype {result.dtype} should match X dtype {x_np_dtype}, not Y dtype {y_np_dtype}"
        np.testing.assert_allclose(result, x**2, rtol=1e-5)

    def test_pow_output_dtype_matches_x_v13(self):
        """Opset 13: output dtype is X's float32, even with INT64 Y."""
        y_val = np.array(3, dtype=np.int64)
        model = _create_pow_model(
            13,
            [(4,), ()],
            input_dtypes=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            output_dtype=onnx.TensorProto.FLOAT,
            y_initializer=y_val,
        )
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        result = comparison["tir_outputs"]["output_0"]
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        np.testing.assert_allclose(result, x**3, rtol=1e-5)


# ============================================================================
# CONSTANT INITIALIZER TESTS (Y AS COMPILE-TIME CONSTANT)
# ============================================================================


@pytest.mark.transpiler
class TestPowConstantY:
    """
    Tests where Y is an ONNX constant initializer.

    Even though Y is a compile-time constant, the transpiler wires it as a
    named tensor input to PowNode (not as an attribute), matching the binary
    ``forge.op.Power`` / TTIR PowOp interface.
    """

    def test_pow_square_1d(self):
        """Y=2.0 constant: square each element of a 1-D tensor."""
        y_val = np.array(2.0, dtype=np.float32)
        model = _create_pow_model(13, [(4,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(
            comparison["tir_outputs"]["output_0"],
            np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32),
            rtol=1e-5,
        )

    def test_pow_cube(self):
        """Y=3.0 constant: cube each element."""
        y_val = np.array(3.0, dtype=np.float32)
        model = _create_pow_model(13, [(2, 2, 2), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones((2, 2, 2), dtype=np.float32) * 2.0
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**3, rtol=1e-5)

    def test_pow_square_root(self):
        """Y=0.5 constant: element-wise square root."""
        y_val = np.array(0.5, dtype=np.float32)
        model = _create_pow_model(13, [(3,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], np.sqrt(x), rtol=1e-5)

    def test_pow_negative_exponent(self):
        """Y=-1.0 constant: element-wise reciprocal."""
        y_val = np.array(-1.0, dtype=np.float32)
        model = _create_pow_model(13, [(3,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], 1.0 / x, rtol=1e-5)


# ============================================================================
# ERROR CASES
# ============================================================================


@pytest.mark.transpiler
class TestPowErrors:
    """Test cases that should raise exceptions."""

    def test_pow_incompatible_shapes_opset7(self):
        """Incompatible shapes at opset 7+ should raise ConversionError."""
        model = _create_pow_model(13, [(2, 3), (2, 4)])
        transpiler = ONNXToForgeTranspiler(validate_model=True)

        with pytest.raises(ConversionError) as exc_info:
            transpiler.transpile(model)

        assert "broadcast" in str(exc_info.value).lower() or "compatible" in str(exc_info.value).lower()


# ============================================================================
# EDGE CASES
# ============================================================================


@pytest.mark.transpiler
class TestPowEdgeCases:
    """Edge cases and special exponent values."""

    def test_pow_exponent_zero(self):
        """Y=0: all outputs must be 1.0 (any base^0 == 1)."""
        y_val = np.array(0.0, dtype=np.float32)
        model = _create_pow_model(13, [(4,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([2.0, -3.0, 0.5, 100.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], np.ones(4, dtype=np.float32), rtol=1e-5)

    def test_pow_exponent_one(self):
        """Y=1: output equals input."""
        y_val = np.array(1.0, dtype=np.float32)
        model = _create_pow_model(13, [(3,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([5.0, -3.0, 0.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x, rtol=1e-5)

    def test_pow_large_exponent(self):
        """Large exponent (Y=10): verify numerical correctness."""
        y_val = np.array(10.0, dtype=np.float32)
        model = _create_pow_model(13, [(3,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([1.0, 2.0, 1.5], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**10, rtol=1e-4)

    def test_pow_fractional_exponent(self):
        """Fractional exponent Y=1/3: element-wise cube root."""
        y_val = np.array(1.0 / 3.0, dtype=np.float32)
        model = _create_pow_model(13, [(3,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([1.0, 8.0, 27.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x ** (1.0 / 3.0), rtol=1e-5)

    def test_pow_base_all_ones(self):
        """Base tensor of all-ones: output is all-ones regardless of exponent."""
        y_val = np.array(5.0, dtype=np.float32)
        model = _create_pow_model(13, [(4,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.ones(4, dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], np.ones(4, dtype=np.float32), rtol=1e-5)

    def test_pow_zero_base(self):
        """Zero base tensor with positive exponent: output must be all zeros."""
        y_val = np.array(3.0, dtype=np.float32)
        model = _create_pow_model(13, [(3,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.zeros(3, dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], np.zeros(3, dtype=np.float32), rtol=1e-5)

    def test_pow_single_element_tensor(self):
        """Single-element tensors for both X and Y."""
        y_val = np.array(3.0, dtype=np.float32)
        model = _create_pow_model(13, [(1,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([4.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], [64.0], rtol=1e-5)

    def test_pow_per_element_runtime_y(self):
        """Runtime Y with one different exponent per element."""
        model = _create_pow_model(13, [(4,), (4,)])
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"]
        np.testing.assert_allclose(
            comparison["tir_outputs"]["output_0"],
            np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32),
            rtol=1e-5,
        )

    def test_pow_negative_base_integer_exponent(self):
        """Negative base with integer exponent: standard math rules apply."""
        model = _create_pow_model(13, [(3,), (3,)])
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"input_0": x, "input_1": y})

        assert not comparison["errors"]
        np.testing.assert_allclose(comparison["tir_outputs"]["output_0"], x**y, rtol=1e-5)


# ============================================================================
# GRAPH STRUCTURE TESTS
# ============================================================================


@pytest.mark.transpiler
class TestPowGraphStructure:
    """Verify the TIR graph structure produced for Pow nodes."""

    def test_pow_always_two_tensor_inputs(self):
        """PowNode must always have exactly 2 tensor inputs — no exponent attribute."""
        y_val = np.array(2.0, dtype=np.float32)
        model = _create_pow_model(13, [(2, 3), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        pow_nodes = [n for n in tir_graph.nodes if n.op_type == "Pow"]
        assert len(pow_nodes) == 1, "Must produce exactly one Pow TIR node"

        node = pow_nodes[0]
        assert len(node.inputs) == 2, (
            "PowNode must always have 2 tensor inputs (X and Y). " "Exponent-as-attribute mode has been removed."
        )
        assert len(node.outputs) == 1
        assert "exponent" not in node.attrs, "PowNode must NOT have an 'exponent' attribute"

    def test_pow_forge_op_name_is_power(self):
        """forge_op_function_name must be 'forge.op.Power' (binary), not 'forge.op.Pow'."""
        y_val = np.array(3.0, dtype=np.float32)
        model = _create_pow_model(13, [(3,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        pow_nodes = [n for n in tir_graph.nodes if n.op_type == "Pow"]
        assert pow_nodes, "No Pow node found"
        assert pow_nodes[0].forge_op_function_name == "forge.op.Power", (
            "Must map to forge.op.Power (binary) → 'power' → TTIR PowOp, "
            "not forge.op.Pow (unary-with-attribute) which has no MLIR lowering path."
        )

    def test_pow_graph_structure_verification(self):
        """verify_tir_graph_structure passes for a basic Pow graph."""
        y_val = np.array(2.0, dtype=np.float32)
        model = _create_pow_model(13, [(2, 3), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        result = verify_tir_graph_structure(tir_graph, model, expected_op_types=["Pow"])
        assert result["output_count_match"], "Output count should match"
        assert "Pow" in result["node_types"], "Graph must contain a Pow node"

    def test_pow_src_layer_populated(self):
        """src_layer should be set on the TIR Pow node."""
        y_val = np.array(2.0, dtype=np.float32)
        model = _create_pow_model(13, [(3,), ()], y_initializer=y_val)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        pow_nodes = [n for n in tir_graph.nodes if n.op_type == "Pow"]
        assert pow_nodes
        assert pow_nodes[0].src_layer is not None, "src_layer should be populated"

    def test_pow_heterogeneous_dtype_cast_node_injected(self):
        """
        When X and Y have different dtypes (opset 12+), PowConverter inserts a
        CastNode Y → X dtype before the PowNode.  The cast output is the second
        input to PowNode.
        """
        y_val = np.array(2, dtype=np.int64)
        model = _create_pow_model(
            12,
            [(3,), ()],
            input_dtypes=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            output_dtype=onnx.TensorProto.FLOAT,
            y_initializer=y_val,
        )
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        node_types = [n.op_type for n in tir_graph.nodes]
        cast_nodes = [n for n in tir_graph.nodes if n.op_type == "Cast"]
        pow_nodes = [n for n in tir_graph.nodes if n.op_type == "Pow"]

        assert pow_nodes, "Must have a Pow node"
        assert cast_nodes, "Heterogeneous dtypes must produce a CastNode Y → X dtype before PowNode"

        cast_idx = node_types.index("Cast")
        pow_idx = node_types.index("Pow")
        assert cast_idx < pow_idx, "CastNode must precede PowNode in the TIR graph"

        pow_node = pow_nodes[0]
        assert len(pow_node.inputs) == 2
        cast_out = list(cast_nodes[0].outputs.keys())[0]
        assert cast_out in pow_node.inputs, (
            f"PowNode's Y input should be CastNode output '{cast_out}', " f"got {list(pow_node.inputs.keys())}"
        )
