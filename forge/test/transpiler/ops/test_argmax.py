# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for the ONNX ArgMax operation.

ONNX spec summary
-----------------
ArgMax (opset 1+)
  * Returns the index of the maximum value along a specified axis.
  * Attributes:
    - axis         : int, default 0.  Axis to reduce along.
    - keepdims     : int, default 1.  If 1, retain the reduced dim (size 1).
    - select_last_index (v12+): int, default 0.  If 1, return last occurrence.
  * Input  dtype: any numeric tensor (float16, float32, float64, int32, int64, …)
  * Output dtype: always tensor(int64).

Opset history
-------------
- v1  : axis, keepdims attributes.
- v11 : axis supports negative values in [-r, r-1].
- v12 : select_last_index attribute added (default 0 = first occurrence).
- v13 : bfloat16 input support.

TIR mapping
-----------
ArgMax → ArgMaxNode (ReductionShape) → torch.argmax

Covered scenarios
-----------------
Correctness
  - Default attrs (axis=0, keepdims=1) on 2-D float32 input
  - axis=0 on 2-D tensor
  - axis=1 on 2-D tensor
  - axis=0 on 3-D tensor
  - axis=1 on 3-D tensor
  - axis=2 on 3-D tensor
  - axis=0 on 4-D tensor (batch)
  - axis=-1 (negative) on 2-D tensor (v11+)
  - axis=-2 (negative) on 3-D tensor (v11+)
  - keepdims=0: reduced dim is dropped from output shape
  - keepdims=1: reduced dim becomes size 1 in output shape
  - float16 input → int64 output
  - float64 input → int64 output
  - int32  input → int64 output
  - int64  input → int64 output
  - 1-D tensor  (axis=0)
  - 3-D tensor  (various axes)
  - All-zeros input (argmax is first index, 0)
  - All-same values (argmax is 0 = first occurrence)
  - Tie-breaking: first occurrence returned by default

Opset variations
  - opset 1  (axis >= 0, no select_last_index)
  - opset 11 (negative axis accepted by spec)
  - opset 12 (select_last_index=0 → OK; select_last_index=1 → ConversionError)
  - opset 13 (float32 substitute for bfloat16)

Graph structure
  - Exactly 1 ArgMax node (op_type == "ArgMax")
  - forge_op_function_name == "Argmax"
  - Output dtype is always int64
  - Output shape: keepdims=1 → axis dim becomes 1; keepdims=0 → axis dim removed

Error handling
  - select_last_index=1 raises ConversionError

Doc examples (from onnx_argmax.md)
  - Example 1: axis=0, keepdims=1, 2×4 matrix
  - Example 2: axis=1, keepdims=1, 2×4 matrix
  - Example 3: axis=0, keepdims=0, 2×4 matrix
  - Example 4: axis=1, keepdims=0, 2×4 matrix
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
)


# ============================================================================
# HELPERS
# ============================================================================


def _argmax_model(
    input_shape,
    input_dtype=onnx.TensorProto.FLOAT,
    axis: int = 0,
    keepdims: int = 1,
    opset: int = 11,
    select_last_index: int = None,
    node_name: str = "argmax_node",
):
    """Create a single-node ONNX ArgMax model."""
    # Compute expected output shape
    rank = len(input_shape)
    norm_axis = axis if axis >= 0 else rank + axis
    if keepdims:
        output_shape = tuple(1 if i == norm_axis else d for i, d in enumerate(input_shape))
    else:
        output_shape = tuple(d for i, d in enumerate(input_shape) if i != norm_axis)

    attrs = {"axis": axis, "keepdims": keepdims}
    if select_last_index is not None and opset >= 12:
        attrs["select_last_index"] = select_last_index

    return create_onnx_model(
        op_type="ArgMax",
        input_shapes=[input_shape],
        input_dtypes=[input_dtype],
        output_shapes=[output_shape],
        output_dtypes=[onnx.TensorProto.INT64],
        attrs=attrs,
        opset_version=opset,
        node_name=node_name,
        input_names=["data"],
        output_names=["reduced"],
    )


def _run_tir(tir_graph, x: np.ndarray) -> np.ndarray:
    """Run tir_graph with numpy input and return output as numpy int64."""
    result = tir_graph.run({"data": torch.from_numpy(x)})
    return result[tir_graph.outputs[0]].detach().cpu().numpy()


def _np_argmax(x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    """Reference implementation using numpy, always returns int64."""
    result = np.argmax(x, axis=axis)
    if keepdims:
        result = np.expand_dims(result, axis=axis)
    return result.astype(np.int64)


# ============================================================================
# CORRECTNESS — DEFAULT ATTRIBUTES
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxDefaults:
    """ArgMax with default attributes: axis=0, keepdims=1."""

    def test_default_axis_0_keepdims_1_2d(self):
        """Default attrs on a 2×4 float32 matrix: axis=0, keepdims=1."""
        model = _argmax_model((2, 4), axis=0, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=0, keepdims=True)
        np.testing.assert_array_equal(out, expected)

    def test_default_output_shape_is_1xN(self):
        """axis=0, keepdims=1 → output shape is (1, N)."""
        model = _argmax_model((5, 7), axis=0, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.random.default_rng(1).standard_normal((5, 7)).astype(np.float32)
        out = _run_tir(tir, x)
        assert out.shape == (1, 7)

    def test_output_dtype_is_always_int64(self):
        """ArgMax output must be int64, regardless of input dtype."""
        model = _argmax_model((3, 5), input_dtype=onnx.TensorProto.FLOAT, axis=0, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.random.default_rng(2).standard_normal((3, 5)).astype(np.float32)
        out = _run_tir(tir, x)
        assert out.dtype == np.int64


# ============================================================================
# CORRECTNESS — AXIS VARIATIONS (2-D)
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxAxis2D:
    """ArgMax along axis=0 and axis=1 on 2-D tensors."""

    def test_axis_0_keepdims_1(self):
        """axis=0, keepdims=1 on a 3×5 float32 tensor."""
        model = _argmax_model((3, 5), axis=0, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(10)
        x = rng.standard_normal((3, 5)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=0, keepdims=True)
        assert out.shape == (1, 5)
        np.testing.assert_array_equal(out, expected)

    def test_axis_0_keepdims_0(self):
        """axis=0, keepdims=0: reduced dim is removed from output."""
        model = _argmax_model((3, 5), axis=0, keepdims=0)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(11)
        x = rng.standard_normal((3, 5)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=0, keepdims=False)
        assert out.shape == (5,)
        np.testing.assert_array_equal(out, expected)

    def test_axis_1_keepdims_1(self):
        """axis=1, keepdims=1 on a 4×6 float32 tensor."""
        model = _argmax_model((4, 6), axis=1, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(12)
        x = rng.standard_normal((4, 6)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=1, keepdims=True)
        assert out.shape == (4, 1)
        np.testing.assert_array_equal(out, expected)

    def test_axis_1_keepdims_0(self):
        """axis=1, keepdims=0: output loses the column dimension."""
        model = _argmax_model((4, 6), axis=1, keepdims=0)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(13)
        x = rng.standard_normal((4, 6)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=1, keepdims=False)
        assert out.shape == (4,)
        np.testing.assert_array_equal(out, expected)

    def test_compare_with_onnxruntime_2d(self):
        """End-to-end ORT comparison for axis=1, keepdims=1 on 2-D float32."""
        model = _argmax_model((4, 8), axis=1, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(14)
        x = rng.standard_normal((4, 8)).astype(np.float32)
        comparison = compare_tir_with_onnx(tir, model, {"data": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["reduced"],
            comparison["onnx_outputs"]["reduced"],
        )


# ============================================================================
# CORRECTNESS — AXIS VARIATIONS (3-D)
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxAxis3D:
    """ArgMax on 3-D tensors with all positive axes."""

    @pytest.mark.parametrize("axis,expected_shape", [(0, (1, 4, 5)), (1, (3, 1, 5)), (2, (3, 4, 1))])
    def test_axis_keepdims_1(self, axis, expected_shape):
        """axis ∈ {0,1,2}, keepdims=1 on a 3×4×5 float32 tensor."""
        model = _argmax_model((3, 4, 5), axis=axis, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(20 + axis)
        x = rng.standard_normal((3, 4, 5)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=axis, keepdims=True)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("axis,expected_shape", [(0, (4, 5)), (1, (3, 5)), (2, (3, 4))])
    def test_axis_keepdims_0(self, axis, expected_shape):
        """axis ∈ {0,1,2}, keepdims=0 on a 3×4×5 float32 tensor."""
        model = _argmax_model((3, 4, 5), axis=axis, keepdims=0)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(30 + axis)
        x = rng.standard_normal((3, 4, 5)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=axis, keepdims=False)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        np.testing.assert_array_equal(out, expected)

    def test_ort_compare_3d_axis1_keepdims1(self):
        """ORT validation: 3-D tensor, axis=1, keepdims=1."""
        model = _argmax_model((2, 6, 4), axis=1, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(35)
        x = rng.standard_normal((2, 6, 4)).astype(np.float32)
        comparison = compare_tir_with_onnx(tir, model, {"data": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["reduced"],
            comparison["onnx_outputs"]["reduced"],
        )


# ============================================================================
# CORRECTNESS — 4-D BATCH TENSOR
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxAxis4D:
    """ArgMax on 4-D batch tensors (B, C, H, W)."""

    @pytest.mark.parametrize("axis", [0, 1, 2, 3])
    def test_4d_batch_axis(self, axis):
        """axis ∈ {0..3}, keepdims=1 on a 2×3×4×5 float32 tensor."""
        shape = (2, 3, 4, 5)
        model = _argmax_model(shape, axis=axis, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(40 + axis)
        x = rng.standard_normal(shape).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=axis, keepdims=True)
        np.testing.assert_array_equal(out, expected)

    def test_ort_compare_4d_axis1(self):
        """ORT validation: 4-D tensor, axis=1, keepdims=1."""
        model = _argmax_model((2, 3, 4, 5), axis=1, keepdims=1, opset=13)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(44)
        x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
        comparison = compare_tir_with_onnx(tir, model, {"data": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["reduced"],
            comparison["onnx_outputs"]["reduced"],
        )


# ============================================================================
# CORRECTNESS — 1-D TENSOR
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxAxis1D:
    """ArgMax on 1-D tensors."""

    def test_1d_keepdims_1(self):
        """1-D tensor, axis=0, keepdims=1 → output shape (1,)."""
        model = _argmax_model((8,), axis=0, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([1.0, 5.0, 3.0, 2.0, 9.0, 0.0, 4.0, 7.0], dtype=np.float32)
        out = _run_tir(tir, x)

        assert out.shape == (1,)
        np.testing.assert_array_equal(out, [4])

    def test_1d_keepdims_0(self):
        """1-D tensor, axis=0, keepdims=0 → output is a scalar (shape ())."""
        model = _argmax_model((8,), axis=0, keepdims=0)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([1.0, 5.0, 3.0, 2.0, 9.0, 0.0, 4.0, 7.0], dtype=np.float32)
        out = _run_tir(tir, x)

        np.testing.assert_array_equal(out, 4)

    def test_1d_ort_compare(self):
        """ORT validation on a 1-D float32 vector."""
        model = _argmax_model((10,), axis=0, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(50)
        x = rng.standard_normal((10,)).astype(np.float32)
        comparison = compare_tir_with_onnx(tir, model, {"data": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["reduced"],
            comparison["onnx_outputs"]["reduced"],
        )


# ============================================================================
# CORRECTNESS — NEGATIVE AXIS (opset 11+)
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxNegativeAxis:
    """ArgMax with negative axis values (normalized to positive internally)."""

    def test_negative_axis_minus1_2d(self):
        """axis=-1 on 3×5 tensor is equivalent to axis=1."""
        model_neg = _argmax_model((3, 5), axis=-1, keepdims=1, opset=11)
        model_pos = _argmax_model((3, 5), axis=1, keepdims=1, opset=11)
        tir_neg = ONNXToForgeTranspiler(validate_model=True).transpile(model_neg)
        tir_pos = ONNXToForgeTranspiler(validate_model=True).transpile(model_pos)

        rng = np.random.default_rng(60)
        x = rng.standard_normal((3, 5)).astype(np.float32)
        out_neg = _run_tir(tir_neg, x)
        out_pos = _run_tir(tir_pos, x)

        np.testing.assert_array_equal(out_neg, out_pos)

    def test_negative_axis_minus2_3d(self):
        """axis=-2 on 3×4×5 tensor is equivalent to axis=1."""
        model_neg = _argmax_model((3, 4, 5), axis=-2, keepdims=1, opset=11)
        model_pos = _argmax_model((3, 4, 5), axis=1, keepdims=1, opset=11)
        tir_neg = ONNXToForgeTranspiler(validate_model=True).transpile(model_neg)
        tir_pos = ONNXToForgeTranspiler(validate_model=True).transpile(model_pos)

        rng = np.random.default_rng(61)
        x = rng.standard_normal((3, 4, 5)).astype(np.float32)
        out_neg = _run_tir(tir_neg, x)
        out_pos = _run_tir(tir_pos, x)

        np.testing.assert_array_equal(out_neg, out_pos)

    def test_negative_axis_minus1_4d(self):
        """axis=-1 on 2×3×4×5 tensor is equivalent to axis=3."""
        model_neg = _argmax_model((2, 3, 4, 5), axis=-1, keepdims=1, opset=11)
        model_pos = _argmax_model((2, 3, 4, 5), axis=3, keepdims=1, opset=11)
        tir_neg = ONNXToForgeTranspiler(validate_model=True).transpile(model_neg)
        tir_pos = ONNXToForgeTranspiler(validate_model=True).transpile(model_pos)

        rng = np.random.default_rng(62)
        x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
        out_neg = _run_tir(tir_neg, x)
        out_pos = _run_tir(tir_pos, x)

        np.testing.assert_array_equal(out_neg, out_pos)

    def test_negative_axis_ort_compare(self):
        """ORT comparison with axis=-1 on 3-D float32 tensor."""
        model = _argmax_model((2, 4, 6), axis=-1, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(63)
        x = rng.standard_normal((2, 4, 6)).astype(np.float32)
        comparison = compare_tir_with_onnx(tir, model, {"data": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["reduced"],
            comparison["onnx_outputs"]["reduced"],
        )


# ============================================================================
# CORRECTNESS — INPUT DTYPE VARIATIONS
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxInputDtypes:
    """ArgMax with various numeric input dtypes; output is always int64."""

    @pytest.mark.parametrize(
        "np_dtype,onnx_dtype",
        [
            (np.float32, onnx.TensorProto.FLOAT),
            (np.float64, onnx.TensorProto.DOUBLE),
            (np.int32, onnx.TensorProto.INT32),
            (np.int64, onnx.TensorProto.INT64),
        ],
    )
    def test_output_is_int64_for_all_dtypes(self, np_dtype, onnx_dtype):
        """Output dtype is always int64 regardless of input dtype."""
        model = _argmax_model((3, 5), input_dtype=onnx_dtype, axis=0, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(70)
        if np.issubdtype(np_dtype, np.floating):
            x = rng.standard_normal((3, 5)).astype(np_dtype)
        else:
            x = rng.integers(-100, 100, (3, 5)).astype(np_dtype)
        out = _run_tir(tir, x)

        assert out.dtype == np.int64, f"Expected int64, got {out.dtype}"

    def test_float16_input_values_correct(self):
        """float16 input produces correct argmax indices."""
        model = _argmax_model((4, 6), input_dtype=onnx.TensorProto.FLOAT16, axis=1, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(71)
        x = rng.standard_normal((4, 6)).astype(np.float16)
        out = _run_tir(tir, x)

        expected = _np_argmax(x.astype(np.float32), axis=1, keepdims=True)
        np.testing.assert_array_equal(out, expected)

    def test_int32_input_values_correct(self):
        """int32 input produces correct argmax indices."""
        model = _argmax_model((3, 7), input_dtype=onnx.TensorProto.INT32, axis=0, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(72)
        x = rng.integers(-100, 100, (3, 7)).astype(np.int32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=0, keepdims=True)
        np.testing.assert_array_equal(out, expected)

    def test_int64_input_ort_compare(self):
        """ORT comparison for int64 input, axis=1."""
        model = _argmax_model((5, 4), input_dtype=onnx.TensorProto.INT64, axis=1, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(73)
        x = rng.integers(-50, 50, (5, 4)).astype(np.int64)
        comparison = compare_tir_with_onnx(tir, model, {"data": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(
            comparison["tir_outputs"]["reduced"],
            comparison["onnx_outputs"]["reduced"],
        )


# ============================================================================
# CORRECTNESS — EDGE CASES
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxEdgeCases:
    """Edge cases: all-zeros, all-same, tie-breaking, single-element rows."""

    def test_all_zeros_first_index_returned(self):
        """All-zeros input → argmax is 0 (first occurrence)."""
        model = _argmax_model((4, 5), axis=1, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.zeros((4, 5), dtype=np.float32)
        out = _run_tir(tir, x)

        expected = np.zeros((4, 1), dtype=np.int64)
        np.testing.assert_array_equal(out, expected)

    def test_single_maximum_value(self):
        """Only one maximum → its index is returned correctly."""
        model = _argmax_model((3, 4), axis=1, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array(
            [
                [1.0, 3.0, 2.0, 0.5],
                [0.1, 0.2, 0.9, 0.3],
                [5.0, 1.0, 2.0, 3.0],
            ],
            dtype=np.float32,
        )
        out = _run_tir(tir, x)
        expected = np.array([[1], [2], [0]], dtype=np.int64)
        np.testing.assert_array_equal(out, expected)

    def test_tie_breaking_returns_first_occurrence(self):
        """When multiple elements share the maximum, the first index is returned."""
        model = _argmax_model((1, 5), axis=1, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([[3.0, 1.0, 3.0, 2.0, 3.0]], dtype=np.float32)
        out = _run_tir(tir, x)

        # Index 0 is the first occurrence of 3.0
        np.testing.assert_array_equal(out, [[0]])

    def test_single_element_per_row(self):
        """Reduce over an axis of size 1: output index is always 0."""
        model = _argmax_model((4, 1), axis=1, keepdims=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(80)
        x = rng.standard_normal((4, 1)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = np.zeros((4, 1), dtype=np.int64)
        np.testing.assert_array_equal(out, expected)

    def test_large_tensor_random(self):
        """Large 16×512 tensor matches numpy reference."""
        model = _argmax_model((16, 512), axis=1, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(81)
        x = rng.standard_normal((16, 512)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=1, keepdims=True)
        np.testing.assert_array_equal(out, expected)


# ============================================================================
# OPSET VERSIONS
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxOpsetVersions:
    """ArgMax behaviour across opset versions."""

    def test_opset_1_positive_axis(self):
        """Opset 1: basic axis=0, keepdims=1."""
        model = _argmax_model((3, 4), axis=0, keepdims=1, opset=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(90)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=0, keepdims=True)
        np.testing.assert_array_equal(out, expected)

    def test_opset_11_negative_axis(self):
        """Opset 11: negative axis is normalised correctly."""
        model = _argmax_model((4, 6), axis=-1, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(91)
        x = rng.standard_normal((4, 6)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=1, keepdims=True)
        np.testing.assert_array_equal(out, expected)

    def test_opset_12_select_last_index_0_ok(self):
        """Opset 12: select_last_index=0 is accepted (first occurrence, no error)."""
        model = _argmax_model((3, 4), axis=1, keepdims=1, opset=12, select_last_index=0)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(92)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=1, keepdims=True)
        np.testing.assert_array_equal(out, expected)

    def test_opset_12_select_last_index_1_raises(self):
        """Opset 12: select_last_index=1 must raise ConversionError (not yet supported)."""
        model = _argmax_model((3, 4), axis=1, keepdims=1, opset=12, select_last_index=1)
        with pytest.raises(ConversionError):
            ONNXToForgeTranspiler(validate_model=True).transpile(model)

    def test_opset_13_float32_as_bfloat16_substitute(self):
        """Opset 13: float32 input (bfloat16 substitute) works correctly."""
        model = _argmax_model((2, 8), axis=0, keepdims=1, opset=13)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(93)
        x = rng.standard_normal((2, 8)).astype(np.float32)
        out = _run_tir(tir, x)

        expected = _np_argmax(x, axis=0, keepdims=True)
        np.testing.assert_array_equal(out, expected)


# ============================================================================
# GRAPH STRUCTURE
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxGraphStructure:
    """TIR graph structure validation for ArgMax."""

    def _transpile(self, shape, axis=0, keepdims=1, opset=11):
        model = _argmax_model(shape, axis=axis, keepdims=keepdims, opset=opset)
        return ONNXToForgeTranspiler(validate_model=True).transpile(model), model

    def test_exactly_one_argmax_node(self):
        """TIR graph for a single ArgMax node contains exactly 1 node."""
        tir, _ = self._transpile((3, 5), axis=0)
        assert len(tir.nodes) == 1, f"Expected 1 node, got {len(tir.nodes)}"

    def test_op_type_is_argmax(self):
        """The single node has op_type == 'ArgMax'."""
        tir, _ = self._transpile((3, 5), axis=1)
        assert tir.nodes[0].op_type == "ArgMax"

    def test_forge_op_function_name(self):
        """forge_op_function_name is 'forge.op.Argmax' (property prepends 'forge.op.')."""
        tir, _ = self._transpile((3, 5), axis=0)
        node = tir.nodes[0]
        assert node.forge_op_function_name == "forge.op.Argmax"

    def _get_output_info(self, tir, output_name):
        """Return the TensorInfo for an output by looking it up in the producing node's outputs dict."""
        for node in tir.nodes:
            if output_name in node.outputs:
                return node.outputs[output_name]
        return None

    def test_output_tensor_info_dtype_is_int64(self):
        """TIR graph output TensorInfo has onnx_dtype INT64."""
        import onnx as onnx_mod

        tir, _ = self._transpile((4, 6), axis=1)
        output_name = tir.outputs[0]
        output_info = self._get_output_info(tir, output_name)
        assert output_info is not None, "Output TensorInfo not found"
        assert output_info.onnx_dtype == onnx_mod.TensorProto.INT64

    def test_output_shape_keepdims_1(self):
        """keepdims=1: output shape retains axis dim as size 1."""
        tir, _ = self._transpile((3, 5), axis=0, keepdims=1)
        output_name = tir.outputs[0]
        output_info = self._get_output_info(tir, output_name)
        assert output_info is not None, "Output TensorInfo not found"
        assert output_info.shape == (1, 5), f"Expected (1, 5), got {output_info.shape}"

    def test_output_shape_keepdims_0(self):
        """keepdims=0: axis dim is removed from output shape."""
        tir, _ = self._transpile((3, 5), axis=0, keepdims=0)
        output_name = tir.outputs[0]
        output_info = self._get_output_info(tir, output_name)
        assert output_info is not None, "Output TensorInfo not found"
        assert output_info.shape == (5,), f"Expected (5,), got {output_info.shape}"

    def test_graph_has_single_input_output(self):
        """Graph has exactly 1 input and 1 output."""
        tir, _ = self._transpile((4, 6), axis=1)
        assert len(tir.inputs) == 1
        assert len(tir.outputs) == 1

    def test_src_layer_populated(self):
        """src_layer attribute is set on the node."""
        tir, _ = self._transpile((3, 5), axis=0)
        node = tir.nodes[0]
        assert hasattr(node, "src_layer")


# ============================================================================
# DOC EXAMPLES (from docs/onnx_argmax.md)
# ============================================================================


@pytest.mark.transpiler
class TestArgMaxDocExamples:
    """
    Direct verification of all examples given in docs/onnx_argmax.md.

    Input matrix used across examples:
        x = [[2, 4, 1, 5],
             [3, 0, 6, 2]]   shape (2, 4)  dtype float32
    """

    X = np.array([[2.0, 4.0, 1.0, 5.0], [3.0, 0.0, 6.0, 2.0]], dtype=np.float32)

    def test_example1_axis0_keepdims1(self):
        """
        Example 1: axis=0, keepdims=1
        Expected: [[1, 0, 1, 0]]  shape (1, 4)
        """
        model = _argmax_model((2, 4), axis=0, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)
        out = _run_tir(tir, self.X)

        expected = np.array([[1, 0, 1, 0]], dtype=np.int64)
        assert out.shape == (1, 4)
        np.testing.assert_array_equal(out, expected)

    def test_example2_axis1_keepdims1(self):
        """
        Example 2: axis=1, keepdims=1
        Expected: [[3], [2]]  shape (2, 1)
        """
        model = _argmax_model((2, 4), axis=1, keepdims=1, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)
        out = _run_tir(tir, self.X)

        expected = np.array([[3], [2]], dtype=np.int64)
        assert out.shape == (2, 1)
        np.testing.assert_array_equal(out, expected)

    def test_example3_axis0_keepdims0(self):
        """
        Example 3: axis=0, keepdims=0
        Expected: [1, 0, 1, 0]  shape (4,)
        """
        model = _argmax_model((2, 4), axis=0, keepdims=0, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)
        out = _run_tir(tir, self.X)

        expected = np.array([1, 0, 1, 0], dtype=np.int64)
        assert out.shape == (4,)
        np.testing.assert_array_equal(out, expected)

    def test_example4_axis1_keepdims0(self):
        """
        Example 4: axis=1, keepdims=0
        Expected: [3, 2]  shape (2,)
        """
        model = _argmax_model((2, 4), axis=1, keepdims=0, opset=11)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)
        out = _run_tir(tir, self.X)

        expected = np.array([3, 2], dtype=np.int64)
        assert out.shape == (2,)
        np.testing.assert_array_equal(out, expected)
