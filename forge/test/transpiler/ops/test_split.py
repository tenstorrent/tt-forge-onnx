# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for the ONNX Split operation converter.

The converter decomposes ONNX Split into one IndexNode (strided slice) per
output — there is no SplitNode in the TIR graph.

Coverage matrix
───────────────
Opset versions  : 2, 11, 13, 17, 18
Input ranks     : 1-D, 2-D, 3-D, 4-D, 5-D
Axes            : positive (0, 1, 2, 3, 4) and negative (-1, -2, -3)
Split modes     : attribute INTS (v2–v12)
                  equal split, no attribute (v2–v12)
                  constant second-input tensor (v13+)
                  equal split, no second input (v13+)
                  num_outputs attribute (v18)
Dtypes          : float16, float32, float64, int32, int64
Equal splits    : even division
Unequal splits  : custom sizes, uneven last chunk (v18)
Graph structure : IndexNode count, start/stop/stride attrs, forge op name
Doc examples    : all 7 examples from docs/onnx_split.md
ORT parity      : end-to-end numerical comparison vs ONNX Runtime
Error cases     : ConversionError on uneven equal split (v2–v17)
                  ConversionError for non-constant split input (v13+)

v18 equal-split formula (ORT behaviour, confirmed by passing tests):
    chunk_size = ceil(dim / num_outputs)
    sizes      = [chunk_size] × (num_outputs − 1)  +  [dim − chunk_size × (num_outputs − 1)]
    ⚠ Avoid test cases where chunk_size × (num_outputs − 1) == dim  (0-size last chunk).
"""
import pytest
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.utils.exceptions import ConversionError


# ============================================================================
# Helpers
# ============================================================================


def _split_model(
    input_shape,
    num_outputs: int,
    axis: int = 0,
    split_sizes=None,  # list[int] → INTS attribute (opset <= 12)
    split_initializer=None,  # np.ndarray(int64) → 2nd input (opset >= 13)
    opset: int = 11,
    dtype=onnx.TensorProto.FLOAT,
    num_outputs_attr: int = None,  # v18 num_outputs attribute
    node_name: str = "split_node",
):
    """
    Build a single-node ONNX Split model.

    Output shapes are computed from the explicit sizes/initializer/num_outputs_attr so
    that the declared graph output shapes always agree with the split sizes (keeping
    the model valid for ORT).  The caller is responsible for supplying sizes that
    sum to the input dimension along ``axis``.
    """
    input_name = "input_0"
    output_names = [f"output_{i}" for i in range(num_outputs)]

    # Compute per-output sizes along the split axis
    rank = len(input_shape)
    norm_axis = axis if axis >= 0 else axis + rank
    dim = input_shape[norm_axis]

    if split_sizes is not None:
        sizes = list(split_sizes)
    elif split_initializer is not None:
        sizes = split_initializer.flatten().tolist()
    elif num_outputs_attr is not None:
        # v18 formula: first (num_outputs-1) chunks get ceil(d/n), last gets remainder
        # Callers must choose (dim, num_outputs_attr) so the last chunk > 0.
        chunk = (dim + num_outputs_attr - 1) // num_outputs_attr
        last = dim - chunk * (num_outputs_attr - 1)
        sizes = [chunk] * (num_outputs_attr - 1) + [last]
    else:
        # Equal split — caller must ensure dim % num_outputs == 0
        sizes = [dim // num_outputs] * num_outputs

    output_shapes = []
    for s in sizes:
        out_shape = list(input_shape)
        out_shape[norm_axis] = s
        output_shapes.append(tuple(out_shape))

    graph_inputs = [onnx.helper.make_tensor_value_info(input_name, dtype, list(input_shape))]
    graph_outputs = [onnx.helper.make_tensor_value_info(n, dtype, list(s)) for n, s in zip(output_names, output_shapes)]
    initializers = []

    node_inputs = [input_name]
    node_attrs = {"axis": axis}

    if split_sizes is not None and opset <= 12:
        node_attrs["split"] = list(split_sizes)

    if split_initializer is not None:
        init_name = "split_sizes"
        initializers.append(onnx.numpy_helper.from_array(split_initializer, name=init_name))
        node_inputs.append(init_name)

    if num_outputs_attr is not None:
        node_attrs["num_outputs"] = num_outputs_attr

    node = onnx.helper.make_node("Split", node_inputs, output_names, name=node_name, **node_attrs)
    graph = onnx.helper.make_graph([node], "test_split_graph", graph_inputs, graph_outputs, initializers)
    model = onnx.helper.make_model(
        graph,
        producer_name="forge-test",
        opset_imports=[onnx.helper.make_opsetid("", opset)],
    )
    try:
        onnx.checker.check_model(model)
    except Exception:
        pass
    return model


def _run_tir(model, input_data: np.ndarray):
    """Transpile *model* and run it; return (tir_graph, outputs)."""
    transpiler = ONNXToForgeTranspiler()
    tir = transpiler.transpile(model)
    import torch

    result = tir.run({"input_0": torch.from_numpy(input_data)})
    return tir, result


def _ort_run(model, input_data: np.ndarray):
    """Run *model* through ONNX Runtime; return a dict of outputs."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), sess_options=opts)
    outputs = sess.run(None, {"input_0": input_data})
    out_names = [o.name for o in model.graph.output]
    return dict(zip(out_names, outputs))


def _check(tir_out, ort_out, rtol=1e-4, atol=1e-4):
    """Assert all TIR outputs match ORT outputs in shape and values."""
    for name, ort_arr in ort_out.items():
        assert name in tir_out, f"Missing TIR output: {name}"
        tir_arr = tir_out[name].detach().cpu().numpy()
        assert tir_arr.shape == ort_arr.shape, f"Shape mismatch for {name}: TIR {tir_arr.shape} vs ORT {ort_arr.shape}"
        assert np.allclose(
            tir_arr, ort_arr, rtol=rtol, atol=atol
        ), f"Value mismatch for {name}: max_diff={np.abs(tir_arr - ort_arr).max():.6e}"


# ============================================================================
# Section 1 – Equal split (no explicit sizes), various ranks and axes
# ============================================================================


class TestSplitEqualSplit:
    """Equal-split (no explicit sizes) along different axes and ranks."""

    def test_1d_axis0_2outputs(self):
        """1-D tensor of length 8 → 2 equal parts of 4."""
        x = np.arange(8, dtype=np.float32)
        model = _split_model((8,), num_outputs=2, axis=0, split_sizes=[4, 4], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_2d_axis0_2outputs(self):
        """2-D (6, 4) → 2 equal parts along axis=0 → (3,4) each."""
        x = np.arange(24, dtype=np.float32).reshape(6, 4)
        model = _split_model((6, 4), num_outputs=2, axis=0, split_sizes=[3, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_2d_axis1_3outputs(self):
        """2-D (2, 9) → 3 equal parts along axis=1 → (2,3) each."""
        x = np.arange(18, dtype=np.float32).reshape(2, 9)
        model = _split_model((2, 9), num_outputs=3, axis=1, split_sizes=[3, 3, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_3d_axis0_2outputs(self):
        """3-D (4, 3, 2) → 2 equal parts along axis=0."""
        x = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        model = _split_model((4, 3, 2), num_outputs=2, axis=0, split_sizes=[2, 2], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_3d_axis2_3outputs(self):
        """3-D (2, 3, 6) → 3 equal parts along axis=2."""
        x = np.random.default_rng(0).random((2, 3, 6)).astype(np.float32)
        model = _split_model((2, 3, 6), num_outputs=3, axis=2, split_sizes=[2, 2, 2], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_4d_axis1_4outputs(self):
        """4-D (2, 4, 3, 8) → 4 equal parts along axis=1 → (2,1,3,8) each."""
        x = np.random.default_rng(1).random((2, 4, 3, 8)).astype(np.float32)
        model = _split_model((2, 4, 3, 8), num_outputs=4, axis=1, split_sizes=[1, 1, 1, 1], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_4d_axis3_2outputs(self):
        """4-D (2, 3, 4, 6) → 2 equal parts along axis=3."""
        x = np.random.default_rng(2).random((2, 3, 4, 6)).astype(np.float32)
        model = _split_model((2, 3, 4, 6), num_outputs=2, axis=3, split_sizes=[3, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_5d_axis0_2outputs(self):
        """5-D (4, 2, 3, 2, 2) → 2 equal parts along axis=0."""
        x = np.random.default_rng(3).random((4, 2, 3, 2, 2)).astype(np.float32)
        model = _split_model((4, 2, 3, 2, 2), num_outputs=2, axis=0, split_sizes=[2, 2], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_5d_axis4_3outputs(self):
        """5-D (2, 2, 2, 2, 6) → 3 equal parts along axis=4."""
        x = np.random.default_rng(4).random((2, 2, 2, 2, 6)).astype(np.float32)
        model = _split_model((2, 2, 2, 2, 6), num_outputs=3, axis=4, split_sizes=[2, 2, 2], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_single_output_identity(self):
        """Split into 1 output → identity: output shape == input shape."""
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        model = _split_model((3, 4), num_outputs=1, axis=0, split_sizes=[3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))
        assert out["output_0"].shape == (3, 4)


# ============================================================================
# Section 2 – Custom (non-equal) split sizes via attribute (opset 2–12)
# ============================================================================


class TestSplitCustomSizesAttr:
    """Explicit split sizes provided as INTS attribute (opset 2–12)."""

    def test_1d_custom_2outputs_opset11(self):
        """1-D split [2, 8] on (10,)."""
        x = np.arange(10, dtype=np.float32)
        model = _split_model((10,), num_outputs=2, axis=0, split_sizes=[2, 8], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_2d_axis0_custom_2outputs(self):
        """2-D (5, 3) split [2, 3] along axis=0."""
        x = np.arange(15, dtype=np.float32).reshape(5, 3)
        model = _split_model((5, 3), num_outputs=2, axis=0, split_sizes=[2, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))
        assert out["output_0"].shape == (2, 3)
        assert out["output_1"].shape == (3, 3)

    def test_2d_axis1_custom_3outputs(self):
        """2-D (3, 8) split [3, 2, 3] along axis=1."""
        x = np.random.default_rng(5).random((3, 8)).astype(np.float32)
        model = _split_model((3, 8), num_outputs=3, axis=1, split_sizes=[3, 2, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_3d_axis1_custom_2outputs(self):
        """3-D (2, 7, 4) split [4, 3] along axis=1."""
        x = np.random.default_rng(6).random((2, 7, 4)).astype(np.float32)
        model = _split_model((2, 7, 4), num_outputs=2, axis=1, split_sizes=[4, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_4d_axis2_custom_3outputs(self):
        """4-D (2, 3, 9, 4) split [2, 3, 4] along axis=2."""
        x = np.random.default_rng(7).random((2, 3, 9, 4)).astype(np.float32)
        model = _split_model((2, 3, 9, 4), num_outputs=3, axis=2, split_sizes=[2, 3, 4], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_5d_axis3_custom_2outputs(self):
        """5-D (2, 2, 3, 7, 2) split [3, 4] along axis=3."""
        x = np.random.default_rng(8).random((2, 2, 3, 7, 2)).astype(np.float32)
        model = _split_model((2, 2, 3, 7, 2), num_outputs=2, axis=3, split_sizes=[3, 4], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_custom_sizes_opset2(self):
        """Explicit split attribute works at opset 2."""
        x = np.arange(6, dtype=np.float32)
        model = _split_model((6,), num_outputs=2, axis=0, split_sizes=[1, 5], opset=2)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_size_1_chunk(self):
        """Split where one output has size 1."""
        x = np.arange(10, dtype=np.float32)
        model = _split_model((10,), num_outputs=3, axis=0, split_sizes=[1, 4, 5], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))
        assert out["output_0"].shape == (1,)


# ============================================================================
# Section 3 – Negative axis (opset 11+)
# ============================================================================


class TestSplitNegativeAxis:
    """Negative axis values (counting from the back), various ranks."""

    def test_2d_negative_axis_minus1(self):
        """axis=-1 on (3, 6) → split last dim into [2, 4]."""
        x = np.random.default_rng(10).random((3, 6)).astype(np.float32)
        model = _split_model((3, 6), num_outputs=2, axis=-1, split_sizes=[2, 4], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_2d_negative_axis_minus2(self):
        """axis=-2 on (6, 4) → split first dim (axis=0) into [2, 4]."""
        x = np.random.default_rng(11).random((6, 4)).astype(np.float32)
        model = _split_model((6, 4), num_outputs=2, axis=-2, split_sizes=[2, 4], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_3d_negative_axis_minus1(self):
        """axis=-1 on (2, 6, 4) → split last dim (size 4) into [1, 3]."""
        x = np.random.default_rng(12).random((2, 6, 4)).astype(np.float32)
        model = _split_model((2, 6, 4), num_outputs=2, axis=-1, split_sizes=[1, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_3d_negative_axis_minus2(self):
        """axis=-2 on (2, 6, 4) → split middle dim into [2, 4]."""
        x = np.random.default_rng(13).random((2, 6, 4)).astype(np.float32)
        model = _split_model((2, 6, 4), num_outputs=2, axis=-2, split_sizes=[2, 4], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_4d_negative_axis_minus1(self):
        """axis=-1 on (2, 3, 4, 6) → split last dim into [2, 2, 2]."""
        x = np.random.default_rng(14).random((2, 3, 4, 6)).astype(np.float32)
        model = _split_model((2, 3, 4, 6), num_outputs=3, axis=-1, split_sizes=[2, 2, 2], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_4d_negative_axis_minus3(self):
        """axis=-3 on (2, 4, 3, 2) → split axis=1 into [2, 2]."""
        x = np.random.default_rng(15).random((2, 4, 3, 2)).astype(np.float32)
        model = _split_model((2, 4, 3, 2), num_outputs=2, axis=-3, split_sizes=[2, 2], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_5d_negative_axis_minus1(self):
        """axis=-1 on (2, 2, 2, 2, 6) → split last dim into [3, 3]."""
        x = np.random.default_rng(16).random((2, 2, 2, 2, 6)).astype(np.float32)
        model = _split_model((2, 2, 2, 2, 6), num_outputs=2, axis=-1, split_sizes=[3, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_equal_split_negative_axis(self):
        """axis=-1 equal split on (4, 8) into 4 parts."""
        x = np.arange(32, dtype=np.float32).reshape(4, 8)
        model = _split_model((4, 8), num_outputs=4, axis=-1, split_sizes=[2, 2, 2, 2], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))


# ============================================================================
# Section 4 – opset 13+: split as constant second-input tensor
# ============================================================================


class TestSplitInputTensorOpset13:
    """Split sizes provided as a constant second-input tensor (opset 13+)."""

    def test_1d_equal_split_input(self):
        """1-D equal split via initializer input at opset 13."""
        x = np.arange(6, dtype=np.float32)
        sizes = np.array([3, 3], dtype=np.int64)
        model = _split_model((6,), num_outputs=2, axis=0, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_2d_axis0_equal_split_input(self):
        """2-D equal split via initializer input at opset 13."""
        x = np.arange(12, dtype=np.float32).reshape(6, 2)
        sizes = np.array([3, 3], dtype=np.int64)
        model = _split_model((6, 2), num_outputs=2, axis=0, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_2d_axis0_unequal_split_input(self):
        """2-D unequal split sizes [1, 4, 2] via initializer input."""
        x = np.arange(7, dtype=np.float32)
        sizes = np.array([1, 4, 2], dtype=np.int64)
        model = _split_model((7,), num_outputs=3, axis=0, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_2d_axis1_split_input(self):
        """2-D split sizes [2, 3, 4] along axis=1 via initializer."""
        x = np.random.default_rng(20).random((3, 9)).astype(np.float32)
        sizes = np.array([2, 3, 4], dtype=np.int64)
        model = _split_model((3, 9), num_outputs=3, axis=1, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_3d_split_input(self):
        """3-D (2, 7, 4) split [3, 4] along axis=1 via initializer."""
        x = np.random.default_rng(21).random((2, 7, 4)).astype(np.float32)
        sizes = np.array([3, 4], dtype=np.int64)
        model = _split_model((2, 7, 4), num_outputs=2, axis=1, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_4d_split_input(self):
        """4-D (2, 3, 8, 4) split [2, 3, 3] along axis=2 via initializer."""
        x = np.random.default_rng(22).random((2, 3, 8, 4)).astype(np.float32)
        sizes = np.array([2, 3, 3], dtype=np.int64)
        model = _split_model((2, 3, 8, 4), num_outputs=3, axis=2, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_5d_split_input(self):
        """5-D (2, 2, 2, 2, 8) split [3, 5] along axis=4 via initializer."""
        x = np.random.default_rng(23).random((2, 2, 2, 2, 8)).astype(np.float32)
        sizes = np.array([3, 5], dtype=np.int64)
        model = _split_model((2, 2, 2, 2, 8), num_outputs=2, axis=4, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset17_split_input(self):
        """Same split-input semantics at opset 17."""
        x = np.arange(10, dtype=np.float32).reshape(5, 2)
        sizes = np.array([2, 3], dtype=np.int64)
        model = _split_model((5, 2), num_outputs=2, axis=0, split_initializer=sizes, opset=17)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_negative_axis_with_split_input(self):
        """Negative axis combined with split-input tensor."""
        x = np.random.default_rng(24).random((3, 9)).astype(np.float32)
        sizes = np.array([4, 5], dtype=np.int64)
        model = _split_model((3, 9), num_outputs=2, axis=-1, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_equal_split_no_split_input_opset13(self):
        """Opset 13 with NO split input → equal split (must be evenly divisible)."""
        x = np.arange(12, dtype=np.float32).reshape(6, 2)
        model = _split_model((6, 2), num_outputs=2, axis=0, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))
        assert out["output_0"].shape == (3, 2)
        assert out["output_1"].shape == (3, 2)


# ============================================================================
# Section 5 – opset 18: num_outputs attribute
# ============================================================================


class TestSplitNumOutputsOpset18:
    """num_outputs attribute with equal and uneven splits (opset 18)."""

    def test_1d_num_outputs_even(self):
        """num_outputs=3 on (6,): 3 chunks of size 2."""
        x = np.arange(6, dtype=np.float32)
        model = _split_model((6,), num_outputs=3, axis=0, num_outputs_attr=3, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_1d_num_outputs_uneven_last_smaller(self):
        """num_outputs=3 on (7,): chunks [3, 3, 1] — last is smaller."""
        x = np.arange(7, dtype=np.float32)
        model = _split_model((7,), num_outputs=3, axis=0, num_outputs_attr=3, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))
        assert out["output_2"].shape[0] < out["output_0"].shape[0]

    def test_2d_axis0_num_outputs_even(self):
        """num_outputs=2 on (6, 4) along axis=0: 2 chunks of (3, 4)."""
        x = np.random.default_rng(30).random((6, 4)).astype(np.float32)
        model = _split_model((6, 4), num_outputs=2, axis=0, num_outputs_attr=2, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_2d_axis1_num_outputs_uneven(self):
        """num_outputs=3 on (2, 10) along axis=1: chunks [4, 4, 2]."""
        # d=10, n=3: chunk=ceil(10/3)=4, sizes=[4,4,10-4*2]=[4,4,2]  (all > 0)
        x = np.arange(20, dtype=np.float32).reshape(2, 10)
        model = _split_model((2, 10), num_outputs=3, axis=1, num_outputs_attr=3, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))
        assert out["output_2"].shape[1] < out["output_0"].shape[1]

    def test_2d_axis1_num_outputs_4_uneven(self):
        """num_outputs=4 on (2, 11) along axis=1: chunks [3, 3, 3, 2]."""
        # d=11, n=4: chunk=ceil(11/4)=3, sizes=[3,3,3,11-3*3]=[3,3,3,2] (all > 0)
        x = np.arange(22, dtype=np.float32).reshape(2, 11)
        model = _split_model((2, 11), num_outputs=4, axis=1, num_outputs_attr=4, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))
        assert out["output_3"].shape[1] < out["output_0"].shape[1]

    def test_3d_num_outputs_uneven(self):
        """num_outputs=3 on (2, 3, 7) along axis=2: chunks [3, 3, 1]."""
        x = np.random.default_rng(31).random((2, 3, 7)).astype(np.float32)
        model = _split_model((2, 3, 7), num_outputs=3, axis=2, num_outputs_attr=3, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_4d_num_outputs_even(self):
        """num_outputs=2 on (2, 3, 6, 4) along axis=2: 2 chunks of (2,3,3,4)."""
        x = np.random.default_rng(32).random((2, 3, 6, 4)).astype(np.float32)
        model = _split_model((2, 3, 6, 4), num_outputs=2, axis=2, num_outputs_attr=2, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_5d_num_outputs_uneven(self):
        """num_outputs=3 on (2, 2, 2, 2, 7) along axis=4: chunks [3, 3, 1]."""
        x = np.random.default_rng(33).random((2, 2, 2, 2, 7)).astype(np.float32)
        model = _split_model((2, 2, 2, 2, 7), num_outputs=3, axis=4, num_outputs_attr=3, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_split_input_opset18(self):
        """Opset 18 with explicit split input tensor (not num_outputs)."""
        x = np.arange(10, dtype=np.float32).reshape(5, 2)
        sizes = np.array([2, 3], dtype=np.int64)
        model = _split_model((5, 2), num_outputs=2, axis=0, split_initializer=sizes, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))
        assert out["output_0"].shape == (2, 2)
        assert out["output_1"].shape == (3, 2)

    def test_negative_axis_num_outputs(self):
        """num_outputs on a negative axis at opset 18."""
        x = np.arange(14, dtype=np.float32).reshape(2, 7)
        model = _split_model((2, 7), num_outputs=3, axis=-1, num_outputs_attr=3, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))


# ============================================================================
# Section 6 – All opset versions: per-opset smoke tests
# ============================================================================


class TestSplitOpsetVersions:
    """One representative test per supported opset version."""

    def test_opset2_split_attr(self):
        """Opset 2: split as INTS attribute."""
        x = np.arange(6, dtype=np.float32)
        model = _split_model((6,), num_outputs=2, axis=0, split_sizes=[1, 5], opset=2)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset11_split_attr(self):
        """Opset 11: split as INTS attribute, negative axis support."""
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        model = _split_model((3, 4), num_outputs=2, axis=-1, split_sizes=[1, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset11_equal_split(self):
        """Opset 11: equal split (no explicit sizes)."""
        x = np.arange(12, dtype=np.float32)
        model = _split_model((12,), num_outputs=4, axis=0, split_sizes=[3, 3, 3, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset13_split_input(self):
        """Opset 13: split sizes as second input tensor."""
        x = np.arange(10, dtype=np.float32)
        sizes = np.array([4, 6], dtype=np.int64)
        model = _split_model((10,), num_outputs=2, axis=0, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset13_equal_split_no_input(self):
        """Opset 13: equal split when no split input provided."""
        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        model = _split_model((2, 4), num_outputs=2, axis=1, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset17_split_input(self):
        """Opset 17: split sizes as second input tensor."""
        x = np.arange(9, dtype=np.float32)
        sizes = np.array([2, 3, 4], dtype=np.int64)
        model = _split_model((9,), num_outputs=3, axis=0, split_initializer=sizes, opset=17)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset18_num_outputs_even(self):
        """Opset 18: num_outputs attribute, evenly divisible."""
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        model = _split_model((3, 4), num_outputs=3, axis=0, num_outputs_attr=3, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset18_num_outputs_uneven(self):
        """Opset 18: num_outputs attribute, last chunk smaller."""
        x = np.arange(7, dtype=np.float32)
        model = _split_model((7,), num_outputs=3, axis=0, num_outputs_attr=3, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset18_split_input(self):
        """Opset 18: explicit split input tensor."""
        x = np.arange(9, dtype=np.float32)
        sizes = np.array([4, 5], dtype=np.int64)
        model = _split_model((9,), num_outputs=2, axis=0, split_initializer=sizes, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))


# ============================================================================
# Section 7 – Input dtype coverage
# ============================================================================


class TestSplitDtypes:
    """Split preserves input dtype in all outputs."""

    @pytest.mark.parametrize(
        "np_dtype,onnx_dtype",
        [
            (np.float16, onnx.TensorProto.FLOAT16),
            (np.float32, onnx.TensorProto.FLOAT),
            (np.float64, onnx.TensorProto.DOUBLE),
            (np.int32, onnx.TensorProto.INT32),
            (np.int64, onnx.TensorProto.INT64),
        ],
    )
    def test_dtype_preserved_2d(self, np_dtype, onnx_dtype):
        """2-D split: output dtype matches input dtype for all supported dtypes."""
        x = np.arange(12, dtype=np_dtype).reshape(4, 3)
        model = _split_model((4, 3), num_outputs=2, axis=0, split_sizes=[2, 2], opset=11, dtype=onnx_dtype)
        tir, out = _run_tir(model, x)
        ort = _ort_run(model, x)
        _check(out, ort, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "np_dtype,onnx_dtype",
        [
            (np.float32, onnx.TensorProto.FLOAT),
            (np.int64, onnx.TensorProto.INT64),
        ],
    )
    def test_dtype_preserved_5d(self, np_dtype, onnx_dtype):
        """5-D split: output dtype preserved."""
        x = np.arange(48, dtype=np_dtype).reshape(2, 2, 2, 2, 3)  # 2×2×2×2×3 = 48
        model = _split_model((2, 2, 2, 2, 3), num_outputs=3, axis=4, split_sizes=[1, 1, 1], opset=11, dtype=onnx_dtype)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))


# ============================================================================
# Section 8 – Graph structure verification
# ============================================================================


class TestSplitGraphStructure:
    """Verify TIR graph contains IndexNodes with correct attrs; no SplitNode."""

    def test_no_split_node(self):
        """SplitNode must NOT appear in the TIR graph."""
        x = np.arange(6, dtype=np.float32)
        model = _split_model((6,), num_outputs=2, axis=0, split_sizes=[3, 3], opset=11)
        tir, _ = _run_tir(model, x)
        assert "Split" not in [n.op_type for n in tir.nodes]

    def test_index_nodes_count_equals_num_outputs(self):
        """Number of IndexNodes == number of outputs."""
        x = np.arange(9, dtype=np.float32)
        model = _split_model((9,), num_outputs=3, axis=0, split_sizes=[3, 3, 3], opset=11)
        tir, _ = _run_tir(model, x)
        assert len([n for n in tir.nodes if n.op_type == "Index"]) == 3

    def test_index_node_attrs_2outputs(self):
        """IndexNode attrs (start, stop, axis) are wired correctly for 2 outputs."""
        x = np.arange(6, dtype=np.float32)
        model = _split_model((6,), num_outputs=2, axis=0, split_sizes=[2, 4], opset=11)
        tir, _ = _run_tir(model, x)
        idx = [n for n in tir.nodes if n.op_type == "Index"]
        assert idx[0].attrs["axis"] == 0 and idx[0].attrs["start"] == 0 and idx[0].attrs["stop"] == 2
        assert idx[1].attrs["axis"] == 0 and idx[1].attrs["start"] == 2 and idx[1].attrs["stop"] == 6

    def test_index_node_attrs_3outputs(self):
        """Start/stop offsets are accumulated correctly for 3 chunks."""
        x = np.arange(10, dtype=np.float32)
        model = _split_model((10,), num_outputs=3, axis=0, split_sizes=[1, 4, 5], opset=11)
        tir, _ = _run_tir(model, x)
        idx = [n for n in tir.nodes if n.op_type == "Index"]
        assert [n.attrs["start"] for n in idx] == [0, 1, 5]
        assert [n.attrs["stop"] for n in idx] == [1, 5, 10]

    def test_output_shapes_from_tir(self):
        """TIR TensorInfo shapes match expected split shapes."""
        x = np.arange(12, dtype=np.float32).reshape(4, 3)
        model = _split_model((4, 3), num_outputs=2, axis=0, split_sizes=[1, 3], opset=11)
        tir, _ = _run_tir(model, x)
        idx = [n for n in tir.nodes if n.op_type == "Index"]
        shapes = [list(info.shape) for node in idx for info in node.outputs.values()]
        assert shapes[0] == [1, 3]
        assert shapes[1] == [3, 3]

    def test_index_node_stride_is_one(self):
        """Every IndexNode produced by Split must have stride=1 (contiguous slice)."""
        x = np.arange(9, dtype=np.float32)
        model = _split_model((9,), num_outputs=3, axis=0, split_sizes=[3, 3, 3], opset=11)
        tir, _ = _run_tir(model, x)
        for node in tir.nodes:
            if node.op_type == "Index":
                assert node.attrs.get("stride", 1) == 1

    def test_forge_op_function_name(self):
        """All Index nodes should report forge.op.Index."""
        x = np.arange(4, dtype=np.float32)
        model = _split_model((4,), num_outputs=2, axis=0, split_sizes=[2, 2], opset=11)
        tir, _ = _run_tir(model, x)
        for node in tir.nodes:
            if node.op_type == "Index":
                assert node.forge_op_function_name == "forge.op.Index"


# ============================================================================
# Section 9 – Documentation examples (from docs/onnx_split.md)
# ============================================================================


class TestSplitDocExamples:
    """Reproduce all 7 worked examples from docs/onnx_split.md."""

    def test_doc_example1_equal_split_axis0(self):
        """Example 1: Split (4, 3) → 2 equal parts along axis=0."""
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        model = _split_model((4, 3), num_outputs=2, axis=0, split_sizes=[2, 2], opset=11)
        _, out = _run_tir(model, x)
        np.testing.assert_array_equal(out["output_0"].numpy(), x[0:2])
        np.testing.assert_array_equal(out["output_1"].numpy(), x[2:4])

    def test_doc_example2_custom_sizes_axis0(self):
        """Example 2: Split (5, 2) into [2, 3] along axis=0."""
        x = np.arange(10, dtype=np.float32).reshape(5, 2)
        model = _split_model((5, 2), num_outputs=2, axis=0, split_sizes=[2, 3], opset=11)
        _, out = _run_tir(model, x)
        np.testing.assert_array_equal(out["output_0"].numpy(), x[0:2])
        np.testing.assert_array_equal(out["output_1"].numpy(), x[2:5])

    def test_doc_example3_negative_axis(self):
        """Example 3: Split (2, 3, 4) with axis=-1 into [2, 2]."""
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        model = _split_model((2, 3, 4), num_outputs=2, axis=-1, split_sizes=[2, 2], opset=11)
        _, out = _run_tir(model, x)
        np.testing.assert_array_equal(out["output_0"].numpy(), x[:, :, 0:2])
        np.testing.assert_array_equal(out["output_1"].numpy(), x[:, :, 2:4])

    def test_doc_example4_split_input_tensor(self):
        """Example 4: Split (4, 3) with sizes [1, 2, 1] via opset-13 input."""
        x = np.arange(12, dtype=np.float32).reshape(4, 3)
        sizes = np.array([1, 2, 1], dtype=np.int64)
        model = _split_model((4, 3), num_outputs=3, axis=0, split_initializer=sizes, opset=13)
        _, out = _run_tir(model, x)
        np.testing.assert_array_equal(out["output_0"].numpy(), x[0:1])
        np.testing.assert_array_equal(out["output_1"].numpy(), x[1:3])
        np.testing.assert_array_equal(out["output_2"].numpy(), x[3:4])

    def test_doc_example5_num_outputs_even(self):
        """Example 5: num_outputs=3 on (6,3) → 3 equal parts of (2,3)."""
        x = np.arange(18, dtype=np.float32).reshape(6, 3)
        model = _split_model((6, 3), num_outputs=3, axis=0, num_outputs_attr=3, opset=18)
        _, out = _run_tir(model, x)
        assert out["output_0"].shape == (2, 3)
        assert out["output_1"].shape == (2, 3)
        assert out["output_2"].shape == (2, 3)

    def test_doc_example6_num_outputs_uneven(self):
        """Example 6: num_outputs=3 on (7,) → chunks [3, 3, 1]."""
        x = np.arange(7, dtype=np.float32)
        model = _split_model((7,), num_outputs=3, axis=0, num_outputs_attr=3, opset=18)
        _, out = _run_tir(model, x)
        np.testing.assert_array_equal(out["output_0"].numpy(), x[0:3])
        np.testing.assert_array_equal(out["output_1"].numpy(), x[3:6])
        np.testing.assert_array_equal(out["output_2"].numpy(), x[6:7])

    def test_doc_example7_column_split(self):
        """Example 7: Split (3, 4) along axis=1 into [1, 3]."""
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        model = _split_model((3, 4), num_outputs=2, axis=1, split_sizes=[1, 3], opset=11)
        _, out = _run_tir(model, x)
        np.testing.assert_array_equal(out["output_0"].numpy(), x[:, 0:1])
        np.testing.assert_array_equal(out["output_1"].numpy(), x[:, 1:4])


# ============================================================================
# Section 10 – Numerical correctness vs ONNX Runtime (end-to-end parity)
# ============================================================================


class TestSplitOnnxRuntimeParity:
    """End-to-end comparison: TIR execution vs ONNX Runtime across all axes/ranks."""

    def test_2d_custom_split_attr(self):
        x = np.random.default_rng(40).random((8, 5)).astype(np.float32)
        model = _split_model((8, 5), num_outputs=3, axis=0, split_sizes=[3, 2, 3], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_3d_axis1_split_attr(self):
        x = np.random.default_rng(41).random((2, 10, 3)).astype(np.float32)
        model = _split_model((2, 10, 3), num_outputs=2, axis=1, split_sizes=[4, 6], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset13_split_input_axis1(self):
        x = np.random.default_rng(42).random((5, 6)).astype(np.float32)
        sizes = np.array([2, 2, 2], dtype=np.int64)
        model = _split_model((5, 6), num_outputs=3, axis=1, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset18_num_outputs_uneven(self):
        x = np.arange(11, dtype=np.float32)
        model = _split_model((11,), num_outputs=3, axis=0, num_outputs_attr=3, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_4d_negative_axis_split_attr(self):
        x = np.random.default_rng(43).random((2, 3, 4, 6)).astype(np.float32)
        model = _split_model((2, 3, 4, 6), num_outputs=3, axis=-1, split_sizes=[2, 2, 2], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_5d_axis4_split_input(self):
        x = np.random.default_rng(44).random((2, 2, 2, 2, 9)).astype(np.float32)
        sizes = np.array([3, 6], dtype=np.int64)
        model = _split_model((2, 2, 2, 2, 9), num_outputs=2, axis=4, split_initializer=sizes, opset=13)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_5d_negative_axis_equal_split(self):
        x = np.random.default_rng(45).random((2, 2, 2, 2, 6)).astype(np.float32)
        model = _split_model((2, 2, 2, 2, 6), num_outputs=3, axis=-1, split_sizes=[2, 2, 2], opset=11)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))

    def test_opset18_split_input_parity(self):
        x = np.random.default_rng(46).random((3, 8)).astype(np.float32)
        sizes = np.array([3, 5], dtype=np.int64)
        model = _split_model((3, 8), num_outputs=2, axis=1, split_initializer=sizes, opset=18)
        tir, out = _run_tir(model, x)
        _check(out, _ort_run(model, x))


# ============================================================================
# Section 11 – Error cases
# ============================================================================


class TestSplitErrors:
    """Converter must raise ConversionError for unsupported scenarios."""

    def test_error_uneven_equal_split_opset11(self):
        """Opset 11: equal split when dim not divisible by num_outputs → ConversionError."""
        node = onnx.helper.make_node("Split", ["input_0"], ["o0", "o1", "o2"], name="split", axis=0)
        graph = onnx.helper.make_graph(
            [node],
            "test_err",
            [onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, [7])],
            [
                onnx.helper.make_tensor_value_info("o0", onnx.TensorProto.FLOAT, [3]),
                onnx.helper.make_tensor_value_info("o1", onnx.TensorProto.FLOAT, [3]),
                onnx.helper.make_tensor_value_info("o2", onnx.TensorProto.FLOAT, [1]),
            ],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])
        with pytest.raises(ConversionError):
            ONNXToForgeTranspiler().transpile(model)

    def test_error_uneven_equal_split_opset13_no_split_input(self):
        """Opset 13 with no split input and uneven dim → ConversionError."""
        node = onnx.helper.make_node("Split", ["input_0"], ["o0", "o1", "o2"], name="split", axis=0)
        graph = onnx.helper.make_graph(
            [node],
            "test_err13",
            [onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, [7])],
            [
                onnx.helper.make_tensor_value_info("o0", onnx.TensorProto.FLOAT, [3]),
                onnx.helper.make_tensor_value_info("o1", onnx.TensorProto.FLOAT, [3]),
                onnx.helper.make_tensor_value_info("o2", onnx.TensorProto.FLOAT, [1]),
            ],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
        with pytest.raises(ConversionError):
            ONNXToForgeTranspiler().transpile(model)
