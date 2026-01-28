# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for the ONNX logical operations: Not (unary) and And (binary).

ONNX spec summary
-----------------
Not (opset 1+)
  * Unary element-wise logical negation: Y = ~X
  * Input/output type: tensor(bool) only.
  * No attributes, no broadcasting.

And (opset 7+, also opset 1-6 with broadcast attribute)
  * Binary element-wise logical AND: C = A & B
  * Input/output type: tensor(bool) only.
  * Opset 1-6: limited broadcasting (broadcast=1 + optional axis attribute).
  * Opset 7+: full NumPy-style multidirectional broadcasting; attrs removed.

TIR mapping
-----------
Not  → LogicalNotNode   (ElementwiseUnaryShape)  → torch.logical_not
And  → LogicalAndNode   (BinaryBroadcastShape)   → torch.logical_and

Covered scenarios
-----------------
Not
  - All-True input  → all-False
  - All-False input → all-True
  - Mixed values
  - 2-D / 3-D / 4-D tensors
  - 1×1 edge case
  - Large tensor (128×128)
  - Double negation (NOT NOT X == X)
And
  - Both all-True  → all-True
  - Both all-False → all-False
  - Mixed values: only True & True stays True
  - Same shape (no broadcasting needed)
  - Broadcasting: scalar × 2-D tensor
  - Broadcasting: [1, N] × [M, N]
  - Broadcasting: [B, 1, N] × [1, M, N]
  - 3-D / 4-D batch tensors
  - Opset 1 with broadcast=1 attribute
  - Opset 7 (default, multidirectional)

Graph structure
  - Not  → exactly 1 LogicalNot node (op_type == "LogicalNot")
  - And  → exactly 1 LogicalAnd node (op_type == "LogicalAnd")
  - forge_op_function_name correct for both
  - Output shape matches input shape (Not)
  - Output shape is the broadcasted shape (And)
  - Output dtype is always bool for both
  - src_layer populated

Error cases
  - And with incompatible shapes raises ValueError/ConversionError
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
# HELPERS
# ============================================================================


def _not_model(input_shape, opset: int = 7, node_name: str = "not_node"):
    """Create a single-node ONNX Not model."""
    return create_onnx_model(
        op_type="Not",
        input_shapes=[input_shape],
        input_dtypes=[onnx.TensorProto.BOOL],
        output_shapes=[input_shape],
        output_dtypes=[onnx.TensorProto.BOOL],
        attrs={},
        opset_version=opset,
        node_name=node_name,
        input_names=["X"],
        output_names=["Y"],
    )


def _and_model(
    shape_a,
    shape_b,
    output_shape,
    opset: int = 7,
    broadcast: int = None,
    axis: int = None,
    node_name: str = "and_node",
):
    """Create a single-node ONNX And model."""
    attrs = {}
    if broadcast is not None:
        attrs["broadcast"] = broadcast
    if axis is not None:
        attrs["axis"] = axis
    return create_onnx_model(
        op_type="And",
        input_shapes=[shape_a, shape_b],
        input_dtypes=[onnx.TensorProto.BOOL, onnx.TensorProto.BOOL],
        output_shapes=[output_shape],
        output_dtypes=[onnx.TensorProto.BOOL],
        attrs=attrs,
        opset_version=opset,
        node_name=node_name,
        input_names=["A", "B"],
        output_names=["C"],
    )


def _run_tir(tir_graph, inputs: dict) -> np.ndarray:
    """Run tir_graph with torch-bool tensors and return the output as numpy."""
    torch_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
    result = tir_graph.run(torch_inputs)
    return result[tir_graph.outputs[0]].detach().cpu().numpy()


def _bool(arr: np.ndarray) -> np.ndarray:
    """Cast a numpy array to bool dtype."""
    return arr.astype(bool)


# ============================================================================
# NOT — BASIC CORRECTNESS
# ============================================================================


@pytest.mark.transpiler
class TestLogicalNotBasic:
    """Element-wise correctness of ONNX Not."""

    def test_not_all_true_gives_all_false(self):
        """NOT [True, True, True] == [False, False, False]."""
        model = _not_model((3,))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = _bool(np.ones((3,)))
        comparison = compare_tir_with_onnx(tir, model, {"X": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["Y"], ~x)

    def test_not_all_false_gives_all_true(self):
        """NOT [False, False, False] == [True, True, True]."""
        model = _not_model((3,))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = _bool(np.zeros((3,)))
        comparison = compare_tir_with_onnx(tir, model, {"X": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["Y"], ~x)

    def test_not_mixed_values(self):
        """NOT [T, F, T, F] == [F, T, F, T]."""
        model = _not_model((4,))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.array([True, False, True, False])
        comparison = compare_tir_with_onnx(tir, model, {"X": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["Y"], ~x)

    def test_not_2d_matrix(self):
        """NOT on a 2-D boolean matrix."""
        model = _not_model((3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=(3, 4)).astype(bool)
        comparison = compare_tir_with_onnx(tir, model, {"X": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["Y"], ~x)

    def test_not_3d_tensor(self):
        """NOT on a 3-D boolean tensor."""
        model = _not_model((2, 3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(1)
        x = rng.integers(0, 2, size=(2, 3, 4)).astype(bool)
        comparison = compare_tir_with_onnx(tir, model, {"X": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["Y"], ~x)

    def test_not_4d_tensor(self):
        """NOT on a 4-D boolean tensor."""
        model = _not_model((2, 2, 3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(2)
        x = rng.integers(0, 2, size=(2, 2, 3, 3)).astype(bool)
        comparison = compare_tir_with_onnx(tir, model, {"X": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["Y"], ~x)

    def test_not_1x1_scalar_like(self):
        """NOT on a [1, 1] matrix (edge case: minimal rank-2 tensor)."""
        model = _not_model((1, 1))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        for val in [True, False]:
            x = np.array([[val]])
            result = _run_tir(tir, {"X": x})
            np.testing.assert_array_equal(result, ~x)

    def test_not_double_negation_is_identity(self):
        """NOT NOT X == X for any boolean tensor."""
        model_not = _not_model((4, 4), node_name="not_a")
        tir_a = ONNXToForgeTranspiler(validate_model=True).transpile(model_not)
        tir_b = ONNXToForgeTranspiler(validate_model=True).transpile(model_not)

        rng = np.random.default_rng(3)
        x = rng.integers(0, 2, size=(4, 4)).astype(bool)

        out_a = _run_tir(tir_a, {"X": x})
        out_b = _run_tir(tir_b, {"X": out_a})

        np.testing.assert_array_equal(out_b, x)

    def test_not_large_tensor(self):
        """NOT on a [128, 128] tensor — numerical sanity at scale."""
        model = _not_model((128, 128))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=(128, 128)).astype(bool)
        comparison = compare_tir_with_onnx(tir, model, {"X": x})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["Y"], ~x)

    def test_not_output_dtype_is_bool(self):
        """Output tensor dtype must be bool regardless of the framework default."""
        model = _not_model((3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        x = np.ones((3, 3), dtype=bool)
        comparison = compare_tir_with_onnx(tir, model, {"X": x})

        assert not comparison["errors"], comparison["errors"]
        result = comparison["tir_outputs"]["Y"]
        assert result.dtype == bool, f"Expected bool, got {result.dtype}"

    def test_not_output_shape_equals_input_shape(self):
        """NOT must preserve the input shape exactly."""
        for shape in [(4,), (2, 3), (2, 3, 4), (2, 2, 3, 3)]:
            model = _not_model(shape)
            tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)
            x = np.ones(shape, dtype=bool)
            result = _run_tir(tir, {"X": x})
            assert result.shape == shape, f"Shape {result.shape} != {shape}"


# ============================================================================
# NOT — GRAPH STRUCTURE
# ============================================================================


@pytest.mark.transpiler
class TestLogicalNotGraphStructure:
    """Verify TIR graph topology for ONNX Not."""

    def test_not_produces_exactly_one_logical_not_node(self):
        """Transpiling Not must yield exactly 1 LogicalNot node."""
        model = _not_model((3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        op_types = [n.op_type for n in tir.nodes]
        assert op_types.count("LogicalNot") == 1, f"Expected 1 LogicalNot, got {op_types}"
        assert len(tir.nodes) == 1, f"Expected exactly 1 node, got {op_types}"

    def test_not_forge_op_function_name(self):
        """LogicalNotNode must map to forge.op.LogicalNot."""
        model = _not_model((3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        not_nodes = [n for n in tir.nodes if n.op_type == "LogicalNot"]
        assert not_nodes, "No LogicalNot node found"
        assert (
            not_nodes[0].forge_op_function_name == "forge.op.LogicalNot"
        ), f"Expected forge.op.LogicalNot, got {not_nodes[0].forge_op_function_name}"

    def test_not_node_has_one_input_one_output(self):
        """LogicalNotNode must have exactly 1 input and 1 output."""
        model = _not_model((3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        node = next(n for n in tir.nodes if n.op_type == "LogicalNot")
        assert len(node.inputs) == 1, f"Expected 1 input, got {list(node.inputs.keys())}"
        assert len(node.outputs) == 1, f"Expected 1 output, got {list(node.outputs.keys())}"

    def test_not_src_layer_populated(self):
        """Every TIR node produced by Not must have src_layer set."""
        model = _not_model((3, 3), node_name="my_not")
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        for node in tir.nodes:
            assert node.src_layer is not None, f"Node '{node.name}' ({node.op_type}) has src_layer=None"

    def test_not_output_count_matches_onnx(self):
        """verify_tir_graph_structure must report output_count_match=True."""
        model = _not_model((3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        result = verify_tir_graph_structure(tir, model, expected_op_types=["LogicalNot"])
        assert result["output_count_match"], "Output count mismatch"


# ============================================================================
# AND — BASIC CORRECTNESS (same shape, no broadcasting)
# ============================================================================


@pytest.mark.transpiler
class TestLogicalAndBasic:
    """Element-wise correctness of ONNX And with same-shape inputs."""

    def test_and_all_true_gives_true(self):
        """True & True == True for all elements."""
        model = _and_model((3,), (3,), (3,))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        a = np.ones((3,), dtype=bool)
        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": a})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a)

    def test_and_all_false_gives_false(self):
        """False & False == False for all elements."""
        model = _and_model((3,), (3,), (3,))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        a = np.zeros((3,), dtype=bool)
        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": a})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a)

    def test_and_true_and_false_gives_false(self):
        """True & False == False."""
        model = _and_model((4,), (4,), (4,))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})

        assert not comparison["errors"], comparison["errors"]
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], expected)

    def test_and_mixed_2d(self):
        """AND on 2-D matrices, result matches numpy reference."""
        model = _and_model((3, 4), (3, 4), (3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(10)
        a = rng.integers(0, 2, size=(3, 4)).astype(bool)
        b = rng.integers(0, 2, size=(3, 4)).astype(bool)
        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a & b)

    def test_and_3d_tensor(self):
        """AND on 3-D tensors, result matches numpy reference."""
        model = _and_model((2, 3, 4), (2, 3, 4), (2, 3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(11)
        a = rng.integers(0, 2, size=(2, 3, 4)).astype(bool)
        b = rng.integers(0, 2, size=(2, 3, 4)).astype(bool)
        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a & b)

    def test_and_4d_tensor(self):
        """AND on 4-D tensors."""
        model = _and_model((2, 2, 3, 3), (2, 2, 3, 3), (2, 2, 3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(12)
        a = rng.integers(0, 2, size=(2, 2, 3, 3)).astype(bool)
        b = rng.integers(0, 2, size=(2, 2, 3, 3)).astype(bool)
        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a & b)

    def test_and_1x1_all_combos(self):
        """AND with [1,1] matrices: verify all 4 boolean truth-table entries."""
        model = _and_model((1, 1), (1, 1), (1, 1))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        for va in [True, False]:
            for vb in [True, False]:
                a = np.array([[va]])
                b = np.array([[vb]])
                result = _run_tir(tir, {"A": a, "B": b})
                np.testing.assert_array_equal(result, np.array([[va and vb]]))

    def test_and_output_dtype_is_bool(self):
        """AND output tensor dtype must be bool."""
        model = _and_model((3, 3), (3, 3), (3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        a = np.ones((3, 3), dtype=bool)
        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": a})

        assert not comparison["errors"], comparison["errors"]
        result = comparison["tir_outputs"]["C"]
        assert result.dtype == bool, f"Expected bool, got {result.dtype}"

    def test_and_large_tensor(self):
        """AND on a [64, 64] tensor — numerical sanity at scale."""
        model = _and_model((64, 64), (64, 64), (64, 64))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(42)
        a = rng.integers(0, 2, size=(64, 64)).astype(bool)
        b = rng.integers(0, 2, size=(64, 64)).astype(bool)
        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})

        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a & b)


# ============================================================================
# AND — BROADCASTING (opset 7+)
# ============================================================================


@pytest.mark.transpiler
class TestLogicalAndBroadcasting:
    """NumPy-style multidirectional broadcasting for ONNX And (opset 7+)."""

    def test_and_broadcast_scalar_vs_2d(self):
        """A=[1,1] (scalar-like) broadcasts against B=[3,4]."""
        model = _and_model((1, 1), (3, 4), (3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        a_true = np.ones((1, 1), dtype=bool)
        b = np.array([[True, False, True, False], [False, True, False, True], [True, True, False, False]])

        comparison = compare_tir_with_onnx(tir, model, {"A": a_true, "B": b})
        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a_true & b)

        a_false = np.zeros((1, 1), dtype=bool)
        result_false = _run_tir(tir, {"A": a_false, "B": b})
        np.testing.assert_array_equal(result_false, np.zeros((3, 4), dtype=bool))

    def test_and_broadcast_row_vs_matrix(self):
        """A=[1, 4] broadcasts over B=[3, 4] (row broadcast)."""
        model = _and_model((1, 4), (3, 4), (3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(20)
        a = rng.integers(0, 2, size=(1, 4)).astype(bool)
        b = rng.integers(0, 2, size=(3, 4)).astype(bool)

        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})
        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a & b)

    def test_and_broadcast_column_vs_matrix(self):
        """A=[3, 1] broadcasts over B=[3, 4] (column broadcast)."""
        model = _and_model((3, 1), (3, 4), (3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(21)
        a = rng.integers(0, 2, size=(3, 1)).astype(bool)
        b = rng.integers(0, 2, size=(3, 4)).astype(bool)

        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})
        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a & b)

    def test_and_broadcast_3d_batch(self):
        """A=[2, 1, 4] broadcasts against B=[2, 3, 4] over middle dim."""
        model = _and_model((2, 1, 4), (2, 3, 4), (2, 3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(22)
        a = rng.integers(0, 2, size=(2, 1, 4)).astype(bool)
        b = rng.integers(0, 2, size=(2, 3, 4)).astype(bool)

        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})
        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a & b)

    def test_and_broadcast_4d_batch_dim(self):
        """A=[1, 2, 3, 4] broadcasts against B=[2, 2, 3, 4] over batch dim."""
        model = _and_model((1, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(23)
        a = rng.integers(0, 2, size=(1, 2, 3, 4)).astype(bool)
        b = rng.integers(0, 2, size=(2, 2, 3, 4)).astype(bool)

        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})
        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a & b)

    def test_and_broadcast_output_shape_is_broadcasted(self):
        """Output shape must be the NumPy-broadcasted shape of the two inputs."""
        model = _and_model((1, 4), (3, 1), (3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        a = np.ones((1, 4), dtype=bool)
        b = np.ones((3, 1), dtype=bool)
        result = _run_tir(tir, {"A": a, "B": b})
        assert result.shape == (3, 4), f"Expected (3,4), got {result.shape}"
        np.testing.assert_array_equal(result, np.ones((3, 4), dtype=bool))

    def test_and_broadcast_where_all_false_dominates(self):
        """If one broadcast row is all-False, all output rows for that row are False."""
        model = _and_model((3, 1), (3, 4), (3, 4))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        a = np.array([[True], [False], [True]])
        b = np.ones((3, 4), dtype=bool)
        result = _run_tir(tir, {"A": a, "B": b})
        expected = a & b
        np.testing.assert_array_equal(result, expected)
        # Row 1 (index 1) should be all False because a[1]=False
        np.testing.assert_array_equal(result[1], np.zeros((4,), dtype=bool))


# ============================================================================
# AND — OPSET VERSIONS
# ============================================================================


@pytest.mark.transpiler
class TestLogicalAndOpset:
    """Test opset-specific behaviour for ONNX And."""

    def test_and_opset7_same_shape(self):
        """And at opset 7 (modern): same-shape inputs, no attributes."""
        model = _and_model((3, 4), (3, 4), (3, 4), opset=7)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(30)
        a = rng.integers(0, 2, size=(3, 4)).astype(bool)
        b = rng.integers(0, 2, size=(3, 4)).astype(bool)
        comparison = compare_tir_with_onnx(tir, model, {"A": a, "B": b})
        assert not comparison["errors"], comparison["errors"]
        np.testing.assert_array_equal(comparison["tir_outputs"]["C"], a & b)

    def test_and_opset1_same_shape_no_broadcast_attr(self):
        """And at opset 1: same-shape inputs don't need broadcast=1."""
        model = _and_model((3, 4), (3, 4), (3, 4), opset=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(31)
        a = rng.integers(0, 2, size=(3, 4)).astype(bool)
        b = rng.integers(0, 2, size=(3, 4)).astype(bool)
        result = _run_tir(tir, {"A": a, "B": b})
        np.testing.assert_array_equal(result, a & b)

    def test_and_opset1_broadcast_flag(self):
        """And at opset 1 with broadcast=1: suffix broadcasting."""
        model = _and_model((3, 4), (4,), (3, 4), opset=1, broadcast=1)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        rng = np.random.default_rng(32)
        a = rng.integers(0, 2, size=(3, 4)).astype(bool)
        b = rng.integers(0, 2, size=(4,)).astype(bool)
        result = _run_tir(tir, {"A": a, "B": b})
        np.testing.assert_array_equal(result, a & b)


# ============================================================================
# COMBINED NOT + AND
# ============================================================================


@pytest.mark.transpiler
class TestLogicalCombined:
    """Combinations of Not and And: NAND pattern and De Morgan identities."""

    def test_nand_via_not_and(self):
        """NOT(A AND B) == NAND(A, B) verified by De Morgan."""
        not_model = _not_model((3, 4), node_name="nand_not")
        and_model = _and_model((3, 4), (3, 4), (3, 4), node_name="nand_and")

        tir_not = ONNXToForgeTranspiler(validate_model=True).transpile(not_model)
        tir_and = ONNXToForgeTranspiler(validate_model=True).transpile(and_model)

        rng = np.random.default_rng(50)
        a = rng.integers(0, 2, size=(3, 4)).astype(bool)
        b = rng.integers(0, 2, size=(3, 4)).astype(bool)

        and_result = _run_tir(tir_and, {"A": a, "B": b})
        nand_result = _run_tir(tir_not, {"X": and_result})

        np.testing.assert_array_equal(nand_result, ~(a & b))

    def test_not_a_and_not_b_equals_not_a_or_b(self):
        """NOT A AND NOT B == NOT (A OR B)  (De Morgan)."""
        not_model = _not_model((4,), node_name="de_morgan_not")
        and_model = _and_model((4,), (4,), (4,), node_name="de_morgan_and")

        tir_not = ONNXToForgeTranspiler(validate_model=True).transpile(not_model)
        tir_and = ONNXToForgeTranspiler(validate_model=True).transpile(and_model)

        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])

        not_a = _run_tir(tir_not, {"X": a})
        not_b = _run_tir(tir_not, {"X": b})
        not_a_and_not_b = _run_tir(tir_and, {"A": not_a, "B": not_b})

        # De Morgan: NOT A AND NOT B == NOT (A OR B)
        expected = ~(a | b)
        np.testing.assert_array_equal(not_a_and_not_b, expected)


# ============================================================================
# GRAPH STRUCTURE — AND
# ============================================================================


@pytest.mark.transpiler
class TestLogicalAndGraphStructure:
    """Verify TIR graph topology for ONNX And."""

    def test_and_produces_exactly_one_logical_and_node(self):
        """Transpiling And must yield exactly 1 LogicalAnd node."""
        model = _and_model((3, 3), (3, 3), (3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        op_types = [n.op_type for n in tir.nodes]
        assert op_types.count("LogicalAnd") == 1, f"Expected 1 LogicalAnd, got {op_types}"
        assert len(tir.nodes) == 1, f"Expected exactly 1 node, got {op_types}"

    def test_and_forge_op_function_name(self):
        """LogicalAndNode must map to forge.op.LogicalAnd."""
        model = _and_model((3, 3), (3, 3), (3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        and_nodes = [n for n in tir.nodes if n.op_type == "LogicalAnd"]
        assert and_nodes, "No LogicalAnd node found"
        assert (
            and_nodes[0].forge_op_function_name == "forge.op.LogicalAnd"
        ), f"Expected forge.op.LogicalAnd, got {and_nodes[0].forge_op_function_name}"

    def test_and_node_has_two_inputs_one_output(self):
        """LogicalAndNode must have exactly 2 inputs and 1 output."""
        model = _and_model((3, 3), (3, 3), (3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        node = next(n for n in tir.nodes if n.op_type == "LogicalAnd")
        assert len(node.inputs) == 2, f"Expected 2 inputs, got {list(node.inputs.keys())}"
        assert len(node.outputs) == 1, f"Expected 1 output, got {list(node.outputs.keys())}"

    def test_and_src_layer_populated(self):
        """Every TIR node produced by And must have src_layer set."""
        model = _and_model((3, 3), (3, 3), (3, 3), node_name="my_and")
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        for node in tir.nodes:
            assert node.src_layer is not None, f"Node '{node.name}' ({node.op_type}) has src_layer=None"

    def test_and_output_count_matches_onnx(self):
        """verify_tir_graph_structure must report output_count_match=True."""
        model = _and_model((3, 3), (3, 3), (3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        result = verify_tir_graph_structure(tir, model, expected_op_types=["LogicalAnd"])
        assert result["output_count_match"], "Output count mismatch"

    def test_and_output_dtype_is_bool_in_tir(self):
        """TIR output TensorInfo for And must be ONNX BOOL dtype."""
        model = _and_model((3, 3), (3, 3), (3, 3))
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        a = np.ones((3, 3), dtype=bool)
        result = _run_tir(tir, {"A": a, "B": a})
        assert result.dtype == bool, f"Expected bool, got {result.dtype}"


# ============================================================================
# ERROR CASES
# ============================================================================


@pytest.mark.transpiler
class TestLogicalErrors:
    """Edge cases that must be handled correctly or raise errors."""

    def test_and_incompatible_shapes_raises(self):
        """And with shapes [2, 3] and [4, 5] (incompatible) must raise an error."""
        model = _and_model((2, 3), (4, 5), (2, 3), opset=7)
        transpiler = ONNXToForgeTranspiler(validate_model=False)

        with pytest.raises(Exception):
            transpiler.transpile(model)

    def test_not_opset1_still_works(self):
        """Not has been in ONNX since opset 1 and must work at opset 1."""
        model = _not_model((3,), opset=1)
        tir = ONNXToForgeTranspiler(validate_model=False).transpile(model)

        x = np.array([True, False, True])
        result = _run_tir(tir, {"X": x})
        np.testing.assert_array_equal(result, ~x)

    def test_and_opset7_works(self):
        """And at opset 7 must be registered and produce correct results."""
        model = _and_model((3,), (3,), (3,), opset=7)
        tir = ONNXToForgeTranspiler(validate_model=True).transpile(model)

        a = np.array([True, True, False])
        b = np.array([True, False, False])
        result = _run_tir(tir, {"A": a, "B": b})
        np.testing.assert_array_equal(result, a & b)
