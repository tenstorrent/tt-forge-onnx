# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Gather operation.
Tests different input shapes, dtypes, opset versions, axis values, and conversion paths.
"""
import pytest
import numpy as np
import onnx
import torch

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.utils.exceptions import ConversionError
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestGather:
    """Comprehensive test cases for Gather operation."""

    @pytest.mark.parametrize("opset_version", [1, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, indices_shape, axis, dtype",
        [
            # Basic cases - axis=0 (Embedding path)
            ((10, 5), (3,), 0, onnx.TensorProto.FLOAT),
            ((20, 8), (5,), 0, onnx.TensorProto.FLOAT),
            ((100, 10), (10,), 0, onnx.TensorProto.FLOAT),
            # axis != 0 (IndexSelect path)
            ((5, 10), (3,), 1, onnx.TensorProto.FLOAT),
            ((3, 4, 5), (2,), 0, onnx.TensorProto.FLOAT),
            ((3, 4, 5), (2,), 1, onnx.TensorProto.FLOAT),
            ((3, 4, 5), (2,), 2, onnx.TensorProto.FLOAT),
            # Higher dimensions
            ((2, 3, 4, 5), (3,), 0, onnx.TensorProto.FLOAT),
            ((2, 3, 4, 5), (3,), 1, onnx.TensorProto.FLOAT),
            ((2, 3, 4, 5), (3,), 2, onnx.TensorProto.FLOAT),
            ((2, 3, 4, 5), (3,), 3, onnx.TensorProto.FLOAT),
            # Multi-dimensional indices
            ((10, 5), (2, 3), 0, onnx.TensorProto.FLOAT),
            ((5, 10), (2, 3), 1, onnx.TensorProto.FLOAT),
            ((3, 4, 5), (2, 3), 1, onnx.TensorProto.FLOAT),
            # Different dtypes
            ((10, 5), (3,), 0, onnx.TensorProto.DOUBLE),
            ((10, 5), (3,), 0, onnx.TensorProto.FLOAT16),
            # Integer indices (should be converted to int64)
            ((10, 5), (3,), 0, onnx.TensorProto.INT32),
        ],
    )
    def test_gather_basic(self, opset_version, data_shape, indices_shape, axis, dtype):
        """Test basic Gather operations across opset versions with various shapes and axes."""
        # Skip bfloat16 for opset < 13
        if opset_version < 13 and dtype == onnx.TensorProto.BFLOAT16:
            pytest.skip(f"BFLOAT16 is only supported in opset 13+")

        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create valid indices (within range)
        axis_size = data_shape[axis]
        indices = np.random.randint(0, axis_size, size=indices_shape, dtype=np.int64)

        # Compute expected output shape
        output_shape = list(data_shape)
        output_shape[axis] = indices_shape[0] if len(indices_shape) == 1 else indices_shape[0] * indices_shape[1]
        if len(indices_shape) > 1:
            # Multi-dimensional indices: output shape replaces axis dimension
            output_shape = output_shape[:axis] + list(indices_shape) + output_shape[axis + 1 :]
        else:
            output_shape[axis] = indices_shape[0]

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Gather",
            input_shapes=[data_shape, indices_shape],
            input_dtypes=[dtype, onnx.TensorProto.INT64],
            output_shapes=[tuple(output_shape)],
            output_dtypes=[dtype],
            attrs={"axis": axis},
            opset_version=opset_version,
            node_name="gather_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have preprocessing nodes + main node
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # Check that we have the expected node types
        node_types = [n.op_type for n in tir_graph.nodes]

        # For axis=0, should have EmbeddingNode
        if axis == 0:
            assert "Embedding" in node_types, f"Expected EmbeddingNode for axis=0, got {node_types}"
        # For axis != 0, should have IndexSelectNode
        else:
            assert "IndexSelect" in node_types, f"Expected IndexSelectNode for axis={axis}, got {node_types}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data, "input_1": indices},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, start_idx, num_indices, axis",
        [
            # Contiguous constant indices (IndexNode path)
            # Test various axes to ensure IndexNode works for all axes
            ((10, 5), 0, 3, 0),  # axis=0
            ((10, 5), 2, 5, 0),  # axis=0, different start
            ((5, 10), 0, 3, 1),  # axis=1
            ((3, 4, 5), 1, 2, 1),  # axis=1, 3D data
            ((3, 4, 5), 0, 2, 2),  # axis=2, 3D data
            ((2, 3, 4, 5), 1, 2, 3),  # axis=3, 4D data
        ],
    )
    def test_gather_contiguous_constant_indices(self, opset_version, data_shape, start_idx, num_indices, axis):
        """Test Gather with contiguous constant indices (should use IndexNode optimization)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create contiguous indices: [start_idx, start_idx+1, ..., start_idx+num_indices-1]
        indices = np.arange(start_idx, start_idx + num_indices, dtype=np.int64)

        # Compute expected output shape
        output_shape = list(data_shape)
        output_shape[axis] = num_indices

        # Create ONNX model with constant indices
        onnx_model = create_onnx_model(
            op_type="Gather",
            input_shapes=[data_shape, (num_indices,)],
            input_dtypes=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            output_shapes=[tuple(output_shape)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={"axis": axis},
            opset_version=opset_version,
            node_name="gather_test",
            initializers={"input_1": indices},  # Make indices a constant
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should use IndexNode for contiguous constant indices
        node_types = [n.op_type for n in tir_graph.nodes]
        assert "Index" in node_types, f"Expected IndexNode for contiguous constant indices, got {node_types}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, indices_shape, axis, negative_indices",
        [
            # Negative indices (should be normalized)
            ((10, 5), (3,), 0, True),
            ((5, 10), (3,), 1, True),
            ((3, 4, 5), (2,), 1, True),
        ],
    )
    def test_gather_negative_indices(self, opset_version, data_shape, indices_shape, axis, negative_indices):
        """Test Gather with negative indices (should be normalized using Where)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create indices with negative values
        axis_size = data_shape[axis]
        indices = np.array([-1, -2, 0], dtype=np.int64)  # Mix of negative and positive

        # Compute expected output shape
        output_shape = list(data_shape)
        output_shape[axis] = len(indices)

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Gather",
            input_shapes=[data_shape, indices_shape],
            input_dtypes=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            output_shapes=[tuple(output_shape)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={"axis": axis},
            opset_version=opset_version,
            node_name="gather_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have WhereNode for normalization
        node_types = [n.op_type for n in tir_graph.nodes]
        assert "Where" in node_types, f"Expected WhereNode for negative index normalization, got {node_types}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data, "input_1": indices},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, indices_shape, axis, out_of_range_indices",
        [
            # Out-of-range indices (should be clamped)
            ((10, 5), (3,), 0, True),
            ((5, 10), (3,), 1, True),
        ],
    )
    def test_gather_out_of_range_indices(self, opset_version, data_shape, indices_shape, axis, out_of_range_indices):
        """Test Gather with out-of-range indices (should be clamped using Clip)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create indices with out-of-range values
        axis_size = data_shape[axis]
        indices = np.array([-1, axis_size, axis_size + 1], dtype=np.int64)  # Mix of negative, valid, and out-of-range

        # Compute expected output shape
        output_shape = list(data_shape)
        output_shape[axis] = len(indices)

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Gather",
            input_shapes=[data_shape, indices_shape],
            input_dtypes=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            output_shapes=[tuple(output_shape)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={"axis": axis},
            opset_version=opset_version,
            node_name="gather_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have ClipNode for clamping
        node_types = [n.op_type for n in tir_graph.nodes]
        assert "Clip" in node_types, f"Expected ClipNode for index clamping, got {node_types}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data, "input_1": indices},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 11, 13])
    def test_gather_2d_indices_reshaping(self, opset_version):
        """Test Gather with 2D indices (should be reshaped for IndexSelectNode/AdvIndex)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(5, 10).astype(np.float32)

        # Create 2D indices (AdvIndex constraint: indices must be 1D or 2D)
        indices = np.random.randint(0, 10, size=(2, 3), dtype=np.int64)

        # Compute expected output shape
        output_shape = (5, 2, 3)  # axis=1, so shape becomes (5, 2, 3)

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Gather",
            input_shapes=[(5, 10), (2, 3)],
            input_dtypes=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            output_shapes=[output_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={"axis": 1},
            opset_version=opset_version,
            node_name="gather_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have IndexSelectNode (axis != 0)
        node_types = [n.op_type for n in tir_graph.nodes]
        assert "IndexSelect" in node_types, f"Expected IndexSelectNode for axis != 0, got {node_types}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data, "input_1": indices},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 11, 13])
    def test_gather_embedding_input_order(self, opset_version):
        """Test that EmbeddingNode correctly converts input order (embedding_table, indices) -> (indices, embedding_table)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(10, 5).astype(np.float32)
        indices = np.array([0, 2, 4], dtype=np.int64)

        # Create ONNX model with axis=0 (uses EmbeddingNode)
        onnx_model = create_onnx_model(
            op_type="Gather",
            input_shapes=[(10, 5), (3,)],
            input_dtypes=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            output_shapes=[(3, 5)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={"axis": 0},
            opset_version=opset_version,
            node_name="gather_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Find EmbeddingNode
        embedding_nodes = [n for n in tir_graph.nodes if n.op_type == "Embedding"]
        assert len(embedding_nodes) == 1, f"Expected 1 EmbeddingNode, got {len(embedding_nodes)}"

        embedding_node = embedding_nodes[0]

        # Check that convert_inputs_to_forge_order method exists
        assert hasattr(
            embedding_node, "convert_inputs_to_forge_order"
        ), "EmbeddingNode should have convert_inputs_to_forge_order method"

        # Test input order conversion
        original_order = embedding_node.input_names
        forge_order = embedding_node.convert_inputs_to_forge_order(original_order)

        # TIR order: (embedding_table, indices)
        # Forge order: (indices, embedding_table) - should be reversed
        assert len(forge_order) == 2, f"Expected 2 inputs, got {len(forge_order)}"
        assert forge_order[0] == original_order[1], "First input should be indices"
        assert forge_order[1] == original_order[0], "Second input should be embedding_table"

        # Verify emit() uses converted order
        emit_info = embedding_node.emit()
        assert (
            emit_info["input_names"] == forge_order
        ), f"emit() should return converted input order, got {emit_info['input_names']}, expected {forge_order}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data, "input_1": indices},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 11, 13])
    def test_gather_indexselect_attrs_conversion(self, opset_version):
        """Test that IndexSelectNode correctly converts attrs (dim -> dim for AdvIndex)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(5, 10).astype(np.float32)
        indices = np.array([0, 2, 4], dtype=np.int64)

        # Create ONNX model with axis=1 (uses IndexSelectNode)
        onnx_model = create_onnx_model(
            op_type="Gather",
            input_shapes=[(5, 10), (3,)],
            input_dtypes=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            output_shapes=[(5, 3)],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={"axis": 1},
            opset_version=opset_version,
            node_name="gather_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Find IndexSelectNode
        indexselect_nodes = [n for n in tir_graph.nodes if n.op_type == "IndexSelect"]
        assert len(indexselect_nodes) == 1, f"Expected 1 IndexSelectNode, got {len(indexselect_nodes)}"

        indexselect_node = indexselect_nodes[0]

        # Check attrs conversion
        assert indexselect_node.attrs.get("dim") == 1, f"Expected dim=1 in attrs, got {indexselect_node.attrs}"

        # Check forge_attrs conversion
        assert (
            indexselect_node.forge_attrs.get("dim") == 1
        ), f"Expected dim=1 in forge_attrs, got {indexselect_node.forge_attrs}"

        # Verify emit() includes correct attrs
        emit_info = indexselect_node.emit()
        assert emit_info["args"].get("dim") == 1, f"Expected dim=1 in emit() args, got {emit_info['args']}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data, "input_1": indices},
            atol=1e-5,
            rtol=1e-5,
        )
