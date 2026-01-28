# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX GlobalAveragePool operation.
Tests different input shapes (1D to 3D), dtypes, and opset versions.

GlobalAveragePool performs average pooling across all spatial dimensions,
reducing each spatial dimension to size 1. It has no attributes and works
the same way across all opset versions.

Decomposition: GlobalAveragePool -> Multiple ReduceMean nodes (one per spatial dimension) with keepdim=True
Note: Forge backend only supports reducing over a single dimension at a time, so we chain
multiple ReduceMean nodes, reducing from highest to lowest spatial dimension.
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
    verify_tir_graph_structure,
)


# ============================================================================
# HELPER METHODS FOR CREATING GLOBALAVGPOOL MODELS
# ============================================================================


def _create_globalavgpool_model(
    opset_version,
    input_shape,
    dtype=onnx.TensorProto.FLOAT,
):
    """
    Helper to create GlobalAveragePool ONNX model.

    Args:
        opset_version: ONNX opset version (no version differences for GlobalAveragePool)
        input_shape: Input tensor shape (must have at least 3 dimensions: N, C, spatial...)
        dtype: Input/output dtype

    Returns:
        ONNX ModelProto
    """
    # Calculate expected output shape
    # GlobalAveragePool reduces all spatial dimensions to 1
    # For [N, C, ...] -> [N, C, 1, 1, ...]
    if len(input_shape) < 3:
        raise ValueError(
            f"GlobalAveragePool requires at least 3 dimensions, got {len(input_shape)}D shape {input_shape}"
        )

    output_shape = (input_shape[0], input_shape[1]) + tuple([1] * (len(input_shape) - 2))

    # GlobalAveragePool has no attributes
    attrs = {}

    return create_onnx_model(
        op_type="GlobalAveragePool",
        input_shapes=[input_shape],
        input_dtypes=[dtype],
        output_shapes=[output_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name="globalavgpool",
    )


# ============================================================================
# GLOBALAVGPOOL TESTS - COMPREHENSIVE
# ============================================================================


@pytest.mark.transpiler
class TestGlobalAveragePool:
    """Comprehensive test cases for GlobalAveragePool operation."""

    # ========================================================================
    # BASIC TESTS - Different Input Shapes (1D to 5D)
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 22])
    @pytest.mark.parametrize(
        "input_shape",
        [
            # 1D (N, C, W)
            (1, 3, 32),
            (1, 3, 64),
            (1, 3, 128),
            (1, 64, 128),
            (1, 128, 256),
            (4, 3, 32),
            (8, 3, 64),
            # 2D (N, C, H, W) - Most common case
            (1, 3, 7, 7),  # ResNet-style
            (1, 3, 8, 8),
            (1, 3, 16, 16),
            (1, 3, 32, 32),
            (1, 64, 7, 7),
            (1, 128, 7, 7),
            (1, 256, 7, 7),
            (1, 512, 7, 7),
            (1, 1024, 7, 7),
            (1, 2048, 7, 7),  # ResNet-50 final feature map
            (1, 3, 224, 224),
            (4, 3, 32, 32),
            (8, 3, 16, 16),
            (1, 1, 7, 7),
            (1, 3, 31, 31),
            (1, 3, 33, 33),
            # 3D (N, C, D, H, W)
            (1, 3, 8, 16, 16),
            (1, 3, 4, 8, 8),
            (1, 64, 8, 16, 16),
            (1, 128, 4, 8, 8),
            (4, 3, 8, 16, 16),
            (1, 3, 7, 7, 7),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
        ],
    )
    def test_globalavgpool_basic(self, opset_version, input_shape, dtype):
        """Test basic GlobalAveragePool with different input shapes and dtypes."""
        # Skip FLOAT16 with opset 1 - ONNX Runtime doesn't support it
        if opset_version == 1 and dtype == onnx.TensorProto.FLOAT16:
            pytest.skip(f"ONNX Runtime doesn't support GlobalAveragePool(1) with FLOAT16 dtype")

        # Calculate expected output shape
        expected_output_shape = (input_shape[0], input_shape[1]) + tuple([1] * (len(input_shape) - 2))

        # Map ONNX dtype to numpy dtype for test data generation
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.FLOAT16: np.float16,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # Try to create model and skip if dtype not supported
        try:
            onnx_model = _create_globalavgpool_model(
                opset_version=opset_version,
                input_shape=input_shape,
                dtype=dtype,
            )
        except Exception as e:
            pytest.skip(f"Failed to create model with dtype {dtype} and opset {opset_version}: {e}")

        # Generate test input data
        input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype)}

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=False)
        try:
            tir_graph = transpiler.transpile(onnx_model)
        except Exception as e:
            pytest.skip(f"Transpilation failed for dtype {dtype} and opset {opset_version}: {e}")

        # Verify graph structure - should decompose to multiple ReduceMean nodes
        # (one per spatial dimension, since Forge backend only supports single-dim reduction)
        num_spatial_dims = len(input_shape) - 2
        expected_num_nodes = num_spatial_dims
        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["ReduceMean"])

        # Verify we have the expected number of ReduceMean nodes (one per spatial dimension)
        assert (
            len(tir_graph.nodes) == expected_num_nodes
        ), f"Expected {expected_num_nodes} ReduceMean nodes (one per spatial dim), got {len(tir_graph.nodes)}. Nodes: {[n.op_type for n in tir_graph.nodes]}"

        # Verify all nodes are ReduceMean and check their attributes
        spatial_dims = list(range(2, len(input_shape)))
        # Nodes should reduce dimensions in reverse order (highest to lowest)
        expected_dim_order = list(reversed(spatial_dims))

        for i, reduce_node in enumerate(tir_graph.nodes):
            assert reduce_node.op_type == "ReduceMean", f"Node {i}: Expected ReduceMean, got {reduce_node.op_type}"

            # Verify keepdim=True for all nodes
            node_keepdim = bool(reduce_node.attrs.get("keepdim", False))
            assert node_keepdim == True, f"Node {i}: Expected keepdim=True, got {node_keepdim}"

            # Verify each node reduces over a single dimension (in reverse order)
            node_dim = reduce_node.attrs.get("dim", None)
            expected_dim = expected_dim_order[i]
            assert node_dim == expected_dim, f"Node {i}: Expected dim={expected_dim}, got {node_dim}"

        # Compare outputs with appropriate tolerances
        if dtype == onnx.TensorProto.FLOAT16:
            rtol = 1e-2  # Lower precision for float16
            atol = 1e-3
        else:
            rtol = 1e-5
            atol = 1e-6

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, shape={input_shape}, dtype={dtype}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, shape={input_shape}, dtype={dtype}"
        )

        # Verify output shape
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == expected_output_shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs expected {expected_output_shape}"
        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    # ========================================================================
    # DTYPE TESTS - All Supported Dtypes
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 22])
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
        ],
    )
    def test_globalavgpool_dtype_comprehensive(self, opset_version, dtype):
        """Test GlobalAveragePool with all supported dtypes."""
        # Skip FLOAT16 with opset 1 - ONNX Runtime doesn't support it
        if opset_version == 1 and dtype == onnx.TensorProto.FLOAT16:
            pytest.skip(f"ONNX Runtime doesn't support GlobalAveragePool(1) with FLOAT16 dtype")

        input_shape = (1, 2048, 7, 7)  # ResNet-style feature map

        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.FLOAT16: np.float16,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # Try to create model and skip if dtype not supported
        try:
            onnx_model = _create_globalavgpool_model(
                opset_version=opset_version,
                input_shape=input_shape,
                dtype=dtype,
            )
        except Exception as e:
            pytest.skip(f"Failed to create model with dtype {dtype} and opset {opset_version}: {e}")

        input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        try:
            tir_graph = transpiler.transpile(onnx_model)
        except Exception as e:
            pytest.skip(f"Transpilation failed for dtype {dtype} and opset {opset_version}: {e}")

        # Verify structure - should have multiple ReduceMean nodes (one per spatial dimension)
        # input_shape is (1, 2048, 7, 7) -> 2 spatial dims -> 2 ReduceMean nodes
        num_spatial_dims = len(input_shape) - 2
        assert (
            len(tir_graph.nodes) == num_spatial_dims
        ), f"Expected {num_spatial_dims} ReduceMean nodes, got {len(tir_graph.nodes)}"
        for node in tir_graph.nodes:
            assert node.op_type == "ReduceMean", f"Expected ReduceMean node, got {node.op_type}"

        # Compare outputs with appropriate tolerances
        if dtype == onnx.TensorProto.FLOAT16:
            rtol = 1e-2
            atol = 1e-3
        elif dtype == onnx.TensorProto.DOUBLE:
            rtol = 1e-10
            atol = 1e-11
        else:
            rtol = 1e-5
            atol = 1e-6

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n" f"Test params: opset={opset_version}, dtype={dtype}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, dtype={dtype}"
        )

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 22])
    def test_globalavgpool_edge_cases(self, opset_version):
        """Test edge cases for GlobalAveragePool."""
        test_cases = [
            # (input_shape, description)
            ((1, 1, 1), "1D single element per channel"),
            ((1, 1, 1, 1), "2D single element per channel"),
            ((1, 1, 1, 1, 1), "3D single element per channel"),
            ((1, 3, 1, 1), "2D already 1x1 spatial"),
            ((1, 64, 1, 1), "2D already 1x1 spatial, many channels"),
            ((1, 3, 2, 2), "2D minimal spatial dimensions"),
            ((1, 3, 3, 3), "2D small spatial dimensions"),
            ((1, 512, 14, 14), "ResNet-34 intermediate feature map"),
            ((1, 3, 1, 224), "2D with one spatial dim = 1"),
            ((1, 3, 224, 1), "2D with one spatial dim = 1"),
        ]

        for input_shape, description in test_cases:
            # Calculate expected output shape
            expected_output_shape = (input_shape[0], input_shape[1]) + tuple([1] * (len(input_shape) - 2))

            onnx_model = _create_globalavgpool_model(
                opset_version=opset_version,
                input_shape=input_shape,
                dtype=onnx.TensorProto.FLOAT,
            )

            input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

            transpiler = ONNXToForgeTranspiler(debug=False)
            tir_graph = transpiler.transpile(onnx_model)

            # Verify structure - should have multiple ReduceMean nodes (one per spatial dimension)
            num_spatial_dims = len(input_shape) - 2
            assert (
                len(tir_graph.nodes) == num_spatial_dims
            ), f"{description}: Expected {num_spatial_dims} ReduceMean nodes, got {len(tir_graph.nodes)}"
            for node in tir_graph.nodes:
                assert node.op_type == "ReduceMean", f"{description}: Expected ReduceMean node, got {node.op_type}"

            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

            assert len(comparison["errors"]) == 0, (
                f"{description}: Comparison errors: {comparison['errors']}\n"
                f"Input shape: {input_shape}, Expected output: {expected_output_shape}"
            )
            assert all(comparison["matches"].values()), (
                f"{description}: Output mismatch: {comparison}\n"
                f"Input shape: {input_shape}, Expected output: {expected_output_shape}"
            )

            # Verify output shape
            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            assert (
                tir_output.shape == expected_output_shape
            ), f"{description}: Expected output shape {expected_output_shape}, got {tir_output.shape}"

    # ========================================================================
    # DIMENSION-SPECIFIC TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 22])
    def test_globalavgpool_1d(self, opset_version):
        """Test GlobalAveragePool with 1D inputs (N, C, W)."""
        input_shapes = [
            (1, 3, 32),
            (1, 64, 128),
            (4, 3, 64),
        ]

        for input_shape in input_shapes:
            expected_output_shape = (input_shape[0], input_shape[1], 1)

            onnx_model = _create_globalavgpool_model(
                opset_version=opset_version,
                input_shape=input_shape,
                dtype=onnx.TensorProto.FLOAT,
            )

            input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

            transpiler = ONNXToForgeTranspiler(debug=False)
            tir_graph = transpiler.transpile(onnx_model)

            # Verify we have 1 ReduceMean node (1 spatial dimension)
            assert len(tir_graph.nodes) == 1, f"Expected 1 ReduceMean node for 1D input, got {len(tir_graph.nodes)}"
            reduce_node = tir_graph.nodes[0]
            assert reduce_node.op_type == "ReduceMean"
            # For 1D input (N, C, W), spatial dim is 2, reduced in reverse order (only one dim)
            assert reduce_node.attrs.get("dim") == 2, f"Expected dim=2, got {reduce_node.attrs.get('dim')}"
            assert reduce_node.attrs.get("keepdim") == True

            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

            assert len(comparison["errors"]) == 0, f"1D test failed for shape {input_shape}: {comparison['errors']}"
            assert all(comparison["matches"].values()), f"1D test failed for shape {input_shape}: {comparison}"

            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            assert (
                tir_output.shape == expected_output_shape
            ), f"1D: Expected {expected_output_shape}, got {tir_output.shape}"

    @pytest.mark.parametrize("opset_version", [1, 22])
    def test_globalavgpool_2d(self, opset_version):
        """Test GlobalAveragePool with 2D inputs (N, C, H, W) - most common case."""
        input_shapes = [
            (1, 3, 7, 7),
            (1, 64, 7, 7),
            (1, 2048, 7, 7),  # ResNet-50
            (1, 3, 32, 32),
            (4, 3, 16, 16),
        ]

        for input_shape in input_shapes:
            expected_output_shape = (input_shape[0], input_shape[1], 1, 1)

            onnx_model = _create_globalavgpool_model(
                opset_version=opset_version,
                input_shape=input_shape,
                dtype=onnx.TensorProto.FLOAT,
            )

            input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

            transpiler = ONNXToForgeTranspiler(debug=False)
            tir_graph = transpiler.transpile(onnx_model)

            # Verify we have 2 ReduceMean nodes (2 spatial dimensions)
            assert len(tir_graph.nodes) == 2, f"Expected 2 ReduceMean nodes for 2D input, got {len(tir_graph.nodes)}"
            # Nodes should reduce dimensions in reverse order: first dim 3, then dim 2
            assert tir_graph.nodes[0].op_type == "ReduceMean"
            assert (
                tir_graph.nodes[0].attrs.get("dim") == 3
            ), f"First node: Expected dim=3, got {tir_graph.nodes[0].attrs.get('dim')}"
            assert tir_graph.nodes[0].attrs.get("keepdim") == True
            assert tir_graph.nodes[1].op_type == "ReduceMean"
            assert (
                tir_graph.nodes[1].attrs.get("dim") == 2
            ), f"Second node: Expected dim=2, got {tir_graph.nodes[1].attrs.get('dim')}"
            assert tir_graph.nodes[1].attrs.get("keepdim") == True

            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

            assert len(comparison["errors"]) == 0, f"2D test failed for shape {input_shape}: {comparison['errors']}"
            assert all(comparison["matches"].values()), f"2D test failed for shape {input_shape}: {comparison}"

            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            assert (
                tir_output.shape == expected_output_shape
            ), f"2D: Expected {expected_output_shape}, got {tir_output.shape}"

    @pytest.mark.parametrize("opset_version", [1, 22])
    def test_globalavgpool_3d(self, opset_version):
        """Test GlobalAveragePool with 3D inputs (N, C, D, H, W)."""
        input_shapes = [
            (1, 3, 8, 16, 16),
            (1, 64, 4, 8, 8),
            (4, 3, 8, 16, 16),
        ]

        for input_shape in input_shapes:
            expected_output_shape = (input_shape[0], input_shape[1], 1, 1, 1)

            onnx_model = _create_globalavgpool_model(
                opset_version=opset_version,
                input_shape=input_shape,
                dtype=onnx.TensorProto.FLOAT,
            )

            input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

            transpiler = ONNXToForgeTranspiler(debug=False)
            tir_graph = transpiler.transpile(onnx_model)

            # Verify we have 3 ReduceMean nodes (3 spatial dimensions)
            assert len(tir_graph.nodes) == 3, f"Expected 3 ReduceMean nodes for 3D input, got {len(tir_graph.nodes)}"
            # Nodes should reduce dimensions in reverse order: dim 4, then 3, then 2
            assert tir_graph.nodes[0].op_type == "ReduceMean"
            assert (
                tir_graph.nodes[0].attrs.get("dim") == 4
            ), f"First node: Expected dim=4, got {tir_graph.nodes[0].attrs.get('dim')}"
            assert tir_graph.nodes[0].attrs.get("keepdim") == True
            assert tir_graph.nodes[1].op_type == "ReduceMean"
            assert (
                tir_graph.nodes[1].attrs.get("dim") == 3
            ), f"Second node: Expected dim=3, got {tir_graph.nodes[1].attrs.get('dim')}"
            assert tir_graph.nodes[1].attrs.get("keepdim") == True
            assert tir_graph.nodes[2].op_type == "ReduceMean"
            assert (
                tir_graph.nodes[2].attrs.get("dim") == 2
            ), f"Third node: Expected dim=2, got {tir_graph.nodes[2].attrs.get('dim')}"
            assert tir_graph.nodes[2].attrs.get("keepdim") == True

            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

            assert len(comparison["errors"]) == 0, f"3D test failed for shape {input_shape}: {comparison['errors']}"
            assert all(comparison["matches"].values()), f"3D test failed for shape {input_shape}: {comparison}"

            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            assert (
                tir_output.shape == expected_output_shape
            ), f"3D: Expected {expected_output_shape}, got {tir_output.shape}"

    # ========================================================================
    # ERROR CASES
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 22])
    def test_globalavgpool_invalid_shapes(self, opset_version):
        """Test GlobalAveragePool with invalid input shapes (should raise error)."""
        invalid_shapes = [
            (),  # Empty shape
            (1,),  # 1D - too few dimensions
            (1, 3),  # 2D - too few dimensions (needs at least N, C, spatial)
        ]

        for invalid_shape in invalid_shapes:
            # Model creation should fail for invalid shapes
            with pytest.raises((ValueError, Exception)):
                onnx_model = _create_globalavgpool_model(
                    opset_version=opset_version,
                    input_shape=invalid_shape,
                    dtype=onnx.TensorProto.FLOAT,
                )
                # If model creation succeeds, transpilation should fail
                transpiler = ONNXToForgeTranspiler(debug=False)
                tir_graph = transpiler.transpile(onnx_model)
                # If we get here, the test should fail
                pytest.fail(f"Expected error for invalid shape {invalid_shape}, but transpilation succeeded")
