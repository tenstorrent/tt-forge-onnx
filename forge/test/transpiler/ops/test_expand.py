# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Expand operation.
Tests all broadcasting cases for opset versions 8 and 13.
"""
import pytest
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


def _create_expand_model(
    opset_version: int,
    input_shape: tuple,
    target_shape: tuple,
    dtype: int = onnx.TensorProto.FLOAT,
    node_name: str = "expand_test",
) -> onnx.ModelProto:
    """
    Create an ONNX Expand model.
    
    Args:
        opset_version: ONNX opset version (8 or 13)
        input_shape: Input tensor shape
        target_shape: Target shape for broadcasting
        dtype: Input/output dtype
        node_name: Name for the Expand node
        
    Returns:
        ONNX ModelProto
    """
    # Shape input must be 1D tensor of int64
    shape_array = np.array(target_shape, dtype=np.int64)
    
    # Create model with shape as initializer (constant input)
    return create_onnx_model(
        op_type="Expand",
        input_shapes=[input_shape, (len(target_shape),)],  # Second input is shape tensor
        input_dtypes=[dtype, onnx.TensorProto.INT64],
        output_shapes=[target_shape],  # Output shape matches target_shape after broadcasting
        output_dtypes=[dtype],
        attrs={},
        opset_version=opset_version,
        node_name=node_name,
        input_names=["input_0", "shape"],
        initializers={"shape": shape_array},  # Shape is a constant initializer
    )


@pytest.mark.transpiler
class TestExpand:
    """
    Comprehensive test cases for Expand operation.
    Tests all broadcasting scenarios for opset versions 8 and 13.
    """

    @pytest.mark.parametrize("opset_version", [8, 13])
    @pytest.mark.parametrize(
        "input_shape, target_shape",
        [
            # Case 1: No broadcasting needed (input == target) -> IdentityNode
            ((3, 4), (3, 4)),
            ((2, 3, 4), (2, 3, 4)),
            ((1,), (1,)),
            
            # Case 2: Single dimension broadcasting (1 -> N)
            ((1, 3), (5, 3)),  # Broadcast first dim
            ((3, 1), (3, 5)),  # Broadcast second dim
            ((1, 1, 4), (3, 5, 4)),  # Broadcast first two dims
            ((2, 1, 3), (2, 4, 3)),  # Broadcast middle dim
            
            # Case 3: Rank expansion (lower -> higher rank)
            ((3,), (2, 3)),  # 1D -> 2D: (1,3) -> (2,3) ✓
            ((3, 4), (2, 3, 4)),  # 2D -> 3D: (1,3,4) -> (2,3,4) ✓
            ((3, 4), (1, 2, 3, 4)),  # 2D -> 4D: (1,1,3,4) -> (1,2,3,4) ✓
            ((1,), (3, 4, 5)),  # Scalar-like -> 3D: (1,1,1) -> (3,4,5) ✓
            
            # Case 4: Multiple dimension broadcasting
            ((1, 1), (3, 4)),  # Both dims broadcast: (1,1) -> (3,4) ✓
            ((1, 1, 1), (2, 3, 4)),  # All dims broadcast: (1,1,1) -> (2,3,4) ✓
            ((1, 3, 1), (2, 3, 4)),  # First and last dims broadcast: (1,3,1) -> (2,3,4) ✓
            
            # Case 5: Mixed broadcasting with rank expansion
            ((1, 3), (2, 3)),  # Rank expansion + dimension broadcast: (1,3) -> (2,3) ✓
            ((3, 1), (2, 3, 4)),  # Rank expansion + dimension broadcast: (1,3,1) -> (2,3,4) ✓
            ((1,), (2, 3, 4, 5)),  # Scalar -> 4D with broadcasting: (1,1,1,1) -> (2,3,4,5) ✓
            
            # Case 6: Higher dimensional cases
            ((1, 2, 3), (4, 2, 3)),  # 3D broadcast first dim
            ((1, 1, 3, 4), (2, 5, 3, 4)),  # 4D broadcast first two dims
            ((2, 1, 3, 1), (2, 4, 3, 5)),  # 4D broadcast middle dims
            
            # Case 7: Edge cases
            ((1, 1, 1, 1), (2, 3, 4, 5)),  # All 1s -> all broadcast
            ((1,), (1, 1, 1)),  # Scalar -> all 1s (no actual broadcast)
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ],
    )
    def test_expand_broadcasting_cases(self, opset_version, input_shape, target_shape, dtype):
        """Test Expand with various broadcasting scenarios."""
        # Skip float16 for opset < 13
        if opset_version < 13 and dtype == onnx.TensorProto.FLOAT16:
            pytest.skip(f"Float16 may not be fully supported in opset {opset_version}")

        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.FLOAT16: np.float16,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # Create ONNX model
        onnx_model = _create_expand_model(
            opset_version=opset_version,
            input_shape=input_shape,
            target_shape=target_shape,
            dtype=dtype,
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify graph structure
        # Expand should decompose into Unsqueeze + Broadcast nodes (or Identity if no broadcast)
        nodes = tir_graph.nodes
        
        # Check that we have the expected node types
        node_types = [n.op_type for n in nodes]
        
        # If input_shape == target_shape, should be IdentityNode
        if input_shape == target_shape:
            assert len(nodes) == 1, f"Expected 1 node for identical shapes, got {len(nodes)}"
            assert nodes[0].op_type == "Identity", f"Expected IdentityNode, got {nodes[0].op_type}"
        else:
            # Should have Unsqueeze and/or Broadcast nodes
            assert len(nodes) >= 1, f"Expected at least 1 node, got {len(nodes)}"
            assert "Broadcast" in node_types or "Unsqueeze" in node_types, (
                f"Expected Broadcast or Unsqueeze nodes, got {node_types}"
            )

        # Create test input
        np.random.seed(42)
        input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype)}

        # Compare with ONNX Runtime
        try:
            comparison = compare_tir_with_onnx(
                tir_graph,
                onnx_model,
                input_data,
                rtol=1e-5 if dtype == onnx.TensorProto.FLOAT16 else 1e-6,
                atol=1e-4 if dtype == onnx.TensorProto.FLOAT16 else 1e-6,
            )
            
            assert len(comparison["errors"]) == 0, (
                f"Comparison errors: {comparison['errors']}\n"
                f"Test params: opset={opset_version}, input_shape={input_shape}, "
                f"target_shape={target_shape}, dtype={dtype}"
            )
            assert all(comparison["matches"].values()), (
                f"Output mismatch: {comparison}\n"
                f"Test params: opset={opset_version}, input_shape={input_shape}, "
                f"target_shape={target_shape}, dtype={dtype}"
            )
            
            # Verify output shape matches expected broadcasted shape (after broadcasting rules)
            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            # Note: Output shape may differ from target_shape if target_shape has 1s
            # The actual output shape follows NumPy broadcasting rules (max of each dimension)
            # So we verify it matches the expected broadcasted shape, not necessarily target_shape
            expected_output_shape = comparison["onnx_outputs"][output_name].shape
            assert tir_output.shape == expected_output_shape, (
                f"Output shape {tir_output.shape} != expected shape {expected_output_shape} "
                f"(target_shape was {target_shape})"
            )
        except Exception as e:
            error_msg = str(e)
            # If ONNX Runtime doesn't support Expand, skip the test
            if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Expand" in error_msg:
                pytest.skip(f"ONNX Runtime doesn't support Expand operator: {error_msg}")
            raise

    @pytest.mark.parametrize("opset_version", [8, 13])
    def test_expand_single_dim_broadcast(self, opset_version):
        """Test single dimension broadcasting (most common case)."""
        input_shape = (3, 1, 5)
        target_shape = (3, 4, 5)
        
        onnx_model = _create_expand_model(
            opset_version=opset_version,
            input_shape=input_shape,
            target_shape=target_shape,
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have exactly 1 BroadcastNode (single dim broadcast)
        broadcast_nodes = [n for n in tir_graph.nodes if n.op_type == "Broadcast"]
        assert len(broadcast_nodes) == 1, (
            f"Expected 1 BroadcastNode for single-dim broadcast, got {len(broadcast_nodes)}"
        )
        
        # Verify execution
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)
        
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

    @pytest.mark.parametrize("opset_version", [8, 13])
    def test_expand_rank_expansion(self, opset_version):
        """Test rank expansion (adding leading dimensions)."""
        input_shape = (3, 4)
        target_shape = (2, 3, 4)
        
        onnx_model = _create_expand_model(
            opset_version=opset_version,
            input_shape=input_shape,
            target_shape=target_shape,
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have UnsqueezeNode(s) for rank expansion
        unsqueeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Unsqueeze"]
        assert len(unsqueeze_nodes) >= 1, (
            f"Expected at least 1 UnsqueezeNode for rank expansion, got {len(unsqueeze_nodes)}"
        )
        
        # Verify execution
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)
        
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

    @pytest.mark.parametrize("opset_version", [8, 13])
    def test_expand_multi_dim_broadcast(self, opset_version):
        """Test multiple dimension broadcasting (decomposed into multiple BroadcastNodes)."""
        input_shape = (1, 1, 4)
        target_shape = (3, 5, 4)
        
        onnx_model = _create_expand_model(
            opset_version=opset_version,
            input_shape=input_shape,
            target_shape=target_shape,
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have multiple BroadcastNodes (one per dimension)
        broadcast_nodes = [n for n in tir_graph.nodes if n.op_type == "Broadcast"]
        assert len(broadcast_nodes) >= 2, (
            f"Expected at least 2 BroadcastNodes for multi-dim broadcast, got {len(broadcast_nodes)}"
        )
        
        # Verify execution
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)
        
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

    @pytest.mark.parametrize("opset_version", [8, 13])
    def test_expand_no_broadcast(self, opset_version):
        """Test case where no broadcasting is needed (should use IdentityNode)."""
        input_shape = (3, 4, 5)
        target_shape = (3, 4, 5)
        
        onnx_model = _create_expand_model(
            opset_version=opset_version,
            input_shape=input_shape,
            target_shape=target_shape,
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have exactly 1 IdentityNode
        identity_nodes = [n for n in tir_graph.nodes if n.op_type == "Identity"]
        assert len(identity_nodes) == 1, (
            f"Expected 1 IdentityNode for no-broadcast case, got {len(identity_nodes)}"
        )
        
        # Verify execution
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)
        
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

    @pytest.mark.parametrize("opset_version", [8, 13])
    @pytest.mark.parametrize(
        "input_shape, target_shape",
        [
            # Scalar-like cases
            ((1,), (5,)),
            ((1,), (3, 4)),
            ((1,), (2, 3, 4)),
            
            # All 1s cases
            ((1, 1), (3, 4)),
            ((1, 1, 1), (2, 3, 4)),
            ((1, 1, 1, 1), (2, 3, 4, 5)),
        ],
    )
    def test_expand_from_ones(self, opset_version, input_shape, target_shape):
        """Test broadcasting from all-ones input."""
        onnx_model = _create_expand_model(
            opset_version=opset_version,
            input_shape=input_shape,
            target_shape=target_shape,
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create input with all ones
        input_data = {"input_0": np.ones(input_shape, dtype=np.float32)}
        
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)
        
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"
        
        # Verify output is all ones with expected broadcasted shape
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        expected_output_shape = comparison["onnx_outputs"][output_name].shape
        assert tir_output.shape == expected_output_shape, (
            f"Output shape {tir_output.shape} != expected shape {expected_output_shape}"
        )
        assert np.allclose(tir_output, 1.0), "Output should be all ones"

    @pytest.mark.parametrize("opset_version", [8, 13])
    def test_expand_high_dimensional(self, opset_version):
        """Test high-dimensional broadcasting."""
        input_shape = (1, 2, 1, 3, 1)
        target_shape = (4, 2, 5, 3, 6)
        
        onnx_model = _create_expand_model(
            opset_version=opset_version,
            input_shape=input_shape,
            target_shape=target_shape,
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)
        
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

    @pytest.mark.parametrize("opset_version", [8, 13])
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ],
    )
    def test_expand_dtypes(self, opset_version, dtype):
        """Test Expand with different dtypes."""
        if opset_version < 13 and dtype == onnx.TensorProto.FLOAT16:
            pytest.skip(f"Float16 may not be fully supported in opset {opset_version}")

        input_shape = (1, 3, 4)
        target_shape = (2, 3, 4)
        
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.FLOAT16: np.float16,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        onnx_model = _create_expand_model(
            opset_version=opset_version,
            input_shape=input_shape,
            target_shape=target_shape,
            dtype=dtype,
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype)}
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5 if dtype == onnx.TensorProto.FLOAT16 else 1e-6,
            atol=1e-4 if dtype == onnx.TensorProto.FLOAT16 else 1e-6,
        )
        
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

    @pytest.mark.parametrize("opset_version", [8, 13])
    def test_expand_known_values(self, opset_version):
        """Test Expand with known values to verify correctness."""
        input_shape = (1, 3)
        target_shape = (2, 3)
        
        onnx_model = _create_expand_model(
            opset_version=opset_version,
            input_shape=input_shape,
            target_shape=target_shape,
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create input with known values
        input_data = {"input_0": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}
        
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)
        
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"
        
        # Verify output shape and values
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        assert tir_output.shape == target_shape
        # Output should have the same row repeated
        assert np.allclose(tir_output[0], [1.0, 2.0, 3.0])
        assert np.allclose(tir_output[1], [1.0, 2.0, 3.0])
