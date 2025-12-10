"""
Test cases for ONNX Transpose operation.
Tests different input shapes, dtypes, opset versions, and permutation behaviors.
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


class TestTranspose:
    """Comprehensive test cases for Transpose operation."""
    
    @pytest.mark.parametrize("opset_version", [1, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize("input_shape, perm, expected_shape", [
        # 2D transposes
        ((2, 3), [1, 0], (3, 2)),           # Swap dims 0 and 1
        ((2, 3), [0, 1], (2, 3)),           # Identity (no change)
        # 3D transposes
        ((2, 3, 4), [0, 1, 2], (2, 3, 4)),  # Identity
        ((2, 3, 4), [0, 2, 1], (2, 4, 3)),  # Swap last two dims
        ((2, 3, 4), [1, 0, 2], (3, 2, 4)),  # Swap first two dims
        ((2, 3, 4), [1, 2, 0], (3, 4, 2)),  # Rotate dimensions
        ((2, 3, 4), [2, 0, 1], (4, 2, 3)),  # Rotate dimensions
        ((2, 3, 4), [2, 1, 0], (4, 3, 2)),  # Reverse all dims
        # 4D transposes
        ((2, 3, 4, 5), [0, 1, 2, 3], (2, 3, 4, 5)),  # Identity
        ((2, 3, 4, 5), [0, 1, 3, 2], (2, 3, 5, 4)),  # Swap last two
        ((2, 3, 4, 5), [0, 2, 1, 3], (2, 4, 3, 5)),  # Swap middle two
        ((2, 3, 4, 5), [1, 0, 2, 3], (3, 2, 4, 5)),  # Swap first two
        ((2, 3, 4, 5), [3, 2, 1, 0], (5, 4, 3, 2)),  # Reverse all
        ((2, 3, 4, 5), [0, 2, 3, 1], (2, 4, 5, 3)),  # Complex permutation
        # 5D transposes
        ((1, 2, 3, 4, 5), [0, 1, 2, 3, 4], (1, 2, 3, 4, 5)),  # Identity
        ((1, 2, 3, 4, 5), [4, 3, 2, 1, 0], (5, 4, 3, 2, 1)),  # Reverse all
        ((1, 2, 3, 4, 5), [0, 1, 3, 2, 4], (1, 2, 4, 3, 5)),  # Swap middle
        # Edge cases (size-1 dimensions covered in test_transpose_size_one_dimensions)
    ])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    def test_transpose_basic(self, opset_version, input_shape, perm, expected_shape, dtype):
        """Test basic Transpose operations across opset versions."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip opset 1 for non-float types (opset 1 only supports float)
        if opset_version == 1 and dtype not in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")
        
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        # Create ONNX model
        attrs = {'perm': perm}
        onnx_model = create_onnx_model(
            op_type="Transpose",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="transpose_test"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        
        # If identity permutation, expect Identity node; otherwise Transpose nodes
        is_identity = perm == list(range(len(perm)))
        if is_identity:
            assert tir_graph.nodes[0].op_type == "Identity", \
                f"Expected Identity node for identity permutation, got {tir_graph.nodes[0].op_type}"
        else:
            # Should have at least one Transpose node
            transpose_nodes = [n for n in tir_graph.nodes if n.op_type == "Transpose"]
            assert len(transpose_nodes) >= 1, \
                f"Expected at least 1 Transpose node, got {[n.op_type for n in tir_graph.nodes]}"
        
        # Create test input
        if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
            input_data = {
                'input_0': np.random.randint(1, 100, size=input_shape, dtype=np_dtype)
            }
            rtol, atol = 0, 0
        else:
            input_data = {
                'input_0': np.random.randn(*input_shape).astype(np_dtype)
            }
            rtol, atol = 1e-5, 1e-6
        
        # Compare outputs
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=rtol,
            atol=atol
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, " \
            f"perm={perm}, dtype={dtype}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, " \
            f"perm={perm}, dtype={dtype}"
        
        logger.info(
            f"✓ Transpose basic test passed: opset={opset_version}, "
            f"input_shape={input_shape}, perm={perm}, dtype={dtype}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize("input_shape", [
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 5),
        (1, 2, 3),
    ])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
    ])
    def test_transpose_default_perm(self, opset_version, input_shape, dtype):
        """Test Transpose with default perm (omitted, should reverse dimensions)."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Default perm is (n-1, ..., 0) - reverse all dimensions
        expected_shape = tuple(reversed(input_shape))
        
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        # Create ONNX model without perm attribute (default behavior)
        attrs = {}  # No perm attribute
        onnx_model = create_onnx_model(
            op_type="Transpose",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="transpose_default"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure - should have Transpose nodes (not Identity, unless rank <= 1)
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        
        if len(input_shape) <= 1:
            # Rank 0 or 1: identity permutation
            assert tir_graph.nodes[0].op_type == "Identity", \
                f"Expected Identity node for rank {len(input_shape)}, got {tir_graph.nodes[0].op_type}"
        else:
            # Should have Transpose nodes
            transpose_nodes = [n for n in tir_graph.nodes if n.op_type == "Transpose"]
            assert len(transpose_nodes) >= 1, \
                f"Expected at least 1 Transpose node, got {[n.op_type for n in tir_graph.nodes]}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np_dtype)
        }
        
        # Compare outputs
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, dtype={dtype}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, dtype={dtype}"
        
        logger.info(
            f"✓ Transpose default perm test passed: opset={opset_version}, "
            f"input_shape={input_shape}, dtype={dtype}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 13, 21, 23, 24, 25])
    def test_transpose_node_indices(self, opset_version):
        """Test that TransposeNode uses positive indices (normalized from perm)."""
        input_shape = (2, 3, 4)
        perm = [0, 2, 1]  # Swap last two dims
        dtype = onnx.TensorProto.FLOAT
        
        # Create ONNX model
        attrs = {'perm': perm}
        onnx_model = create_onnx_model(
            op_type="Transpose",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[(2, 4, 3)],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="transpose_indices"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Find Transpose nodes and verify they use positive indices
        transpose_nodes = [n for n in tir_graph.nodes if n.op_type == "Transpose"]
        assert len(transpose_nodes) >= 1, \
            f"Expected at least 1 Transpose node, got {[n.op_type for n in tir_graph.nodes]}"
        
        for node in transpose_nodes:
            dim0 = node.attrs.get('dim0')
            dim1 = node.attrs.get('dim1')
            assert dim0 is not None and dim1 is not None, \
                f"TransposeNode should have dim0 and dim1 attributes"
            # Should use positive indices (normalized)
            assert isinstance(dim0, int) and dim0 >= 0, \
                f"dim0 should be non-negative int, got {dim0} (type: {type(dim0)})"
            assert isinstance(dim1, int) and dim1 >= 0, \
                f"dim1 should be non-negative int, got {dim1} (type: {type(dim1)})"
            assert dim0 < len(input_shape), \
                f"dim0 should be < {len(input_shape)}, got {dim0}"
            assert dim1 < len(input_shape), \
                f"dim1 should be < {len(input_shape)}, got {dim1}"
        
        logger.info(f"✓ Transpose node indices test passed: opset={opset_version}")
    
    
    @pytest.mark.parametrize("opset_version", [1, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize("input_shape, perm, num_swaps_expected", [
        ((2, 3), [1, 0], 1),              # Single swap
        ((2, 3, 4), [0, 2, 1], 1),         # Single swap (last two)
        ((2, 3, 4), [1, 0, 2], 1),         # Single swap (first two)
        ((2, 3, 4), [1, 2, 0], 2),         # Two swaps needed
        ((2, 3, 4), [2, 0, 1], 2),         # Two swaps needed
        ((2, 3, 4), [2, 1, 0], 1),         # Single swap (swap first and last)
        ((2, 3, 4, 5), [0, 1, 3, 2], 1),   # Single swap
        ((2, 3, 4, 5), [1, 0, 2, 3], 1),   # Single swap
        ((2, 3, 4, 5), [3, 2, 1, 0], 2),   # Two swaps needed
        ((2, 3, 4, 5), [0, 2, 3, 1], 2),   # Two swaps needed
    ])
    def test_transpose_swap_sequence(self, opset_version, input_shape, perm, num_swaps_expected):
        """Test that permutation is correctly decomposed into swap sequence."""
        dtype = onnx.TensorProto.FLOAT
        
        # Create ONNX model
        attrs = {'perm': perm}
        expected_shape = tuple(input_shape[i] for i in perm)
        
        onnx_model = create_onnx_model(
            op_type="Transpose",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="transpose_swaps"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Count Transpose nodes (excluding Identity)
        transpose_nodes = [n for n in tir_graph.nodes if n.op_type == "Transpose"]
        assert len(transpose_nodes) == num_swaps_expected, \
            f"Expected {num_swaps_expected} Transpose nodes, got {len(transpose_nodes)}. " \
            f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        
        logger.info(
            f"✓ Transpose swap sequence test passed: opset={opset_version}, "
            f"input_shape={input_shape}, perm={perm}, swaps={num_swaps_expected}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize("input_shape, perm", [
        # High-dimensional cases
        ((1, 2, 3, 4, 5, 6), [5, 4, 3, 2, 1, 0]),  # 6D reverse
        ((1, 1, 2, 3, 4, 5), [0, 1, 5, 4, 3, 2]),  # 6D partial reverse
        ((1, 1, 1, 2, 3, 4), [0, 1, 2, 5, 4, 3]),  # 6D with many ones
        # Large tensors
        ((10, 20, 30), [2, 0, 1]),  # Large 3D
        ((5, 10, 15, 20), [3, 2, 1, 0]),  # Large 4D
        # Complex permutations requiring many swaps
        ((2, 3, 4, 5, 6), [4, 3, 2, 1, 0]),  # 5D reverse (multiple swaps)
        ((1, 2, 3, 4, 5, 6, 7), [6, 5, 4, 3, 2, 1, 0]),  # 7D reverse (fixed: 7D input with 7-element perm)
    ])
    def test_transpose_high_dimensional(self, opset_version, input_shape, perm):
        """Test Transpose with high-dimensional tensors and complex permutations."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        dtype = onnx.TensorProto.FLOAT
        expected_shape = tuple(input_shape[i] for i in perm)
        
        # Create ONNX model
        attrs = {'perm': perm}
        onnx_model = create_onnx_model(
            op_type="Transpose",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="transpose_high_dim"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        # Compare outputs
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, perm={perm}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, perm={perm}"
        
        logger.info(
            f"✓ Transpose high-dimensional test passed: "
            f"opset={opset_version}, input_shape={input_shape}, perm={perm}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize("input_shape, perm", [
        # Edge cases with size-1 dimensions
        ((1, 1, 1), [2, 1, 0]),  # All ones, reverse
        ((1, 1, 1, 1), [3, 2, 1, 0]),  # All ones, reverse
        ((1, 10, 1), [2, 1, 0]),  # Mixed ones
        ((10, 1, 10), [1, 0, 2]),  # Middle dimension is 1
        ((1, 2, 1, 3, 1), [4, 3, 2, 1, 0]),  # Multiple ones
        # Single element tensors
        ((1,), [0]),  # 1D scalar-like
        ((1, 1), [1, 0]),  # 2D single element
    ])
    def test_transpose_size_one_dimensions(self, opset_version, input_shape, perm):
        """Test Transpose with size-1 dimensions and edge cases."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        dtype = onnx.TensorProto.FLOAT
        expected_shape = tuple(input_shape[i] for i in perm)
        
        # Create ONNX model
        attrs = {'perm': perm}
        onnx_model = create_onnx_model(
            op_type="Transpose",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="transpose_size_one"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        # Compare outputs
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, perm={perm}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, perm={perm}"
        
        logger.info(
            f"✓ Transpose size-one dimensions test passed: "
            f"opset={opset_version}, input_shape={input_shape}, perm={perm}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 13, 21, 23, 24, 25])
    def test_transpose_intermediate_outputs(self, opset_version):
        """Test that intermediate outputs are created correctly for multi-swap permutations."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        input_shape = (2, 3, 4)
        perm = [2, 0, 1]  # Requires 2 swaps: (0,2) then (1,2)
        dtype = onnx.TensorProto.FLOAT
        expected_shape = (4, 2, 3)
        
        # Create ONNX model
        attrs = {'perm': perm}
        onnx_model = create_onnx_model(
            op_type="Transpose",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="transpose_multi_swap"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have 2 Transpose nodes
        transpose_nodes = [n for n in tir_graph.nodes if n.op_type == "Transpose"]
        assert len(transpose_nodes) == 2, \
            f"Expected 2 Transpose nodes, got {len(transpose_nodes)}. " \
            f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        
        # Verify intermediate output exists
        all_outputs = set()
        for node in tir_graph.nodes:
            all_outputs.update(node.outputs)
        
        # Should have at least one intermediate output
        intermediate_outputs = [out for out in all_outputs if 'intermediate' in out]
        assert len(intermediate_outputs) >= 1, \
            f"Expected at least 1 intermediate output, got {intermediate_outputs}"
        
        # Verify intermediate output has correct shape
        # After first swap (0,2): (2,3,4) -> (4,3,2)
        # After second swap (1,2): (4,3,2) -> (4,2,3)
        for node in transpose_nodes:
            if node.outputs[0] in intermediate_outputs:
                # Check intermediate output tensor info
                output_tensors = node.output_tensors
                if node.outputs[0] in output_tensors:
                    intermediate_info = output_tensors[node.outputs[0]]
                    # Shape should be (4, 3, 2) after first swap
                    assert intermediate_info.shape == (4, 3, 2), \
                        f"Intermediate output shape should be (4, 3, 2), got {intermediate_info.shape}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        # Compare outputs
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
        
        logger.info(f"✓ Transpose intermediate outputs test passed: opset={opset_version}")
    
    @pytest.mark.parametrize("opset_version", [1, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize("input_shape, perm", [
        # Permutations requiring maximum swaps
        ((2, 3, 4, 5, 6), [4, 3, 2, 1, 0]),  # 5D reverse (2 swaps)
        ((1, 2, 3, 4, 5, 6, 7), [6, 5, 4, 3, 2, 1, 0]),  # 7D reverse (3 swaps) - fixed: 7D input with 7-element perm
        # Complex rotations
        ((2, 3, 4, 5), [1, 2, 3, 0]),  # Rotate left
        ((2, 3, 4, 5), [3, 0, 1, 2]),  # Rotate right
    ])
    def test_transpose_complex_permutations(self, opset_version, input_shape, perm):
        """Test complex permutations requiring multiple swaps."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        dtype = onnx.TensorProto.FLOAT
        expected_shape = tuple(input_shape[i] for i in perm)
        
        # Create ONNX model
        attrs = {'perm': perm}
        onnx_model = create_onnx_model(
            op_type="Transpose",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="transpose_complex"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure - should have multiple Transpose nodes
        transpose_nodes = [n for n in tir_graph.nodes if n.op_type == "Transpose"]
        assert len(transpose_nodes) >= 1, \
            f"Expected at least 1 Transpose node, got {[n.op_type for n in tir_graph.nodes]}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        # Compare outputs
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, perm={perm}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, perm={perm}"
        
        logger.info(
            f"✓ Transpose complex permutations test passed: "
            f"opset={opset_version}, input_shape={input_shape}, perm={perm}"
        )
    
