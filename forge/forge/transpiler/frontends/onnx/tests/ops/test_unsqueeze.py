"""
Test cases for ONNX Unsqueeze operation.
Tests different input shapes, dtypes, opset versions, axes, and edge cases.
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


class TestUnsqueeze:
    """Comprehensive test cases for Unsqueeze operation."""
    
    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize("input_shape, axes, expected_shape", [
        # Single axis unsqueeze
        ((3, 4), [0], (1, 3, 4)),              # Insert at first position
        ((3, 4), [1], (3, 1, 4)),              # Insert at middle position
        ((3, 4), [2], (3, 4, 1)),              # Insert at last position
        ((3, 4, 5), [0], (1, 3, 4, 5)),       # Insert at first (4D output)
        ((3, 4, 5), [3], (3, 4, 5, 1)),       # Insert at last (4D output)
        # Multiple axes unsqueeze
        ((3, 4), [0, 2], (1, 3, 1, 4)),        # Insert at first and last
        ((3, 4), [0, 3], (1, 3, 4, 1)),       # Insert at first and after last
        ((3, 4, 5), [0, 4], (1, 3, 4, 5, 1)), # Insert at first and last (5D output)
        ((3, 4), [1, 3], (3, 1, 4, 1)),       # Insert at middle positions
        ((3, 4, 5), [0, 2, 5], (1, 3, 1, 4, 5, 1)), # Insert at multiple positions
        # Negative indices (v11+)
        ((3, 4), [-1], (3, 4, 1)),            # Negative last position
        ((3, 4), [-2], (3, 1, 4)),            # Negative middle position
        ((3, 4), [-3], (1, 3, 4)),            # Negative first position
        ((3, 4, 5), [-1], (3, 4, 5, 1)),      # Negative last (4D output)
        ((3, 4), [-4, -1], (1, 3, 4, 1)),     # Negative first and last
        ((3, 4, 5), [-5, -1], (1, 3, 4, 5, 1)), # Negative indices (5D output, -5 is first, -1 is last)
        # Edge cases
        ((3,), [0], (1, 3)),                  # 1D to 2D (insert at start)
        ((3,), [1], (3, 1)),                  # 1D to 2D (insert at end)
        ((3,), [0, 1], (1, 3, 1)),            # 1D to 3D
        ((3, 4), [0, 1, 2, 3], (1, 3, 1, 4, 1, 1)), # Insert at all positions
        ((1,), [0, 1], (1, 1, 1)),           # Scalar-like to 3D
        ((2, 3), [0], (1, 2, 3)),             # 2D to 3D
        ((2, 3, 4), [0, 4], (1, 2, 3, 4, 1)), # 3D to 5D
        # Unsorted axes (order should not matter)
        ((3, 4), [2, 0], (1, 3, 1, 4)),       # Unsorted: should be same as [0, 2]
        ((3, 4, 5), [4, 0, 2], (1, 3, 1, 4, 5, 1)), # Unsorted multiple
    ])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    def test_unsqueeze_basic(self, opset_version, input_shape, axes, expected_shape, dtype):
        """Test basic Unsqueeze operations across opset versions."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")
        
        # Skip negative indices for opset 1 (not supported)
        if opset_version == 1:
            axes_list = axes if isinstance(axes, list) else [axes]
            if any(ax < 0 for ax in axes_list):
                pytest.skip(f"Opset 1 does not support negative indices")
        
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        # Create ONNX model
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        # Handle axes based on opset version
        if opset_version >= 13:
            # v13+: axes is required input tensor
            axes_array = np.array(axes, dtype=np.int64)
            input_names.append("axes")
            initializers["axes"] = axes_array
            # Create model with 2 inputs (data + axes)
            onnx_model = create_onnx_model(
                op_type="Unsqueeze",
                input_shapes=[input_shape, axes_array.shape],
                input_dtypes=[dtype, onnx.TensorProto.INT64],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="unsqueeze_test",
                input_names=input_names,
                initializers=initializers
            )
        else:
            # v1-v12: axes is required attribute
            attrs['axes'] = axes
            onnx_model = create_onnx_model(
                op_type="Unsqueeze",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="unsqueeze_test"
            )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        
        # For multiple axes, we should have multiple UnsqueezeNode instances
        unsqueeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Unsqueeze"]
        expected_node_count = len(axes) if isinstance(axes, list) else 1
        assert len(unsqueeze_nodes) == expected_node_count, \
            f"Expected {expected_node_count} UnsqueezeNode(s), got {len(unsqueeze_nodes)}. " \
            f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        
        # Verify each UnsqueezeNode has single int dim
        for node in unsqueeze_nodes:
            assert 'dim' in node.attrs, f"UnsqueezeNode {node.name} missing 'dim' attribute"
            assert isinstance(node.attrs['dim'], int), \
                f"UnsqueezeNode {node.name} 'dim' must be int, got {type(node.attrs['dim'])}"
        
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
        
        # Compare with ONNX runtime
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
            f"axes={axes}, dtype={dtype}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, " \
            f"axes={axes}, dtype={dtype}"
    
    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    def test_unsqueeze_negative_indices(self, opset_version):
        """Test Unsqueeze with negative indices (v11+)."""
        if opset_version < 11:
            pytest.skip(f"Negative indices not supported in opset {opset_version}")
        
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        input_shape = (3, 4, 5)
        dtype = onnx.TensorProto.FLOAT
        
        # Test cases with negative indices
        # For input shape (3, 4, 5) with axes=[-1]:
        #   Output rank = 3 + 1 = 4
        #   -1 normalized = -1 + 4 = 3
        #   Output shape = (3, 4, 5, 1)
        test_cases = [
            ([-1], (3, 4, 5, 1)),       # Insert at last position
            ([-4], (1, 3, 4, 5)),       # Insert at first position
            ([-4, -1], (1, 3, 4, 5, 1)), # Insert at first and last
            ([-2], (3, 4, 1, 5)),       # Insert at second-to-last
        ]
        
        for axes, expected_shape in test_cases:
            attrs = {}
            initializers = {}
            input_names = ["input_0"]
            
            if opset_version >= 13:
                axes_array = np.array(axes, dtype=np.int64)
                input_names.append("axes")
                initializers["axes"] = axes_array
                onnx_model = create_onnx_model(
                    op_type="Unsqueeze",
                    input_shapes=[input_shape, axes_array.shape],
                    input_dtypes=[dtype, onnx.TensorProto.INT64],
                    output_shapes=[expected_shape],
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="unsqueeze_negative",
                    input_names=input_names,
                    initializers=initializers
                )
            else:
                attrs['axes'] = axes
                onnx_model = create_onnx_model(
                    op_type="Unsqueeze",
                    input_shapes=[input_shape],
                    input_dtypes=[dtype],
                    output_shapes=[expected_shape],
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="unsqueeze_negative"
                )
            
            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)
            
            # Verify multiple nodes for multiple axes
            unsqueeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Unsqueeze"]
            assert len(unsqueeze_nodes) == len(axes), \
                f"Expected {len(axes)} UnsqueezeNode(s) for axes={axes}, got {len(unsqueeze_nodes)}"
            
            # Create test input
            input_data = {
                'input_0': np.random.randn(*input_shape).astype(np.float32)
            }
            
            # Compare with ONNX runtime
            comparison = compare_tir_with_onnx(
                tir_graph,
                onnx_model,
                input_data,
                rtol=1e-5,
                atol=1e-6
            )
            
            assert len(comparison['errors']) == 0, \
                f"Comparison errors: {comparison['errors']}\n" \
                f"Test params: opset={opset_version}, axes={axes}"
            assert all(comparison['matches'].values()), \
                f"Output mismatch: {comparison}\n" \
                f"Test params: opset={opset_version}, axes={axes}"
    
    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    def test_unsqueeze_intermediate_outputs(self, opset_version):
        """Test that intermediate outputs are created correctly for multiple axes."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        input_shape = (3, 4)
        axes = [0, 2]  # Should create 2 UnsqueezeNode instances
        expected_shape = (1, 3, 1, 4)
        dtype = onnx.TensorProto.FLOAT
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        if opset_version >= 13:
            axes_array = np.array(axes, dtype=np.int64)
            input_names.append("axes")
            initializers["axes"] = axes_array
            onnx_model = create_onnx_model(
                op_type="Unsqueeze",
                input_shapes=[input_shape, axes_array.shape],
                input_dtypes=[dtype, onnx.TensorProto.INT64],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="unsqueeze_multi",
                input_names=input_names,
                initializers=initializers
            )
        else:
            attrs['axes'] = axes
            onnx_model = create_onnx_model(
                op_type="Unsqueeze",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="unsqueeze_multi"
            )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have 2 Unsqueeze nodes
        unsqueeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Unsqueeze"]
        assert len(unsqueeze_nodes) == 2, \
            f"Expected 2 UnsqueezeNode instances, got {len(unsqueeze_nodes)}. " \
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
        # After first unsqueeze at axis 0: (3, 4) -> (1, 3, 4)
        # After second unsqueeze at axis 2: (1, 3, 4) -> (1, 3, 1, 4)
        for node in tir_graph.nodes:
            if 'intermediate' in str(node.outputs):
                # Find the intermediate tensor info
                for output_name in node.outputs:
                    if 'intermediate' in output_name:
                        # Should have shape (1, 3, 4) after first unsqueeze
                        assert node.output_tensors[output_name].shape == (1, 3, 4), \
                            f"Intermediate output {output_name} has incorrect shape: " \
                            f"{node.output_tensors[output_name].shape}, expected (1, 3, 4)"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}"
    
    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    def test_unsqueeze_single_axis(self, opset_version):
        """Test Unsqueeze with single axis (should create single node)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        input_shape = (3, 4)
        axes = [1]  # Single axis
        expected_shape = (3, 1, 4)
        dtype = onnx.TensorProto.FLOAT
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        if opset_version >= 13:
            axes_array = np.array(axes, dtype=np.int64)
            input_names.append("axes")
            initializers["axes"] = axes_array
            onnx_model = create_onnx_model(
                op_type="Unsqueeze",
                input_shapes=[input_shape, axes_array.shape],
                input_dtypes=[dtype, onnx.TensorProto.INT64],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="unsqueeze_single",
                input_names=input_names,
                initializers=initializers
            )
        else:
            attrs['axes'] = axes
            onnx_model = create_onnx_model(
                op_type="Unsqueeze",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="unsqueeze_single"
            )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have exactly 1 Unsqueeze node (no intermediate outputs needed)
        unsqueeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Unsqueeze"]
        assert len(unsqueeze_nodes) == 1, \
            f"Expected 1 UnsqueezeNode for single axis, got {len(unsqueeze_nodes)}"
        
        # Should not have intermediate outputs
        all_outputs = set()
        for node in tir_graph.nodes:
            all_outputs.update(node.outputs)
        
        intermediate_outputs = [out for out in all_outputs if 'intermediate' in out]
        assert len(intermediate_outputs) == 0, \
            f"Expected no intermediate outputs for single axis, got {intermediate_outputs}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}"
    
    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    def test_unsqueeze_high_dimensional(self, opset_version):
        """Test Unsqueeze with high-dimensional tensors."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        test_cases = [
            ((2, 3, 4, 5), [0, 5], (1, 2, 3, 4, 5, 1)),  # 4D to 6D
            ((1, 2, 3), [0, 3], (1, 1, 2, 3, 1)),       # 3D to 5D
            ((2, 3, 4, 5, 6), [0, 6], (1, 2, 3, 4, 5, 6, 1)),  # 5D to 7D
        ]
        
        dtype = onnx.TensorProto.FLOAT
        
        for input_shape, axes, expected_shape in test_cases:
            attrs = {}
            initializers = {}
            input_names = ["input_0"]
            
            if opset_version >= 13:
                axes_array = np.array(axes, dtype=np.int64)
                input_names.append("axes")
                initializers["axes"] = axes_array
                onnx_model = create_onnx_model(
                    op_type="Unsqueeze",
                    input_shapes=[input_shape, axes_array.shape],
                    input_dtypes=[dtype, onnx.TensorProto.INT64],
                    output_shapes=[expected_shape],
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="unsqueeze_high_dim",
                    input_names=input_names,
                    initializers=initializers
                )
            else:
                attrs['axes'] = axes
                onnx_model = create_onnx_model(
                    op_type="Unsqueeze",
                    input_shapes=[input_shape],
                    input_dtypes=[dtype],
                    output_shapes=[expected_shape],
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="unsqueeze_high_dim"
                )
            
            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)
            
            # Create test input
            input_data = {
                'input_0': np.random.randn(*input_shape).astype(np.float32)
            }
            
            # Compare with ONNX runtime
            comparison = compare_tir_with_onnx(
                tir_graph,
                onnx_model,
                input_data,
                rtol=1e-5,
                atol=1e-6
            )
            
            assert len(comparison['errors']) == 0, \
                f"Comparison errors: {comparison['errors']}\n" \
                f"Test params: opset={opset_version}, input_shape={input_shape}"
            assert all(comparison['matches'].values()), \
                f"Output mismatch: {comparison}\n" \
                f"Test params: opset={opset_version}, input_shape={input_shape}"
    
    def test_unsqueeze_missing_axes_error(self):
        """Test that missing axes raises an error."""
        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT
        
        # Create model without axes (should fail)
        onnx_model = create_onnx_model(
            op_type="Unsqueeze",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[(3, 4)],  # Dummy output shape
            output_dtypes=[dtype],
            attrs={},  # No axes attribute
            opset_version=1,
            node_name="unsqueeze_no_axes"
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        
        # Should raise ValueError because axes is required
        with pytest.raises((ValueError, Exception)) as exc_info:
            tir_graph = transpiler.transpile(onnx_model)
        
        # Verify error message mentions axes
        error_msg = str(exc_info.value).lower()
        assert 'axes' in error_msg or 'required' in error_msg, \
            f"Error message should mention 'axes' or 'required', got: {error_msg}"
    
    def test_unsqueeze_duplicate_axes_error(self):
        """Test that duplicate axes raises an error."""
        input_shape = (3, 4)
        axes = [0, 0]  # Duplicate axes
        dtype = onnx.TensorProto.FLOAT
        
        attrs = {'axes': axes}
        onnx_model = create_onnx_model(
            op_type="Unsqueeze",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[(1, 3, 1, 4)],  # Dummy output shape
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=1,
            node_name="unsqueeze_duplicate"
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        
        # Transpiler catches exceptions and logs them, so check for empty graph
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have no nodes because conversion failed
        assert len(tir_graph.nodes) == 0, \
            f"Expected no nodes when duplicate axes are provided, got {len(tir_graph.nodes)} nodes"
    
    def test_unsqueeze_out_of_range_error(self):
        """Test that out-of-range axes raises an error."""
        input_shape = (3, 4)
        # Output rank would be 2 + 1 = 3, so valid axes are [0, 1, 2] or [-3, -2, -1]
        # Using axis 5 which is out of range
        axes = [5]
        dtype = onnx.TensorProto.FLOAT
        
        attrs = {'axes': axes}
        onnx_model = create_onnx_model(
            op_type="Unsqueeze",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[(3, 4, 1)],  # Dummy output shape
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=1,
            node_name="unsqueeze_out_of_range"
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        
        # Transpiler catches exceptions and logs them, so check for empty graph
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have no nodes because conversion failed
        assert len(tir_graph.nodes) == 0, \
            f"Expected no nodes when out-of-range axis is provided, got {len(tir_graph.nodes)} nodes"
    
    @pytest.mark.parametrize("opset_version", [13, 21, 23])
    def test_unsqueeze_v13_axes_as_input(self, opset_version):
        """Test that v13+ correctly handles axes as input tensor."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        input_shape = (3, 4)
        axes = [0, 2]
        expected_shape = (1, 3, 1, 4)
        dtype = onnx.TensorProto.FLOAT
        
        axes_array = np.array(axes, dtype=np.int64)
        initializers = {"axes": axes_array}
        input_names = ["input_0", "axes"]
        
        onnx_model = create_onnx_model(
            op_type="Unsqueeze",
            input_shapes=[input_shape, axes_array.shape],
            input_dtypes=[dtype, onnx.TensorProto.INT64],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="unsqueeze_v13",
            input_names=input_names,
            initializers=initializers
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have 2 Unsqueeze nodes (one per axis)
        unsqueeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Unsqueeze"]
        assert len(unsqueeze_nodes) == 2, \
            f"Expected 2 UnsqueezeNode instances for axes={axes}, got {len(unsqueeze_nodes)}"
        
        # Verify only data input is used (axes is embedded, not passed as input)
        # First node should use original input, second node uses intermediate output
        assert len(unsqueeze_nodes[0].inputs) == 1, \
            f"First UnsqueezeNode should have only data input, got inputs: {unsqueeze_nodes[0].inputs}"
        assert unsqueeze_nodes[0].inputs[0] == "input_0", \
            f"First UnsqueezeNode input should be 'input_0', got: {unsqueeze_nodes[0].inputs[0]}"
        
        # Second node should use intermediate output
        assert len(unsqueeze_nodes[1].inputs) == 1, \
            f"Second UnsqueezeNode should have only one input, got inputs: {unsqueeze_nodes[1].inputs}"
        assert 'intermediate' in unsqueeze_nodes[1].inputs[0], \
            f"Second UnsqueezeNode should use intermediate output, got: {unsqueeze_nodes[1].inputs[0]}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}"
    
    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    def test_unsqueeze_different_dtypes(self, opset_version, dtype):
        """Test Unsqueeze with different data types."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        if opset_version == 1 and dtype not in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")
        
        input_shape = (2, 3)
        axes = [1]
        expected_shape = (2, 1, 3)
        
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        if opset_version >= 13:
            axes_array = np.array(axes, dtype=np.int64)
            input_names.append("axes")
            initializers["axes"] = axes_array
            onnx_model = create_onnx_model(
                op_type="Unsqueeze",
                input_shapes=[input_shape, axes_array.shape],
                input_dtypes=[dtype, onnx.TensorProto.INT64],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="unsqueeze_dtype",
                input_names=input_names,
                initializers=initializers
            )
        else:
            attrs['axes'] = axes
            onnx_model = create_onnx_model(
                op_type="Unsqueeze",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="unsqueeze_dtype"
            )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
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
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=rtol,
            atol=atol
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}, dtype={dtype}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, dtype={dtype}"

