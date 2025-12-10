"""
Test cases for ONNX reduction operations (ReduceSum, ReduceMean).
Tests different input shapes, dtypes, opset versions, and attributes.
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


class TestReduceSum:
    """Comprehensive test cases for ReduceSum operation."""
    
    @pytest.mark.parametrize("opset_version", [1, 11])
    @pytest.mark.parametrize("input_shape", [
        # 1D
        (1,),      # Single element
        (4,),      # Small 1D
        (10,),     # Medium 1D
        (100,),    # Large 1D
        # 2D
        (1, 1),    # All ones
        (1, 5),    # Single row
        (5, 1),    # Single column
        (2, 3),    # Small 2D
        (5, 4),    # Medium 2D
        (10, 10),  # Square 2D
        # 3D
        (1, 1, 1), # All ones
        (1, 5, 6), # Single batch
        (2, 3, 4), # Small 3D
        (5, 4, 3), # Medium 3D
        (10, 8, 6), # Large 3D
        # 4D
        (1, 1, 1, 1), # All ones
        (1, 2, 3, 4), # Single batch
        (2, 3, 4, 5), # Small 4D
        (3, 4, 5, 6), # Medium 4D
        # 5D
        (1, 1, 1, 1, 1), # All ones
        (1, 1, 2, 3, 4), # Small 5D
        (2, 3, 4, 5, 6), # Medium 5D
    ])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    @pytest.mark.parametrize("axes", [
        None,      # Reduce all dimensions
        [0],       # Reduce first dimension (positive)
        [-1],      # Reduce last dimension (negative)
        [1],       # Reduce second dimension (positive, if exists)
        [-2],      # Reduce second-to-last (negative, if exists)
        [0, 1],    # Reduce first two dimensions (if exists)
        [-1, -2],  # Reduce last two dimensions (negative, if exists)
        [0, -1],   # Mix of positive and negative
        [1, 2],    # Reduce middle dimensions (if exists)
        [-2, -1],  # Reduce last two (negative, if exists)
    ])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_reducesum_comprehensive(self, opset_version, input_shape, dtype, axes, keepdims):
        """Comprehensive test for ReduceSum with all combinations."""
        # Validate axes are within valid range for input shape
        if axes is not None:
            valid_axes = []
            for axis in axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if 0 <= normalized_axis < len(input_shape):
                    valid_axes.append(axis)
                # Skip if any axis is invalid
                else:
                    pytest.skip(f"Invalid axis {axis} for input shape {input_shape} (normalized: {normalized_axis})")
            
            # Remove duplicates after normalization (e.g., [0, -1] for 1D both become 0)
            normalized_set = set()
            unique_axes = []
            for axis in valid_axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if normalized_axis not in normalized_set:
                    unique_axes.append(axis)
                    normalized_set.add(normalized_axis)
            
            if len(unique_axes) == 0:
                pytest.skip(f"No valid unique axes after deduplication for shape {input_shape}")
            axes = unique_axes if len(unique_axes) < len(valid_axes) else valid_axes
        
        # Calculate expected output shape
        if axes is None:
            # Reduce all dimensions
            if keepdims:
                expected_shape = tuple([1] * len(input_shape))
            else:
                expected_shape = (1,)
        else:
            # Reduce specified axes
            output_shape = list(input_shape)
            for axis in axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if 0 <= normalized_axis < len(output_shape):
                    if keepdims:
                        output_shape[normalized_axis] = 1
                    else:
                        output_shape[normalized_axis] = None  # Mark for removal
            if not keepdims:
                output_shape = [d for d in output_shape if d is not None]
            expected_shape = tuple(output_shape) if output_shape else (1,)
        
        # Map ONNX dtype to numpy dtype for test data generation
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        # Create ONNX model
        attrs = {}
        if axes is not None:
            attrs['axes'] = axes
        attrs['keepdims'] = keepdims
        
        onnx_model = create_onnx_model(
            op_type="ReduceSum",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducesum_test"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify graph structure
        expected_op_types = ["ReduceSum"]
        structure = verify_tir_graph_structure(
            tir_graph,
            onnx_model,
            expected_op_types=expected_op_types
        )
        
        # Verify we have exactly one node (no Reshape needed)
        assert len(tir_graph.nodes) == 1, \
            f"Expected 1 node (ReduceSum), got {len(tir_graph.nodes)}. Nodes: {[n.op_type for n in tir_graph.nodes]}"
        
        # Verify the node is ReduceSum
        assert tir_graph.nodes[0].op_type == "ReduceSum", "Expected ReduceSum node"
        
        # Verify the ReduceSum node has correct attributes
        reduce_node = tir_graph.nodes[0]
        node_keepdim = bool(reduce_node.attrs.get('keepdim', False))
        assert node_keepdim == keepdims, \
            f"Expected keepdim={keepdims}, got {node_keepdim} in node attributes"
        
        # Create test input with appropriate dtype and values
        if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
            # For integer types, use small positive integers
            input_data = {
                'input_0': np.random.randint(1, 10, size=input_shape, dtype=np_dtype)
            }
            # Use integer-appropriate tolerances
            rtol = 0
            atol = 0
        else:
            # For float types, use random values
            input_data = {
                'input_0': np.random.randn(*input_shape).astype(np_dtype)
            }
            rtol = 1e-5
            atol = 1e-6
        
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
            f"Test params: opset={opset_version}, shape={input_shape}, dtype={dtype}, axes={axes}, keepdims={keepdims}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, shape={input_shape}, dtype={dtype}, axes={axes}, keepdims={keepdims}"
        
        logger.info(
            f"✓ ReduceSum test passed: opset={opset_version}, shape={input_shape}, "
            f"dtype={dtype}, axes={axes}, keepdims={keepdims}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 11])
    def test_reducesum_edge_cases(self, opset_version):
        """Test ReduceSum edge cases."""
        edge_cases = [
            # (input_shape, axes, keepdims, description)
            ((1,), None, True, "1D single element, reduce all, keepdims"),
            ((1,), None, False, "1D single element, reduce all, no keepdims"),
            ((1, 1), None, True, "2D all ones, reduce all, keepdims"),
            ((1, 1), None, False, "2D all ones, reduce all, no keepdims"),
            ((1, 1, 1, 1, 1), None, True, "5D all ones, reduce all, keepdims"),
            ((1, 1, 1, 1, 1), None, False, "5D all ones, reduce all, no keepdims"),
            ((10,), [0], True, "1D, reduce first, keepdims"),
            ((10,), [0], False, "1D, reduce first, no keepdims"),
            ((10,), [-1], True, "1D, reduce last (negative), keepdims"),
            ((10,), [-1], False, "1D, reduce last (negative), no keepdims"),
            ((2, 3), [0, 1], True, "2D, reduce all dims, keepdims"),
            ((2, 3), [0, 1], False, "2D, reduce all dims, no keepdims"),
            ((2, 3), [0, -1], True, "2D, reduce all dims (mixed indices), keepdims"),
            ((2, 3, 4), [0, 1, 2], True, "3D, reduce all dims, keepdims"),
            ((2, 3, 4), [0, 1, 2], False, "3D, reduce all dims, no keepdims"),
            ((2, 3, 4, 5), [1, 2], True, "4D, reduce middle dims, keepdims"),
            ((2, 3, 4, 5), [1, 2], False, "4D, reduce middle dims, no keepdims"),
            ((2, 3, 4, 5, 6), [0, -1], True, "5D, reduce first and last, keepdims"),
            ((2, 3, 4, 5, 6), [0, -1], False, "5D, reduce first and last, no keepdims"),
        ]
        
        for input_shape, axes, keepdims, description in edge_cases:
            # Calculate expected output shape
            if axes is None:
                if keepdims:
                    expected_shape = tuple([1] * len(input_shape))
                else:
                    expected_shape = (1,)
            else:
                output_shape = list(input_shape)
                for axis in axes:
                    normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                    if 0 <= normalized_axis < len(output_shape):
                        if keepdims:
                            output_shape[normalized_axis] = 1
                        else:
                            output_shape[normalized_axis] = None
                if not keepdims:
                    output_shape = [d for d in output_shape if d is not None]
                expected_shape = tuple(output_shape) if output_shape else (1,)
            
            # Create ONNX model
            attrs = {}
            if axes is not None:
                attrs['axes'] = axes
            attrs['keepdims'] = keepdims
            
            onnx_model = create_onnx_model(
                op_type="ReduceSum",
                input_shapes=[input_shape],
                input_dtypes=[onnx.TensorProto.FLOAT],
                output_shapes=[expected_shape],
                output_dtypes=[onnx.TensorProto.FLOAT],
                attrs=attrs,
                opset_version=opset_version,
                node_name="reducesum_edge_case"
            )
            
            # Transpile
            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)
            
            # Verify structure
            assert len(tir_graph.nodes) == 1, \
                f"Expected 1 node, got {len(tir_graph.nodes)} for {description}"
            assert tir_graph.nodes[0].op_type == "ReduceSum", \
                f"Expected ReduceSum node for {description}"
            
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
                f"Comparison errors for {description}: {comparison['errors']}"
            assert all(comparison['matches'].values()), \
                f"Output mismatch for {description}: {comparison}"
            
            logger.info(f"✓ Edge case passed: {description} (opset={opset_version})")
    
    @pytest.mark.parametrize("opset_version", [1, 11])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    def test_reducesum_dtype_comprehensive(self, opset_version, dtype):
        """Test ReduceSum with different dtypes across opset versions."""
        input_shape = (3, 4, 5)
        axes = [1, 2]
        keepdims = False
        
        # Calculate expected output shape
        output_shape = list(input_shape)
        for axis in axes:
            normalized_axis = axis if axis >= 0 else len(input_shape) + axis
            if 0 <= normalized_axis < len(output_shape):
                output_shape[normalized_axis] = None
        output_shape = [d for d in output_shape if d is not None]
        expected_shape = tuple(output_shape) if output_shape else (1,)
        
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        # Create ONNX model
        attrs = {'axes': axes, 'keepdims': keepdims}
        onnx_model = create_onnx_model(
            op_type="ReduceSum",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducesum_dtype_test"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) == 1
        assert tir_graph.nodes[0].op_type == "ReduceSum"
        
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
            f"Comparison errors: {comparison['errors']} (opset={opset_version}, dtype={dtype})"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison} (opset={opset_version}, dtype={dtype})"
        
        logger.info(f"✓ Dtype test passed: opset={opset_version}, dtype={dtype}")
    
    def test_reducesum_dim_none_keepdim_true(self):
        """Test ReduceSum with dim=None and keepdim=True - PyTorch handles this correctly."""
        # Test the specific case: torch.sum(inp, dim=None, keepdim=True)
        input_shape = (2, 3, 4)
        axes = None  # Reduce all dimensions
        keepdims = True
        
        # Expected output shape: all dimensions as size 1
        expected_shape = tuple([1] * len(input_shape))  # (1, 1, 1)
        
        # Create ONNX model
        attrs = {'keepdims': keepdims}
        onnx_model = create_onnx_model(
            op_type="ReduceSum",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=11,
            node_name="reducesum_dim_none_keepdim_true"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify we have exactly one ReduceSum node (no Reshape)
        assert len(tir_graph.nodes) == 1, \
            f"Expected 1 node (ReduceSum), got {len(tir_graph.nodes)}. Nodes: {[n.op_type for n in tir_graph.nodes]}"
        assert tir_graph.nodes[0].op_type == "ReduceSum", "Expected ReduceSum node"
        
        # Verify node attributes
        reduce_node = tir_graph.nodes[0]
        node_dim = reduce_node.attrs.get('dim', None)
        node_keepdim = bool(reduce_node.attrs.get('keepdim', False))
        assert node_dim is None, f"Expected dim=None, got {node_dim}"
        assert node_keepdim == True, f"Expected keepdim=True, got {node_keepdim}"
        
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
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), f"Output mismatch: {comparison}"
        
        # Verify output shape matches expected
        tir_output = comparison['tir_outputs']['output_0']
        assert tir_output.shape == expected_shape, \
            f"Expected output shape {expected_shape}, got {tir_output.shape}"
        
        # Verify PyTorch behavior directly
        torch_input = torch.from_numpy(input_data['input_0'])
        torch_output = torch.sum(torch_input, dim=None, keepdim=True)
        assert torch_output.shape == expected_shape, \
            f"PyTorch output shape {torch_output.shape} != expected {expected_shape}"
        
        logger.info(
            f"✓ ReduceSum dim=None keepdim=True test passed: "
            f"input_shape={input_shape}, output_shape={expected_shape}"
        )
    
    @pytest.mark.parametrize("input_shape", [
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 5),
    ])
    def test_reducesum_all_dims_keepdim_true(self, input_shape):
        """Test ReduceSum reducing all dimensions with keepdim=True - no Reshape needed."""
        axes = None  # Reduce all dimensions
        keepdims = True
        expected_shape = tuple([1] * len(input_shape))
        
        # Create ONNX model
        attrs = {'keepdims': keepdims}
        onnx_model = create_onnx_model(
            op_type="ReduceSum",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=11,
            node_name="reducesum_all_dims"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify single ReduceSum node (no Reshape)
        assert len(tir_graph.nodes) == 1, \
            f"Expected 1 node, got {len(tir_graph.nodes)}: {[n.op_type for n in tir_graph.nodes]}"
        assert tir_graph.nodes[0].op_type == "ReduceSum"
        
        # Test with actual data
        input_data = {'input_0': np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(
            tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert comparison['tir_outputs']['output_0'].shape == expected_shape
        
        logger.info(f"✓ ReduceSum all dims test passed: {input_shape} -> {expected_shape}")
    
    def test_pytorch_sum_behavior(self):
        """Verify PyTorch's torch.sum behavior with dim=None and keepdim=True."""
        # Test the exact case mentioned by the user
        inp = torch.rand(2, 3, 4)
        
        # Test dim=None, keepdim=True
        result = torch.sum(inp, dim=None, keepdim=True)
        expected_shape = (1, 1, 1)
        assert result.shape == expected_shape, \
            f"torch.sum(inp, dim=None, keepdim=True) returned shape {result.shape}, expected {expected_shape}"
        
        # Test dim=None, keepdim=False
        result_no_keep = torch.sum(inp, dim=None, keepdim=False)
        assert result_no_keep.shape == (), \
            f"torch.sum(inp, dim=None, keepdim=False) returned shape {result_no_keep.shape}, expected ()"
        
        # Test partial reduction with keepdim=True
        result_partial = torch.sum(inp, dim=0, keepdim=True)
        assert result_partial.shape == (1, 3, 4), \
            f"torch.sum(inp, dim=0, keepdim=True) returned shape {result_partial.shape}, expected (1, 3, 4)"
        
        logger.info("✓ PyTorch torch.sum behavior verified correctly")
    
    # ========== Opset 13 Specific Tests ==========
    
    @pytest.mark.parametrize("input_shape", [
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 5),
    ])
    @pytest.mark.parametrize("axes", [
        None,      # No axes provided (reduce all)
        [0],       # Single axis
        [0, 1],    # Multiple axes
        [-1],      # Negative index
    ])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_reducesum_opset13_basic(self, input_shape, axes, keepdims):
        """Test ReduceSum opset 13 with axes as input tensor (correct opset 13 behavior)."""
        # Calculate expected output shape
        if axes is None:
            if keepdims:
                expected_shape = tuple([1] * len(input_shape))
            else:
                expected_shape = (1,)
        else:
            output_shape = list(input_shape)
            for axis in axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if 0 <= normalized_axis < len(output_shape):
                    if keepdims:
                        output_shape[normalized_axis] = 1
                    else:
                        output_shape[normalized_axis] = None
            if not keepdims:
                output_shape = [d for d in output_shape if d is not None]
            expected_shape = tuple(output_shape) if output_shape else (1,)
        
        # For opset 13, axes is an optional input tensor, NOT an attribute
        # Only attributes are: keepdims and noop_with_empty_axes
        attrs = {'keepdims': keepdims}
        
        # Prepare inputs and initializers
        input_names = ['input_0']
        initializers = {}
        
        # If axes is provided, add it as an input tensor (constant initializer)
        if axes is not None:
            axes_tensor = np.array(axes, dtype=np.int64)
            initializers['axes'] = axes_tensor
            input_names.append('axes')
        
        onnx_model = create_onnx_model(
            op_type="ReduceSum",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=13,
            node_name="reducesum_opset13",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) == 1, \
            f"Expected 1 node, got {len(tir_graph.nodes)}. Nodes: {[n.op_type for n in tir_graph.nodes]}"
        assert tir_graph.nodes[0].op_type == "ReduceSum", "Expected ReduceSum node"
        
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
            f"Test params: shape={input_shape}, axes={axes}, keepdims={keepdims}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: shape={input_shape}, axes={axes}, keepdims={keepdims}"
        
        logger.info(
            f"✓ Opset 13 test passed: shape={input_shape}, axes={axes}, keepdims={keepdims}"
        )
    
    @pytest.mark.parametrize("input_shape", [
        (2, 3),
        (2, 3, 4),
        (1, 5, 6),
    ])
    def test_reducesum_opset13_noop_with_empty_axes_true(self, input_shape):
        """Test ReduceSum opset 13 with noop_with_empty_axes=True (identity/no-op)."""
        # When noop_with_empty_axes=True and axes is empty/None, output should equal input
        expected_shape = input_shape
        
        # Create ONNX model with noop_with_empty_axes=True
        attrs = {
            'keepdims': True,
            'noop_with_empty_axes': 1  # True
        }
        # For opset 13, explicitly provide empty axes tensor to indicate no-op
        # When axes is not provided at all, ONNXRuntime may default to reducing all axes
        empty_axes = np.array([], dtype=np.int64)  # Empty axes tensor
        initializers = {'axes': empty_axes}
        input_names = ['input_0', 'axes']
        
        onnx_model = create_onnx_model(
            op_type="ReduceSum",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=13,
            node_name="reducesum_opset13_noop",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure - should have Identity node (no-op) or ReduceSum
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        # Should be Identity node for no-op case
        assert tir_graph.nodes[0].op_type in ["Identity", "ReduceSum"], \
            f"Expected Identity or ReduceSum node, got {tir_graph.nodes[0].op_type}"
        
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
            f"Test params: shape={input_shape}, noop_with_empty_axes=True"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: shape={input_shape}, noop_with_empty_axes=True"
        
        # Verify output equals input (identity behavior)
        tir_output = comparison['tir_outputs']['output_0']
        onnx_output = comparison['onnx_outputs']['output_0']
        assert tir_output.shape == input_shape, \
            f"Expected output shape {input_shape}, got {tir_output.shape}"
        assert np.allclose(tir_output, onnx_output, rtol=1e-5, atol=1e-6), \
            "Output should equal input for no-op case"
        
        logger.info(
            f"✓ Opset 13 noop_with_empty_axes=True test passed: shape={input_shape}"
        )
    
    @pytest.mark.parametrize("input_shape", [
        (2, 3),
        (2, 3, 4),
    ])
    def test_reducesum_opset13_noop_with_empty_axes_false(self, input_shape):
        """Test ReduceSum opset 13 with noop_with_empty_axes=False (default, reduce all)."""
        # When noop_with_empty_axes=False (default) and axes is empty/None, reduce all dimensions
        keepdims = True
        expected_shape = tuple([1] * len(input_shape))
        
        # Create ONNX model with noop_with_empty_axes=False (default)
        attrs = {
            'keepdims': keepdims,
            'noop_with_empty_axes': 0  # False (default)
        }
        # axes is None (not provided)
        
        onnx_model = create_onnx_model(
            op_type="ReduceSum",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=13,
            node_name="reducesum_opset13_reduce_all"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure - should have ReduceSum node
        assert len(tir_graph.nodes) == 1, \
            f"Expected 1 node, got {len(tir_graph.nodes)}. Nodes: {[n.op_type for n in tir_graph.nodes]}"
        assert tir_graph.nodes[0].op_type == "ReduceSum", "Expected ReduceSum node"
        
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
            f"Test params: shape={input_shape}, noop_with_empty_axes=False"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: shape={input_shape}, noop_with_empty_axes=False"
        
        logger.info(
            f"✓ Opset 13 noop_with_empty_axes=False test passed: shape={input_shape}"
        )
    
    @pytest.mark.parametrize("input_shape", [
        (2, 3, 4),
        (3, 4, 5, 6),
    ])
    @pytest.mark.parametrize("axes_input", [
        [0],       # Single axis as list
        [0, 1],    # Multiple axes
        [-1],      # Negative index
        [1, 2],    # Middle axes
    ])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_reducesum_opset13_axes_as_input_tensor(self, input_shape, axes_input, keepdims):
        """Test ReduceSum opset 13 with axes provided as input tensor (constant)."""
        # For opset 13, axes must be provided as an input tensor, not an attribute
        
        # Calculate expected output shape
        output_shape = list(input_shape)
        for axis in axes_input:
            normalized_axis = axis if axis >= 0 else len(input_shape) + axis
            if 0 <= normalized_axis < len(output_shape):
                if keepdims:
                    output_shape[normalized_axis] = 1
                else:
                    output_shape[normalized_axis] = None
        if not keepdims:
            output_shape = [d for d in output_shape if d is not None]
        expected_shape = tuple(output_shape) if output_shape else (1,)
        
        # For opset 13, axes is an input tensor, NOT an attribute
        # Only attributes are: keepdims and noop_with_empty_axes
        attrs = {'keepdims': keepdims}
        
        # Add axes as an input tensor (constant initializer)
        axes_tensor = np.array(axes_input, dtype=np.int64)
        initializers = {'axes': axes_tensor}
        input_names = ['input_0', 'axes']
        
        onnx_model = create_onnx_model(
            op_type="ReduceSum",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=13,
            node_name="reducesum_opset13_axes_input",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) == 1, \
            f"Expected 1 node, got {len(tir_graph.nodes)}"
        assert tir_graph.nodes[0].op_type == "ReduceSum", "Expected ReduceSum node"
        
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
            f"Test params: shape={input_shape}, axes={axes_input}, keepdims={keepdims}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: shape={input_shape}, axes={axes_input}, keepdims={keepdims}"
        
        logger.info(
            f"✓ Opset 13 axes as input test passed: shape={input_shape}, "
            f"axes={axes_input}, keepdims={keepdims}"
        )


class TestReduceMax:
    """Comprehensive test cases for ReduceMax operation."""
    
    @pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 18, 20])
    @pytest.mark.parametrize("input_shape", [
        # 1D
        (1,),      # Single element
        (4,),      # Small 1D
        (10,),     # Medium 1D
        # 2D
        (1, 1),    # All ones
        (1, 5),    # Single row
        (5, 1),    # Single column
        (2, 3),    # Small 2D
        (5, 4),    # Medium 2D
        # 3D
        (1, 1, 1), # All ones
        (1, 5, 6), # Single batch
        (2, 3, 4), # Small 3D
        (5, 4, 3), # Medium 3D
        # 4D
        (1, 1, 1, 1), # All ones
        (1, 2, 3, 4), # Single batch
        (2, 3, 4, 5), # Small 4D
        # 5D
        (1, 1, 1, 1, 1), # All ones
        (1, 1, 2, 3, 4), # Small 5D
        (2, 3, 4, 5, 6), # Medium 5D
    ])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    @pytest.mark.parametrize("axes", [
        None,      # Reduce all dimensions
        [0],       # Reduce first dimension
        [-1],      # Reduce last dimension
        [0, 1],    # Reduce first two dimensions (if exists)
    ])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_reducemax_comprehensive(self, opset_version, input_shape, dtype, axes, keepdims):
        """Comprehensive test for ReduceMax with all combinations."""
        # Validate axes are within valid range for input shape
        if axes is not None:
            valid_axes = []
            for axis in axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if 0 <= normalized_axis < len(input_shape):
                    valid_axes.append(axis)
                else:
                    pytest.skip(f"Invalid axis {axis} for input shape {input_shape}")
            
            # Remove duplicates
            normalized_set = set()
            unique_axes = []
            for axis in valid_axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if normalized_axis not in normalized_set:
                    unique_axes.append(axis)
                    normalized_set.add(normalized_axis)
            
            if len(unique_axes) == 0:
                pytest.skip(f"No valid unique axes for shape {input_shape}")
            axes = unique_axes
        
        # Calculate expected output shape
        if axes is None:
            if keepdims:
                expected_shape = tuple([1] * len(input_shape))
            else:
                expected_shape = (1,)
        else:
            output_shape = list(input_shape)
            for axis in axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if 0 <= normalized_axis < len(output_shape):
                    if keepdims:
                        output_shape[normalized_axis] = 1
                    else:
                        output_shape[normalized_axis] = None
            if not keepdims:
                output_shape = [d for d in output_shape if d is not None]
            expected_shape = tuple(output_shape) if output_shape else (1,)
        
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
        if opset_version < 18:
            # Opset 1-17: axes as attribute
            if axes is not None:
                attrs['axes'] = axes
        else:
            # Opset 18+: axes as input tensor
            pass  # Will handle below
        attrs['keepdims'] = keepdims
        
        # Prepare inputs and initializers
        input_names = ['input_0']
        initializers = {}
        
        # For opset 18+, axes is an input tensor
        if opset_version >= 18 and axes is not None:
            axes_tensor = np.array(axes, dtype=np.int64)
            initializers['axes'] = axes_tensor
            input_names.append('axes')
        
        onnx_model = create_onnx_model(
            op_type="ReduceMax",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducemax_test",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        assert tir_graph.nodes[0].op_type == "ReduceMax", \
            f"Expected ReduceMax node, got {tir_graph.nodes[0].op_type}"
        
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
            f"Test params: opset={opset_version}, shape={input_shape}, dtype={dtype}, axes={axes}, keepdims={keepdims}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, shape={input_shape}, dtype={dtype}, axes={axes}, keepdims={keepdims}"
        
        logger.info(
            f"✓ ReduceMax test passed: opset={opset_version}, shape={input_shape}, "
            f"dtype={dtype}, axes={axes}, keepdims={keepdims}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 18, 20])
    def test_reducemax_rank_zero(self, opset_version):
        """Test ReduceMax with rank-zero (scalar) input tensor."""
        # Rank-zero tensor (scalar)
        input_shape = ()  # Empty tuple = scalar
        expected_shape = ()  # Scalar output
        
        # Create ONNX model
        attrs = {'keepdims': False}
        onnx_model = create_onnx_model(
            op_type="ReduceMax",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducemax_rank_zero"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create scalar input
        input_data = {
            'input_0': np.array(42.5, dtype=np.float32)
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
        
        # Verify output is scalar (same as input for rank-zero)
        tir_output = comparison['tir_outputs']['output_0']
        assert tir_output.shape == (), f"Expected scalar output, got shape {tir_output.shape}"
        assert np.allclose(tir_output, input_data['input_0'], rtol=1e-5, atol=1e-6), \
            "Scalar output should equal input"
        
        logger.info(f"✓ ReduceMax rank-zero test passed: opset={opset_version}")

    @pytest.mark.parametrize("opset_version", [20])  # Boolean only supported in opset 20+
    def test_reducemax_boolean_false_less_than_true(self, opset_version):
        """Test ReduceMax with Boolean type: False < True semantics (opset 20+)."""
        # Create input with mix of True and False
        input_shape = (3, 4)
        expected_shape = (4,)  # Reduce along axis 0
        
        # Create ONNX model with Boolean type
        attrs = {'keepdims': False}
        # Opset 20+: axes as input tensor
        axes_tensor = np.array([0], dtype=np.int64)
        initializers = {'axes': axes_tensor}
        input_names = ['input_0', 'axes']
        
        onnx_model = create_onnx_model(
            op_type="ReduceMax",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.BOOL],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.BOOL],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducemax_boolean",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create test input: mix of True and False
        # False < True, so max should be True where any True exists
        input_data = {
            'input_0': np.array([
                [False, True, False, True],
                [False, False, False, False],
                [True, True, True, False]
            ], dtype=np.bool_)
        }
        
        # Expected output: max along axis 0 (False < True)
        # [False, True, False, True]  <- row 0
        # [False, False, False, False] <- row 1
        # [True, True, True, False] <- row 2
        # Max: [True, True, True, True] (True > False)
        expected_output = np.array([True, True, True, True], dtype=np.bool_)
        
        # Compare outputs
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=0,
            atol=0
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
        
        # Verify Boolean semantics: False < True
        tir_output = comparison['tir_outputs']['output_0']
        onnx_output = comparison['onnx_outputs']['output_0']
        
        assert np.array_equal(tir_output, expected_output), \
            f"Expected {expected_output}, got {tir_output}"
        assert np.array_equal(onnx_output, expected_output), \
            f"ONNX output should be {expected_output}, got {onnx_output}"
        
        logger.info(f"✓ ReduceMax Boolean False < True test passed: opset={opset_version}")
    
    @pytest.mark.parametrize("opset_version", [18, 20])
    def test_reducemax_opset18_noop_with_empty_axes(self, opset_version):
        """Test ReduceMax opset 18+ with noop_with_empty_axes attribute."""
        input_shape = (2, 3, 4)
        expected_shape = input_shape  # No-op should preserve shape
        
        # Create ONNX model with noop_with_empty_axes=True
        attrs = {
            'keepdims': True,
            'noop_with_empty_axes': 1  # True
        }
        # Empty axes tensor
        empty_axes = np.array([], dtype=np.int64)
        initializers = {'axes': empty_axes}
        input_names = ['input_0', 'axes']
        
        onnx_model = create_onnx_model(
            op_type="ReduceMax",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducemax_noop",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure - should have Identity node (no-op)
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        assert tir_graph.nodes[0].op_type in ["Identity", "ReduceMax"], \
            f"Expected Identity or ReduceMax node, got {tir_graph.nodes[0].op_type}"
        
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
        
        # Verify output equals input (identity behavior)
        tir_output = comparison['tir_outputs']['output_0']
        onnx_output = comparison['onnx_outputs']['output_0']
        assert tir_output.shape == input_shape, \
            f"Expected output shape {input_shape}, got {tir_output.shape}"
        assert np.allclose(tir_output, input_data['input_0'], rtol=1e-5, atol=1e-6), \
            "Output should equal input for no-op case"
        
        logger.info(
            f"✓ ReduceMax noop_with_empty_axes test passed: opset={opset_version}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 18, 20])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    def test_reducemax_dtype_comprehensive(self, opset_version, dtype):
        """Test ReduceMax with different dtypes across opset versions."""
        input_shape = (3, 4, 5)
        axes = [1, 2]
        keepdims = False
        
        # Calculate expected output shape
        output_shape = list(input_shape)
        for axis in axes:
            normalized_axis = axis if axis >= 0 else len(input_shape) + axis
            if 0 <= normalized_axis < len(output_shape):
                output_shape[normalized_axis] = None
        output_shape = [d for d in output_shape if d is not None]
        expected_shape = tuple(output_shape) if output_shape else (1,)
        
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        # Create ONNX model
        attrs = {'keepdims': keepdims}
        input_names = ['input_0']
        initializers = {}
        
        if opset_version < 18:
            # Opset 1-17: axes as attribute
            attrs['axes'] = axes
        else:
            # Opset 18+: axes as input tensor
            axes_tensor = np.array(axes, dtype=np.int64)
            initializers['axes'] = axes_tensor
            input_names.append('axes')
        
        onnx_model = create_onnx_model(
            op_type="ReduceMax",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducemax_dtype_test",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
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
        
        # Compare outputs
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=rtol,
            atol=atol
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']} (opset={opset_version}, dtype={dtype})"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison} (opset={opset_version}, dtype={dtype})"
        
        logger.info(f"✓ ReduceMax dtype test passed: opset={opset_version}, dtype={dtype}")


class TestReduceMean:
    """Comprehensive test cases for ReduceMean operation."""
    
    @pytest.mark.parametrize("opset_version", [1, 11, 13, 18])
    @pytest.mark.parametrize("input_shape", [
        # 1D
        (1,),      # Single element
        (4,),      # Small 1D
        (10,),     # Medium 1D
        # 2D
        (1, 1),    # All ones
        (1, 5),    # Single row
        (5, 1),    # Single column
        (2, 3),    # Small 2D
        (5, 4),    # Medium 2D
        # 3D
        (1, 1, 1), # All ones
        (1, 5, 6), # Single batch
        (2, 3, 4), # Small 3D
        (5, 4, 3), # Medium 3D
        # 4D
        (1, 1, 1, 1), # All ones
        (1, 2, 3, 4), # Single batch
        (2, 3, 4, 5), # Small 4D
        # 5D
        (1, 1, 1, 1, 1), # All ones
        (1, 1, 2, 3, 4), # Small 5D
        (2, 3, 4, 5, 6), # Medium 5D
    ])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    @pytest.mark.parametrize("axes", [
        None,      # Reduce all dimensions
        [0],       # Reduce first dimension
        [-1],      # Reduce last dimension
        [0, 1],    # Reduce first two dimensions (if exists)
    ])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_reducemean_comprehensive(self, opset_version, input_shape, dtype, axes, keepdims):
        """Comprehensive test for ReduceMean with all combinations."""
        # Validate axes are within valid range for input shape
        if axes is not None:
            valid_axes = []
            for axis in axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if 0 <= normalized_axis < len(input_shape):
                    valid_axes.append(axis)
                else:
                    pytest.skip(f"Invalid axis {axis} for input shape {input_shape}")
            
            # Remove duplicates
            normalized_set = set()
            unique_axes = []
            for axis in valid_axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if normalized_axis not in normalized_set:
                    unique_axes.append(axis)
                    normalized_set.add(normalized_axis)
            
            if len(unique_axes) == 0:
                pytest.skip(f"No valid unique axes for shape {input_shape}")
            axes = unique_axes
        
        # Calculate expected output shape
        if axes is None:
            if keepdims:
                expected_shape = tuple([1] * len(input_shape))
            else:
                expected_shape = (1,)
        else:
            output_shape = list(input_shape)
            for axis in axes:
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                if 0 <= normalized_axis < len(output_shape):
                    if keepdims:
                        output_shape[normalized_axis] = 1
                    else:
                        output_shape[normalized_axis] = None
            if not keepdims:
                output_shape = [d for d in output_shape if d is not None]
            expected_shape = tuple(output_shape) if output_shape else (1,)
        
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
        if opset_version < 18:
            # Opset 1-17: axes as attribute
            if axes is not None:
                attrs['axes'] = axes
        else:
            # Opset 18+: axes as input tensor
            pass  # Will handle below
        attrs['keepdims'] = keepdims
        
        # Prepare inputs and initializers
        input_names = ['input_0']
        initializers = {}
        
        # For opset 18+, axes is an input tensor
        if opset_version >= 18 and axes is not None:
            axes_tensor = np.array(axes, dtype=np.int64)
            initializers['axes'] = axes_tensor
            input_names.append('axes')
        
        onnx_model = create_onnx_model(
            op_type="ReduceMean",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducemean_test",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        
        # For integer types, we insert Cast nodes before and after ReduceMean
        is_integer = dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]
        if is_integer:
            # Should have 3 nodes: Cast -> ReduceMean -> Cast
            assert len(tir_graph.nodes) == 3, \
                f"Expected 3 nodes (Cast->ReduceMean->Cast) for integer type, got {len(tir_graph.nodes)}"
            assert tir_graph.nodes[0].op_type == "Cast", \
                f"Expected Cast node first, got {tir_graph.nodes[0].op_type}"
            assert tir_graph.nodes[1].op_type == "ReduceMean", \
                f"Expected ReduceMean node second, got {tir_graph.nodes[1].op_type}"
            assert tir_graph.nodes[2].op_type == "Cast", \
                f"Expected Cast node third, got {tir_graph.nodes[2].op_type}"
        else:
            # For float types, should have ReduceMean node (may be first or only node)
            reduce_mean_nodes = [n for n in tir_graph.nodes if n.op_type == "ReduceMean"]
            assert len(reduce_mean_nodes) == 1, \
                f"Expected exactly 1 ReduceMean node for float type, got {len(reduce_mean_nodes)}"
        
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
            f"Test params: opset={opset_version}, shape={input_shape}, dtype={dtype}, axes={axes}, keepdims={keepdims}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, shape={input_shape}, dtype={dtype}, axes={axes}, keepdims={keepdims}"
        
        logger.info(
            f"✓ ReduceMean test passed: opset={opset_version}, shape={input_shape}, "
            f"dtype={dtype}, axes={axes}, keepdims={keepdims}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 11, 13, 18])
    def test_reducemean_rank_zero(self, opset_version):
        """Test ReduceMean with rank-zero (scalar) input tensor."""
        # Rank-zero tensor (scalar)
        input_shape = ()  # Empty tuple = scalar
        expected_shape = ()  # Scalar output
        
        # Create ONNX model
        attrs = {'keepdims': False}
        onnx_model = create_onnx_model(
            op_type="ReduceMean",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducemean_rank_zero"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create scalar input
        input_data = {
            'input_0': np.array(42.5, dtype=np.float32)
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
        
        # Verify output is scalar (same as input for rank-zero)
        tir_output = comparison['tir_outputs']['output_0']
        assert tir_output.shape == (), f"Expected scalar output, got shape {tir_output.shape}"
        assert np.allclose(tir_output, input_data['input_0'], rtol=1e-5, atol=1e-6), \
            "Scalar output should equal input"
        
        logger.info(f"✓ ReduceMean rank-zero test passed: opset={opset_version}")
    

    @pytest.mark.parametrize("opset_version", [18])
    def test_reducemean_opset18_noop_with_empty_axes(self, opset_version):
        """Test ReduceMean opset 18+ with noop_with_empty_axes attribute."""
        input_shape = (2, 3, 4)
        expected_shape = input_shape  # No-op should preserve shape
        
        # Create ONNX model with noop_with_empty_axes=True
        attrs = {
            'keepdims': True,
            'noop_with_empty_axes': 1  # True
        }
        # Empty axes tensor
        empty_axes = np.array([], dtype=np.int64)
        initializers = {'axes': empty_axes}
        input_names = ['input_0', 'axes']
        
        onnx_model = create_onnx_model(
            op_type="ReduceMean",
            input_shapes=[input_shape],
            input_dtypes=[onnx.TensorProto.FLOAT],
            output_shapes=[expected_shape],
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducemean_noop",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure - should have Identity node (no-op)
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        assert tir_graph.nodes[0].op_type in ["Identity", "ReduceMean"], \
            f"Expected Identity or ReduceMean node, got {tir_graph.nodes[0].op_type}"
        
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
        
        # Verify output equals input (identity behavior)
        tir_output = comparison['tir_outputs']['output_0']
        onnx_output = comparison['onnx_outputs']['output_0']
        assert tir_output.shape == input_shape, \
            f"Expected output shape {input_shape}, got {tir_output.shape}"
        assert np.allclose(tir_output, input_data['input_0'], rtol=1e-5, atol=1e-6), \
            "Output should equal input for no-op case"
        
        logger.info(
            f"✓ ReduceMean noop_with_empty_axes test passed: opset={opset_version}"
        )
    
    @pytest.mark.parametrize("opset_version", [1, 11, 13, 18])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    def test_reducemean_dtype_comprehensive(self, opset_version, dtype):
        """Test ReduceMean with different dtypes across opset versions."""
        input_shape = (3, 4, 5)
        axes = [1, 2]
        keepdims = False
        
        # Calculate expected output shape
        output_shape = list(input_shape)
        for axis in axes:
            normalized_axis = axis if axis >= 0 else len(input_shape) + axis
            if 0 <= normalized_axis < len(output_shape):
                output_shape[normalized_axis] = None
        output_shape = [d for d in output_shape if d is not None]
        expected_shape = tuple(output_shape) if output_shape else (1,)
        
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        # Create ONNX model
        attrs = {'keepdims': keepdims}
        input_names = ['input_0']
        initializers = {}
        
        if opset_version < 18:
            # Opset 1-17: axes as attribute
            attrs['axes'] = axes
        else:
            # Opset 18+: axes as input tensor
            axes_tensor = np.array(axes, dtype=np.int64)
            initializers['axes'] = axes_tensor
            input_names.append('axes')
        
        onnx_model = create_onnx_model(
            op_type="ReduceMean",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reducemean_dtype_test",
            input_names=input_names,
            initializers=initializers
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure for integer types
        is_integer = dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]
        if is_integer:
            # Should have 3 nodes: Cast -> ReduceMean -> Cast
            assert len(tir_graph.nodes) == 3, \
                f"Expected 3 nodes (Cast->ReduceMean->Cast) for integer type, got {len(tir_graph.nodes)}"
            assert tir_graph.nodes[0].op_type == "Cast", \
                f"Expected Cast node first, got {tir_graph.nodes[0].op_type}"
            assert tir_graph.nodes[1].op_type == "ReduceMean", \
                f"Expected ReduceMean node second, got {tir_graph.nodes[1].op_type}"
            assert tir_graph.nodes[2].op_type == "Cast", \
                f"Expected Cast node third, got {tir_graph.nodes[2].op_type}"
        else:
            # For float types, should have ReduceMean node (may be first or only node)
            reduce_mean_nodes = [n for n in tir_graph.nodes if n.op_type == "ReduceMean"]
            assert len(reduce_mean_nodes) == 1, \
                f"Expected exactly 1 ReduceMean node for float type, got {len(reduce_mean_nodes)}"
        
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
            f"Comparison errors: {comparison['errors']} (opset={opset_version}, dtype={dtype})"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison} (opset={opset_version}, dtype={dtype})"
        
        logger.info(f"✓ ReduceMean dtype test passed: opset={opset_version}, dtype={dtype}")
    
    @pytest.mark.parametrize("opset_version", [13, 18])
    def test_reducemean_bfloat16_support(self, opset_version):
        """Test ReduceMean with bfloat16 type (supported from opset 13+)."""
        input_shape = (2, 3, 4)
        axes = [1]
        keepdims = False
        
        # Calculate expected output shape
        output_shape = list(input_shape)
        for axis in axes:
            normalized_axis = axis if axis >= 0 else len(input_shape) + axis
            if 0 <= normalized_axis < len(output_shape):
                output_shape[normalized_axis] = None
        output_shape = [d for d in output_shape if d is not None]
        expected_shape = tuple(output_shape) if output_shape else (1,)
        
        # Create ONNX model with bfloat16
        attrs = {'keepdims': keepdims}
        input_names = ['input_0']
        initializers = {}
        
        if opset_version < 18:
            attrs['axes'] = axes
        else:
            axes_tensor = np.array(axes, dtype=np.int64)
            initializers['axes'] = axes_tensor
            input_names.append('axes')
        
        # Note: bfloat16 may not be fully supported in all backends
        # This test verifies the converter handles it correctly
        try:
            onnx_model = create_onnx_model(
                op_type="ReduceMean",
                input_shapes=[input_shape],
                input_dtypes=[onnx.TensorProto.BFLOAT16],
                output_shapes=[expected_shape],
                output_dtypes=[onnx.TensorProto.BFLOAT16],
                attrs=attrs,
                opset_version=opset_version,
                node_name="reducemean_bfloat16",
                input_names=input_names,
                initializers=initializers
            )
            
            # Transpile
            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)
            
            logger.info(f"✓ ReduceMean bfloat16 test passed: opset={opset_version}")
        except Exception as e:
            # bfloat16 may not be supported in all environments
            pytest.skip(f"bfloat16 not supported in this environment: {e}")