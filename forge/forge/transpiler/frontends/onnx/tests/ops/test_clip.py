"""
Test cases for ONNX Clip operation.
Tests different input shapes, dtypes, opset versions, min/max values, and edge cases.
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


class TestClip:
    """Comprehensive test cases for Clip operation."""
    
    @pytest.mark.parametrize("opset_version", [1, 6, 11, 12, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize("input_shape, min_val, max_val, dtype", [
        # Basic clipping
        ((3, 4), 0.0, 1.0, onnx.TensorProto.FLOAT),
        ((2, 3, 4), -1.0, 1.0, onnx.TensorProto.FLOAT),
        ((5,), 0.0, 5.0, onnx.TensorProto.FLOAT),
        # Only min
        ((3, 4), 0.0, None, onnx.TensorProto.FLOAT),
        ((2, 3), -1.0, None, onnx.TensorProto.FLOAT),
        # Only max
        ((3, 4), None, 1.0, onnx.TensorProto.FLOAT),
        ((2, 3), None, 5.0, onnx.TensorProto.FLOAT),
        # No clipping (None, None)
        ((3, 4), None, None, onnx.TensorProto.FLOAT),
        # Integer types (v12+)
        ((3, 4), 0, 255, onnx.TensorProto.INT32),
        ((2, 3), -128, 127, onnx.TensorProto.INT32),
        ((3, 4), 0, 100, onnx.TensorProto.INT64),
        # Double precision
        ((3, 4), 0.0, 1.0, onnx.TensorProto.DOUBLE),
        # Edge cases
        ((1,), 0.0, 1.0, onnx.TensorProto.FLOAT),
        ((1, 1), -1.0, 1.0, onnx.TensorProto.FLOAT),
        ((10, 10, 10), 0.0, 1.0, onnx.TensorProto.FLOAT),
    ])
    def test_clip_basic(self, opset_version, input_shape, min_val, max_val, dtype):
        """Test basic Clip operations across opset versions."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip opset 1 - ONNXRuntime doesn't support Clip(1)
        if opset_version == 1:
            pytest.skip(f"Opset 1 not supported by ONNXRuntime (Clip(1) not implemented)")
        
        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")
        
        # Skip integer types for opset < 12
        if opset_version < 12 and dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64, 
                                            onnx.TensorProto.INT8, onnx.TensorProto.UINT8]:
            pytest.skip(f"Integer types not supported in opset {opset_version} (requires v12+)")
        
        # Skip DOUBLE dtype for opset 6 - ONNX Runtime doesn't support Clip(6) for DOUBLE
        if opset_version == 6 and dtype == onnx.TensorProto.DOUBLE:
            pytest.skip(f"ONNX Runtime doesn't support Clip(6) for DOUBLE dtype")
        
        # Skip DOUBLE dtype for opset 11 - ONNX Runtime may not support Clip(11) for DOUBLE
        if opset_version == 11 and dtype == onnx.TensorProto.DOUBLE:
            pytest.skip(f"ONNX Runtime doesn't support Clip(11) for DOUBLE dtype")
        
        # Skip "only max" tests for v11+ - test creates invalid ONNX model (max in input[1] instead of input[2])
        # ONNX Runtime interprets input[1] as min by position, causing mismatch
        if opset_version >= 11 and min_val is None and max_val is not None:
            pytest.skip(f"Test creates invalid ONNX model for 'only max' case in opset {opset_version} "
                       f"(max should be in input[2], not input[1])")
        
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
        
        # Handle min/max based on opset version
        if opset_version >= 11:
            # v11+: min and max are optional input tensors
            if min_val is not None:
                min_array = np.array(min_val, dtype=np_dtype)
                input_names.append("min")
                initializers["min"] = min_array
            
            if max_val is not None:
                max_array = np.array(max_val, dtype=np_dtype)
                input_names.append("max")
                initializers["max"] = max_array
            
            onnx_model = create_onnx_model(
                op_type="Clip",
                input_shapes=[input_shape] + ([()] * (len(input_names) - 1)),  # Scalar shapes for min/max
                input_dtypes=[dtype] * len(input_names),
                output_shapes=[input_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="clip_test",
                input_names=input_names,
                initializers=initializers
            )
        else:
            # v1-v6: min and max are attributes
            if min_val is not None:
                attrs['min'] = float(min_val)
            if max_val is not None:
                attrs['max'] = float(max_val)
            
            onnx_model = create_onnx_model(
                op_type="Clip",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[input_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="clip_test"
            )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        
        # If both min and max are None:
        # - For v6: should use defaults (ClipNode with defaults)
        # - For v11+: should have IdentityNode (no clipping)
        if min_val is None and max_val is None:
            if opset_version >= 11:
                identity_nodes = [n for n in tir_graph.nodes if n.op_type == "Identity"]
                assert len(identity_nodes) == 1, \
                    f"Expected 1 IdentityNode when min=None and max=None for v11+, got {len(identity_nodes)}. " \
                    f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
                # Skip attribute checks for IdentityNode
            else:
                # v6: should use defaults (ClipNode)
                clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
                assert len(clip_nodes) == 1, \
                    f"Expected 1 ClipNode with defaults for v6 when min=None and max=None, got {len(clip_nodes)}. " \
                    f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
                # Skip comparison for v6 defaults (ONNX Runtime may not support Clip(6) for all dtypes)
                clip_node = clip_nodes[0]
                assert 'min' in clip_node.attrs and 'max' in clip_node.attrs, \
                    f"ClipNode should have default min/max attributes for v6"
                # Skip ONNX comparison for v6 None,None case
                return
        else:
            # Should have one ClipNode
            clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
            assert len(clip_nodes) == 1, \
                f"Expected 1 ClipNode, got {len(clip_nodes)}. " \
                f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
            
            # Verify ClipNode has correct attributes and single input
            clip_node = clip_nodes[0]
            # ClipNode should always have only one input (data input)
            assert len(clip_node.inputs) == 1, \
                f"ClipNode should have exactly 1 input (data), got {len(clip_node.inputs)}: {clip_node.inputs}"
            assert clip_node.inputs[0] == "input_0", \
                f"ClipNode input should be 'input_0', got {clip_node.inputs[0]}"
            
            # min/max should be in attrs (extracted from input tensors in v11+)
            if min_val is not None:
                assert 'min' in clip_node.attrs, \
                    f"ClipNode {clip_node.name} missing 'min' attribute"
                assert abs(clip_node.attrs['min'] - float(min_val)) < 1e-6, \
                    f"ClipNode min mismatch: expected {min_val}, got {clip_node.attrs['min']}"
            if max_val is not None:
                assert 'max' in clip_node.attrs, \
                    f"ClipNode {clip_node.name} missing 'max' attribute"
                assert abs(clip_node.attrs['max'] - float(max_val)) < 1e-6, \
                    f"ClipNode max mismatch: expected {max_val}, got {clip_node.attrs['max']}"
        
        # Create test input
        if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
            # Create input with values outside the clip range
            input_data = {
                'input_0': np.random.randint(-100, 100, size=input_shape, dtype=np_dtype)
            }
            rtol, atol = 0, 0
        else:
            # Create input with values outside the clip range
            input_data = {
                'input_0': np.random.randn(*input_shape).astype(np_dtype) * 5  # Values in range [-5, 5]
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
            f"min={min_val}, max={max_val}, dtype={dtype}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, " \
            f"min={min_val}, max={max_val}, dtype={dtype}"
    
    @pytest.mark.parametrize("opset_version", [13, 21, 23, 24, 25])
    def test_clip_min_greater_than_max(self, opset_version):
        """Test Clip v13+ behavior when min > max (all values set to max)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        dtype = onnx.TensorProto.FLOAT
        input_shape = (3, 4)
        min_val = 10.0  # Greater than max
        max_val = 5.0
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        # v13+: min and max are input tensors
        min_array = np.array(min_val, dtype=np.float32)
        max_array = np.array(max_val, dtype=np.float32)
        input_names.extend(["min", "max"])
        initializers["min"] = min_array
        initializers["max"] = max_array
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape, (), ()],  # Scalar shapes for min/max
            input_dtypes=[dtype, dtype, dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_min_gt_max",
            input_names=input_names,
            initializers=initializers
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify ClipNode
        clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
        assert len(clip_nodes) == 1, \
            f"Expected 1 ClipNode, got {len(clip_nodes)}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32) * 10
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
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
        
        # Verify all values are set to max (v13+ behavior)
        tir_output = comparison['tir_outputs'].get('output_0')
        if tir_output is not None:
            # All values should be max_val (5.0)
            assert np.allclose(tir_output, max_val, rtol=1e-5, atol=1e-6), \
                f"Expected all values to be {max_val} when min > max, but got values in range " \
                f"[{tir_output.min():.6f}, {tir_output.max():.6f}]"
    
    def test_clip_no_limits(self):
        """Test Clip with no min/max (should pass through unchanged as IdentityNode)."""
        opset_version = 11
        dtype = onnx.TensorProto.FLOAT
        input_shape = (3, 4)
        
        # No min, no max
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_no_limits"
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Should have IdentityNode instead of ClipNode when both min and max are None
        identity_nodes = [n for n in tir_graph.nodes if n.op_type == "Identity"]
        assert len(identity_nodes) == 1, \
            f"Expected 1 IdentityNode when min=None and max=None, got {len(identity_nodes)}. " \
            f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        
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
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
        
        # Verify output equals input (no clipping)
        tir_output = comparison['tir_outputs'].get('output_0')
        onnx_output = comparison['onnx_outputs'].get('output_0')
        if tir_output is not None and onnx_output is not None:
            assert np.allclose(tir_output, input_data['input_0'], rtol=1e-5, atol=1e-6), \
                "Output should equal input when no min/max limits"
    
    @pytest.mark.parametrize("opset_version", [1, 6, 11, 12, 13])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    ])
    def test_clip_different_dtypes(self, opset_version, dtype):
        """Test Clip with different data types."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip opset 1 - ONNXRuntime doesn't support Clip(1)
        if opset_version == 1:
            pytest.skip(f"Opset 1 not supported by ONNXRuntime (Clip(1) not implemented)")
        
        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")
        
        # Skip integer types for opset < 12
        if opset_version < 12 and dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
            pytest.skip(f"Integer types not supported in opset {opset_version} (requires v12+)")
        
        # Skip DOUBLE dtype for opset 6 and 11 - ONNX Runtime doesn't support Clip for DOUBLE
        if opset_version in [6, 11] and dtype == onnx.TensorProto.DOUBLE:
            pytest.skip(f"ONNX Runtime doesn't support Clip({opset_version}) for DOUBLE dtype")
        
        input_shape = (3, 4)
        min_val = 0.0 if dtype in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE] else 0
        max_val = 1.0 if dtype in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE] else 100
        
        # Map ONNX dtype to numpy dtype
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
        
        if opset_version >= 11:
            min_array = np.array(min_val, dtype=np_dtype)
            max_array = np.array(max_val, dtype=np_dtype)
            input_names.extend(["min", "max"])
            initializers["min"] = min_array
            initializers["max"] = max_array
            
            onnx_model = create_onnx_model(
                op_type="Clip",
                input_shapes=[input_shape, (), ()],
                input_dtypes=[dtype, dtype, dtype],
                output_shapes=[input_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="clip_dtype",
                input_names=input_names,
                initializers=initializers
            )
        else:
            attrs['min'] = float(min_val) if min_val is not None else None
            attrs['max'] = float(max_val) if max_val is not None else None
            
            onnx_model = create_onnx_model(
                op_type="Clip",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[input_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="clip_dtype"
            )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify ClipNode
        clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
        assert len(clip_nodes) == 1, \
            f"Expected 1 ClipNode, got {len(clip_nodes)}"
        
        # Create test input
        input_data = {}
        if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
            input_data['input_0'] = np.random.randint(-100, 100, size=input_shape, dtype=np_dtype)
        else:
            input_data['input_0'] = np.random.randn(*input_shape).astype(np_dtype) * 5
        
        rtol, atol = (0, 0) if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64] else (1e-5, 1e-6)
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=rtol,
            atol=atol
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors for dtype={dtype}: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch for dtype={dtype}: {comparison}"
    
    def test_clip_v6_defaults(self):
        """Test Clip v6 with default min/max values."""
        opset_version = 6
        dtype = onnx.TensorProto.FLOAT
        input_shape = (3, 4)
        
        # Don't set min/max - should use defaults
        attrs = {}
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_defaults"
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify ClipNode
        clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
        assert len(clip_nodes) == 1, \
            f"Expected 1 ClipNode, got {len(clip_nodes)}"
        
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
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
    
    def test_clip_extreme_values(self):
        """Test Clip with extreme min/max values."""
        opset_version = 11
        dtype = onnx.TensorProto.FLOAT
        input_shape = (3, 4)
        
        # Very large min/max values
        min_val = -1e10
        max_val = 1e10
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        min_array = np.array(min_val, dtype=np.float32)
        max_array = np.array(max_val, dtype=np.float32)
        input_names.extend(["min", "max"])
        initializers["min"] = min_array
        initializers["max"] = max_array
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape, (), ()],
            input_dtypes=[dtype, dtype, dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_extreme",
            input_names=input_names,
            initializers=initializers
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify ClipNode
        clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
        assert len(clip_nodes) == 1, \
            f"Expected 1 ClipNode, got {len(clip_nodes)}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32) * 1000
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
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
    
    @pytest.mark.parametrize("opset_version", [11, 12, 13])
    @pytest.mark.parametrize("min_val, max_val", [
        (0.0, 0.0),      # Equal min and max
        (-1.0, -1.0),    # Equal negative
        (1.0, 1.0),      # Equal positive
    ])
    def test_clip_equal_min_max(self, opset_version, min_val, max_val):
        """Test Clip when min equals max (all values should be set to that value)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        dtype = onnx.TensorProto.FLOAT
        input_shape = (3, 4)
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        min_array = np.array(min_val, dtype=np.float32)
        max_array = np.array(max_val, dtype=np.float32)
        input_names.extend(["min", "max"])
        initializers["min"] = min_array
        initializers["max"] = max_array
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape, (), ()],
            input_dtypes=[dtype, dtype, dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_equal",
            input_names=input_names,
            initializers=initializers
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify ClipNode
        clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
        assert len(clip_nodes) == 1, \
            f"Expected 1 ClipNode, got {len(clip_nodes)}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32) * 10
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
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
        
        # Verify all values are set to min/max (they're equal)
        tir_output = comparison['tir_outputs'].get('output_0')
        if tir_output is not None:
            assert np.allclose(tir_output, min_val, rtol=1e-5, atol=1e-6), \
                f"Expected all values to be {min_val} when min == max, but got values in range " \
                f"[{tir_output.min():.6f}, {tir_output.max():.6f}]"
    
    @pytest.mark.parametrize("opset_version", [11, 12, 13])
    @pytest.mark.parametrize("input_shape, min_val, max_val", [
        ((3, 4), -1e-6, 1e-6),      # Very small values
        ((2, 3), -1e-10, 1e-10),    # Extremely small values
        ((5,), 0.0, 0.0),           # Zero range
        ((3, 4), -100.0, -50.0),    # Negative range
        ((2, 3), 50.0, 100.0),      # Positive range
    ])
    def test_clip_edge_values(self, opset_version, input_shape, min_val, max_val):
        """Test Clip with edge case min/max values."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        dtype = onnx.TensorProto.FLOAT
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        min_array = np.array(min_val, dtype=np.float32)
        max_array = np.array(max_val, dtype=np.float32)
        input_names.extend(["min", "max"])
        initializers["min"] = min_array
        initializers["max"] = max_array
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape, (), ()],
            input_dtypes=[dtype, dtype, dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_edge",
            input_names=input_names,
            initializers=initializers
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify ClipNode
        clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
        assert len(clip_nodes) == 1, \
            f"Expected 1 ClipNode, got {len(clip_nodes)}"
        
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
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
    
    @pytest.mark.parametrize("opset_version", [12, 13])
    @pytest.mark.parametrize("dtype, min_val, max_val", [
        (onnx.TensorProto.INT8, -128, 127),
        (onnx.TensorProto.UINT8, 0, 255),
        (onnx.TensorProto.INT16, -32768, 32767),
        (onnx.TensorProto.UINT16, 0, 65535),
        (onnx.TensorProto.UINT32, 0, 1000),
        (onnx.TensorProto.UINT64, 0, 1000),
    ])
    def test_clip_integer_types(self, opset_version, dtype, min_val, max_val):
        """Test Clip with various integer types (v12+)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip integer types that ONNX Runtime doesn't support
        # ONNX Runtime doesn't support Clip for UINT16, UINT32, UINT64 in opset 12
        if opset_version == 12 and dtype in [onnx.TensorProto.UINT16, onnx.TensorProto.UINT32, onnx.TensorProto.UINT64]:
            pytest.skip(f"ONNX Runtime doesn't support Clip(12) for dtype={dtype}")
        
        # ONNX Runtime doesn't support Clip for INT16 in opset 12/13
        if dtype == onnx.TensorProto.INT16:
            pytest.skip(f"ONNX Runtime doesn't support Clip for INT16 in opset {opset_version}")
        
        # ONNX Runtime doesn't support UINT16 in opset 13 (Where operator issue)
        if opset_version == 13 and dtype == onnx.TensorProto.UINT16:
            pytest.skip(f"ONNX Runtime doesn't support UINT16 in opset 13 (Where operator not implemented)")
        
        input_shape = (3, 4)
        
        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.INT8: np.int8,
            onnx.TensorProto.UINT8: np.uint8,
            onnx.TensorProto.INT16: np.int16,
            onnx.TensorProto.UINT16: np.uint16,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.UINT32: np.uint32,
            onnx.TensorProto.INT64: np.int64,
            onnx.TensorProto.UINT64: np.uint64,
        }
        np_dtype = dtype_map.get(dtype, np.int32)
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        min_array = np.array(min_val, dtype=np_dtype)
        max_array = np.array(max_val, dtype=np_dtype)
        input_names.extend(["min", "max"])
        initializers["min"] = min_array
        initializers["max"] = max_array
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape, (), ()],
            input_dtypes=[dtype, dtype, dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_int",
            input_names=input_names,
            initializers=initializers
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify ClipNode
        clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
        assert len(clip_nodes) == 1, \
            f"Expected 1 ClipNode, got {len(clip_nodes)}"
        
        # Create test input - use safe range to avoid overflow for small integer types
        if dtype in [onnx.TensorProto.INT8]:
            # For int8, use a safe range within [-128, 127]
            safe_min = max(-128, min_val - 10)
            safe_max = min(127, max_val + 10)
            input_data = {
                'input_0': np.random.randint(safe_min, safe_max, size=input_shape, dtype=np_dtype)
            }
        elif dtype in [onnx.TensorProto.UINT8]:
            # For uint8, use a safe range within [0, 255]
            safe_min = max(0, min_val - 10)
            safe_max = min(255, max_val + 10)
            input_data = {
                'input_0': np.random.randint(safe_min, safe_max, size=input_shape, dtype=np_dtype)
            }
        elif dtype in [onnx.TensorProto.INT16]:
            # For int16, use a safe range within [-32768, 32767]
            safe_min = max(-32768, min_val - 100)
            safe_max = min(32767, max_val + 100)
            input_data = {
                'input_0': np.random.randint(safe_min, safe_max, size=input_shape, dtype=np_dtype)
            }
        elif dtype in [onnx.TensorProto.UINT16]:
            # For uint16, use a safe range within [0, 65535]
            safe_min = max(0, min_val - 100)
            safe_max = min(65535, max_val + 100)
            input_data = {
                'input_0': np.random.randint(safe_min, safe_max, size=input_shape, dtype=np_dtype)
            }
        elif dtype in [onnx.TensorProto.UINT32]:
            # For uint32, use a safe range within [0, max_uint32]
            # Can't subtract from 0 for unsigned types
            safe_min = max(0, min_val)
            safe_max = min(np.iinfo(np.uint32).max, max_val + 50)
            input_data = {
                'input_0': np.random.randint(safe_min, safe_max, size=input_shape, dtype=np_dtype)
            }
        elif dtype in [onnx.TensorProto.UINT64]:
            # For uint64, use a safe range within [0, max_uint64]
            # Can't subtract from 0 for unsigned types
            safe_min = max(0, min_val)
            safe_max = min(np.iinfo(np.uint64).max, max_val + 50)
            input_data = {
                'input_0': np.random.randint(safe_min, safe_max, size=input_shape, dtype=np_dtype)
            }
        else:
            # For larger signed integer types, use original logic
            input_data = {
                'input_0': np.random.randint(min_val - 50, max_val + 50, size=input_shape, dtype=np_dtype)
            }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=0,
            atol=0
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors for dtype={dtype}: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch for dtype={dtype}: {comparison}"
    
    def test_clip_v1_consumed_inputs_ignored(self):
        """Test that Clip v1 ignores consumed_inputs legacy attribute."""
        opset_version = 1
        # Skip opset 1 - ONNXRuntime doesn't support Clip(1)
        pytest.skip(f"Opset 1 not supported by ONNXRuntime (Clip(1) not implemented)")
        dtype = onnx.TensorProto.FLOAT
        input_shape = (3, 4)
        
        # Set consumed_inputs (legacy attribute) - should be ignored
        attrs = {
            'min': 0.0,
            'max': 1.0,
            'consumed_inputs': [0]  # Legacy attribute
        }
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_legacy"
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify ClipNode
        clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
        assert len(clip_nodes) == 1, \
            f"Expected 1 ClipNode, got {len(clip_nodes)}"
        
        # Verify consumed_inputs is not in attrs (it's ignored)
        clip_node = clip_nodes[0]
        assert 'consumed_inputs' not in clip_node.attrs, \
            "ClipNode should not have 'consumed_inputs' attribute (legacy, ignored)"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32) * 5
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
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
    
    def test_clip_zero_tensor(self):
        """Test Clip with zero tensor."""
        opset_version = 11
        dtype = onnx.TensorProto.FLOAT
        input_shape = (3, 4)
        min_val = -1.0
        max_val = 1.0
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        min_array = np.array(min_val, dtype=np.float32)
        max_array = np.array(max_val, dtype=np.float32)
        input_names.extend(["min", "max"])
        initializers["min"] = min_array
        initializers["max"] = max_array
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape, (), ()],
            input_dtypes=[dtype, dtype, dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_zero",
            input_names=input_names,
            initializers=initializers
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create zero input
        input_data = {
            'input_0': np.zeros(input_shape, dtype=np.float32)
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
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"
        
        # Verify output is still zero (within clip range)
        tir_output = comparison['tir_outputs'].get('output_0')
        if tir_output is not None:
            assert np.allclose(tir_output, 0.0, rtol=1e-5, atol=1e-6), \
                "Zero tensor should remain zero when within clip range"
    
    def test_clip_high_dimensional(self):
        """Test Clip with high-dimensional tensors."""
        opset_version = 11
        dtype = onnx.TensorProto.FLOAT
        input_shape = (2, 3, 4, 5, 6)
        min_val = -1.0
        max_val = 1.0
        
        attrs = {}
        initializers = {}
        input_names = ["input_0"]
        
        min_array = np.array(min_val, dtype=np.float32)
        max_array = np.array(max_val, dtype=np.float32)
        input_names.extend(["min", "max"])
        initializers["min"] = min_array
        initializers["max"] = max_array
        
        onnx_model = create_onnx_model(
            op_type="Clip",
            input_shapes=[input_shape, (), ()],
            input_dtypes=[dtype, dtype, dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="clip_high_dim",
            input_names=input_names,
            initializers=initializers
        )
        
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify ClipNode
        clip_nodes = [n for n in tir_graph.nodes if n.op_type == "Clip"]
        assert len(clip_nodes) == 1, \
            f"Expected 1 ClipNode, got {len(clip_nodes)}"
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32) * 5
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
            f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), \
            f"Output mismatch: {comparison}"

