"""
Test cases for ONNX Sigmoid operation.
Tests different input shapes, dtypes, opset versions, and edge cases.
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


class TestSigmoid:
    """Comprehensive test cases for Sigmoid operation."""
    
    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14, 21, 23, 24, 25])
    @pytest.mark.parametrize("input_shape", [
        # Scalar-like
        (1,),
        # 1D
        (5,),
        (10,),
        # 2D
        (3, 4),
        (2, 3),
        (10, 10),
        # 3D
        (2, 3, 4),
        (5, 5, 5),
        # 4D
        (2, 3, 4, 5),
        # Higher dimensions
        (1, 2, 3, 4, 5),
    ])
    @pytest.mark.parametrize("dtype", [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.FLOAT16,
    ])
    def test_sigmoid_basic(self, opset_version, input_shape, dtype):
        """Test basic Sigmoid operations across opset versions."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip opset 1 - ONNX Runtime doesn't support Sigmoid(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Sigmoid(1)")
        
        # Skip float16 for opset < 13 (may not be fully supported)
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
        attrs = {}
        # v1 has consumed_inputs attribute (legacy, should be ignored)
        if opset_version == 1:
            attrs['consumed_inputs'] = [0]  # Legacy attribute, should be ignored
        
        onnx_model = create_onnx_model(
            op_type="Sigmoid",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sigmoid_test"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) == 1, \
            f"Expected 1 node, got {len(tir_graph.nodes)}"
        
        sigmoid_nodes = [n for n in tir_graph.nodes if n.op_type == "Sigmoid"]
        assert len(sigmoid_nodes) == 1, \
            f"Expected 1 SigmoidNode, got {len(sigmoid_nodes)}. " \
            f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        
        sigmoid_node = sigmoid_nodes[0]
        assert len(sigmoid_node.inputs) == 1, \
            f"SigmoidNode should have exactly 1 input, got {len(sigmoid_node.inputs)}: {sigmoid_node.inputs}"
        assert sigmoid_node.inputs[0] == "input_0", \
            f"SigmoidNode input should be 'input_0', got {sigmoid_node.inputs[0]}"
        assert sigmoid_node.outputs[0] == "output_0", \
            f"SigmoidNode output should be 'output_0', got {sigmoid_node.outputs[0]}"
        
        # Create test input with mixed positive and negative values
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np_dtype) * 5  # Values in range [-5, 5]
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5 if dtype == onnx.TensorProto.FLOAT16 else 1e-6,
            atol=1e-4 if dtype == onnx.TensorProto.FLOAT16 else 1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}\n" \
            f"Test params: opset={opset_version}, input_shape={input_shape}, dtype={dtype}"
    
    @pytest.mark.parametrize("opset_version", [13, 14, 21, 23])
    @pytest.mark.parametrize("input_shape", [
        (3, 4),
        (2, 3, 4),
    ])
    def test_sigmoid_bfloat16(self, opset_version, input_shape):
        """Test Sigmoid with bfloat16 type (v13+)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip bfloat16 - ONNX Runtime doesn't support Sigmoid with bfloat16
        pytest.skip("ONNX Runtime doesn't support Sigmoid with bfloat16 type")
        
        dtype = onnx.TensorProto.BFLOAT16
        
        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Sigmoid",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="sigmoid_bfloat16"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) == 1
        sigmoid_nodes = [n for n in tir_graph.nodes if n.op_type == "Sigmoid"]
        assert len(sigmoid_nodes) == 1
        
        # Create test input
        # bfloat16 is not directly supported by numpy, use float32 and convert
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32) * 5
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-2,  # bfloat16 has lower precision
            atol=1e-2
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}"
    
    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14])
    def test_sigmoid_positive_values(self, opset_version):
        """Test Sigmoid with all positive values."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip opset 1 - ONNX Runtime doesn't support Sigmoid(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Sigmoid(1)")
        
        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT
        
        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs['consumed_inputs'] = [0]
        
        onnx_model = create_onnx_model(
            op_type="Sigmoid",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sigmoid_positive"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create test input with all positive values
        input_data = {
            'input_0': np.random.rand(*input_shape).astype(np.float32) * 10  # Values in range [0, 10]
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-6,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}"
        
        # Verify output is in range (0, 1) for positive inputs
        tir_output = comparison['tir_outputs']['output_0']
        assert np.all(tir_output > 0), "Sigmoid output should be > 0 for all inputs"
        assert np.all(tir_output < 1), "Sigmoid output should be < 1 for all inputs"
    
    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14])
    def test_sigmoid_negative_values(self, opset_version):
        """Test Sigmoid with all negative values."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip opset 1 - ONNX Runtime doesn't support Sigmoid(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Sigmoid(1)")
        
        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT
        
        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs['consumed_inputs'] = [0]
        
        onnx_model = create_onnx_model(
            op_type="Sigmoid",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sigmoid_negative"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create test input with all negative values
        # Use abs to ensure all values are positive, then negate to make them all negative
        input_data = {
            'input_0': -np.abs(np.random.randn(*input_shape).astype(np.float32)) * 5  # Values in range [-5, 0)
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-6,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}"
        
        # Verify output is in range (0, 0.5) for negative inputs
        tir_output = comparison['tir_outputs']['output_0']
        assert np.all(tir_output > 0), "Sigmoid output should be > 0 for all inputs"
        assert np.all(tir_output < 0.5), "Sigmoid output should be < 0.5 for negative inputs"
    
    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14])
    def test_sigmoid_zero_values(self, opset_version):
        """Test Sigmoid with zero values (should output 0.5)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip opset 1 - ONNX Runtime doesn't support Sigmoid(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Sigmoid(1)")
        
        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT
        
        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs['consumed_inputs'] = [0]
        
        onnx_model = create_onnx_model(
            op_type="Sigmoid",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sigmoid_zero"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create test input with all zeros
        input_data = {
            'input_0': np.zeros(input_shape, dtype=np.float32)
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-6,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}"
        
        # Verify output is 0.5 for zero input (sigmoid(0) = 0.5)
        tir_output = comparison['tir_outputs']['output_0']
        expected = np.full(input_shape, 0.5, dtype=np.float32)
        np.testing.assert_allclose(tir_output, expected, rtol=1e-6, atol=1e-6)
    
    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14])
    def test_sigmoid_edge_values(self, opset_version):
        """Test Sigmoid with edge values (very large, very small, zero)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")
        
        # Skip opset 1 - ONNX Runtime doesn't support Sigmoid(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Sigmoid(1)")
        
        input_shape = (2, 3)
        dtype = onnx.TensorProto.FLOAT
        
        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs['consumed_inputs'] = [0]
        
        onnx_model = create_onnx_model(
            op_type="Sigmoid",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sigmoid_edge"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Create test input with edge values
        input_data = {
            'input_0': np.array([
                [10.0, -10.0, 0.0],      # Very large positive, very large negative, zero
                [1e-5, -1e-5, 1.0]       # Very small positive, very small negative, one
            ], dtype=np.float32)
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-5,
            atol=1e-5
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}"
        
        # Verify expected behavior
        tir_output = comparison['tir_outputs']['output_0']
        # sigmoid(10) ≈ 0.99995, sigmoid(-10) ≈ 0.00005, sigmoid(0) = 0.5
        # sigmoid(1e-5) ≈ 0.5000025, sigmoid(-1e-5) ≈ 0.4999975, sigmoid(1) ≈ 0.731
        expected = np.array([
            [0.99995, 0.00005, 0.5],     # Large positive -> ~1, large negative -> ~0, zero -> 0.5
            [0.5000025, 0.4999975, 0.731]  # Small positive -> ~0.5, small negative -> ~0.5, one -> ~0.731
        ], dtype=np.float32)
        np.testing.assert_allclose(tir_output, expected, rtol=1e-3, atol=1e-3)
    
    def test_sigmoid_v1_consumed_inputs_ignored(self):
        """Test that consumed_inputs attribute in v1 is ignored."""
        opset_version = 1
        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT
        
        # Skip opset 1 - ONNX Runtime doesn't support Sigmoid(1)
        pytest.skip("ONNX Runtime doesn't support Sigmoid(1)")
        
        # Create ONNX model with consumed_inputs attribute
        attrs = {'consumed_inputs': [0]}
        
        onnx_model = create_onnx_model(
            op_type="Sigmoid",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="sigmoid_v1"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure (should work normally, ignoring consumed_inputs)
        assert len(tir_graph.nodes) == 1
        sigmoid_nodes = [n for n in tir_graph.nodes if n.op_type == "Sigmoid"]
        assert len(sigmoid_nodes) == 1
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32) * 5
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-6,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}"
    
    def test_sigmoid_high_dimensional(self):
        """Test Sigmoid with high-dimensional tensors."""
        opset_version = 13
        input_shape = (2, 3, 4, 5, 6)
        dtype = onnx.TensorProto.FLOAT
        
        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Sigmoid",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="sigmoid_high_dim"
        )
        
        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)
        
        # Verify structure
        assert len(tir_graph.nodes) == 1
        sigmoid_nodes = [n for n in tir_graph.nodes if n.op_type == "Sigmoid"]
        assert len(sigmoid_nodes) == 1
        
        # Create test input
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32) * 5
        }
        
        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data,
            rtol=1e-6,
            atol=1e-6
        )
        
        assert len(comparison['errors']) == 0, \
            f"Comparison errors: {comparison['errors']}"
        
        # Verify output is in valid range (0, 1]
        # Note: Due to floating point precision, sigmoid can be exactly 1.0 for very large inputs
        tir_output = comparison['tir_outputs']['output_0']
        assert np.all(tir_output >= 0), "Sigmoid output should be >= 0"
        assert np.all(tir_output <= 1), "Sigmoid output should be <= 1"

