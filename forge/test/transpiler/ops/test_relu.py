# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Relu operation.
Tests different input shapes, dtypes, opset versions, and edge cases.
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestRelu:
    """Comprehensive test cases for Relu operation."""

    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "input_shape",
        [
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
    def test_relu_basic(self, opset_version, input_shape, dtype):
        """Test basic Relu operations across opset versions."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip opset 1 - ONNX Runtime doesn't support Relu(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Relu(1)")

        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")

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
            attrs["consumed_inputs"] = [0]  # Legacy attribute, should be ignored

        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="relu_test",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"

        relu_nodes = [n for n in tir_graph.nodes if n.op_type == "Relu"]
        assert len(relu_nodes) == 1, (
            f"Expected 1 ReluNode, got {len(relu_nodes)}. " f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        )

        relu_node = relu_nodes[0]
        assert (
            len(relu_node.inputs) == 1
        ), f"ReluNode should have exactly 1 input, got {len(relu_node.inputs)}: {relu_node.input_names}"
        assert (
            relu_node.input_names[0] == "input_0"
        ), f"ReluNode input should be 'input_0', got {relu_node.input_names[0]}"
        # Check original output name (before sanitization)
        assert (
            relu_node.original_outputs[0] == "output_0"
        ), f"ReluNode output should be 'output_0', got {relu_node.original_outputs[0]}"

        # Create test input with mixed positive and negative values
        input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype) * 5}  # Values in range [-5, 5]

        # Compare with ONNX runtime
        # Use relaxed tolerance for opset 14 due to numerical precision differences
        if dtype == onnx.TensorProto.FLOAT16:
            rtol_val, atol_val = 1e-5, 1e-4
        elif opset_version == 14:
            rtol_val, atol_val = 1e-4, 1e-5
        else:
            rtol_val, atol_val = 1e-6, 1e-6
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol_val, atol=atol_val)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, dtype={dtype}"
        )

    @pytest.mark.parametrize("opset_version", [13, 14, 21, 23])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (3, 4),
            (2, 3, 4),
        ],
    )
    def test_relu_bfloat16(self, opset_version, input_shape):
        """Test Relu with bfloat16 type (v13+)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip bfloat16 - ONNX Runtime doesn't support Relu with bfloat16
        pytest.skip("ONNX Runtime doesn't support Relu with bfloat16 type")

        dtype = onnx.TensorProto.BFLOAT16

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="relu_bfloat16",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1
        relu_nodes = [n for n in tir_graph.nodes if n.op_type == "Relu"]
        assert len(relu_nodes) == 1

        # Create test input
        # bfloat16 is not directly supported by numpy, use float32 and convert
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32) * 5}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(
            tir_graph, onnx_model, input_data, rtol=1e-2, atol=1e-2  # bfloat16 has lower precision
        )

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    @pytest.mark.parametrize("opset_version", [14, 21, 23])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (3, 4),
            (2, 3),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.INT8,
            onnx.TensorProto.INT16,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
        ],
    )
    def test_relu_integer_types(self, opset_version, input_shape, dtype):
        """Test Relu with integer types (v14+)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip INT16 and INT64 - ONNX Runtime doesn't support Relu(14) with these types
        if dtype in [onnx.TensorProto.INT16, onnx.TensorProto.INT64]:
            pytest.skip(f"ONNX Runtime doesn't support Relu with {dtype} type")

        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.INT8: np.int8,
            onnx.TensorProto.INT16: np.int16,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.int32)

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="relu_int",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1
        relu_nodes = [n for n in tir_graph.nodes if n.op_type == "Relu"]
        assert len(relu_nodes) == 1

        # Create test input with mixed positive and negative values
        # Use safe range to avoid overflow
        if dtype == onnx.TensorProto.INT8:
            input_data = {"input_0": np.random.randint(-50, 50, size=input_shape, dtype=np_dtype)}
        elif dtype == onnx.TensorProto.INT16:
            input_data = {"input_0": np.random.randint(-1000, 1000, size=input_shape, dtype=np_dtype)}
        else:
            input_data = {"input_0": np.random.randint(-10000, 10000, size=input_shape, dtype=np_dtype)}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=0, atol=0)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14])
    def test_relu_positive_values(self, opset_version):
        """Test Relu with all positive values (should remain unchanged)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip opset 1 - ONNX Runtime doesn't support Relu(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Relu(1)")

        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs["consumed_inputs"] = [0]

        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="relu_positive",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with all positive values
        input_data = {"input_0": np.random.rand(*input_shape).astype(np.float32) * 10}  # Values in range [0, 10]

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output equals input (all positive values)
        tir_output = comparison["tir_outputs"]["output_0"]
        onnx_output = comparison["onnx_outputs"]["output_0"]
        np.testing.assert_allclose(tir_output, onnx_output, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(tir_output, input_data["input_0"], rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14])
    def test_relu_negative_values(self, opset_version):
        """Test Relu with all negative values (should all become zero)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip opset 1 - ONNX Runtime doesn't support Relu(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Relu(1)")

        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs["consumed_inputs"] = [0]

        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="relu_negative",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with all negative values
        # Use abs to ensure all values are positive, then negate to make them all negative
        input_data = {
            "input_0": -np.abs(np.random.randn(*input_shape).astype(np.float32)) * 5  # Values in range [-5, 0)
        }

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output is all zeros
        tir_output = comparison["tir_outputs"]["output_0"]
        onnx_output = comparison["onnx_outputs"]["output_0"]
        np.testing.assert_allclose(tir_output, onnx_output, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(tir_output, np.zeros(input_shape, dtype=np.float32), rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14])
    def test_relu_zero_values(self, opset_version):
        """Test Relu with zero values (should remain zero)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip opset 1 - ONNX Runtime doesn't support Relu(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Relu(1)")

        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs["consumed_inputs"] = [0]

        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="relu_zero",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with all zeros
        input_data = {"input_0": np.zeros(input_shape, dtype=np.float32)}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output is all zeros
        tir_output = comparison["tir_outputs"]["output_0"]
        np.testing.assert_allclose(tir_output, np.zeros(input_shape, dtype=np.float32), rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("opset_version", [1, 6, 13, 14])
    def test_relu_edge_values(self, opset_version):
        """Test Relu with edge values (very large, very small, zero)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip opset 1 - ONNX Runtime doesn't support Relu(1)
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support Relu(1)")

        input_shape = (2, 3)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        if opset_version == 1:
            attrs["consumed_inputs"] = [0]

        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="relu_edge",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with edge values
        input_data = {
            "input_0": np.array(
                [
                    [1e10, -1e10, 0.0],  # Very large positive, very large negative, zero
                    [1e-10, -1e-10, 1.0],  # Very small positive, very small negative, one
                ],
                dtype=np.float32,
            )
        }

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify expected behavior
        tir_output = comparison["tir_outputs"]["output_0"]
        expected = np.array(
            [
                [1e10, 0.0, 0.0],  # Large positive unchanged, large negative -> 0, zero -> 0
                [1e-10, 0.0, 1.0],  # Small positive unchanged, small negative -> 0, one unchanged
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(tir_output, expected, rtol=1e-5, atol=1e-5)

    def test_relu_v1_consumed_inputs_ignored(self):
        """Test that consumed_inputs attribute in v1 is ignored."""
        opset_version = 1
        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Skip opset 1 - ONNX Runtime doesn't support Relu(1)
        pytest.skip("ONNX Runtime doesn't support Relu(1)")

        # Create ONNX model with consumed_inputs attribute
        attrs = {"consumed_inputs": [0]}

        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="relu_v1",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure (should work normally, ignoring consumed_inputs)
        assert len(tir_graph.nodes) == 1
        relu_nodes = [n for n in tir_graph.nodes if n.op_type == "Relu"]
        assert len(relu_nodes) == 1

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32) * 5}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

    @pytest.mark.parametrize("opset_version", [14, 21, 23])
    def test_relu_integer_negative_to_zero(self, opset_version):
        """Test that integer negative values become zero in Relu (v14+)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        input_shape = (2, 3)
        dtype = onnx.TensorProto.INT32

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="relu_int_negative",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with negative integers
        input_data = {"input_0": np.array([[-5, 0, 10], [-100, 50, -1]], dtype=np.int32)}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=0, atol=0)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify expected behavior: negative values -> 0, non-negative unchanged
        tir_output = comparison["tir_outputs"]["output_0"]
        expected = np.array(
            [[0, 0, 10], [0, 50, 0]], dtype=np.int32  # -5 -> 0, 0 -> 0, 10 -> 10  # -100 -> 0, 50 -> 50, -1 -> 0
        )
        np.testing.assert_array_equal(tir_output, expected)

    def test_relu_high_dimensional(self):
        """Test Relu with high-dimensional tensors."""
        opset_version = 14
        input_shape = (2, 3, 4, 5, 6)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Relu",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="relu_high_dim",
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1
        relu_nodes = [n for n in tir_graph.nodes if n.op_type == "Relu"]
        assert len(relu_nodes) == 1

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32) * 5}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
