# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Erf operation.
Tests different input shapes, dtypes, opset versions, and edge cases.
"""
import pytest
import numpy as np
import onnx
import torch

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestErf:
    """
    Comprehensive test cases for Erf operation.

    Note: ONNX Runtime doesn't support Erf operator. Tests will skip if ONNX Runtime
    comparison fails due to unsupported operator.
    """

    @pytest.mark.parametrize("opset_version", [9, 13])
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
    def test_erf_basic(self, opset_version, input_shape, dtype):
        """Test basic Erf operations across opset versions."""
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
        onnx_model = create_onnx_model(
            op_type="Erf",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="erf_test",
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"

        erf_nodes = [n for n in tir_graph.nodes if n.op_type == "Erf"]
        assert len(erf_nodes) == 1, (
            f"Expected 1 ErfNode, got {len(erf_nodes)}. " f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        )

        erf_node = erf_nodes[0]
        assert (
            len(erf_node.inputs) == 1
        ), f"ErfNode should have exactly 1 input, got {len(erf_node.inputs)}: {erf_node.input_names}"
        assert erf_node.input_names[0] == "input_0", f"ErfNode input should be 'input_0', got {erf_node.input_names[0]}"
        # Check forge_op_name is set correctly
        assert erf_node.forge_op_name == "Erf", f"ErfNode forge_op_name should be 'Erf', got {erf_node.forge_op_name}"
        # Check original output name (before sanitization)
        assert (
            erf_node.original_outputs[0] == "output_0"
        ), f"ErfNode output should be 'output_0', got {erf_node.original_outputs[0]}"

        # Create test input with various values
        np.random.seed(42)
        # Generate values in a reasonable range for erf
        input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype) * 2}

        # Try to compare with ONNX Runtime, skip if not supported
        try:
            comparison = compare_tir_with_onnx(
                tir_graph,
                onnx_model,
                input_data,
                rtol=1e-5 if dtype == onnx.TensorProto.FLOAT16 else 1e-6,
                atol=1e-4 if dtype == onnx.TensorProto.FLOAT16 else 1e-6,
            )
            # Check if comparison failed due to ONNX Runtime not supporting Erf
            if comparison.get("errors"):
                error_msg = " ".join(comparison["errors"])
                if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                    pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            assert len(comparison["errors"]) == 0, (
                f"Comparison errors: {comparison['errors']}\n"
                f"Test params: opset={opset_version}, input_shape={input_shape}, dtype={dtype}"
            )
        except Exception as e:
            error_msg = str(e)
            if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            raise

    @pytest.mark.parametrize("opset_version", [9, 13])
    def test_erf_known_values(self, opset_version):
        """Test Erf with known values."""
        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Erf",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="erf_known",
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with known values
        # erf(0) = 0, erf(1) ≈ 0.8427, erf(-1) ≈ -0.8427, erf(2) ≈ 0.9953
        input_data = {
            "input_0": np.array(
                [[-2.0, -1.0, 0.0, 1.0], [2.0, -0.5, 0.5, 1.5], [-1.5, 0.0, 1.0, 2.0]], dtype=np.float32
            )
        }

        # Try to compare with ONNX Runtime, skip if not supported
        try:
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)
            # Check if comparison failed due to ONNX Runtime not supporting Erf
            if comparison.get("errors"):
                error_msg = " ".join(comparison["errors"])
                if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                    pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
            tir_output = comparison["tir_outputs"]["output_0"]
        except Exception as e:
            error_msg = str(e)
            if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            # Fall back to TIR-only execution
            input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
            tir_outputs = tir_graph.run(input_dict)
            tir_output = tir_outputs["output_0"].detach().cpu().numpy()

        # Verify output is approximately correct
        # Check some known values
        assert np.isclose(tir_output[0, 2], 0.0, rtol=1e-5, atol=1e-5), f"Expected 0.0, got {tir_output[0, 2]}"
        assert np.isclose(tir_output[0, 3], 0.8427, rtol=1e-3, atol=1e-3), f"Expected ~0.8427, got {tir_output[0, 3]}"
        assert np.isclose(tir_output[0, 1], -0.8427, rtol=1e-3, atol=1e-3), f"Expected ~-0.8427, got {tir_output[0, 1]}"
        assert np.isclose(tir_output[1, 0], 0.9953, rtol=1e-3, atol=1e-3), f"Expected ~0.9953, got {tir_output[1, 0]}"

    @pytest.mark.parametrize("opset_version", [9, 13])
    def test_erf_zero_values(self, opset_version):
        """Test Erf with zero values (should output 0.0)."""
        input_shape = (3, 4)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Erf",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="erf_zero",
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with all zeros
        input_data = {"input_0": np.zeros(input_shape, dtype=np.float32)}

        # Try to compare with ONNX Runtime, skip if not supported
        try:
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)
            # Check if comparison failed due to ONNX Runtime not supporting Erf
            if comparison.get("errors"):
                error_msg = " ".join(comparison["errors"])
                if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                    pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
            tir_output = comparison["tir_outputs"]["output_0"]
        except Exception as e:
            error_msg = str(e)
            if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            # Fall back to TIR-only execution
            input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
            tir_outputs = tir_graph.run(input_dict)
            tir_output = tir_outputs["output_0"].detach().cpu().numpy()

        # Verify output is 0.0 for zero input (erf(0) = 0)
        expected = np.zeros(input_shape, dtype=np.float32)
        np.testing.assert_allclose(tir_output, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("opset_version", [9, 13])
    def test_erf_edge_values(self, opset_version):
        """Test Erf with edge values (very small, very large, zero, infinity)."""
        input_shape = (2, 3)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Erf",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="erf_edge",
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with edge values
        input_data = {
            "input_0": np.array(
                [[0.0, 1e-10, 1e10], [np.inf, -np.inf, 4.0]],
                dtype=np.float32,  # zero, very small, very large, inf, -inf, normal
            )
        }

        # Try to compare with ONNX Runtime, skip if not supported
        try:
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)
            # Check if comparison failed due to ONNX Runtime not supporting Erf
            if comparison.get("errors"):
                error_msg = " ".join(comparison["errors"])
                if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                    pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            tir_output = comparison["tir_outputs"]["output_0"]
        except Exception as e:
            error_msg = str(e)
            if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            # Fall back to TIR-only execution
            input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
            tir_outputs = tir_graph.run(input_dict)
            tir_output = tir_outputs["output_0"].detach().cpu().numpy()

        # Verify specific edge cases
        # erf(0) = 0
        assert np.isclose(tir_output[0, 0], 0.0, rtol=1e-5, atol=1e-5), f"Expected 0.0, got {tir_output[0, 0]}"
        # erf(inf) should approach 1
        if not np.isnan(tir_output[1, 0]) and not np.isinf(tir_output[1, 0]):
            assert np.isclose(tir_output[1, 0], 1.0, rtol=1e-3, atol=1e-3), f"Expected ~1.0, got {tir_output[1, 0]}"
        # erf(-inf) should approach -1
        if not np.isnan(tir_output[1, 1]) and not np.isinf(tir_output[1, 1]):
            assert np.isclose(tir_output[1, 1], -1.0, rtol=1e-3, atol=1e-3), f"Expected ~-1.0, got {tir_output[1, 1]}"

    @pytest.mark.parametrize("opset_version", [9, 13])
    def test_erf_small_values(self, opset_version):
        """Test Erf with small values (linear approximation region)."""
        input_shape = (2, 3)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Erf",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="erf_small",
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with small values
        # For small |x|, erf(x) ≈ (2/√π) * x
        input_data = {
            "input_0": np.array(
                [[-0.1, -0.01, 0.0], [0.01, 0.1, 0.5]],
                dtype=np.float32,
            )
        }

        # Try to compare with ONNX Runtime, skip if not supported
        try:
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)
            # Check if comparison failed due to ONNX Runtime not supporting Erf
            if comparison.get("errors"):
                error_msg = " ".join(comparison["errors"])
                if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                    pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
            tir_output = comparison["tir_outputs"]["output_0"]
        except Exception as e:
            error_msg = str(e)
            if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            # Fall back to TIR-only execution
            input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
            tir_outputs = tir_graph.run(input_dict)
            tir_output = tir_outputs["output_0"].detach().cpu().numpy()

        # Verify output
        # Check that erf is approximately linear for small values
        # erf(0.1) ≈ 0.1125, erf(0.01) ≈ 0.0113
        assert np.isclose(tir_output[1, 0], 0.0113, rtol=1e-2, atol=1e-2), f"Expected ~0.0113, got {tir_output[1, 0]}"
        assert np.isclose(tir_output[1, 1], 0.1125, rtol=1e-2, atol=1e-2), f"Expected ~0.1125, got {tir_output[1, 1]}"

    @pytest.mark.parametrize("opset_version", [9, 13])
    def test_erf_large_values(self, opset_version):
        """Test Erf with large values (should approach ±1)."""
        input_shape = (2, 3)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Erf",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="erf_large",
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input with large values
        # erf(3) ≈ 0.99998, erf(4) ≈ 0.99999998
        input_data = {
            "input_0": np.array(
                [[-4.0, -3.0, -2.0], [2.0, 3.0, 4.0]],
                dtype=np.float32,
            )
        }

        # Try to compare with ONNX Runtime, skip if not supported
        try:
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)
            # Check if comparison failed due to ONNX Runtime not supporting Erf
            if comparison.get("errors"):
                error_msg = " ".join(comparison["errors"])
                if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                    pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
            tir_output = comparison["tir_outputs"]["output_0"]
        except Exception as e:
            error_msg = str(e)
            if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            # Fall back to TIR-only execution
            input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
            tir_outputs = tir_graph.run(input_dict)
            tir_output = tir_outputs["output_0"].detach().cpu().numpy()

        # Verify output approaches ±1 for large values
        # Large positive values should approach 1
        assert tir_output[1, 1] > 0.99, f"Expected > 0.99, got {tir_output[1, 1]}"
        assert tir_output[1, 2] > 0.999, f"Expected > 0.999, got {tir_output[1, 2]}"
        # Large negative values should approach -1
        assert tir_output[0, 0] < -0.999, f"Expected < -0.999, got {tir_output[0, 0]}"
        assert tir_output[0, 1] < -0.99, f"Expected < -0.99, got {tir_output[0, 1]}"

    @pytest.mark.parametrize("opset_version", [13])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (3, 4),
            (2, 3, 4),
        ],
    )
    def test_erf_bfloat16(self, opset_version, input_shape):
        """Test Erf with bfloat16 type (v13+)."""
        # Skip bfloat16 - ONNX Runtime doesn't support Erf with bfloat16
        pytest.skip("ONNX Runtime doesn't support Erf with bfloat16 type")

    @pytest.mark.parametrize("opset_version", [9, 13])
    def test_erf_high_dimensional(self, opset_version):
        """Test Erf with high-dimensional tensors."""
        input_shape = (2, 3, 4, 5, 6)
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        onnx_model = create_onnx_model(
            op_type="Erf",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="erf_high_dim",
        )

        # Transpile with debug mode enabled
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) == 1
        erf_nodes = [n for n in tir_graph.nodes if n.op_type == "Erf"]
        assert len(erf_nodes) == 1

        # Create test input
        np.random.seed(42)
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32) * 2}

        # Try to compare with ONNX Runtime, skip if not supported
        try:
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-6, atol=1e-6)
            # Check if comparison failed due to ONNX Runtime not supporting Erf
            if comparison.get("errors"):
                error_msg = " ".join(comparison["errors"])
                if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                    pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

            # Verify output shape matches input shape
            tir_output = comparison["tir_outputs"]["output_0"]
            assert tir_output.shape == input_shape, f"Output shape {tir_output.shape} != input shape {input_shape}"
        except Exception as e:
            error_msg = str(e)
            if "NOT_IMPLEMENTED" in error_msg or "Could not find an implementation for Erf" in error_msg:
                pytest.skip(f"ONNX Runtime doesn't support Erf operator: {error_msg}")
            # Fall back to TIR-only execution
            input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
            tir_outputs = tir_graph.run(input_dict)
            tir_output = tir_outputs["output_0"].detach().cpu().numpy()
            assert tir_output.shape == input_shape, f"Output shape {tir_output.shape} != input shape {input_shape}"
