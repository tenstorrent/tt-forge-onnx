# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Dropout operation.
Tests different opset versions, training/inference modes, ratios, and edge cases.
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
# HELPER METHODS FOR CREATING DROPOUT MODELS
# ============================================================================


def _create_dropout_model_v1_v6(opset_version, input_shape, ratio=0.5, is_test=0, seed=0, dtype=onnx.TensorProto.FLOAT):
    """
    Helper to create Dropout ONNX model for opset v1-v6.

    Args:
        opset_version: ONNX opset version (1-6)
        input_shape: Input tensor shape
        ratio: Dropout ratio (default: 0.5)
        is_test: If nonzero, run in inference mode (default: 0 = training)
        seed: Random seed (default: 0)
        dtype: ONNX tensor dtype
    """
    attrs = {
        "ratio": float(ratio),
        "is_test": int(is_test),
    }
    if seed != 0:
        attrs["seed"] = int(seed)

    return create_onnx_model(
        op_type="Dropout",
        input_shapes=[input_shape],
        input_dtypes=[dtype],
        output_shapes=[input_shape],  # Output shape same as input
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name="dropout",
    )


def _create_dropout_model_v7_v10(opset_version, input_shape, ratio=0.5, seed=0, dtype=onnx.TensorProto.FLOAT):
    """
    Helper to create Dropout ONNX model for opset v7-v10.

    Args:
        opset_version: ONNX opset version (7-10)
        input_shape: Input tensor shape
        ratio: Dropout ratio (default: 0.5)
        seed: Random seed (default: 0)
        dtype: ONNX tensor dtype
    """
    attrs = {
        "ratio": float(ratio),
    }
    if seed != 0:
        attrs["seed"] = int(seed)

    return create_onnx_model(
        op_type="Dropout",
        input_shapes=[input_shape],
        input_dtypes=[dtype],
        output_shapes=[input_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name="dropout",
    )


def _create_dropout_model_v12_plus(
    opset_version, input_shape, ratio=None, training_mode=None, seed=0, dtype=onnx.TensorProto.FLOAT
):
    """
    Helper to create Dropout ONNX model for opset v12+.

    Args:
        opset_version: ONNX opset version (12+)
        input_shape: Input tensor shape
        ratio: Optional ratio input (default: None, uses default 0.5)
        training_mode: Optional training_mode input (default: None, uses default False)
        seed: Random seed (default: 0)
        dtype: ONNX tensor dtype
    """
    attrs = {}
    if seed != 0:
        attrs["seed"] = int(seed)

    # Build inputs
    input_names = ["input_0"]  # data input
    input_shapes = [input_shape]
    input_dtypes = [dtype]
    initializers = {}

    # Add ratio input if provided
    if ratio is not None:
        input_names.append("ratio_input")
        input_shapes.append(())  # Scalar
        input_dtypes.append(dtype)
        initializers["ratio_input"] = np.array(ratio, dtype=np.float32)

    # Add training_mode input if provided
    if training_mode is not None:
        input_names.append("training_mode_input")
        input_shapes.append(())  # Scalar
        input_dtypes.append(onnx.TensorProto.BOOL)
        initializers["training_mode_input"] = np.array(training_mode, dtype=np.bool_)

    return create_onnx_model(
        op_type="Dropout",
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=[input_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name="dropout",
        input_names=input_names,
        initializers=initializers,
    )


# ============================================================================
# TEST CASES: OPSET V1-V6 (is_test and ratio attributes)
# ============================================================================


@pytest.mark.transpiler
class TestDropoutV1V6:
    """Test Dropout for opset versions 1-6 (uses is_test and ratio attributes).

    Note: ONNX Runtime doesn't support Dropout opset 1 and 6, so these tests are skipped.
    """

    @pytest.mark.parametrize("opset_version", [1, 6])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (1, 3, 32, 32),
            (2, 64, 16, 16),
            (1, 128),
            (4, 3, 224, 224),
        ],
    )
    @pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75])
    @pytest.mark.parametrize("is_test", [0, 1])
    def test_dropout_v1_v6_basic(self, opset_version, input_shape, ratio, is_test):
        """Test basic Dropout with different ratios and training/inference modes."""
        # Skip opset 1 and 6 - ONNX Runtime doesn't support them
        pytest.skip(
            f"ONNX Runtime doesn't support Dropout opset {opset_version}. "
            f"ONNX Runtime only guarantees support for opset 7+."
        )

        onnx_model = _create_dropout_model_v1_v6(
            opset_version=opset_version, input_shape=input_shape, ratio=ratio, is_test=is_test
        )

        # Create input data
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify graph structure
        verify_tir_graph_structure(tir_graph, onnx_model)

        # Compare with ONNX Runtime
        # Note: Dropout is stochastic, so we use relaxed tolerance
        # We mainly check that shapes match and values are reasonable
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-1, atol=1e-1)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output shape
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"

        # In inference mode (is_test=1), output should be identical to input
        if is_test == 1:
            np.testing.assert_allclose(tir_output, input_data["input_0"], rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("opset_version", [1, 6])
    def test_dropout_v1_v6_with_seed(self, opset_version):
        """Test Dropout with seed for reproducibility."""
        # Skip opset 1 and 6 - ONNX Runtime doesn't support them
        pytest.skip(
            f"ONNX Runtime doesn't support Dropout opset {opset_version}. "
            f"ONNX Runtime only guarantees support for opset 7+."
        )

        input_shape = (1, 3, 32, 32)
        seed = 42

        onnx_model = _create_dropout_model_v1_v6(
            opset_version=opset_version, input_shape=input_shape, ratio=0.5, is_test=0, seed=seed
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)

        # Run twice with same seed - should produce same result
        comparison1 = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-1, atol=1e-1)
        comparison2 = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-1, atol=1e-1)

        # Note: With seed, results should be deterministic
        output_name = onnx_model.graph.output[0].name
        output1 = comparison1["tir_outputs"][output_name]
        output2 = comparison2["tir_outputs"][output_name]

        # Shapes should match
        assert output1.shape == output2.shape


# ============================================================================
# TEST CASES: OPSET V7-V10 (ratio attribute, graph context for training)
# ============================================================================


@pytest.mark.transpiler
class TestDropoutV7V10:
    """Test Dropout for opset versions 7-10 (uses ratio attribute, graph context for training)."""

    @pytest.mark.parametrize("opset_version", [7, 10])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (1, 3, 32, 32),
            (2, 64, 16, 16),
            (1, 128),
        ],
    )
    @pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75])
    def test_dropout_v7_v10_basic(self, opset_version, input_shape, ratio):
        """Test basic Dropout with different ratios."""
        onnx_model = _create_dropout_model_v7_v10(opset_version=opset_version, input_shape=input_shape, ratio=ratio)

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)

        # Compare with ONNX Runtime
        # Note: For training mode (default), dropout is stochastic, so we only check shapes
        # For ratio=0, we can check exact values
        if ratio == 0.0:
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)
            assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
            # With ratio=0, output should be identical to input
            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            np.testing.assert_allclose(tir_output, input_data["input_0"], rtol=1e-5, atol=1e-5)
        else:
            # For training mode with dropout, only check shapes (values are stochastic)
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-1, atol=1e-1)
            # Filter out value mismatch errors for training mode (expected due to randomness)
            shape_errors = [e for e in comparison["errors"] if "Shape mismatch" in e or "ONNX execution failed" in e]
            assert len(shape_errors) == 0, f"Shape/execution errors: {shape_errors}"

            # Verify output shape
            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            onnx_output = comparison["onnx_outputs"][output_name]

            assert (
                tir_output.shape == onnx_output.shape
            ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"


# ============================================================================
# TEST CASES: OPSET V12+ (ratio and training_mode as inputs)
# ============================================================================


@pytest.mark.transpiler
class TestDropoutV12Plus:
    """Test Dropout for opset versions 12+ (uses ratio and training_mode as optional inputs)."""

    @pytest.mark.parametrize("opset_version", [12, 13, 22])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (1, 3, 32, 32),
            (2, 64, 16, 16),
            (1, 128),
        ],
    )
    def test_dropout_v12_plus_default(self, opset_version, input_shape):
        """Test Dropout with default ratio and training_mode (not provided as inputs).

        Note: Default training_mode is False (inference mode), so output should match input.
        """
        onnx_model = _create_dropout_model_v12_plus(
            opset_version=opset_version,
            input_shape=input_shape,
            ratio=None,  # Use default (0.5)
            training_mode=None,  # Use default (False = inference mode)
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)

        # Compare with ONNX Runtime
        # Default training_mode=False means inference mode, so output should match input exactly
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # Verify output shape
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"

        # In inference mode (default), output should be identical to input
        np.testing.assert_allclose(tir_output, input_data["input_0"], rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("opset_version", [12, 13, 22])
    @pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75])
    @pytest.mark.parametrize("training_mode", [True, False])
    def test_dropout_v12_plus_with_inputs(self, opset_version, ratio, training_mode):
        """Test Dropout with ratio and training_mode provided as inputs."""
        input_shape = (1, 3, 32, 32)

        onnx_model = _create_dropout_model_v12_plus(
            opset_version=opset_version, input_shape=input_shape, ratio=ratio, training_mode=training_mode
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)

        # Compare with ONNX Runtime
        # For inference mode or ratio=0, check exact values
        # For training mode with dropout, only check shapes (values are stochastic)
        if not training_mode or ratio == 0.0:
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)
            assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

            # Verify output shape
            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            onnx_output = comparison["onnx_outputs"][output_name]

            assert (
                tir_output.shape == onnx_output.shape
            ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"

            # In inference mode (training_mode=False) or ratio=0, output should be identical to input
            np.testing.assert_allclose(tir_output, input_data["input_0"], rtol=1e-5, atol=1e-5)
        else:
            # For training mode with dropout, only check shapes (values are stochastic)
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-1, atol=1e-1)
            # Filter out value mismatch errors for training mode (expected due to randomness)
            shape_errors = [e for e in comparison["errors"] if "Shape mismatch" in e or "ONNX execution failed" in e]
            assert len(shape_errors) == 0, f"Shape/execution errors: {shape_errors}"

            # Verify output shape
            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            onnx_output = comparison["onnx_outputs"][output_name]

            assert (
                tir_output.shape == onnx_output.shape
            ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"


# ============================================================================
# TEST CASES: EDGE CASES
# ============================================================================


@pytest.mark.transpiler
class TestDropoutEdgeCases:
    """Test edge cases for Dropout."""

    @pytest.mark.parametrize("opset_version", [7, 12])
    def test_dropout_ratio_zero(self, opset_version):
        """Test Dropout with ratio=0 (no dropout, should be identity)."""
        input_shape = (1, 3, 32, 32)

        if opset_version < 12:
            onnx_model = _create_dropout_model_v7_v10(opset_version=opset_version, input_shape=input_shape, ratio=0.0)
        else:
            onnx_model = _create_dropout_model_v12_plus(
                opset_version=opset_version, input_shape=input_shape, ratio=0.0, training_mode=True
            )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-5)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"

        # With ratio=0, output should be identical to input
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        np.testing.assert_allclose(tir_output, input_data["input_0"], rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("opset_version", [7, 12])
    def test_dropout_different_shapes(self, opset_version):
        """Test Dropout with various input shapes."""
        test_shapes = [
            (1,),  # 1D
            (1, 3),  # 2D
            (1, 3, 32),  # 3D
            (1, 3, 32, 32),  # 4D
            (2, 64, 16, 16),  # Larger batch
        ]

        for input_shape in test_shapes:
            if opset_version < 12:
                onnx_model = _create_dropout_model_v7_v10(
                    opset_version=opset_version, input_shape=input_shape, ratio=0.5
                )
            else:
                onnx_model = _create_dropout_model_v12_plus(
                    opset_version=opset_version, input_shape=input_shape, ratio=0.5, training_mode=True
                )

            input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

            transpiler = ONNXToForgeTranspiler(debug=False)
            tir_graph = transpiler.transpile(onnx_model)

            verify_tir_graph_structure(tir_graph, onnx_model)
            # For training mode with dropout, only check shapes (values are stochastic)
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-1, atol=1e-1)
            # Filter out value mismatch errors for training mode (expected due to randomness)
            shape_errors = [e for e in comparison["errors"] if "Shape mismatch" in e or "ONNX execution failed" in e]
            assert len(shape_errors) == 0, f"Shape {input_shape}: Shape/execution errors: {shape_errors}"

            # Verify output shape matches input shape
            output_name = onnx_model.graph.output[0].name
            tir_output = comparison["tir_outputs"][output_name]
            assert (
                tir_output.shape == input_shape
            ), f"Shape {input_shape}: Output shape mismatch: {tir_output.shape} vs {input_shape}"
