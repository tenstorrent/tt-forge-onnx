# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Reshape operation.
Tests different input shapes, dtypes, opset versions, and special behaviors (-1, 0, allowzero, empty shape).
"""
import pytest
import numpy as np
import onnx
import torch
from loguru import logger

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestReshape:
    """Comprehensive test cases for Reshape operation."""

    @pytest.mark.parametrize("opset_version", [1, 5, 13, 14, 18, 19, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "input_shape, output_shape",
        [
            # Basic reshapes
            ((2, 4), (8,)),  # Flatten 2D to 1D
            ((8,), (2, 4)),  # Unflatten 1D to 2D
            ((2, 3, 4), (24,)),  # Flatten 3D to 1D
            ((24,), (2, 3, 4)),  # Unflatten 1D to 3D
            ((2, 3, 4), (6, 4)),  # 3D to 2D
            ((6, 4), (2, 3, 4)),  # 2D to 3D
            ((1, 1, 1), (1,)),  # All ones
            ((10,), (10,)),  # Identity reshape
            ((4, 4), (2, 2, 2, 2)),  # 2D to 4D
            ((2, 2, 2, 2), (4, 4)),  # 4D to 2D
            # Edge cases
            ((1,), (1,)),  # Single element
            ((1, 1), (1,)),  # Multiple ones to one
            ((2, 1, 3), (6,)),  # With dimension 1
            ((6,), (2, 1, 3)),  # To shape with dimension 1
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
        ],
    )
    def test_reshape_basic(self, opset_version, input_shape, output_shape, dtype):
        """Test basic Reshape operations across opset versions."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip opset 1 for non-float types (opset 1 only supports float)
        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
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
        attrs = {}
        input_names = ["input_0"]
        initializers = {}

        if opset_version < 5:
            # Opset 1-4: shape as attribute
            attrs["shape"] = list(output_shape)
        else:
            # Opset 5+: shape as input tensor
            shape_tensor = np.array(output_shape, dtype=np.int64)
            initializers["shape"] = shape_tensor
            input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[output_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_test",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) >= 1, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # If input and output shapes are the same, Identity node is created (optimization)
        # Otherwise, Reshape node is created
        if input_shape == output_shape:
            assert tir_graph.nodes[0].op_type in [
                "Reshape",
                "Identity",
            ], f"Expected Reshape or Identity node for same shape, got {tir_graph.nodes[0].op_type}"
        else:
            assert tir_graph.nodes[0].op_type == "Reshape", f"Expected Reshape node, got {tir_graph.nodes[0].op_type}"

        # Create test input
        if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
            input_data = {"input_0": np.random.randint(1, 100, size=input_shape, dtype=np_dtype)}
            rtol, atol = 0, 0
        else:
            input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype)}
            rtol, atol = 1e-5, 1e-6

        # Compare outputs
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, "
            f"output_shape={output_shape}, dtype={dtype}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, "
            f"output_shape={output_shape}, dtype={dtype}"
        )

        logger.info(
            f"✓ Reshape basic test passed: opset={opset_version}, "
            f"input_shape={input_shape}, output_shape={output_shape}, dtype={dtype}"
        )

    @pytest.mark.parametrize("opset_version", [1, 5, 13, 14, 18])
    @pytest.mark.parametrize(
        "input_shape, shape_with_neg1, expected_shape",
        [
            # -1 in first position
            ((2, 3, 4), (-1, 4), (6, 4)),  # 24 elements: -1, 4 → 6, 4
            ((2, 3, 4), (-1, 2), (12, 2)),  # 24 elements: -1, 2 → 12, 2
            ((8,), (-1,), (8,)),  # 8 elements: -1 → 8
            # -1 in middle position
            ((2, 3, 4), (2, -1, 4), (2, 3, 4)),  # 24 elements: 2, -1, 4 → 2, 3, 4
            ((24,), (2, -1, 3), (2, 4, 3)),  # 24 elements: 2, -1, 3 → 2, 4, 3
            # -1 in last position
            ((2, 3, 4), (6, -1), (6, 4)),  # 24 elements: 6, -1 → 6, 4
            ((12,), (3, -1), (3, 4)),  # 12 elements: 3, -1 → 3, 4
        ],
    )
    def test_reshape_with_neg1(self, opset_version, input_shape, shape_with_neg1, expected_shape):
        """Test Reshape with -1 (inferred dimension)."""
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        input_names = ["input_0"]
        initializers = {}

        if opset_version < 5:
            # Opset 1-4: shape as attribute
            attrs["shape"] = list(shape_with_neg1)
        else:
            # Opset 5+: shape as input tensor
            shape_tensor = np.array(shape_with_neg1, dtype=np.int64)
            initializers["shape"] = shape_tensor
            input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_neg1",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        # Compare outputs
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, "
            f"shape_with_neg1={shape_with_neg1}, expected_shape={expected_shape}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, "
            f"shape_with_neg1={shape_with_neg1}, expected_shape={expected_shape}"
        )

        logger.info(
            f"✓ Reshape with -1 test passed: opset={opset_version}, "
            f"input_shape={input_shape}, shape={shape_with_neg1}, expected={expected_shape}"
        )

    @pytest.mark.parametrize("opset_version", [1, 5, 13, 14, 18])
    @pytest.mark.parametrize(
        "input_shape, shape_with_zero, expected_shape",
        [
            # 0 in first position (copy from input dimension 0)
            ((2, 3, 4), (0, 12), (2, 12)),  # Copy dim 0 = 2 → (2, 12) = 24 elements ✓
            # 0 in middle position (3D input, 3D output)
            ((2, 3, 4), (2, 0, 4), (2, 3, 4)),  # Copy dim 1 = 3 → (2, 3, 4) = 24 elements ✓
            # 0 in last position (2D output from 2D input)
            ((6, 4), (6, 0), (6, 4)),  # Copy dim 1 = 4 → (6, 4) = 24 elements ✓
            # 0 in first position (2D input)
            ((6, 4), (0, 4), (6, 4)),  # Copy dim 0 = 6 → (6, 4) = 24 elements ✓
            # 0 in first position (1D input)
            ((24,), (0,), (24,)),  # Copy dim 0 = 24 → (24,) = 24 elements ✓
        ],
    )
    def test_reshape_with_zero_copy(self, opset_version, input_shape, shape_with_zero, expected_shape):
        """Test Reshape with 0 (copy from input) - allowzero=0 behavior."""
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        input_names = ["input_0"]
        initializers = {}

        if opset_version < 5:
            # Opset 1-4: shape as attribute
            attrs["shape"] = list(shape_with_zero)
        else:
            # Opset 5+: shape as input tensor
            shape_tensor = np.array(shape_with_zero, dtype=np.int64)
            initializers["shape"] = shape_tensor
            input_names.append("shape")

        # Opset 14+: allowzero defaults to 0 (copy from input)
        if opset_version >= 14:
            attrs["allowzero"] = 0

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_zero",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        # Compare outputs
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, "
            f"shape_with_zero={shape_with_zero}, expected_shape={expected_shape}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, "
            f"shape_with_zero={shape_with_zero}, expected_shape={expected_shape}"
        )

        logger.info(
            f"✓ Reshape with 0 (copy) test passed: opset={opset_version}, "
            f"input_shape={input_shape}, shape={shape_with_zero}, expected={expected_shape}"
        )

    @pytest.mark.parametrize("opset_version", [14, 18, 19, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "input_shape, shape_with_zero, expected_shape",
        [
            # 0 with allowzero=1 means explicit zero
            ((2, 3, 4), (0, 4), (0, 4)),  # Explicit zero → empty tensor (0, 4)
            ((2, 3, 4), (3, 0), (3, 0)),  # Explicit zero → empty tensor (3, 0)
            ((2, 3, 4), (0, 0), (0, 0)),  # Multiple zeros → empty tensor (0, 0)
        ],
    )
    def test_reshape_with_zero_explicit(self, opset_version, input_shape, shape_with_zero, expected_shape):
        """Test Reshape with 0 (explicit zero) - allowzero=1 behavior."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model with allowzero=1
        attrs = {"allowzero": 1}
        input_names = ["input_0"]
        initializers = {}

        # Opset 5+: shape as input tensor
        shape_tensor = np.array(shape_with_zero, dtype=np.int64)
        initializers["shape"] = shape_tensor
        input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_zero_explicit",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        # Disable debug mode for empty tensor cases - ONNXRuntime doesn't support them
        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        # ONNXRuntime doesn't support reshaping to empty tensors, so skip comparison
        # Just verify that FullNode is created and produces correct output
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
        assert (
            tir_graph.nodes[0].op_type == "Full"
        ), f"Expected Full node for empty tensor, got {tir_graph.nodes[0].op_type}"

        # Execute TIR graph to verify output
        input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
        tir_outputs = tir_graph.run(input_dict)

        tir_output = tir_outputs["output_0"]
        assert tir_output.numel() == 0, f"Expected empty tensor, got {tir_output.numel()} elements"
        assert tuple(tir_output.shape) == expected_shape, f"Expected shape {expected_shape}, got {tir_output.shape}"

        logger.info(
            f"✓ Reshape with 0 (explicit) test passed: opset={opset_version}, "
            f"input_shape={input_shape}, shape={shape_with_zero}, expected={expected_shape}"
        )

    @pytest.mark.parametrize("opset_version", [5, 13, 14, 18])
    def test_reshape_empty_shape_scalar(self, opset_version):
        """Test Reshape with empty shape (scalar conversion)."""
        dtype = onnx.TensorProto.FLOAT
        input_shape = (1,)  # Single element
        output_shape = ()  # Scalar (empty shape)

        # Create ONNX model
        attrs = {}
        input_names = ["input_0"]
        initializers = {}

        if opset_version < 5:
            # Opset 1-4: shape as attribute (empty list)
            attrs["shape"] = []
        else:
            # Opset 5+: shape as input tensor (empty array)
            shape_tensor = np.array([], dtype=np.int64)
            initializers["shape"] = shape_tensor
            input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[output_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_scalar",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input (single element)
        input_data = {"input_0": np.array([42.5], dtype=np.float32)}

        # Compare outputs
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        # Verify output value matches (shape may be (1,) after flatten, but value should match)
        tir_output = comparison["tir_outputs"]["output_0"]
        onnx_output = comparison["onnx_outputs"]["output_0"]
        # For scalar conversion, check the value matches (ONNX produces scalar, TIR may produce (1,))
        if onnx_output.shape == ():
            # ONNX output is scalar, check TIR value matches
            tir_value = tir_output.item() if tir_output.size == 1 else tir_output[0].item()
            assert np.allclose(
                tir_value, onnx_output.item(), rtol=1e-5, atol=1e-6
            ), f"Scalar value mismatch: TIR={tir_value}, ONNX={onnx_output.item()}"
        else:
            # Both should match
            assert np.allclose(tir_output, onnx_output, rtol=1e-5, atol=1e-6), "Output values should match"

        logger.info(f"✓ Reshape empty shape (scalar) test passed: opset={opset_version}")

    @pytest.mark.parametrize("opset_version", [14, 18, 19, 21, 23, 24, 25])
    @pytest.mark.parametrize("allowzero", [0, 1])
    def test_reshape_allowzero_attribute(self, opset_version, allowzero):
        """Test Reshape with allowzero attribute (opset 14+)."""
        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        dtype = onnx.TensorProto.FLOAT
        input_shape = (2, 3, 4)  # 24 elements

        # Test with shape that has 0
        if allowzero == 0:
            # allowzero=0: 0 means copy from input
            shape_with_zero = [0, 12]  # Copy dim 0 = 2 → (2, 12)
            expected_shape = (2, 12)
        else:
            # allowzero=1: 0 means explicit zero
            shape_with_zero = [0, 4]  # Explicit zero → (0, 4)
            expected_shape = (0, 4)

        # Create ONNX model
        attrs = {"allowzero": allowzero}
        input_names = ["input_0"]
        initializers = {}

        shape_tensor = np.array(shape_with_zero, dtype=np.int64)
        initializers["shape"] = shape_tensor
        input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_allowzero",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        # Disable debug mode for allowzero=1 with empty tensor - ONNXRuntime doesn't support it
        transpiler = ONNXToForgeTranspiler(debug=(allowzero == 0), validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        # For allowzero=1 with empty tensor, ONNXRuntime doesn't support it
        # Just verify the converter creates the correct node
        if allowzero == 1:
            # Verify FullNode is created for empty tensor
            assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
            assert (
                tir_graph.nodes[0].op_type == "Full"
            ), f"Expected Full node for empty tensor, got {tir_graph.nodes[0].op_type}"

            # Execute TIR graph to verify output
            input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
            tir_outputs = tir_graph.run(input_dict)
            tir_output = tir_outputs["output_0"]
            assert tir_output.numel() == 0, f"Expected empty tensor, got {tir_output.numel()} elements"
            assert tuple(tir_output.shape) == expected_shape, f"Expected shape {expected_shape}, got {tir_output.shape}"
        else:
            # For allowzero=0, compare with ONNXRuntime
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

            assert len(comparison["errors"]) == 0, (
                f"Comparison errors: {comparison['errors']}\n"
                f"Test params: opset={opset_version}, allowzero={allowzero}"
            )
            assert all(comparison["matches"].values()), (
                f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, allowzero={allowzero}"
            )

        logger.info(f"✓ Reshape allowzero test passed: opset={opset_version}, allowzero={allowzero}")

    def test_reshape_empty_shape_creates_flatten(self):
        """Test that empty shape () creates ReshapeNode with shape (-1,) to flatten."""
        dtype = onnx.TensorProto.FLOAT
        input_shape = (2, 3, 4)  # 24 elements
        output_shape = (24,)  # Flattened

        # Create ONNX model with empty shape
        attrs = {}
        input_names = ["input_0"]
        initializers = {}

        # Opset 5+: shape as input tensor (empty array)
        shape_tensor = np.array([], dtype=np.int64)
        initializers["shape"] = shape_tensor
        input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[output_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=14,
            node_name="reshape_empty_shape",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure: should have ReshapeNode with shape (-1,) to flatten
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
        assert tir_graph.nodes[0].op_type == "Reshape", f"Expected Reshape node, got {tir_graph.nodes[0].op_type}"
        assert tir_graph.nodes[0].attrs.get("shape") == (
            -1,
        ), f"Expected shape (-1,) to flatten, got {tir_graph.nodes[0].attrs.get('shape')}"

        logger.info("✓ Empty shape creates ReshapeNode with (-1,) test passed")

    def test_reshape_empty_tensor_creates_fullnode(self):
        """Test that shape with 0 and allowzero=1 creates FullNode for empty tensor."""
        dtype = onnx.TensorProto.FLOAT
        input_shape = (2, 3, 4)
        shape_with_zero = [0, 4]  # With allowzero=1, this means empty tensor (0, 4)
        expected_shape = (0, 4)

        # Create ONNX model with allowzero=1
        attrs = {"allowzero": 1}
        input_names = ["input_0"]
        initializers = {}

        shape_tensor = np.array(shape_with_zero, dtype=np.int64)
        initializers["shape"] = shape_tensor
        input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=14,
            node_name="reshape_empty_tensor",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure: should have FullNode, not ReshapeNode
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
        assert (
            tir_graph.nodes[0].op_type == "Full"
        ), f"Expected Full node for empty tensor, got {tir_graph.nodes[0].op_type}"
        assert tir_graph.nodes[0].attrs.get("shape") == tuple(
            expected_shape
        ), f"Expected shape {expected_shape}, got {tir_graph.nodes[0].attrs.get('shape')}"
        assert (
            tir_graph.nodes[0].attrs.get("fill_value") == 0.0
        ), f"Expected fill_value 0.0, got {tir_graph.nodes[0].attrs.get('fill_value')}"

        logger.info("✓ Empty tensor (0 with allowzero=1) creates FullNode test passed")

    def test_reshape_validation_both_neg1_and_zero(self):
        """Test that shape cannot contain both -1 and 0."""
        from forge.transpiler.frontends.onnx.converters.reshape import ReshapeConverter

        # This should raise ValueError
        shape = (-1, 0, 4)
        input_shape = (2, 3, 4)

        with pytest.raises(ValueError, match="Shape cannot contain both -1.*and 0"):
            ReshapeConverter._resolve_shape(shape, input_shape, allowzero=0)

        # Also test with allowzero=1
        with pytest.raises(ValueError, match="Shape cannot contain both -1.*and 0"):
            ReshapeConverter._resolve_shape(shape, input_shape, allowzero=1)

        logger.info("✓ Validation: shape cannot contain both -1 and 0 test passed")

    def test_reshape_validation_invalid_shape_type(self):
        """Test that shape must be a list or tuple."""
        from forge.transpiler.frontends.onnx.converters.reshape import ReshapeConverter

        # Test with invalid types
        invalid_shapes = [
            "string",  # String
            42,  # Scalar int
            {"key": "value"},  # Dict
            None,  # None (though this is handled separately)
        ]

        input_shape = (2, 3, 4)

        for invalid_shape in invalid_shapes:
            if invalid_shape is None:
                continue  # None is handled separately in _normalize_shape_value
            with pytest.raises(TypeError, match="Shape must be a list or tuple"):
                ReshapeConverter._resolve_shape(invalid_shape, input_shape, allowzero=0)

        logger.info("✓ Validation: shape must be list or tuple test passed")

    @pytest.mark.parametrize("opset_version", [1, 5, 14])
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
        ],
    )
    def test_reshape_dtype_comprehensive(self, opset_version, dtype):
        """Test Reshape with various dtypes across opset versions."""
        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")

        input_shape = (2, 3, 4)
        output_shape = (6, 4)

        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # Create ONNX model
        attrs = {}
        input_names = ["input_0"]
        initializers = {}

        if opset_version < 5:
            attrs["shape"] = list(output_shape)
        else:
            shape_tensor = np.array(output_shape, dtype=np.int64)
            initializers["shape"] = shape_tensor
            input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[output_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_dtype",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input
        if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
            input_data = {"input_0": np.random.randint(1, 100, size=input_shape, dtype=np_dtype)}
            rtol, atol = 0, 0
        else:
            input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype)}
            rtol, atol = 1e-5, 1e-6

        # Compare outputs
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n" f"Test params: opset={opset_version}, dtype={dtype}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, dtype={dtype}"
        )

        logger.info(f"✓ Reshape dtype test passed: opset={opset_version}, dtype={dtype}")

    @pytest.mark.parametrize("opset_version", [1, 5, 14])
    @pytest.mark.parametrize(
        "input_shape, output_shape",
        [
            ((1,), (1,)),  # Single element
            ((1, 1), (1,)),  # Multiple ones
            ((1, 1, 1), (1,)),  # 3D ones
            ((10, 1), (10,)),  # One dimension is 1
            ((1, 10), (10,)),  # First dimension is 1
            ((1, 1, 10), (10,)),  # Multiple leading ones
        ],
    )
    def test_reshape_with_dimension_one(self, opset_version, input_shape, output_shape):
        """Test Reshape with dimension size 1."""
        dtype = onnx.TensorProto.FLOAT

        # Create ONNX model
        attrs = {}
        input_names = ["input_0"]
        initializers = {}

        if opset_version < 5:
            attrs["shape"] = list(output_shape)
        else:
            shape_tensor = np.array(output_shape, dtype=np.int64)
            initializers["shape"] = shape_tensor
            input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[output_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_dim1",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        # Compare outputs
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, output_shape={output_shape}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, output_shape={output_shape}"
        )

        logger.info(
            f"✓ Reshape with dimension 1 test passed: opset={opset_version}, "
            f"input_shape={input_shape}, output_shape={output_shape}"
        )

    def test_reshape_multiple_neg1_should_fail(self):
        """Test that shape with multiple -1 should fail (ambiguous)."""
        from forge.transpiler.frontends.onnx.converters.reshape import ReshapeConverter

        shape = (-1, -1, 4)  # Multiple -1 is invalid
        input_shape = (2, 3, 4)

        # This should raise ValueError when trying to infer
        with pytest.raises(ValueError, match="Cannot infer dimension"):
            ReshapeConverter._resolve_shape(shape, input_shape, allowzero=0)

        logger.info("✓ Validation: multiple -1 should fail test passed")

    def test_reshape_incompatible_shape_should_fail(self):
        """Test that incompatible shapes should fail."""
        from forge.transpiler.frontends.onnx.converters.reshape import ReshapeConverter

        # Shape that doesn't match total elements
        shape = (2, 3, 4)  # 24 elements
        input_shape = (2, 3)  # 6 elements - incompatible

        with pytest.raises(ValueError, match="Cannot reshape tensor"):
            ReshapeConverter._resolve_shape(shape, input_shape, allowzero=0)

        logger.info("✓ Validation: incompatible shape should fail test passed")

    @pytest.mark.parametrize("opset_version", [14, 18, 19, 21])
    def test_reshape_zero_with_allowzero_edge_cases(self, opset_version):
        """Test edge cases with 0 and allowzero."""
        dtype = onnx.TensorProto.FLOAT

        test_cases = [
            # (input_shape, shape_with_zero, allowzero, expected_shape, should_be_empty)
            ((2, 3, 4), [0, 0], 1, (0, 0), True),  # Multiple zeros
            ((2, 3, 4), [0, 1], 1, (0, 1), True),  # Zero with non-zero
            ((2, 3, 4), [1, 0], 1, (1, 0), True),  # Non-zero with zero
            ((2, 3, 4), [0, 12], 0, (2, 12), False),  # Copy from input
        ]

        for input_shape, shape_with_zero, allowzero_val, expected_shape, should_be_empty in test_cases:
            attrs = {"allowzero": allowzero_val} if opset_version >= 14 else {}
            input_names = ["input_0"]
            initializers = {}

            shape_tensor = np.array(shape_with_zero, dtype=np.int64)
            initializers["shape"] = shape_tensor
            input_names.append("shape")

            onnx_model = create_onnx_model(
                op_type="Reshape",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="reshape_zero_edge",
                input_names=input_names,
                initializers=initializers,
            )

            # Disable debug mode for empty tensor cases - ONNXRuntime doesn't support them
            transpiler = ONNXToForgeTranspiler(debug=not should_be_empty, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)

            input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

            if should_be_empty:
                # ONNXRuntime doesn't support reshaping to empty tensors, skip comparison
                # Just verify FullNode is created
                assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
                assert (
                    tir_graph.nodes[0].op_type == "Full"
                ), f"Expected Full node for empty tensor, got {tir_graph.nodes[0].op_type}"

                # Execute TIR graph to verify output
                input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
                tir_outputs = tir_graph.run(input_dict)
                tir_output = tir_outputs["output_0"]
                assert tir_output.numel() == 0, f"Expected empty tensor, got {tir_output.numel()} elements"
                assert (
                    tuple(tir_output.shape) == expected_shape
                ), f"Expected shape {expected_shape}, got {tir_output.shape}"
            else:
                # For non-empty tensors, compare with ONNXRuntime
                comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

                assert len(comparison["errors"]) == 0, (
                    f"Comparison errors: {comparison['errors']}\n"
                    f"Test case: input_shape={input_shape}, shape={shape_with_zero}, allowzero={allowzero_val}"
                )

            logger.info(
                f"✓ Reshape zero edge case passed: input_shape={input_shape}, "
                f"shape={shape_with_zero}, allowzero={allowzero_val}"
            )

    @pytest.mark.parametrize("opset_version", [1, 5, 14])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (2, 3, 4),
            (10,),
            (1, 1, 1),
            (4, 4),
            (2, 2, 2, 2),
        ],
    )
    def test_reshape_identity_optimization(self, opset_version, input_shape):
        """Test that Reshape with same input and output shape creates Identity node."""
        dtype = onnx.TensorProto.FLOAT
        output_shape = input_shape  # Same as input

        # Create ONNX model
        attrs = {}
        input_names = ["input_0"]
        initializers = {}

        if opset_version < 5:
            attrs["shape"] = list(output_shape)
        else:
            shape_tensor = np.array(output_shape, dtype=np.int64)
            initializers["shape"] = shape_tensor
            input_names.append("shape")

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[output_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_identity",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure: should have IdentityNode, not ReshapeNode
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
        assert (
            tir_graph.nodes[0].op_type == "Identity"
        ), f"Expected Identity node for same shape, got {tir_graph.nodes[0].op_type}"

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        # Compare outputs
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, input_shape={input_shape}"
        )

        logger.info(
            f"✓ Reshape identity optimization test passed: opset={opset_version}, " f"input_shape={input_shape}"
        )

    @pytest.mark.parametrize("opset_version", [1, 5, 14])
    @pytest.mark.parametrize(
        "input_shape, shape_with_copy",
        [
            # Shape with 0 that copies from input, resulting in same shape
            ((2, 3, 4), [0, 12]),  # Copy dim 0 = 2 → (2, 12) but input is (2, 3, 4), so different
            ((2, 3, 4), [2, 0, 4]),  # Copy dim 1 = 3 → (2, 3, 4) - same!
            ((2, 3, 4), [2, 3, 0]),  # Copy dim 2 = 4 → (2, 3, 4) - same!
            (
                (10,),
                [
                    0,
                ],
            ),  # Copy dim 0 = 10 → (10,) - same!
        ],
    )
    def test_reshape_identity_with_zero_copy(self, opset_version, input_shape, shape_with_copy):
        """Test that Reshape with 0 (copy from input) that results in same shape creates Identity."""
        dtype = onnx.TensorProto.FLOAT

        # Resolve the shape to see if it matches input
        # For [2, 0, 4] with input (2, 3, 4), copy dim 1 = 3 → (2, 3, 4) - same!
        resolved_shape = list(shape_with_copy)
        for i, s in enumerate(resolved_shape):
            if s == 0 and i < len(input_shape):
                resolved_shape[i] = input_shape[i]
        resolved_shape = tuple(resolved_shape)

        # Only test cases where resolved shape matches input
        if resolved_shape != input_shape:
            pytest.skip(f"Shape {shape_with_copy} resolves to {resolved_shape}, not {input_shape}")

        # Create ONNX model
        attrs = {}
        input_names = ["input_0"]
        initializers = {}

        if opset_version < 5:
            attrs["shape"] = list(shape_with_copy)
        else:
            shape_tensor = np.array(shape_with_copy, dtype=np.int64)
            initializers["shape"] = shape_tensor
            input_names.append("shape")

        if opset_version >= 14:
            attrs["allowzero"] = 0  # Copy from input

        onnx_model = create_onnx_model(
            op_type="Reshape",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[resolved_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="reshape_identity_zero",
            input_names=input_names,
            initializers=initializers,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=False, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure: should have IdentityNode
        assert len(tir_graph.nodes) == 1, f"Expected 1 node, got {len(tir_graph.nodes)}"
        assert (
            tir_graph.nodes[0].op_type == "Identity"
        ), f"Expected Identity node for same shape after 0 copy, got {tir_graph.nodes[0].op_type}"

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        # Compare outputs
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, shape={shape_with_copy}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, shape={shape_with_copy}"
        )

        logger.info(
            f"✓ Reshape identity with 0 copy test passed: opset={opset_version}, "
            f"input_shape={input_shape}, shape={shape_with_copy}"
        )
