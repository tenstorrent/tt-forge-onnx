# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Concat operation.
Tests different input shapes, dtypes, opset versions, axes, and edge cases.
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.core.exceptions import ConversionError
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestConcat:
    """Comprehensive test cases for Concat operation."""

    @pytest.mark.parametrize("opset_version", [4, 11, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "input_shapes, axis, expected_shape, dtype",
        [
            # 2D concatenation along axis 0
            ([(2, 3), (3, 3)], 0, (5, 3), onnx.TensorProto.FLOAT),
            ([(2, 3), (4, 3)], 0, (6, 3), onnx.TensorProto.FLOAT),
            # 2D concatenation along axis 1
            ([(3, 2), (3, 3)], 1, (3, 5), onnx.TensorProto.FLOAT),
            ([(3, 2), (3, 4)], 1, (3, 6), onnx.TensorProto.FLOAT),
            # 3D concatenation along axis 0
            ([(2, 3, 4), (3, 3, 4)], 0, (5, 3, 4), onnx.TensorProto.FLOAT),
            # 3D concatenation along axis 1
            ([(2, 3, 4), (2, 5, 4)], 1, (2, 8, 4), onnx.TensorProto.FLOAT),
            # 3D concatenation along axis 2
            ([(2, 3, 4), (2, 3, 5)], 2, (2, 3, 9), onnx.TensorProto.FLOAT),
            # Multiple inputs (3 tensors)
            ([(2, 3), (2, 3), (2, 3)], 0, (6, 3), onnx.TensorProto.FLOAT),
            ([(3, 2), (3, 2), (3, 2)], 1, (3, 6), onnx.TensorProto.FLOAT),
            # Multiple inputs (4 tensors)
            ([(1, 4), (1, 4), (1, 4), (1, 4)], 0, (4, 4), onnx.TensorProto.FLOAT),
            ([(4, 1), (4, 1), (4, 1), (4, 1)], 1, (4, 4), onnx.TensorProto.FLOAT),
            # Different sizes along concat axis
            ([(2, 3), (1, 3), (3, 3)], 0, (6, 3), onnx.TensorProto.FLOAT),
            ([(3, 2), (3, 1), (3, 3)], 1, (3, 6), onnx.TensorProto.FLOAT),
            # 1D concatenation
            ([(3,), (5,)], 0, (8,), onnx.TensorProto.FLOAT),
            ([(4,), (2,), (3,)], 0, (9,), onnx.TensorProto.FLOAT),
            # 4D concatenation
            ([(2, 3, 4, 5), (3, 3, 4, 5)], 0, (5, 3, 4, 5), onnx.TensorProto.FLOAT),
            ([(2, 3, 4, 5), (2, 3, 4, 3)], 3, (2, 3, 4, 8), onnx.TensorProto.FLOAT),
            # Integer types
            ([(2, 3), (3, 3)], 0, (5, 3), onnx.TensorProto.INT32),
            ([(2, 3), (3, 3)], 0, (5, 3), onnx.TensorProto.INT64),
            # Double precision
            ([(2, 3), (3, 3)], 0, (5, 3), onnx.TensorProto.DOUBLE),
        ],
    )
    def test_concat_basic(self, opset_version, input_shapes, axis, expected_shape, dtype):
        """Test basic Concat operations across opset versions."""
        # Skip opset 1 - ONNXRuntime doesn't support Concat(1)
        if opset_version == 1:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (Concat(1) not implemented)")

        # Skip opset 24 and 25 - ONNXRuntime doesn't support them yet
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")

        # Skip negative indices for opset 1 and v4-v10 (not supported)
        if opset_version < 11 and axis < 0:
            pytest.skip(f"Negative indices not supported in opset {opset_version}")

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

        # v1: axis is optional (defaults to 1), but we'll set it explicitly
        # v4+: axis is required
        if opset_version == 1:
            # v1: axis defaults to 1, but we can set it explicitly
            attrs["axis"] = axis
        else:
            # v4+: axis is required
            attrs["axis"] = axis

        # Create input names and shapes
        input_names = [f"input_{i}" for i in range(len(input_shapes))]
        input_dtypes = [dtype] * len(input_shapes)

        onnx_model = create_onnx_model(
            op_type="Concat",
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="concat_test",
            input_names=input_names,
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) >= 1, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # Should have one ConcatNode
        concat_nodes = [n for n in tir_graph.nodes if n.op_type == "Concat"]
        assert len(concat_nodes) == 1, (
            f"Expected 1 ConcatNode, got {len(concat_nodes)}. " f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        )

        # Verify ConcatNode has dim attribute
        concat_node = concat_nodes[0]
        assert "dim" in concat_node.attrs, f"ConcatNode {concat_node.name} missing 'dim' attribute"
        assert isinstance(
            concat_node.attrs["dim"], int
        ), f"ConcatNode {concat_node.name} 'dim' must be int, got {type(concat_node.attrs['dim'])}"

        # Normalize axis for comparison (negative indices are normalized)
        input_rank = len(input_shapes[0])
        normalized_axis = axis if axis >= 0 else axis + input_rank
        assert (
            concat_node.attrs["dim"] == normalized_axis
        ), f"ConcatNode dim mismatch: expected {normalized_axis}, got {concat_node.attrs['dim']}"

        # Create test inputs
        input_data = {}
        for i, shape in enumerate(input_shapes):
            input_name = f"input_{i}"
            if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
                input_data[input_name] = np.random.randint(1, 100, size=shape, dtype=np_dtype)
            else:
                input_data[input_name] = np.random.randn(*shape).astype(np_dtype)

        rtol, atol = (0, 0) if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64] else (1e-5, 1e-6)

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, input_shapes={input_shapes}, "
            f"axis={axis}, dtype={dtype}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n"
            f"Test params: opset={opset_version}, input_shapes={input_shapes}, "
            f"axis={axis}, dtype={dtype}"
        )

    @pytest.mark.parametrize("opset_version", [11, 13, 21, 23, 24, 25])
    def test_concat_negative_indices(self, opset_version):
        """Test Concat with negative indices (v11+)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        dtype = onnx.TensorProto.FLOAT

        # Test cases with negative indices
        # For 2D tensor (3, 4):
        #   axis=-1 means last dimension (axis=1)
        #   axis=-2 means second-to-last (axis=0)
        test_cases = [
            ([(2, 3), (2, 3)], -1, (2, 6)),  # Concatenate along last dim (axis=1)
            ([(2, 3), (3, 3)], -2, (5, 3)),  # Concatenate along first dim (axis=0)
            ([(2, 3, 4), (3, 3, 4)], -3, (5, 3, 4)),  # Concatenate along first dim (axis=0)
            ([(2, 3, 4), (2, 3, 5)], -1, (2, 3, 9)),  # Concatenate along last dim (axis=2)
            ([(2, 3, 4), (2, 5, 4)], -2, (2, 8, 4)),  # Concatenate along middle dim (axis=1)
        ]

        for input_shapes, axis, expected_shape in test_cases:
            attrs = {"axis": axis}
            input_names = [f"input_{i}" for i in range(len(input_shapes))]
            input_dtypes = [dtype] * len(input_shapes)

            onnx_model = create_onnx_model(
                op_type="Concat",
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="concat_negative",
                input_names=input_names,
            )

            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)

            # Verify ConcatNode
            concat_nodes = [n for n in tir_graph.nodes if n.op_type == "Concat"]
            assert len(concat_nodes) == 1, f"Expected 1 ConcatNode for axis={axis}, got {len(concat_nodes)}"

            # Verify normalized axis
            concat_node = concat_nodes[0]
            input_rank = len(input_shapes[0])
            normalized_axis = axis if axis >= 0 else axis + input_rank
            assert (
                concat_node.attrs["dim"] == normalized_axis
            ), f"Expected normalized axis {normalized_axis}, got {concat_node.attrs['dim']}"

            # Create test inputs
            input_data = {}
            for i, shape in enumerate(input_shapes):
                input_name = f"input_{i}"
                input_data[input_name] = np.random.randn(*shape).astype(np.float32)

            # Compare with ONNX runtime
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

            assert len(comparison["errors"]) == 0, f"Comparison errors for axis={axis}: {comparison['errors']}"
            assert all(comparison["matches"].values()), f"Output mismatch for axis={axis}: {comparison}"

    def test_concat_single_input(self):
        """Test Concat with single input (should return Identity)."""
        opset_version = 11
        dtype = onnx.TensorProto.FLOAT
        input_shape = (3, 4)

        attrs = {"axis": 0}
        onnx_model = create_onnx_model(
            op_type="Concat",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[input_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="concat_single",
        )

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Should have IdentityNode, not ConcatNode
        concat_nodes = [n for n in tir_graph.nodes if n.op_type == "Concat"]
        identity_nodes = [n for n in tir_graph.nodes if n.op_type == "Identity"]

        assert len(concat_nodes) == 0, f"Expected no ConcatNode for single input, got {len(concat_nodes)}"
        assert len(identity_nodes) == 1, f"Expected 1 IdentityNode for single input, got {len(identity_nodes)}"

        # Create test input
        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

    def test_concat_v1_default_axis(self):
        """Test Concat v1 with default axis (should default to 1)."""
        opset_version = 1
        # Skip opset 1 - ONNXRuntime doesn't support Concat(1)
        pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (Concat(1) not implemented)")

        dtype = onnx.TensorProto.FLOAT
        input_shapes = [(3, 2), (3, 3)]
        expected_shape = (3, 5)  # Concatenate along axis 1 (default)

        # Don't set axis attribute - should default to 1
        attrs = {}

        input_names = [f"input_{i}" for i in range(len(input_shapes))]
        input_dtypes = [dtype] * len(input_shapes)

        onnx_model = create_onnx_model(
            op_type="Concat",
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="concat_default",
            input_names=input_names,
        )

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify ConcatNode has dim=1 (default)
        concat_nodes = [n for n in tir_graph.nodes if n.op_type == "Concat"]
        assert len(concat_nodes) == 1, f"Expected 1 ConcatNode, got {len(concat_nodes)}"

        concat_node = concat_nodes[0]
        assert concat_node.attrs["dim"] == 1, f"Expected dim=1 (default), got {concat_node.attrs['dim']}"

        # Create test inputs
        input_data = {}
        for i, shape in enumerate(input_shapes):
            input_name = f"input_{i}"
            input_data[input_name] = np.random.randn(*shape).astype(np.float32)

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

    @pytest.mark.parametrize("opset_version", [4, 11, 13])
    def test_concat_missing_axis_error(self, opset_version):
        """Test that Concat v4+ raises error when axis is missing."""
        dtype = onnx.TensorProto.FLOAT
        input_shapes = [(3, 2), (3, 3)]

        # Don't set axis attribute - should cause error for v4+
        attrs = {}

        input_names = [f"input_{i}" for i in range(len(input_shapes))]
        input_dtypes = [dtype] * len(input_shapes)

        # Create model with missing axis (this might fail at ONNX validation)
        try:
            onnx_model = create_onnx_model(
                op_type="Concat",
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                output_shapes=[(3, 5)],  # Dummy shape
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="concat_missing_axis",
                input_names=input_names,
            )
        except Exception as e:
            # ONNX validation might catch this, which is fine
            # This is expected behavior - ONNX checker should reject models with missing required attributes
            return  # Test passes if ONNX validation catches the error

        # If ONNX validation didn't catch it, try transpiling
        # Transpile with validate_model=False to bypass ONNX validation
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=False)

        # Should raise ConversionError because axis is required
        from forge.transpiler.core.exceptions import ConversionError

        with pytest.raises(ConversionError) as exc_info:
            tir_graph = transpiler.transpile(onnx_model)

        # Verify error message mentions axis
        error_msg = str(exc_info.value).lower()
        assert "axis" in error_msg, f"Error message should mention 'axis', got: {error_msg}"

    @pytest.mark.parametrize("opset_version", [4, 11, 13])
    def test_concat_out_of_range_error(self, opset_version):
        """Test that Concat raises error when axis is out of range."""
        dtype = onnx.TensorProto.FLOAT
        input_shapes = [(3, 4)]  # Rank 2, valid axes: 0, 1 (or -2, -1)

        # Test invalid axes
        invalid_axes = []
        if opset_version < 11:
            # v1-v10: only non-negative, so test axis=2, axis=3
            invalid_axes = [2, 3, 10]
        else:
            # v11+: supports negative, so test axis=2, axis=-3
            invalid_axes = [2, -3, 10]

        for invalid_axis in invalid_axes:
            attrs = {"axis": invalid_axis}

            input_names = [f"input_{i}" for i in range(len(input_shapes))]
            input_dtypes = [dtype] * len(input_shapes)

            # Create model - ONNX might validate this
            try:
                onnx_model = create_onnx_model(
                    op_type="Concat",
                    input_shapes=input_shapes,
                    input_dtypes=input_dtypes,
                    output_shapes=[(3, 4)],  # Dummy shape
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="concat_invalid_axis",
                    input_names=input_names,
                )
            except Exception:
                # ONNX validation might catch this, which is fine
                continue

            # Transpile - should raise ConversionError
            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            with pytest.raises(ConversionError):
                transpiler.transpile(onnx_model)

    def test_concat_mismatched_ranks_error(self):
        """Test that Concat raises error when input ranks don't match."""
        opset_version = 11
        dtype = onnx.TensorProto.FLOAT

        # Different ranks: (3, 4) vs (2, 3, 4)
        input_shapes = [(3, 4), (2, 3, 4)]

        attrs = {"axis": 0}
        input_names = [f"input_{i}" for i in range(len(input_shapes))]
        input_dtypes = [dtype] * len(input_shapes)

        # Create model - ONNX might validate this
        try:
            onnx_model = create_onnx_model(
                op_type="Concat",
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                output_shapes=[(5, 3, 4)],  # Dummy shape
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="concat_mismatched_ranks",
                input_names=input_names,
            )
        except Exception:
            # ONNX validation might catch this, which is fine
            pytest.skip("ONNX validation caught mismatched ranks")

        # Transpile - should raise ConversionError
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        with pytest.raises(ConversionError):
            transpiler.transpile(onnx_model)

    def test_concat_mismatched_shapes_error(self):
        """Test that Concat raises error when input shapes don't match (except concat axis)."""
        opset_version = 11
        dtype = onnx.TensorProto.FLOAT

        # Same rank but different shapes in non-concat dimensions
        # Concatenating along axis=0, but shapes differ in axis=1: (3, 4) vs (3, 5)
        input_shapes = [(3, 4), (3, 5)]

        attrs = {"axis": 0}  # Concatenate along axis 0, but axis 1 differs
        input_names = [f"input_{i}" for i in range(len(input_shapes))]
        input_dtypes = [dtype] * len(input_shapes)

        # Create model - ONNX might validate this
        try:
            onnx_model = create_onnx_model(
                op_type="Concat",
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                output_shapes=[(6, 4)],  # Dummy shape
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="concat_mismatched_shapes",
                input_names=input_names,
            )
        except Exception:
            # ONNX validation might catch this, which is fine
            pytest.skip("ONNX validation caught mismatched shapes")

        # Transpile - should raise ConversionError
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        with pytest.raises(ConversionError):
            transpiler.transpile(onnx_model)

    def test_concat_high_dimensional(self):
        """Test Concat with high-dimensional tensors."""
        opset_version = 11
        dtype = onnx.TensorProto.FLOAT

        test_cases = [
            # 5D tensors
            ([(2, 3, 4, 5, 6), (3, 3, 4, 5, 6)], 0, (5, 3, 4, 5, 6)),
            ([(2, 3, 4, 5, 6), (2, 3, 4, 5, 3)], 4, (2, 3, 4, 5, 9)),
            # 6D tensors
            ([(1, 2, 3, 4, 5, 6), (2, 2, 3, 4, 5, 6)], 0, (3, 2, 3, 4, 5, 6)),
            ([(1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 3)], 5, (1, 2, 3, 4, 5, 9)),
        ]

        for input_shapes, axis, expected_shape in test_cases:
            attrs = {"axis": axis}
            input_names = [f"input_{i}" for i in range(len(input_shapes))]
            input_dtypes = [dtype] * len(input_shapes)

            onnx_model = create_onnx_model(
                op_type="Concat",
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="concat_high_dim",
                input_names=input_names,
            )

            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)

            # Verify ConcatNode
            concat_nodes = [n for n in tir_graph.nodes if n.op_type == "Concat"]
            assert len(concat_nodes) == 1, f"Expected 1 ConcatNode, got {len(concat_nodes)}"

            # Create test inputs
            input_data = {}
            for i, shape in enumerate(input_shapes):
                input_name = f"input_{i}"
                input_data[input_name] = np.random.randn(*shape).astype(np.float32)

            # Compare with ONNX runtime
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

            assert (
                len(comparison["errors"]) == 0
            ), f"Comparison errors for shapes={input_shapes}, axis={axis}: {comparison['errors']}"
            assert all(
                comparison["matches"].values()
            ), f"Output mismatch for shapes={input_shapes}, axis={axis}: {comparison}"

    def test_concat_many_inputs(self):
        """Test Concat with many input tensors."""
        opset_version = 11
        dtype = onnx.TensorProto.FLOAT

        # Create 10 input tensors
        num_inputs = 10
        input_shapes = [(2, 3)] * num_inputs
        expected_shape = (2 * num_inputs, 3)  # Concatenate along axis 0

        attrs = {"axis": 0}
        input_names = [f"input_{i}" for i in range(num_inputs)]
        input_dtypes = [dtype] * num_inputs

        onnx_model = create_onnx_model(
            op_type="Concat",
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="concat_many",
            input_names=input_names,
        )

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify ConcatNode
        concat_nodes = [n for n in tir_graph.nodes if n.op_type == "Concat"]
        assert len(concat_nodes) == 1, f"Expected 1 ConcatNode, got {len(concat_nodes)}"

        # Verify all inputs are connected
        concat_node = concat_nodes[0]
        assert len(concat_node.inputs) == num_inputs, f"Expected {num_inputs} inputs, got {len(concat_node.inputs)}"

        # Create test inputs
        input_data = {}
        for i in range(num_inputs):
            input_name = f"input_{i}"
            input_data[input_name] = np.random.randn(*input_shapes[0]).astype(np.float32)

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

    @pytest.mark.parametrize("opset_version", [4, 11, 13])
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
        ],
    )
    def test_concat_different_dtypes(self, opset_version, dtype):
        """Test Concat with different data types."""
        # Skip opset 1 - ONNXRuntime doesn't support Concat(1)
        if opset_version == 1:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (Concat(1) not implemented)")

        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        # Skip opset 1 for non-float types
        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")

        input_shapes = [(2, 3), (3, 3)]
        expected_shape = (5, 3)

        attrs = {"axis": 0}
        input_names = [f"input_{i}" for i in range(len(input_shapes))]
        input_dtypes = [dtype] * len(input_shapes)

        # Map ONNX dtype to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        onnx_model = create_onnx_model(
            op_type="Concat",
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="concat_dtype",
            input_names=input_names,
        )

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify ConcatNode
        concat_nodes = [n for n in tir_graph.nodes if n.op_type == "Concat"]
        assert len(concat_nodes) == 1, f"Expected 1 ConcatNode, got {len(concat_nodes)}"

        # Create test inputs
        input_data = {}
        for i, shape in enumerate(input_shapes):
            input_name = f"input_{i}"
            if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
                input_data[input_name] = np.random.randint(1, 100, size=shape, dtype=np_dtype)
            else:
                input_data[input_name] = np.random.randn(*shape).astype(np_dtype)

        rtol, atol = (0, 0) if dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64] else (1e-5, 1e-6)

        # Compare with ONNX runtime
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors for dtype={dtype}: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch for dtype={dtype}: {comparison}"
