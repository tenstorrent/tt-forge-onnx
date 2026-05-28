# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Squeeze operation.
Tests different input shapes, dtypes, opset versions, axes, and edge cases.
"""
import pytest
import numpy as np
import onnx
from loguru import logger

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestSqueeze:
    """Comprehensive test cases for Squeeze operation."""

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "input_shape, axes, expected_shape",
        [
            # Single axis squeeze
            ((1, 3, 4), [0], (3, 4)),  # Squeeze first dim
            ((2, 1, 4), [1], (2, 4)),  # Squeeze middle dim
            ((2, 3, 1), [2], (2, 3)),  # Squeeze last dim
            ((1, 1, 4), [0], (1, 4)),  # Squeeze first, leave second
            ((1, 1, 4), [1], (1, 4)),  # Squeeze second, leave first
            # Multiple axes squeeze
            ((1, 1, 4), [0, 1], (4,)),  # Squeeze first two dims
            ((1, 3, 1), [0, 2], (3,)),  # Squeeze first and last
            ((1, 1, 1), [0, 1], (1,)),  # Squeeze first two, leave one
            ((1, 1, 1), [0, 1, 2], ()),  # Squeeze all (scalar)
            # Negative indices (v11+)
            ((1, 3, 4), [-3], (3, 4)),  # Negative first dim
            ((2, 1, 4), [-2], (2, 4)),  # Negative middle dim
            ((2, 3, 1), [-1], (2, 3)),  # Negative last dim
            ((1, 1, 4), [-3, -2], (4,)),  # Negative multiple dims
            # No axes specified (squeeze all size-1)
            ((1, 3, 4), None, (3, 4)),  # Single size-1 dim
            ((1, 1, 4), None, (4,)),  # Multiple size-1 dims
            ((1, 1, 1), None, ()),  # All size-1 (scalar)
            ((2, 3, 4), None, (2, 3, 4)),  # No size-1 dims (no-op)
            # Edge cases
            ((1,), [0], ()),  # 1D to scalar
            ((1,), None, ()),  # 1D to scalar (no axes)
            ((1, 1), [0], (1,)),  # 2D all ones, squeeze first
            ((1, 1), [1], (1,)),  # 2D all ones, squeeze second
            ((1, 1), None, ()),  # 2D all ones, squeeze all
            ((1, 5, 1, 3), [0, 2], (5, 3)),  # 4D squeeze first and third
            ((1, 5, 1, 3), None, (5, 3)),  # 4D squeeze all size-1
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
    def test_squeeze_basic(self, opset_version, input_shape, axes, expected_shape, dtype):
        """Test basic Squeeze operations across opset versions."""
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

        # Skip negative indices for opset 1 (not supported)
        if opset_version == 1 and axes is not None:
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
            # v13+: axes is optional input tensor
            if axes is not None:
                axes_array = np.array(axes, dtype=np.int64)
                input_names.append("axes")
                initializers["axes"] = axes_array
                # Create model with 2 inputs (data + axes)
                onnx_model = create_onnx_model(
                    op_type="Squeeze",
                    input_shapes=[input_shape, axes_array.shape],
                    input_dtypes=[dtype, onnx.TensorProto.INT64],
                    output_shapes=[expected_shape],
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="squeeze_test",
                    input_names=input_names,
                    initializers=initializers,
                )
            else:
                # No axes - only data input
                onnx_model = create_onnx_model(
                    op_type="Squeeze",
                    input_shapes=[input_shape],
                    input_dtypes=[dtype],
                    output_shapes=[expected_shape],
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="squeeze_test",
                )
        else:
            # v1-v11: axes is attribute
            # ONNX can't handle empty list attributes, so skip setting it when axes=[]
            if axes is not None and len(axes) > 0:
                attrs["axes"] = axes
            # If axes=[] or axes=None, don't set the attribute (let converter handle it)
            onnx_model = create_onnx_model(
                op_type="Squeeze",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="squeeze_test",
            )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) >= 1, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # If no size-1 dims to squeeze, expect Identity node
        if input_shape == expected_shape:
            assert (
                tir_graph.nodes[0].op_type == "Identity"
            ), f"Expected Identity node when no squeeze needed, got {tir_graph.nodes[0].op_type}"
        else:
            # Should have Squeeze nodes
            squeeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Squeeze"]
            assert (
                len(squeeze_nodes) >= 1
            ), f"Expected at least 1 Squeeze node, got {[n.op_type for n in tir_graph.nodes]}"

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
            f"axes={axes}, dtype={dtype}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n"
            f"Test params: opset={opset_version}, input_shape={input_shape}, "
            f"axes={axes}, dtype={dtype}"
        )

        logger.info(
            f"✓ Squeeze basic test passed: opset={opset_version}, "
            f"input_shape={input_shape}, axes={axes}, dtype={dtype}"
        )

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    def test_squeeze_negative_indices(self, opset_version):
        """Test Squeeze with negative indices (v11+)."""
        if opset_version < 11:
            pytest.skip(f"Negative indices not supported in opset {opset_version}")

        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        input_shape = (1, 3, 1, 5)
        dtype = onnx.TensorProto.FLOAT

        # Test cases with negative indices
        # For input shape (1, 3, 1, 5):
        #   -4 (dim 0): size 1, can squeeze
        #   -3 (dim 1): size 3, cannot squeeze
        #   -2 (dim 2): size 1, can squeeze
        #   -1 (dim 3): size 5, cannot squeeze
        test_cases = [
            ([-2], (1, 3, 5)),  # Squeeze second-to-last (dim 2, size 1)
            ([-4], (3, 1, 5)),  # Squeeze first dim (dim 0, size 1)
            ([-4, -2], (3, 5)),  # Squeeze first and second-to-last (dims 0 and 2, both size 1)
        ]

        for axes, expected_shape in test_cases:
            attrs = {}
            initializers = {}

            if opset_version >= 13:
                axes_array = np.array(axes, dtype=np.int64)
                initializers["axes"] = axes_array
                onnx_model = create_onnx_model(
                    op_type="Squeeze",
                    input_shapes=[input_shape, axes_array.shape],
                    input_dtypes=[dtype, onnx.TensorProto.INT64],
                    output_shapes=[expected_shape],
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="squeeze_neg",
                    input_names=["input_0", "axes"],
                    initializers=initializers,
                )
            else:
                # For opset < 13, axes is an attribute
                # ONNX can't handle empty list attributes, so skip setting it when axes=[]
                if axes is not None and len(axes) > 0:
                    attrs["axes"] = axes
                # If axes=[] or axes=None, don't set the attribute (let converter handle it)
                onnx_model = create_onnx_model(
                    op_type="Squeeze",
                    input_shapes=[input_shape],
                    input_dtypes=[dtype],
                    output_shapes=[expected_shape],
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="squeeze_neg",
                )

            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)

            # Verify nodes use positive indices (normalized)
            squeeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Squeeze"]
            for node in squeeze_nodes:
                dim = node.attrs.get("dim")
                assert dim is not None, "SqueezeNode should have dim attribute"
                # Handle both int and tuple/list cases
                if isinstance(dim, (tuple, list)):
                    assert all(d >= 0 for d in dim), f"All dims should be non-negative after normalization, got {dim}"
                else:
                    assert dim >= 0, f"dim should be non-negative after normalization, got {dim}"

            input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

            assert len(comparison["errors"]) == 0, (
                f"Comparison errors: {comparison['errors']}\n" f"Test params: opset={opset_version}, axes={axes}"
            )
            assert all(comparison["matches"].values()), (
                f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, axes={axes}"
            )

            logger.info(f"✓ Squeeze negative indices test passed: opset={opset_version}, axes={axes}")

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    def test_squeeze_multiple_axes(self, opset_version):
        """Test Squeeze with multiple axes (should create multiple SqueezeNode operations)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        input_shape = (1, 2, 1, 3, 1)
        axes = [0, 2, 4]  # Squeeze first, third, and fifth dims
        expected_shape = (2, 3)
        dtype = onnx.TensorProto.FLOAT

        attrs = {}
        initializers = {}

        if opset_version >= 13:
            axes_array = np.array(axes, dtype=np.int64)
            initializers["axes"] = axes_array
            onnx_model = create_onnx_model(
                op_type="Squeeze",
                input_shapes=[input_shape, axes_array.shape],
                input_dtypes=[dtype, onnx.TensorProto.INT64],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="squeeze_multi",
                input_names=["input_0", "axes"],
                initializers=initializers,
            )
        else:
            # For opset < 13, axes is an attribute
            # ONNX can't handle empty list attributes, so skip setting it when axes=[]
            if axes is not None and len(axes) > 0:
                attrs["axes"] = axes
            # If axes=[] or axes=None, don't set the attribute (let converter handle it)
            onnx_model = create_onnx_model(
                op_type="Squeeze",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="squeeze_multi",
            )

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Should have a single Squeeze node with tuple of dims
        squeeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Squeeze"]
        assert len(squeeze_nodes) == 1, (
            f"Expected 1 Squeeze node, got {len(squeeze_nodes)}. " f"Nodes: {[n.op_type for n in tir_graph.nodes]}"
        )

        # Verify the node has tuple of dims for multiple axes
        squeeze_node = squeeze_nodes[0]
        dim = squeeze_node.attrs.get("dim")
        if len(axes) > 1:
            assert isinstance(dim, (tuple, list)), f"Expected tuple/list of dims for multiple axes, got {type(dim)}"
            assert len(dim) == len(axes), f"Expected {len(axes)} dims, got {len(dim)}"
        else:
            assert isinstance(dim, int), f"Expected int dim for single axis, got {type(dim)}"

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

        logger.info(f"✓ Squeeze multiple axes test passed: opset={opset_version}")

    @pytest.mark.parametrize("opset_version", [13, 21, 23, 24, 25])
    def test_squeeze_no_axes_input(self, opset_version):
        """Test Squeeze v13+ with no axes input (should squeeze all size-1 dims)."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        input_shape = (1, 3, 1, 5)
        expected_shape = (3, 5)  # All size-1 dims removed
        dtype = onnx.TensorProto.FLOAT

        # Create model with only data input (no axes input)
        onnx_model = create_onnx_model(
            op_type="Squeeze",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[expected_shape],
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name="squeeze_no_axes",
        )

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Should have 1 Squeeze node with tuple of dims [0, 2] (auto-detected size-1 dims)
        squeeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Squeeze"]
        assert len(squeeze_nodes) == 1, f"Expected 1 Squeeze node, got {len(squeeze_nodes)}"

        # Verify the node has tuple of dims
        squeeze_node = squeeze_nodes[0]
        dim = squeeze_node.attrs.get("dim")
        assert isinstance(dim, (tuple, list)), f"Expected tuple/list of dims, got {type(dim)}"
        assert set(dim) == {0, 2}, f"Expected dims [0, 2], got {dim}"

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), f"Output mismatch: {comparison}"

        logger.info(f"✓ Squeeze no axes input test passed: opset={opset_version}")

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    def test_squeeze_edge_cases(self, opset_version):
        """Test Squeeze edge cases."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        test_cases = [
            # Scalar output
            ((1, 1, 1), None, ()),
            ((1, 1, 1), [0, 1, 2], ()),
            # No-op (no size-1 dims)
            ((2, 3, 4), None, (2, 3, 4)),
            ((2, 3, 4), [], (2, 3, 4)),
            # Single element
            ((1,), None, ()),
            ((1,), [0], ()),
            # High dimensional
            ((1, 2, 1, 3, 1, 4), [0, 2, 4], (2, 3, 4)),
            ((1, 2, 1, 3, 1, 4), None, (2, 3, 4)),
        ]

        dtype = onnx.TensorProto.FLOAT

        for input_shape, axes, expected_shape in test_cases:
            attrs = {}
            initializers = {}

            if opset_version >= 13:
                if axes is not None and len(axes) > 0:
                    axes_array = np.array(axes, dtype=np.int64)
                    initializers["axes"] = axes_array
                    onnx_model = create_onnx_model(
                        op_type="Squeeze",
                        input_shapes=[input_shape, axes_array.shape],
                        input_dtypes=[dtype, onnx.TensorProto.INT64],
                        output_shapes=[expected_shape],
                        output_dtypes=[dtype],
                        attrs=attrs,
                        opset_version=opset_version,
                        node_name="squeeze_edge",
                        input_names=["input_0", "axes"],
                        initializers=initializers,
                    )
                else:
                    onnx_model = create_onnx_model(
                        op_type="Squeeze",
                        input_shapes=[input_shape],
                        input_dtypes=[dtype],
                        output_shapes=[expected_shape],
                        output_dtypes=[dtype],
                        attrs=attrs,
                        opset_version=opset_version,
                        node_name="squeeze_edge",
                    )
            else:
                # For opset < 13, axes is an attribute
                # ONNX can't handle empty list attributes, so skip setting it when axes=[]
                if axes is not None and len(axes) > 0:
                    attrs["axes"] = axes
                # If axes=[] or axes=None, don't set the attribute (let converter handle it)
                onnx_model = create_onnx_model(
                    op_type="Squeeze",
                    input_shapes=[input_shape],
                    input_dtypes=[dtype],
                    output_shapes=[expected_shape],
                    output_dtypes=[dtype],
                    attrs=attrs,
                    opset_version=opset_version,
                    node_name="squeeze_edge",
                )

            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)

            # For no-op case, expect Identity node
            if input_shape == expected_shape:
                assert (
                    tir_graph.nodes[0].op_type == "Identity"
                ), f"Expected Identity node for no-op, got {tir_graph.nodes[0].op_type}"
            else:
                squeeze_nodes = [n for n in tir_graph.nodes if n.op_type == "Squeeze"]
                assert (
                    len(squeeze_nodes) >= 1
                ), f"Expected at least 1 Squeeze node, got {[n.op_type for n in tir_graph.nodes]}"

            # Create input data (handle scalar output case)
            if len(input_shape) > 0:
                input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}
            else:
                input_data = {"input_0": np.array(1.0, dtype=np.float32)}

            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-5, atol=1e-6)

            assert len(comparison["errors"]) == 0, (
                f"Comparison errors: {comparison['errors']}\n" f"Test params: input_shape={input_shape}, axes={axes}"
            )
            assert all(comparison["matches"].values()), (
                f"Output mismatch: {comparison}\n" f"Test params: input_shape={input_shape}, axes={axes}"
            )

            logger.info(
                f"✓ Squeeze edge case test passed: input_shape={input_shape}, "
                f"axes={axes}, expected_shape={expected_shape}"
            )

    @pytest.mark.parametrize("opset_version", [1, 11, 13, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
            onnx.TensorProto.BOOL,
        ],
    )
    def test_squeeze_all_dtypes(self, opset_version, dtype):
        """Test Squeeze with all supported dtypes."""
        if opset_version in [24, 25]:
            pytest.skip(f"Opset {opset_version} not supported by ONNXRuntime (max supported: 23)")

        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
            pytest.skip(f"Opset 1 only supports float types, got dtype={dtype}")

        input_shape = (1, 3, 1, 5)
        axes = [0, 2]
        expected_shape = (3, 5)

        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
            onnx.TensorProto.BOOL: np.bool_,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        attrs = {"axes": axes} if opset_version < 13 else {}
        initializers = {}

        if opset_version >= 13:
            axes_array = np.array(axes, dtype=np.int64)
            initializers["axes"] = axes_array
            onnx_model = create_onnx_model(
                op_type="Squeeze",
                input_shapes=[input_shape, axes_array.shape],
                input_dtypes=[dtype, onnx.TensorProto.INT64],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="squeeze_dtype",
                input_names=["input_0", "axes"],
                initializers=initializers,
            )
        else:
            onnx_model = create_onnx_model(
                op_type="Squeeze",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[expected_shape],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="squeeze_dtype",
            )

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        if dtype == onnx.TensorProto.BOOL:
            input_data = {"input_0": np.random.choice([True, False], size=input_shape).astype(np_dtype)}
            rtol, atol = 0, 0
        elif dtype in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
            input_data = {"input_0": np.random.randint(1, 100, size=input_shape, dtype=np_dtype)}
            rtol, atol = 0, 0
        else:
            input_data = {"input_0": np.random.randn(*input_shape).astype(np_dtype)}
            rtol, atol = 1e-5, 1e-6

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n" f"Test params: opset={opset_version}, dtype={dtype}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, dtype={dtype}"
        )

        logger.info(f"✓ Squeeze all dtypes test passed: opset={opset_version}, dtype={dtype}")
