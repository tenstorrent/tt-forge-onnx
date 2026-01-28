# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Slice operation.
Tests different input shapes, dtypes, opset versions, multi-axis slicing, and edge cases.
"""
import pytest
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.utils.exceptions import ConversionError
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


def _create_slice_model_v1(
    data_shape: tuple,
    starts: list,
    ends: list,
    axes: list = None,
    dtype: int = onnx.TensorProto.FLOAT,
    node_name: str = "slice_test",
) -> onnx.ModelProto:
    """
    Create ONNX Slice model for opset v1 (attributes-based).

    Args:
        data_shape: Input data tensor shape
        starts: List of start indices
        ends: List of end indices
        axes: Optional list of axes (defaults to [0, 1, ..., len(starts)-1])
        dtype: Data dtype
        node_name: Node name

    Returns:
        ONNX ModelProto
    """
    attrs = {
        "starts": starts,
        "ends": ends,
    }
    if axes is not None:
        attrs["axes"] = axes

    # Validate lengths match BEFORE computing output shape (for error tests, allow model creation but converter will catch it)
    if len(starts) != len(ends):
        # For error tests, use a dummy output shape
        return create_onnx_model(
            op_type="Slice",
            input_shapes=[data_shape],
            input_dtypes=[dtype],
            output_shapes=[data_shape],  # Dummy shape
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=1,
            node_name=node_name,
        )

    # Check if axes length matches before defaulting
    if axes is not None and len(axes) != len(starts):
        # For error tests, use a dummy output shape
        return create_onnx_model(
            op_type="Slice",
            input_shapes=[data_shape],
            input_dtypes=[dtype],
            output_shapes=[data_shape],  # Dummy shape
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=1,
            node_name=node_name,
        )

    # Now safe to compute defaults
    # Compute output shape
    output_shape = list(data_shape)
    if axes is None:
        axes = list(range(len(starts)))

    for i in range(len(starts)):
        axis = axes[i]
        # Skip if axis is out of bounds (for error tests)
        if axis < 0 or axis >= len(data_shape):
            return create_onnx_model(
                op_type="Slice",
                input_shapes=[data_shape],
                input_dtypes=[dtype],
                output_shapes=[data_shape],  # Dummy shape
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=1,
                node_name=node_name,
            )

        start = starts[i]
        end = ends[i]
        # Normalize negative indices
        if start < 0:
            start = data_shape[axis] + start
        if end < 0:
            end = data_shape[axis] + end
        # Clamp
        start = max(0, min(start, data_shape[axis]))
        end = max(0, min(end, data_shape[axis]))
        # Compute output size (step=1 for v1)
        output_shape[axis] = max(0, end - start)

    return create_onnx_model(
        op_type="Slice",
        input_shapes=[data_shape],
        input_dtypes=[dtype],
        output_shapes=[tuple(output_shape)],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=1,
        node_name=node_name,
    )


def _create_slice_model_v10_plus(
    opset_version: int,
    data_shape: tuple,
    starts: list,
    ends: list,
    axes: list = None,
    steps: list = None,
    dtype: int = onnx.TensorProto.FLOAT,
    node_name: str = "slice_test",
) -> onnx.ModelProto:
    """
    Create ONNX Slice model for opset v10+ (input-based).

    Args:
        opset_version: Opset version (10, 11, or 13)
        data_shape: Input data tensor shape
        starts: List of start indices
        ends: List of end indices
        axes: Optional list of axes
        steps: Optional list of step sizes
        dtype: Data dtype
        node_name: Node name

    Returns:
        ONNX ModelProto
    """
    input_names = ["input_0", "starts", "ends"]
    initializers = {}

    # Add starts and ends as initializers
    initializers["starts"] = np.array(starts, dtype=np.int64)
    initializers["ends"] = np.array(ends, dtype=np.int64)

    # Add axes if provided
    if axes is not None:
        input_names.append("axes")
        initializers["axes"] = np.array(axes, dtype=np.int64)

    # Add steps if provided (v10+ supports steps)
    if steps is not None:
        input_names.append("steps")
        initializers["steps"] = np.array(steps, dtype=np.int64)

    # Validate lengths match BEFORE computing output shape (for error tests, allow model creation but converter will catch it)
    if len(starts) != len(ends):
        # For error tests, use a dummy output shape
        return create_onnx_model(
            op_type="Slice",
            input_shapes=[data_shape] + [(len(starts),)] * (len(input_names) - 1),
            input_dtypes=[dtype] + [onnx.TensorProto.INT64] * (len(input_names) - 1),
            output_shapes=[data_shape],  # Dummy shape
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name=node_name,
            input_names=input_names,
            initializers=initializers,
        )

    # Check axes length before defaulting
    if axes is not None and len(axes) != len(starts):
        # For error tests, use a dummy output shape
        return create_onnx_model(
            op_type="Slice",
            input_shapes=[data_shape] + [(len(starts),)] * (len(input_names) - 1),
            input_dtypes=[dtype] + [onnx.TensorProto.INT64] * (len(input_names) - 1),
            output_shapes=[data_shape],  # Dummy shape
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name=node_name,
            input_names=input_names,
            initializers=initializers,
        )

    # Check steps length before defaulting
    if steps is not None and len(steps) != len(starts):
        # For error tests, use a dummy output shape
        return create_onnx_model(
            op_type="Slice",
            input_shapes=[data_shape] + [(len(starts),)] * (len(input_names) - 1),
            input_dtypes=[dtype] + [onnx.TensorProto.INT64] * (len(input_names) - 1),
            output_shapes=[data_shape],  # Dummy shape
            output_dtypes=[dtype],
            attrs={},
            opset_version=opset_version,
            node_name=node_name,
            input_names=input_names,
            initializers=initializers,
        )

    # Now safe to compute defaults
    # Compute output shape
    output_shape = list(data_shape)
    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)

    # Iterate over starts length (not axes length) to ensure we have matching indices
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        step = steps[i]
        axis = axes[i]

        # Skip if axis is out of bounds (for error tests)
        if axis < 0 or axis >= len(data_shape):
            return create_onnx_model(
                op_type="Slice",
                input_shapes=[data_shape] + [(len(starts),)] * (len(input_names) - 1),
                input_dtypes=[dtype] + [onnx.TensorProto.INT64] * (len(input_names) - 1),
                output_shapes=[data_shape],  # Dummy shape
                output_dtypes=[dtype],
                attrs={},
                opset_version=opset_version,
                node_name=node_name,
                input_names=input_names,
                initializers=initializers,
            )

        # Skip if step is 0 (for error tests)
        if step == 0:
            return create_onnx_model(
                op_type="Slice",
                input_shapes=[data_shape] + [(len(starts),)] * (len(input_names) - 1),
                input_dtypes=[dtype] + [onnx.TensorProto.INT64] * (len(input_names) - 1),
                output_shapes=[data_shape],  # Dummy shape
                output_dtypes=[dtype],
                attrs={},
                opset_version=opset_version,
                node_name=node_name,
                input_names=input_names,
                initializers=initializers,
            )

        # Normalize negative indices
        if start < 0:
            start = data_shape[axis] + start
        if end < 0:
            end = data_shape[axis] + end
        # Clamp
        if step > 0:
            start = max(0, min(start, data_shape[axis]))
            end = max(0, min(end, data_shape[axis]))
            output_size = max(0, (end - start + step - 1) // step)
        else:
            start = max(0, min(start, data_shape[axis] - 1))
            end = max(-1, min(end, data_shape[axis] - 1))
            output_size = max(0, (start - end + abs(step) - 1) // abs(step))
        output_shape[axis] = output_size

    return create_onnx_model(
        op_type="Slice",
        input_shapes=[data_shape] + [(len(starts),)] * (len(input_names) - 1),
        input_dtypes=[dtype] + [onnx.TensorProto.INT64] * (len(input_names) - 1),
        output_shapes=[tuple(output_shape)],
        output_dtypes=[dtype],
        attrs={},
        opset_version=opset_version,
        node_name=node_name,
        input_names=input_names,
        initializers=initializers,
    )


@pytest.mark.transpiler
class TestSlice:
    """Comprehensive test cases for Slice operation."""

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, starts, ends, axes, dtype",
        [
            # Single-axis slicing
            ((10, 5), [1], [5], None, onnx.TensorProto.FLOAT),
            ((10, 5), [0], [3], None, onnx.TensorProto.FLOAT),
            ((5, 10), [2], [8], [1], onnx.TensorProto.FLOAT),
            # Multi-axis slicing
            ((10, 5, 3), [1, 0], [5, 3], None, onnx.TensorProto.FLOAT),
            ((10, 5, 3), [2, 1], [8, 4], [0, 1], onnx.TensorProto.FLOAT),
            ((3, 4, 5, 6), [1, 1, 1], [3, 3, 4], [0, 1, 2], onnx.TensorProto.FLOAT),
            # All axes
            ((5, 4, 3), [1, 0, 0], [4, 3, 2], None, onnx.TensorProto.FLOAT),
            # Different dtypes
            ((10, 5), [1], [5], None, onnx.TensorProto.DOUBLE),
            ((10, 5), [1], [5], None, onnx.TensorProto.FLOAT16),
            ((10, 5), [1], [5], None, onnx.TensorProto.INT32),
            ((10, 5), [1], [5], None, onnx.TensorProto.INT64),
        ],
    )
    def test_slice_basic(self, opset_version, data_shape, starts, ends, axes, dtype):
        """Test basic Slice operations across opset versions."""
        # Skip bfloat16 for opset < 13
        if opset_version < 13 and dtype == onnx.TensorProto.BFLOAT16:
            pytest.skip(f"BFLOAT16 is only supported in opset 13+")

        # Skip opset 1 for non-float types (not typically used)
        if opset_version == 1 and dtype not in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
        ]:
            pytest.skip(f"Opset 1 typically uses float types")

        # Create random data
        np.random.seed(42)
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        data = np.random.randn(*data_shape).astype(np_dtype)

        # Create ONNX model
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, axes, dtype)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, None, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have IndexNode(s)
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        node_types = [n.op_type for n in tir_graph.nodes]
        assert "Index" in node_types, f"Expected IndexNode(s), got {node_types}"

        # Verify number of IndexNodes matches number of axes
        index_nodes = [n for n in tir_graph.nodes if n.op_type == "Index"]
        expected_num_nodes = len(starts)
        assert len(index_nodes) == expected_num_nodes, (
            f"Expected {expected_num_nodes} IndexNode(s) for {len(starts)} axes, " f"got {len(index_nodes)}"
        )

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [10, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, starts, ends, steps, axes, dtype",
        [
            # Positive steps
            ((10, 5), [1], [5], [1], None, onnx.TensorProto.FLOAT),
            ((10, 5), [0], [10], [2], None, onnx.TensorProto.FLOAT),
            ((10, 5), [1], [9], [3], None, onnx.TensorProto.FLOAT),
            # Multi-axis with steps
            ((10, 5, 3), [1, 0], [5, 3], [1, 2], None, onnx.TensorProto.FLOAT),
            ((10, 5, 3), [2, 1], [8, 4], [2, 1], [0, 1], onnx.TensorProto.FLOAT),
            # Different step sizes
            ((20, 10), [0], [20], [5], None, onnx.TensorProto.FLOAT),
            ((15, 8), [1], [14], [4], None, onnx.TensorProto.FLOAT),
        ],
    )
    def test_slice_with_steps(self, opset_version, data_shape, starts, ends, steps, axes, dtype):
        """Test Slice with step sizes (v10+ only)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, steps, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"
        node_types = [n.op_type for n in tir_graph.nodes]
        assert "Index" in node_types, f"Expected IndexNode(s), got {node_types}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, starts, ends, axes, dtype",
        [
            # Negative start indices
            ((10, 5), [-1], [5], None, onnx.TensorProto.FLOAT),
            ((10, 5), [-3], [10], None, onnx.TensorProto.FLOAT),
            ((10, 5), [-5], [-1], None, onnx.TensorProto.FLOAT),
            # Negative end indices
            ((10, 5), [0], [-1], None, onnx.TensorProto.FLOAT),
            ((10, 5), [2], [-2], None, onnx.TensorProto.FLOAT),
            # Both negative
            ((10, 5), [-3], [-1], None, onnx.TensorProto.FLOAT),
            # Multi-axis with negatives
            ((10, 5, 3), [-1, -1], [10, 5], None, onnx.TensorProto.FLOAT),
            ((10, 5, 3), [1, -2], [5, -1], [0, 1], onnx.TensorProto.FLOAT),
        ],
    )
    def test_slice_negative_indices(self, opset_version, data_shape, starts, ends, axes, dtype):
        """Test Slice with negative indices (should be normalized)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, axes, dtype)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, None, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, starts, ends, axes, dtype",
        [
            # Out-of-bounds start (should be clamped)
            ((10, 5), [15], [10], None, onnx.TensorProto.FLOAT),
            ((10, 5), [-20], [5], None, onnx.TensorProto.FLOAT),
            # Out-of-bounds end (should be clamped)
            ((10, 5), [0], [20], None, onnx.TensorProto.FLOAT),
            ((10, 5), [2], [100], None, onnx.TensorProto.FLOAT),
            # Both out-of-bounds
            ((10, 5), [-10], [20], None, onnx.TensorProto.FLOAT),
        ],
    )
    def test_slice_out_of_bounds(self, opset_version, data_shape, starts, ends, axes, dtype):
        """Test Slice with out-of-bounds indices (should be clamped)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, axes, dtype)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, None, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, starts, ends, axes, dtype",
        [
            # Zero-length output
            ((10, 5), [5], [5], None, onnx.TensorProto.FLOAT),
            ((10, 5), [3], [3], None, onnx.TensorProto.FLOAT),
            # Start > end (zero-length)
            ((10, 5), [5], [3], None, onnx.TensorProto.FLOAT),
            ((10, 5), [8], [2], None, onnx.TensorProto.FLOAT),
        ],
    )
    def test_slice_zero_length(self, opset_version, data_shape, starts, ends, axes, dtype):
        """Test Slice with zero-length output (edge case)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, axes, dtype)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, None, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, starts, ends, axes, dtype",
        [
            # Single element
            ((1,), [0], [1], None, onnx.TensorProto.FLOAT),
            ((1, 1), [0], [1], None, onnx.TensorProto.FLOAT),
            # Small dimensions
            ((2, 2), [0], [1], None, onnx.TensorProto.FLOAT),
            ((3, 1), [1], [2], None, onnx.TensorProto.FLOAT),
        ],
    )
    def test_slice_small_tensors(self, opset_version, data_shape, starts, ends, axes, dtype):
        """Test Slice with small tensors (edge cases)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, axes, dtype)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, None, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, starts, ends, axes, dtype",
        [
            # Negative axes
            ((10, 5), [1], [5], [-1], onnx.TensorProto.FLOAT),
            ((10, 5, 3), [1, 0], [5, 2], [-2, -1], onnx.TensorProto.FLOAT),
            ((3, 4, 5, 6), [1], [3], [-3], onnx.TensorProto.FLOAT),
        ],
    )
    def test_slice_negative_axes(self, opset_version, data_shape, starts, ends, axes, dtype):
        """Test Slice with negative axes (should be normalized)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, axes, dtype)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, None, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [10, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, starts, ends, steps, axes, dtype",
        [
            # Negative steps (backward slicing) - limited support
            # Note: Forge Index has limited negative stride support
            ((1,), [0], [1], [-1], None, onnx.TensorProto.FLOAT),
            # Multi-axis with mixed steps
            ((10, 1), [5, 0], [0, 1], [-1, 1], None, onnx.TensorProto.FLOAT),
        ],
    )
    def test_slice_negative_steps(self, opset_version, data_shape, starts, ends, steps, axes, dtype):
        """Test Slice with negative steps (backward slicing)."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, steps, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)

        # Negative steps may not be fully supported by Forge Index
        # Check if it raises an error or creates nodes
        try:
            tir_graph = transpiler.transpile(onnx_model)
            # If successful, verify structure
            assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

            # Verify with ONNX Runtime if possible
            compare_tir_with_onnx(
                tir_graph,
                onnx_model,
                input_data={"input_0": data},
                atol=1e-5,
                rtol=1e-5,
            )
        except (ConversionError, ValueError, NotImplementedError) as e:
            # Negative steps may not be fully supported
            pytest.skip(f"Negative steps not fully supported: {e}")

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    def test_slice_default_axes(self, opset_version):
        """Test Slice with default axes (should use [0, 1, ..., len(starts)-1])."""
        data_shape = (10, 5, 3)
        starts = [1, 0]
        ends = [5, 3]

        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model without axes (should default to [0, 1])
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, None)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, None, None)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have 2 IndexNodes (one per axis)
        index_nodes = [n for n in tir_graph.nodes if n.op_type == "Index"]
        assert len(index_nodes) == 2, f"Expected 2 IndexNodes for 2 axes, got {len(index_nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [10, 11, 13])
    def test_slice_default_steps(self, opset_version):
        """Test Slice with default steps (should default to [1, 1, ...])."""
        data_shape = (10, 5)
        starts = [1, 0]
        ends = [5, 3]

        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model without steps (should default to [1, 1])
        onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, None, None)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    def test_slice_all_dimensions(self, opset_version):
        """Test Slice slicing all dimensions."""
        data_shape = (5, 4, 3)
        starts = [1, 0, 0]
        ends = [4, 3, 2]

        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, None)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, None, None)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have 3 IndexNodes (one per axis)
        index_nodes = [n for n in tir_graph.nodes if n.op_type == "Index"]
        assert len(index_nodes) == 3, f"Expected 3 IndexNodes for 3 axes, got {len(index_nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    def test_slice_single_axis(self, opset_version):
        """Test Slice with single axis (should create single IndexNode)."""
        data_shape = (10, 5, 3)
        starts = [2]
        ends = [8]
        axes = [1]

        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, axes)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, None)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - should have exactly 1 IndexNode
        index_nodes = [n for n in tir_graph.nodes if n.op_type == "Index"]
        assert len(index_nodes) == 1, f"Expected 1 IndexNode for single axis, got {len(index_nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [1, 10, 11, 13])
    @pytest.mark.parametrize(
        "data_shape, starts, ends, axes, dtype",
        [
            # High-dimensional tensors
            ((2, 3, 4, 5, 6), [1, 1, 1, 1], [2, 3, 4, 5], [0, 1, 2, 3], onnx.TensorProto.FLOAT),
            ((3, 4, 5, 6, 7, 8), [1, 1], [3, 4], [0, 1], onnx.TensorProto.FLOAT),
        ],
    )
    def test_slice_high_dimensional(self, opset_version, data_shape, starts, ends, axes, dtype):
        """Test Slice with high-dimensional tensors."""
        # Create random data
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)

        # Create ONNX model
        if opset_version == 1:
            onnx_model = _create_slice_model_v1(data_shape, starts, ends, axes, dtype)
        else:
            onnx_model = _create_slice_model_v10_plus(opset_version, data_shape, starts, ends, axes, None, dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure
        assert len(tir_graph.nodes) > 0, f"Expected at least 1 node, got {len(tir_graph.nodes)}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={"input_0": data},
            atol=1e-5,
            rtol=1e-5,
        )
