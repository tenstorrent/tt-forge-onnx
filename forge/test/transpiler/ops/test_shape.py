# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Shape operation.
Tests different input shapes, dtypes, opset versions, shape slicing, and edge cases.
"""
import pytest
import numpy as np
import onnx
import onnx.helper
import torch

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.utils.exceptions import ConversionError
from forge.transpiler.frontends.onnx.utils.validation import ConverterValidationError
from forge.transpiler.utils.exceptions import ONNXModelValidationError
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
)


def _get_shape_constant(tir_graph, output_name):
    """Get shape constant from either tir_graph.constants or tir_graph.computed_constants.

    Depending on the opset and ONNX shape inference, Shape outputs may be stored in either:
    - tir_graph.constants: when ONNX inference folds the Shape op into a graph initializer
    - tir_graph.computed_constants: when the ShapeConverter eagerly computes a ConstantResult
    """
    if output_name in tir_graph.constants:
        return tir_graph.constants[output_name]
    if output_name in tir_graph.computed_constants:
        return tir_graph.computed_constants[output_name]
    return None


def _create_shape_model(
    opset_version: int,
    input_shape: tuple,
    input_dtype: int = onnx.TensorProto.FLOAT,
    start: int = None,
    end: int = None,
    node_name: str = "shape_test",
) -> onnx.ModelProto:
    """
    Create ONNX Shape model.

    Args:
        opset_version: Opset version (1+)
        input_shape: Input tensor shape
        input_dtype: Input tensor dtype
        start: Optional start attribute (v13+)
        end: Optional end attribute (v13+)
        node_name: Node name

    Returns:
        ONNX ModelProto
    """
    # Calculate expected output shape
    # Output is always 1D int64 tensor with shape values
    if start is not None or end is not None:
        # Shape slicing (v13+)
        r = len(input_shape)
        start_val = start if start is not None else 0
        end_val = end if end is not None else r

        # Normalize negative indices
        if start_val < 0:
            start_val = start_val + r
        if end_val is not None and end_val < 0:
            end_val = end_val + r

        # Clamp to valid range
        start_val = max(0, min(start_val, r))
        if end_val is not None:
            end_val = max(0, min(end_val, r))
        else:
            end_val = r

        # Extract slice
        if start_val >= end_val:
            output_shape = (0,)
        else:
            output_shape = (end_val - start_val,)
    else:
        # Full shape
        output_shape = (len(input_shape),)

    # Prepare attributes
    # Note: start/end attributes are supported in opset 15+, not 13
    # Opset 13 Shape operation does not support start/end attributes
    attrs = {}
    if opset_version >= 15:
        if start is not None:
            attrs["start"] = start
        if end is not None:
            attrs["end"] = end

    # Create model
    onnx_model = create_onnx_model(
        op_type="Shape",
        input_shapes=[input_shape],
        input_dtypes=[input_dtype],
        output_shapes=[output_shape],
        output_dtypes=[onnx.TensorProto.INT64],
        attrs=attrs,
        opset_version=opset_version,
        node_name=node_name,
    )

    return onnx_model


@pytest.mark.transpiler
class TestShape:
    """Comprehensive test cases for Shape operation."""

    @pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_shape, input_dtype",
        [
            # 1D tensors
            ((5,), onnx.TensorProto.FLOAT),
            ((10,), onnx.TensorProto.FLOAT),
            ((1,), onnx.TensorProto.FLOAT),
            # 2D tensors
            ((3, 4), onnx.TensorProto.FLOAT),
            ((5, 5), onnx.TensorProto.FLOAT),
            ((2, 10), onnx.TensorProto.FLOAT),
            # 3D tensors
            ((2, 3, 4), onnx.TensorProto.FLOAT),
            ((1, 1, 1), onnx.TensorProto.FLOAT),
            # 4D tensors
            ((2, 3, 4, 5), onnx.TensorProto.FLOAT),
            # Different dtypes
            ((3, 4), onnx.TensorProto.DOUBLE),
            ((3, 4), onnx.TensorProto.INT32),
            ((3, 4), onnx.TensorProto.INT64),
            ((3, 4), onnx.TensorProto.BOOL),
        ],
    )
    def test_shape_basic(self, opset_version, input_shape, input_dtype):
        """Test basic Shape operations across opset versions with various shapes and dtypes."""
        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape, input_dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify structure - Shape should create a constant, not a node.
        # Depending on opset and ONNX inference, the constant may be in either
        # tir_graph.constants (folded by ONNX) or tir_graph.computed_constants (ShapeConverter).
        output_name = onnx_model.graph.output[0].name

        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, (
            f"Shape constant '{output_name}' not found in tir_graph.constants or computed_constants. "
            f"constants={list(tir_graph.constants.keys())}, "
            f"computed_constants={list(tir_graph.computed_constants.keys())}"
        )

        # Verify it's a torch tensor
        assert isinstance(shape_constant, torch.Tensor), f"Expected torch.Tensor, got {type(shape_constant)}"

        # Verify dtype is int64
        assert shape_constant.dtype == torch.int64, f"Expected int64, got {shape_constant.dtype}"

        # Verify shape matches input shape
        expected_shape = torch.tensor(input_shape, dtype=torch.int64)
        assert torch.equal(
            shape_constant, expected_shape
        ), f"Shape mismatch: expected {expected_shape.tolist()}, got {shape_constant.tolist()}"

        # Compare with ONNX Runtime if available
        try:
            input_data = {onnx_model.graph.input[0].name: np.random.randn(*input_shape).astype(np.float32)}
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)

            # For Shape, we need to check the constant value matches ONNX output
            if "onnx_outputs" in comparison and output_name in comparison["onnx_outputs"]:
                onnx_output = comparison["onnx_outputs"][output_name]
                assert np.array_equal(shape_constant.numpy(), onnx_output), (
                    f"Shape constant doesn't match ONNX output: "
                    f"expected {onnx_output}, got {shape_constant.numpy()}"
                )
        except Exception as e:
            # ONNX Runtime comparison is optional
            pytest.skip(f"ONNX Runtime comparison skipped: {e}")

    @pytest.mark.parametrize("opset_version", [15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_shape, start, expected_output",
        [
            # Basic slicing
            ((2, 3, 4), 0, [2, 3, 4]),
            ((2, 3, 4), 1, [3, 4]),
            ((2, 3, 4), 2, [4]),
            ((2, 3, 4), 3, []),
            # Negative start
            ((2, 3, 4), -1, [4]),
            ((2, 3, 4), -2, [3, 4]),
            ((2, 3, 4), -3, [2, 3, 4]),
            # High-dimensional
            ((2, 3, 4, 5), 1, [3, 4, 5]),
            ((2, 3, 4, 5), -2, [4, 5]),
        ],
    )
    def test_shape_with_start(self, opset_version, input_shape, start, expected_output):
        """Test Shape operation with start attribute (v15+)."""
        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape, start=start)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor(expected_output, dtype=torch.int64)

        assert torch.equal(
            shape_constant, expected
        ), f"Shape with start={start} mismatch: expected {expected.tolist()}, got {shape_constant.tolist()}"

    @pytest.mark.parametrize("opset_version", [15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_shape, end, expected_output",
        [
            # Basic slicing
            ((2, 3, 4), 1, [2]),
            ((2, 3, 4), 2, [2, 3]),
            ((2, 3, 4), 3, [2, 3, 4]),
            # Negative end
            ((2, 3, 4), -1, [2, 3]),
            ((2, 3, 4), -2, [2]),
            ((2, 3, 4), -3, []),
            # High-dimensional
            ((2, 3, 4, 5), 2, [2, 3]),
            ((2, 3, 4, 5), -1, [2, 3, 4]),
        ],
    )
    def test_shape_with_end(self, opset_version, input_shape, end, expected_output):
        """Test Shape operation with end attribute (v15+)."""
        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape, end=end)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor(expected_output, dtype=torch.int64)

        assert torch.equal(
            shape_constant, expected
        ), f"Shape with end={end} mismatch: expected {expected.tolist()}, got {shape_constant.tolist()}"

    @pytest.mark.parametrize("opset_version", [15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_shape, start, end, expected_output",
        [
            # Basic slicing
            ((2, 3, 4), 0, 1, [2]),
            ((2, 3, 4), 1, 2, [3]),
            ((2, 3, 4), 1, 3, [3, 4]),
            ((2, 3, 4), 0, 2, [2, 3]),
            # Negative indices
            ((2, 3, 4), -2, -1, [3]),
            ((2, 3, 4), -3, -1, [2, 3]),
            ((2, 3, 4), -2, None, [3, 4]),  # end=None means to the end
            # Edge cases
            ((2, 3, 4), 1, 1, []),  # Empty slice
            ((2, 3, 4), 2, 1, []),  # start > end
            # High-dimensional
            ((2, 3, 4, 5), 1, 3, [3, 4]),
            ((2, 3, 4, 5), -3, -1, [3, 4]),
        ],
    )
    def test_shape_with_start_and_end(self, opset_version, input_shape, start, end, expected_output):
        """Test Shape operation with both start and end attributes (v15+)."""
        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape, start=start, end=end)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor(expected_output, dtype=torch.int64)

        assert torch.equal(shape_constant, expected), (
            f"Shape with start={start}, end={end} mismatch: "
            f"expected {expected.tolist()}, got {shape_constant.tolist()}"
        )

    @pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 15, 19, 21, 23, 25])
    def test_shape_scalar(self, opset_version):
        """Test Shape operation with scalar input."""
        input_shape = ()  # Scalar

        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor([], dtype=torch.int64)  # Empty tensor for scalar

        assert torch.equal(
            shape_constant, expected
        ), f"Scalar shape mismatch: expected {expected.tolist()}, got {shape_constant.tolist()}"

    @pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (1,),
            (100,),
            (1, 1),
            (1, 1, 1),
            (100, 100),
            (10, 20, 30),
        ],
    )
    def test_shape_single_element(self, opset_version, input_shape):
        """Test Shape operation with single-element tensors."""
        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor(input_shape, dtype=torch.int64)

        assert torch.equal(
            shape_constant, expected
        ), f"Single-element shape mismatch: expected {expected.tolist()}, got {shape_constant.tolist()}"

    @pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (5,),
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6),
        ],
    )
    def test_shape_high_dimensional(self, opset_version, input_shape):
        """Test Shape operation with high-dimensional tensors."""
        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor(input_shape, dtype=torch.int64)

        assert torch.equal(
            shape_constant, expected
        ), f"High-dimensional shape mismatch: expected {expected.tolist()}, got {shape_constant.tolist()}"

    @pytest.mark.parametrize("opset_version", [15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_shape, start, end",
        [
            # Out-of-range indices (should be clamped)
            ((2, 3, 4), 10, None),  # start > rank
            ((2, 3, 4), -10, None),  # start < -rank
            ((2, 3, 4), None, 10),  # end > rank
            ((2, 3, 4), None, -10),  # end < -rank
            ((2, 3, 4), 10, 20),  # Both out of range
        ],
    )
    def test_shape_out_of_range_indices(self, opset_version, input_shape, start, end):
        """Test Shape operation with out-of-range indices (should be clamped)."""
        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape, start=start, end=end)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant exists (clamping should prevent errors)
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"
        assert isinstance(shape_constant, torch.Tensor)
        assert shape_constant.dtype == torch.int64

    @pytest.mark.parametrize("opset_version", [15, 19, 21, 23, 25])
    def test_shape_empty_slice(self, opset_version):
        """Test Shape operation with empty slice (start >= end)."""
        input_shape = (2, 3, 4)

        # Create ONNX model with start >= end (should produce empty shape)
        onnx_model = _create_shape_model(opset_version, input_shape, start=2, end=1)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant is empty
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor([], dtype=torch.int64)

        assert torch.equal(
            shape_constant, expected
        ), f"Empty slice mismatch: expected {expected.tolist()}, got {shape_constant.tolist()}"

    @pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_dtype",
        [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
            onnx.TensorProto.BOOL,
        ],
    )
    def test_shape_different_dtypes(self, opset_version, input_dtype):
        """Test Shape operation with different input dtypes."""
        input_shape = (2, 3, 4)

        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape, input_dtype=input_dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant (should be same regardless of input dtype)
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor(input_shape, dtype=torch.int64)

        assert torch.equal(shape_constant, expected), (
            f"Shape with dtype {input_dtype} mismatch: " f"expected {expected.tolist()}, got {shape_constant.tolist()}"
        )

    def test_shape_error_no_input(self):
        """Test that Shape operation raises error when no input is provided."""
        # create_onnx_model only warns for invalid models, doesn't raise
        # The model will be created but will fail during transpilation
        onnx_model = create_onnx_model(
            op_type="Shape",
            input_shapes=[],  # Empty - invalid
            input_dtypes=[],
            output_shapes=[(3,)],
            output_dtypes=[onnx.TensorProto.INT64],
            opset_version=13,
        )

        # Transpilation should fail because Shape requires an input
        # Model validation will catch this and raise ONNXModelValidationError
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        with pytest.raises((ONNXModelValidationError, ConversionError, ConverterValidationError)):
            transpiler.transpile(onnx_model)

    @pytest.mark.parametrize("opset_version", [15, 19, 21, 23, 25])
    def test_shape_constant_reusable(self, opset_version):
        """Test that Shape constant can be accessed by other nodes."""
        input_shape = (2, 3, 4)

        # Create ONNX model with Shape
        onnx_model = _create_shape_model(opset_version, input_shape)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant exists in either constants or computed_constants
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"
        assert isinstance(shape_constant, torch.Tensor)

        # Verify constant value
        expected = torch.tensor(input_shape, dtype=torch.int64)
        assert torch.equal(shape_constant, expected)

        # Verify constant is int64 (required for shape operations)
        assert shape_constant.dtype == torch.int64

    @pytest.mark.parametrize("opset_version", [15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_shape, start, end, description",
        [
            ((2, 3, 4), 0, None, "start=0, end=None (full shape)"),
            ((2, 3, 4), None, None, "start=None, end=None (full shape)"),
            ((2, 3, 4), 0, 3, "start=0, end=3 (full shape)"),
            ((2, 3, 4), -3, None, "start=-3, end=None (full shape)"),
            ((2, 3, 4), None, -1, "start=None, end=-1 (exclude last)"),
        ],
    )
    def test_shape_default_attributes(self, opset_version, input_shape, start, end, description):
        """Test Shape operation with default attribute values (v13+)."""
        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape, start=start, end=end)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        # Calculate expected output based on start/end
        r = len(input_shape)
        start_val = start if start is not None else 0
        end_val = end if end is not None else r

        # Normalize negative indices
        if start_val < 0:
            start_val = start_val + r
        if end_val is not None and end_val < 0:
            end_val = end_val + r

        # Clamp to valid range
        start_val = max(0, min(start_val, r))
        if end_val is not None:
            end_val = max(0, min(end_val, r))
        else:
            end_val = r

        # Extract slice
        if start_val >= end_val:
            expected_output = []
        else:
            expected_output = list(input_shape[start_val:end_val])

        expected = torch.tensor(expected_output, dtype=torch.int64)

        assert torch.equal(
            shape_constant, expected
        ), f"Shape with {description} mismatch: expected {expected.tolist()}, got {shape_constant.tolist()}"

    @pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 15, 19, 21, 23, 25])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (0,),  # Zero-sized dimension
            (0, 5),  # Zero-sized first dimension
            (5, 0),  # Zero-sized second dimension
            (0, 0),  # Multiple zero-sized dimensions
        ],
    )
    def test_shape_zero_sized_dimensions(self, opset_version, input_shape):
        """Test Shape operation with zero-sized dimensions."""
        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor(input_shape, dtype=torch.int64)

        assert torch.equal(
            shape_constant, expected
        ), f"Zero-sized dimension shape mismatch: expected {expected.tolist()}, got {shape_constant.tolist()}"

    @pytest.mark.parametrize("opset_version", [15, 19, 21, 23, 25])
    def test_shape_large_tensor(self, opset_version):
        """Test Shape operation with large tensor dimensions."""
        input_shape = (1000, 2000, 3000)

        # Create ONNX model
        onnx_model = _create_shape_model(opset_version, input_shape)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify constant
        output_name = onnx_model.graph.output[0].name
        shape_constant = _get_shape_constant(tir_graph, output_name)
        assert shape_constant is not None, f"Shape constant '{output_name}' not found"

        expected = torch.tensor(input_shape, dtype=torch.int64)

        assert torch.equal(
            shape_constant, expected
        ), f"Large tensor shape mismatch: expected {expected.tolist()}, got {shape_constant.tolist()}"
