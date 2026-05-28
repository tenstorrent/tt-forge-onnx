# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX ConstantOfShape operation.
Tests different shapes, dtypes, opset versions, custom values, and edge cases.
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


def _create_constantofshape_model(
    opset_version: int,
    shape: tuple,
    value: np.ndarray = None,
    value_dtype: int = None,
    output_dtype: int = None,
    node_name: str = "constantofshape_test",
) -> onnx.ModelProto:
    """
    Create ONNX ConstantOfShape model.

    Args:
        opset_version: Opset version (9+)
        shape: Output tensor shape (tuple of non-negative integers)
        value: Optional one-element numpy array for fill value
        value_dtype: Optional ONNX dtype for value tensor (if value provided)
        output_dtype: Output tensor dtype (defaults to value dtype or FLOAT)
        node_name: Node name

    Returns:
        ONNX ModelProto
    """
    # Shape input must be 1D tensor of int64
    # Handle empty shape (scalar) - create empty array
    if len(shape) == 0:
        shape_array = np.array([], dtype=np.int64)
        shape_input_shape = (0,)  # Empty 1D tensor
    else:
        shape_array = np.array(shape, dtype=np.int64)
        shape_input_shape = (len(shape),)

    # Prepare attributes
    attrs = {}
    initializers = {"shape": shape_array}

    # Add value attribute if provided
    if value is not None:
        # Create TensorProto for value attribute
        if value_dtype is None:
            # Infer dtype from numpy array
            if value.dtype == np.float32:
                value_dtype = onnx.TensorProto.FLOAT
            elif value.dtype == np.float64:
                value_dtype = onnx.TensorProto.DOUBLE
            elif value.dtype == np.int32:
                value_dtype = onnx.TensorProto.INT32
            elif value.dtype == np.int64:
                value_dtype = onnx.TensorProto.INT64
            elif value.dtype == np.bool_ or value.dtype == bool:
                value_dtype = onnx.TensorProto.BOOL
            else:
                value_dtype = onnx.TensorProto.FLOAT

        # Convert value to appropriate numpy dtype for TensorProto
        if value_dtype == onnx.TensorProto.FLOAT:
            np_value = value.astype(np.float32)
        elif value_dtype == onnx.TensorProto.DOUBLE:
            np_value = value.astype(np.float64)
        elif value_dtype == onnx.TensorProto.INT32:
            np_value = value.astype(np.int32)
        elif value_dtype == onnx.TensorProto.INT64:
            np_value = value.astype(np.int64)
        elif value_dtype == onnx.TensorProto.BOOL:
            np_value = value.astype(bool)
        else:
            np_value = value.astype(np.float32)

        # Create TensorProto for value (one-element tensor)
        value_tensor = onnx.numpy_helper.from_array(np_value, name="value_attr")
        attrs["value"] = value_tensor

        # Determine output dtype
        if output_dtype is None:
            output_dtype = value_dtype
    else:
        # Default: 0.0, float32
        if output_dtype is None:
            output_dtype = onnx.TensorProto.FLOAT

    # Create model
    return create_onnx_model(
        op_type="ConstantOfShape",
        input_shapes=[shape_input_shape],  # Shape input is 1D tensor (empty for scalar)
        input_dtypes=[onnx.TensorProto.INT64],
        output_shapes=[shape],
        output_dtypes=[output_dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name=node_name,
        input_names=["shape"],
        initializers=initializers,
    )


@pytest.mark.transpiler
class TestConstantOfShape:
    """Comprehensive test cases for ConstantOfShape operation."""

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "shape, dtype",
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
    def test_constantofshape_basic(self, opset_version, shape, dtype):
        """Test basic ConstantOfShape operations with default value (0.0).

        ConstantOfShape is eagerly folded into a computed constant at transpilation time,
        so no TIR nodes are generated. The result is in tir_graph.computed_constants.
        """
        # Create ONNX model
        onnx_model = _create_constantofshape_model(opset_version, shape, value=None, output_dtype=dtype)

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # ConstantOfShape is folded into a computed constant — no TIR nodes generated
        output_name = "output_0"
        assert (
            output_name in tir_graph.computed_constants
        ), f"Expected constant '{output_name}' in tir_graph.computed_constants, keys: {list(tir_graph.computed_constants.keys())}"

        constant = tir_graph.computed_constants[output_name]
        assert tuple(constant.shape) == shape, f"Shape mismatch: {constant.shape} != {shape}"

        # Verify with ONNX Runtime (no input data needed - shape is constant)
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={},  # No inputs - shape is in initializers
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "shape, value, value_dtype, output_dtype",
        [
            # Float values
            ((3, 4), np.array([5.0]), onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT),
            ((2, 3), np.array([-1.5]), onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT),
            ((5,), np.array([3.14]), onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT),
            # Double values
            ((3, 4), np.array([5.0]), onnx.TensorProto.DOUBLE, onnx.TensorProto.DOUBLE),
            # Integer values
            ((3, 4), np.array([10]), onnx.TensorProto.INT32, onnx.TensorProto.INT32),
            ((2, 5), np.array([42]), onnx.TensorProto.INT64, onnx.TensorProto.INT64),
            # Boolean values
            ((3, 4), np.array([True]), onnx.TensorProto.BOOL, onnx.TensorProto.BOOL),
            ((2, 2), np.array([False]), onnx.TensorProto.BOOL, onnx.TensorProto.BOOL),
        ],
    )
    def test_constantofshape_custom_value(self, opset_version, shape, value, value_dtype, output_dtype):
        """Test ConstantOfShape with custom fill values.

        ConstantOfShape is eagerly folded into a computed constant at transpilation time.
        """
        # Create ONNX model
        onnx_model = _create_constantofshape_model(
            opset_version, shape, value=value, value_dtype=value_dtype, output_dtype=output_dtype
        )

        # Transpile
        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # ConstantOfShape is folded into a computed constant — no TIR nodes generated
        output_name = "output_0"
        assert (
            output_name in tir_graph.computed_constants
        ), f"Expected constant '{output_name}' in tir_graph.computed_constants, keys: {list(tir_graph.computed_constants.keys())}"

        constant = tir_graph.computed_constants[output_name]
        assert tuple(constant.shape) == shape, f"Shape mismatch: {constant.shape} != {shape}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    def test_constantofshape_scalar(self, opset_version):
        """Test ConstantOfShape with empty shape (scalar output).

        ConstantOfShape is folded into a computed constant — result is in computed_constants.
        """
        shape = ()  # Empty shape = scalar

        # Test with default value
        onnx_model = _create_constantofshape_model(opset_version, shape)

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify the result is stored as a computed constant
        assert (
            "output_0" in tir_graph.computed_constants
        ), f"Expected 'output_0' in computed_constants, got: {list(tir_graph.computed_constants.keys())}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={},
            atol=1e-5,
            rtol=1e-5,
        )

        # Test with custom value
        onnx_model_custom = _create_constantofshape_model(opset_version, shape, value=np.array([7.5]))

        tir_graph_custom = transpiler.transpile(onnx_model_custom)
        compare_tir_with_onnx(
            tir_graph_custom,
            onnx_model_custom,
            input_data={},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "shape",
        [
            (0,),  # Zero-sized 1D
            (0, 5),  # Zero-sized first dimension
            (5, 0),  # Zero-sized second dimension
            (0, 0),  # Zero-sized 2D
            (2, 0, 3),  # Zero-sized middle dimension
        ],
    )
    def test_constantofshape_zero_sized(self, opset_version, shape):
        """Test ConstantOfShape with zero-sized dimensions.

        ConstantOfShape is folded into a computed constant — result is in computed_constants.
        """
        onnx_model = _create_constantofshape_model(opset_version, shape)

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify the result is stored as a computed constant
        output_name = "output_0"
        assert (
            output_name in tir_graph.computed_constants
        ), f"Expected '{output_name}' in computed_constants, keys: {list(tir_graph.computed_constants.keys())}"
        constant = tir_graph.computed_constants[output_name]
        assert tuple(constant.shape) == shape, f"Shape mismatch: {constant.shape} != {shape}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "shape",
        [
            (1,),
            (1, 1),
            (1, 1, 1),
            (1, 1, 1, 1),
        ],
    )
    def test_constantofshape_single_element(self, opset_version, shape):
        """Test ConstantOfShape with single-element tensors."""
        onnx_model = _create_constantofshape_model(opset_version, shape, value=np.array([99.0]))

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "shape",
        [
            (2, 3, 4, 5, 6),  # 5D
            (2, 3, 4, 5, 6, 7),  # 6D
        ],
    )
    def test_constantofshape_high_dimensional(self, opset_version, shape):
        """Test ConstantOfShape with high-dimensional tensors.

        ConstantOfShape is folded into a computed constant — result is in computed_constants.
        """
        onnx_model = _create_constantofshape_model(opset_version, shape)

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify the result is stored as a computed constant
        output_name = "output_0"
        assert (
            output_name in tir_graph.computed_constants
        ), f"Expected '{output_name}' in computed_constants, keys: {list(tir_graph.computed_constants.keys())}"
        constant = tir_graph.computed_constants[output_name]
        assert tuple(constant.shape) == shape, f"Shape mismatch: {constant.shape} != {shape}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    def test_constantofshape_large_tensor(self, opset_version):
        """Test ConstantOfShape with large tensors."""
        shape = (100, 100)
        onnx_model = _create_constantofshape_model(opset_version, shape)

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    @pytest.mark.parametrize(
        "value, dtype",
        [
            (np.array([1.0]), onnx.TensorProto.FLOAT),
            (np.array([2.0]), onnx.TensorProto.DOUBLE),
            (np.array([3]), onnx.TensorProto.INT32),
            (np.array([4]), onnx.TensorProto.INT64),
            (np.array([True]), onnx.TensorProto.BOOL),
        ],
    )
    def test_constantofshape_different_dtypes(self, opset_version, value, dtype):
        """Test ConstantOfShape with different value dtypes."""
        shape = (3, 4)
        onnx_model = _create_constantofshape_model(
            opset_version, shape, value=value, value_dtype=dtype, output_dtype=dtype
        )

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    def test_constantofshape_empty_shape_tensor(self, opset_version):
        """Test ConstantOfShape with empty shape tensor (scalar output)."""
        # Empty shape array = scalar
        shape_array = np.array([], dtype=np.int64)
        initializers = {"shape": shape_array}

        onnx_model = create_onnx_model(
            op_type="ConstantOfShape",
            input_shapes=[(0,)],  # Empty 1D tensor
            input_dtypes=[onnx.TensorProto.INT64],
            output_shapes=[()],  # Scalar output
            output_dtypes=[onnx.TensorProto.FLOAT],
            attrs={},
            opset_version=opset_version,
            node_name="constantofshape_test",
            input_names=["shape"],
            initializers=initializers,
        )

        transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify the result is stored as a computed constant (scalar = 0-dim tensor)
        assert (
            "output_0" in tir_graph.computed_constants
        ), f"Expected 'output_0' in computed_constants, got: {list(tir_graph.computed_constants.keys())}"

        # Verify with ONNX Runtime
        compare_tir_with_onnx(
            tir_graph,
            onnx_model,
            input_data={},
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("opset_version", [9, 20, 21, 23, 24, 25])
    def test_constantofshape_various_values(self, opset_version):
        """Test ConstantOfShape with various fill values."""
        shape = (2, 3)

        test_cases = [
            (np.array([0.0]), onnx.TensorProto.FLOAT),
            (np.array([1.0]), onnx.TensorProto.FLOAT),
            (np.array([-1.0]), onnx.TensorProto.FLOAT),
            (np.array([100.0]), onnx.TensorProto.FLOAT),
            (np.array([0.5]), onnx.TensorProto.FLOAT),
            (np.array([0]), onnx.TensorProto.INT32),
            (np.array([1]), onnx.TensorProto.INT32),
            (np.array([-1]), onnx.TensorProto.INT32),
            (np.array([True]), onnx.TensorProto.BOOL),
            (np.array([False]), onnx.TensorProto.BOOL),
        ]

        for value, dtype in test_cases:
            onnx_model = _create_constantofshape_model(
                opset_version, shape, value=value, value_dtype=dtype, output_dtype=dtype
            )

            transpiler = ONNXToForgeTranspiler(debug=True, validate_model=True)
            tir_graph = transpiler.transpile(onnx_model)

            # Verify with ONNX Runtime
            compare_tir_with_onnx(
                tir_graph,
                onnx_model,
                input_data={},
                atol=1e-5,
                rtol=1e-5,
            )
