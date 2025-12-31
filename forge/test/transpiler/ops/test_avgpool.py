# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX AveragePool operations (1D, 2D, and 3D).
Tests different input shapes, kernel sizes, attributes, opset versions, and edge cases.
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
# HELPER METHODS FOR CREATING AVGPOOL MODELS
# ============================================================================


def _create_avgpool1d_model(
    opset_version,
    input_shape,
    kernel_shape,
    stride=1,
    padding=0,
    ceil_mode=False,
    count_include_pad=False,
    auto_pad="NOTSET",
    dtype=onnx.TensorProto.FLOAT,
):
    """Helper to create AveragePool ONNX model for 1D."""
    attrs = {
        "kernel_shape": [kernel_shape] if isinstance(kernel_shape, int) else list(kernel_shape),
    }

    if stride != 1:
        if isinstance(stride, int):
            attrs["strides"] = [stride]
        else:
            attrs["strides"] = list(stride) if isinstance(stride, (list, tuple)) else [stride[0]]

    if auto_pad == "NOTSET":
        if padding != 0:
            if isinstance(padding, int):
                attrs["pads"] = [padding, padding]
            elif isinstance(padding, (list, tuple)):
                if len(padding) == 2:
                    attrs["pads"] = list(padding)
                else:
                    attrs["pads"] = [padding[0], padding[0]]
    else:
        attrs["auto_pad"] = auto_pad

    if opset_version >= 10:
        attrs["ceil_mode"] = 1 if ceil_mode else 0

    if opset_version >= 7:
        attrs["count_include_pad"] = 1 if count_include_pad else 0

    if opset_version >= 19:
        attrs["dilations"] = [1]

    return create_onnx_model(
        op_type="AveragePool",
        input_shapes=[input_shape],
        input_dtypes=[dtype],
        output_shapes=[input_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name="avgpool1d",
    )


def _create_avgpool2d_model(
    opset_version,
    input_shape,
    kernel_shape,
    stride=1,
    padding=0,
    ceil_mode=False,
    count_include_pad=False,
    auto_pad="NOTSET",
    dtype=onnx.TensorProto.FLOAT,
):
    """Helper to create AveragePool ONNX model for 2D."""
    attrs = {
        "kernel_shape": list(kernel_shape),
    }

    if stride != 1:
        if isinstance(stride, int):
            attrs["strides"] = [stride, stride]
        else:
            attrs["strides"] = list(stride) if isinstance(stride, (list, tuple)) else [stride[0], stride[1]]

    if auto_pad == "NOTSET":
        if padding != 0:
            if isinstance(padding, int):
                attrs["pads"] = [padding, padding, padding, padding]
            elif isinstance(padding, (list, tuple)):
                if len(padding) == 2:
                    attrs["pads"] = [padding[0], padding[1], padding[0], padding[1]]
                elif len(padding) == 4:
                    attrs["pads"] = list(padding)
    else:
        attrs["auto_pad"] = auto_pad

    if opset_version >= 10:
        attrs["ceil_mode"] = 1 if ceil_mode else 0

    if opset_version >= 7:
        attrs["count_include_pad"] = 1 if count_include_pad else 0

    if opset_version >= 19:
        attrs["dilations"] = [1, 1]

    return create_onnx_model(
        op_type="AveragePool",
        input_shapes=[input_shape],
        input_dtypes=[dtype],
        output_shapes=[input_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name="avgpool2d",
    )


def _create_avgpool3d_model(
    opset_version,
    input_shape,
    kernel_shape,
    stride=1,
    padding=0,
    ceil_mode=False,
    count_include_pad=False,
    auto_pad="NOTSET",
    dtype=onnx.TensorProto.FLOAT,
):
    """Helper to create AveragePool ONNX model for 3D."""
    attrs = {
        "kernel_shape": list(kernel_shape),
    }

    if stride != 1:
        if isinstance(stride, int):
            attrs["strides"] = [stride, stride, stride]
        else:
            attrs["strides"] = list(stride) if isinstance(stride, (list, tuple)) else [stride[0], stride[1], stride[2]]

    if auto_pad == "NOTSET":
        if padding != 0:
            if isinstance(padding, int):
                attrs["pads"] = [padding, padding, padding, padding, padding, padding]
            elif isinstance(padding, (list, tuple)):
                if len(padding) == 6:
                    attrs["pads"] = list(padding)
                elif len(padding) == 3:
                    attrs["pads"] = [padding[0], padding[1], padding[2], padding[0], padding[1], padding[2]]
                else:
                    attrs["pads"] = [padding[0], padding[0], padding[0], padding[0], padding[0], padding[0]]
    else:
        attrs["auto_pad"] = auto_pad

    if opset_version >= 10:
        attrs["ceil_mode"] = 1 if ceil_mode else 0

    if opset_version >= 7:
        attrs["count_include_pad"] = 1 if count_include_pad else 0

    if opset_version >= 19:
        attrs["dilations"] = [1, 1, 1]

    return create_onnx_model(
        op_type="AveragePool",
        input_shapes=[input_shape],
        input_dtypes=[dtype],
        output_shapes=[input_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name="avgpool3d",
    )


# ============================================================================
# AVGPOOL 1D TESTS
# ============================================================================


@pytest.mark.transpiler
class TestAveragePool1d:
    """Comprehensive test cases for AveragePool1d operation."""

    # ========================================================================
    # BASIC TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 32), (2,)),
            ((1, 3, 32), (3,)),
            ((1, 3, 32), (5,)),
            ((1, 3, 32), (7,)),
            ((1, 3, 32), (1,)),
            ((1, 3, 64), (3,)),
            ((1, 3, 16), (3,)),
            ((1, 3, 128), (7,)),
            ((1, 1, 32), (3,)),
            ((1, 64, 32), (3,)),
            ((1, 128, 32), (3,)),
            ((4, 3, 32), (3,)),
            ((8, 3, 32), (3,)),
            ((1, 3, 31), (3,)),
            ((1, 3, 33), (3,)),
        ],
    )
    @pytest.mark.parametrize("stride", [1, 2, 3])
    @pytest.mark.parametrize("padding", [0, 1, 2, (1, 1), (2, 2)])
    def test_avgpool1d_basic(self, opset_version, input_shape, kernel_shape, stride, padding):
        """Test basic AveragePool1d with different kernel sizes, strides, and padding."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support AveragePool(1)")

        if isinstance(padding, int):
            pad_w = padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:
                pad_w = max(padding[0], padding[1])
            else:
                pad_w = padding[0] if len(padding) > 0 else 0
        else:
            pad_w = 0

        kW = kernel_shape[0] if isinstance(kernel_shape, (list, tuple)) else kernel_shape

        if pad_w > kW // 2:
            pytest.skip(
                f"Invalid padding: padding={padding}, kernel_shape={kernel_shape}. "
                f"PyTorch requires padding <= kernel_size/2"
            )

        onnx_model = _create_avgpool1d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=False,
            auto_pad="NOTSET",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version >= 19 else 1e-4
        atol = 1e-4 if opset_version >= 19 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    # ========================================================================
    # AUTO_PAD TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 32), (3,)),
            ((1, 3, 31), (3,)),
            ((1, 3, 16), (2,)),
        ],
    )
    @pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
    @pytest.mark.parametrize("stride", [2])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_avgpool1d_auto_pad(self, opset_version, input_shape, kernel_shape, auto_pad, stride, ceil_mode):
        """Test AveragePool1d with different auto_pad modes."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support AveragePool(1)")

        onnx_model = _create_avgpool1d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            count_include_pad=False,
            auto_pad=auto_pad,
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)

        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            rtol = 1e-2
            atol = 1.5
        else:
            rtol = 1e-3 if opset_version >= 19 else 1e-4
            atol = 1e-4 if opset_version >= 19 else 1e-5

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    # ========================================================================
    # CEIL_MODE TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [7, 11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 32), (3,)),
            ((1, 3, 31), (3,)),
            ((1, 3, 33), (3,)),
            ((1, 3, 5), (2,)),
        ],
    )
    @pytest.mark.parametrize("stride", [2, 3])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_avgpool1d_ceil_mode(self, opset_version, input_shape, kernel_shape, stride, ceil_mode):
        """Test AveragePool1d with ceil_mode."""
        if opset_version == 7:
            pytest.skip(
                f"ceil_mode attribute is not supported in AveragePool opset {opset_version}. "
                f"It was introduced in opset 10."
            )

        onnx_model = _create_avgpool1d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            count_include_pad=False,
            auto_pad="NOTSET",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version >= 19 else 1e-4
        atol = 1e-4 if opset_version >= 19 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    # ========================================================================
    # COUNT_INCLUDE_PAD TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 32), (3,)),
            ((1, 3, 16), (3,)),
            ((1, 1, 8), (3,)),
        ],
    )
    @pytest.mark.parametrize("padding", [1, 2, (1, 1), (2, 2)])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    def test_avgpool1d_count_include_pad(self, opset_version, input_shape, kernel_shape, padding, count_include_pad):
        """Test AveragePool1d with count_include_pad attribute."""
        if isinstance(padding, int):
            pad_w = padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:
                pad_w = max(padding[0], padding[1])
            else:
                pad_w = padding[0] if len(padding) > 0 else 0
        else:
            pad_w = 0

        kW = kernel_shape[0] if isinstance(kernel_shape, (list, tuple)) else kernel_shape

        if pad_w > kW // 2:
            pytest.skip(
                f"Invalid padding: padding={padding}, kernel_shape={kernel_shape}. "
                f"PyTorch requires padding <= kernel_size/2"
            )

        onnx_model = _create_avgpool1d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=1,
            padding=padding,
            ceil_mode=False,
            count_include_pad=count_include_pad,
            auto_pad="NOTSET",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version >= 19 else 1e-4
        atol = 1e-4 if opset_version >= 19 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 19])
    def test_avgpool1d_edge_cases(self, opset_version):
        """Test edge cases for AveragePool1d."""
        test_cases = [
            ((1, 1, 1), (1,), 1, 0, "1x1 input with 1x1 kernel"),
            ((1, 3, 2), (3,), 1, 0, "Small input, kernel larger than input"),
            ((1, 3, 10), (10,), 1, 0, "Kernel same size as input"),
            ((1, 3, 32), (3,), 10, 0, "Stride larger than kernel"),
        ]

        for input_shape, kernel_shape, stride, padding, description in test_cases:
            kW = kernel_shape[0] if isinstance(kernel_shape, (list, tuple)) else kernel_shape
            if isinstance(padding, int) and padding > kW // 2:
                continue

            # Calculate output shape and skip if invalid
            W_in = input_shape[2]
            if isinstance(padding, int):
                total_pad = padding * 2
            elif isinstance(padding, (list, tuple)) and len(padding) == 2:
                total_pad = padding[0] + padding[1]
            else:
                total_pad = 0

            sW = stride if isinstance(stride, int) else stride[0] if isinstance(stride, (list, tuple)) else 1
            W_out = (W_in + total_pad - kW) // sW + 1

            if W_out <= 0:
                continue  # Skip invalid output shapes

            onnx_model = _create_avgpool1d_model(
                opset_version=opset_version,
                input_shape=input_shape,
                kernel_shape=kernel_shape,
                stride=stride,
                padding=padding,
                ceil_mode=False,
                count_include_pad=False,
                auto_pad="NOTSET",
                dtype=onnx.TensorProto.FLOAT,
            )

            input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

            transpiler = ONNXToForgeTranspiler(debug=False)
            tir_graph = transpiler.transpile(onnx_model)

            verify_tir_graph_structure(tir_graph, onnx_model)
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)

            assert len(comparison["errors"]) == 0, f"{description}: Comparison errors: {comparison['errors']}"
            assert all(comparison["matches"].values()), f"{description}: Output values don't match"


# ============================================================================
# AVGPOOL 2D TESTS
# ============================================================================


@pytest.mark.transpiler
class TestAveragePool2d:
    """Comprehensive test cases for AveragePool2d operation."""

    # ========================================================================
    # BASIC TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 32, 32), (2, 2)),
            ((1, 3, 32, 32), (3, 3)),
            ((1, 3, 32, 32), (5, 5)),
            ((1, 3, 32, 32), (7, 7)),
            ((1, 3, 32, 32), (1, 1)),
            ((1, 3, 32, 32), (3, 5)),
            ((1, 3, 32, 32), (5, 3)),
            ((1, 3, 32, 32), (2, 3)),
            ((1, 3, 64, 64), (3, 3)),
            ((1, 3, 16, 16), (3, 3)),
            ((1, 3, 28, 28), (3, 3)),
            ((1, 3, 224, 224), (7, 7)),
            ((1, 1, 32, 32), (3, 3)),
            ((1, 64, 32, 32), (3, 3)),
            ((1, 128, 32, 32), (3, 3)),
            ((4, 3, 32, 32), (3, 3)),
            ((8, 3, 32, 32), (3, 3)),
            ((1, 3, 31, 31), (3, 3)),
            ((1, 3, 33, 33), (3, 3)),
        ],
    )
    @pytest.mark.parametrize("stride", [1, 2, 3, (1, 1), (2, 2), (1, 2), (2, 1)])
    @pytest.mark.parametrize("padding", [0, 1, 2, (1, 1), (2, 2), (1, 2, 1, 2)])
    def test_avgpool2d_basic(self, opset_version, input_shape, kernel_shape, stride, padding):
        """Test basic AveragePool2d with different kernel sizes, strides, and padding."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support AveragePool(1)")

        if isinstance(padding, int):
            pad_h, pad_w = padding, padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:
                pad_h, pad_w = padding[0], padding[1]
            elif len(padding) == 4:
                pad_h = padding[0] + padding[2]
                pad_w = padding[1] + padding[3]
            else:
                pad_h, pad_w = 0, 0
        else:
            pad_h, pad_w = 0, 0

        if pad_h > kernel_shape[0] // 2 or pad_w > kernel_shape[1] // 2:
            pytest.skip(
                f"Invalid padding: padding={padding}, kernel_shape={kernel_shape}. "
                f"PyTorch requires padding <= kernel_size/2"
            )

        onnx_model = _create_avgpool2d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=False,
            auto_pad="NOTSET",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version >= 19 else 1e-4
        atol = 1e-4 if opset_version >= 19 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    # ========================================================================
    # AUTO_PAD TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 32, 32), (3, 3)),
            ((1, 3, 31, 31), (3, 3)),
            ((1, 3, 16, 16), (2, 2)),
        ],
    )
    @pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
    @pytest.mark.parametrize("stride", [2, (2, 2)])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_avgpool2d_auto_pad(self, opset_version, input_shape, kernel_shape, auto_pad, stride, ceil_mode):
        """Test AveragePool2d with different auto_pad modes."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support AveragePool(1)")

        onnx_model = _create_avgpool2d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            count_include_pad=False,
            auto_pad=auto_pad,
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)

        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            rtol = 1e-2
            atol = 1.5
        else:
            rtol = 1e-3 if opset_version >= 19 else 1e-4
            atol = 1e-4 if opset_version >= 19 else 1e-5

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    # ========================================================================
    # CEIL_MODE TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [7, 11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 32, 32), (3, 3)),
            ((1, 3, 31, 31), (3, 3)),
            ((1, 3, 33, 33), (3, 3)),
            ((1, 3, 5, 5), (2, 2)),
        ],
    )
    @pytest.mark.parametrize("stride", [2, 3, (2, 2), (3, 3)])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_avgpool2d_ceil_mode(self, opset_version, input_shape, kernel_shape, stride, ceil_mode):
        """Test AveragePool2d with ceil_mode."""
        if opset_version == 7:
            pytest.skip(
                f"ceil_mode attribute is not supported in AveragePool opset {opset_version}. "
                f"It was introduced in opset 10."
            )

        onnx_model = _create_avgpool2d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            count_include_pad=False,
            auto_pad="NOTSET",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version >= 19 else 1e-4
        atol = 1e-4 if opset_version >= 19 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    # ========================================================================
    # COUNT_INCLUDE_PAD TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [7, 11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 32, 32), (3, 3)),
            ((1, 3, 16, 16), (3, 3)),
            ((1, 1, 8, 8), (3, 3)),
        ],
    )
    @pytest.mark.parametrize("padding", [1, 2, (1, 1), (2, 2)])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    def test_avgpool2d_count_include_pad(self, opset_version, input_shape, kernel_shape, padding, count_include_pad):
        """Test AveragePool2d with count_include_pad attribute."""
        if isinstance(padding, int):
            pad_h, pad_w = padding, padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:
                pad_h, pad_w = padding[0], padding[1]
            else:
                pad_h, pad_w = 0, 0
        else:
            pad_h, pad_w = 0, 0

        if pad_h > kernel_shape[0] // 2 or pad_w > kernel_shape[1] // 2:
            pytest.skip(
                f"Invalid padding: padding={padding}, kernel_shape={kernel_shape}. "
                f"PyTorch requires padding <= kernel_size/2"
            )

        onnx_model = _create_avgpool2d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=1,
            padding=padding,
            ceil_mode=False,
            count_include_pad=count_include_pad,
            auto_pad="NOTSET",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version >= 19 else 1e-4
        atol = 1e-4 if opset_version >= 19 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    # ========================================================================
    # COMBINATION TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 19])
    def test_avgpool2d_ceil_mode_with_auto_pad(self, opset_version):
        """Test AveragePool2d with ceil_mode and auto_pad together."""
        onnx_model = _create_avgpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 31, 31),
            kernel_shape=(3, 3),
            stride=2,
            padding=0,
            ceil_mode=True,
            count_include_pad=False,
            auto_pad="SAME_UPPER",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(1, 3, 31, 31).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-2
        atol = 6e-1  # Increased tolerance for ceil_mode with auto_pad combination
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    @pytest.mark.parametrize("opset_version", [11, 19])
    def test_avgpool2d_count_include_pad_with_padding(self, opset_version):
        """Test AveragePool2d with count_include_pad and explicit padding."""
        for count_include_pad in [True, False]:
            onnx_model = _create_avgpool2d_model(
                opset_version=opset_version,
                input_shape=(1, 1, 8, 8),
                kernel_shape=(3, 3),
                stride=1,
                padding=1,
                ceil_mode=False,
                count_include_pad=count_include_pad,
                auto_pad="NOTSET",
                dtype=onnx.TensorProto.FLOAT,
            )

            input_data = {"input_0": np.random.randn(1, 1, 8, 8).astype(np.float32)}

            transpiler = ONNXToForgeTranspiler(debug=False)
            tir_graph = transpiler.transpile(onnx_model)

            verify_tir_graph_structure(tir_graph, onnx_model)
            rtol = 1e-3 if opset_version >= 19 else 1e-4
            atol = 1e-4 if opset_version >= 19 else 1e-5
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

            assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
            assert all(comparison["matches"].values()), "Output values don't match"


# ============================================================================
# AVGPOOL 3D TESTS
# ============================================================================


@pytest.mark.transpiler
class TestAveragePool3d:
    """Comprehensive test cases for AveragePool3d operation."""

    # ========================================================================
    # BASIC TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 8, 16, 16), (2, 2, 2)),
            ((1, 3, 8, 16, 16), (3, 3, 3)),
            ((1, 3, 8, 16, 16), (2, 3, 3)),
            ((1, 3, 8, 16, 16), (1, 1, 1)),
            ((1, 3, 16, 32, 32), (3, 3, 3)),
            ((1, 3, 4, 8, 8), (3, 3, 3)),
            ((1, 1, 8, 16, 16), (3, 3, 3)),
            ((1, 64, 8, 16, 16), (3, 3, 3)),
            ((4, 3, 8, 16, 16), (3, 3, 3)),
            ((1, 3, 7, 15, 15), (3, 3, 3)),
            ((1, 3, 9, 17, 17), (3, 3, 3)),
        ],
    )
    @pytest.mark.parametrize("stride", [1, 2, (1, 1, 1), (2, 2, 2)])
    @pytest.mark.parametrize("padding", [0, 1, (1, 1, 1, 1, 1, 1)])
    def test_avgpool3d_basic(self, opset_version, input_shape, kernel_shape, stride, padding):
        """Test basic AveragePool3d with different kernel sizes, strides, and padding."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support AveragePool(1)")

        if isinstance(padding, int):
            pad_d = pad_h = pad_w = padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 6:
                pad_d = max(padding[0], padding[3])
                pad_h = max(padding[1], padding[4])
                pad_w = max(padding[2], padding[5])
            elif len(padding) == 3:
                pad_d = pad_h = pad_w = padding[0]
            else:
                pad_d = pad_h = pad_w = 0
        else:
            pad_d = pad_h = pad_w = 0

        kD, kH, kW = kernel_shape[0], kernel_shape[1], kernel_shape[2]

        if pad_d > kD // 2 or pad_h > kH // 2 or pad_w > kW // 2:
            pytest.skip(
                f"Invalid padding: padding={padding}, kernel_shape={kernel_shape}. "
                f"PyTorch requires padding <= kernel_size/2"
            )

        onnx_model = _create_avgpool3d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=False,
            auto_pad="NOTSET",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version >= 19 else 1e-4
        atol = 1e-4 if opset_version >= 19 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    # ========================================================================
    # AUTO_PAD TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 8, 16, 16), (3, 3, 3)),
            ((1, 3, 7, 15, 15), (3, 3, 3)),
            ((1, 3, 4, 8, 8), (2, 2, 2)),
        ],
    )
    @pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
    @pytest.mark.parametrize("stride", [2])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_avgpool3d_auto_pad(self, opset_version, input_shape, kernel_shape, auto_pad, stride, ceil_mode):
        """Test AveragePool3d with different auto_pad modes."""
        if opset_version == 1:
            pytest.skip(f"ONNX Runtime doesn't support AveragePool(1)")

        onnx_model = _create_avgpool3d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            count_include_pad=False,
            auto_pad=auto_pad,
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)

        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            rtol = 1e-2
            atol = 1.5
        else:
            rtol = 1e-3 if opset_version >= 19 else 1e-4
            atol = 1e-4 if opset_version >= 19 else 1e-5

        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    # ========================================================================
    # CEIL_MODE TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [7, 11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 8, 16, 16), (3, 3, 3)),
            ((1, 3, 7, 15, 15), (3, 3, 3)),
            ((1, 3, 9, 17, 17), (3, 3, 3)),
            ((1, 3, 5, 5, 5), (2, 2, 2)),
        ],
    )
    @pytest.mark.parametrize("stride", [2, 3, (2, 2, 2), (3, 3, 3)])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_avgpool3d_ceil_mode(self, opset_version, input_shape, kernel_shape, stride, ceil_mode):
        """Test AveragePool3d with ceil_mode."""
        if opset_version == 7:
            pytest.skip(
                f"ceil_mode attribute is not supported in AveragePool opset {opset_version}. "
                f"It was introduced in opset 10."
            )

        onnx_model = _create_avgpool3d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            count_include_pad=False,
            auto_pad="NOTSET",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version >= 19 else 1e-4
        atol = 1e-4 if opset_version >= 19 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    # ========================================================================
    # COUNT_INCLUDE_PAD TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 19])
    @pytest.mark.parametrize(
        "input_shape, kernel_shape",
        [
            ((1, 3, 8, 16, 16), (3, 3, 3)),
            ((1, 3, 4, 8, 8), (3, 3, 3)),
            ((1, 1, 4, 4, 4), (3, 3, 3)),
        ],
    )
    @pytest.mark.parametrize("padding", [1, (1, 1, 1, 1, 1, 1)])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    def test_avgpool3d_count_include_pad(self, opset_version, input_shape, kernel_shape, padding, count_include_pad):
        """Test AveragePool3d with count_include_pad attribute."""
        if isinstance(padding, int):
            pad_d = pad_h = pad_w = padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 6:
                pad_d = max(padding[0], padding[3])
                pad_h = max(padding[1], padding[4])
                pad_w = max(padding[2], padding[5])
            else:
                pad_d = pad_h = pad_w = padding[0] if len(padding) > 0 else 0
        else:
            pad_d = pad_h = pad_w = 0

        kD, kH, kW = kernel_shape[0], kernel_shape[1], kernel_shape[2]

        if pad_d > kD // 2 or pad_h > kH // 2 or pad_w > kW // 2:
            pytest.skip(
                f"Invalid padding: padding={padding}, kernel_shape={kernel_shape}. "
                f"PyTorch requires padding <= kernel_size/2"
            )

        onnx_model = _create_avgpool3d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=1,
            padding=padding,
            ceil_mode=False,
            count_include_pad=count_include_pad,
            auto_pad="NOTSET",
            dtype=onnx.TensorProto.FLOAT,
        )

        input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version >= 19 else 1e-4
        atol = 1e-4 if opset_version >= 19 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 19])
    def test_avgpool3d_edge_cases(self, opset_version):
        """Test edge cases for AveragePool3d."""
        test_cases = [
            ((1, 1, 1, 1, 1), (1, 1, 1), 1, 0, "1x1x1 input with 1x1x1 kernel"),
            ((1, 3, 2, 2, 2), (3, 3, 3), 1, 0, "Small input, kernel larger than input"),
            ((1, 3, 4, 4, 4), (4, 4, 4), 1, 0, "Kernel same size as input"),
            ((1, 3, 8, 16, 16), (3, 3, 3), 10, 0, "Stride larger than kernel"),
        ]

        for input_shape, kernel_shape, stride, padding, description in test_cases:
            kD, kH, kW = kernel_shape[0], kernel_shape[1], kernel_shape[2]
            if isinstance(padding, int) and padding > min(kD, kH, kW) // 2:
                continue

            # Calculate output shape and skip if invalid
            D_in, H_in, W_in = input_shape[2], input_shape[3], input_shape[4]
            if isinstance(padding, int):
                total_pad_d = total_pad_h = total_pad_w = padding * 2
            elif isinstance(padding, (list, tuple)) and len(padding) == 6:
                total_pad_d = padding[0] + padding[3]
                total_pad_h = padding[1] + padding[4]
                total_pad_w = padding[2] + padding[5]
            else:
                total_pad_d = total_pad_h = total_pad_w = 0

            if isinstance(stride, int):
                sD = sH = sW = stride
            elif isinstance(stride, (list, tuple)) and len(stride) == 3:
                sD, sH, sW = stride[0], stride[1], stride[2]
            else:
                sD = sH = sW = 1

            D_out = (D_in + total_pad_d - kD) // sD + 1
            H_out = (H_in + total_pad_h - kH) // sH + 1
            W_out = (W_in + total_pad_w - kW) // sW + 1

            if D_out <= 0 or H_out <= 0 or W_out <= 0:
                continue  # Skip invalid output shapes

            onnx_model = _create_avgpool3d_model(
                opset_version=opset_version,
                input_shape=input_shape,
                kernel_shape=kernel_shape,
                stride=stride,
                padding=padding,
                ceil_mode=False,
                count_include_pad=False,
                auto_pad="NOTSET",
                dtype=onnx.TensorProto.FLOAT,
            )

            input_data = {"input_0": np.random.randn(*input_shape).astype(np.float32)}

            transpiler = ONNXToForgeTranspiler(debug=False)
            tir_graph = transpiler.transpile(onnx_model)

            verify_tir_graph_structure(tir_graph, onnx_model)
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)

            assert len(comparison["errors"]) == 0, f"{description}: Comparison errors: {comparison['errors']}"
            assert all(comparison["matches"].values()), f"{description}: Output values don't match"
