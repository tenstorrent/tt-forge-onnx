# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Conv operation (Conv1d and Conv2d).
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


@pytest.mark.transpiler
class TestConv:
    """Comprehensive test cases for Conv operation (Conv1d and Conv2d)."""

    @staticmethod
    def _calculate_conv_output_shape(input_shape, kernel_shape, stride, padding, dilation, groups=1):
        """
        Calculate expected output shape for convolution.

        Args:
            input_shape: (N, C_in, *spatial_dims)
            kernel_shape: (*kernel_dims,)
            stride: tuple of strides
            padding: padding values (int, tuple, or list)
            dilation: tuple of dilations
            groups: number of groups

        Returns:
            Output shape tuple
        """
        N, C_in = input_shape[0], input_shape[1]
        spatial_dims = input_shape[2:]
        kernel_dims = len(kernel_shape)

        # Normalize stride and dilation
        if isinstance(stride, int):
            stride = (stride,) * kernel_dims
        elif isinstance(stride, (list, tuple)):
            if len(stride) == 1:
                stride = (stride[0],) * kernel_dims
            else:
                stride = tuple(stride[:kernel_dims])

        if isinstance(dilation, int):
            dilation = (dilation,) * kernel_dims
        elif isinstance(dilation, (list, tuple)):
            if len(dilation) == 1:
                dilation = (dilation[0],) * kernel_dims
            else:
                dilation = tuple(dilation[:kernel_dims])

        # Calculate output spatial dimensions
        output_spatial = []
        for i, (in_size, k_size, s, d) in enumerate(zip(spatial_dims, kernel_shape, stride, dilation)):
            # Get padding for this dimension
            if isinstance(padding, int):
                pad = padding
            elif isinstance(padding, (list, tuple)):
                if len(padding) == 2 * kernel_dims:
                    # ONNX format: [pad_dim0_begin, pad_dim1_begin, ..., pad_dim0_end, pad_dim1_end, ...]
                    pad = padding[i] + padding[i + kernel_dims]
                elif len(padding) == kernel_dims:
                    pad = padding[i] * 2  # Symmetric
                else:
                    pad = padding[0] * 2 if len(padding) > 0 else 0
            else:
                pad = 0

            # Output size formula
            effective_kernel = (k_size - 1) * d + 1
            out_size = (in_size + pad - effective_kernel) // s + 1
            output_spatial.append(out_size)

        # Output channels = weight shape[0]
        # For now, we'll infer from weight shape in the test
        C_out = C_in  # Default, will be overridden by weight shape

        return (N, C_out, *output_spatial)

    @staticmethod
    def _create_conv_model(
        opset_version,
        input_shape,
        weight_shape,
        kernel_shape=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        auto_pad="NOTSET",
        bias=True,
        dtype=onnx.TensorProto.FLOAT,
    ):
        """
        Helper to create Conv ONNX model.

        Args:
            opset_version: ONNX opset version
            input_shape: (N, C_in, *spatial_dims)
            weight_shape: (C_out, C_in/groups, *kernel_dims)
            kernel_shape: Optional kernel shape (inferred from weight if not provided)
            stride: Stride value(s)
            padding: Padding value(s) or 'NOTSET' for auto_pad
            dilation: Dilation value(s)
            groups: Number of groups
            auto_pad: 'NOTSET', 'SAME_UPPER', 'SAME_LOWER', or 'VALID'
            bias: Whether to include bias
            dtype: Data type
        """
        np_dtype = np.float32 if dtype == onnx.TensorProto.FLOAT else np.float64
        attrs = {}
        initializers = {}
        input_names = ["X"]

        # Set kernel_shape attribute
        if kernel_shape is None:
            kernel_shape = weight_shape[2:]
        attrs["kernel_shape"] = list(kernel_shape)

        # Set stride attribute
        if isinstance(stride, int):
            attrs["strides"] = [stride] * len(kernel_shape)
        elif isinstance(stride, (list, tuple)):
            attrs["strides"] = list(stride[: len(kernel_shape)])
        else:
            attrs["strides"] = [1] * len(kernel_shape)

        # Set dilation attribute
        if isinstance(dilation, int):
            attrs["dilations"] = [dilation] * len(kernel_shape)
        elif isinstance(dilation, (list, tuple)):
            attrs["dilations"] = list(dilation[: len(kernel_shape)])
        else:
            attrs["dilations"] = [1] * len(kernel_shape)

        # Set groups attribute
        attrs["group"] = groups

        # Handle padding and auto_pad
        if auto_pad != "NOTSET":
            attrs["auto_pad"] = auto_pad
        else:
            # Set pads attribute
            if isinstance(padding, int):
                # Symmetric padding: create pads list
                pads = [padding] * (2 * len(kernel_shape))
            elif isinstance(padding, (list, tuple)):
                if len(padding) == len(kernel_shape):
                    # Symmetric per dimension: [padH, padW] -> [padH, padW, padH, padW]
                    pads = []
                    for p in padding:
                        pads.extend([p, p])
                elif len(padding) == 2 * len(kernel_shape):
                    # Full ONNX format: [pad_dim0_begin, pad_dim1_begin, ..., pad_dim0_end, pad_dim1_end, ...]
                    pads = list(padding)
                else:
                    pads = [0] * (2 * len(kernel_shape))
            else:
                pads = [0] * (2 * len(kernel_shape))
            attrs["pads"] = pads

        # Create weight initializer
        weight_data = np.random.randn(*weight_shape).astype(np_dtype)
        initializers["W"] = weight_data
        input_names.append("W")

        # Create bias initializer if needed
        if bias:
            C_out = weight_shape[0]
            bias_data = np.random.randn(C_out).astype(np_dtype)
            initializers["B"] = bias_data
            input_names.append("B")

        # Calculate output shape
        C_out = weight_shape[0]
        N = input_shape[0]
        spatial_dims = input_shape[2:]
        kernel_dims = len(kernel_shape)

        # Calculate output spatial dimensions
        if auto_pad == "VALID":
            # No padding
            output_spatial = []
            for i, (in_size, k_size, s, d) in enumerate(
                zip(spatial_dims, kernel_shape, attrs["strides"], attrs["dilations"])
            ):
                effective_kernel = (k_size - 1) * d + 1
                out_size = (in_size - effective_kernel) // s + 1
                output_spatial.append(max(1, out_size))
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # Output size = ceil(input_size / stride)
            output_spatial = []
            for in_size, s in zip(spatial_dims, attrs["strides"]):
                out_size = (in_size + s - 1) // s
                output_spatial.append(max(1, out_size))
        else:
            # Explicit padding
            output_spatial = []
            pads = attrs.get("pads", [0] * (2 * kernel_dims))
            for i, (in_size, k_size, s, d) in enumerate(
                zip(spatial_dims, kernel_shape, attrs["strides"], attrs["dilations"])
            ):
                pad_before = pads[i]
                pad_after = pads[i + kernel_dims]
                total_pad = pad_before + pad_after
                effective_kernel = (k_size - 1) * d + 1
                out_size = (in_size + total_pad - effective_kernel) // s + 1
                output_spatial.append(max(1, out_size))

        output_shape = (N, C_out, *output_spatial)

        # Create model
        return create_onnx_model(
            op_type="Conv",
            input_shapes=[input_shape],
            input_dtypes=[dtype],
            output_shapes=[output_shape],
            output_dtypes=[dtype],
            attrs=attrs,
            opset_version=opset_version,
            node_name="conv_test",
            input_names=input_names,
            initializers=initializers,
        )

    # ========================================================================
    # CONV2D TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape",
        [
            # Basic 2D convolutions
            ((1, 3, 32, 32), (16, 3, 3, 3), (3, 3)),  # Standard 3x3 conv
            ((1, 3, 32, 32), (16, 3, 5, 5), (5, 5)),  # 5x5 kernel
            ((1, 3, 32, 32), (16, 3, 1, 1), (1, 1)),  # 1x1 pointwise
            ((1, 3, 32, 32), (16, 3, 7, 7), (7, 7)),  # 7x7 kernel
            # Different input sizes
            ((1, 3, 64, 64), (32, 3, 3, 3), (3, 3)),  # Larger input
            ((1, 3, 16, 16), (8, 3, 3, 3), (3, 3)),  # Smaller input
            ((1, 3, 28, 28), (16, 3, 3, 3), (3, 3)),  # MNIST-like
            # Different channel counts
            ((1, 1, 32, 32), (8, 1, 3, 3), (3, 3)),  # Grayscale
            ((1, 64, 32, 32), (128, 64, 3, 3), (3, 3)),  # More channels
            # Non-square kernels
            ((1, 3, 32, 32), (16, 3, 3, 5), (3, 5)),  # 3x5 kernel
            ((1, 3, 32, 32), (16, 3, 5, 3), (5, 3)),  # 5x3 kernel
            # Batch size > 1
            ((4, 3, 32, 32), (16, 3, 3, 3), (3, 3)),  # Batch of 4
            ((8, 3, 32, 32), (16, 3, 3, 3), (3, 3)),  # Batch of 8
        ],
    )
    @pytest.mark.parametrize("stride", [1, 2, (1, 1), (2, 2), (1, 2), (2, 1)])
    @pytest.mark.parametrize("padding", [0, 1, 2, (1, 1), (2, 2), (1, 2, 1, 2)])
    @pytest.mark.parametrize("dilation", [1, 2, (1, 1), (2, 2)])
    def test_conv2d_basic(self, opset_version, input_shape, weight_shape, kernel_shape, stride, padding, dilation):
        """Test basic Conv2d operations with various configurations."""
        # Skip opset 1 for ONNX Runtime (may not support)
        if opset_version == 1:
            pytest.skip("Opset 1 may not be fully supported by ONNX Runtime")

        # Validate weight shape matches input
        C_in = input_shape[1]
        C_out, W_in, *W_kernel = weight_shape
        if W_in * 1 != C_in:  # groups=1 for this test
            pytest.skip(f"Weight input channels {W_in} doesn't match input channels {C_in}")
        if tuple(W_kernel) != kernel_shape:
            pytest.skip(f"Weight kernel shape {W_kernel} doesn't match kernel_shape {kernel_shape}")

        # Normalize stride and padding for validation
        if isinstance(stride, int):
            stride_tuple = (stride, stride)
        elif isinstance(stride, (list, tuple)):
            stride_tuple = tuple(stride[:2]) if len(stride) >= 2 else (stride[0], stride[0])
        else:
            stride_tuple = (1, 1)

        # Check if output would be valid (positive size)
        H_in, W_in = input_shape[2], input_shape[3]
        kH, kW = kernel_shape

        # Get effective padding
        if isinstance(padding, int):
            pad_h = pad_w = padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:
                pad_h, pad_w = padding[0] * 2, padding[1] * 2
            elif len(padding) == 4:
                pad_h = padding[0] + padding[2]
                pad_w = padding[1] + padding[3]
            else:
                pad_h = pad_w = 0
        else:
            pad_h = pad_w = 0

        # Normalize dilation
        if isinstance(dilation, int):
            dil_h = dil_w = dilation
        elif isinstance(dilation, (list, tuple)):
            dil_h = dilation[0] if len(dilation) > 0 else 1
            dil_w = dilation[1] if len(dilation) > 1 else dil_h
        else:
            dil_h = dil_w = 1

        # Calculate output size
        effective_kH = (kH - 1) * dil_h + 1
        effective_kW = (kW - 1) * dil_w + 1
        H_out = (H_in + pad_h - effective_kH) // stride_tuple[0] + 1
        W_out = (W_in + pad_w - effective_kW) // stride_tuple[1] + 1

        # Skip if output would be invalid
        if H_out <= 0 or W_out <= 0:
            pytest.skip(f"Invalid output shape: ({H_out}, {W_out})")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            auto_pad="NOTSET",
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        np_dtype = np.float32
        input_data = {"X": np.random.randn(*input_shape).astype(np_dtype)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify graph structure
        verify_tir_graph_structure(tir_graph, onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 22 and 11 due to numerical differences
        if opset_version == 22:
            rtol = 1e-3
            atol = 1e-4
        elif opset_version == 11:
            rtol = 1e-3
            atol = 1e-4
        else:
            rtol = 1e-4
            atol = 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify output shape and dtype
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    @pytest.mark.parametrize("opset_version", [1, 11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape",
        [
            ((1, 3, 32, 32), (16, 3, 3, 3), (3, 3)),
            ((1, 64, 32, 32), (128, 64, 3, 3), (3, 3)),
            ((4, 3, 64, 64), (32, 3, 5, 5), (5, 5)),
        ],
    )
    @pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
    @pytest.mark.parametrize("stride", [1, 2, (1, 1), (2, 2)])
    def test_conv2d_auto_pad(self, opset_version, input_shape, weight_shape, kernel_shape, auto_pad, stride):
        """Test Conv2d with auto_pad modes."""
        if opset_version == 1:
            pytest.skip("Opset 1 may not be fully supported by ONNX Runtime")

        # Normalize stride
        if isinstance(stride, int):
            stride_tuple = (stride, stride)
        else:
            stride_tuple = tuple(stride[:2]) if len(stride) >= 2 else (stride[0], stride[0])

        # Validate weight shape
        C_in = input_shape[1]
        C_out, W_in, *W_kernel = weight_shape
        if W_in != C_in:
            pytest.skip(f"Weight input channels {W_in} doesn't match input channels {C_in}")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,  # Ignored when auto_pad is set
            dilation=1,
            groups=1,
            auto_pad=auto_pad,
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 22 and 11 due to numerical differences
        if opset_version == 22:
            rtol = 1e-3
            atol = 1e-4
        elif opset_version == 11:
            rtol = 1e-3
            atol = 1e-4
        else:
            rtol = 1e-4
            atol = 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify shape and dtype
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    @pytest.mark.parametrize("opset_version", [1, 11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape, groups",
        [
            # Grouped convolutions
            ((1, 64, 32, 32), (128, 32, 3, 3), (3, 3), 2),  # groups=2
            ((1, 64, 32, 32), (128, 16, 3, 3), (3, 3), 4),  # groups=4
            ((1, 64, 32, 32), (128, 8, 3, 3), (3, 3), 8),  # groups=8
            # Depthwise convolution (groups = in_channels)
            ((1, 64, 32, 32), (64, 1, 3, 3), (3, 3), 64),  # Depthwise
            ((1, 32, 32, 32), (32, 1, 3, 3), (3, 3), 32),  # Depthwise
        ],
    )
    @pytest.mark.parametrize("stride", [1, 2, (1, 1), (2, 2)])
    @pytest.mark.parametrize("padding", [0, 1, (1, 1)])
    def test_conv2d_groups(self, opset_version, input_shape, weight_shape, kernel_shape, groups, stride, padding):
        """Test Conv2d with grouped convolutions."""
        if opset_version == 1:
            pytest.skip("Opset 1 may not be fully supported by ONNX Runtime")

        # Validate groups
        C_in = input_shape[1]
        C_out, W_in, *W_kernel = weight_shape

        # Check: C_in must be divisible by groups
        if C_in % groups != 0:
            pytest.skip(f"Input channels {C_in} not divisible by groups {groups}")

        # Check: C_out must be divisible by groups
        if C_out % groups != 0:
            pytest.skip(f"Output channels {C_out} not divisible by groups {groups}")

        # Check: W_in should equal C_in / groups
        expected_W_in = C_in // groups
        if W_in != expected_W_in:
            pytest.skip(f"Weight input channels {W_in} should be {expected_W_in} for groups={groups}")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
            auto_pad="NOTSET",
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 22 and 11 due to numerical differences
        if opset_version == 22:
            rtol = 1e-3
            atol = 1e-4
        elif opset_version == 11:
            rtol = 1e-3
            atol = 1e-4
        else:
            rtol = 1e-4
            atol = 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify shape and dtype
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    @pytest.mark.parametrize("opset_version", [1, 11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape",
        [
            ((1, 3, 32, 32), (16, 3, 3, 3), (3, 3)),
            ((1, 64, 32, 32), (128, 64, 3, 3), (3, 3)),
        ],
    )
    @pytest.mark.parametrize("bias", [True, False])
    def test_conv2d_bias(self, opset_version, input_shape, weight_shape, kernel_shape, bias):
        """Test Conv2d with and without bias."""
        if opset_version == 1:
            pytest.skip("Opset 1 may not be fully supported by ONNX Runtime")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            auto_pad="NOTSET",
            bias=bias,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 22 and 11 due to numerical differences
        if opset_version == 22:
            rtol = 1e-3
            atol = 1e-4
        elif opset_version == 11:
            rtol = 1e-3
            atol = 1e-4
        else:
            rtol = 1e-4
            atol = 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify shape and dtype
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
    # CONV1D TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape",
        [
            # Basic 1D convolutions
            ((1, 3, 32), (16, 3, 3), (3,)),  # Standard 1D conv
            ((1, 3, 32), (16, 3, 5), (5,)),  # 5-tap kernel
            ((1, 3, 32), (16, 3, 7), (7,)),  # 7-tap kernel
            ((1, 3, 32), (16, 3, 1), (1,)),  # 1-tap pointwise
            # Different input sizes
            ((1, 3, 64), (32, 3, 3), (3,)),  # Longer sequence
            ((1, 3, 16), (8, 3, 3), (3,)),  # Shorter sequence
            ((1, 3, 128), (16, 3, 3), (3,)),  # Very long sequence
            # Different channel counts
            ((1, 1, 32), (8, 1, 3), (3,)),  # Single channel
            ((1, 64, 32), (128, 64, 3), (3,)),  # More channels
            # Batch size > 1
            ((4, 3, 32), (16, 3, 3), (3,)),  # Batch of 4
            ((8, 3, 32), (16, 3, 3), (3,)),  # Batch of 8
        ],
    )
    @pytest.mark.parametrize("stride", [1, 2, 3, (1,), (2,)])
    @pytest.mark.parametrize("padding", [0, 1, 2, (1,), (2,), (1, 2)])
    @pytest.mark.parametrize("dilation", [1, 2, (1,), (2,)])
    def test_conv1d_basic(self, opset_version, input_shape, weight_shape, kernel_shape, stride, padding, dilation):
        """Test basic Conv1d operations with various configurations."""
        if opset_version == 1:
            pytest.skip("Opset 1 may not be fully supported by ONNX Runtime")

        # Validate weight shape
        C_in = input_shape[1]
        C_out, W_in, *W_kernel = weight_shape
        if W_in != C_in:
            pytest.skip(f"Weight input channels {W_in} doesn't match input channels {C_in}")
        if tuple(W_kernel) != kernel_shape:
            pytest.skip(f"Weight kernel shape {W_kernel} doesn't match kernel_shape {kernel_shape}")

        # Normalize stride
        if isinstance(stride, int):
            stride_val = stride
        elif isinstance(stride, (list, tuple)):
            stride_val = stride[0] if len(stride) > 0 else 1
        else:
            stride_val = 1

        # Get effective padding
        if isinstance(padding, int):
            pad_w = padding * 2
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 1:
                pad_w = padding[0] * 2
            elif len(padding) == 2:
                pad_w = padding[0] + padding[1]
            else:
                pad_w = 0
        else:
            pad_w = 0

        # Normalize dilation
        if isinstance(dilation, int):
            dil_w = dilation
        elif isinstance(dilation, (list, tuple)):
            dil_w = dilation[0] if len(dilation) > 0 else 1
        else:
            dil_w = 1

        # Calculate output size
        W_in_size = input_shape[2]
        kW = kernel_shape[0]
        effective_kW = (kW - 1) * dil_w + 1
        W_out = (W_in_size + pad_w - effective_kW) // stride_val + 1

        # Skip if output would be invalid
        if W_out <= 0:
            pytest.skip(f"Invalid output width: {W_out}")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            auto_pad="NOTSET",
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Verify graph structure
        verify_tir_graph_structure(tir_graph, onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 22 and 11 due to numerical differences
        if opset_version == 22:
            rtol = 1e-3
            atol = 1e-4
        elif opset_version == 11:
            rtol = 1e-3
            atol = 1e-4
        else:
            rtol = 1e-4
            atol = 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify output shape and dtype
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    @pytest.mark.parametrize("opset_version", [1, 11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape",
        [
            ((1, 3, 32), (16, 3, 3), (3,)),
            ((1, 64, 32), (128, 64, 3), (3,)),
            ((4, 3, 64), (32, 3, 5), (5,)),
        ],
    )
    @pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
    @pytest.mark.parametrize("stride", [1, 2, (1,), (2,)])
    def test_conv1d_auto_pad(self, opset_version, input_shape, weight_shape, kernel_shape, auto_pad, stride):
        """Test Conv1d with auto_pad modes."""
        if opset_version == 1:
            pytest.skip("Opset 1 may not be fully supported by ONNX Runtime")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,  # Ignored when auto_pad is set
            dilation=1,
            groups=1,
            auto_pad=auto_pad,
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 22 and 11 due to numerical differences
        if opset_version == 22:
            rtol = 1e-3
            atol = 1e-4
        elif opset_version == 11:
            rtol = 1e-3
            atol = 1e-4
        else:
            rtol = 1e-4
            atol = 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify shape and dtype
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    @pytest.mark.parametrize("opset_version", [1, 11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape, groups",
        [
            # Grouped convolutions
            ((1, 64, 32), (128, 32, 3), (3,), 2),  # groups=2
            ((1, 64, 32), (128, 16, 3), (3,), 4),  # groups=4
            ((1, 64, 32), (128, 8, 3), (3,), 8),  # groups=8
            # Depthwise convolution
            ((1, 64, 32), (64, 1, 3), (3,), 64),  # Depthwise
            ((1, 32, 32), (32, 1, 3), (3,), 32),  # Depthwise
        ],
    )
    @pytest.mark.parametrize("stride", [1, 2, (1,), (2,)])
    @pytest.mark.parametrize("padding", [0, 1, (1,), (1, 2)])
    def test_conv1d_groups(self, opset_version, input_shape, weight_shape, kernel_shape, groups, stride, padding):
        """Test Conv1d with grouped convolutions."""
        if opset_version == 1:
            pytest.skip("Opset 1 may not be fully supported by ONNX Runtime")

        # Validate groups
        C_in = input_shape[1]
        C_out, W_in, *W_kernel = weight_shape

        if C_in % groups != 0:
            pytest.skip(f"Input channels {C_in} not divisible by groups {groups}")
        if C_out % groups != 0:
            pytest.skip(f"Output channels {C_out} not divisible by groups {groups}")

        expected_W_in = C_in // groups
        if W_in != expected_W_in:
            pytest.skip(f"Weight input channels {W_in} should be {expected_W_in} for groups={groups}")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
            auto_pad="NOTSET",
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 22 and 11 due to numerical differences
        if opset_version == 22:
            rtol = 1e-3
            atol = 1e-4
        elif opset_version == 11:
            rtol = 1e-3
            atol = 1e-4
        else:
            rtol = 1e-4
            atol = 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify shape and dtype
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
    # EDGE CASES AND SPECIAL COMBINATIONS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape, stride, padding",
        [
            # Large stride cases
            ((1, 3, 32, 32), (16, 3, 3, 3), (3, 3), 4, 0),
            ((1, 3, 64, 64), (16, 3, 3, 3), (3, 3), 8, 0),
            # Large padding cases
            ((1, 3, 16, 16), (16, 3, 3, 3), (3, 3), 1, 4),
            ((1, 3, 16, 16), (16, 3, 3, 3), (3, 3), 1, (2, 4, 2, 4)),  # Asymmetric
            # Small input cases
            ((1, 3, 4, 4), (16, 3, 3, 3), (3, 3), 1, 0),
            ((1, 3, 5, 5), (16, 3, 3, 3), (3, 3), 1, 1),
        ],
    )
    def test_conv2d_edge_cases(self, opset_version, input_shape, weight_shape, kernel_shape, stride, padding):
        """Test Conv2d edge cases with extreme parameter values."""
        # Validate output would be valid
        H_in, W_in = input_shape[2], input_shape[3]
        kH, kW = kernel_shape

        # Get padding
        if isinstance(padding, int):
            pad_h = pad_w = padding * 2
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:
                pad_h = padding[0] * 2
                pad_w = padding[1] * 2
            elif len(padding) == 4:
                pad_h = padding[0] + padding[2]
                pad_w = padding[1] + padding[3]
            else:
                pad_h = pad_w = 0
        else:
            pad_h = pad_w = 0

        # Normalize stride
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h = stride[0] if len(stride) > 0 else 1
            stride_w = stride[1] if len(stride) > 1 else stride_h

        H_out = (H_in + pad_h - kH) // stride_h + 1
        W_out = (W_in + pad_w - kW) // stride_w + 1

        if H_out <= 0 or W_out <= 0:
            pytest.skip(f"Invalid output shape: ({H_out}, {W_out})")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            auto_pad="NOTSET",
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 22 and 11 due to numerical differences
        if opset_version == 22:
            rtol = 1e-3
            atol = 1e-4
        elif opset_version == 11:
            rtol = 1e-3
            atol = 1e-4
        else:
            rtol = 1e-4
            atol = 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify shape and dtype
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    @pytest.mark.parametrize("opset_version", [11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape, dilation",
        [
            # Dilated convolutions
            ((1, 3, 32, 32), (16, 3, 3, 3), (3, 3), 2),
            ((1, 3, 32, 32), (16, 3, 3, 3), (3, 3), (2, 2)),
            ((1, 3, 64, 64), (16, 3, 5, 5), (5, 5), 3),
            ((1, 3, 32, 32), (16, 3, 3, 3), (3, 3), (1, 2)),  # Asymmetric dilation
        ],
    )
    @pytest.mark.parametrize("stride", [1, (1, 1)])
    @pytest.mark.parametrize("padding", [0, 2, (2, 2)])
    def test_conv2d_dilation(self, opset_version, input_shape, weight_shape, kernel_shape, dilation, stride, padding):
        """Test Conv2d with dilation."""
        # Validate output would be valid
        H_in, W_in = input_shape[2], input_shape[3]
        kH, kW = kernel_shape

        # Normalize dilation
        if isinstance(dilation, int):
            dil_h = dil_w = dilation
        else:
            dil_h = dilation[0] if len(dilation) > 0 else 1
            dil_w = dilation[1] if len(dilation) > 1 else dil_h

        # Get padding
        if isinstance(padding, int):
            pad_h = pad_w = padding * 2
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:
                pad_h = padding[0] * 2
                pad_w = padding[1] * 2
            else:
                pad_h = pad_w = 0
        else:
            pad_h = pad_w = 0

        # Normalize stride
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h = stride[0] if len(stride) > 0 else 1
            stride_w = stride[1] if len(stride) > 1 else stride_h

        effective_kH = (kH - 1) * dil_h + 1
        effective_kW = (kW - 1) * dil_w + 1
        H_out = (H_in + pad_h - effective_kH) // stride_h + 1
        W_out = (W_in + pad_w - effective_kW) // stride_w + 1

        if H_out <= 0 or W_out <= 0:
            pytest.skip(f"Invalid output shape: ({H_out}, {W_out})")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            auto_pad="NOTSET",
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 22 and 11 due to numerical differences
        if opset_version == 22:
            rtol = 1e-3
            atol = 1e-4
        elif opset_version == 11:
            rtol = 1e-3
            atol = 1e-4
        else:
            rtol = 1e-4
            atol = 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify shape and dtype
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
    # CONV3D TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape, stride, padding, dilation, groups, bias",
        [
            # Basic Conv3d cases
            ((1, 3, 8, 8, 8), (16, 3, 3, 3, 3), (3, 3, 3), 1, 0, 1, 1, True),
            ((1, 3, 8, 8, 8), (16, 3, 3, 3, 3), (3, 3, 3), 1, 1, 1, 1, True),
            ((1, 3, 8, 8, 8), (16, 3, 3, 3, 3), (3, 3, 3), 1, 0, 1, 1, False),
            ((2, 3, 10, 10, 10), (16, 3, 3, 3, 3), (3, 3, 3), 1, 0, 1, 1, True),
            # Different kernel sizes
            ((1, 3, 8, 8, 8), (16, 3, 5, 5, 5), (5, 5, 5), 1, 0, 1, 1, True),
            ((1, 3, 8, 8, 8), (16, 3, 1, 1, 1), (1, 1, 1), 1, 0, 1, 1, True),
            # Different strides
            ((1, 3, 16, 16, 16), (16, 3, 3, 3, 3), (3, 3, 3), 2, 0, 1, 1, True),
            ((1, 3, 16, 16, 16), (16, 3, 3, 3, 3), (3, 3, 3), (2, 2, 2), 0, 1, 1, True),
            ((1, 3, 16, 16, 16), (16, 3, 3, 3, 3), (3, 3, 3), (1, 2, 1), 0, 1, 1, True),
            # Different padding
            ((1, 3, 8, 8, 8), (16, 3, 3, 3, 3), (3, 3, 3), 1, 2, 1, 1, True),
            ((1, 3, 8, 8, 8), (16, 3, 3, 3, 3), (3, 3, 3), 1, (1, 1, 1), 1, 1, True),
            ((1, 3, 8, 8, 8), (16, 3, 3, 3, 3), (3, 3, 3), 1, (1, 2, 1, 2, 1, 2), 1, 1, True),  # Asymmetric
            # Different dilations
            ((1, 3, 16, 16, 16), (16, 3, 3, 3, 3), (3, 3, 3), 1, 0, 2, 1, True),
            ((1, 3, 16, 16, 16), (16, 3, 3, 3, 3), (3, 3, 3), 1, 0, (2, 2, 2), 1, True),
            ((1, 3, 16, 16, 16), (16, 3, 3, 3, 3), (3, 3, 3), 1, 0, (1, 2, 1), 1, True),
            # Groups
            ((1, 6, 8, 8, 8), (18, 2, 3, 3, 3), (3, 3, 3), 1, 0, 1, 3, True),  # Fixed: 18 divisible by 3
            ((1, 8, 8, 8, 8), (16, 2, 3, 3, 3), (3, 3, 3), 1, 0, 1, 4, True),
        ],
    )
    def test_conv3d_basic(
        self, opset_version, input_shape, weight_shape, kernel_shape, stride, padding, dilation, groups, bias
    ):
        """Test basic Conv3d operations with various parameters."""
        # Validate output would be valid
        D_in, H_in, W_in = input_shape[2], input_shape[3], input_shape[4]
        kD, kH, kW = kernel_shape

        # Normalize stride
        if isinstance(stride, int):
            stride_d = stride_h = stride_w = stride
        elif isinstance(stride, (list, tuple)):
            if len(stride) == 1:
                stride_d = stride_h = stride_w = stride[0]
            elif len(stride) >= 3:
                stride_d, stride_h, stride_w = stride[0], stride[1], stride[2]
            else:
                stride_d = stride[0] if len(stride) > 0 else 1
                stride_h = stride[1] if len(stride) > 1 else stride_d
                stride_w = stride[2] if len(stride) > 2 else stride_h
        else:
            stride_d = stride_h = stride_w = 1

        # Normalize dilation
        if isinstance(dilation, int):
            dil_d = dil_h = dil_w = dilation
        elif isinstance(dilation, (list, tuple)):
            if len(dilation) == 1:
                dil_d = dil_h = dil_w = dilation[0]
            elif len(dilation) >= 3:
                dil_d, dil_h, dil_w = dilation[0], dilation[1], dilation[2]
            else:
                dil_d = dilation[0] if len(dilation) > 0 else 1
                dil_h = dilation[1] if len(dilation) > 1 else dil_d
                dil_w = dilation[2] if len(dilation) > 2 else dil_h
        else:
            dil_d = dil_h = dil_w = 1

        # Get padding
        if isinstance(padding, int):
            pad_d = pad_h = pad_w = padding * 2
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 3:
                pad_d = padding[0] * 2
                pad_h = padding[1] * 2
                pad_w = padding[2] * 2
            elif len(padding) == 6:
                pad_d = padding[0] + padding[3]  # D_begin + D_end
                pad_h = padding[1] + padding[4]  # H_begin + H_end
                pad_w = padding[2] + padding[5]  # W_begin + W_end
            else:
                pad_d = pad_h = pad_w = 0
        else:
            pad_d = pad_h = pad_w = 0

        effective_kD = (kD - 1) * dil_d + 1
        effective_kH = (kH - 1) * dil_h + 1
        effective_kW = (kW - 1) * dil_w + 1

        D_out = (D_in + pad_d - effective_kD) // stride_d + 1
        H_out = (H_in + pad_h - effective_kH) // stride_h + 1
        W_out = (W_in + pad_w - effective_kW) // stride_w + 1

        if D_out <= 0 or H_out <= 0 or W_out <= 0:
            pytest.skip(f"Invalid output shape: ({D_out}, {H_out}, {W_out})")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            auto_pad="NOTSET",
            bias=bias,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 11 and 22 due to numerical differences
        rtol = 1e-3 if opset_version in (11, 22) else 1e-4
        atol = 1e-4 if opset_version in (11, 22) else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify output shape and dtype
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    @pytest.mark.parametrize("opset_version", [11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape, auto_pad",
        [
            # SAME_UPPER
            ((1, 3, 8, 8, 8), (16, 3, 3, 3, 3), (3, 3, 3), "SAME_UPPER"),
            ((1, 3, 10, 10, 10), (16, 3, 3, 3, 3), (3, 3, 3), "SAME_UPPER"),
            ((1, 3, 8, 8, 8), (16, 3, 5, 5, 5), (5, 5, 5), "SAME_UPPER"),
            # SAME_LOWER
            ((1, 3, 8, 8, 8), (16, 3, 3, 3, 3), (3, 3, 3), "SAME_LOWER"),
            ((1, 3, 10, 10, 10), (16, 3, 3, 3, 3), (3, 3, 3), "SAME_LOWER"),
            # VALID
            ((1, 3, 8, 8, 8), (16, 3, 3, 3, 3), (3, 3, 3), "VALID"),
            ((1, 3, 10, 10, 10), (16, 3, 5, 5, 5), (5, 5, 5), "VALID"),
        ],
    )
    @pytest.mark.parametrize("stride", [1, (1, 1, 1), 2, (2, 2, 2)])
    def test_conv3d_auto_pad(self, opset_version, input_shape, weight_shape, kernel_shape, auto_pad, stride):
        """Test Conv3d with auto_pad attribute."""
        # Normalize stride
        if isinstance(stride, int):
            stride_d = stride_h = stride_w = stride
        elif isinstance(stride, (list, tuple)):
            if len(stride) == 1:
                stride_d = stride_h = stride_w = stride[0]
            elif len(stride) >= 3:
                stride_d, stride_h, stride_w = stride[0], stride[1], stride[2]
            else:
                stride_d = stride[0] if len(stride) > 0 else 1
                stride_h = stride[1] if len(stride) > 1 else stride_d
                stride_w = stride[2] if len(stride) > 2 else stride_h
        else:
            stride_d = stride_h = stride_w = 1

        D_in, H_in, W_in = input_shape[2], input_shape[3], input_shape[4]
        kD, kH, kW = kernel_shape

        # Calculate expected output shape
        if auto_pad == "VALID":
            D_out = (D_in - kD) // stride_d + 1
            H_out = (H_in - kH) // stride_h + 1
            W_out = (W_in - kW) // stride_w + 1
        else:  # SAME_UPPER or SAME_LOWER
            D_out = (D_in + stride_d - 1) // stride_d
            H_out = (H_in + stride_h - 1) // stride_h
            W_out = (W_in + stride_w - 1) // stride_w

        if D_out <= 0 or H_out <= 0 or W_out <= 0:
            pytest.skip(f"Invalid output shape: ({D_out}, {H_out}, {W_out})")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,  # Ignored when auto_pad is set
            dilation=1,
            groups=1,
            auto_pad=auto_pad,
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 11 and 22 due to numerical differences
        rtol = 1e-3 if opset_version in (11, 22) else 1e-4
        atol = 1e-4 if opset_version in (11, 22) else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify output shape and dtype
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"

    @pytest.mark.parametrize("opset_version", [11, 22])
    @pytest.mark.parametrize(
        "input_shape, weight_shape, kernel_shape, groups",
        [
            # Depthwise and grouped convolutions
            ((1, 6, 8, 8, 8), (6, 1, 3, 3, 3), (3, 3, 3), 6),  # Depthwise
            ((1, 8, 8, 8, 8), (16, 2, 3, 3, 3), (3, 3, 3), 4),
            ((1, 12, 8, 8, 8), (24, 4, 3, 3, 3), (3, 3, 3), 3),
            ((1, 16, 10, 10, 10), (32, 4, 5, 5, 5), (5, 5, 5), 4),
        ],
    )
    @pytest.mark.parametrize("stride", [1, (1, 1, 1), 2])
    @pytest.mark.parametrize("padding", [0, 1, (1, 1, 1)])
    def test_conv3d_groups(self, opset_version, input_shape, weight_shape, kernel_shape, groups, stride, padding):
        """Test Conv3d with groups (including depthwise convolution)."""
        # Validate output would be valid
        D_in, H_in, W_in = input_shape[2], input_shape[3], input_shape[4]
        kD, kH, kW = kernel_shape

        # Normalize stride
        if isinstance(stride, int):
            stride_d = stride_h = stride_w = stride
        elif isinstance(stride, (list, tuple)):
            if len(stride) == 1:
                stride_d = stride_h = stride_w = stride[0]
            elif len(stride) >= 3:
                stride_d, stride_h, stride_w = stride[0], stride[1], stride[2]
            else:
                stride_d = stride[0] if len(stride) > 0 else 1
                stride_h = stride[1] if len(stride) > 1 else stride_d
                stride_w = stride[2] if len(stride) > 2 else stride_h
        else:
            stride_d = stride_h = stride_w = 1

        # Get padding
        if isinstance(padding, int):
            pad_d = pad_h = pad_w = padding * 2
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 3:
                pad_d = padding[0] * 2
                pad_h = padding[1] * 2
                pad_w = padding[2] * 2
            else:
                pad_d = pad_h = pad_w = 0
        else:
            pad_d = pad_h = pad_w = 0

        D_out = (D_in + pad_d - kD) // stride_d + 1
        H_out = (H_in + pad_h - kH) // stride_h + 1
        W_out = (W_in + pad_w - kW) // stride_w + 1

        if D_out <= 0 or H_out <= 0 or W_out <= 0:
            pytest.skip(f"Invalid output shape: ({D_out}, {H_out}, {W_out})")

        # Create model
        onnx_model = TestConv._create_conv_model(
            opset_version=opset_version,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
            auto_pad="NOTSET",
            bias=True,
            dtype=onnx.TensorProto.FLOAT,
        )

        # Create input data
        input_data = {"X": np.random.randn(*input_shape).astype(np.float32)}

        # Transpile and compare
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)

        # Compare with ONNX Runtime
        # Use slightly relaxed tolerance for opset 11 and 22 due to numerical differences
        rtol = 1e-3 if opset_version in (11, 22) else 1e-4
        atol = 1e-4 if opset_version in (11, 22) else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)

        # Verify results
        assert len(comparison["errors"]) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison["matches"].values()), "Output values don't match"

        # Verify output shape and dtype
        output_name = onnx_model.graph.output[0].name
        tir_output = comparison["tir_outputs"][output_name]
        onnx_output = comparison["onnx_outputs"][output_name]

        assert (
            tir_output.shape == onnx_output.shape
        ), f"Shape mismatch: TIR {tir_output.shape} vs ONNX {onnx_output.shape}"
        assert (
            tir_output.dtype == onnx_output.dtype
        ), f"Dtype mismatch: TIR {tir_output.dtype} vs ONNX {onnx_output.dtype}"
