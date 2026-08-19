# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import numpy as np
import onnx
import onnx.helper as oh
from onnx import helper, TensorProto, numpy_helper
from onnx import TensorProto as otp

import forge
from forge._C import MLIRConfig
from forge.verify.verify import verify


ONNX_OPSET_VERSION = 21
opset_imports = [helper.make_operatorsetid("", ONNX_OPSET_VERSION)]


@pytest.mark.push
def test_add():
    input_A = helper.make_tensor_value_info("input_A", TensorProto.FLOAT, [2, 32, 32])
    input_B = helper.make_tensor_value_info("input_B", TensorProto.FLOAT, [2, 32, 32])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 32, 32])

    add_node = helper.make_node(
        "Add",
        inputs=["input_A", "input_B"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        nodes=[add_node],
        name="AddGraph",
        inputs=[input_A, input_B],
        outputs=[output],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="AddModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand([2, 32, 32]), torch.rand([2, 32, 32])]

    onnx_module = forge.OnnxModule("add", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


@pytest.mark.push
def test_arithmetic():
    input_A = helper.make_tensor_value_info("input_A", TensorProto.FLOAT, [2, 32, 32])
    input_B = helper.make_tensor_value_info("input_B", TensorProto.FLOAT, [2, 32, 32])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 32, 32])

    sqrt_node = helper.make_node(
        "Sqrt",
        inputs=["input_A"],
        outputs=["Sqrt_A"],
    )
    exp_node = helper.make_node(
        "Exp",
        inputs=["input_B"],
        outputs=["Exp_B"],
    )
    add_node = helper.make_node(
        "Add",
        inputs=["Sqrt_A", "Exp_B"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        nodes=[sqrt_node, exp_node, add_node],
        name="ArithGraph",
        inputs=[input_A, input_B],
        outputs=[output],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="ArithModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand([2, 32, 32]), torch.rand([2, 32, 32])]

    onnx_module = forge.OnnxModule("arith", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


@pytest.mark.push
def test_matmul():
    input_A = helper.make_tensor_value_info("input_A", TensorProto.FLOAT, [32, 64])
    input_B = helper.make_tensor_value_info("input_B", TensorProto.FLOAT, [64, 32])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32])

    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input_A", "input_B"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        nodes=[matmul_node],
        name="MatMulGraph",
        inputs=[input_A, input_B],
        outputs=[output],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="MatMulModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand([32, 64]), torch.rand([64, 32])]

    onnx_module = forge.OnnxModule("matmul", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


@pytest.mark.push
def test_squeeze():
    input_A = helper.make_tensor_value_info("input_A", TensorProto.FLOAT, [1, 32, 32])
    input_B = helper.make_tensor_value_info("input_B", TensorProto.FLOAT, [1, 32, 32])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32])

    squeeze_a_node = helper.make_node(
        "Squeeze",
        inputs=["input_A"],
        outputs=["squeezed_A"],
    )

    squeeze_b_node = helper.make_node(
        "Squeeze",
        inputs=["input_B"],
        outputs=["squeezed_B"],
    )

    transpose_a_node = helper.make_node("Transpose", inputs=["squeezed_A"], outputs=["transposed_A"])

    add_node = helper.make_node("Add", inputs=["transposed_A", "squeezed_B"], outputs=["output"])

    graph = helper.make_graph(
        nodes=[squeeze_a_node, squeeze_b_node, transpose_a_node, add_node],
        name="SqueezeGraph",
        inputs=[input_A, input_B],
        outputs=[output],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="SqueezeModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand([1, 32, 32]), torch.rand([1, 32, 32])]

    onnx_module = forge.OnnxModule("squeeze", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


@pytest.mark.push
def test_flatten():
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 32, 32])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 1024])

    flatten_node = helper.make_node(
        "Flatten",
        inputs=["input"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        nodes=[flatten_node],
        name="FlattenGraph",
        inputs=[input],
        outputs=[output],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="FlattenModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand([2, 32, 32])]

    onnx_module = forge.OnnxModule("flatten", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


@pytest.mark.push
def test_linear_layer():
    input_features, output_dim = (784, 10)

    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_features])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_dim])

    weight_data = np.random.rand(input_features, output_dim).astype(np.float32)
    bias_data = np.random.rand(output_dim).astype(np.float32)
    weight_initializer = numpy_helper.from_array(weight_data, name="weight")
    bias_initializer = numpy_helper.from_array(bias_data, name="bias")

    matmul_node = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["matmul_A"])

    add_node = helper.make_node("Add", inputs=["matmul_A", "bias"], outputs=["output"])

    graph = helper.make_graph(
        nodes=[matmul_node, add_node],
        name="LinearLayerGraph",
        inputs=[input],
        outputs=[output],
        initializer=[weight_initializer, bias_initializer],
    )

    onnx_model = helper.make_model(
        graph,
        producer_name="LinearLayerModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand([1, input_features])]

    onnx_module = forge.OnnxModule("linear", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


@pytest.mark.push
def test_multiple_layers():
    num_classes = 10

    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, num_classes])

    conv1_weight = np.random.rand(16, 3, 3, 3).astype(np.float32)
    conv1_bias = np.random.rand(16).astype(np.float32)
    conv2_weight = np.random.rand(32, 16, 3, 3).astype(np.float32)
    conv2_bias = np.random.rand(32).astype(np.float32)
    fc1_weight = np.random.rand(32 * 8 * 8, 128).astype(np.float32)
    fc1_bias = np.random.rand(128).astype(np.float32)
    fc2_weight = np.random.rand(128, num_classes).astype(np.float32)
    fc2_bias = np.random.rand(num_classes).astype(np.float32)

    initializer = [
        numpy_helper.from_array(conv1_weight, "conv1_weight"),
        numpy_helper.from_array(conv1_bias, "conv1_bias"),
        numpy_helper.from_array(conv2_weight, "conv2_weight"),
        numpy_helper.from_array(conv2_bias, "conv2_bias"),
        numpy_helper.from_array(fc1_weight, "fc1_weight"),
        numpy_helper.from_array(fc1_bias, "fc1_bias"),
        numpy_helper.from_array(fc2_weight, "fc2_weight"),
        numpy_helper.from_array(fc2_bias, "fc2_bias"),
    ]

    nodes = [
        helper.make_node(
            "Conv", ["input", "conv1_weight", "conv1_bias"], ["conv1_out"], pads=[1, 1, 1, 1], strides=[1, 1]
        ),
        helper.make_node("Relu", ["conv1_out"], ["relu1_out"]),
        helper.make_node("MaxPool", ["relu1_out"], ["pool1_out"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node(
            "Conv", ["pool1_out", "conv2_weight", "conv2_bias"], ["conv2_out"], pads=[1, 1, 1, 1], strides=[1, 1]
        ),
        helper.make_node("Relu", ["conv2_out"], ["relu2_out"]),
        helper.make_node("MaxPool", ["relu2_out"], ["pool2_out"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Flatten", ["pool2_out"], ["flatten_out"], axis=1),
        helper.make_node("MatMul", ["flatten_out", "fc1_weight"], ["fc1_matmul_out"]),
        helper.make_node("Add", ["fc1_matmul_out", "fc1_bias"], ["fc1_out"]),
        helper.make_node("Relu", ["fc1_out"], ["relu_fc1_out"]),
        helper.make_node("MatMul", ["relu_fc1_out", "fc2_weight"], ["fc2_matmul_out"]),
        helper.make_node("Add", ["fc2_matmul_out", "fc2_bias"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="CNNClassifierGraph",
        inputs=[input],
        outputs=[output],
        initializer=initializer,
    )

    onnx_model = helper.make_model(
        graph,
        producer_name="CNNClassifierModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand([1, 3, 32, 32])]

    onnx_module = forge.OnnxModule("multiple_linears", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


@pytest.mark.push
def test_mnist_linear():
    input_size = 784
    hidden_size = 512
    output_size = 10

    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_size])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_size])

    fc1_weight = np.random.rand(input_size, hidden_size).astype(np.float32)
    fc1_bias = np.random.rand(hidden_size).astype(np.float32)
    fc2_weight = np.random.rand(hidden_size, hidden_size).astype(np.float32)
    fc2_bias = np.random.rand(hidden_size).astype(np.float32)
    fc3_weight = np.random.rand(hidden_size, output_size).astype(np.float32)
    fc3_bias = np.random.rand(output_size).astype(np.float32)

    initializer = [
        numpy_helper.from_array(fc1_weight, "fc1_weight"),
        numpy_helper.from_array(fc1_bias, "fc1_bias"),
        numpy_helper.from_array(fc2_weight, "fc2_weight"),
        numpy_helper.from_array(fc2_bias, "fc2_bias"),
        numpy_helper.from_array(fc3_weight, "fc3_weight"),
        numpy_helper.from_array(fc3_bias, "fc3_bias"),
    ]

    nodes = [
        helper.make_node("MatMul", ["input", "fc1_weight"], ["fc1_matmul_out"]),
        helper.make_node("Add", ["fc1_matmul_out", "fc1_bias"], ["fc1_out"]),
        helper.make_node("Relu", ["fc1_out"], ["relu1_out"]),
        helper.make_node("MatMul", ["relu1_out", "fc2_weight"], ["fc2_matmul_out"]),
        helper.make_node("Add", ["fc2_matmul_out", "fc2_bias"], ["fc2_out"]),
        helper.make_node("Relu", ["fc2_out"], ["relu2_out"]),
        helper.make_node("MatMul", ["relu2_out", "fc3_weight"], ["fc3_matmul_out"]),
        helper.make_node("Add", ["fc3_matmul_out", "fc3_bias"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="MNISTLinearGraph",
        inputs=[input],
        outputs=[output],
        initializer=initializer,
    )

    onnx_model = helper.make_model(
        graph,
        producer_name="MNISTLinearModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand([1, 784])]

    onnx_module = forge.OnnxModule("mnist_linear", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


@pytest.mark.push
def test_batchnorm():
    num_features = 32
    input_shape = [1, 32, 56, 56]

    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)

    scale = np.random.rand(num_features).astype(np.float32)
    bias = np.random.rand(num_features).astype(np.float32)
    mean = np.random.rand(num_features).astype(np.float32)
    var = np.random.rand(num_features).astype(np.float32)

    initializer = [
        numpy_helper.from_array(scale, "scale"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(mean, "mean"),
        numpy_helper.from_array(var, "var"),
    ]

    batch_norm_node = helper.make_node(
        "BatchNormalization", inputs=["input", "scale", "bias", "mean", "var"], outputs=["output"], epsilon=1e-5
    )

    graph = helper.make_graph(
        nodes=[batch_norm_node],
        name="BatchNormGraph",
        inputs=[input],
        outputs=[output],
        initializer=initializer,
    )

    onnx_model = helper.make_model(
        graph,
        producer_name="BatchNormModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand(input_shape)]

    onnx_module = forge.OnnxModule("batchnorm", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


@pytest.mark.push
def test_convbn():
    in_c = 3
    out_c = 64
    filter_size = 3
    stride = 1
    padding = 1
    num_groups = 1
    input_shape = [1, in_c, 64, 64]

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, out_c, 64, 64])

    conv_weight = np.random.rand(out_c, in_c // num_groups, filter_size, filter_size).astype(np.float32)
    conv_bias = np.random.rand(out_c).astype(np.float32)

    scale = np.random.rand(out_c).astype(np.float32)
    bias = np.random.rand(out_c).astype(np.float32)
    mean = np.random.rand(out_c).astype(np.float32)
    var = np.random.rand(out_c).astype(np.float32)

    initializer = [
        numpy_helper.from_array(conv_weight, "conv_weight"),
        numpy_helper.from_array(conv_bias, "conv_bias"),
        numpy_helper.from_array(scale, "bn_scale"),
        numpy_helper.from_array(bias, "bn_bias"),
        numpy_helper.from_array(mean, "bn_mean"),
        numpy_helper.from_array(var, "bn_var"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            inputs=["input", "conv_weight", "conv_bias"],
            outputs=["conv_out"],
            kernel_shape=[filter_size, filter_size],
            strides=[stride, stride],
            pads=[padding, padding, padding, padding],
            group=num_groups,
        ),
        helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            epsilon=1e-5,
        ),
        helper.make_node(
            "Relu",
            inputs=["bn_out"],
            outputs=["output"],
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="ConvBNLayerGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializer,
    )

    onnx_model = helper.make_model(
        graph,
        producer_name="ConvBNLayerModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand(input_shape)]

    onnx_module = forge.OnnxModule("convbn", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)

    verify(inputs, onnx_module, compiled_model)


def _grid_with_padding_coverage(shape, seed=1):
    """
    Coordinates spanning [-1.1, 1.1], so roughly 9% of them land outside the
    image and must come back as zeros under padding_mode="zeros".

    Sampling uniformly inside [-1, 1] never leaves the image, which leaves the
    padding path completely untested. Out-of-range coordinates are not an edge
    case for this op -- the model's own lookup tables carry 30-75% of their
    coordinates outside the border (see _bev_lut). This matches the convention
    in tt-metal's own grid_sample suite, which draws grids as
    `torch.rand(...) * 2.2 - 1.1` for exactly this reason.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.1, 1.1, shape).astype(np.float32)


@pytest.mark.push
@pytest.mark.parametrize(
    "data_shape, grid_shape, mode, padding_mode, align_corners",
    [
        # data_shape is NCHW (N, C, H_in, W_in); the tt-metal kernel receives NHWC
        # so the channel dim C must be divisible by TILE_WIDTH=32.
        pytest.param((1, 32, 8, 8), (1, 4, 4, 2), "bilinear", "zeros", 1),
        pytest.param((1, 32, 8, 8), (1, 4, 4, 2), "bilinear", "zeros", 0),
        pytest.param((1, 32, 8, 8), (1, 4, 4, 2), "nearest", "zeros", 1),
        pytest.param((1, 32, 8, 8), (1, 4, 4, 2), "nearest", "zeros", 0),
        pytest.param((1, 64, 96, 96), (1, 128, 64, 2), "bilinear", "zeros", 1),
        pytest.param((1, 64, 96, 96), (1, 128, 64, 2), "nearest", "zeros", 1),
    ],
)
def test_gridsample(data_shape, grid_shape, mode, padding_mode, align_corners):
    n, c, h, w = data_shape
    gn, gh, gw, _ = grid_shape

    data_vi = oh.make_tensor_value_info("data", otp.FLOAT, list(data_shape))
    grid_vi = oh.make_tensor_value_info("grid", otp.FLOAT, list(grid_shape))
    out_vi = oh.make_tensor_value_info("output", otp.FLOAT, [n, c, gh, gw])
    node = oh.make_node(
        "GridSample",
        inputs=["data", "grid"],
        outputs=["output"],
        align_corners=align_corners,
        mode=mode,
        padding_mode=padding_mode,
    )
    graph = oh.make_graph([node], "gridsample", [data_vi, grid_vi], [out_vi])
    onnx_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    onnx.checker.check_model(onnx_model)

    rng = np.random.default_rng(0)
    data = torch.from_numpy(rng.standard_normal(data_shape).astype(np.float32))
    grid = torch.from_numpy(_grid_with_padding_coverage(grid_shape))

    inputs = [data, grid]

    framework_model = forge.OnnxModule("gridsample", onnx_model)
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name="gridsample")
    verify(inputs, framework_model=framework_model, compiled_model=compiled_model)



# Lookup-table geometry shared by both BEV transform blocks: a (1, 128, 64, 8, 2)
# table holding K=8 coordinate pairs per output cell.
_LUT_H, _LUT_W, _LUT_K = 128, 64, 8


def _bev_lut(shape, seed=0):
    """
    Synthesise a coordinate table with the same character as the model's real
    ones (BEV_model/input_samples/.../input_lut_*.npy).

    Those tables are NOT confined to [-1, 1]: measured across the five real
    tables, 30-75% of coordinates sit just outside the border (block D's
    input_lut_4: 36.4% out of range, spanning [-1.025, +1.024]), piled up a
    fraction below -1. Those are rays that miss the image plane, and
    padding_mode="zeros" has to return zero for them. Sampling uniformly inside
    [-1, 1] would leave that path -- the dominant one for this model --
    completely untested.
    """
    rng = np.random.default_rng(seed)
    lut = rng.uniform(-1.0, 1.0, shape).astype(np.float32)
    below = rng.random(shape) < 0.35
    lut[below] = rng.uniform(-1.026, -1.0, shape).astype(np.float32)[below]
    above = rng.random(shape) < 0.001
    lut[above] = rng.uniform(1.0, 1.024, shape).astype(np.float32)[above]
    return lut


@pytest.mark.push
@pytest.mark.parametrize(
    "data_shape",
    [
        # One camera group of the deformed BEV transform — eight GridSample ops
        # over a 96x96 feature map. The block runs four such groups (32 ops).
        pytest.param((1, 64, 96, 96), id="block_B_one_camera"),
        # Cylinder BEV transform — the same eight-op structure over 80x144.
        pytest.param((1, 64, 80, 144), id="block_D"),
    ],
)
@pytest.mark.parametrize(
    "df_override",
    [
        # bf16 is what the model runs. Its coarse coordinate spacing snaps many
        # grid points onto exact half-integer ties, which is where nearest-mode
        # rounding has to agree with the framework.
        pytest.param(forge._C.DataFormat.Float16_b, id="bf16"),
        # f32 keeps full coordinate resolution — ties become measure-zero, so a
        # failure here points at the kernel rather than at input quantisation.
        pytest.param(forge._C.DataFormat.Float32, id="f32"),
    ],
)
@pytest.mark.parametrize(
    "enable_trace",
    [
        pytest.param(True, id="trace_enabled"),
        pytest.param(False, id="trace_disabled"),
    ],
)
def test_gridsample_lut_batched(data_shape, df_override, enable_trace):
    """
    The BEV transform's real GridSample structure: one feature map sampled eight
    times, each against a different depth slice of a (1, 128, 64, 8, 2) lookup
    table, with the results concatenated along the channel axis.

        Gather(axis=3, scalar k)  (1,128,64,8,2) -> (1,128,64,2)
        GridSample(data, grid_k)  (1,64,H,W)     -> (1,64,128,64)
        Concat(axis=1)                           -> (1,512,128,64)

    Gather takes a rank-0 index, so it drops the sliced axis and needs no
    reshape -- this mirrors the exported graph node for node.

    The compiler fuses the eight ops into a single ttnn.grid_sample carrying
    batch_output_channels=true and a (1, 128, 64, 16) grid, producing
    (1, 128, 64, 512) directly. A single-node GridSample test never reaches that
    path, so this covers the K-batched kernel the model actually dispatches.

    mode="nearest" with align_corners=True is the combination that used to
    require a host-precomputed grid; it is now resolved on device from the raw
    grid.
    """
    n, c, h, w = data_shape
    lut_shape = (1, _LUT_H, _LUT_W, _LUT_K, 2)

    data_vi = oh.make_tensor_value_info("data", otp.FLOAT, list(data_shape))
    lut_vi = oh.make_tensor_value_info("lut", otp.FLOAT, list(lut_shape))
    out_vi = oh.make_tensor_value_info("output", otp.FLOAT, [n, c * _LUT_K, _LUT_H, _LUT_W])

    initializers, nodes = [], []
    for k in range(_LUT_K):
        # Rank-0 (scalar) index, matching the exported graph: Gather then drops
        # axis 3 rather than keeping it as a length-1 dimension.
        initializers.append(numpy_helper.from_array(np.array(k, dtype=np.int64), f"k_{k}"))
        nodes.append(oh.make_node("Gather", ["lut", f"k_{k}"], [f"grid_{k}"], axis=3))
        nodes.append(
            oh.make_node(
                "GridSample",
                inputs=["data", f"grid_{k}"],
                outputs=[f"sampled_{k}"],
                align_corners=1,
                mode="nearest",
                padding_mode="zeros",
            )
        )
    nodes.append(oh.make_node("Concat", [f"sampled_{k}" for k in range(_LUT_K)], ["output"], axis=1))

    graph = oh.make_graph(
        nodes,
        "gridsample_lut_batched",
        [data_vi, lut_vi],
        [out_vi],
        initializer=initializers,
    )
    onnx_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    onnx.checker.check_model(onnx_model)

    rng = np.random.default_rng(0)
    data = torch.from_numpy(rng.standard_normal(data_shape).astype(np.float32))
    lut = torch.from_numpy(_bev_lut(lut_shape))

    # Snap the grid onto bf16's representable values before EITHER side sees it.
    #
    # df_override casts the device-side grid to bf16 while verify() keeps running
    # the framework model in f32, so without this the two sample from different
    # coordinates. In nearest mode a coordinate is an address, not a value: bf16's
    # spacing near 1.0 is 2^-8, which (x -> pixel) scales by (W-1)/2 into up to
    # 0.28 pixels of movement, and any coordinate landing within that of a .5
    # boundary rounds to the neighbouring pixel. The two sides then read unrelated
    # feature values -- e.g. coord -0.600969136 becomes -0.601562500, pixel
    # 28.5307 becomes 28.4883, and the golden reads pixel 29 while the device
    # reads 28. Both are answering their own question correctly; they were simply
    # handed different coordinates. It costs ~4.6-6.7% of in-range points and
    # drags PCC to 0.907 (block_D) / 0.930 (block_B).
    #
    # This is input quantisation, not a kernel disagreement -- no kernel can
    # recover f32 resolution from a bf16 grid. Widening the PCC gate instead would
    # be wrong: the round-half-to-even tie-breaking bug scored 0.943, i.e. BETTER
    # than this noise floor, so a relaxed gate could no longer tell a correct
    # kernel from one that rounds ties incorrectly.
    #
    # Scoped to the grid on purpose. bf16 in the feature map only perturbs sampled
    # values, which degrades gracefully; bf16 in the grid changes which pixel is
    # read. The f32 param is left untouched so it still covers the full-resolution
    # path, where ties are measure-zero.
    if df_override == forge._C.DataFormat.Float16_b:
        lut = lut.to(torch.bfloat16).to(torch.float32)

    inputs = [data, lut]

    mlir_cfg = MLIRConfig().set_optimization_level(2).set_enable_trace(enable_trace)
    compiler_cfg = forge.CompilerConfig(mlir_config=mlir_cfg)
    compiler_cfg.enable_optimization_passes = True
    if df_override == forge._C.DataFormat.Float16_b:
        compiler_cfg.default_df_override = df_override

    # Program cache stays on for every case — trace capture is built on it, and
    # it is how the model is deployed, so the no-trace cases should not run
    # against a different device configuration.
    from forge._C import runtime as forge_runtime

    ds = forge_runtime.experimental.DeviceSettings()
    ds.enable_program_cache = True
    forge_runtime.experimental.configure_devices(ds)

    framework_model = forge.OnnxModule("gridsample_lut_batched", onnx_model)
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name="gridsample_lut_batched",
        compiler_cfg=compiler_cfg,
    )

    # With trace on the first invocation captures and the second replays. Run
    # twice regardless so both variants execute the same number of times.
    for _ in range(2):
        verify(inputs, framework_model=framework_model, compiled_model=compiled_model)
