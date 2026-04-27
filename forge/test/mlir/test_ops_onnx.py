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


@pytest.mark.push
@pytest.mark.parametrize(
    "data_shape, grid_shape, mode, padding_mode, align_corners",
    [
        pytest.param((1, 4, 8, 8), (1, 4, 4, 2), "bilinear", "zeros", 1),
        pytest.param((1, 4, 8, 8), (1, 4, 4, 2), "bilinear", "zeros", 0),
        pytest.param((1, 4, 8, 8), (1, 4, 4, 2), "nearest", "zeros", 1),
        pytest.param((1, 4, 8, 8), (1, 4, 4, 2), "nearest", "zeros", 0),
        pytest.param((1, 64, 96, 96), (1, 128, 64, 2), "bilinear", "zeros", 1),
        pytest.param((1, 64, 96, 96), (1, 128, 64, 2), "nearest", "zeros", 1),
        pytest.param((1, 64, 80, 144), (1, 128, 64, 2), "nearest", "zeros", 1),
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
    grid = torch.from_numpy(rng.uniform(-1.0, 1.0, grid_shape).astype(np.float32))

    inputs = [data, grid]

    framework_model = forge.OnnxModule("gridsample", onnx_model)
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name="gridsample")
    verify(inputs, framework_model=framework_model, compiled_model=compiled_model)
