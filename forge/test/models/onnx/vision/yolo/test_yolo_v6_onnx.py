# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import pytest
import requests
import torch
import onnx
from yolov6.layers.common import DetectBackend

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.model_utils.yolov6_utils import (
    check_img_size,
    process_image,
)

variants = [
    "yolov6n",
]


class YoloV6Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y, _ = self.model(x)
        # The model outputs float32, even if the input is bfloat16
        # Cast the output back to the input dtype
        return y.to(x.dtype)


@pytest.mark.pr_models_regression
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolo_v6_pytorch(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.YOLOV6,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.CV_OBJECT_DETECTION,
    )

    # STEP 2 :prepare model
    url = f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{variant}.pt"
    weights = f"{variant}.pt"

    try:
        response = requests.get(url)
        with open(weights, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {url} to {weights}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

    model = DetectBackend(weights)
    framework_model = model.model

    # prepare input
    stride = 32
    input_size = 640
    img_size = check_img_size(input_size, s=stride)
    img, img_src = process_image(img_size, stride, half=False)
    input_batch = img.unsqueeze(0)

    inputs = [input_batch]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/yolov6.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_module = forge.OnnxModule(module_name, onnx_model)

    # Issue: https://github.com/tenstorrent/tt-mlir/issues/6915
    # Once the issue is fixed, CPU Hoisted Const Eval should be enabled
    mlir_config = forge.config.MLIRConfig()
    mlir_config.set_custom_config("enable-cpu-hoisted-const-eval=false")

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=forge.CompilerConfig(mlir_config=mlir_config),
    )

    # Model Verification
    verify(
        inputs,
        onnx_module,
        compiled_model,
    )

    # STEP 5 : remove downloaded weights
    os.remove(weights)
