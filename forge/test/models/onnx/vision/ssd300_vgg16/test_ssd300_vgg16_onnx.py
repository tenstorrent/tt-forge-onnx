# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.onnx.vision.ssd300_vgg16.model_utils.model_utils import SSDWrapper
from test.models.onnx.vision.vision_utils.utils import load_vision_model_and_input

variants_with_weights = {
    "ssd300_vgg16": "SSD300_VGG16_Weights",
}


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_ssd300_vgg16(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SSD300VGG16,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "detection", weight_name)
    model = SSDWrapper(framework_model)
    inputs = [inputs[0]]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}_ssd.onnx"
    torch.onnx.export(
        model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["bbox_regression", "cls_logits", "features"],
    )

    # Load ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile with Forge
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify
    verify(inputs, framework_model, compiled_model)
