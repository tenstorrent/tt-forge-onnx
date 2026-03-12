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

from test.models.onnx.vision.vision_utils.utils import load_vision_model_and_input

variants_with_weights = {
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights",
}


@pytest.mark.pr_models_regression
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_ssdlite320_mobilenetv3(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SSDLITE320MOBILENETV3,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    pytest.xfail(reason="Fatal Python error: Segmentation fault")

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "detection", weight_name)
    inputs = [inputs[0]]

    # STEP 3: Export to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["bbox_regression", "cls_logits", "features"],
        dynamic_axes={"input": {0: "batch_size"}},
    )

    # STEP 4: Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # STEP 5: Compile with Forge
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # STEP 6: Verify
    verify(inputs, framework_model, compiled_model)
