# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# From: https://huggingface.co/alibaba-damo/mgp-str-base
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
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.onnx.vision.mgp_str_base.model_utils.utils import (
    load_input,
    load_model,
)


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)[0]


@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2969")
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        "alibaba-damo/mgp-str-base",
    ],
)
def test_mgp_scene_text_recognition_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MGP,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_ENCODING,
    )

    # Load model and wrap
    framework_model = load_model(variant)
    framework_model = Wrapper(framework_model).eval()

    # Load input
    inputs, processor = load_input(variant)
    inputs = [inputs[0]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/mgp_str_base.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(
        inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))
    )

    # Post-processing
    output = (co_out[0], co_out[1], co_out[2])
    generated_text = processor.batch_decode(output)["generated_text"]

    print(f"Generated text: {generated_text}")
