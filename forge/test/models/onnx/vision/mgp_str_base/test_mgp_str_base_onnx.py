# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# From: https://huggingface.co/alibaba-damo/mgp-str-base
import pytest

import forge
from third_party.tt_forge_models.mgp_str_base.image_classification.onnx import ModelLoader
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


@pytest.mark.xfail(reason="PCC drop: 0.545, below required 0.95")
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

    # Load model and input
    loader = ModelLoader()
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = loader.load_inputs()
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(
        inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))
    )

    output = (co_out[0], co_out[1], co_out[2])
    generated_text = loader.torch_loader.processor.batch_decode(output)["generated_text"]

    print(f"Generated text: {generated_text}")
