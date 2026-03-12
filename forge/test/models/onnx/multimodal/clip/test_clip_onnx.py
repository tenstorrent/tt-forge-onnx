# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.onnx.multimodal.clip.model_utils.clip_model import CLIPTextWrapper
import onnx
import torch
from third_party.tt_forge_models.clip.pytorch import ModelLoader


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "openai/clip-vit-base-patch32",
        ),
    ],
)
def test_clip_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.CLIP,
        variant=variant,
        suffix="text",
        source=Source.HUGGINGFACE,
        task=Task.NLP_CAUSAL_LM,
    )

    # Load Model and input
    loader = ModelLoader()
    model = loader.load_model()
    torch_model = CLIPTextWrapper(model)
    inputs = loader.load_inputs()
    inputs = [inputs["input_ids"], inputs["attention_mask"]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
