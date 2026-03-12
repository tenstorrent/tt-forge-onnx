# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx
from transformers import AutoModelForCausalLM, AutoProcessor

import forge
from forge.forge_property_utils import Framework, ModelArch, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.onnx.text.phi3.model_utils.utils import load_input
from test.utils import download_model


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values, image_sizes):
        return self.model(input_ids, attention_mask, None, None, None, pixel_values, image_sizes)


variants = ["microsoft/Phi-3.5-vision-instruct"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_phi3_5_vision(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.PHI35VISION,
        variant=variant,
        task=Task.MM_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Segmentation Fault")

    # Load model and processor
    model = download_model(
        AutoModelForCausalLM.from_pretrained,
        variant,
        return_dict=False,
        trust_remote_code=True,
        use_cache=False,
        _attn_implementation="eager",
    )
    model.eval()
    framework_model = Wrapper(model)
    processor = download_model(AutoProcessor.from_pretrained, variant, trust_remote_code=True, num_crops=4)

    # prepare input
    inputs = load_input(processor)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/model.onnx"
    input_ids, attention_mask, pixel_values, image_sizes = inputs
    torch.onnx.export(
        framework_model, (input_ids, attention_mask, pixel_values, image_sizes), onnx_path, opset_version=17
    )

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
