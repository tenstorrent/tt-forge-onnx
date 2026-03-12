# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify

from test.models.onnx.text.mistral.model_utils.utils import get_current_weather
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.utils import download_model
import torch
import onnx

variants = ["mistralai/Mistral-7B-Instruct-v0.3"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mistral_v0_3_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MISTRAL,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Segmentation Fault")

    # Load tokenizer and model
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AutoModelForCausalLM.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()

    # prepare input
    conversation = [{"role": "user", "content": "What's the weather like in Paris?"}]
    input = tokenizer.apply_chat_template(
        conversation,
        tools=[get_current_weather],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = [input["input_ids"]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/mistral_7b_v0_3.onnx"
    torch.onnx.export(framework_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
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
