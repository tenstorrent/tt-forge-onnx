# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from test.utils import download_model
import onnx

# Wrapper to get around past key values
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, None, attention_mask)
        return output


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        "bigscience/bloom-1b1",
    ],
)
def test_bloom_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.BLOOM,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.NLP_CAUSAL_LM,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, padding_side="left")
    model = download_model(AutoModelForCausalLM.from_pretrained, variant, use_cache=False, return_dict=False)
    model.eval()
    torch_model = Wrapper(model)

    # Prepare input
    test_input = "This is a sample text from "
    input_tokens = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification and Inference
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
