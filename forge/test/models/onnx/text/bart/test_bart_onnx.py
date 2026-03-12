# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# BART Demo Script - SQuADv1.1 QA
import pytest
import torch
from transformers import BartForSequenceClassification, BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right

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


class BartWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        out = self.model(input_ids, attention_mask, decoder_input_ids)[0]
        return out


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        "facebook/bart-large-mnli",
    ],
)
def test_bart_classifier_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.BART,
        variant=variant,
        task=Task.NLP_SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    model = download_model(BartForSequenceClassification.from_pretrained, variant, torchscript=True)
    model.eval()
    tokenizer = download_model(BartTokenizer.from_pretrained, variant, pad_to_max_length=True)
    hypothesis = "Most of Mrinal Sen's work can be found in European collections."
    premise = "Calcutta seems to be the only other production center having any pretensions to artistic creativity at all, but ironically you're actually more likely to see the works of Satyajit Ray or Mrinal Sen shown in Europe or North America than in India itself."

    # generate inputs
    inputs_dict = tokenizer(
        premise,
        hypothesis,
        truncation="only_first",
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    decoder_input_ids = shift_tokens_right(
        inputs_dict["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id
    )
    inputs = [inputs_dict["input_ids"], inputs_dict["attention_mask"], decoder_input_ids]
    torch_model = BartWrapper(model)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1], inputs[2]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
