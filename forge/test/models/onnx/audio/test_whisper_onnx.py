# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx
from transformers import AutoProcessor, WhisperConfig, WhisperForConditionalGeneration

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


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        inputs = {"input_features": input_features, "decoder_input_ids": decoder_input_ids}
        output = self.model(**inputs)
        return output.logits


variants = [
    pytest.param(
        "openai/whisper-tiny",
        marks=[
            pytest.mark.pr_models_regression,
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-mlir/issues/6937"),
        ],
    ),
    pytest.param(
        "openai/whisper-base",
        marks=[
            pytest.mark.pr_models_regression,
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-mlir/issues/6937"),
        ],
    ),
    pytest.param(
        "openai/whisper-small",
        marks=[
            pytest.mark.pr_models_regression,
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-mlir/issues/6937"),
        ],
    ),
    pytest.param("openai/whisper-medium", marks=pytest.mark.xfail),
    pytest.param("openai/whisper-large", marks=pytest.mark.xfail),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_whisper_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.WHISPER,
        variant=variant,
        task=Task.AUDIO_SPEECH_RECOGNITION,
        source=Source.HUGGINGFACE,
    )

    if variant == "openai/whisper-medium":
        pytest.xfail(reason="Skipping the test because it takes longer time to run")
    elif variant == "openai/whisper-large":
        pytest.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2767")

    # Load model (with tokenizer and feature extractor)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    model_config = WhisperConfig.from_pretrained(variant)
    model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        config=model_config,
    )
    model.config.use_cache = False

    # Load and preprocess sample audio
    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt", weights_only=False)
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    input_features = inputs.input_features

    # Get decoder inputs
    decoder_start_token_tensor = torch.tensor(model.generation_config.decoder_start_token_id, dtype=torch.long)
    decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor

    inputs = [input_features, decoder_input_ids]

    torch_model = Wrapper(model)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    if variant in ["openai/whisper-medium", "openai/whisper-large"]:
        onnx.checker.check_model(onnx_path)
        framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)
        model = framework_model
    else:
        onnx.checker.check_model(onnx_model)
        framework_model = forge.OnnxModule(module_name, onnx_model)
        model = onnx_model

    # Compile model
    compiled_model = forge.compile(model, inputs, module_name=module_name)

    # Model Verification and Inference
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
