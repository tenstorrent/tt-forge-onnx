# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from third_party.tt_forge_models.whisper.audio_classification.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


variants = [
    pytest.param(
        ModelVariant.WHISPER_TINY,
        marks=[
            pytest.mark.pr_models_regression,
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-mlir/issues/6937"),
        ],
    ),
    pytest.param(
        ModelVariant.WHISPER_BASE,
        marks=[
            pytest.mark.pr_models_regression,
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-mlir/issues/6937"),
        ],
    ),
    pytest.param(
        ModelVariant.WHISPER_SMALL,
        marks=[
            pytest.mark.pr_models_regression,
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-mlir/issues/6937"),
        ],
    ),
    pytest.param(ModelVariant.WHISPER_MEDIUM, marks=pytest.mark.xfail),
    pytest.param(ModelVariant.WHISPER_LARGE, marks=pytest.mark.xfail),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_whisper_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.WHISPER,
        variant=variant.value,
        task=Task.AUDIO_SPEECH_RECOGNITION,
        source=Source.HUGGINGFACE,
    )

    if variant == ModelVariant.WHISPER_MEDIUM:
        pytest.xfail(reason="Skipping the test because it takes longer time to run")
    elif variant == ModelVariant.WHISPER_LARGE:
        pytest.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2767")

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = loader.load_inputs()
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
