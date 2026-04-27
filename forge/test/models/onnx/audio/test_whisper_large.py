# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import onnx

import forge
from third_party.tt_forge_models.whisper.audio_classification.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import Framework, ModelArch, Source, Task, record_model_properties
from forge.verify.verify import verify


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
def test_whisper_large_v3_onnx(forge_tmp_path, variant=ModelVariant.WHISPER_LARGE_V3):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.WHISPER,
        variant=variant.value,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = loader.load_inputs()
    prep = loader.torch_loader._variant_config.pretrained_model_name
    onnx_path = f"{forge_tmp_path}/{prep}.onnx"

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification and inference
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
