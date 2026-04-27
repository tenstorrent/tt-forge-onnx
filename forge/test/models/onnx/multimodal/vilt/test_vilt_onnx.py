# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from third_party.tt_forge_models.vilt.question_answering.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2969")
@pytest.mark.nightly
def test_vilt_question_answering_onnx(forge_tmp_path, variant=ModelVariant.VQA):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.VILT,
        variant=variant.value,
        task=Task.NLP_QA,
        source=Source.HUGGINGFACE,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = loader.load_inputs()
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    logits = co_out[0]
    idx = logits.argmax(-1).item()
    print(f"Predicted answer: {loader.torch_loader.model.config.id2label[idx]}")
