# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from third_party.tt_forge_models.oft_stable_diffusion.text_to_image.onnx import ModelLoader
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from forge.verify.verify import verify


@pytest.mark.parametrize("variant", ["runwayml/stable-diffusion-v1-5"])
@pytest.mark.nightly
def test_oft(forge_tmp_path, variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.OFT,
        variant=variant.split("/")[-1],
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Segmentation Fault")

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = list(loader.load_inputs())
    framework_model = forge.OnnxModule(module_name, onnx_model)

    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
