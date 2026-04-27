# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from third_party.tt_forge_models.ghostnet.image_classification.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

variants = [
    pytest.param(ModelVariant.GHOSTNET_100, marks=pytest.mark.pr_models_regression),
    ModelVariant.GHOSTNET_100_IN1K,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_ghostnet_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.GHOSTNET,
        variant=variant.value,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLASSIFICATION,
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

    loader.print_cls_results(co_out)
