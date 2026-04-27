# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from third_party.tt_forge_models.dla.image_classification.onnx import ModelLoader, ModelVariant

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


variants = [
    ModelVariant.DLA34,
    ModelVariant.DLA46_C,
    ModelVariant.DLA46X_C,
    ModelVariant.DLA60,
    pytest.param(ModelVariant.DLA60X, marks=pytest.mark.pr_models_regression),
    ModelVariant.DLA60X_C,
    ModelVariant.DLA102,
    ModelVariant.DLA102X,
    ModelVariant.DLA102X2,
    ModelVariant.DLA169,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_dla_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DLA,
        variant=variant.value,
        task=Task.CV_IMAGE_ENCODING,
        source=Source.TIMM,
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
