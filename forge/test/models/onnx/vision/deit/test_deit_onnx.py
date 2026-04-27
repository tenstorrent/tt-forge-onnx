# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from third_party.tt_forge_models.deit.image_classification.onnx import ModelLoader, ModelVariant
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

variants = [
    pytest.param(ModelVariant.BASE, marks=pytest.mark.pr_models_regression),
    pytest.param(ModelVariant.SMALL),
    pytest.param(ModelVariant.TINY),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_deit_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DEIT,
        variant=variant.value,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = loader.load_inputs()
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    pcc = 0.99
    if variant == ModelVariant.BASE:
        pcc = 0.96

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post processing
    loader.output_postprocess(co_out=co_out)
