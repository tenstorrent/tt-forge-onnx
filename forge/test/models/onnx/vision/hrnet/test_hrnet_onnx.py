# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from third_party.tt_forge_models.hrnet.image_classification.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results

variants = [
    pytest.param(ModelVariant.HRNET_W18_SMALL_V1_OSMR, marks=pytest.mark.pr_models_regression),
    ModelVariant.HRNET_W18_SMALL_V2_OSMR,
    ModelVariant.HRNETV2_W18_OSMR,
    ModelVariant.HRNETV2_W30_OSMR,
    ModelVariant.HRNETV2_W44_OSMR,
    ModelVariant.HRNETV2_W48_OSMR,
    ModelVariant.HRNETV2_W64_OSMR,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_hrnet_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.HRNET,
        variant=variant.value,
        source=Source.OSMR,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = loader.load_inputs()
    framework_model = forge.OnnxModule(module_name, onnx_model)

    pcc = 0.99
    if variant in (ModelVariant.HRNETV2_W64_OSMR, ModelVariant.HRNETV2_W44_OSMR):
        pcc = 0.95

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    print_cls_results(fw_out[0], co_out[0])
