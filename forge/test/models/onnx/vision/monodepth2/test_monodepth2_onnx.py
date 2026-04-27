# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from third_party.tt_forge_models.monodepth2.image_classification.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

variants = [
    pytest.param(ModelVariant.MONO_640X192),
    pytest.param(ModelVariant.STEREO_640X192),
    pytest.param(ModelVariant.MONO_STEREO_640X192, marks=pytest.mark.pr_models_regression),
    pytest.param(ModelVariant.MONO_NO_PT_640X192),
    pytest.param(ModelVariant.STEREO_NO_PT_640X192),
    pytest.param(ModelVariant.MONO_STEREO_NO_PT_640X192),
    pytest.param(ModelVariant.MONO_1024X320),
    pytest.param(ModelVariant.STEREO_1024X320),
    pytest.param(ModelVariant.MONO_STEREO_1024X320),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_monodepth2(variant, forge_tmp_path):
    pcc = 0.99
    if variant in (
        ModelVariant.MONO_640X192,
        ModelVariant.STEREO_640X192,
        ModelVariant.MONO_STEREO_640X192,
        ModelVariant.MONO_NO_PT_640X192,
        ModelVariant.STEREO_NO_PT_640X192,
    ):
        pcc = 0.95

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MONODEPTH2,
        variant=variant.value,
        source=Source.TORCHVISION,
        task=Task.CV_DEPTH_ESTIMATION,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = loader.load_inputs()
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    loader.torch_loader.postprocess_and_save_disparity_map(co_out, str(forge_tmp_path))
