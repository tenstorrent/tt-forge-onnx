# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from third_party.tt_forge_models.mlp_mixer.image_classification.onnx import ModelLoader, ModelVariant
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

varaints = [
    pytest.param(
        ModelVariant.MIXER_B16_224,
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        ModelVariant.MIXER_B16_224_IN21K,
        marks=[pytest.mark.xfail],
    ),
    pytest.param(ModelVariant.MIXER_B16_224_MIIL),
    pytest.param(
        ModelVariant.MIXER_B16_224_MIIL_IN21K,
        marks=[pytest.mark.xfail],
    ),
    pytest.param(ModelVariant.MIXER_B32_224),
    pytest.param(
        ModelVariant.MIXER_L16_224,
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        ModelVariant.MIXER_L16_224_IN21K,
        marks=[pytest.mark.xfail],
    ),
    pytest.param(ModelVariant.MIXER_L32_224),
    pytest.param(
        ModelVariant.MIXER_S16_224,
        marks=pytest.mark.pr_models_regression,
    ),
    pytest.param(ModelVariant.MIXER_S32_224),
    pytest.param(
        ModelVariant.MIXER_B16_224_GOOG_IN21K,
        marks=[pytest.mark.xfail],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", varaints)
def test_mlp_mixer_timm_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MLPMIXER,
        variant=variant.value,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = loader.load_inputs()
    framework_model = forge.OnnxModule(module_name, onnx_model)

    pcc = 0.99
    if variant == ModelVariant.MIXER_B16_224_MIIL:
        pcc = 0.95

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify model
    fw_out, co_out = verify(
        inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    loader.print_cls_results(co_out)
