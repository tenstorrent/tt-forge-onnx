# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig, AutomaticValueChecker
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from third_party.tt_forge_models.efficientnet.image_classification.onnx import ModelLoader, ModelVariant

params = [
    pytest.param(ModelVariant.B0, marks=[pytest.mark.pr_models_regression]),
    pytest.param(ModelVariant.B1),
    pytest.param(ModelVariant.B2),
    pytest.param(ModelVariant.B3),
    pytest.param(ModelVariant.B4),
    pytest.param(ModelVariant.B5),
    pytest.param(ModelVariant.B6),
    pytest.param(ModelVariant.B7),
    pytest.param(ModelVariant.TIMM_EFFICIENTNET_B0),
    pytest.param(ModelVariant.TIMM_EFFICIENTNET_B4),
    pytest.param(ModelVariant.HF_TIMM_EFFICIENTNET_B0_RA_IN1K),
    pytest.param(ModelVariant.HF_TIMM_EFFICIENTNET_B4_RA2_IN1K),
    pytest.param(ModelVariant.HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K),
    pytest.param(ModelVariant.HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K),
    pytest.param(ModelVariant.HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K),
]


@pytest.mark.parametrize("variant", params)
@pytest.mark.nightly
def test_efficientnet_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.EFFICIENTNET,
        variant=variant.value,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )
    if variant == ModelVariant.B5:
        pytest.xfail(reason="Requires multi-chip support")

    # Load inputs
    loader = ModelLoader(variant=variant)
    inputs = loader.load_inputs().contiguous()

    # Load framework model
    framework_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    framework_model = forge.OnnxModule(module_name, framework_model)

    # Compile model
    compiled_model = forge.compile(framework_model, [inputs], module_name=module_name)

    pcc = 0.99

    if variant == ModelVariant.B1:
        pcc = 0.95

    # Model Verification and Inference
    _, co_out = verify(
        [inputs],
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(
            value_checker=AutomaticValueChecker(pcc=pcc),
        ),
    )

    # Print classification results
    loader.print_cls_results(co_out)
