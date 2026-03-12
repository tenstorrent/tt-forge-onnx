# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from third_party.tt_forge_models.mobilenetv1.image_classification.onnx import ModelLoader, ModelVariant

variants = [
    pytest.param(ModelVariant.MOBILENET_V1_GITHUB, marks=pytest.mark.pr_models_regression),
    ModelVariant.MOBILENET_V1_075_192_HF,
    ModelVariant.MOBILENET_V1_100_224_HF,
    ModelVariant.MOBILENET_V1_100_TIMM,
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_mobilenetv1_onnx(forge_tmp_path, variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MOBILENETV1,
        variant=variant.value,
        source=Source.TORCHVISION,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load inputs
    loader = ModelLoader(variant=variant)
    inputs = loader.load_inputs()

    # Load framework model
    framework_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    framework_model = forge.OnnxModule(module_name, framework_model)

    # Set data format override
    data_format_override = forge._C.DataFormat.Float16_b
    compiler_cfg = forge.config.CompilerConfig(default_df_override=data_format_override)

    # Compile model
    compiled_model = forge.compile(framework_model, [inputs], module_name=module_name, compiler_cfg=compiler_cfg)

    # Model Verification and Inference
    _, co_out = verify(
        [inputs],
        framework_model,
        compiled_model,
    )

    # Print classification results
    loader.print_cls_results(co_out)
