# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## Inception V4
import pytest

import forge
from third_party.tt_forge_models.inception.image_classification.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results

variants = [
    pytest.param(
        ModelVariant.INCEPTION_V4,
        marks=pytest.mark.pr_models_regression,
    ),
    pytest.param(
        ModelVariant.INCEPTION_V4_TF_IN1K,
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_inception_v4_timm_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.INCEPTION,
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

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    print_cls_results(fw_out[0], co_out[0])
