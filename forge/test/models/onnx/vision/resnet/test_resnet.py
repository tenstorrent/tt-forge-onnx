# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

from third_party.tt_forge_models.resnet.image_classification.onnx import ModelLoader, ModelVariant

variants = [
    ModelVariant.RESNET_50,
    ModelVariant.RESNET_101,
    ModelVariant.RESNET_152,
]


@pytest.mark.pr_models_regression
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_resnet_onnx(variant, forge_tmp_path):

    # Record model details
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.RESNET,
        variant=variant.value,
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load inputs
    loader = ModelLoader(variant=ModelVariant(variant))
    inputs = loader.load_inputs().contiguous()

    # Load framework model
    framework_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    framework_model = forge.OnnxModule(module_name, framework_model)

    # Compile model
    compiled_model = forge.compile(framework_model, [inputs], module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        [inputs],
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )

    # Print classification results
    loader.print_cls_results(co_out)
