# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import paddle
import pytest

from third_party.tt_forge_models.resnet.image_classification.paddlepaddle import ModelLoader, ModelVariant

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

variants = [
    pytest.param(ModelVariant.RESNET18, marks=pytest.mark.pr_models_regression),
    ModelVariant.RESNET34,
    ModelVariant.RESNET50,
    ModelVariant.RESNET101,
    ModelVariant.RESNET152,
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_resnet_pd(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.RESNET,
        variant=variant.value,
        source=Source.PADDLE,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load inputs
    loader = ModelLoader(variant=ModelVariant(variant))
    input_sample = loader.load_inputs()

    # Load framework model
    framework_model = loader.load_model()

    # Compile model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input_sample,
        module_name=module_name,
    )

    # Model Verification and Inference
    _, co_out = verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )

    # Print classification results
    loader.print_results(co_out)
