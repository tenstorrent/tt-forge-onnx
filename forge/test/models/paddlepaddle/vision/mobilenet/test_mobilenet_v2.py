# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import paddle
import pytest

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from third_party.tt_forge_models.mobilenetv2.image_classification.paddlepaddle import ModelLoader

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_mobilenetv2_basic():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.MOBILENETV2,
        source=Source.PADDLE,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load inputs
    loader = ModelLoader()
    input_sample = loader.load_inputs()

    # Load framework model
    framework_model = loader.load_model()

    # Compile model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input_sample,
        module_name=module_name,
    )

    # TODO: Currently, there is pcc drop when we execute
    # the test cases along with other tests in the nightly
    # but when we run the test cases alone it passes.
    # Need to investigate why this is happening.
    # Until then, we are disabling the value comparison.
    # Issue: https://github.com/tenstorrent/tt-forge-onnx/issues/3161

    # Model Verification and Inference
    _, co_out = verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(verify_values=False),
    )

    # Print classification results
    loader.print_results(co_out)
