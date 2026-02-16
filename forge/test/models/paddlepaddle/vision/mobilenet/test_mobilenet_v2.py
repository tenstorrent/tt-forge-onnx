# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import paddle
import pytest

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from paddle.vision.models import mobilenet_v2

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_mobilenetv2_basic():
    # Record model details
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.MOBILENETV2,
        variant="basic",
        source=Source.PADDLE,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load framework model
    framework_model = mobilenet_v2(pretrained=True)

    # Compile model
    input_sample = [paddle.rand([1, 3, 224, 224])]
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

    # Verify data on sample input
    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(verify_values=False),
    )
