# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from test.models.models_utils import print_cls_results, preprocess_inputs
from test.models.onnx.vision.alexnet.model_utils.alexnet_utils import load_model


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_alexnet_onnx(forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.ALEXNET,
        source=Source.TORCH_HUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load input
    inputs = preprocess_inputs()

    # Load model
    onnx_model = load_model(forge_tmp_path, inputs)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    print_cls_results(fw_out[0], co_out[0])
