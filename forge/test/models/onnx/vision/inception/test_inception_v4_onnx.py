# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## Inception V4
import pytest
import torch
import onnx

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.onnx.vision.inception.model_utils.model_utils import (
    preprocess_timm_model,
)
from test.utils import download_model
from test.models.models_utils import print_cls_results


def generate_model_inceptionV4_imgcls_timm_pytorch(variant):
    # Load model & Preprocess image
    framework_model, img_tensor = download_model(preprocess_timm_model, variant)
    return framework_model, [img_tensor]


variants = [
    pytest.param(
        "inception_v4",
        marks=pytest.mark.pr_models_regression,
    ),
    pytest.param(
        "inception_v4.tf_in1k",
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_inception_v4_timm_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.INCEPTION,
        variant=variant,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model + input
    framework_model, inputs = generate_model_inceptionV4_imgcls_timm_pytorch(variant)

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}_inception_v4.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # Load and verify ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
