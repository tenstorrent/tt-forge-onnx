# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.onnx.vision.resnext.model_utils.utils import (
    get_image_tensor,
    post_processing,
)
from test.utils import download_model
import onnx

variants = [
    pytest.param("resnext14_32x4d", marks=pytest.mark.pr_models_regression),
    "resnext26_32x4d",
    "resnext50_32x4d",
    "resnext101_64x4d",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_resnext_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.RESNEXT,
        source=Source.OSMR,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load the model and prepare input data
    torch_model = download_model(ptcv_get_model, variant, pretrained=True)
    torch_model.eval()
    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    post_processing(co_out)
