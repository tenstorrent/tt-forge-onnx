# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import numpy as np
import forge
import onnx
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.models.onnx.vision.unet.model_utils.utils import load_inputs
from test.utils import download_model


@pytest.mark.nightly
def test_unet_onnx(forge_tmp_path):

    # Build Module Name
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.UNET,
        variant="base",
        source=Source.TORCH_HUB,
        task=Task.CV_IMAGE_SEGMENTATION,
    )

    # Load the torch model
    # trust_repo=True bypasses GitHub API validation that triggers rate limit (403) in CI/shared envs
    torch_model = download_model(
        torch.hub.load,
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
        trust_repo=True,
    )
    torch_model.eval()

    # Load the inputs
    url, filename = (
        "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
        "TCGA_CS_4944.png",
    )
    inputs = load_inputs(url, filename)

    onnx_path = f"{forge_tmp_path}/unet.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
