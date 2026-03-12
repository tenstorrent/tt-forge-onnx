# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import requests
import torch
import onnx
from third_party.tt_forge_models.tools.utils import get_file

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.onnx.vision.ssd300_resnet50.model_utils.image_utils import (
    prepare_input,
)
from test.utils import download_model


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_pytorch_ssd300_resnet50(forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SSD300RESNET50,
        source=Source.TORCH_HUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # STEP 2 : prepare model
    framework_model = download_model(
        torch.hub.load, "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", pretrained=False
    )
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt"
    checkpoint_path = "nvidia_ssdpyt_fp16_190826.pt"

    response = requests.get(url)
    with open(checkpoint_path, "wb") as f:
        f.write(response.content)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    framework_model.load_state_dict(checkpoint["model"])
    framework_model.eval()

    # STEP 3 : prepare input
    input_image = get_file("http://images.cocodataset.org/val2017/000000397133.jpg")
    HWC = prepare_input(input_image)
    CHW = np.swapaxes(np.swapaxes(HWC, 0, 2), 1, 2)
    batch = np.expand_dims(CHW, axis=0)
    input_batch = torch.from_numpy(batch).float().contiguous()
    inputs = [input_batch]

    # STEP 4: Export to ONNX
    onnx_path = f"{forge_tmp_path}/ssd300_resnet50.onnx"
    torch.onnx.export(
        framework_model,
        input_batch,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # STEP 5: Load and wrap ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # STEP 6: Forge compile
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # STEP 7: Verify model
    verify(inputs, framework_model, compiled_model)
