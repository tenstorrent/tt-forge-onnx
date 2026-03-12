# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import zipfile

import pytest
import requests
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
from test.models.onnx.vision.retinanet.model_utils.model_utils import img_preprocess
from test.models.onnx.vision.retinanet.model_utils.model import Model

variants = [
    "retinanet_rn18fpn",
    "retinanet_rn34fpn",
    "retinanet_rn50fpn",
    "retinanet_rn101fpn",
    "retinanet_rn152fpn",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_retinanet(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.RETINANET,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CV_OBJECT_DETECTION,
    )

    # Prepare model
    url = f"https://github.com/NVIDIA/retinanet-examples/releases/download/19.04/{variant}.zip"
    local_zip_path = f"{variant}.zip"

    response = requests.get(url)
    with open(local_zip_path, "wb") as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    # Find the path of the .pth file
    extracted_path = f"{variant}"
    checkpoint_path = ""
    for root, _, files in os.walk(extracted_path):
        for file in files:
            if file.endswith(".pth"):
                checkpoint_path = os.path.join(root, file)

    framework_model = Model.load(checkpoint_path)
    framework_model.eval()

    # Prepare input
    input_batch = img_preprocess()
    inputs = [input_batch]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}_retinanet.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify model correctness
    verify(inputs, framework_model, compiled_model)

    # Cleanup
    shutil.rmtree(extracted_path)
    os.remove(local_zip_path)
