# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import onnx
import torchvision.transforms as transforms
from datasets import load_dataset

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.onnx.vision.autoencoder.model_utils.linear_autoencoder import (
    LinearAE,
)


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_linear_ae_pytorch(forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.AUTOENCODER,
        variant="linear",
        task=Task.CV_IMAGE_ENCODING,
        source=Source.GITHUB,
    )

    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    framework_model = LinearAE()

    # Define transform to normalize data
    transform = transforms.Compose(
        [
            transforms.Resize((1, 784)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load sample from MNIST dataset
    dataset = load_dataset("ylecun/mnist", split="train")
    sample = dataset[0]["image"]
    sample_tensor = transform(sample).squeeze(0)

    inputs = [sample_tensor]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/linear_ae.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    output_image = co_out[0].view(1, 28, 28).detach().numpy()
    save_path = "forge/test/models/onnx/vision/autoencoder/results"
    os.makedirs(save_path, exist_ok=True)
    reconstructed_image_path = f"{save_path}/reconstructed_image.png"
    plt.imsave(reconstructed_image_path, np.squeeze(output_image), cmap="gray")
