# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from test.models.onnx.vision.mnist.model_utils.model import MnistModel


def load_model():
    model = MnistModel()
    model.eval()
    return model


def load_input():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    dataloader = DataLoader(test_dataset, batch_size=1)
    test_input, _ = next(iter(dataloader))
    return [test_input]
