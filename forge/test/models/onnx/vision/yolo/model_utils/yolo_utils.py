# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from datasets import load_dataset
from torchvision import transforms
from ultralytics.nn.tasks import DetectionModel


class YoloWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.model[-1].end2end = False  # Disable internal post processing steps

    def forward(self, image: torch.Tensor):
        y, x = self.model(image)
        # Post processing inside model casts output to float32, even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return (y.to(image.dtype), *x)


def load_yolo_model_and_image(url):
    # Load YOLO model weights
    weights = torch.hub.load_state_dict_from_url(url, map_location="cpu")

    # Initialize and load model
    model = DetectionModel(cfg=weights["model"].yaml)
    model.load_state_dict(weights["model"].float().state_dict())
    model.eval()

    # Load sample image and preprocess
    dataset = load_dataset("huggingface/cats-image", split="test[:1]")
    image = dataset[0]["image"]
    preprocess = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0)

    return model, image_tensor
