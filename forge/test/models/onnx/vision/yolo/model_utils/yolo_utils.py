# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from torchvision import transforms
from typing import Optional
from ultralytics.nn.tasks import DetectionModel

from test.utils import download_model
from third_party.tt_forge_models.tools.utils import get_file

# S3 path for YOLO weights
YOLO_S3_WEIGHT_PATH = "test_files/pytorch/yolo/weights/"


class YoloWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.model[-1].end2end = False  # Disable internal post processing steps

    def forward(self, image: torch.Tensor):
        y, x = self.model(image)
        return (y, *x)


def _create_image_input() -> torch.Tensor:
    """Create preprocessed image tensor. Tries cats-image dataset, falls back to random tensor."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"]
    except Exception as e:
        print(f"Failed to load dataset image, using random tensor: {e}")
        return torch.rand(1, 3, 640, 640)

    preprocess = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ]
    )
    return preprocess(image).unsqueeze(0)


def _load_weights_from_s3(model_file: str) -> Optional[dict]:
    """Try loading YOLOv10 weights from S3 bucket."""
    try:
        s3_weight_file_path = os.path.join(YOLO_S3_WEIGHT_PATH, model_file)
        weight_file = get_file(s3_weight_file_path)
        return torch.load(str(weight_file), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load weights from S3 ({s3_weight_file_path}): {e}")
        return None


def _load_weights_from_url(url: str) -> dict:
    """Load weights from URL"""
    return download_model(
        torch.hub.load_state_dict_from_url,
        url,
        map_location="cpu",
        num_retries=3,
    )


def load_yolo_model_and_image(url: str):
    """Load YOLOv10 model and sample image with S3-first, URL fallback strategy.

    Strategy:
        1. Try loading weights from S3 bucket (avoids GitHub 502/rate limits)
        2. Fall back to URL with retry logic via download_model

    Args:
        url: GitHub releases URL for weights (e.g. ultralytics assets)

    Returns:
        Tuple of (model, image_tensor)
    """
    model_file = url.split("/")[-1]

    # Tier 1: Try loading from S3
    weights = _load_weights_from_s3(model_file)

    # Tier 2: Fall back to URL with retries
    if weights is None:
        print("Falling back to URL download with retries")
        weights = _load_weights_from_url(url)

    # Initialize and load model
    model = DetectionModel(cfg=weights["model"].yaml)
    model.load_state_dict(weights["model"].float().state_dict())
    model.eval()

    image_tensor = _create_image_input()
    return model, image_tensor
