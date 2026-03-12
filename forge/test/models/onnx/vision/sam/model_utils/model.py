# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from transformers import SamModel, SamProcessor


def get_model_inputs(variant):
    framework_model = SamModel.from_pretrained(variant).to("cpu")
    processor = SamProcessor.from_pretrained(variant)

    try:
        image_path = get_file("https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png")
        raw_image = Image.open(str(image_path)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to fetch image from URL. Using random fallback tensor. Reason: {e}")
        raw_image = Image.fromarray((torch.rand(3, 1024, 1024) * 255).byte().permute(1, 2, 0).numpy())

    input_points = [[[450, 600]]]
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to("cpu")
    sample_inputs = (
        inputs["pixel_values"],
        inputs["input_points"],
    )
    return framework_model, sample_inputs


class SamWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_points):
        return self.model(pixel_values=pixel_values, input_points=input_points).pred_masks.cpu()
