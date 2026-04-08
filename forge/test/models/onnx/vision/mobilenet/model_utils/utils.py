# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import timm
import torch
from datasets import load_dataset
from loguru import logger


def load_inputs(model):
    """Load inputs from the ImageNet validation set, with random-tensor fallback."""
    try:
        dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=True)
        img = next(iter(dataset.skip(10)))["image"]
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        img_tensor = transform(img).unsqueeze(0)
        logger.info(f"img_tensor: {img_tensor.shape}")
    except Exception as e:
        logger.warning(f"Error loading dataset: {e}")
        img_tensor = torch.rand(1, 3, 224, 224)

    return [img_tensor]
