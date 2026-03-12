# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import timm
import torch
from loguru import logger
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms


def get_image():
    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        input_image = Image.open(file_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(input_image)
        img_tensor = img_tensor.unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    return img_tensor


def preprocess_steps(model_type):
    model = model_type(True, True).eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    return model, img_tensor


def preprocess_timm_model(model_name):
    use_pretrained_weights = True
    if model_name == "ese_vovnet99b":
        use_pretrained_weights = False
    model = timm.create_model(model_name, pretrained=use_pretrained_weights)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    return model, img_tensor
