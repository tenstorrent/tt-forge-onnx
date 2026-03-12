# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Reference: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
# Image URL from: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/mobilenet_ssd/tflite_mobilenet_v2_ssd_1x1.py

import torch
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import models, transforms


def load_model(variant, weights):
    weights = getattr(models, weights).DEFAULT
    model = getattr(models, variant)(weights=weights)
    model.eval()
    return model


def load_input():
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image))
    transform = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor()])
    img_tensor = [transform(image).unsqueeze(0)]
    batch_tensor = torch.cat(img_tensor, dim=0)
    return [batch_tensor]
