# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import models


def load_image(feature_extractor):
    image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(image_path)
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    return [img_tensor]


variants_with_weights = {
    "swin_t": "Swin_T_Weights",
    "swin_s": "Swin_S_Weights",
    "swin_b": "Swin_B_Weights",
    "swin_v2_t": "Swin_V2_T_Weights",
    "swin_v2_s": "Swin_V2_S_Weights",
    "swin_v2_b": "Swin_V2_B_Weights",
}


def load_model(variant):
    weight_name = variants_with_weights[variant]
    weights = getattr(models, weight_name).DEFAULT
    model = getattr(models, variant)(weights=weights)
    model.eval()
    return model
