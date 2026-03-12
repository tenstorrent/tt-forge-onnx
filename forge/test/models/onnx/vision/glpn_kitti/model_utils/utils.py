# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from transformers import GLPNForDepthEstimation, GLPNImageProcessor


def load_model(variant):
    model = GLPNForDepthEstimation.from_pretrained(variant)
    model.eval()
    return model


def load_input(variant):
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image))
    processor = GLPNImageProcessor.from_pretrained(variant)
    inputs = processor(images=image, return_tensors="pt")
    return [inputs["pixel_values"]]
