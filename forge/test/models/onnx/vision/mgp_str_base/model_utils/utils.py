# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from transformers import MgpstrForSceneTextRecognition, MgpstrProcessor


def load_model(variant):
    model = MgpstrForSceneTextRecognition.from_pretrained(variant, return_dict=False)
    model.eval()
    return model


def load_input(variant):
    input_image = get_file("https://i.postimg.cc/ZKwLg2Gw/367-14.png")
    image = Image.open(str(input_image)).convert("RGB")
    processor = MgpstrProcessor.from_pretrained(variant)
    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    return [inputs["pixel_values"]], processor
