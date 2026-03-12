# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file


def load_input(processor):
    input_image = get_file(
        "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg"
    )
    image = Image.open(str(input_image))
    placeholder = "<|image_1|>\n"
    messages = [{"role": "user", "content": placeholder + "Summarize the slide."}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt")
    inputs = [inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_sizes"]]
    return inputs
