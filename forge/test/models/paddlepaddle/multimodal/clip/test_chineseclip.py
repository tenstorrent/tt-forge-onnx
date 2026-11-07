# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from PIL import Image

import pytest

import paddle
from paddlenlp.transformers import (
    ChineseCLIPProcessor,
    ChineseCLIPTokenizer,
    ChineseCLIPModel,
    ChineseCLIPTextModel,
    ChineseCLIPVisionModel,
)

from forge.tvm_calls.forge_utils import paddle_trace
import forge
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
from third_party.tt_forge_models.tools.utils import get_file

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

variants = ["OFA-Sys/chinese-clip-vit-base-patch16"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_chineseclip_text(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.CHINESECLIPTEXT,
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.NLP_TEXT_ENCODING,
    )

    # Load Model and Tokenizer
    model = ChineseCLIPTextModel.from_pretrained(variant)
    model.eval()
    tokenizer = ChineseCLIPTokenizer.from_pretrained(variant)

    # Load sample
    inputs = tokenizer("一只猫的照片", padding=True, return_tensors="pd")
    inputs = [inputs["input_ids"]]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_chineseclip_vision(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.CHINESECLIPVISION,
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.CV_IMAGE_ENCODING,
    )

    # Load Model and Tokenizer
    model = ChineseCLIPVisionModel.from_pretrained(variant)
    model.eval()
    processor = ChineseCLIPProcessor.from_pretrained(variant)

    # Load sample
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image))

    inputs = processor(images=image, return_tensors="pd")
    inputs = [inputs["pixel_values"]]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.96)))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_chineseclip(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.CHINESECLIP,
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.MM_IMAGE_TEXT_PAIRING,
    )

    # Load Model and Tokenizer
    model = ChineseCLIPModel.from_pretrained(variant)
    model.eval()
    processor = ChineseCLIPProcessor.from_pretrained(variant)

    # Load sample
    # Prompts describe the COCO sample image (two cats on a couch), so the
    # similarities printed below stay meaningful: "猫" should score highest and the
    # rest act as distractors.
    text = ["猫", "狗", "玫瑰", "椅子"]
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image))
    inputs = processor(images=image, text=text, return_tensors="pd", padding=True)
    inputs = [inputs["input_ids"], inputs["pixel_values"]]

    # Test framework model
    outputs = model(*inputs)

    image_embed = outputs.image_embeds
    text_embeds = outputs.text_embeds

    image_embed = paddle.nn.functional.normalize(image_embed, axis=-1)
    text_embeds = paddle.nn.functional.normalize(text_embeds, axis=-1)

    similarities = paddle.matmul(text_embeds, image_embed.T)
    similarities = similarities.squeeze().numpy()

    for t, sim in zip(text, similarities):
        print(f"{t}: similarity = {sim:.4f}")

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.96)))
