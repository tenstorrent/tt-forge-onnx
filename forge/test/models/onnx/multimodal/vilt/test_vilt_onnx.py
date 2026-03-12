# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)

import forge
import onnx
from forge.verify.verify import verify
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from transformers import (
    ViltConfig,
    ViltForQuestionAnswering,
    ViltProcessor,
)
from test.models.onnx.multimodal.vilt.model_utils.model import (
    ViLtEmbeddingWrapper,
    ViltModelWrapper,
)
from test.utils import download_model

text1 = "How many cats are there?"


def get_image():
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    return Image.open(str(input_image))


def generate_model_vilt_question_answering_hf_pytorch(variant):

    # Set model configurations
    config = ViltConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)

    # Load model and processor from HuggingFace
    processor = download_model(ViltProcessor.from_pretrained, variant)
    model = download_model(ViltForQuestionAnswering.from_pretrained, variant, config=config)
    model.eval()
    encoding = processor(get_image(), text1, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model, task=Task.NLP_QA.short)
    embedding_output, attention_mask = text_vision_embedding_model(**encoding)
    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], model


variants = ["dandelin/vilt-b32-finetuned-vqa"]


@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2969")
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_question_answering_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX, model=ModelArch.VILT, variant=variant, task=Task.NLP_QA, source=Source.HUGGINGFACE
    )
    torch_model, inputs, model = generate_model_vilt_question_answering_hf_pytorch(variant)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    logits = co_out[0]
    idx = logits.argmax(-1).item()
    print(f"Predicted answer: {model.config.id2label[idx]}")
