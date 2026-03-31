# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import timm
import torch
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import transforms

from test.models.onnx.vision.mobilenet.model_utils.mobilenet_v1 import MobileNetV1
from test.utils import download_model
from datasets import load_dataset
from loguru import logger


def load_inputs(model):

    try:
        dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=True)
        img = next(iter(dataset.skip(10)))["image"]
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        img_tensor = transforms(img).unsqueeze(0)
        logger.info(f"img_tensor: {img_tensor.shape}")
    except Exception as e:
        logger.warning(f"Error loading dataset: {e}")
        img_tensor = torch.rand(1, 3, 224, 224)

    return [img_tensor]


def load_mobilenet_model(model_name):

    # Create model
    if model_name == "mobilenet_v1":
        model = MobileNetV1(9)
    else:
        model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", model_name, pretrained=True)

    model.eval()

    # Load data sample
    input_image = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")

    # Preprocessing
    input_image = Image.open(str(input_image))
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    return model, [input_batch]


def post_processing(output, top_k=5):

    probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
    class_file_path = get_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

    with open(class_file_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    topk_prob, topk_catid = torch.topk(probabilities, top_k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())
