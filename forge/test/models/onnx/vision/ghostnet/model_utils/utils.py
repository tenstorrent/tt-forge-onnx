# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import timm
import torch
from datasets import load_dataset
from third_party.tt_forge_models.tools.utils import get_file
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from test.utils import download_model


def load_ghostnet_model(variant):

    # Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # Prepare input
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    img = next(iter(dataset.skip(10)))["image"]
    data_config = resolve_data_config({}, model=framework_model)
    transforms = create_transform(**data_config, is_training=False)
    img_tensor = transforms(img).unsqueeze(0)
    return framework_model, [img_tensor]


def post_processing(output, top_k=5):

    probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
    class_file_path = get_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

    with open(class_file_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    topk_prob, topk_catid = torch.topk(probabilities, top_k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())
