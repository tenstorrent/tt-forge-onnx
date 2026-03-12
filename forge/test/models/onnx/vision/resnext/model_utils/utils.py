# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import transforms

from test.utils import download_model


def get_image_tensor():
    # Load data sample
    input_image = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
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
    return input_batch


def get_resnext_model_and_input(repo_or_dir, model_name):

    # Create model
    model = download_model(torch.hub.load, repo_or_dir, model_name, pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    return model, [input_batch]


def post_processing(output, top_k=5):

    probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
    class_file_path = get_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

    with open(class_file_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    topk_prob, topk_catid = torch.topk(probabilities, top_k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())
