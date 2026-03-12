# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from test.utils import download_model
from third_party.tt_forge_models.tools.utils import get_file


def post_processing(output, top_k=5):

    probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
    class_file_path = get_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

    with open(class_file_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    topk_prob, topk_catid = torch.topk(probabilities, top_k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())


def generate_model_xception_imgcls_timm(variant):
    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)
    file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
    img = Image.open(file_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return framework_model, [img_tensor]
