# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import requests
import torch
from PIL import Image
from torchvision import transforms


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_batch):
        outputs = self.model(input_batch)
        outputs = [outputs[0]["boxes"], outputs[0]["labels"], outputs[0]["scores"]]
        return outputs


def img_preprocess():

    url = "https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg"
    pil_img = Image.open(requests.get(url, stream=True).raw)
    new_size = (640, 480)
    pil_img = pil_img.resize(new_size, resample=Image.BICUBIC)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = preprocess(pil_img)
    img = img.unsqueeze(0)
    return img
