# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import requests
import torch

from third_party.tt_forge_models.tools.utils import get_file
from test.utils import download_model
from test.models.onnx.vision.ssd300_resnet50.model_utils.image_utils import prepare_input

_CHECKPOINT_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp"
    "/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt"
)
_CHECKPOINT_PATH = "nvidia_ssdpyt_fp16_190826.pt"
_COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/000000397133.jpg"


def load_ssd300_torch_model() -> torch.nn.Module:
    """Load the NVIDIA SSD300 ResNet50 model via torch hub and apply pretrained weights."""
    model = download_model(
        torch.hub.load,
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_ssd",
        pretrained=False,
    )
    response = requests.get(_CHECKPOINT_URL)
    with open(_CHECKPOINT_PATH, "wb") as f:
        f.write(response.content)
    checkpoint = torch.load(_CHECKPOINT_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_ssd300_inputs() -> list:
    """Load and preprocess a COCO sample image for SSD300 ResNet50."""
    input_image = get_file(_COCO_IMAGE_URL)
    HWC = prepare_input(input_image)
    CHW = np.swapaxes(np.swapaxes(HWC, 0, 2), 1, 2)
    batch = np.expand_dims(CHW, axis=0)
    return [torch.from_numpy(batch).float().contiguous()]
