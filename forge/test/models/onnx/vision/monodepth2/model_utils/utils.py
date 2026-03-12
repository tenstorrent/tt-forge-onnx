# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import hashlib
import os
import zipfile

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as pil
import torch
from PIL import Image
from six.moves import urllib
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import transforms

from test.models.onnx.vision.monodepth2.model_utils.depth_decoder import DepthDecoder
from test.models.onnx.vision.monodepth2.model_utils.resnet_encoder import (
    ResnetEncoder,
)


def download_model(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it"""
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
            "a964b8356e08a02d009609d9e3928f7c",
        ),
        "stereo_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
            "3dfb76bcff0786e4ec07ac00f658dd07",
        ),
        "mono+stereo_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
            "c024d69012485ed05d7eaa9617a96b81",
        ),
        "mono_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
            "9c2f071e35027c895a4728358ffc913a",
        ),
        "stereo_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
            "41ec2de112905f85541ac33a854742d1",
        ),
        "mono+stereo_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
            "46c3b824f541d143a45c37df65fbab0a",
        ),
        "mono_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
            "0ab0766efdfeea89a0d9ea8ba90e1e63",
        ),
        "stereo_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
            "afc2f2126d70cf3fdf26b550898b501a",
        ),
        "mono+stereo_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
            "cdc5fc9b23513c07d5b19235d9ef08f7",
        ),
    }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, "rb") as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", "r") as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


class MonoDepth2(torch.nn.Module):
    def __init__(self, encoder, depth_decoder):
        super().__init__()
        self.encoder = encoder
        self.depth_decoder = depth_decoder

    def forward(self, input):
        features = self.encoder(input)
        outputs = self.depth_decoder(features)
        return outputs[("disp", 0)]


def load_model(variant):
    encoder_path = os.path.join("models", variant, "encoder.pth")
    depth_decoder_path = os.path.join("models", variant, "depth.pth")

    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location="cpu")
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
    depth_decoder.load_state_dict(loaded_dict)

    model = MonoDepth2(encoder, depth_decoder)
    model.eval()

    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]

    return model, feed_height, feed_width


def load_input(feed_height, feed_width):

    input_image = get_file("https://raw.githubusercontent.com/nianticlabs/monodepth2/master/assets/test_image.jpg")
    input_image = Image.open(input_image).convert("RGB")
    original_width, original_height = input_image.size
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    return input_tensor, original_width, original_height


def postprocess_and_save_disparity_map(co_out, original_height, original_width, variant):
    disp = co_out[0].to(torch.float32)
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False
    )

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    save_path = "forge/test/models/pytorch/vision/monodepth2/results"
    os.makedirs(save_path, exist_ok=True)
    name_dest_im = f"{save_path}/{variant}_pred_disp_vis.png"
    im.save(name_dest_im)
