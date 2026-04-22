# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import numpy as np
from PIL import Image


def load_image(image_path):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    mean, std = 128, 128
    img = Image.open(image_path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0  # Convert to float [0, 1]
    return img


def rescale(img, input_height, input_width):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    aspect = img.shape[1] / float(img.shape[0])
    # Convert to PIL Image for resizing
    img_pil = Image.fromarray((img * 255).astype(np.uint8))

    if aspect > 1:
        # landscape orientation - wide image
        res = int(aspect * input_height)
        # PIL resize takes (width, height), skimage takes (height, width)
        imgScaled = img_pil.resize((res, input_width), Image.Resampling.BILINEAR)
    elif aspect < 1:
        # portrait orientation - tall image
        res = int(input_width / aspect)
        imgScaled = img_pil.resize((input_height, res), Image.Resampling.BILINEAR)
    else:
        imgScaled = img_pil.resize((input_height, input_width), Image.Resampling.BILINEAR)

    # Convert back to float numpy array [0, 1]
    return np.array(imgScaled).astype(np.float32) / 255.0


def crop_center(img, cropx, cropy):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def normalize(img, mean=128, std=128):
    img = (img * 256 - mean) / std
    return img


def prepare_input(img_uri):
    img = load_image(img_uri)
    img = rescale(img, 300, 300)
    img = crop_center(img, 300, 300)
    img = normalize(img)
    return img
