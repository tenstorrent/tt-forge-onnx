# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import timm
import torch
import onnx
from typing import Callable
from PIL import Image
from loguru import logger
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import models, transforms

from test.utils import download_model

ONNX_OPSET_VERSION = 17
_IMAGENET_DOG_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
_IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def load_inputs(img, model):
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    # Apply the transformation
    img_tensor = transforms(img).unsqueeze(0)  # Add batch dimension

    return [img_tensor]


def load_timm_model_and_input(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    input_image = get_file(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
    img = Image.open(str(input_image))
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    input_batch = transforms(img).unsqueeze(0)
    return model, [input_batch]


def load_vision_model_and_input(variant, task, weight_name):
    if task == "detection":
        weights = getattr(models.detection, weight_name).DEFAULT
        model = getattr(models.detection, variant)(weights=weights)
    else:
        weights = getattr(models, weight_name).DEFAULT
        model = getattr(models, variant)(weights=weights)

    model.eval()

    # Preprocess image
    preprocess = weights.transforms()
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image)).convert("RGB")
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)

    # Make the tensor contiguous.
    # Current limitation of compiler/runtime is that it does not support non-contiguous tensors properly.
    batch_t = batch_t.contiguous()

    return model, [batch_t]


def load_torch_hub_model(model_name: str) -> torch.nn.Module:
    """Load a pretrained model from pytorch/vision torch hub in eval mode.

    Args:
        model_name: Model name as accepted by pytorch/vision:v0.10.0
            (e.g. ``"mobilenet_v3_small"``, ``"wide_resnet50_2"``).

    Returns:
        ``torch.nn.Module`` in eval mode.
    """
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", model_name, pretrained=True)
    model.eval()
    return model


def load_imagenet_inputs(image_size: int = 224) -> list:
    """Load a standard ImageNet-preprocessed input tensor.

    Downloads the canonical dog.jpg sample image and applies the standard
    ImageNet preprocessing pipeline (Resize 256 → CenterCrop → ToTensor →
    Normalize). Falls back to a random tensor if the download fails.

    Args:
        image_size: Final square crop size (default 224).

    Returns:
        List containing one preprocessed image tensor of shape
        (1, 3, image_size, image_size).
    """
    try:
        input_image = get_file(_IMAGENET_DOG_URL)
        img = Image.open(str(input_image))
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return [preprocess(img).unsqueeze(0)]
    except Exception as e:
        logger.warning(f"Failed to load image: {e}. Using random input tensor.")
        return [torch.rand(1, 3, image_size, image_size)]


def load_and_validate_onnx(onnx_path: str) -> onnx.ModelProto:
    """Load and validate an ONNX model from a file path."""
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def export_to_onnx(model: torch.nn.Module, inputs: list, output_path: str) -> None:
    """Export a PyTorch model to ONNX format at opset 17."""
    torch.onnx.export(model, inputs[0], output_path, opset_version=ONNX_OPSET_VERSION)


def load_onnx_with_fallback(
    torch_model_loader: Callable[[], torch.nn.Module],
    s3_onnx_path: str,
    onnx_filename: str,
    forge_tmp_path: str,
    inputs: list,
) -> onnx.ModelProto:
    """Load an ONNX model using a 2-tier fallback strategy.

    Strategy:
        1. Call ``torch_model_loader`` to obtain a PyTorch model, export it to
           ONNX, then load and validate the result.
        2. If that fails (e.g. torch.hub unavailable), fetch a pre-exported
           ONNX file directly from S3 via ``get_file``.

    Args:
        torch_model_loader: Zero-argument callable that returns a
            ``torch.nn.Module`` in eval mode.
        s3_onnx_path: S3-relative path for the pre-exported ONNX file,
            e.g. ``"test_files/onnx/mobilenet/mobilenet_v3_small.onnx"``.
        onnx_filename: Filename used when saving the exported ONNX locally,
            e.g. ``"mobilenet_v3_small.onnx"``.
        forge_tmp_path: Temporary directory for the exported ONNX file.
        inputs: Input tensors used for the ONNX export trace.

    Returns:
        Validated ``onnx.ModelProto`` ready for compilation.

    Raises:
        RuntimeError: If both tiers fail.
    """
    # Tier 1: torch model → ONNX export
    try:
        torch_model = torch_model_loader()
        onnx_path = f"{forge_tmp_path}/{onnx_filename}"
        export_to_onnx(torch_model, inputs, onnx_path)
        return load_and_validate_onnx(onnx_path)
    except Exception as e:
        logger.warning(f"Torch loading failed ({onnx_filename}): {e}. Falling back to S3.")

    # Tier 2: pre-exported ONNX from S3
    try:
        onnx_file = get_file(s3_onnx_path)
        return load_and_validate_onnx(str(onnx_file))
    except Exception as e:
        logger.warning(f"Failed to load ONNX from S3 ({s3_onnx_path}): {e}")

    raise RuntimeError(f"All loading strategies failed for {onnx_filename}.")


def post_processing(output, top_k: int = 5) -> None:
    """Print the top-k ImageNet class predictions from model output."""
    probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
    class_file_path = get_file(_IMAGENET_CLASSES_URL)
    with open(class_file_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    topk_prob, topk_catid = torch.topk(probabilities, top_k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())
