# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
from typing import Optional

from test.utils import download_model
from third_party.tt_forge_models.tools.utils import get_file

ONNX_OPSET_VERSION = 17


def _load_and_validate_onnx(onnx_path: str) -> onnx.ModelProto:
    """Load and validate an ONNX model from file path.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Validated ONNX model
    """
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def _export_to_onnx(
    model: torch.nn.Module,
    inputs: list,
    output_path: str,
) -> None:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        inputs: Input tensors for the model
        output_path: Path where ONNX model will be saved
    """
    torch.onnx.export(
        model,
        inputs[0],
        output_path,
        opset_version=ONNX_OPSET_VERSION,
    )


def _load_onnx_from_s3() -> Optional[onnx.ModelProto]:
    """Try to load ONNX model directly from S3 bucket."""
    onnx_path = "test_files/onnx/alexnet/alexnet.onnx"
    try:
        onnx_file = get_file(onnx_path)
        onnx_model = _load_and_validate_onnx(str(onnx_file))
        return onnx_model
    except Exception as e:
        print(f"Failed to load ONNX from S3 ({onnx_path}): {e}")
        return None


def _load_via_torch_hub(forge_tmp_path, inputs: list) -> onnx.ModelProto:
    """Load model via torch.hub (fallback method).

    Args:
        forge_tmp_path: Temporary directory path for saving ONNX model
        inputs: Input tensors for the model

    Returns:
        onnx_model
    """
    model = download_model(
        torch.hub.load,
        "pytorch/vision:v0.10.0",
        "alexnet",
        pretrained=True,
        trust_repo=True,
    )
    model.eval()

    onnx_path = f"{forge_tmp_path}/alexnet.onnx"
    _export_to_onnx(model, inputs, onnx_path)

    onnx_model = _load_and_validate_onnx(onnx_path)
    return onnx_model


def load_model(forge_tmp_path: str, inputs: list) -> onnx.ModelProto:
    """Load AlexNet ONNX model using 2-tier fallback strategy.

    Strategy:
        1. Try loading ONNX directly from S3 bucket
        2. Fall back to torch.hub.load and export to ONNX

    Args:
        forge_tmp_path: Temporary directory path for saving ONNX model
        inputs: Input tensors for the model

    Returns:
        onnx_model
    """
    # Tier 1: Try loading ONNX directly from S3
    result = _load_onnx_from_s3()
    if result is not None:
        return result

    # Tier 2: Fall back to torch.hub.load
    print("Falling back to torch.hub.load method")
    return _load_via_torch_hub(forge_tmp_path, inputs)
