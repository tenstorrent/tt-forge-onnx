# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import paddle
import pytest
import cv2

import forge
from forge.config import CompilerConfig
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.utils import fetch_paddle_model


model_urls = {
    "v4": {
        "ch": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
    }
}

cache_dir = os.path.join("forge/test/models/paddlepaddle/multimodal/paddleocr", "cached_models")
os.makedirs(cache_dir, exist_ok=True)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant,url",
    [(f"{variant}_det_{lang}", url) for variant, urls in model_urls.items() for lang, url in urls.items()],
)
def test_paddleocr_det(variant, url):
    # Record model details
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.PADDLEOCR,
        variant=variant,
        source=Source.PADDLE,
        task=Task.CV_OBJECT_DETECTION,
    )

    # Fetch model
    framework_model = fetch_paddle_model(url, cache_dir)

    # Generate a random image
    image = (np.random.rand(32, 100, 3) * 255).astype("uint8")
    resized_image = cv2.resize(image, (448, 448))
    image = resized_image.transpose(2, 0, 1).astype("float32")
    inputs = [paddle.to_tensor([image])]

    # Compile model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name=module_name,
    )

    # Verify data on sample input
    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.8)),
    )
