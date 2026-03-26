# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import cv2

import forge
from forge.verify.verify import verify

from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from test.utils import fetch_paddle_model
from test.models.paddlepaddle.multimodal.paddleocr.utils import (
    get_boxes_from_pred,
    process_and_pad_images,
    fetch_img_and_charset,
    prep_image_for_recognition,
    prep_image_for_detection,
)

model_urls = {
    "v4": {
        "det": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
        "rec": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
    }
}

image_url = "https://raw.githubusercontent.com/ycdhqzhiai/PaddleOCR-demo/main/imgs/11.jpg"
dict_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt"

cache_dir = os.path.join("forge/test/models/paddlepaddle/multimodal/paddleocr", "cached_models")
os.makedirs(cache_dir, exist_ok=True)


@pytest.mark.skip_model_analysis
@pytest.mark.xfail
@pytest.mark.parametrize("det_url,rec_url", [(urls["det"], urls["rec"]) for _, urls in model_urls.items()])
@pytest.mark.parametrize("image_url", [image_url])
@pytest.mark.parametrize("cache_dir", [cache_dir])
@pytest.mark.parametrize("dict_url", [dict_url])
# Uncomment if you want to save results
# @pytest.mark.parametrize("results_path", ["forge/test/models/paddlepaddle/multimodal/paddleocr/results"])
def test_paddleocr(det_url, rec_url, dict_url, image_url, cache_dir, results_path=None):
    # Fetch model
    detection_model = fetch_paddle_model(det_url, cache_dir)
    recognition_model = fetch_paddle_model(rec_url, cache_dir)

    # Fetch image and dictionary
    image, charset = fetch_img_and_charset(image_url, dict_url)

    # Prepare inputs
    inputs, resized_image = prep_image_for_detection(image)

    # Compile detection model
    compiled_detection_model = forge.compile(detection_model, inputs)

    # Run and verify compiled detection model
    _, co_output = verify(
        inputs, detection_model, compiled_detection_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.8))
    )

    pred = co_output[0].numpy()
    box_cuts = get_boxes_from_pred(pred, resized_image, results_path=results_path)

    # Unify image sizes for recognition with compiled model
    padded_box_cuts = process_and_pad_images(box_cuts, img_size=resized_image.shape[:2])

    # Use first box image for input shape as they are all the same
    inputs = prep_image_for_recognition(padded_box_cuts[0])

    # Compile recognition model
    compiled_recognition_model = forge.compile(recognition_model, inputs)

    for i, box_image in enumerate(padded_box_cuts):
        if results_path:
            cv2.imwrite(f"{results_path}/box_{i}.jpg", box_image)

        # Prepare inputs for recognition
        box_image_input = prep_image_for_recognition(box_image)

        # Run and verify compiled recognition model
        _, co_output = verify(
            box_image_input,
            recognition_model,
            compiled_recognition_model,
            VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.8)),
        )

        output = co_output[0].numpy()

        # Decode output
        pred = output.argmax(axis=2)[0]
        pred_str = "".join([charset[i] for i in pred])
        print(f"Predicted text for box {i}: {pred_str}")
