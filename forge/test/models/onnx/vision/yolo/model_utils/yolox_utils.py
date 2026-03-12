# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
import torch
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import demo_postprocess, multiclass_nms


def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img = torch.from_numpy(padded_img)
    return padded_img


def print_detection_results(co_out, ratio, input_shape):
    """
    Post-processes raw model outputs and prints detected object information.

    This function converts model outputs into bounding boxes, applies non-maximum suppression (NMS),
    and prints the class name, confidence score, and bounding box coordinates for each detected object.
    """
    for i in range(len(co_out)):
        co_out[i] = co_out[i].detach().float().numpy()

    predictions = demo_postprocess(co_out[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        for box, score, cls_ind in zip(final_boxes, final_scores, final_cls_inds):
            class_name = COCO_CLASSES[int(cls_ind)]
            x_min, y_min, x_max, y_max = box
            print(f"Class: {class_name}, Confidence: {score}, Coordinates: ({x_min}, {y_min}, {x_max}, {y_max})")
