# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        x, y, z = self.model(image)
        # Post processing inside model casts output to float32,
        # even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return x.to(image.dtype), y.to(image.dtype), z.to(image.dtype)
