# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        # 1D
        pytest.param(
            (64,), 0, True, marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs")
        ),
        pytest.param(
            (64,),
            -1,
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (64,),
            0,
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (64,),
            -1,
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 2D - single dim
        ((32, 64), 0, True),
        ((32, 64), 0, False),
        ((32, 64), 1, True),
        ((32, 64), 1, False),
        ((32, 64), -1, True),
        ((32, 64), -1, False),
        ((32, 64), -2, True),
        ((32, 64), -2, False),
        # 2D - multi dim
        pytest.param(
            (32, 64),
            [0, 1],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (32, 64),
            [0, 1],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 3D - single dim
        ((4, 32, 64), 0, True),
        ((4, 32, 64), 0, False),
        ((4, 32, 64), 1, True),
        ((4, 32, 64), 1, False),
        ((4, 32, 64), 2, True),
        ((4, 32, 64), 2, False),
        ((4, 32, 64), -1, True),
        ((4, 32, 64), -1, False),
        ((4, 32, 64), -2, True),
        ((4, 32, 64), -2, False),
        ((4, 32, 64), -3, True),
        ((4, 32, 64), -3, False),
        # 3D - multi dim
        ((4, 32, 64), [0, 1], True),
        ((4, 32, 64), [0, 1], False),
        ((4, 32, 64), [0, 2], True),
        ((4, 32, 64), [0, 2], False),
        ((4, 32, 64), [1, 2], True),
        ((4, 32, 64), [1, 2], False),
        pytest.param(
            (4, 32, 64),
            [0, 1, 2],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (4, 32, 64),
            [0, 1, 2],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 4D - single dim
        ((1, 4, 32, 64), 0, True),
        ((1, 4, 32, 64), 0, False),
        ((1, 4, 32, 64), 1, True),
        ((1, 4, 32, 64), 1, False),
        ((1, 4, 32, 64), 2, True),
        ((1, 4, 32, 64), 2, False),
        ((1, 4, 32, 64), 3, True),
        ((1, 4, 32, 64), 3, False),
        ((1, 4, 32, 64), -1, True),
        ((1, 4, 32, 64), -1, False),
        ((1, 4, 32, 64), -2, True),
        ((1, 4, 32, 64), -2, False),
        ((1, 4, 32, 64), -3, True),
        ((1, 4, 32, 64), -3, False),
        ((1, 4, 32, 64), -4, True),
        ((1, 4, 32, 64), -4, False),
        # 4D - 2-dim combinations
        ((1, 4, 32, 64), [0, 1], True),
        ((1, 4, 32, 64), [0, 1], False),
        ((1, 4, 32, 64), [0, 2], True),
        ((1, 4, 32, 64), [0, 2], False),
        ((1, 4, 32, 64), [0, 3], True),
        ((1, 4, 32, 64), [0, 3], False),
        ((1, 4, 32, 64), [1, 2], True),
        ((1, 4, 32, 64), [1, 2], False),
        ((1, 4, 32, 64), [1, 3], True),
        ((1, 4, 32, 64), [1, 3], False),
        ((1, 4, 32, 64), [2, 3], True),
        ((1, 4, 32, 64), [2, 3], False),
        # 4D - 3-dim combinations
        ((1, 4, 32, 64), [0, 1, 2], True),
        ((1, 4, 32, 64), [0, 1, 2], False),
        ((1, 4, 32, 64), [0, 1, 3], True),
        ((1, 4, 32, 64), [0, 1, 3], False),
        ((1, 4, 32, 64), [0, 2, 3], True),
        ((1, 4, 32, 64), [0, 2, 3], False),
        pytest.param(
            (1, 4, 32, 64),
            [1, 2, 3],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (1, 4, 32, 64),
            [1, 2, 3],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 4D - all dims
        pytest.param(
            (1, 4, 32, 64),
            [0, 1, 2, 3],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (1, 4, 32, 64),
            [0, 1, 2, 3],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
    ],
)
@pytest.mark.push
def test_reduce_sum(input_shape, dim, keepdim):
    class ReduceSum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.sum(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape)]
    framework_model = ReduceSum()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        # 1D
        pytest.param(
            (64,), 0, True, marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs")
        ),
        pytest.param(
            (64,),
            -1,
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (64,),
            0,
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (64,),
            -1,
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 2D - single dim
        ((32, 64), 0, True),
        ((32, 64), 0, False),
        ((32, 64), 1, True),
        ((32, 64), 1, False),
        ((32, 64), -1, True),
        ((32, 64), -1, False),
        ((32, 64), -2, True),
        ((32, 64), -2, False),
        # 2D - multi dim
        pytest.param(
            (32, 64),
            [0, 1],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (32, 64),
            [0, 1],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 3D - single dim
        ((4, 32, 64), 0, True),
        ((4, 32, 64), 0, False),
        ((4, 32, 64), 1, True),
        ((4, 32, 64), 1, False),
        ((4, 32, 64), 2, True),
        ((4, 32, 64), 2, False),
        ((4, 32, 64), -1, True),
        ((4, 32, 64), -1, False),
        ((4, 32, 64), -2, True),
        ((4, 32, 64), -2, False),
        ((4, 32, 64), -3, True),
        ((4, 32, 64), -3, False),
        # 3D - multi dim
        ((4, 32, 64), [0, 1], True),
        ((4, 32, 64), [0, 1], False),
        ((4, 32, 64), [0, 2], True),
        ((4, 32, 64), [0, 2], False),
        ((4, 32, 64), [1, 2], True),
        ((4, 32, 64), [1, 2], False),
        pytest.param(
            (4, 32, 64),
            [0, 1, 2],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (4, 32, 64),
            [0, 1, 2],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 4D - single dim
        ((1, 4, 32, 64), 0, True),
        ((1, 4, 32, 64), 0, False),
        ((1, 4, 32, 64), 1, True),
        ((1, 4, 32, 64), 1, False),
        ((1, 4, 32, 64), 2, True),
        ((1, 4, 32, 64), 2, False),
        ((1, 4, 32, 64), 3, True),
        ((1, 4, 32, 64), 3, False),
        ((1, 4, 32, 64), -1, True),
        ((1, 4, 32, 64), -1, False),
        ((1, 4, 32, 64), -2, True),
        ((1, 4, 32, 64), -2, False),
        ((1, 4, 32, 64), -3, True),
        ((1, 4, 32, 64), -3, False),
        ((1, 4, 32, 64), -4, True),
        ((1, 4, 32, 64), -4, False),
        # 4D - 2-dim combinations
        ((1, 4, 32, 64), [0, 1], True),
        ((1, 4, 32, 64), [0, 1], False),
        ((1, 4, 32, 64), [0, 2], True),
        ((1, 4, 32, 64), [0, 2], False),
        ((1, 4, 32, 64), [0, 3], True),
        ((1, 4, 32, 64), [0, 3], False),
        ((1, 4, 32, 64), [1, 2], True),
        ((1, 4, 32, 64), [1, 2], False),
        ((1, 4, 32, 64), [1, 3], True),
        ((1, 4, 32, 64), [1, 3], False),
        ((1, 4, 32, 64), [2, 3], True),
        ((1, 4, 32, 64), [2, 3], False),
        # 4D - 3-dim combinations
        ((1, 4, 32, 64), [0, 1, 2], True),
        ((1, 4, 32, 64), [0, 1, 2], False),
        ((1, 4, 32, 64), [0, 1, 3], True),
        ((1, 4, 32, 64), [0, 1, 3], False),
        ((1, 4, 32, 64), [0, 2, 3], True),
        ((1, 4, 32, 64), [0, 2, 3], False),
        pytest.param(
            (1, 4, 32, 64),
            [1, 2, 3],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (1, 4, 32, 64),
            [1, 2, 3],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 4D - all dims
        pytest.param(
            (1, 4, 32, 64),
            [0, 1, 2, 3],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (1, 4, 32, 64),
            [0, 1, 2, 3],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
    ],
)
@pytest.mark.push
def test_reduce_mean(input_shape, dim, keepdim):
    class ReduceMean(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.mean(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape)]
    framework_model = ReduceMean()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        # 1D
        pytest.param(
            (64,), 0, True, marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs")
        ),
        pytest.param(
            (64,),
            -1,
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (64,),
            0,
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (64,),
            -1,
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 2D - single dim
        ((32, 64), 0, True),
        ((32, 64), 0, False),
        ((32, 64), 1, True),
        ((32, 64), 1, False),
        ((32, 64), -1, True),
        ((32, 64), -1, False),
        ((32, 64), -2, True),
        ((32, 64), -2, False),
        # 2D - multi dim
        pytest.param(
            (32, 64),
            [0, 1],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (32, 64),
            [0, 1],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 3D - single dim
        ((4, 32, 64), 0, True),
        ((4, 32, 64), 0, False),
        ((4, 32, 64), 1, True),
        ((4, 32, 64), 1, False),
        ((4, 32, 64), 2, True),
        ((4, 32, 64), 2, False),
        ((4, 32, 64), -1, True),
        ((4, 32, 64), -1, False),
        ((4, 32, 64), -2, True),
        ((4, 32, 64), -2, False),
        ((4, 32, 64), -3, True),
        ((4, 32, 64), -3, False),
        # 3D - multi dim
        ((4, 32, 64), [0, 1], True),
        ((4, 32, 64), [0, 1], False),
        ((4, 32, 64), [0, 2], True),
        ((4, 32, 64), [0, 2], False),
        ((4, 32, 64), [1, 2], True),
        ((4, 32, 64), [1, 2], False),
        pytest.param(
            (4, 32, 64),
            [0, 1, 2],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (4, 32, 64),
            [0, 1, 2],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 4D - single dim
        ((1, 4, 32, 64), 0, True),
        ((1, 4, 32, 64), 0, False),
        ((1, 4, 32, 64), 1, True),
        ((1, 4, 32, 64), 1, False),
        ((1, 4, 32, 64), 2, True),
        ((1, 4, 32, 64), 2, False),
        ((1, 4, 32, 64), 3, True),
        ((1, 4, 32, 64), 3, False),
        ((1, 4, 32, 64), -1, True),
        ((1, 4, 32, 64), -1, False),
        ((1, 4, 32, 64), -2, True),
        ((1, 4, 32, 64), -2, False),
        ((1, 4, 32, 64), -3, True),
        ((1, 4, 32, 64), -3, False),
        ((1, 4, 32, 64), -4, True),
        ((1, 4, 32, 64), -4, False),
        # 4D - 2-dim combinations
        ((1, 4, 32, 64), [0, 1], True),
        ((1, 4, 32, 64), [0, 1], False),
        ((1, 4, 32, 64), [0, 2], True),
        ((1, 4, 32, 64), [0, 2], False),
        ((1, 4, 32, 64), [0, 3], True),
        ((1, 4, 32, 64), [0, 3], False),
        ((1, 4, 32, 64), [1, 2], True),
        ((1, 4, 32, 64), [1, 2], False),
        ((1, 4, 32, 64), [1, 3], True),
        ((1, 4, 32, 64), [1, 3], False),
        ((1, 4, 32, 64), [2, 3], True),
        ((1, 4, 32, 64), [2, 3], False),
        # 4D - 3-dim combinations
        ((1, 4, 32, 64), [0, 1, 2], True),
        ((1, 4, 32, 64), [0, 1, 2], False),
        ((1, 4, 32, 64), [0, 1, 3], True),
        ((1, 4, 32, 64), [0, 1, 3], False),
        ((1, 4, 32, 64), [0, 2, 3], True),
        ((1, 4, 32, 64), [0, 2, 3], False),
        pytest.param(
            (1, 4, 32, 64),
            [1, 2, 3],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (1, 4, 32, 64),
            [1, 2, 3],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        # 4D - all dims
        pytest.param(
            (1, 4, 32, 64),
            [0, 1, 2, 3],
            True,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
        pytest.param(
            (1, 4, 32, 64),
            [0, 1, 2, 3],
            False,
            marks=pytest.mark.xfail(reason="Data mismatch between framework and compiled model outputs"),
        ),
    ],
)
@pytest.mark.push
def test_reduce_max(input_shape, dim, keepdim):
    class ReduceMax(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            # torch.amax supports both int and list[int] dims.
            return torch.amax(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape)]
    framework_model = ReduceMax()
    framework_model.eval()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        # 2D - single dim
        ((32, 64), 0, True),
        ((32, 64), 0, False),
        ((32, 64), 1, True),
        ((32, 64), 1, False),
        ((32, 64), -1, True),
        ((32, 64), -1, False),
        # 3D - single dim
        ((4, 32, 64), 0, True),
        ((4, 32, 64), 0, False),
        ((4, 32, 64), 1, True),
        ((4, 32, 64), 1, False),
        ((4, 32, 64), 2, True),
        ((4, 32, 64), 2, False),
        # 3D - multi dim
        ((4, 32, 64), [0, 1], True),
        ((4, 32, 64), [0, 1], False),
        ((4, 32, 64), [0, 2], True),
        ((4, 32, 64), [0, 2], False),
        ((4, 32, 64), [1, 2], True),
        ((4, 32, 64), [1, 2], False),
        # 4D - single dim
        ((1, 4, 32, 64), 1, True),
        ((1, 4, 32, 64), 1, False),
        ((1, 4, 32, 64), 2, True),
        ((1, 4, 32, 64), 2, False),
        ((1, 4, 32, 64), 3, True),
        ((1, 4, 32, 64), 3, False),
        # 4D - multi dim
        ((1, 4, 32, 64), [0, 1], True),
        ((1, 4, 32, 64), [0, 1], False),
        ((1, 4, 32, 64), [1, 2], True),
        ((1, 4, 32, 64), [1, 2], False),
        ((1, 4, 32, 64), [2, 3], True),
        ((1, 4, 32, 64), [2, 3], False),
    ],
)
@pytest.mark.push
def test_reduce_sum_backward(input_shape, dim, keepdim):
    class ReduceSum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.sum(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape, requires_grad=True)]
    framework_model = ReduceSum()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, training=True)
    verify(inputs, framework_model, compiled_model, with_backward=True)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        # 2D - single dim
        ((32, 64), 0, True),
        ((32, 64), 0, False),
        ((32, 64), 1, True),
        ((32, 64), 1, False),
        ((32, 64), -1, True),
        ((32, 64), -1, False),
        # 3D - single dim
        ((4, 32, 64), 0, True),
        ((4, 32, 64), 0, False),
        ((4, 32, 64), 1, True),
        ((4, 32, 64), 1, False),
        ((4, 32, 64), 2, True),
        ((4, 32, 64), 2, False),
        # 3D - multi dim
        ((4, 32, 64), [0, 1], True),
        ((4, 32, 64), [0, 1], False),
        ((4, 32, 64), [0, 2], True),
        ((4, 32, 64), [0, 2], False),
        ((4, 32, 64), [1, 2], True),
        ((4, 32, 64), [1, 2], False),
        # 4D - single dim
        ((1, 4, 32, 64), 1, True),
        ((1, 4, 32, 64), 1, False),
        ((1, 4, 32, 64), 2, True),
        ((1, 4, 32, 64), 2, False),
        ((1, 4, 32, 64), 3, True),
        ((1, 4, 32, 64), 3, False),
        # 4D - multi dim
        ((1, 4, 32, 64), [0, 1], True),
        ((1, 4, 32, 64), [0, 1], False),
        ((1, 4, 32, 64), [1, 2], True),
        ((1, 4, 32, 64), [1, 2], False),
        ((1, 4, 32, 64), [2, 3], True),
        ((1, 4, 32, 64), [2, 3], False),
    ],
)
@pytest.mark.push
def test_reduce_mean_backward(input_shape, dim, keepdim):
    class ReduceMean(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.mean(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape, requires_grad=True)]
    framework_model = ReduceMean()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, training=True)
    verify(inputs, framework_model, compiled_model, with_backward=True)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        # 2D - single dim
        ((32, 64), 0, True),
        ((32, 64), 0, False),
        ((32, 64), 1, True),
        ((32, 64), 1, False),
        ((32, 64), -1, True),
        ((32, 64), -1, False),
        # 3D - single dim
        ((4, 32, 64), 0, True),
        ((4, 32, 64), 0, False),
        ((4, 32, 64), 1, True),
        ((4, 32, 64), 1, False),
        ((4, 32, 64), 2, True),
        ((4, 32, 64), 2, False),
        # 3D - multi dim
        ((4, 32, 64), [0, 1], True),
        ((4, 32, 64), [0, 1], False),
        # 4D - single dim
        ((1, 4, 32, 64), 1, True),
        ((1, 4, 32, 64), 1, False),
        ((1, 4, 32, 64), 2, True),
        ((1, 4, 32, 64), 2, False),
        ((1, 4, 32, 64), 3, True),
        ((1, 4, 32, 64), 3, False),
        # 4D - multi dim
        ((1, 4, 32, 64), [0, 1], True),
        ((1, 4, 32, 64), [0, 1], False),
        ((1, 4, 32, 64), [1, 2], True),
        ((1, 4, 32, 64), [1, 2], False),
    ],
)
@pytest.mark.push
def test_reduce_max_backward(input_shape, dim, keepdim):
    class ReduceMax(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.amax(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape, requires_grad=True)]
    framework_model = ReduceMax()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, training=True)
    verify(inputs, framework_model, compiled_model, with_backward=True)
