# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
import torch.nn.functional as F
import onnx
import os


@pytest.mark.parametrize(
    "input_shape, scale_factor",
    [
        ((1, 256, 4, 128), 2),
        (
            (3, 32, 10, 10),
            4,
        ),
        ((2, 98, 6, 6), 7),
        ((4, 18, 8, 8), 3),
        ((2, 50, 12, 12), 5),
        ((2, 64, 7, 7), 8),
        ((5, 100, 2, 4), 10),
        ((1, 49, 4, 4), 7),
        ((4, 36, 5, 5), 6),
    ],
)
@pytest.mark.push
def test_pixel_shuffle(input_shape, scale_factor):
    class PixelShuffleModel(nn.Module):
        def __init__(self, scale_factor):
            super().__init__()
            self.model = nn.PixelShuffle(scale_factor)

        def forward(self, x):
            return self.model(x)

    inputs = [torch.randn(*input_shape)]
    framework_model = PixelShuffleModel(scale_factor)
    framework_model.eval()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(
            (1, 256, 6, 6),
            torch.float32,
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
        pytest.param(
            (224, 224), torch.float32, marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath")
        ),
        pytest.param(
            (1, 8, 224, 224),
            torch.float32,
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
        pytest.param(
            (4, 1, 32, 32),
            torch.float32,
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
        ((2, 128, 8, 8), torch.float16),
        ((512,), torch.float16),
        ((6, 1, 900, 256), torch.float16),
        ((8, 64, 32, 32, 45), torch.float16),
    ],
)
@pytest.mark.push
def test_nan_to_num(shape, dtype):
    class nan_to_num(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.nan_to_num(x1)

    compiler_cfg = forge.config.CompilerConfig()

    if dtype == torch.float16:
        # set compie depth to avoid Unsupported ttnn::DataType , Fatal Python error: Aborted
        compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    inputs = [torch.randn(shape, dtype=dtype)]

    mask_nan = torch.rand(shape, dtype=dtype) < 0.1
    mask_posinf = torch.rand(shape, dtype=dtype) < 0.05
    mask_neginf = torch.rand(shape, dtype=dtype) < 0.05

    inputs[0][mask_nan] = float("nan")
    inputs[0][mask_posinf] = float("inf")
    inputs[0][mask_neginf] = float("-inf")

    framework_model = nan_to_num()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)
    if dtype == torch.float32:
        verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(
            (1, 256),
            torch.float32,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.float32"
            ),
        ),
        pytest.param(
            (1, 512, 14, 14),
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.bfloat16"
            ),
        ),
        pytest.param(
            (1, 3, 224),
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.bfloat16"
            ),
        ),
        pytest.param(
            (1, 8, 224, 224),
            torch.float32,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.float32"
            ),
        ),
        pytest.param(
            (1, 8, 16, 128, 128),
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.bfloat16"
            ),
        ),
        pytest.param(
            (4, 1, 32, 32),
            torch.float32,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.float32"
            ),
        ),
        pytest.param(
            (100,),
            torch.float32,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.float32"
            ),
        ),
        pytest.param(
            (1, 8, 64, 32, 32),
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.bfloat16"
            ),
        ),
    ],
)
@pytest.mark.push
def test_isnan(shape, dtype):
    class isnan(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.isnan(x1)

    inputs = [torch.randn(shape, dtype=dtype)]
    mask_nan = torch.rand(shape) < 0.1
    inputs[0][mask_nan] = float("nan")

    framework_model = isnan()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [(888), (1, 7, 256), (3, 128, 128), (1, 10), (2, 2, 2), (5, 5), (1, 3, 224, 224), (8, 16, 32), (1, 3, 2, 544, 544)],
)
@pytest.mark.push
def test_atan(shape):
    class Atan(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.atan(x1)

    inputs = [torch.randn(shape)]

    framework_model = Atan()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (1, 96),
        (10, 10),
        (1, 96, 54),
        (1, 64, 128),
        (1, 96, 54, 54),
        (1, 3, 224, 224),
        (1, 64, 128, 128),
        (1, 96, 28, 28),
        (1, 1, 128, 128),
    ],
)
@pytest.mark.push
def test_power(shape):
    class power(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.pow(x, 0.75)

    inputs = [torch.rand(shape)]

    framework_model = power()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 7, 256),
    ],
)
@pytest.mark.push
def test_sin(shape):
    class Sin(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sin(x)

    inputs = [torch.rand(shape)]

    framework_model = Sin()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 7, 256),
    ],
)
@pytest.mark.push
def test_cosine(shape):
    class Cosine(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cos(x)

    inputs = [torch.rand(shape)]

    framework_model = Cosine()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 768),
    ],
)
@pytest.mark.xfail(reason="Found Unsupported operations while lowering from TTForge to TTIR in forward graph")
@pytest.mark.push
def test_tanh(shape):
    class Tanh(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tanh(x)

    inputs = [torch.rand(shape)]

    framework_model = Tanh()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 512, 512),
    ],
)
@pytest.mark.push
def test_leakyrelu(shape):

    inputs = [torch.rand(shape)]

    framework_model = nn.LeakyReLU(negative_slope=0.1)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 4096),
    ],
)
@pytest.mark.push
def test_gelu(shape):

    inputs = [torch.rand(shape)]

    framework_model = nn.GELU()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape, min_val, max_val",
    [
        ((1, 1, 256, 256), 0, 1),
        ((1, 96, 96, 24), 6.0, 0.0),
        ((1, 1, 32, 32), -0.5, 0.5),
        ((2, 10, 5, 20), 2.0, -1.0),
        ((3, 3, 3, 3), -3.0, -1.0),
        ((1, 64, 64), -0.5, 0.5),
        ((1, 128, 128), 1.0, -1.0),
        ((2, 2, 2), -1.0, 0.0),
        ((32, 32), -0.2, 0.2),
        ((3, 3), -0.5, -0.2),
        ((4,), 0.0, 2.0),
        ((8,), -3.0, -1.0),
    ],
)
@pytest.mark.push
def test_clip(shape, min_val, max_val):
    class Clip(nn.Module):
        def __init__(self, min_val, max_val):
            super().__init__()
            self.min_val = min_val
            self.max_val = max_val

        def forward(self, x):
            return torch.clamp(x, self.min_val, self.max_val)

    inputs = [torch.rand(shape)]

    framework_model = Clip(min_val, max_val)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((56), 0),
        ((1, 128), 1),
        ((1, 32), -1),
        ((1, 64, 76), 2),
        ((1, 64, 76, 96), 3),
        pytest.param(
            (1, 64, 86, 100, 120),
            4,
            marks=pytest.mark.xfail(
                reason=" RuntimeError: (dim >= 0 && dim <= 3),info: dim should be 0 - 3, but got: 4"
            ),
        ),
    ],
)
@pytest.mark.push
def test_cumsum(shape, dim):
    class CumSum(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.cumsum(x, dim=self.dim)

    inputs = [torch.rand(shape)]

    framework_model = CumSum(dim)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape", [(1, 1, 256, 256), (1, 1, 1, 128), (1, 1, 1, 384), (1, 1, 32, 32), (1, 1, 6, 6), (1, 1, 29, 29)]
)
@pytest.mark.push
def test_abs(shape):
    class Abs(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.abs(x)

    inputs = [torch.rand(shape)]

    framework_model = Abs()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 28, 28),
        (1, 64, 28, 28),
        (1, 256, 28, 28),
        (1, 128, 14, 14),
        (1, 128, 56, 56),
        (1, 32, 64, 64),
        (1, 512, 7, 7),
    ],
)
@pytest.mark.push
def test_exp(shape):
    class Exp(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(x)

    inputs = [torch.rand(shape)]

    framework_model = Exp()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 28, 28),
        (128, 28, 28),
        (28, 28),
        (28),
        (1, 64, 28, 28),
        (64, 28, 28),
        (1, 256, 28, 28),
        (256, 28, 28),
        (1, 128, 14, 14),
        (128, 14, 14),
        (14, 14),
        (14),
        (1, 128, 56, 56),
        (128, 56, 56),
        (56, 56),
        (56),
        (1, 32, 64, 64),
        (32, 64, 64),
        (64, 64),
        (64),
        (1, 512, 7, 7),
        (512, 7, 7),
        (7, 7),
        (7),
    ],
)
@pytest.mark.push
def test_log(shape):
    class Log(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.log(x)

    inputs = [torch.rand(shape)]

    framework_model = Log()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape_x, shape_y",
    [
        ((1, 12, 256, 256), (1,)),
        ((1, 16, 256, 256), (1,)),
        ((1, 32, 256, 256), (1,)),
        ((1, 12, 32, 32), (1,)),
        ((1, 16, 32, 32), (1,)),
        ((1, 32, 32, 32), (1,)),
    ],
)
@pytest.mark.push
def test_maximum(shape_x, shape_y):
    class Maximum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.maximum(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = Maximum()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_relu():
    class ReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, a):
            return self.relu(a)

    inputs = [torch.rand(1, 32)]

    framework_model = ReLU()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("x_shape", [7, 32, 41])
@pytest.mark.parametrize("y_shape", [7, 32, 41])
@pytest.mark.push
def test_sqrt(x_shape, y_shape):
    class Sqrt(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sqrt(x)

    inputs = [
        torch.rand(1, x_shape, y_shape),
    ]

    framework_model = Sqrt()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (7,),  # 1D tensor
        (32,),  # 1D tensor
        (7, 32),  # 2D tensor
        (32, 41),  # 2D tensor
        (1, 7, 32),  # 3D tensor
        (1, 32, 41),  # 3D tensor
        (1, 7, 32, 41),  # 4D tensor
        (2, 7, 32, 41),  # 4D tensor
    ],
)
@pytest.mark.push
def test_reciprocal(shape):
    class Reciprocal(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reciprocal(x)

    inputs = [
        torch.rand(*shape) + 0.1,  # Adding 0.1 to avoid division by zero
    ]

    framework_model = Reciprocal()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (7,),  # 1D tensor
        (32,),  # 1D tensor
        (7, 32),  # 2D tensor
        (32, 41),  # 2D tensor
        (1, 7, 32),  # 3D tensor
        (1, 32, 41),  # 3D tensor
        (1, 7, 32, 41),  # 4D tensor
        (2, 7, 32, 41),  # 4D tensor
    ],
)
@pytest.mark.push
def test_sigmoid(shape):
    class Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sigmoid(x)

    inputs = [
        torch.rand(*shape),
    ]
    framework_model = Sigmoid()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 768),
    ],
)
@pytest.mark.push
def test_tanh(input_shape):
    class Tanh(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.tanh(a)

    inputs = [torch.rand(input_shape)]

    framework_model = Tanh()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(reason="RuntimeError: BinaryOpType cannot be mapped to BcastOpMath")
@pytest.mark.parametrize(
    "input_data",
    [
        torch.tensor([-0.8166, 1.5308, -0.2530, -0.2091]),
        torch.tensor([-3.7, -1.2, 0.0, 1.5, 3.9]),
        torch.tensor([1.0, 2.0, -1.0, -2.0]),
        torch.tensor([-12345.678, 12345.678, -0.999, 0.999, 3.14159, -3.14159]),
    ],
)
def test_floor(input_data):
    class Floor(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.floor(a)

    framework_model = Floor()
    compiled_model = forge.compile(framework_model, sample_inputs=[input_data], module_name="floor")

    verify([input_data], framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape, dim, keepdim",
    [
        # Core test cases for dimension-specific argmax
        pytest.param(
            (56,),
            0,
            False,
        ),
        ((56,), 0, True),
        ((1, 128), 1, False),
        ((1, 128), 1, True),
        pytest.param(
            (1, 64, 76),
            2,
            False,
        ),
        ((1, 64, 76), 2, True),
        ((1, 64, 76), 1, True),
        #################################################
        # Core test cases for global argmax (dim=None)
        pytest.param(
            (56,),
            None,
            False,
        ),
        ((56,), None, True),
        pytest.param(
            (1, 128),
            None,
            False,
        ),
        ((1, 128), None, True),
    ],
)
@pytest.mark.push
def test_argmax(shape, dim, keepdim):
    class ArgMax(nn.Module):
        def __init__(self, dim, keepdim):
            super().__init__()
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, x):
            return torch.argmax(x, dim=self.dim, keepdim=self.keepdim)

    inputs = [torch.rand(shape)]

    framework_model = ArgMax(dim, keepdim)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "pattern, input_shape",
    [
        ("nhwpqc->nchpwq", [(2, 64, 64, 2, 2, 16)]),
        ("nhwpqc->nchpwq", [(1, 32, 32, 4, 4, 8)]),
        ("nhwpqc->nchpwq", [(4, 16, 16, 2, 2, 32)]),
        ("nhwpqc->nchpwq", [(3, 8, 8, 1, 1, 64)]),
        ("...si,...id->...sd", [(2, 3, 4), (2, 4, 5)]),
        ("...si,...id->...sd", [(5, 6, 7), (5, 7, 8)]),
        ("...si,...id->...sd", [(1, 10, 20), (1, 20, 30)]),
        ("...si,...id->...sd", [(4, 2, 3), (4, 3, 6)]),
        ("bhwc,hkc->bhwk", [(1, 8, 8, 16), (8, 32, 16)]),
        ("bhwc,hkc->bhwk", [(2, 16, 16, 32), (16, 64, 32)]),
        ("bhwc,hkc->bhwk", [(5, 10, 10, 20), (10, 40, 20)]),
        ("bhwc,hkc->bhwk", [(6, 12, 12, 128), (12, 256, 128)]),
        ("bhwc,wkc->bhwk", [(1, 8, 8, 16), (8, 32, 16)]),
        ("bhwc,wkc->bhwk", [(2, 16, 16, 32), (16, 64, 32)]),
        ("bhwc,wkc->bhwk", [(5, 10, 10, 20), (10, 40, 20)]),
        ("bhwc,wkc->bhwk", [(6, 12, 12, 128), (12, 256, 128)]),
        ("bchw,bkc->bkhw", [(2, 4, 1, 1), (2, 1, 4)]),
        ("bchw,bkc->bkhw", [(1, 4, 1, 1), (1, 2, 4)]),
        ("bmchw,bnmc->bmhwn", [(1, 2, 2, 1, 1), (1, 1, 2, 2)]),
        ("bmchw,bnmc->bmhwn", [(1, 5, 10, 1, 1), (1, 1, 5, 10)]),
        ("bmnk,bkmc->bnmc", [(1, 4, 6, 8), (1, 8, 4, 16)]),
        ("bmnk,bkmc->bnmc", [(1, 6, 8, 16), (1, 16, 6, 32)]),
    ],
)
@pytest.mark.push
def test_einsum(pattern, input_shape):
    class EinsumModel(torch.nn.Module):
        def __init__(self, pattern):
            super().__init__()
            self.pattern = pattern

        def forward(self, *inputs):
            return torch.einsum(self.pattern, *inputs)

    inputs = [torch.randn(shape) for shape in input_shape]

    model = EinsumModel(pattern)
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=inputs)

    verify(inputs, model, compiled_model)


@pytest.mark.parametrize("shape", [(8,), (4, 4), (3, 5, 7), (2, 6, 4, 8), (2, 3, 5, 7, 9)])
@pytest.mark.push
def test_res_conj(shape):
    class res_conj(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            output = x.resolve_conj()
            return output

    model = res_conj()
    model.eval()

    inputs = [torch.randn(shape)]

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)


@pytest.mark.parametrize("shape", [(8,), (4, 4), (3, 5, 7), (2, 6, 4, 8), (2, 3, 5, 7, 9)])
@pytest.mark.push
def test_res_neg(shape):
    class res_neg(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            output = x.resolve_neg()
            return output

    model = res_neg()
    model.eval()

    inputs = [torch.randn(shape)]

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=inputs)

    # Model Verification
    verify(inputs, model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, dim, unflattened_size",
    [
        ([10], 0, (5, 2)),
        ([12, 15], 1, (3, 5)),
        ([197, 1, 2304], -1, (3, 768)),
        ([11, 5, 8, 2], 2, (2, 2, 2)),
        ([25, 75, 3, 24], 3, (4, 6)),
        ([5, 5, 30, 2, 60], 4, (2, 2, 3, 5)),
    ],
)
@pytest.mark.push
def test_unflatten(input_shape, dim, unflattened_size):
    class UnflattenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.unflatten(x, dim, unflattened_size)

    input_tensor = torch.randn(*input_shape)
    inputs = [input_tensor]

    model = UnflattenModel()
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=inputs)

    verify(inputs, model, compiled_model)


@pytest.mark.parametrize("hidden_dim", [96, 128, 160, 192])
def test_zero(hidden_dim):
    class ZeroModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, qkv_weight, qkv_bias):
            length = qkv_bias.numel() // 3
            qkv_bias[length : 2 * length].zero_()
            qkv = F.linear(x, qkv_weight, qkv_bias)
            return qkv

    model = ZeroModel()
    model.eval()

    x = torch.rand(4, 16, hidden_dim)
    qkv_weight = torch.rand(3 * hidden_dim, hidden_dim)
    qkv_bias = torch.rand(3 * hidden_dim)

    inputs = [x, qkv_weight, qkv_bias]

    compiled_model = forge.compile(model, sample_inputs=inputs)

    verify(inputs, model, compiled_model)


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 3, 4),
        (1, 10),
        (5, 5, 5),
        (8,),
    ],
)
def test_zeros_like(input_shape):
    class ZerosLikeModel(torch.nn.Module):
        def forward(self, x):
            z = torch.zeros_like(x)
            cond = x > 0
            return torch.where(cond, x, z)

    input_tensor = torch.randn(input_shape)
    inputs = [input_tensor]

    model = ZerosLikeModel()
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )

    verify(inputs, model, compiled_model)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 10),
        (4, 10),
        (1, 1),
        (2, 3, 10),
        (32, 1024),
        (64, 2048),
        (8, 128, 512),
        (16, 512, 1024),
        (4, 64, 64, 256),
        (1, 8192),
        (128, 128),
    ],
)
@pytest.mark.xfail
def test_bernoulli(input_shape):
    class BernoulliNet(nn.Module):
        def __init__(self, in_features):
            super(BernoulliNet, self).__init__()
            self.linear = nn.Linear(in_features, 10)

        def forward(self, x):
            x = torch.sigmoid(self.linear(x))
            x = torch.bernoulli(x)
            return x

    input_tensor = torch.randn(input_shape)
    in_features = input_tensor.shape[-1]
    model = BernoulliNet(in_features)
    inputs = [input_tensor]

    model.eval()
    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )

    verify(inputs, model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, upscale_factor",
    [
        ((1, 3072, 8, 8), 32),
        ((1, 128, 16, 16), 8),
        ((1, 144, 12, 12), 6),
        ((1, 256, 4, 4), 4),
        ((1, 36, 10, 10), 6),
    ],
)
@pytest.mark.push
def test_depth_to_space(input_shape, upscale_factor):
    class DepthToSpace(nn.Module):
        def __init__(self, upscale_factor):
            super().__init__()
            self.model = nn.PixelShuffle(upscale_factor)

        def forward(self, x):
            return self.model(x)

    x = torch.randn(*input_shape)
    inputs = [x]
    model = DepthToSpace(upscale_factor)

    # Export model to ONNX
    onnx_path = f"DepthToSpace_up_{upscale_factor}.onnx"
    torch.onnx.export(
        model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs)

    verify(inputs, model, compiled_model)

    # Clean up exported ONNX file
    os.remove(onnx_path)


@pytest.mark.parametrize(
    "input_shape, fill_value, threshold",
    [
        ((8,), 0.0, 0.5),
        ((4, 4), -1.0, 0.3),
        ((2, 3, 5), 2.5, 0.7),
        ((1, 3, 32, 32), 0.5, 0.5),
    ],
)
@pytest.mark.push
def test_masked_fill(input_shape, fill_value, threshold):
    class MaskedFillModel(nn.Module):
        def __init__(self, fill_value, threshold):
            super().__init__()
            self.fill_value = fill_value
            self.threshold = threshold

        def forward(self, x):
            mask = x > self.threshold
            return x.masked_fill(mask, self.fill_value)

    inputs = [torch.randn(*input_shape)]

    framework_model = MaskedFillModel(fill_value, threshold)
    framework_model.eval()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 3, 4),
        (1, 10),
        (5, 5, 5),
        (8,),
    ],
)
@pytest.mark.push
def test_new_zeros(input_shape):
    class NewZerosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = x.new_zeros(x.shape)
            cond = x > 0
            return torch.where(cond, x, z)

    input_tensor = torch.randn(input_shape)
    inputs = [input_tensor]

    model = NewZerosModel()
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )

    verify(inputs, model, compiled_model)
