# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify import DeprecatedVerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((10,), 0),
        ((5, 10), 1),
        ((3, 5, 10), 2),
        ((2, 3, 5, 10), 3),
        ((1, 6, 20, 50, 64), 4),
    ],
)
@pytest.mark.push
def test_stack_and_view(shape, dim):
    class stack_and_view(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x, y):
            stacked = torch.stack((x, y), dim=self.dim)
            new_shape = list(x.shape)
            new_shape[self.dim] *= 2
            return stacked.view(*new_shape)

    x = torch.rand(shape)
    y = torch.rand(shape)

    inputs = [x, y]

    framework_model = stack_and_view(dim)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "einsum_pattern, shape_1, shape_2",
    [
        pytest.param(
            "bqnc,bnchw->bqnhw",
            (1, 100, 8, 32),
            (1, 8, 32, 14, 20),
        ),
        pytest.param(
            "bqnc,bnchw->bqnhw",
            (3, 99, 7, 31),
            (3, 7, 31, 15, 19),
        ),
    ],
)
@pytest.mark.push
def test_einsum(einsum_pattern, shape_1, shape_2):
    class EinsumModel(torch.nn.Module):
        def __init__(self, pattern):
            super().__init__()
            self.pattern = pattern

        def forward(self, input_1, input_2):
            return torch.einsum(self.pattern, input_1, input_2)

    input_1 = torch.randn(*shape_1)
    input_2 = torch.randn(*shape_2)
    inputs = [input_1, input_2]

    framework_model = EinsumModel(einsum_pattern)
    framework_model.eval()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (300, 1),
        (1, 6, 18),
        (2, 2, 2),
        (5, 5),
        (745),
        (1, 256, 6, 6),
        (1, 512, 14, 14),
        (1, 3, 224, 224),
        (1, 34, 200, 224, 53),
    ],
)
@pytest.mark.push
def test_atan2(shape):
    class Atan2(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return torch.atan2(x2, x1)

    inputs = [torch.randn(shape), torch.randn(shape)]
    framework_model = Atan2()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape_x, shape_y",
    [
        ((1, 128, 28, 28), (1, 128, 28, 28)),
        ((1, 64, 28, 28), (1, 64, 28, 28)),
        ((1, 256, 28, 28), (1, 256, 28, 28)),
        ((1, 128, 14, 14), (1, 128, 14, 14)),
        ((1, 128, 56, 56), (1, 128, 56, 56)),
        ((1, 32, 64, 64), (1, 32, 64, 64)),
        ((1, 512, 7, 7), (1, 512, 7, 7)),
        ((1, 32, 32, 32), (1, 32, 32, 32)),
    ],
)
@pytest.mark.push
def test_less(shape_x, shape_y):
    class Less(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.less(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = Less()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(verify_dtype=False),
    )


@pytest.mark.parametrize(
    "shape_x, shape_y",
    [
        ((1, 128, 28, 28), (1, 128, 28, 28)),
        ((1, 64, 28, 28), (1, 64, 28, 28)),
        ((1, 256, 28, 28), (1, 256, 28, 28)),
        ((1, 128, 14, 14), (1, 128, 14, 14)),
        ((1, 128, 56, 56), (1, 128, 56, 56)),
        ((1, 32, 64, 64), (1, 32, 64, 64)),
        ((1, 512, 7, 7), (1, 512, 7, 7)),
        ((1, 32, 32, 32), (1, 32, 32, 32)),
    ],
)
@pytest.mark.push
def test_less_equal(shape_x, shape_y):
    class LessEqual(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.less_equal(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = LessEqual()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(verify_dtype=False),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (888,),
        (1, 7, 256),
        (3, 128, 128),
        (1, 10),
        (2, 2, 2),
        (5, 5),
        (1, 3, 224, 224),
        (8, 16, 32),
        (1, 3, 2, 544, 544),
    ],
)
def test_bitwise_and(shape):
    class BitwiseAnd(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return torch.bitwise_and(x1, x2)

    inputs = [
        torch.randint(0, 256, shape, dtype=torch.int32),
        torch.randint(0, 256, shape, dtype=torch.int32),
    ]

    framework_model = BitwiseAnd()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape_x, shape_y",
    [
        ((1, 128, 28, 28), (1, 128, 28, 28)),
        ((1, 64, 28, 28), (1, 64, 28, 28)),
        ((1, 256, 28, 28), (1, 256, 28, 28)),
        ((1, 128, 14, 14), (1, 128, 14, 14)),
        ((1, 128, 56, 56), (1, 128, 56, 56)),
        ((1, 32, 64, 64), (1, 32, 64, 64)),
        ((1, 512, 7, 7), (1, 512, 7, 7)),
        ((1, 32, 32, 32), (1, 32, 32, 32)),
    ],
)
@pytest.mark.push
def test_greater(shape_x, shape_y):
    class Greater(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.greater(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = Greater()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(verify_dtype=False),
    )


@pytest.mark.parametrize(
    "shape_x, shape_y",
    [
        ((1, 128, 28, 28), (1, 128, 28, 28)),
        ((1, 64, 28, 28), (1, 64, 28, 28)),
        ((1, 256, 28, 28), (1, 256, 28, 28)),
        ((1, 128, 14, 14), (1, 128, 14, 14)),
        ((1, 128, 56, 56), (1, 128, 56, 56)),
        ((1, 32, 64, 64), (1, 32, 64, 64)),
        ((1, 512, 7, 7), (1, 512, 7, 7)),
        ((1, 32, 32, 32), (1, 32, 32, 32)),
    ],
)
@pytest.mark.push
def test_not_equal(shape_x, shape_y):
    class NotEqual(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.ne(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = NotEqual()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(verify_dtype=False),
    )


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
        (1, 32, 32, 32),
        (128, 28, 28),
        (64, 28, 28),
        (256, 28, 28),
        (128, 14, 14),
        (128, 56, 56),
        (32, 64, 64),
        (512, 7, 7),
        (32, 32, 32),
        (128, 28),
        (64, 28),
        (256, 28),
        (128, 14),
        (128, 56),
        (32, 64),
        (512, 7),
        (32, 32),
    ],
)
@pytest.mark.push
def test_equal(shape):
    class Equal(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.eq(x, y)

    x = torch.rand(shape)
    y = x * 2.0

    inputs = [x, y]

    framework_model = Equal()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(verify_dtype=False),
    )


@pytest.mark.parametrize(
    "shape_dtype",
    [
        ((2, 32, 32), torch.float32),
        ((1, 128), torch.int64),
    ],
)
@pytest.mark.push
def test_add(shape_dtype):
    shape, dtype = shape_dtype

    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    # Generate random tensors of the appropriate shape and dtype
    a = torch.rand(size=shape).to(dtype)
    b = torch.rand(size=shape).to(dtype)

    inputs = [a, b]

    framework_model = Add()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("dims", [(1, 32, 64), (6, 33), (4, 16, 17)])
@pytest.mark.push
def test_greater_equal(dims):
    class GreaterEqual(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.greater_equal(a, b)

    inputs = [torch.rand(dims), torch.rand(dims)]

    framework_model = GreaterEqual()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(verify_dtype=False),
    )


@pytest.mark.parametrize("dims", [(1, 32, 64)])
@pytest.mark.push
def test_power(dims):
    class Power(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.pow(a, b)

    # Use positive values for base to avoid complex numbers
    input1, input2 = torch.rand(dims) + 0.1, torch.rand(dims)
    input1.requires_grad = True
    input2.requires_grad = True
    inputs = [input1, input2]

    framework_model = Power()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, training=True)

    verify(
        inputs,
        framework_model,
        compiled_model,
        with_backward=True,
    )


@pytest.mark.push
def test_subtract():
    class Subtract(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a - b

    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]

    framework_model = Subtract()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(verify_dtype=False),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 32),
        (12, 8640),
    ],
)
@pytest.mark.push
def test_multiply(shape):
    class Multiply(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a * b

    inputs = [torch.rand(shape), torch.rand(shape)]

    framework_model = Multiply()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        ((1, 32, 32), (1, 32, 32)),  # Same shapes
        ((32, 32, 1), (1,)),  # Broadcasting scalar
        ((32, 32), (1, 32, 32)),  # Broadcasting different ranks
        ((1, 1, 32), (1, 32, 1)),  # Broadcasting different dimensions
    ],
)
@pytest.mark.push
def test_divide(shape):
    class Divide(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.div(a, b)

    shape_a, shape_b = shape
    is_training = True

    input_a = torch.randn(shape_a, requires_grad=is_training)
    input_b = torch.randn(shape_b, requires_grad=is_training)

    # Avoid division by zero
    with torch.no_grad():
        input_b[input_b.abs() < 0.1] = 1.0

    inputs = [input_a, input_b]

    framework_model = Divide()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, training=is_training)

    verify(inputs, framework_model, compiled_model, with_backward=True)


@pytest.mark.push
def test_remainder():
    class Remainder(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a % b

    inputs = [torch.rand(2, 32, 32), torch.rand(2, 32, 32)]

    framework_model = Remainder()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@torch.jit.script
def make_log_bucket_position(relative_pos, bucket_size: int):
    mid = bucket_size // 2
    abs_pos = relative_pos * mid
    return abs_pos


@pytest.mark.push
@pytest.mark.parametrize("bucket_size", [5, 16, 127, 256, 513])
def test_floordiv(bucket_size):
    class floordiv(nn.Module):
        def __init__(self, bucket_size):
            super().__init__()
            self.bucket_size = bucket_size

        def forward(self, x):
            op = make_log_bucket_position(x, self.bucket_size)
            return op

    x = torch.randn(41, 41)
    inputs = [x]

    framework_model = floordiv(bucket_size)
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        verify_cfg=DeprecatedVerifyConfig(verify_forge_codegen_vs_framework=True),
    )
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (3, 3),
        (10, 10),
        (2, 41, 41),
        (8, 96, 96),
    ],
)
@pytest.mark.xfail(
    reason="NotImplementedError: The following operators are not implemented: ['aten::linalg_solve']"
)  # https://github.com/tenstorrent/tt-forge-onnx/issues/1991
def test_linalg_solve(shape):
    class linalg_solve(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.linalg.solve(a, b)

    inputs = [torch.randn(*shape), torch.randn(*shape)]
    framework_model = linalg_solve()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
