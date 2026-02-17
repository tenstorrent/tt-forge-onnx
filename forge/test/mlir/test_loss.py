# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker

import forge


class NLLLossWrapper(torch.nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = torch.nn.NLLLoss(reduction=reduction)

    def forward(self, prediction, target_one_hot):
        # Forge passes one-hot targets, but Torch NLLLoss expects class indices.
        target_index = torch.argmax(target_one_hot, dim=-1)
        return self.loss_fn(prediction, target_index)


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.push
def test_l1_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.L1Loss("l1_loss", reduction=reduction)
    torch_loss = torch.nn.L1Loss(reduction=reduction)

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.randn((prediction_shape))
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])
    verify(
        inputs=[prediction, target],
        framework_model=torch_loss,
        compiled_model=forge_loss,
        verify_cfg=VerifyConfig(verify_shape=False, value_checker=AllCloseValueChecker(rtol=11e-3)),
    )


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
    ],
)
@pytest.mark.push
def test_cross_entropy_loss(prediction_shape):
    forge_loss = forge.op.loss.CrossEntropyLoss("cross_entropy_loss")
    torch_loss = torch.nn.CrossEntropyLoss()

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.empty(prediction_shape[0], dtype=torch.long).random_(prediction_shape[-1])
    target = nn.functional.one_hot(target, num_classes=prediction_shape[-1]).float()
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])

    verify(
        inputs=[prediction, target],
        framework_model=torch_loss,
        compiled_model=forge_loss,
        verify_cfg=VerifyConfig(verify_shape=False, value_checker=AllCloseValueChecker(rtol=11e-3)),
    )


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_kl_div_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.KLDivLoss("kl_div_loss", reduction=reduction)
    torch_loss = torch.nn.KLDivLoss(reduction=reduction)

    prediction = nn.functional.log_softmax(torch.randn(prediction_shape, requires_grad=True), dim=-1)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.randn(prediction_shape)
    # softmax the target
    target = nn.functional.softmax(target, dim=-1)
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])

    verify(
        inputs=[prediction, target],
        framework_model=torch_loss,
        compiled_model=forge_loss,
        verify_cfg=VerifyConfig(verify_shape=False, value_checker=AllCloseValueChecker(rtol=5e-2)),
    )


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_mse_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.MSELoss("mse_loss", reduction=reduction)
    torch_loss = torch.nn.MSELoss(reduction=reduction)

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.randn((prediction_shape))
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])

    # relative tolerance is 5% and absolute tolerance is 0.005
    verify(
        inputs=[prediction, target],
        framework_model=torch_loss,
        compiled_model=forge_loss,
        verify_cfg=VerifyConfig(verify_shape=False, value_checker=AllCloseValueChecker(rtol=5e-2, atol=5e-3)),
    )


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_nll_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.NLLLoss("nll_loss", reduction=reduction)
    torch_loss = NLLLossWrapper(reduction=reduction)

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction = nn.functional.log_softmax(prediction, dim=-1)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)

    batch_size = prediction_shape[0] if len(prediction_shape) > 1 else 1
    target = torch.randint(0, prediction_shape[-1], (batch_size,), dtype=torch.long)

    # Because of the following error
    # RuntimeError: TT_FATAL @ ../embedding_device_operation.cpp:28: weights.get_dtype() == DataType::BFLOAT16
    # We need to convert the target to one hot, which is different from torch
    # https://github.com/tenstorrent/tt-mlir/issues/1503
    target_one_hot = nn.functional.one_hot(target, num_classes=prediction_shape[-1]).float()

    if batch_size == 1:  # Handle 1D case, remove the batch dimension
        target_one_hot = target_one_hot.squeeze(0)

    target_forge = forge.tensor.Tensor.create_from_torch(target_one_hot)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])
    verify(
        inputs=[prediction, target_one_hot],
        framework_model=torch_loss,
        compiled_model=forge_loss,
        verify_cfg=VerifyConfig(verify_shape=False, value_checker=AllCloseValueChecker(rtol=11e-3)),
    )


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_huber_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.HuberLoss("huber_loss", delta=1.0, reduction=reduction)
    torch_loss = torch.nn.HuberLoss(reduction=reduction, delta=1.0)

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.randn(prediction_shape)
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])

    verify(
        inputs=[prediction, target],
        framework_model=torch_loss,
        compiled_model=forge_loss,
        verify_cfg=VerifyConfig(verify_shape=False, value_checker=AllCloseValueChecker(rtol=5e-2)),
    )


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_bce_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.BCELoss("bce_loss", reduction=reduction)
    torch_loss = torch.nn.BCELoss(reduction=reduction)

    prediction = nn.functional.sigmoid(torch.randn(prediction_shape, requires_grad=True))
    target = torch.rand(prediction_shape)

    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])

    verify(
        inputs=[prediction, target],
        framework_model=torch_loss,
        compiled_model=forge_loss,
        verify_cfg=VerifyConfig(verify_shape=False, value_checker=AllCloseValueChecker(rtol=5e-2, atol=5e-3)),
    )


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_bce_with_logits_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.BCEWithLogitsLoss("bce_with_logits_loss", reduction=reduction)
    torch_loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    prediction = torch.randn(prediction_shape, requires_grad=True)
    target = torch.rand(prediction_shape)

    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])

    verify(
        inputs=[prediction, target],
        framework_model=torch_loss,
        compiled_model=forge_loss,
        verify_cfg=VerifyConfig(verify_shape=False, value_checker=AllCloseValueChecker(rtol=5e-2, atol=5e-3)),
    )


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("margin", [0.5, 2.0])
@pytest.mark.parametrize("eps", [1e-6, 1e-2])
@pytest.mark.parametrize("swap", [True, False])
def test_triplet_margin_loss(prediction_shape, reduction, margin, eps, swap):
    forge_loss = forge.op.loss.TripletMarginLoss(
        "triplet_margin_loss", margin=margin, reduction=reduction, eps=eps, swap=swap
    )
    torch_loss = torch.nn.TripletMarginLoss(margin=margin, p=2.0, reduction=reduction, eps=eps, swap=swap)

    anchor = torch.randn(prediction_shape, requires_grad=True)
    anchor_forge = forge.tensor.Tensor.create_from_torch(anchor)
    positive = torch.randn(prediction_shape, requires_grad=True)
    positive_forge = forge.tensor.Tensor.create_from_torch(positive)
    negative = torch.randn(prediction_shape, requires_grad=True)
    negative_forge = forge.tensor.Tensor.create_from_torch(negative)

    forge_loss = forge.compile(
        forge_loss,
        sample_inputs=[anchor_forge, positive_forge, negative_forge],
    )

    verify(
        inputs=[anchor, positive, negative],
        framework_model=torch_loss,
        compiled_model=forge_loss,
        verify_cfg=VerifyConfig(verify_shape=False, value_checker=AllCloseValueChecker(rtol=5e-2)),
    )
