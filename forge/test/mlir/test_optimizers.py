# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn

import forge
from forge._C.runtime import Tensor as CTensor
from test.mlir.utils import get_param_grads, copy_params
from test.mlir.mnist.utils import MNISTLinear
from forge.verify.compare import compare_with_golden


def train_and_compare_optimizers(
    num_epochs, batch_size, shape, loss_fn, tt_model, tt_optimizer, golden_model, golden_optimizer
):
    for epoch in range(num_epochs):
        # Generate random data
        x = torch.randn(batch_size, shape[0])
        y = torch.randn(batch_size, shape[2])

        golden_optimizer.zero_grad()

        # Forward pass
        gold_out = golden_model(x)

        loss = loss_fn(gold_out, y)
        loss.backward()

        # Copy the gradients to the TT model since just the optimizer should be tested
        ordered_param_names = tt_model.fwd_compiled_graph_state.ordered_parameter_node_names
        grads = get_param_grads(golden_model.named_parameters)
        grad_list = []
        for name in ordered_param_names:
            grad_list.append(grads[name])
        gradient_outputs = grad_list[::-1]
        tt_model.gradient_outputs = [CTensor(grad) for grad in gradient_outputs]

        # Step
        tt_optimizer.step()
        tt_model.update_host_weights()

        golden_optimizer.step()

        # Compare all the parameters
        # TODO: Switch to verify once this is closed
        # https://github.com/tenstorrent/tt-forge-onnx/issues/1310
        for i, (tt_param, golden_param) in enumerate(
            zip(tt_model.framework_module.module.parameters(), golden_model.parameters())
        ):
            assert compare_with_golden(
                tt_param, golden_param, pcc=0.99
            ), f"Weight mismatch at epoch {epoch}\n {tt_param}, {golden_param} and param {i}"

        print(f"Epoch: {epoch}, Loss: {loss.item()}")


@pytest.mark.parametrize(
    "shape",
    [
        # input, hidden, output
        (784, 784, 10),
        (33, 27, 127),
        (128, 10, 20),
    ],
)
@pytest.mark.push
def test_sgd(shape):
    torch.manual_seed(0)
    num_epochs = 10
    # Large learning rate to propagate possible errors faster
    learning_rate = 1
    batch_size = 10

    framework_model = MNISTLinear(input_size=shape[0], hidden_size=shape[1], output_size=shape[2], bias=False)
    golden_model = MNISTLinear(input_size=shape[0], hidden_size=shape[1], output_size=shape[2], bias=False)

    copy_params(framework_model, golden_model)

    tt_optimizer = forge.optimizers.SGD(learning_rate=learning_rate)
    golden_optimizer = torch.optim.SGD(golden_model.parameters(), lr=learning_rate)

    tt_model = forge.compile(
        framework_model,
        sample_inputs=[torch.randn(batch_size, shape[0])],
        optimizer=tt_optimizer,
        training=True,
    )

    loss_fn = nn.MSELoss()

    train_and_compare_optimizers(
        num_epochs=num_epochs,
        batch_size=batch_size,
        shape=shape,
        loss_fn=loss_fn,
        tt_model=tt_model,
        tt_optimizer=tt_optimizer,
        golden_model=golden_model,
        golden_optimizer=golden_optimizer,
    )


@pytest.mark.parametrize(
    "shape",
    [
        # input, hidden, output
        (784, 784, 10),
        (33, 27, 127),
        (128, 10, 20),
    ],
)
@pytest.mark.parametrize(
    "betas",
    [
        (0.9, 0.999),
        (0.8, 0.99),
    ],
)
@pytest.mark.parametrize(
    "weight_decay",
    [
        0.0,
        0.1,
    ],
)
@pytest.mark.push
def test_adam(shape, betas, weight_decay):
    torch.manual_seed(0)
    num_epochs = 10
    # Large learning rate to propagate possible errors faster
    learning_rate = 1
    batch_size = 10
    eps = 1e-8

    framework_model = MNISTLinear(input_size=shape[0], hidden_size=shape[1], output_size=shape[2], bias=False)
    golden_model = MNISTLinear(input_size=shape[0], hidden_size=shape[1], output_size=shape[2], bias=False)

    copy_params(framework_model, golden_model)

    tt_optimizer = forge.optimizers.Adam(
        learning_rate=learning_rate,
        epsilon=eps,
        beta1=betas[0],
        beta2=betas[1],
        weight_decay=weight_decay,
        bias_correction=True,
    )
    golden_optimizer = torch.optim.Adam(
        golden_model.parameters(), lr=learning_rate, eps=eps, betas=betas, weight_decay=weight_decay
    )

    # Issue: https://github.com/tenstorrent/tt-mlir/issues/6915
    # Once the issue is fixed, CPU Hoisted Const Eval should be enabled
    mlir_config = forge.config.MLIRConfig()
    mlir_config.set_custom_config("enable-cpu-hoisted-const-eval=false")

    tt_model = forge.compile(
        framework_model,
        sample_inputs=[torch.randn(batch_size, shape[0])],
        optimizer=tt_optimizer,
        training=True,
        compiler_cfg=forge.CompilerConfig(mlir_config=mlir_config),
    )

    loss_fn = nn.MSELoss()

    train_and_compare_optimizers(
        num_epochs=num_epochs,
        batch_size=batch_size,
        shape=shape,
        loss_fn=loss_fn,
        tt_model=tt_model,
        tt_optimizer=tt_optimizer,
        golden_model=golden_model,
        golden_optimizer=golden_optimizer,
    )
