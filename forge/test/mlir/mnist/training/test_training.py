# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time

import torch
from torch import nn
from loguru import logger

import forge
from forge.op.loss import CrossEntropyLoss, L1Loss
from forge.tensor import to_forge_tensors
from forge.verify import compare_with_golden, verify, VerifyConfig, AutomaticValueChecker
from forge.verify.config import DeprecatedVerifyConfig
from forge.config import CompileDepth
from ..utils import *
from test.mlir.utils import *
from forge.config import CompilerConfig
from forge._C import DataFormat


@pytest.mark.push
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mnist_training(dtype):
    # Model and data type.
    # For bfloat16, the following line should be added to the test_forge_vs_torch function:
    # In file forge/forge/op/eval/forge/eltwise_unary.py:418 should be replaced with: threshold_tensor = ac.tensor(torch.zeros(shape, dtype=torch.bfloat16) + threshold)
    # That sets relu threshold to bfloat16 tensor.

    # Set training hyperparameters
    num_epochs = 3
    batch_size = 1024
    learning_rate = 0.001

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size, dtype=dtype)

    # Define model and instruct it to compile and run on TT device
    framework_model = MNISTLinear(
        bias=False, dtype=dtype
    )  # bias=False because batch_size=1 with bias=True is not supported

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)

    # Configure compiler only if using bfloat16
    compiler_cfg = CompilerConfig()
    if dtype == torch.bfloat16:
        data_format_override = DataFormat.Float16_b
        compiler_cfg.default_df_override = data_format_override

    tt_model = forge.compile(
        framework_model,
        sample_inputs=[torch.rand(batch_size, 784, dtype=dtype)],
        optimizer=framework_optimizer,
        training=True,
        compiler_cfg=compiler_cfg,
    )

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")
    for epoch_idx in range(num_epochs):

        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Reset gradients (every batch)
            framework_optimizer.zero_grad()

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).to(dtype)

            # Forward pass (prediction) on device
            golden_pred, pred = verify(
                inputs=[data],
                framework_model=framework_model,
                compiled_model=tt_model,
                verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
            )
            golden_pred, pred = golden_pred[0], pred[0]

            # Compute loss on CPU
            loss = loss_fn(pred, target)
            total_loss += loss.item()

            golden_loss = loss_fn(golden_pred, target)
            assert torch.allclose(loss, golden_loss, rtol=1e-1)  # 10% tolerance

            # Loss backward pass on CPU.
            loss.backward()

            # Run backward pass on device.
            tt_model.backward()

            # Adjust weights (on CPU)
            framework_optimizer.step()

        print(f"epoch: {epoch_idx} loss: {total_loss}")

    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = tt_model(data)[0]
        target = nn.functional.one_hot(target, num_classes=10).to(dtype)

        test_loss += loss_fn(pred, target)

    print(f"Test (total) loss: {test_loss}")


@pytest.mark.push
def test_mnist_training_with_grad_accumulation():
    # Config
    num_epochs = 3
    batch_size = 1
    learning_rate = 0.001

    # Limit number of batches to run - quicker test
    limit_num_batches = 10

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Define model and instruct it to compile and run on TT device
    framework_model = MNISTLinear(bias=False)  # bias=False because batch_size=1 with bias=True is not supported

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    tt_model = forge.compile(
        framework_model,
        sample_inputs=[torch.rand(batch_size, 784)],
        training=True,
    )

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")
    for epoch_idx in range(num_epochs):
        # Reset gradients (every epoch) - since our batch size is currently 1,
        # we accumulate gradients across multiple batches (limit_num_batches),
        # and then run the optimizer.
        framework_optimizer.zero_grad()

        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            golden_pred, pred = verify(
                inputs=[data],
                framework_model=framework_model,
                compiled_model=tt_model,
                verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
            )
            golden_pred, pred = golden_pred[0], pred[0]

            # Compute loss on CPU
            loss = loss_fn(pred, target)
            total_loss += loss.item()

            golden_loss = loss_fn(golden_pred, target)
            assert torch.allclose(loss, golden_loss, rtol=1e-1)

            # Run backward pass on device
            loss.backward()

            tt_model.backward()

            if batch_idx >= limit_num_batches:
                break

        print(f"epoch: {epoch_idx} loss: {total_loss}")

        # Adjust weights (on CPU)
        framework_optimizer.step()

    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = tt_model(data)[0]
        target = nn.functional.one_hot(target, num_classes=10).float()

        test_loss += loss_fn(pred, target)

        if batch_idx == 1000:
            break

    print(f"Test (total) loss: {test_loss}")


@pytest.mark.parametrize("freeze_layer", [None, 0, 2, 4])
@pytest.mark.push
def test_forge_vs_torch_gradients(freeze_layer):
    logger.disable("")
    batch_size = 64

    dtype = torch.float32
    torch.set_printoptions(precision=10)

    in_features = 28 * 28
    out_features = 10

    torch_model = MNISTLinear(dtype=dtype, bias=True)

    forge_model = MNISTLinear(dtype=dtype, bias=True)

    copy_params(torch_model, forge_model)

    if freeze_layer is not None:
        forge_model.linear_relu_stack[freeze_layer].weight.requires_grad = False
        forge_model.linear_relu_stack[freeze_layer].bias.requires_grad = False
        torch_model.linear_relu_stack[freeze_layer].weight.requires_grad = False
        torch_model.linear_relu_stack[freeze_layer].bias.requires_grad = False

    loss_fn = nn.CrossEntropyLoss()

    sample_inputs = [torch.ones(batch_size, in_features, dtype=dtype)]

    tt_model = forge.compile(forge_model, sample_inputs=sample_inputs, training=True)

    X = torch.ones(batch_size, in_features, dtype=dtype)
    y = torch.zeros(batch_size, out_features, dtype=dtype)

    torch_pred = torch_model(X)
    torch_loss = loss_fn(torch_pred, y)
    torch_loss.backward()
    torch_grads = get_param_grads(torch_model.named_parameters)

    X = torch.ones(batch_size, in_features, dtype=dtype)
    y = torch.zeros(batch_size, out_features, dtype=dtype)

    forge_pred = tt_model(X)[0]
    forge_loss = loss_fn(forge_pred, y)
    forge_loss.backward()
    tt_model.backward()
    forge_grads = get_param_grads(forge_model.named_parameters)

    if freeze_layer is not None:
        assert forge_model.linear_relu_stack[freeze_layer].weight.grad is None
        assert forge_model.linear_relu_stack[freeze_layer].bias.grad is None

    # Compare gradients for each parameter
    for name in reversed(list(torch_grads.keys())):
        assert compare_with_golden(torch_grads[name], forge_grads[name])


# For bfloat16, the following line should be added to the test_forge_vs_torch function:
# In file forge/forge/op/eval/forge/eltwise_unary.py:418 should be replaced with: threshold_tensor = ac.tensor(torch.zeros(shape, dtype=torch.bfloat16) + threshold)
# That sets relu threshold to bfloat16 tensor.
# And in file forge/forge/compile.py::compile_main forced bfloat 16 should be added compiler_cfg.default_df_override = DataFormat.Float16_b
@pytest.mark.skip(reason="Need to be tested with bfloat16 and takes around 10 minutes to run")
@pytest.mark.push
def test_forge_vs_torch():
    batch_size = 64
    learning_rate = 1e-2
    epochs = 10
    verbose = True

    dtype = torch.float32

    torch_model = MNISTLinear(dtype=dtype)
    forge_model = MNISTLinear(dtype=dtype)

    copy_params(torch_model, forge_model)

    torch_writer = load_tb_writer("torch")
    forge_writer = load_tb_writer("forge")

    loss_fn = nn.CrossEntropyLoss()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
    forge_optimizer = torch.optim.SGD(forge_model.parameters(), lr=learning_rate)

    tt_model = forge.compile(
        forge_model,
        sample_inputs=[torch.ones(batch_size, 784, dtype=dtype)],
        optimizer=forge_optimizer,
        training=True,
    )

    test_loader, train_loader = load_dataset(batch_size, dtype=dtype)
    step = 0

    early_stop = EarlyStopping(patience=1, mode="max")

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")
    for i in range(epochs):
        start_time = time.time()
        torch_loop = train_loop(
            train_loader,
            torch_model,
            loss_fn,
            torch_optimizer,
            batch_size,
            torch_model.named_parameters,
            is_tt=False,
            verbose=verbose,
        )
        forge_loop = train_loop(
            train_loader,
            tt_model,
            loss_fn,
            forge_optimizer,
            batch_size,
            forge_model.named_parameters,
            is_tt=True,
            verbose=verbose,
        )
        for torch_data, forge_data in zip(torch_loop, forge_loop):
            step += 1

            torch_loss, torch_pred, torch_grads = torch_data
            forge_loss, forge_pred, forge_grads = forge_data

            if step % 100 == 0:
                torch_val_loss, torch_val_acc = validation_loop(
                    test_loader, torch_model, loss_fn, batch_size, is_tt=False
                )
                forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, is_tt=True)

                torch_writer.add_scalar("train_loss", torch_loss.float(), step)
                forge_writer.add_scalar("train_loss", forge_loss.float(), step)
                torch_writer.add_scalar("validation_acc", torch_val_acc, step)
                forge_writer.add_scalar("validation_acc", forge_val_acc, step)

                torch_writer.flush()
                forge_writer.flush()

        if verbose:
            print(f"Epoch {i} took {time.time() - start_time} seconds")

        forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, is_tt=True)
        early_stop.step(forge_val_acc, i)

        if early_stop.is_best():
            torch.save(torch_model.state_dict(), f"runs/models/torch_model_{i}.pth")
            torch.save(forge_model.state_dict(), f"runs/models/forge_model_{i}.pth")

        if early_stop.is_early_stop():
            break

    # Load best model
    torch_model.load_state_dict(torch.load(f"runs/models/torch_model_{early_stop.get_best_model()}.pth"))
    forge_model.load_state_dict(torch.load(f"runs/models/forge_model_{early_stop.get_best_model()}.pth"))

    torch_val_loss, torch_val_acc = validation_loop(test_loader, torch_model, loss_fn, batch_size, is_tt=False)
    forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, is_tt=True)

    print(f"Validation accuracy for Torch: {torch_val_acc} in epoch {early_stop.get_best_model()}")
    print(f"Validation accuracy for Forge: {forge_val_acc} in epoch {early_stop.get_best_model()}")


@pytest.mark.push
def test_loss_device():
    # Config
    num_epochs = 3
    batch_size = 1
    learning_rate = 0.01

    # Limit number of batches to run - quicker test
    limit_num_batches = 10

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Load TensorBoard writer (for logging)
    writer = load_tb_writer("forge_mnist")

    framework_model = MNISTLinear(bias=False)
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)

    tt_model = forge.compile(
        framework_model,
        sample_inputs=[torch.rand(batch_size, 784)],
        training=True,
    )

    loss_fn = CrossEntropyLoss(name="cross_entropy_loss")

    loss_inputs = [torch.rand(batch_size, 10).requires_grad_(True), torch.rand(batch_size, 10)]
    loss_inputs = to_forge_tensors(loss_inputs)

    tt_loss = forge.compile(
        loss_fn,
        sample_inputs=loss_inputs,
        attach_to=tt_model,
        training=True,
    )

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")
    for epoch_idx in range(num_epochs):
        # Reset gradients (every epoch) - since our batch size is currently 1,
        # we accumulate gradients across multiple batches (limit_num_batches),
        # and then run the optimizer.
        framework_optimizer.zero_grad()

        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            golden_pred, pred = verify(
                inputs=[data],
                framework_model=framework_model,
                compiled_model=tt_model,
                verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
            )
            pred = pred[0]

            loss = tt_loss(pred, target)
            total_loss += loss[0].item()

            # Run backward pass on device
            tt_loss.backward()

            if batch_idx >= limit_num_batches:
                break

        print(f"epoch: {epoch_idx} loss: {total_loss}")

        # Adjust weights (on CPU)
        framework_optimizer.step()

    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        golden_pred, pred = verify(
            inputs=[data],
            framework_model=framework_model,
            compiled_model=tt_model,
            verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        )
        pred = pred[0]

        target = nn.functional.one_hot(target, num_classes=10).float()

        test_loss += tt_loss(pred, target)[0]

        if batch_idx >= limit_num_batches:
            break

    print(f"Test (total) loss: {test_loss}")


@pytest.mark.push
def test_lora():
    # Config
    num_epochs = 3
    batch_size = 128
    learning_rate = 0.1

    # Limit number of batches to run - quicker test
    limit_num_batches = 10

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    framework_model = MNISTLora(bias=False)

    tt_optimizer = forge.optimizers.SGD(learning_rate=learning_rate)
    tt_model = forge.compile(
        framework_model,
        sample_inputs=[torch.rand(batch_size, 784)],
        optimizer=tt_optimizer,
        training=True,
    )

    loss_fn = CrossEntropyLoss(name="cross_entropy_loss")

    loss_inputs = [torch.rand(batch_size, 10).requires_grad_(True), torch.rand(batch_size, 10)]
    loss_inputs = to_forge_tensors(loss_inputs)
    tt_loss = forge.compile(
        loss_fn,
        sample_inputs=loss_inputs,
        attach_to=tt_model,
        training=True,
    )

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")
    prev_total_loss = 1e10
    for epoch_idx in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            golden_pred, pred = verify(
                inputs=[data],
                framework_model=framework_model,
                compiled_model=tt_model,
                verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
            )
            pred = pred[0]

            loss = tt_loss(pred, target)[0]

            total_loss += loss.item()

            # Run backward pass on device
            tt_loss.backward()

            # Adjust weights on the device.
            # NOTE: after executing the step, this will also zero the gradients.
            tt_optimizer.step()

            # Update the pytorch model weights.
            tt_model.update_host_weights()

            if batch_idx >= limit_num_batches:
                break

        print(f"epoch: {epoch_idx} loss: {total_loss}")
        assert prev_total_loss - total_loss > 1e-5, "Loss should go down"
        prev_total_loss = total_loss

    test_loss = 0
    for _, (data, target) in enumerate(test_loader):
        pred = tt_model(data)[0]
        target = nn.functional.one_hot(target, num_classes=10).float()

        test_loss += tt_loss(pred, target)[0]

    print(f"Test (total) loss: {test_loss}")


@pytest.mark.push
def test_optimizer_device():
    # Config
    num_epochs = 32
    batch_size = 1024
    learning_rate = 0.1

    # Limit number of batches to run - quicker test
    limit_num_batches = 1

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    framework_model = MNISTLinear(bias=False)
    framework_loss = torch.nn.CrossEntropyLoss()
    optimizer = forge.optimizers.SGD(learning_rate=learning_rate)

    tt_model = forge.compile(
        framework_model,
        sample_inputs=[torch.rand(batch_size, 784)],
        optimizer=optimizer,
        training=True,
    )

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")

    for epoch_idx in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            golden_pred, pred = verify(
                inputs=[data],
                framework_model=framework_model,
                compiled_model=tt_model,
                verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
            )
            pred = pred[0]

            # Execute loss (and its backward) on CPU.
            loss = framework_loss(pred, target)
            total_loss += loss.item()

            loss.backward()
            tt_model.backward()

            # Adjust weights on the device.
            # NOTE: after executing the step, this will also zero the gradients.
            optimizer.step()
            tt_model.update_host_weights()

            if batch_idx >= limit_num_batches:
                break

        print(f"epoch: {epoch_idx} loss: {total_loss}")

    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = tt_model(data)[0]
        target = nn.functional.one_hot(target, num_classes=10).float()

        test_loss += framework_loss(pred, target)
        break

    print(f"Test (total) loss: {test_loss}")


@pytest.mark.push
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_e2e_device(dtype):
    # Config
    num_epochs = 5
    batch_size = 1024
    learning_rate = 0.1

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size, dtype=dtype)

    framework_model = MNISTLinear(bias=False, dtype=dtype)
    framework_loss = torch.nn.CrossEntropyLoss()
    tt_optimizer = forge.optimizers.SGD(learning_rate=learning_rate)

    # Configure compiler only if using bfloat16
    compiler_cfg = CompilerConfig()
    if dtype == torch.bfloat16:
        data_format_override = DataFormat.Float16_b
        compiler_cfg.default_df_override = data_format_override
        compiler_cfg.mlir_config = forge.config.MLIRConfig().set_custom_config("enable-cpu-hoisted-const-eval=false")

    verify_cfg = DeprecatedVerifyConfig()
    verify_cfg.stages_for_intermediate_verification = {CompileDepth.AUTOGRAD}
    verify_cfg.enable_op_level_comparision = True
    tt_model = forge.compile(
        framework_model,
        sample_inputs=[torch.rand(batch_size, 784, dtype=dtype)],
        optimizer=tt_optimizer,
        training=True,
        verify_cfg=verify_cfg,
        compiler_cfg=compiler_cfg,
    )

    loss_inputs = [
        torch.rand(batch_size, 10, dtype=dtype).requires_grad_(True),
        torch.rand(batch_size, 10, dtype=dtype),
    ]
    loss_inputs = to_forge_tensors(loss_inputs)
    tt_loss = forge.compile(
        CrossEntropyLoss(name="cross_entropy_loss"),
        sample_inputs=loss_inputs,
        training=True,
        attach_to=tt_model,
    )

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")

    prev_total_loss = 1e10
    for epoch_idx in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            # Create target tensor and leave on CPU.
            target = nn.functional.one_hot(target, num_classes=10).to(dtype)

            # Forward pass
            golden_pred, pred = verify(
                inputs=[data],
                framework_model=framework_model,
                compiled_model=tt_model,
                verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
            )
            pred = pred[0]

            # Execute loss (and its backward) on device.
            golden_loss, loss = verify(
                inputs=[pred, target],
                framework_model=framework_loss,
                compiled_model=tt_loss,
                verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(rtol=1e-1), verify_shape=False),
            )
            total_loss += loss[0].item()

            tt_loss.backward()

            # Adjust weights on the device.
            # NOTE: after executing the step, this will also zero the gradients.
            tt_optimizer.step()

            tt_model.update_host_weights()

        print(f"epoch: {epoch_idx} loss: {total_loss}")

        assert prev_total_loss - total_loss > 1e-5, "Loss should go down"
        prev_total_loss = total_loss

    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = tt_model(data)[0]
        target = nn.functional.one_hot(target, num_classes=10).to(dtype)

        test_loss += framework_loss(pred, target)
        break

    print(f"Test (total) loss: {test_loss}")
