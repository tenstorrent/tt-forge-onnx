# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Compile-for-a-named-architecture, with no device attached.

By default forge reads the system descriptor off the device that is physically
present (lower_to_mlir.cpp), so the compile target is whatever hardware you happen
to be sitting in front of, and compiling with no hardware is impossible.
MLIRConfig.set_target_arch / set_system_desc_path override that: the descriptor
comes from the MLIR pipeline instead, so a Wormhole machine can compile for
Blackhole or Quasar, and CI can compile with no accelerator at all.

These tests deliberately do NOT run the compiled model -- executing still needs a
matching device. They assert only that compilation reaches a flatbuffer for the
requested arch without touching hardware, which is what makes them safe to run on
a plain CI runner.
"""

import pytest
import torch
from torch import nn

import forge
from forge.config import CompilerConfig, MLIRConfig

TTSystem = forge._C.runtime.experimental.TTSystem

# Every architecture the ttir-to-ttnn pipeline accepts a mock descriptor for.
ALL_ARCHS = [forge._C.Arch.WORMHOLE_B0, forge._C.Arch.BLACKHOLE, forge._C.Arch.QUASAR]


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


def _compile_for(arch, module_name):
    cfg = CompilerConfig(mlir_config=MLIRConfig().set_target_arch(arch))
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    return forge.compile(Add(), sample_inputs=inputs, module_name=module_name, compiler_cfg=cfg)


@pytest.mark.push
@pytest.mark.parametrize("arch", ALL_ARCHS, ids=lambda a: str(a).split(".")[-1].lower())
def test_compile_for_target_arch(arch):
    """Each supported arch compiles, whatever hardware is (or is not) present."""
    assert _compile_for(arch, f"target_arch_{str(arch).split('.')[-1].lower()}") is not None


@pytest.mark.push
def test_compile_does_not_open_a_device():
    """The point of the flag: naming a target must not initialise hardware.

    Guarded on TTSystem not already being initialised, because it is a process-wide
    singleton -- another test in the same session may have opened a device, and then
    this assertion would say nothing. Skipping is honest; passing would not be.
    """
    if TTSystem.is_initialized():
        pytest.skip("a device was already opened in this process; the check would be vacuous")

    _compile_for(forge._C.Arch.QUASAR, "target_arch_no_device")

    assert not TTSystem.is_initialized(), (
        "compiling for a named target arch must not open a device -- "
        "lower_to_mlir should have skipped reading the live system descriptor"
    )


@pytest.mark.push
def test_target_arch_rejects_unsupported():
    """Only the three archs with a mock descriptor are accepted."""
    for arch in (forge._C.Arch.WORMHOLE, forge._C.Arch.JAWBRIDGE, forge._C.Arch.Invalid):
        with pytest.raises(Exception, match="target_arch only accepts"):
            MLIRConfig().set_target_arch(arch)


@pytest.mark.push
def test_quasar_rejects_block_float_weights():
    """Quasar's format set has no bf8_b/bf4_b; catch it at config time.

    Otherwise it surfaces much later as a tt-metal host format-validator throw.
    """
    cfg = CompilerConfig(
        mlir_config=MLIRConfig()
        .set_target_arch(forge._C.Arch.QUASAR)
        .set_experimental_weight_dtype(forge._C.DataFormat.Bfp8_b)
    )
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    with pytest.raises(Exception, match="not supported on Quasar"):
        forge.compile(Add(), sample_inputs=inputs, module_name="quasar_bfp8", compiler_cfg=cfg)


@pytest.mark.push
def test_system_desc_path_requires_non_empty():
    with pytest.raises(Exception, match="must not be empty"):
        MLIRConfig().set_system_desc_path("")
