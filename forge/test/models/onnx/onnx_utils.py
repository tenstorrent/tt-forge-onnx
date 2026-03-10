# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
import onnx
from onnx import numpy_helper
from loguru import logger

import forge
from forge.config import CompileDepth
from forge.module import OnnxModule
from forge.parameter import Parameter


# ---------------------------------------------------------------------------
# OnnxModule parameter patch
# ---------------------------------------------------------------------------


def _get_parameters_as_bfloat16(self) -> List[Parameter]:
    """Read ONNX graph initializers and return them as Forge ``Parameter`` objects.

    Every floating-point initializer is cast to ``torch.bfloat16`` so that the
    compiled model runs in bfloat16.  Integer initializers (e.g. attention masks,
    positional indices) are kept at their original dtype.

    This function is intended to be monkey-patched onto ``OnnxModule`` via
    ``patch_onnx_module_bfloat16()``.  It replaces the default implementation
    that keeps parameters at their original ONNX dtype (usually float32).

    Parameters
    ----------
    self:
        The ``OnnxModule`` instance whose graph initializers are read.

    Returns
    -------
    List[Parameter]
        One ``forge.parameter.Parameter`` per ONNX initializer, in graph order.
    """
    params = []
    for initializer in self.module.graph.initializer:
        param_data = numpy_helper.to_array(initializer)
        torch_param = torch.tensor(param_data)
        if torch.is_floating_point(torch_param):
            logger.info(f"Casting ONNX initializer '{initializer.name}' float → bfloat16")
            torch_param = torch_param.to(torch.bfloat16)
        params.append(Parameter(torch_param, requires_grad=False, name=initializer.name))
    return params


def patch_onnx_module_bfloat16() -> None:
    """Monkey-patch ``OnnxModule`` so parameters are loaded as bfloat16.

    Call this **once** at the top of a test module (outside any test function)
    before importing or instantiating any ONNX model.  The patch is process-wide
    and persists for the lifetime of the Python interpreter.

    Why this is needed
    ------------------
    Forge reads ONNX model weights via ``OnnxModule.get_parameters``.  The
    default implementation preserves the original ONNX dtype (float32 for most
    pre-trained models).  When we want to run in bfloat16 end-to-end, we need
    the parameters to already be bfloat16 at the point Forge processes them,
    which is what this patch achieves.
    """
    OnnxModule.get_parameters = _get_parameters_as_bfloat16
    logger.debug("OnnxModule.get_parameters patched: parameters will be loaded as bfloat16")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def convert_inputs_to_bfloat16(inputs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Cast all floating-point tensors in *inputs* to ``torch.bfloat16``.

    Parameters
    ----------
    inputs:
        Sample input tensors as returned by the model's data-loader utility.

    Returns
    -------
    List[torch.Tensor]
        New list of tensors with floating-point entries in bfloat16.

    Examples
    --------
    >>> bf16 = convert_inputs_to_bfloat16([torch.randn(1, 3, 224, 224)])
    >>> bf16[0].dtype
    torch.bfloat16
    """
    bfloat16_inputs = []
    for input in inputs:
        if torch.is_floating_point(input):
            bfloat16_input = input.to(torch.bfloat16)
            bfloat16_inputs.append(bfloat16_input)
        else:
            bfloat16_inputs.append(input)
    return bfloat16_inputs


# ---------------------------------------------------------------------------
# Phase 1 – generate the Forge Python module with float32 inputs
# ---------------------------------------------------------------------------


def compile_onnx_initial_graph(
    onnx_model: onnx.ModelProto,
    float32_inputs: List[torch.Tensor],
    module_name: str,
) -> None:
    """Compile an ONNX model with float32 inputs up to the initial-graph stage.

    Forge's ``GENERATE_INITIAL_GRAPH`` stage converts the ONNX graph into a
    Python Forge module and writes it to disk.

    The resulting module file on disk is then reused by
    ``compile_and_run_onnx_bfloat16`` (Phase 2) via the
    ``FORGE_RELOAD_GENERATED_MODULES`` environment variable.

    Parameters
    ----------
    onnx_model:
        A validated ``onnx.ModelProto`` (typically the output of
        ``onnx.load`` + ``onnx.checker.check_model``).
    float32_inputs:
        The **original** float32 sample inputs as returned by the data loader,
        *before* any bfloat16 conversion.  These are used solely for graph
        tracing; no inference is performed.
    module_name:
        Unique name for this compilation artefact.  Must match the name passed
        to ``compile_and_run_onnx_bfloat16`` so that Phase 2 can locate the
        generated file.

    """
    compiler_cfg = forge.config.CompilerConfig(
        compile_depth=CompileDepth.GENERATE_INITIAL_GRAPH,
    )
    forge.compile(
        onnx_model,
        sample_inputs=float32_inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )
    logger.info(f"[Phase 1] '{module_name}': Forge module generated (GENERATE_INITIAL_GRAPH) " "using float32 inputs.")


# ---------------------------------------------------------------------------
# Phase 2 – full compile + inference with bfloat16 inputs
# ---------------------------------------------------------------------------


def compile_and_run_onnx_bfloat16(
    onnx_model: onnx.ModelProto,
    bfloat16_inputs: List[torch.Tensor],
    module_name: str,
) -> List[torch.Tensor]:
    """Fully compile an ONNX model in bfloat16 and run a forward pass.

    This function is the second phase of the two-phase compile workflow.  The forge
    compiler reads the Python module that was written to disk by
    ``compile_onnx_initial_graph`` (Phase 1) instead of regenerating it.
    Skipping regeneration avoids the float32 / bfloat16 dtype conflict that
    would arise if Forge tried to re-trace the model with bfloat16 inputs

    Once compiled, the model is called with ``bfloat16_inputs[0]`` and the
    raw output list is returned.

    Parameters
    ----------
    onnx_model:
        A validated ``onnx.ModelProto``.  Should be the same object that was
        passed to ``compile_onnx_initial_graph``.
    bfloat16_inputs:
        Sample inputs in bfloat16 (e.g. the output of
        ``convert_inputs_to_bfloat16``).  The first element is used for the
        actual inference call.
    module_name:
        Must match the name used in ``compile_onnx_initial_graph`` so Forge
        can locate the pre-generated module on disk.

    Returns
    -------
    List[torch.Tensor]
        Raw output tensors from the compiled model's forward pass.

    """

    compiler_cfg = forge.config.CompilerConfig(
        default_df_override=forge.DataFormat.Float16_b,
    )
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=bfloat16_inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )
    logger.info(f"[Phase 2] '{module_name}': full compile complete; running inference with bfloat16 inputs.")
    return compiled_model(bfloat16_inputs[0])
