# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge.verify.verify import verify
import onnx
from test.models.onnx.multimodal.stable_diffusion.model_utils.model import (
    load_pipe,
    stable_diffusion_preprocessing_xl,
)
from forge.forge_property_utils import Framework, ModelArch, Source, Task, record_model_properties


class StableDiffusionXLWrapper(torch.nn.Module):
    def __init__(self, model, added_cond_kwargs, cross_attention_kwargs=None):
        super().__init__()
        self.model = model
        self.cross_attention_kwargs = cross_attention_kwargs
        self.added_cond_kwargs = added_cond_kwargs

    def forward(self, latent_model_input, timestep, prompt_embeds):
        noise_pred = self.model(
            latent_model_input,
            timestep[0],
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=self.added_cond_kwargs,
        )[0]
        return noise_pred


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["stable-diffusion-xl-base-1.0"])
def test_stable_diffusion_generation(variant, forge_tmp_path):
    # Build Module Name
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.STABLEDIFFUSION,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load the pipeline
    pipe = load_pipe(variant, variant_type="xl")

    # Extract only the unet, as the forward pass occurs here.
    framework_model = pipe.unet

    # Tokenize the prompt to a tensor
    tokenizer = pipe.tokenizer
    prompt = "An astronaut riding a green horse"
    (
        latent_model_input,
        timestep,
        prompt_embeds,
        timestep_cond,
        added_cond_kwargs,
        add_time_ids,
    ) = stable_diffusion_preprocessing_xl(pipe, prompt)
    inputs = [latent_model_input, timestep, prompt_embeds]

    # Wrap the pipeline in the wrapper
    framework_model = StableDiffusionXLWrapper(framework_model, added_cond_kwargs, cross_attention_kwargs=None)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/stable_diffusion_xl.onnx"
    torch.onnx.export(framework_model, (latent_model_input, timestep, prompt_embeds), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
