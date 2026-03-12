# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Stable Diffusion Demo Script


from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)

import forge
from forge.forge_property_utils import Framework, ModelArch, record_model_properties


def load_pipe(variant, variant_type):
    if variant_type == "v35":
        pipe = StableDiffusion3Pipeline.from_pretrained(f"stabilityai/{variant}", torch_dtype=torch.float32)
        modules = [pipe.text_encoder, pipe.transformer, pipe.text_encoder_2, pipe.text_encoder_3, pipe.vae]
    elif variant_type == "xl":
        pipe = DiffusionPipeline.from_pretrained(f"stabilityai/{variant}", torch_dtype=torch.float32)
        modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]

    # Move the pipeline to CPU
    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def stable_diffusion_preprocessing(
    pipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    callback_steps: int = 1,
):

    # 0. Default height and width to unet
    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    print("pipeline.unet.config.sample_size", pipeline.unet.config.sample_size)
    print("pipeline.vae_scale_factor", pipeline.vae_scale_factor)
    print("width", width)

    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = pipeline._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipeline.unet.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    return (
        latents,
        timesteps,
        extra_step_kwargs,
        prompt_embeds,
        extra_step_kwargs,
    )


def stable_diffusion_preprocessing_v35(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.0,
    num_inference_steps=1,
    num_images_per_prompt=1,
    clip_skip=None,
    max_sequence_length=256,
    joint_attention_kwargs=None,
    skip_guidance_layers=None,
    skip_layer_guidance_scale=2.8,
    skip_layer_guidance_start=0.01,
    skip_layer_guidance_stop=0.2,
    do_classifier_free_guidance=True,
    mu=None,
):
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor

    pipe.check_inputs(
        prompt,
        None,  # prompt_2
        None,  # prompt_3
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=max_sequence_length,
    )

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        device=device,
        clip_skip=clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )

    if do_classifier_free_guidance:
        original_prompt_embeds = prompt_embeds
        original_pooled_prompt_embeds = pooled_prompt_embeds

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    num_channels_latents = pipe.transformer.config.in_channels
    shape = (
        num_images_per_prompt,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)

    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        image_seq_len = (height // pipe.transformer.config.patch_size) * (width // pipe.transformer.config.patch_size)
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=1,
        device=device,
        sigmas=None,
        **scheduler_kwargs,
    )

    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds


def stable_diffusion_preprocessing_xl(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=5.0,
    num_inference_steps=50,
    timesteps=None,
    sigmas=None,
    eta=0.0,
    num_images_per_prompt=1,
    height=None,
    width=None,
    clip_skip=None,
    original_size=None,
    target_size=None,
    cross_attention_kwargs=None,
    guidance_rescale=0.0,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_target_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    **kwargs,
):
    # Set default height and width
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # Check inputs
    pipe.check_inputs(
        prompt,
        None,  # prompt_2 (if applicable)
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_steps=None,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    # 1. Encode the prompt
    do_classifier_free_guidance = True
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,  # Assume classifier-free guidance
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
    )
    # 2. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        timesteps=timesteps,
        sigmas=sigmas,
    )

    # 3. Prepare latent variables
    if isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    num_channels_latents = pipe.unet.config.in_channels
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    torch.manual_seed(42)
    latents = torch.randn(
        (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipe._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
    ip_adapter_image = None
    ip_adapter_image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            do_classifier_free_guidance,
        )
    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        added_cond_kwargs["image_embeds"] = image_embeds

    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps[0])

    return latent_model_input, timesteps, prompt_embeds, timestep_cond, added_cond_kwargs, add_time_ids


def denoising_loop(
    pipeline,
    latents,
    timesteps,
    prompt_embeds,
    extra_step_kwargs,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    callback=None,
    callback_steps: int = 1,
):

    do_classifier_free_guidance = guidance_scale > 1.0
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
                timestep_ = torch.cat([t.unsqueeze(0)] * 2).float()
            else:
                latent_model_input = latents
                timestep_ = t

            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # sanity
            # noise_pred_0 = pipeline(latent_model_input.detach()[0:1],timestep_.detach()[0:1],prompt_embeds.detach()[0:1],)

            inputs = [latent_model_input.detach()[0:1], timestep_.detach()[0:1], prompt_embeds.detach()[0:1]]
            module_name = record_model_properties(
                framework=Framework.PYTORCH, model=ModelArch.STABLEDIFFUSION, suffix=f"1_{i}"
            )
            compiled_model = forge.compile(pipeline, sample_inputs=inputs, module_name=module_name)
            noise_pred_0 = compiled_model(*inputs)

            # sanity
            # noise_pred_1 = pipeline(latent_model_input.detach()[1:2],timestep_.detach()[1:2],prompt_embeds.detach()[1:2],)
            inputs = [latent_model_input.detach()[1:2], timestep_.detach()[1:2], prompt_embeds.detach()[1:2]]
            module_name = record_model_properties(
                framework=Framework.PYTORCH, model=ModelArch.STABLEDIFFUSION, suffix=f"2_{i}"
            )
            compiled_model = forge.compile(pipeline, sample_inputs=inputs, module_name=module_name)
            noise_pred_1 = compiled_model(*inputs)

            noise_pred = torch.cat([noise_pred_0[0].value().detach(), noise_pred_1[0].value().detach()], dim=0)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    return latents


def stable_diffusion_postprocessing(
    pipeline,
    latents,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
):

    has_nsfw_concept = None
    if output_type == "latent":
        image = latents
        has_nsfw_concept = None
    elif output_type == "pil":
        # 8. Post-processing
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        image = pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()

        # 9. Convert to PIL
        image = pipeline.numpy_to_pil(image)
    else:
        # 8. Post-processing
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        image = pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()

    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
