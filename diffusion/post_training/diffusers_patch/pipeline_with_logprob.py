# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps as retrieve_flux_timesteps
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from .solver import run_sampling


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def _unwrap_compiled(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# ---------------------------------------------------------------------------
# SD3 pipeline
# ---------------------------------------------------------------------------


@torch.no_grad()
def pipeline_with_logprob_sd3(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
    deterministic: bool = False,
    solver: str = "flow",
    sequential_decode: bool = False,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    if latents is None:
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    else:
        latents = latents.to(device)

    # 5. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=None,
    )
    self._num_timesteps = len(timesteps)

    sigmas = self.scheduler.sigmas.float()

    def v_pred_fn(z, sigma):
        latent_model_input = torch.cat([z] * 2) if self.do_classifier_free_guidance else z
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = torch.full([latent_model_input.shape[0]], sigma * 1000, device=z.device, dtype=torch.long)
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.to(prompt_embeds.dtype)
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []

    # 7. Denoising loop
    latents, all_latents, all_log_probs = run_sampling(v_pred_fn, latents, sigmas, solver, deterministic, noise_level)

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    if sequential_decode and latents.shape[0] > 1:
        decoded_batches = []
        for idx in range(latents.shape[0]):
            decoded_batches.append(self.vae.decode(latents[idx : idx + 1], return_dict=False)[0])
        image = torch.cat(decoded_batches, dim=0)
    else:
        image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    return image, all_latents, all_log_probs


# ---------------------------------------------------------------------------
# FLUX.1 pipeline
# ---------------------------------------------------------------------------


@torch.no_grad()
def pipeline_with_logprob_flux(
    pipeline,
    prompt=None,
    prompt_2=None,
    height=None,
    width=None,
    num_inference_steps=28,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    pooled_prompt_embeds=None,
    text_ids=None,
    output_type="pt",
    joint_attention_kwargs=None,
    max_sequence_length=512,
    noise_level=0.7,
    deterministic=False,
    solver="flow",
    sequential_decode=False,
):
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    pipeline.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device
    lora_scale = joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None

    if prompt_embeds is None or pooled_prompt_embeds is None or text_ids is None:
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    num_channels_latents = pipeline.transformer.config.in_channels // 4
    if latents is None:
        latents, latent_image_ids = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    else:
        latents = latents.to(device)
        latent_image_ids = pipeline._prepare_latent_image_ids(
            batch_size * num_images_per_prompt,
            height // pipeline.vae_scale_factor,
            width // pipeline.vae_scale_factor,
            device,
            prompt_embeds.dtype,
        )

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if hasattr(pipeline.scheduler.config, "use_flow_sigmas") and pipeline.scheduler.config.use_flow_sigmas:
        sigmas = None

    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    _, num_inference_steps = retrieve_flux_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    sigmas = pipeline.scheduler.sigmas.float()

    active_transformer = pipeline.transformer
    guidance_config = _unwrap_compiled(active_transformer).config
    if guidance_config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
    else:
        guidance = None

    def v_pred_fn(z, sigma):
        timestep = torch.full([z.shape[0]], float(sigma), device=z.device, dtype=z.dtype)
        noise_pred = active_transformer(
            hidden_states=z,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        return noise_pred

    all_latents = [latents]
    latents, all_latents, all_log_probs = run_sampling(v_pred_fn, latents, sigmas, solver, deterministic, noise_level)

    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    latents = latents.to(dtype=pipeline.vae.dtype)

    if sequential_decode and latents.shape[0] > 1:
        decoded_batches = []
        for idx in range(latents.shape[0]):
            decoded_batches.append(pipeline.vae.decode(latents[idx : idx + 1], return_dict=False)[0])
        image = torch.cat(decoded_batches, dim=0)
    else:
        image = pipeline.vae.decode(latents, return_dict=False)[0]

    image = pipeline.image_processor.postprocess(image, output_type=output_type)
    pipeline.maybe_free_model_hooks()

    return image, all_latents, latent_image_ids, text_ids, all_log_probs


# ---------------------------------------------------------------------------
# Sana pipeline
# ---------------------------------------------------------------------------


@torch.no_grad()
def pipeline_with_logprob_sana(
    transformer,
    vae,
    *,
    latents=None,
    num_channels=None,
    latent_size=None,
    prompt_embeds=None,
    prompt_attention_mask=None,
    negative_prompt_embeds=None,
    negative_prompt_attention_mask=None,
    num_inference_steps=20,
    guidance_scale=4.5,
    noise_level=0.7,
    deterministic=False,
    sequential_decode=False,
    solver="flow",
):
    assert prompt_embeds is not None

    if latents is None:
        assert num_channels is not None and latent_size is not None
        latents = torch.randn(
            prompt_embeds.shape[0],
            num_channels,
            latent_size,
            latent_size,
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )

    device = latents.device
    dtype = latents.dtype
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype)

    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    caption_4d = prompt_embeds.unsqueeze(1) if prompt_embeds.dim() == 3 else prompt_embeds
    mask_4d = (
        prompt_attention_mask.unsqueeze(1).unsqueeze(1).to(torch.int16)
        if prompt_attention_mask is not None and prompt_attention_mask.dim() == 2
        else prompt_attention_mask
    )

    if do_cfg:
        neg_4d = negative_prompt_embeds.unsqueeze(1) if negative_prompt_embeds.dim() == 3 else negative_prompt_embeds
        neg_mask_4d = (
            negative_prompt_attention_mask.unsqueeze(1).unsqueeze(1).to(torch.int16)
            if negative_prompt_attention_mask is not None and negative_prompt_attention_mask.dim() == 2
            else negative_prompt_attention_mask
        )
        y_in = torch.cat([neg_4d, caption_4d], dim=0)
        m_in = torch.cat([neg_mask_4d, mask_4d], dim=0) if mask_4d is not None else None
    else:
        y_in = caption_4d
        m_in = mask_4d

    def v_pred_fn(z, sigma):
        z_in = torch.cat([z, z], dim=0) if do_cfg else z
        t_batch = sigma.expand(z_in.shape[0]).to(device)
        pred = transformer(z_in, t_batch, y_in, mask=m_in)
        if do_cfg:
            u, c = pred.chunk(2)
            pred = u + guidance_scale * (c - u)
        return pred

    latents, all_latents, _ = run_sampling(
        v_pred_fn,
        latents,
        sigmas,
        solver,
        deterministic,
        noise_level,
    )

    vae_dtype = next(vae.parameters()).dtype
    latents_dec = latents.to(vae_dtype) / vae.config.scaling_factor
    if sequential_decode and latents_dec.shape[0] > 1:
        decoded = []
        for idx in range(latents_dec.shape[0]):
            decoded.append(vae.decode(latents_dec[idx : idx + 1], return_dict=False)[0])
        image = torch.cat(decoded, dim=0)
    else:
        image = vae.decode(latents_dec, return_dict=False)[0]
    images = (image / 2 + 0.5).clamp(0, 1)

    return images, all_latents, sigmas[:-1]
