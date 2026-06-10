# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from ..guiders.adaptive_projected_guidance import AdaptiveProjectedGuidance


class FlowEuler:
    def __init__(self, model_fn, condition, uncondition, cfg_scale, flow_shift=3.0, model_kwargs=None, apg=None):
        self.model = model_fn
        self.condition = condition
        self.uncondition = uncondition
        self.cfg_scale = cfg_scale
        self.model_kwargs = model_kwargs
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        self.apg = apg

    def sample(self, latents, steps=28):
        device = self.condition.device
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, steps, device, None)
        do_classifier_free_guidance = self.cfg_scale > 1

        prompt_embeds = self.condition
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([self.uncondition, self.condition], dim=0)

        for i, t in tqdm(list(enumerate(timesteps)), disable=os.getenv("DPM_TQDM", "False") == "True"):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.model(
                latent_model_input,
                timestep,
                prompt_embeds,
                **self.model_kwargs,
            )

            if isinstance(noise_pred, Transformer2DModelOutput):
                noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                if self.apg is None:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    x0_pred = latent_model_input - timestep * noise_pred
                    x0_pred_uncond, x0_pred_text = x0_pred.chunk(2)
                    x0_pred = self.apg(x0_pred_text, x0_pred_uncond)[0]
                    noise_pred = (latents - x0_pred) / timestep

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        return latents


class LTXFlowEuler(FlowEuler):
    def __init__(self, model_fn, condition, uncondition, cfg_scale, flow_shift=3.0, model_kwargs=None):
        super().__init__(model_fn, condition, uncondition, cfg_scale, flow_shift, model_kwargs)

    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents. This helps with motion continuity, especially
        when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = conditioning_mask > (1.0 - eps)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents

    def sample(self, latents, steps=28, generator=None):
        """
        latents: 1,C,F,H,W
        steps: int

        latents is only one sample but the model kwargs are 2 samples
        """

        device = self.condition.device
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, steps, device, None)
        do_classifier_free_guidance = self.cfg_scale > 1

        condition_frame_info = self.model_kwargs["data_info"].pop(
            "condition_frame_info", {}
        )  # {frame_idx: frame_weight}
        condition_mask = torch.zeros_like(latents)  # 1,C,F,H,W
        image_cond_noise_scale = 0.0
        for frame_idx, frame_weight in condition_frame_info.items():
            condition_mask[:, :, frame_idx] = 1
            image_cond_noise_scale = max(image_cond_noise_scale, frame_weight)

        prompt_embeds = self.condition
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([self.uncondition, self.condition], dim=0)

        init_latents = latents.clone()  # here we need to clone to avoid modifying the original latents

        for i, t in tqdm(list(enumerate(timesteps)), disable=os.getenv("DPM_TQDM", "False") == "True"):
            if image_cond_noise_scale > 0:
                latents = self.add_noise_to_image_conditioning_latents(
                    t / 1000.0, init_latents, latents, image_cond_noise_scale, condition_mask, generator
                )

            condition_mask_input = torch.cat([condition_mask] * 2) if do_classifier_free_guidance else condition_mask
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(condition_mask_input.shape).float()
            timestep = torch.min(timestep, (1 - condition_mask_input) * 1000.0)

            noise_pred = self.model(
                latent_model_input,
                # timestep[:, 0, 0, 0, 0], # b
                timestep[:, :1, :, 0, 0],  # b,c,f,h,w -> b,1,f
                prompt_embeds,
                **self.model_kwargs,
            )  # b,c,f,h,w

            if isinstance(noise_pred, Transformer2DModelOutput):
                noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
                timestep = timestep.chunk(2)[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents_shape = latents.shape
            batch_size, num_latent_channels, num_frames, height, width = latents_shape

            # NOTE if we use per_token_timesteps, the noise_pred should be -noise_pred
            denoised_latents = self.scheduler.step(
                -noise_pred.reshape(batch_size, num_latent_channels, -1).transpose(1, 2),  # b,fhw,c -> b,c,fhw
                t,
                latents.reshape(batch_size, num_latent_channels, -1).transpose(1, 2),  # b,c,fhw -> b,fhw,c
                per_token_timesteps=timestep.reshape(batch_size, num_latent_channels, -1)[:, 0],  # b,c,fhw -> b,fhw
                return_dict=False,
            )[0]
            denoised_latents = denoised_latents.transpose(1, 2).reshape(latents_shape)
            tokens_to_denoise_mask = t / 1000 - 1e-6 < (1.0 - condition_mask)
            latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        return latents


class ChunkFlowEuler(LTXFlowEuler):
    """Euler sampler for non-cached chunk-causal teacher models."""

    @staticmethod
    def create_temporal_chunks(num_frames: int, chunk_index: list[int] | tuple[int, ...] | None) -> list[tuple[int, int]]:
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if not chunk_index:
            return [(0, num_frames)]

        starts = sorted({int(idx) for idx in chunk_index if 0 <= int(idx) < num_frames})
        if not starts or starts[0] != 0:
            starts = [0] + starts
        return [(start, end) for start, end in zip(starts, starts[1:] + [num_frames]) if end > start]

    @staticmethod
    def _slice_temporal_tensor(value: torch.Tensor, *, end_frame: int, total_frames: int) -> torch.Tensor:
        if value.ndim == 5 and value.shape[2] >= end_frame and value.shape[2] in {total_frames, end_frame}:
            return value[:, :, :end_frame]
        if value.ndim >= 3 and value.shape[1] >= end_frame and value.shape[1] in {total_frames, end_frame}:
            return value[:, :end_frame]
        return value

    def _slice_model_kwargs_for_active_prefix(self, *, active_end: int, total_frames: int) -> dict:
        sliced = dict(self.model_kwargs or {})

        data_info = dict(sliced.get("data_info", {}) or {})
        image_vae_embeds = data_info.get("image_vae_embeds")
        if isinstance(image_vae_embeds, torch.Tensor):
            data_info["image_vae_embeds"] = self._slice_temporal_tensor(
                image_vae_embeds,
                end_frame=active_end,
                total_frames=total_frames,
            )
        sliced["data_info"] = data_info

        for key in ("camera_conditions", "chunk_plucker", "delta_actions", "cam_pos_embeds"):
            value = sliced.get(key)
            if isinstance(value, torch.Tensor):
                sliced[key] = self._slice_temporal_tensor(value, end_frame=active_end, total_frames=total_frames)
            elif isinstance(value, dict):
                sliced[key] = {
                    k: self._slice_temporal_tensor(v, end_frame=active_end, total_frames=total_frames)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in value.items()
                }
        return sliced

    def sample(self, latents, steps=50, generator=None, chunk_index=None, interval_k=0.5):
        device = self.condition.device
        timesteps, _ = retrieve_timesteps(self.scheduler, steps, device, None)
        do_classifier_free_guidance = self.cfg_scale > 1

        batch_size, num_latent_channels, num_frames, height, width = latents.shape
        chunks = self.create_temporal_chunks(num_frames, chunk_index or [0])
        num_chunks = len(chunks)
        if num_chunks <= 1:
            return super().sample(latents, steps=steps, generator=generator)
        if interval_k <= 0:
            raise ValueError(f"interval_k must be positive for ChunkFlowEuler, got {interval_k}")

        condition_frame_info = dict(
            ((self.model_kwargs or {}).get("data_info", {}) or {}).get("condition_frame_info", {}) or {}
        )
        condition_mask = torch.zeros_like(latents)
        for frame_idx in condition_frame_info:
            if 0 <= int(frame_idx) < num_frames:
                condition_mask[:, :, int(frame_idx)] = 1

        chunk_start_steps = [int(i * float(interval_k) * steps) for i in range(num_chunks)]
        total_steps = chunk_start_steps[-1] + steps
        timestep_matrix = torch.full((num_chunks, total_steps), -1, dtype=torch.float32, device=device)
        for chunk_idx, chunk_start in enumerate(chunk_start_steps):
            for step_idx, t in enumerate(timesteps):
                timestep_matrix[chunk_idx, chunk_start + step_idx] = float(t.item())
            if chunk_start + steps < total_steps:
                timestep_matrix[chunk_idx, chunk_start + steps :] = 0.0

        prompt_embeds = self.condition
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([self.uncondition, self.condition], dim=0)

        chunk_latents = [latents[:, :, start:end].clone() for start, end in chunks]

        for global_step in tqdm(range(total_steps), disable=os.getenv("DPM_TQDM", "False") == "True"):
            active_chunk_indices = [
                chunk_idx for chunk_idx in range(num_chunks) if timestep_matrix[chunk_idx, global_step] >= 0
            ]
            if not active_chunk_indices:
                continue

            active_latents = torch.cat([chunk_latents[idx] for idx in active_chunk_indices], dim=2)
            active_end = chunks[active_chunk_indices[-1]][1]
            active_condition_mask = condition_mask[:, :, :active_end]
            model_kwargs = self._slice_model_kwargs_for_active_prefix(active_end=active_end, total_frames=num_frames)
            model_kwargs["chunk_index"] = [chunks[idx][0] for idx in active_chunk_indices]
            model_kwargs["data_info"] = {**model_kwargs.get("data_info", {}), "condition_frame_info": {}}

            latent_model_input = torch.cat([active_latents] * 2) if do_classifier_free_guidance else active_latents

            timestep_list = []
            for chunk_idx in active_chunk_indices:
                start, end = chunks[chunk_idx]
                timestep_list.extend([timestep_matrix[chunk_idx, global_step]] * (end - start))
            timestep_tensor = torch.stack(timestep_list).to(device=device, dtype=torch.float32)
            timestep_tensor = timestep_tensor.view(1, 1, -1, 1, 1).expand(
                batch_size,
                num_latent_channels,
                -1,
                height,
                width,
            )
            timestep_tensor = (1 - active_condition_mask) * timestep_tensor
            if do_classifier_free_guidance:
                timestep_tensor = torch.cat([timestep_tensor, timestep_tensor], dim=0)

            noise_pred = self.model(
                latent_model_input,
                timestep_tensor[:, :1, :, 0, 0],
                prompt_embeds,
                **model_kwargs,
            )
            if isinstance(noise_pred, Transformer2DModelOutput):
                noise_pred = noise_pred[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
                timestep_tensor = timestep_tensor.chunk(2)[0]

            latents_dtype = active_latents.dtype
            active_shape = active_latents.shape
            t = timesteps[min(global_step, len(timesteps) - 1)]
            denoised = self.scheduler.step(
                -noise_pred.reshape(batch_size, num_latent_channels, -1).transpose(1, 2),
                t,
                active_latents.reshape(batch_size, num_latent_channels, -1).transpose(1, 2),
                per_token_timesteps=timestep_tensor.reshape(batch_size, num_latent_channels, -1)[:, 0],
                return_dict=False,
            )[0]
            denoised = denoised.transpose(1, 2).reshape(active_shape)
            if denoised.dtype != latents_dtype:
                denoised = denoised.to(latents_dtype)

            frame_offset = 0
            for chunk_idx in active_chunk_indices:
                start, end = chunks[chunk_idx]
                chunk_len = end - start
                chunk_latents[chunk_idx] = denoised[:, :, frame_offset : frame_offset + chunk_len]
                frame_offset += chunk_len

        for chunk_latent, (start, end) in zip(chunk_latents, chunks):
            latents[:, :, start:end] = chunk_latent
        return latents
