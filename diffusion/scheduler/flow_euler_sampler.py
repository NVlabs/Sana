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
