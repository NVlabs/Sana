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
    def __init__(self, model_fn, condition, uncondition, cfg_scale, flow_shift=3.0, model_kwargs=None):
        super().__init__(model_fn, condition, uncondition, cfg_scale, flow_shift, model_kwargs)

    def create_temporal_chunks(self, num_frames, chunk_index):
        """
        Create temporal chunks based on provided indices.

        Args:
            num_frames: Total number of frames
            chunk_index: List of chunk start indices (must be sorted)

        Returns:
            List of tuples (start_idx, end_idx) for each chunk
        """
        if not chunk_index:
            return [(0, num_frames)]

        chunks = []
        chunk_index = sorted(chunk_index)

        # Add intermediate chunks
        for i in range(len(chunk_index) - 1):
            chunks.append((chunk_index[i], chunk_index[i + 1]))

        # Add last chunk
        chunks.append((chunk_index[-1], num_frames))

        return chunks

    def sample(self, latents, steps=50, generator=None, chunk_index=None, interval_k=0.5):
        """
        Sample with chunked temporal denoising.

        Args:
            latents: Input latents of shape (1, C, F, H, W)
            steps: Total number of denoising steps
            generator: Random generator for noise
            chunk_index: List of chunk start indices for temporal splitting
            interval_k: Interval ratio (0-1) for staggered denoising start

        Returns:
            Denoised latents
        """
        device = self.condition.device
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, steps, device, None)
        do_classifier_free_guidance = self.cfg_scale > 1

        # Extract shape information
        batch_size, num_latent_channels, num_frames, height, width = latents.shape

        # Create temporal chunks
        chunks = self.create_temporal_chunks(num_frames, chunk_index or [0])
        num_chunks = len(chunks)

        condition_frame_info = self.model_kwargs["data_info"].pop(
            "condition_frame_info", {}
        )  # {frame_idx: frame_weight}
        condition_mask = torch.zeros_like(latents)  # 1,C,F,H,W
        image_cond_noise_scale = 0.0
        for frame_idx, frame_weight in condition_frame_info.items():
            condition_mask[:, :, frame_idx] = 1
            image_cond_noise_scale = max(
                image_cond_noise_scale, frame_weight
            )  # NOTE image condition noise scale is not supported yet

        # Calculate when each chunk starts denoising
        chunk_start_steps = []
        for i in range(num_chunks):
            start_step = int(i * interval_k * steps)
            chunk_start_steps.append(start_step)

        # Total steps needed
        total_steps_needed = chunk_start_steps[-1] + steps if num_chunks > 1 else steps

        # Initialize timestep tracking matrix
        # Shape: (num_chunks, total_steps_needed)
        # -1: not started, 0: fully denoised, >0: current timestep
        timestep_matrix = torch.full((num_chunks, total_steps_needed), -1, dtype=torch.float32, device=device)

        # Fill the timestep matrix based on when each chunk is active
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_start_steps[chunk_idx]
            for step_idx, t in enumerate(timesteps):
                global_step = chunk_start + step_idx
                timestep_matrix[chunk_idx, global_step] = t.item()
            # Set remaining steps to 0 (fully denoised)
            for global_step in range(chunk_start + steps, total_steps_needed):
                timestep_matrix[chunk_idx, global_step] = 0

        prompt_embeds = self.condition
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([self.uncondition, self.condition], dim=0)

        init_latents = latents.clone()

        # Split latents by chunks
        chunk_latents = []
        for start, end in chunks:
            chunk_latents.append(latents[:, :, start:end].clone())

        # Main denoising loop
        for global_step in tqdm(range(total_steps_needed), disable=os.getenv("DPM_TQDM", "False") == "True"):

            # Determine which chunks are active (started denoising)
            active_chunk_indices = []
            active_timesteps = []

            for chunk_idx in range(num_chunks):
                current_timestep = timestep_matrix[chunk_idx, global_step]
                if current_timestep >= 0:  # Chunk has started denoising
                    active_chunk_indices.append(chunk_idx)
                    active_timesteps.append(current_timestep)

            # Skip if no chunks are active
            if not active_chunk_indices:
                continue

            # Concatenate active chunks along frame dimension
            active_latents_list = []
            condition_mask_list = []
            for idx in active_chunk_indices:
                active_latents_list.append(chunk_latents[idx])
                start, end = chunks[idx]
                condition_mask_list.append(condition_mask[:, :, start:end])  # b,c,f,h,w

            # Concatenate along frame dimension
            concatenated_latents = torch.cat(active_latents_list, dim=2)  # Shape: (1, C, total_active_frames, H, W)

            # Prepare model inputs
            latent_model_input = (
                torch.cat([concatenated_latents] * 2) if do_classifier_free_guidance else concatenated_latents
            )

            # Create timestep tensor for concatenated frames
            timestep_list = []
            for idx, chunk_idx in enumerate(active_chunk_indices):
                start, end = chunks[chunk_idx]
                chunk_frames = end - start
                timestep_value = active_timesteps[idx]
                timestep_list.extend([timestep_value] * chunk_frames)

            timestep_tensor = torch.tensor(timestep_list, device=device, dtype=torch.float32)
            timestep_tensor = timestep_tensor.view(1, 1, -1, 1, 1).expand(
                batch_size, num_latent_channels, -1, height, width
            )  # b,1,f,h,w
            timestep_tensor = (1 - torch.cat(condition_mask_list, dim=2)) * timestep_tensor

            if do_classifier_free_guidance:
                timestep_tensor = torch.cat([timestep_tensor, timestep_tensor], dim=0)

            # Model forward pass
            # chunk index should be the start index of the active chunks
            self.model_kwargs["chunk_index"] = [chunks[idx][0] for idx in active_chunk_indices]
            noise_pred = self.model(
                latent_model_input,
                timestep_tensor[:, :1, :, 0, 0],  # b,1,f
                prompt_embeds,
                **self.model_kwargs,
            )

            if isinstance(noise_pred, Transformer2DModelOutput):
                noise_pred = noise_pred[0]

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
                timestep_tensor = timestep_tensor.chunk(2)[0]

            # Compute the previous noisy sample x_t -> x_t-1
            latents_dtype = concatenated_latents.dtype
            concat_shape = concatenated_latents.shape

            # Get the actual timestep value for scheduler
            t = timesteps[min(global_step, len(timesteps) - 1)]

            # Reshape for scheduler
            denoised_latents = self.scheduler.step(
                -noise_pred.reshape(batch_size, num_latent_channels, -1).transpose(1, 2),
                t,
                concatenated_latents.reshape(batch_size, num_latent_channels, -1).transpose(1, 2),
                per_token_timesteps=timestep_tensor.reshape(batch_size, num_latent_channels, -1)[:, 0],
                return_dict=False,
            )[0]
            denoised_latents = denoised_latents.transpose(1, 2).reshape(concat_shape)

            if denoised_latents.dtype != latents_dtype:
                denoised_latents = denoised_latents.to(latents_dtype)

            # Split denoised latents back to chunks
            frame_offset = 0
            for idx, chunk_idx in enumerate(active_chunk_indices):
                start, end = chunks[chunk_idx]
                chunk_frames = end - start
                chunk_latents[chunk_idx] = denoised_latents[:, :, frame_offset : frame_offset + chunk_frames]
                frame_offset += chunk_frames

        # Reconstruct full latents from chunks
        for chunk_idx, (start, end) in enumerate(chunks):
            latents[:, :, start:end] = chunk_latents[chunk_idx]

        return latents
