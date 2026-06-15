# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed-RoPE streaming sampler for standalone Sana V2V modules."""

from __future__ import annotations

import copy
import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from tqdm import tqdm


class SANAStreamingSampler:
    """Self-forcing streaming sampler that only supports fixed RoPE caching.

    The sampler expects attention modules to be built directly in their runtime
    cache format:
      - state-cache blocks expose ``fixed_rope_cache_type = "state"``
      - softmax blocks expose ``fixed_rope_cache_type = "softmax"``

    It does not dynamically swap attention classes.
    """

    _STATE_CACHE_CLASS_NAMES = {
        "V2VStateCachedBiGDNAttention",
        "QuantizedStateCachedGDN",
    }

    def __init__(
        self,
        model_fn,
        condition,
        uncondition,
        cfg_scale,
        flow_shift=3.0,
        model_kwargs=None,
        base_chunk_frames=10,
        num_cached_blocks=-1,
        cache_strategy="fixed_rope",
        efficient_cache=False,
        **kwargs,
    ):
        if cache_strategy not in ("fixed_rope", ""):
            raise ValueError(f"SANAStreamingSampler only supports fixed_rope, got {cache_strategy!r}")
        self.model = model_fn
        self.condition = condition
        self.uncondition = uncondition
        self.cfg_scale = cfg_scale
        self.model_kwargs = model_kwargs or {}
        self.mask = self.model_kwargs.pop("mask", None)
        self.flow_shift = flow_shift
        self.base_chunk_frames = base_chunk_frames
        self.num_cached_blocks = num_cached_blocks
        self.efficient_cache = efficient_cache
        self.sink_token = kwargs.get("sink_token", False)
        self._fixed_rope_full_history_softmax_cache = self.num_cached_blocks < 0
        self.block_is_state_cached = self._detect_cache_blocks()
        self.num_model_blocks = len(self.block_is_state_cached)

    def _model_blocks(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model, "blocks"):
            return model.blocks
        if hasattr(model, "transformer_blocks"):
            return model.transformer_blocks
        if hasattr(model, "layers"):
            return model.layers
        raise ValueError("Model does not have blocks/transformer_blocks/layers")

    def _detect_cache_blocks(self) -> list[bool]:
        block_is_state_cached = []
        for block in self._model_blocks():
            attn = getattr(block, "attn", None)
            cls_name = type(attn).__name__
            cache_type = getattr(attn, "fixed_rope_cache_type", None)
            if cache_type == "state":
                block_is_state_cached.append(True)
            elif cache_type == "softmax":
                block_is_state_cached.append(False)
            else:
                block_is_state_cached.append(cls_name in self._STATE_CACHE_CLASS_NAMES)
        return block_is_state_cached

    def create_autoregressive_segments(self, total_frames):
        remained_frames = total_frames % self.base_chunk_frames
        num_chunks = total_frames // self.base_chunk_frames
        chunk_indices = [0]
        for i in range(num_chunks):
            cur_idx = chunk_indices[-1] + self.base_chunk_frames
            if i == 0:
                cur_idx += remained_frames
            chunk_indices.append(cur_idx)
        return chunk_indices

    def _initialize_kv_cache(self, num_chunks: int):
        return [[[None] * 6 for _ in range(self.num_model_blocks)] for _ in range(num_chunks)]

    def _accumulate_fixed_rope_kv_cache(self, kv_cache, chunk_idx):
        cur_kv_cache = kv_cache[chunk_idx]
        start_chunk_idx = max(chunk_idx - self.num_cached_blocks, 0) if self.num_cached_blocks > 0 else 0
        num_cached_frames = 0
        sink_num = 0

        for block_id, is_state_cached in enumerate(self.block_is_state_cached):
            if is_state_cached:
                prev = kv_cache[chunk_idx - 1][block_id]
                cur_kv_cache[block_id][0] = prev[0]
                cur_kv_cache[block_id][1] = prev[1]
                cur_kv_cache[block_id][-1] = prev[-1]
                continue

            if self._fixed_rope_full_history_softmax_cache:
                prev = kv_cache[chunk_idx - 1][block_id]
                previous_q, previous_k, previous_v = prev[0], prev[1], prev[2]
                previous_tconv = prev[-1]
                cur_kv_cache[block_id] = [previous_q, previous_k, previous_v, None, None, previous_tconv]
                if previous_q is not None:
                    hw = getattr(self, "_spatial_hw", 0)
                    if hw > 0:
                        num_cached_frames = previous_q.shape[-1] // hw
                continue

            previous_q, previous_k, previous_v = None, None, None
            previous_tconv = None
            valid_cached_chunks = list(range(start_chunk_idx, chunk_idx))

            if self.num_cached_blocks > 0 and self.sink_token:
                window_start_chunk = max(chunk_idx - self.num_cached_blocks + 1, 0)
                if window_start_chunk > 0:
                    valid_cached_chunks = [0] + list(range(window_start_chunk, chunk_idx))
                    if sink_num == 0:
                        sink_num = self._chunk_indices[1] - self._chunk_indices[0]

            for cache_idx in range(chunk_idx):
                if cache_idx not in valid_cached_chunks:
                    kv_cache[cache_idx][block_id] = [None] * 6
                    continue

                prev = kv_cache[cache_idx][block_id]
                if prev[0] is not None:
                    if previous_q is None:
                        previous_q = prev[0].clone()
                        previous_k = prev[1].clone()
                        previous_v = prev[2].clone()
                    else:
                        previous_q = torch.cat([previous_q, prev[0]], dim=-1)
                        previous_k = torch.cat([previous_k, prev[1]], dim=-1)
                        previous_v = torch.cat([previous_v, prev[2]], dim=-1)

                if prev[-1] is not None:
                    if previous_tconv is None:
                        previous_tconv = prev[-1].clone()
                    else:
                        previous_tconv = torch.cat([previous_tconv, prev[-1]], dim=2)

            cur_kv_cache[block_id] = [previous_q, previous_k, previous_v, None, None, previous_tconv]
            if previous_q is not None:
                hw = getattr(self, "_spatial_hw", 0)
                if hw > 0:
                    num_cached_frames = previous_q.shape[-1] // hw

        return cur_kv_cache, chunk_idx - start_chunk_idx, sink_num, num_cached_frames

    def accumulate_kv_cache(self, kv_cache, chunk_idx):
        if chunk_idx == 0:
            return kv_cache[0], 0, 0, 0
        return self._accumulate_fixed_rope_kv_cache(kv_cache, chunk_idx)

    def _promote_fixed_rope_full_history_cache(self, kv_cache, chunk_idx):
        if not self._fixed_rope_full_history_softmax_cache or chunk_idx == 0:
            return

        for block_id, is_state_cached in enumerate(self.block_is_state_cached):
            if is_state_cached:
                continue

            prev = kv_cache[chunk_idx - 1][block_id]
            cur = kv_cache[chunk_idx][block_id]

            if prev[0] is not None and cur[0] is not None:
                cur[0] = torch.cat([prev[0], cur[0]], dim=-1)
                cur[1] = torch.cat([prev[1], cur[1]], dim=-1)
                cur[2] = torch.cat([prev[2], cur[2]], dim=-1)
            elif prev[0] is not None:
                cur[0], cur[1], cur[2] = prev[0], prev[1], prev[2]

            if prev[-1] is not None and cur[-1] is not None:
                cur[-1] = torch.cat([prev[-1], cur[-1]], dim=2)
            elif prev[-1] is not None:
                cur[-1] = prev[-1]

            kv_cache[chunk_idx - 1][block_id] = [None] * len(prev)

    @staticmethod
    def _expand_per_chunk(tensor, batch_size, num_chunks, name, allow_no_batch=False):
        if tensor is None:
            return None
        shape = tensor.shape
        if tensor.dim() >= 2 and shape[0] == batch_size and shape[1] == num_chunks:
            return tensor
        if shape[0] == batch_size:
            return tensor.unsqueeze(1).expand(batch_size, num_chunks, *shape[1:])
        if shape[0] == num_chunks and batch_size == 1:
            return tensor.unsqueeze(0)
        if allow_no_batch:
            return tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, num_chunks, *shape)
        raise AssertionError(
            f"{name} shape {tuple(shape)} incompatible with batch_size={batch_size}, num_chunks={num_chunks}"
        )

    @staticmethod
    def _timesteps_for_steps(scheduler, steps, device):
        timesteps, _ = retrieve_timesteps(scheduler, steps, device, None)
        if steps == 4:
            timesteps = torch.tensor([1000, 961, 893, 743], device=device)
            scheduler.timesteps = timesteps
            scheduler.sigmas = torch.cat([timesteps / 1000, torch.zeros(1, device=device)])
        elif steps == 2:
            timesteps = torch.tensor([1000, 743], device=device)
            scheduler.timesteps = timesteps
            scheduler.sigmas = torch.cat([timesteps / 1000, torch.zeros(1, device=device)])
        return timesteps

    @torch.no_grad()
    def sample(self, latents, steps=50, **kwargs):
        device = self.condition.device
        do_classifier_free_guidance = self.cfg_scale > 1
        batch_size, _, total_frames, height, width = latents.shape
        self._spatial_hw = height * width

        if total_frames <= self.base_chunk_frames:
            raise ValueError("Use the standard flow sampler for short videos")

        chunk_indices = self.create_autoregressive_segments(total_frames)
        self._chunk_indices = chunk_indices
        num_chunks = len(chunk_indices) - 1
        kv_cache = self._initialize_kv_cache(num_chunks)

        cond_per_chunk = self._expand_per_chunk(self.condition, batch_size, num_chunks, "condition")
        mask_per_chunk = self._expand_per_chunk(self.mask, batch_size, num_chunks, "mask", allow_no_batch=True)
        uncond = self.uncondition
        if uncond.shape[0] == 1 and batch_size > 1:
            uncond = uncond.expand(batch_size, *uncond.shape[1:])
        elif uncond.shape[0] not in (1, batch_size):
            raise AssertionError(f"uncondition first dim must be 1 or batch_size={batch_size}, got {uncond.shape[0]}")

        data_info = self.model_kwargs.pop("data_info", {})
        image_vae_embeds = data_info.get("image_vae_embeds", None)

        for chunk_idx in tqdm(
            range(num_chunks),
            disable=os.getenv("DPM_TQDM", "False") == "True",
            desc="Processing chunks",
        ):
            chunk_kv_cache, _, sink_num, num_cached_frames = self.accumulate_kv_cache(kv_cache, chunk_idx)

            prompt_embeds = cond_per_chunk[:, chunk_idx]
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([uncond, prompt_embeds], dim=0)

            mask = mask_per_chunk[:, chunk_idx] if mask_per_chunk is not None else None
            scheduler = FlowMatchEulerDiscreteScheduler(shift=self.flow_shift)
            timesteps = self._timesteps_for_steps(scheduler, steps, device)

            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            current_num_frames = end_f - start_f
            cache_start_chunk_idx = max(chunk_idx - self.num_cached_blocks, 0) if self.num_cached_blocks > 0 else 0

            frame_index = None
            if sink_num > 0:
                sink_fi = torch.arange(sink_num, device=device)
                non_sink_count = num_cached_frames - sink_num + current_num_frames
                window_start_f = end_f - non_sink_count
                remaining_fi = torch.arange(window_start_f, end_f, device=device)
                frame_index = torch.cat([sink_fi, remaining_fi], dim=0)
                rope_start_f = 0
                rope_end_f = end_f
            else:
                rope_start_f = chunk_indices[cache_start_chunk_idx]
                rope_end_f = end_f

            local_data_info = copy.deepcopy(data_info)
            if image_vae_embeds is not None:
                local_data_info["image_vae_embeds"] = image_vae_embeds[:, :, start_f:end_f]

            is_last_chunk = chunk_idx == num_chunks - 1
            for step_idx, t in enumerate(timesteps):
                is_last_step = step_idx == len(timesteps) - 1
                save_cache_now = self.efficient_cache and is_last_step and not is_last_chunk
                latent_model_input = (
                    torch.cat([latents[:, :, start_f:end_f]] * 2)
                    if do_classifier_free_guidance
                    else latents[:, :, start_f:end_f]
                )
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred, step_kv_cache = self.model(
                    latent_model_input,
                    timestep,
                    prompt_embeds,
                    start_f=rope_start_f,
                    end_f=rope_end_f,
                    frame_index=frame_index,
                    save_kv_cache=save_cache_now,
                    kv_cache=chunk_kv_cache,
                    mask=mask,
                    data_info=local_data_info,
                    **self.model_kwargs,
                )

                if save_cache_now:
                    kv_cache[chunk_idx] = step_kv_cache
                    self._promote_fixed_rope_full_history_cache(kv_cache, chunk_idx)

                if isinstance(noise_pred, Transformer2DModelOutput):
                    noise_pred = noise_pred[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)

                latents_dtype = latents.dtype
                latents[:, :, start_f:end_f] = scheduler.step(
                    noise_pred, t, latents[:, :, start_f:end_f], return_dict=False
                )[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

            if not self.efficient_cache and not is_last_chunk:
                latent_model_input = (
                    torch.cat([latents[:, :, start_f:end_f]] * 2)
                    if do_classifier_free_guidance
                    else latents[:, :, start_f:end_f]
                )
                timestep = torch.zeros(latent_model_input.shape[0], device=device)
                _, updated_kv_cache = self.model(
                    latent_model_input,
                    timestep,
                    prompt_embeds,
                    start_f=rope_start_f,
                    end_f=rope_end_f,
                    frame_index=frame_index,
                    save_kv_cache=True,
                    kv_cache=chunk_kv_cache,
                    mask=mask,
                    data_info=local_data_info,
                    **self.model_kwargs,
                )
                kv_cache[chunk_idx] = updated_kv_cache
                self._promote_fixed_rope_full_history_cache(kv_cache, chunk_idx)

        return latents
