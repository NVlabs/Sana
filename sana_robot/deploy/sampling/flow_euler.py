# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

TRAIN_TIMESTEPS = 1000
MAX_GUIDANCE_WEIGHT = 10.0


def session_verb(method):
    """Run a session method in inference mode, under bf16 autocast when the session's policy has bf16 weights, with the session's own memo of the packed text on the policy."""

    @functools.wraps(method)
    def run(self, *arguments, **keyword_arguments):
        policy = self.policy
        policy.packed_text_cache = self.packed_text
        device_type = next(policy.parameters()).device.type
        with (
            torch.inference_mode(),
            torch.autocast(device_type, dtype=torch.bfloat16, enabled=policy.dtype == torch.bfloat16),
        ):
            return method(self, *arguments, **keyword_arguments)

    return run


def sample_dtype(policy) -> torch.dtype:
    """Return the dtype the sampler keeps its samples in: float32, or the policy's when that is wider."""
    return torch.promote_types(policy.dtype, torch.float32)


def flow_schedule(steps: int, shift: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the timesteps [steps] of a Flow-Euler sampler with the given shift and its noise levels [steps + 1], the last one 0."""
    scheduler = FlowMatchEulerDiscreteScheduler(shift=float(shift))
    scheduler.set_timesteps(int(steps), device=device)
    return (scheduler.timesteps, scheduler.sigmas)


def euler_step(
    sample: torch.Tensor, flow: torch.Tensor, timestep: torch.Tensor, next_level: torch.Tensor
) -> torch.Tensor:
    """Move a sample from the noise level of ``timestep`` to ``next_level`` along the predicted flow."""
    dtype = torch.promote_types(sample.dtype, torch.float32)
    level = timestep / TRAIN_TIMESTEPS
    return (sample.to(dtype) + (next_level - level) * flow.to(dtype)).to(sample.dtype)


def draw_noise(shape, generator, device, dtype) -> torch.Tensor:
    """Draw Gaussian noise on the device."""
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def host_geometry(data_info: dict) -> dict:
    """Return a copy of ``data_info`` with the view geometry on the CPU and the view count and the frame stride as ints: the policy reads them on the host at every forward, which a captured CUDA graph must not wait for."""
    info = dict(data_info)
    for name in ("view_latent_shape", "view_slot_ids"):
        if isinstance(info.get(name), torch.Tensor):
            info[name] = info[name].detach().cpu()
    for name in ("view_count", "video_frame_stride"):
        if isinstance(info.get(name), torch.Tensor):
            info[name] = int(info[name].reshape(-1)[0])
    return info


def guidance_weight(noise_level: torch.Tensor) -> torch.Tensor:
    """Return the weight of the real-time chunking guidance at a noise level, capped at ``MAX_GUIDANCE_WEIGHT``."""
    clean = 1 - noise_level
    ratio = torch.nan_to_num(noise_level / clean, posinf=MAX_GUIDANCE_WEIGHT)
    spread = (noise_level**2 + clean**2) / noise_level**2
    return torch.clamp(torch.nan_to_num(ratio * spread, posinf=MAX_GUIDANCE_WEIGHT), max=MAX_GUIDANCE_WEIGHT)


def guided_flow(
    flow: torch.Tensor, action: torch.Tensor, noise_level: torch.Tensor, guide: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Pull an action flow toward the actions ``guide`` under ``weights`` (real-time chunking): the clean actions the step implies, action - noise_level * flow, are compared with the guide and the flow becomes flow - weight * error. Zero weights leave the flow as it is."""
    error = (guide.float() - (action.float() - noise_level * flow.float())) * weights.float()
    return (flow.float() - guidance_weight(noise_level) * error).to(flow.dtype)
