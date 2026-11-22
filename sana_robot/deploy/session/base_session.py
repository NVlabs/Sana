# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.deploy.runtime.cuda_graph import CudaGraphDriver
from sana_robot.deploy.sampling.flow_euler import (
    draw_noise,
    euler_step,
    flow_schedule,
    guided_flow,
    host_geometry,
    sample_dtype,
    session_verb,
)
from sana_robot.policy.attention.condition import PREFILL, TARGET, ConditionPrefix
from sana_robot.policy.sana_robot_base import SanaRobotBase


class BaseSession:
    """Inference of SANA-Robot-base: one observation and state in, one action chunk out, nothing kept between chunks.

    The observation (the current latent frame of every view) and the state row form the condition chunk, which
    does not read the noisy prediction. It runs once per chunk and every denoising step runs the prediction chunk
    alone against what it left (prefix caching). An optimized session replays the whole generation as one CUDA
    graph.
    """

    def __init__(
        self,
        policy: SanaRobotBase,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        data_info: dict,
        steps: int,
        video_flow_shift: float = 5.0,
        action_flow_shift: float = 1.0,
        optimized: bool = True,
    ) -> None:
        """Build a session for one task prompt.

        Args:
            policy: The policy, in eval mode on its device.
            text_embeddings: Text embeddings of the prompt groups [1, V + 1, 1, L_text, C_text].
            text_mask: Text token mask [1, V + 1, L_text].
            data_info: What stays the same over the episode: ``view_count``, ``view_latent_shape``,
                ``view_slot_ids``, ``model_fps`` and ``video_frame_stride``.
            steps: Denoising steps per chunk.
            video_flow_shift: Flow shift of the video latents.
            action_flow_shift: Flow shift of the actions.
            optimized: Replay a generation as a CUDA graph; needs a CUDA device.
        """
        self.policy = policy
        self.text_embeddings = text_embeddings.to(policy.dtype)
        self.text_mask = text_mask
        self.data_info = host_geometry(data_info)
        self.steps = int(steps)
        device = next(policy.parameters()).device
        (self.video_timesteps, self.video_levels) = flow_schedule(self.steps, video_flow_shift, device)
        (self.action_timesteps, self.action_levels) = flow_schedule(self.steps, action_flow_shift, device)
        self.frames_per_latent = policy.vae_temporal_stride * policy.video_frame_stride(self.data_info)
        self.optimized = optimized and device.type == "cuda"
        self.packed_text = {} if self.optimized else None
        self.generate_step = CudaGraphDriver(self._generate) if self.optimized else self._generate

    @session_verb
    def generate(
        self,
        observation_latent: torch.Tensor,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        video_noise: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        guidance: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one chunk from the current observation and state.

        Args:
            observation_latent: The current frame of every view encoded alone, [1, C, 1, 1, sum(H_v * W_v)].
            state: The normalized robot state [1, 80].
            state_mask: Which state dimensions the robot has, [1, 80].
            action_mask: Which action dimensions the robot has, [1, A, 80].
            video_noise: Starting noise of the future latents [1, C, f, 1, sum(H_v * W_v)]; drawn when None.
            action_noise: Starting noise of the actions [1, A, 80]; drawn when None.
            generator: Random generator of the noise that is drawn.
            guidance: Real-time chunking guidance: the actions to stay close to [1, A, 80] and their weights
                [1, A, 80], as ``sana_robot.deploy.async_exec.rtc.rtc_guidance`` returns them; None for none.

        Returns:
            The normalized actions [1, A, 80] and the predicted future latents [1, C, f, 1, sum(H_v * W_v)].
        """
        (device, dtype) = (observation_latent.device, sample_dtype(self.policy))
        if video_noise is None:
            future_frames = action_mask.shape[1] // self.frames_per_latent
            shape = (1, observation_latent.shape[1], future_frames, *observation_latent.shape[3:])
            video_noise = draw_noise(shape, generator, device, dtype)
        if action_noise is None:
            action_noise = draw_noise(tuple(action_mask.shape), generator, device, dtype)
        if guidance is None:
            guidance = (action_noise.new_zeros(action_mask.shape), action_noise.new_zeros(action_mask.shape))
        (action, future) = self.generate_step(
            observation_latent.to(dtype),
            state.to(dtype),
            state_mask,
            action_mask,
            video_noise.to(dtype),
            action_noise.to(dtype),
            guidance[0].to(dtype),
            guidance[1].to(dtype),
        )
        return (action.clone(), future.clone())

    def _generate(self, observation, state, state_mask, action_mask, future, action, guide, guide_weights) -> tuple:
        """Run the condition chunk once, then denoise the prediction chunk from the given noise; returns the actions and the future latents."""
        (chunk_size, future_frames) = (action.shape[1], future.shape[2])
        action = action.masked_fill(~action_mask, 0)
        info = dict(
            self.data_info, initial_state=state, initial_state_condition_mask=state_mask, action_mask=action_mask
        )
        prefix = ConditionPrefix()
        for step in range(self.steps):
            frame_timesteps = self.video_timesteps[step].expand(1, 1 + future_frames).clone()
            frame_timesteps[:, 0] = 0
            step_info = dict(info, action=action, action_timestep=self.action_timesteps[step].expand(1, chunk_size))
            inputs = (torch.cat((observation, future), dim=2), frame_timesteps, self.text_embeddings, self.text_mask)
            if step == 0:
                self.policy(*inputs, data_info=step_info, condition_prefix=(prefix, PREFILL))
            prediction = self.policy(*inputs, data_info=step_info, condition_prefix=(prefix, TARGET))
            action_flow = guided_flow(prediction["action_pred"], action, self.action_levels[step], guide, guide_weights)
            future = euler_step(
                future, prediction["video_pred"][:, :, 1:], self.video_timesteps[step], self.video_levels[step + 1]
            )
            action = euler_step(action, action_flow, self.action_timesteps[step], self.action_levels[step + 1])
            action = action.masked_fill(~action_mask, 0)
        return (action, future)
