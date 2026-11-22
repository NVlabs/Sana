# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque

import torch

from sana_robot.deploy.async_exec.memory_rtc import MemoryRTCController
from sana_robot.deploy.runtime.deployment import Deployment


class EpisodePolicy:
    """SANA-Robot-AR in the closed loop of a simulator, one episode at a time.

    The policy generates a chunk every ``n_action_steps`` control steps from the current frames and state, on the
    memory of the frames it has seen since the start of the episode (memory RTC without delay: the simulator
    waits for every chunk).

    Args:
        deployment: The deployment of the checkpoint.
    """

    def __init__(self, deployment: Deployment) -> None:
        config = deployment.config
        self.deployment = deployment
        self.robot = deployment.robot
        self.n_action_steps = config.n_action_steps or config.chunk_size
        self.frame_stride = config.video_frame_stride
        self.window = deque(maxlen=config.chunk_size // config.video_frame_stride)
        (self.instruction, self.session, self.controller, self.step) = (None, None, None, 0)

    def start(self, images, instruction: str) -> None:
        """Open an episode at the camera frames of its first control step, uint8 [H, W, 3] per camera."""
        if self.session is None or instruction != self.instruction:
            self.session = self.deployment.session(instruction)
            self.controller = MemoryRTCController(self.session, self.deployment.stream, 0, self.n_action_steps)
            self.instruction = instruction
        frames = self.robot.frames(images)
        self.window.clear()
        self.window.extend([frames] * self.window.maxlen)
        self.step = 0

    def observe(self, images) -> None:
        """Take the camera frames after another ``video_frame_stride`` control steps."""
        self.step += self.frame_stride
        self.window.append(self.robot.frames(images))

    def plan(self, state_row, instruction: str, generate: bool = True, seed: int | None = None):
        """Bring the memory to the current step and generate the next chunk.

        Args:
            state_row: The robot state [80] at the current step, a multiple of ``n_action_steps`` control steps.
            instruction: The task text at this step; a new text replaces the prompt and the memory stays.
            generate: Generate a chunk; when off the policy only takes the frames into its memory.
            seed: The seed of the noise the chunk is denoised from.

        Returns:
            The first ``n_action_steps`` action rows of the chunk [n, 80] in the dataset's units, or None.
        """
        if instruction != self.instruction:
            self.session.set_prompt(*self.deployment.encode_text([self.robot.prompts(instruction)]))
            self.instruction = instruction
        clips = [torch.cat(view, dim=1) for view in zip(*self.window)]
        (state, state_mask) = self.robot.state(state_row)
        noise = {} if seed is None else {"generator": torch.Generator(self.deployment.device).manual_seed(seed)}
        if self.step == 0:
            actions = self.controller.open(clips, state, state_mask, self.robot.action_mask, **noise)
        elif generate:
            actions = self.controller.generate(self.step, clips, state, state_mask, self.robot.action_mask, **noise)
        else:
            self.controller.observe(self.step, clips)
        return self.robot.actions(actions)[: self.n_action_steps] if generate else None
