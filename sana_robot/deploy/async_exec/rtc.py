# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

from sana_robot.deploy.memory.view_stream import ViewStream
from sana_robot.deploy.session.base_session import BaseSession


def prefix_row_weights(overlap_steps: int, async_delay_steps: int) -> torch.Tensor:
    """Return the guidance weight of every row a new chunk shares with the running one: 1 on the first ``async_delay_steps``, which the robot runs while the chunk is generated, then a decay to 0 over the rest."""
    delay = min(int(async_delay_steps), int(overlap_steps))
    weights = torch.zeros(int(overlap_steps))
    weights[:delay] = 1
    if overlap_steps > delay:
        decay = torch.linspace(1, 0, int(overlap_steps) - delay + 2)[1:-1]
        weights[delay:] = decay * torch.expm1(decay) / (math.e - 1)
    return weights


def rtc_guidance(running_rows: torch.Tensor, row_weights: torch.Tensor, action_mask: torch.Tensor) -> tuple:
    """Return the guidance of one chunk for ``BaseSession.generate``: the rows of the running chunk the new chunk shares, ``running_rows`` [1, R, 80], as actions [1, A, 80] on the chunk's first R rows, and their weights [1, A, 80] from ``row_weights`` [R] on the slots the robot has."""
    shared = min(running_rows.shape[1], action_mask.shape[1])
    guide = running_rows.new_zeros(action_mask.shape, dtype=torch.float32)
    guide[:, :shared] = running_rows[:, :shared].float()
    weights = torch.zeros_like(guide)
    weights[:, :shared] = row_weights[:shared].to(guide).view(1, shared, 1)
    return (guide, weights * action_mask)


class RTCController:
    """Real-time chunking of SANA-Robot-base: asynchronous execution with guidance toward the running chunk.

    A chunk is generated every P control steps from the frame and the state observed at its first step, and takes
    over d steps later, 0 <= d <= P <= C - d; P = C - d by default. Its first C - P rows are also rows of the
    running chunk; the guidance pulls them toward it, fully on the d rows the robot runs meanwhile. With d = 0 the
    robot waits for every chunk: P = C is synchronous execution.
    """

    def __init__(
        self,
        session: BaseSession,
        stream: ViewStream,
        chunk_size: int,
        async_delay_steps: int,
        n_action_steps: int | None = None,
        guided: bool = True,
    ) -> None:
        """Build the controller.

        Args:
            session: The session.
            stream: The video stream; only single frames are encoded.
            chunk_size: C, the control steps of a chunk.
            async_delay_steps: d, the control steps the robot runs from the running chunk while the next one is
                generated.
            n_action_steps: P, the control steps between two generated chunks; C - d when None.
            guided: Guide every chunk after the first toward the running one.
        """
        self.session = session
        self.stream = stream
        self.chunk_size = int(chunk_size)
        self.async_delay_steps = int(async_delay_steps)
        self.n_action_steps = (
            self.chunk_size - self.async_delay_steps if n_action_steps is None else int(n_action_steps)
        )
        self.frame_stride = session.policy.video_frame_stride(session.data_info)
        self.window_steps = self.n_action_steps
        self.guided = guided
        self.row_weights = prefix_row_weights(self.chunk_size - self.n_action_steps, self.async_delay_steps)

    def open(self, window, state, state_mask, action_mask, **noise) -> torch.Tensor:
        """Generate chunk 0 from the first frame. Returns its normalized actions [1, C, 80]."""
        observation = self.stream.encode_observation([clip[:, :1] for clip in window])
        return self.session.generate(observation, state, state_mask, action_mask, **noise)[0]

    def observe(self, step: int, window) -> None:
        """Nothing is kept between chunks."""

    def generate(self, step: int, window, state, state_mask, action_mask, running=None, **noise) -> torch.Tensor:
        """Generate the chunk that starts at ``step`` from the newest frame, guided toward the rows it shares with ``running``, the normalized actions [1, C, 80] of the chunk before. Returns its normalized actions [1, C, 80]."""
        observation = self.stream.encode_observation([clip[:, -1:] for clip in window])
        guidance = None
        if self.guided and running is not None and self.chunk_size > self.n_action_steps:
            shared = running[:, self.n_action_steps :]
            guidance = rtc_guidance(shared, self.row_weights, action_mask)
        return self.session.generate(observation, state, state_mask, action_mask, guidance=guidance, **noise)[0]
