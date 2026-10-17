# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
#
# `validate_sigmas` and `shift_sigmas` are derived from `MiniMaxH3Scheduler.set_timesteps` in diffusers
# (`schedulers/scheduling_minimax_h3.py`), Copyright 2025 The MiniMax authors and The HuggingFace Team,
# Apache License 2.0. See THIRD_PARTY_NOTICES.md.
#
# `build_row_time_pairs` is derived from `MiniMaxH3SetTimestepsStep.build_row_timesteps` in diffusers
# (`modular_pipelines/minimax_h3/before_denoise.py`), Copyright 2026 The MiniMax and HuggingFace Teams,
# Apache License 2.0. See THIRD_PARTY_NOTICES.md.
"""The HyperFlow sigma grid and the per-row ``(t, r)`` plan.

MiniMax-H3 runs one packed sequence — text, conditioning rows, target audio, target video — through a single forward
per step, and its scheduler works on the ``t = 1 - sigma`` axis (``t = 1`` is clean). HyperFlow adds one thing: every
row also carries the *endpoint* ``r`` the step is aiming at, ``r_i = 1 - sigma_{i+1}``, so the transformer is
conditioned on the interval ``(t, r)`` rather than the point ``t``.

Only two facts in this module are HyperFlow's own: the 9-point raw sigma grid (8 model evaluations) and the endpoint
rule. Everything else — the exponential shift, the per-modality schedules, which rows are pinned and where — is the
official MiniMax-H3 recipe, reproduced here so the plan can be built for pairs instead of single timesteps.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

#: Raw (unshifted) sigma grid the adapter was trained to sample on: 9 points, 8 model evaluations. Each modality
#: shifts it with its own scheduler shift (12.0 video, 3.0 audio), exactly as the official pipeline shifts its
#: ``linspace(1, 0, num_inference_steps)`` grid.
DEFAULT_SIGMAS_8STEP: tuple[float, ...] = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)

#: The shifts the adapter was trained with. They are read from the official schedulers at run time; these constants
#: only exist so the loader can warn when a pipeline was configured differently.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def validate_sigmas(sigmas: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Return ``sigmas`` as a float32 CPU tensor after checking it is a valid rectified-flow grid.

    The grid must be strictly decreasing, start at or below ``1.0`` and end at exactly ``0.0`` — the same contract
    ``MiniMaxH3Scheduler.set_timesteps(sigmas=...)`` enforces.
    """
    grid = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
    if grid.numel() < 2:
        raise ValueError(f"A sigma grid needs at least two points, got {grid.numel()}.")
    if not bool((grid[1:] < grid[:-1]).all()):
        raise ValueError(f"The sigma grid must be strictly decreasing, got {grid.tolist()}.")
    if grid[0].item() > 1.0 or grid[-1].item() != 0.0:
        raise ValueError(f"The sigma grid must start at or below 1.0 and end at exactly 0.0, got {grid.tolist()}.")
    return grid


def shift_sigmas(sigmas: Sequence[float] | torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the exponential shift ``s * sigma / (1 + (s - 1) * sigma)``, the formula ``MiniMaxH3Scheduler`` uses.

    The shift maps ``0`` to ``0`` and ``1`` to ``1``, so a valid grid stays valid.
    """
    if shift <= 0:
        raise ValueError(f"`shift` must be positive, got {shift}.")
    base = validate_sigmas(sigmas)
    return shift * base / (1.0 + (shift - 1.0) * base)


def endpoints_from_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    """The endpoint of every step: ``r_i = 1 - sigma_{i+1}`` on the ``t = 1 - sigma`` axis.

    Pairs with ``MiniMaxH3Scheduler.timesteps == 1 - sigmas[:-1]``: step ``i`` moves the generated rows from
    ``t_i = 1 - sigma_i`` to ``r_i = 1 - sigma_{i+1}``.
    """
    return 1.0 - sigmas[1:]


def build_row_time_pairs(
    *,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_timestep: float,
    video_endpoint: float,
    audio_timestep: float,
    audio_endpoint: float,
    condition_video_timestep: float,
    condition_audio_timestep: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign a ``(t, r)`` pair to every row of the packed sequence and reduce it to the transformer's inputs.

    This is the official ``MiniMaxH3SetTimestepsStep.build_row_timesteps`` extended with the endpoint. The pin rule
    is unchanged:

    * generated video rows step ``(t_video, r_video)``; text rows never reach an output head and follow video;
    * generated audio rows step ``(t_audio, r_audio)`` on their own schedule;
    * visual conditioning rows — ``fl2va`` keyframes, ``ref2va`` image and video references — are held at the
      noise-augmentation level, and an anchor has nowhere to go, so ``r == t`` there;
    * ``ref2va`` audio reference rows are never noised: ``t == r == 1.0``.

    Returns:
        ``(timestep, endpoint, indices)``: the distinct ``(t, r)`` pairs (sorted, as two ``(num_pairs,)`` float32
        tensors) and, for every row, the index of its pair. ``timestep`` and ``indices`` are what the official
        transformer forward takes; ``endpoint`` is handed to the [`TwoTimeEmbedder`].
    """
    video_indices = video_indices.cpu()
    audio_indices = audio_indices.cpu()
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)

    timestep = torch.full((sequence_length,), float(video_timestep), dtype=torch.float32)
    endpoint = torch.full((sequence_length,), float(video_endpoint), dtype=torch.float32)

    condition_video = video_indices[:num_condition_video_rows]
    timestep[condition_video] = float(condition_video_timestep)
    endpoint[condition_video] = float(condition_video_timestep)

    generated_audio = audio_indices[num_condition_audio_rows:]
    timestep[generated_audio] = float(audio_timestep)
    endpoint[generated_audio] = float(audio_endpoint)

    condition_audio = audio_indices[:num_condition_audio_rows]
    timestep[condition_audio] = float(condition_audio_timestep)
    endpoint[condition_audio] = float(condition_audio_timestep)

    pairs = torch.stack((timestep, endpoint), dim=-1)
    unique_pairs, indices = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    return unique_pairs[:, 0].contiguous(), unique_pairs[:, 1].contiguous(), indices


__all__ = [
    "AUDIO_SHIFT",
    "DEFAULT_SIGMAS_8STEP",
    "VIDEO_SHIFT",
    "build_row_time_pairs",
    "endpoints_from_sigmas",
    "shift_sigmas",
    "validate_sigmas",
]
