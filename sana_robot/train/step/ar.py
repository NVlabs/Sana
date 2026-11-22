# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.data.ar.teacher_forcing import (
    action_row_positions,
    assemble_teacher_forcing_latents,
    encode_teacher_forcing_latents,
    strip_view_latents,
    target_frame_positions,
    teacher_forcing_action_timesteps,
    teacher_forcing_video_timesteps,
    with_state_rows,
)
from sana_robot.data.format.video import VAE_TEMPORAL_STRIDE, frames_to_model_range
from sana_robot.train.flow_matching.loss import (
    chunk_action_losses,
    chunk_video_losses,
    global_chunk_scale,
    mean_over_trained_chunks,
)
from sana_robot.train.flow_matching.noise import FlowNoise, add_noise
from sana_robot.train.flow_matching.timesteps import (
    ACTION_FLOW_SHIFT,
    VIDEO_FLOW_SHIFT,
    model_timesteps,
    sample_chunk_timesteps,
    shifted_sigmas,
)
from sana_robot.train.flow_matching.weights import timestep_weights

POLICY_FIELDS = (
    "view_latent_shape",
    "view_slot_ids",
    "model_fps",
    "video_frame_stride",
    "initial_state",
    "initial_state_condition_mask",
    "chunk_states",
    "chunk_states_mask",
    "action_mask",
    "chunk_counts",
)


def pad_chunks(latents: torch.Tensor, chunks: int) -> torch.Tensor:
    """Return latents [1, C, F, tokens] with zero frames appended for ``chunks`` more chunks."""
    return torch.cat((latents, latents.new_zeros(*latents.shape[:2], int(chunks), latents.shape[3])), dim=2)


@torch.no_grad()
def encode_window_latents(batch: dict, encode, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the clean latents of a batch of windows: the causal encoding of every window [B, C, 1 + L * f, tokens] and its observations [B, C, L - 1, tokens], read from the batch or encoded from each window's own frames."""
    if "window_latents" in batch:
        return batch["window_latents"].to(device).float(), batch["observation_latents"].to(device).float()
    longest = int(batch["teacher_forcing_num_chunks"])
    frames_per_chunk = (batch["frames"][0].shape[1] - 1) // longest
    windows, observations = [], []
    for row, count in enumerate(batch["chunk_counts"].tolist()):
        views = []
        for frames in batch["frames"]:
            clip = frames_to_model_range(frames[row : row + 1, : 1 + frames_per_chunk * count].to(device))
            views.append(encode_teacher_forcing_latents(encode, clip, count))
        window = strip_view_latents([view[0] for view in views])
        observation = strip_view_latents([view[1] for view in views])
        windows.append(pad_chunks(window, (longest - count) * (frames_per_chunk // VAE_TEMPORAL_STRIDE)))
        observations.append(pad_chunks(observation, longest - count))
    return torch.cat(windows), torch.cat(observations)


def draw_window_noise(window_latents: torch.Tensor, chunks: int, chunk_size: int, slots: int) -> FlowNoise:
    """Draw the timesteps and the noise of a training step for a batch of windows of ``chunks`` chunks."""
    batch, device = window_latents.shape[0], window_latents.device
    timesteps = sample_chunk_timesteps(batch, int(chunks), device)
    video = torch.randn_like(window_latents[:, :, 1:])
    action = torch.randn(batch, int(chunks) * int(chunk_size), int(slots), device=device)
    return FlowNoise(timesteps, video, action)


def window_flow_inputs(
    batch: dict,
    window_latents: torch.Tensor,
    observation_latents: torch.Tensor,
    noise: FlowNoise,
    timestep_weighting: bool = True,
) -> tuple[dict, dict]:
    """Build the policy inputs and the flow targets of a training step.

    Args:
        batch: A batch of ``EpisodeWindowDataset`` on the training device.
        window_latents: The causal encoding of every window [B, C, 1 + L * f, tokens].
        observation_latents: The observations of chunks 1 to L - 1 [B, C, L - 1, tokens].
        noise: The draws of the step.
        timestep_weighting: Weigh the loss of a chunk by its timestep; every chunk weighs 1 when off.

    Returns:
        The keyword arguments of ``SanaRobotAR.forward`` without the text, and a dict with the flow targets
        ``video`` [B, C, L * f, tokens] and ``action`` [B, L * A, 80], the loss weights ``video_weights`` [B, L * f]
        and ``action_weights`` [B, L * A], and ``video_positions`` and ``action_positions``, where the targets sit
        in the policy's outputs.
    """
    chunks = int(batch["teacher_forcing_num_chunks"])
    per_chunk = (window_latents.shape[2] - 1) // chunks
    chunk_size = noise.action.shape[1] // chunks
    device = window_latents.device
    video_positions = target_frame_positions(chunks, per_chunk).to(device)
    action_positions = action_row_positions(chunks, chunk_size).to(device)
    frame_timesteps = noise.timesteps.repeat_interleave(per_chunk, dim=1)
    row_timesteps = noise.timesteps.repeat_interleave(chunk_size, dim=1)

    video_sigmas = shifted_sigmas(frame_timesteps, VIDEO_FLOW_SHIFT)[:, None, :, None]
    noisy_video, video_target = add_noise(window_latents[:, :, 1:], noise.video, video_sigmas)
    latents = assemble_teacher_forcing_latents(window_latents, observation_latents, chunks)
    latents[:, :, video_positions] = noisy_video

    action_sigmas = shifted_sigmas(row_timesteps, ACTION_FLOW_SHIFT)[:, :, None]
    noisy_action, action_target = add_noise(batch["action"][:, action_positions], noise.action, action_sigmas)
    read_slots = batch["action_mask"][:, action_positions]
    noisy_action = torch.where(read_slots, noisy_action, torch.zeros_like(noisy_action))

    data_info = {name: batch[name] for name in POLICY_FIELDS}
    data_info["teacher_forcing_num_chunks"] = chunks
    data_info["action"] = with_state_rows(noisy_action, chunks, chunk_size)
    data_info["action_timestep"] = teacher_forcing_action_timesteps(
        model_timesteps(noise.timesteps, ACTION_FLOW_SHIFT), chunk_size
    )
    video_weights = timestep_weights(frame_timesteps, VIDEO_FLOW_SHIFT)
    action_weights = timestep_weights(row_timesteps, ACTION_FLOW_SHIFT)
    if not timestep_weighting:
        video_weights, action_weights = torch.ones_like(video_weights), torch.ones_like(action_weights)
    inputs = {
        "video_latents": latents.unsqueeze(3),
        "timestep": teacher_forcing_video_timesteps(model_timesteps(noise.timesteps, VIDEO_FLOW_SHIFT), per_chunk),
        "data_info": data_info,
    }
    targets = {
        "video": video_target,
        "action": action_target,
        "video_weights": video_weights,
        "action_weights": action_weights,
        "video_positions": video_positions,
        "action_positions": action_positions,
    }
    return inputs, targets


def window_losses(output: dict, targets: dict, batch: dict, action_loss_weight: float) -> dict:
    """Compute the training loss of a batch of windows from the policy's output.

    Args:
        output: The output of ``SanaRobotAR.forward``.
        targets: The targets from ``window_flow_inputs``.
        batch: The batch of the step, for ``chunk_supervised`` and ``action_loss_mask``.
        action_loss_weight: The weight of the action loss in the total.

    Returns:
        A dict with ``loss``, the mean over the trained chunks of the batch, its parts ``video_loss`` and
        ``action_loss``, and ``trained``, the trained chunks [B, L].
    """
    trained = batch["chunk_supervised"].bool()
    chunks = trained.shape[1]
    per_chunk = targets["video"].shape[2] // chunks
    video_prediction = output["video_pred"].squeeze(3)[:, :, targets["video_positions"]]
    action_prediction = output["action_pred"][:, targets["action_positions"]]
    video_losses, video_counts = chunk_video_losses(
        video_prediction,
        targets["video"],
        targets["video_weights"],
        trained.repeat_interleave(per_chunk, dim=1),
        chunks,
    )
    action_losses, action_counts = chunk_action_losses(
        action_prediction,
        targets["action"],
        targets["action_weights"],
        batch["action_loss_mask"][:, targets["action_positions"]],
        chunks,
    )
    video_loss = mean_over_trained_chunks(video_losses, trained & (video_counts > 0))
    action_loss = mean_over_trained_chunks(action_losses, trained & (action_counts > 0))
    return {
        "loss": video_loss + action_loss_weight * action_loss,
        "video_loss": video_loss,
        "action_loss": action_loss,
        "trained": trained,
    }


def window_training_loss(
    model,
    batch: dict,
    window_latents: torch.Tensor,
    observation_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    text_mask: torch.Tensor,
    noise: FlowNoise,
    action_loss_weight: float,
    world_size: int,
    timestep_weighting: bool = True,
) -> dict:
    """Run the policy on a noised batch of windows and compute the loss to backpropagate on this rank.

    Args:
        model: ``SanaRobotAR``.
        batch: A batch of ``EpisodeWindowDataset`` on the training device.
        window_latents: The causal encoding of every window [B, C, 1 + L * f, tokens].
        observation_latents: The observations of chunks 1 to L - 1 [B, C, L - 1, tokens].
        text_embeddings: The text embeddings [B, V + 1, 1, L_text, C_text].
        text_mask: The text token mask [B, V + 1, L_text].
        noise: The draws of the step.
        action_loss_weight: The weight of the action loss in the total.
        world_size: The number of ranks.
        timestep_weighting: Weigh the loss of a chunk by its timestep; every chunk weighs 1 when off.

    Returns:
        The dict of ``window_losses`` with ``scaled_loss`` added: the loss times this rank's share of the trained
        chunks of all ranks, so that every trained chunk of the step weighs the same once the ranks' gradients are
        averaged.
    """
    inputs, targets = window_flow_inputs(batch, window_latents, observation_latents, noise, timestep_weighting)
    scale = global_chunk_scale(batch["chunk_supervised"], world_size)
    output = model(text_embeddings=text_embeddings, text_mask=text_mask, **inputs)
    losses = window_losses(output, targets, batch, action_loss_weight)
    losses["scaled_loss"] = losses["loss"] * scale
    return losses
