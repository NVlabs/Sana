# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.data.ar.teacher_forcing import strip_view_latents
from sana_robot.data.format.video import frames_to_model_range
from sana_robot.train.flow_matching.loss import chunk_action_losses, chunk_video_losses, mean_over_trained_chunks
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
    "view_count",
    "view_latent_shape",
    "view_slot_ids",
    "model_fps",
    "video_frame_stride",
    "initial_state",
    "initial_state_condition_mask",
    "action_mask",
)


@torch.no_grad()
def encode_chunk_latents(batch: dict, encode, device) -> torch.Tensor:
    """Return the clean latents of a batch of one-chunk windows [B, C, 1 + f, tokens], the observation first: read from the batch or encoded from its frames."""
    if "latents" in batch:
        return batch["latents"].to(device).float()
    views = [encode(frames_to_model_range(frames.to(device)).transpose(1, 2)) for frames in batch["frames"]]
    return strip_view_latents(views)


def draw_chunk_noise(latents: torch.Tensor, chunk_size: int, slots: int) -> FlowNoise:
    """Draw the timestep and the noise of a training step for a batch of one-chunk windows."""
    batch, device = latents.shape[0], latents.device
    timesteps = sample_chunk_timesteps(batch, 1, device)
    video = torch.randn_like(latents[:, :, 1:])
    action = torch.randn(batch, int(chunk_size), int(slots), device=device)
    return FlowNoise(timesteps, video, action)


def chunk_flow_inputs(batch: dict, latents: torch.Tensor, noise: FlowNoise) -> tuple[dict, dict]:
    """Build the policy inputs and the flow targets of a training step.

    Args:
        batch: A batch of ``ChunkDataset`` on the training device.
        latents: The clean latents of every window [B, C, 1 + f, tokens], the observation first.
        noise: The draws of the step, one timestep per window.

    Returns:
        The keyword arguments of ``SanaRobotBase.forward`` without the text, and a dict with the flow targets
        ``video`` [B, C, f, tokens] and ``action`` [B, A, 80] and the loss weights ``video_weights`` [B, f] and
        ``action_weights`` [B, A].
    """
    targets_per_window = latents.shape[2] - 1
    chunk_size = noise.action.shape[1]
    frame_timesteps = noise.timesteps.expand(-1, targets_per_window)
    row_timesteps = noise.timesteps.expand(-1, chunk_size)

    video_sigmas = shifted_sigmas(frame_timesteps, VIDEO_FLOW_SHIFT)[:, None, :, None]
    noisy_video, video_target = add_noise(latents[:, :, 1:], noise.video, video_sigmas)
    noisy_latents = torch.cat((latents[:, :, :1], noisy_video), dim=2)
    view_shapes = batch["view_latent_shape"][0].tolist()
    if len(view_shapes) == 1:
        noisy_latents = noisy_latents.unflatten(-1, tuple(view_shapes[0]))
    else:
        noisy_latents = noisy_latents.unsqueeze(3)

    action_sigmas = shifted_sigmas(row_timesteps, ACTION_FLOW_SHIFT)[:, :, None]
    noisy_action, action_target = add_noise(batch["action"], noise.action, action_sigmas)
    noisy_action = torch.where(batch["action_mask"], noisy_action, torch.zeros_like(noisy_action))

    data_info = {name: batch[name] for name in POLICY_FIELDS}
    data_info["action"] = noisy_action
    data_info["action_timestep"] = model_timesteps(row_timesteps, ACTION_FLOW_SHIFT)
    video_timesteps = model_timesteps(frame_timesteps, VIDEO_FLOW_SHIFT)
    inputs = {
        "video_latents": noisy_latents,
        "timestep": torch.cat((torch.zeros_like(video_timesteps[:, :1]), video_timesteps), dim=1),
        "data_info": data_info,
    }
    targets = {
        "video": video_target,
        "action": action_target,
        "video_weights": timestep_weights(frame_timesteps, VIDEO_FLOW_SHIFT),
        "action_weights": timestep_weights(row_timesteps, ACTION_FLOW_SHIFT),
    }
    return inputs, targets


def chunk_losses(output: dict, targets: dict, batch: dict, action_loss_weight: float) -> dict:
    """Compute the training loss of a batch of one-chunk windows from the policy's output.

    Args:
        output: The output of ``SanaRobotBase.forward``.
        targets: The targets from ``chunk_flow_inputs``.
        batch: The batch of the step, for ``action_loss_mask``.
        action_loss_weight: The weight of the action loss in the total.

    Returns:
        A dict with ``loss``, the mean over the windows of the batch, and its parts ``video_loss`` and
        ``action_loss``.
    """
    video_prediction = output["video_pred"].flatten(3)[:, :, 1:]
    valid = torch.ones_like(targets["video_weights"], dtype=torch.bool)
    video_losses, _ = chunk_video_losses(video_prediction, targets["video"], targets["video_weights"], valid, 1)
    action_losses, _ = chunk_action_losses(
        output["action_pred"], targets["action"], targets["action_weights"], batch["action_loss_mask"], 1
    )
    trained = torch.ones_like(video_losses, dtype=torch.bool)
    video_loss = mean_over_trained_chunks(video_losses, trained)
    action_loss = mean_over_trained_chunks(action_losses, trained)
    return {"loss": video_loss + action_loss_weight * action_loss, "video_loss": video_loss, "action_loss": action_loss}


def chunk_training_loss(
    model,
    batch: dict,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    text_mask: torch.Tensor,
    noise: FlowNoise,
    action_loss_weight: float,
) -> dict:
    """Run the policy on a noised batch of one-chunk windows and compute the loss to backpropagate.

    Args:
        model: ``SanaRobotBase``.
        batch: A batch of ``ChunkDataset`` on the training device.
        latents: The clean latents of every window [B, C, 1 + f, tokens], the observation first.
        text_embeddings: The text embeddings [B, V + 1, 1, L_text, C_text].
        text_mask: The text token mask [B, V + 1, L_text].
        noise: The draws of the step.
        action_loss_weight: The weight of the action loss in the total.

    Returns:
        The dict of ``chunk_losses`` with ``scaled_loss``, the loss itself: every rank holds the same number of
        windows.
    """
    inputs, targets = chunk_flow_inputs(batch, latents, noise)
    output = model(text_embeddings=text_embeddings, text_mask=text_mask, **inputs)
    losses = chunk_losses(output, targets, batch, action_loss_weight)
    losses["scaled_loss"] = losses["loss"]
    return losses
