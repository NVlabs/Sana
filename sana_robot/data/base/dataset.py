# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

import numpy as np
import torch

from sana_robot.data.base.sampler import ChunkBatchSampler
from sana_robot.data.base.windows import build_chunk_windows, window_frame_rows, window_robot_rows
from sana_robot.data.format.normalization import normalize_rows
from sana_robot.data.format.prompt import build_prompts
from sana_robot.data.format.video import VAE_SPATIAL_STRIDE, model_frames


def read_window_frames(storage, record, frame_rows, view_sizes, augment=None) -> tuple:
    """Return the frames of a window as one uint8 tensor [T, 3, H, W] per view, read at the given episode rows and brought to the view's pixel size under the dataset's resize rule. ``augment`` takes that last step over: called as ``augment(frames, info, view_sizes)`` with the stored uint8 frames [T, H, W, 3] of every view and the dataset description, it returns the views at their sizes."""
    info = storage.info
    frames = [storage.read_frames(record, view.name, frame_rows) for view in info.views]
    if augment is not None:
        return tuple(augment(frames, info, view_sizes))
    return tuple(model_frames(stored, size, info.resize) for stored, size in zip(frames, view_sizes))


def window_prompts(info, record) -> dict:
    """Return the prompts of a window with one of the episode's instructions drawn at random, and the unconditional prompts without it."""
    descriptions = [view.description for view in info.views]
    instruction = random.choice(record.instructions)
    return {
        "prompts": build_prompts(info.embodiment, info.action_mode, descriptions, instruction),
        "unconditional_prompts": build_prompts(info.embodiment, info.action_mode, descriptions, None),
    }


def view_layout(info, view_sizes) -> dict:
    """Return what the policy reads about the views of a sample: the latent shape of each view and its slot in the spatial layout."""
    latent_shapes = [
        [int(height) // VAE_SPATIAL_STRIDE, int(width) // VAE_SPATIAL_STRIDE] for height, width in view_sizes
    ]
    return {
        "view_latent_shape": torch.tensor(latent_shapes, dtype=torch.long),
        "view_slot_ids": torch.tensor([view.slot for view in info.views], dtype=torch.long),
    }


def normalized_robot_rows(states, actions, real_rows, info, normalization) -> dict:
    """Return the normalized robot rows of a window as tensors: the chunk states [L, 80] and the actions [L * C, 80] with the masks of the slots the policy reads, and the mask of the real action rows."""
    state_mask = torch.from_numpy(info.state_mask).expand(len(states), -1).clone()
    action_mask = torch.from_numpy(info.action_mask).expand(len(actions), -1).clone()
    return {
        "states": normalize_rows(torch.from_numpy(np.ascontiguousarray(states)), state_mask, normalization),
        "state_mask": state_mask,
        "actions": normalize_rows(torch.from_numpy(np.ascontiguousarray(actions)), action_mask, normalization),
        "action_mask": action_mask,
        "real_action_mask": action_mask & torch.from_numpy(real_rows)[:, None],
    }


def collate_samples(samples) -> dict:
    """Return a batch of samples: tensors and numbers stacked, the frames stacked per view, and the prompts kept as one tuple per sample."""
    batch = {}
    for name, first in samples[0].items():
        values = [sample[name] for sample in samples]
        if torch.is_tensor(first):
            batch[name] = torch.stack(values)
        elif isinstance(first, tuple) and torch.is_tensor(first[0]):
            batch[name] = tuple(torch.stack(view) for view in zip(*values))
        elif isinstance(first, (bool, int, float)):
            batch[name] = torch.tensor(values)
        else:
            batch[name] = values
    return batch


class ChunkDataset(torch.utils.data.Dataset):
    """One-chunk windows of a dataset for SANA-Robot-base.

    A window starts at any row of an episode: the state of that row, the actions of the next ``chunk_size`` rows
    and the video frames of these rows at the video stride, the first of them being the observation. The windows
    near the end of an episode run past its last row; the missing rows repeat the last frame and the last action
    and are left out of the action loss.

    A sample is a dict with ``frames`` (one uint8 tensor [T, 3, H, W] per view) or, with a latent store,
    ``latents`` [C, 2, tokens]; ``prompts`` and ``unconditional_prompts`` (one text per view, then one for the
    robot rows); ``initial_state`` and ``initial_state_condition_mask`` [80]; ``action``, ``action_mask`` and
    ``action_loss_mask`` [chunk_size, 80]; ``view_count``, ``view_latent_shape``, ``view_slot_ids``,
    ``model_fps`` and ``video_frame_stride``; and the ``episode`` and ``first_row`` it came from.

    Args:
        storage: The dataset reader, a ``ZipArchiveStorage`` or a ``LeRobotStorage``.
        normalization: The normalization of the robot rows, from ``load_normalization``.
        view_sizes: The pixel size (height, width) of each view.
        chunk_size: The control rows of one action chunk.
        video_frame_stride: The control rows between two video frames.
        latent_store: A ``LatentStore`` to read precomputed latents from instead of decoding frames, or None.
    """

    def __init__(self, storage, normalization, view_sizes, chunk_size=32, video_frame_stride=4, latent_store=None):
        self.storage = storage
        self.normalization = normalization
        self.view_sizes = [tuple(size) for size in view_sizes]
        self.chunk_size = int(chunk_size)
        self.video_frame_stride = int(video_frame_stride)
        self.latent_store = latent_store
        episode_rows = [record.rows for record in storage.episodes]
        self.window_episode, self.window_first_row = build_chunk_windows(episode_rows, self.video_frame_stride)

    def __len__(self) -> int:
        return len(self.window_episode)

    def __getitem__(self, index: int) -> dict:
        episode, first_row = int(self.window_episode[index]), int(self.window_first_row[index])
        info, record = self.storage.info, self.storage.episodes[episode]
        state, action = self.storage.read_rows(record)
        states, actions, real_rows = window_robot_rows(state, action, first_row, 1, self.chunk_size)
        robot = normalized_robot_rows(states, actions, real_rows, info, self.normalization)
        sample = {
            "initial_state": robot["states"][0],
            "initial_state_condition_mask": robot["state_mask"][0],
            "action": robot["actions"],
            "action_mask": robot["action_mask"],
            "action_loss_mask": robot["real_action_mask"],
            "view_count": len(info.views),
            "model_fps": float(info.fps),
            "video_frame_stride": self.video_frame_stride,
            "episode": episode,
            "first_row": first_row,
            **view_layout(info, self.view_sizes),
            **window_prompts(info, record),
        }
        if self.latent_store is not None:
            sample["latents"] = self.latent_store.base_latents(record.name, first_row)
            return sample
        frame_rows = window_frame_rows(first_row, 1, self.chunk_size, self.video_frame_stride, record.rows)
        sample["frames"] = read_window_frames(self.storage, record, frame_rows, self.view_sizes)
        return sample


def chunk_dataloader(dataset, batch_size: int, world_size: int = 1, rank: int = 0, seed: int = 0, num_workers: int = 4):
    """Build the training loader of SANA-Robot-base: the one-chunk windows in a random order, split over the ranks.

    Args:
        dataset: A ``ChunkDataset``.
        batch_size: The windows of one batch on one rank.
        world_size: The number of ranks.
        rank: The rank this loader serves.
        seed: The seed of the orders of all epochs.
        num_workers: The worker processes that read the windows.

    Returns:
        The loader; its ``batch_sampler`` is a ``ChunkBatchSampler``, whose ``set_epoch`` selects the epoch and the
        step to resume from.
    """
    sampler = ChunkBatchSampler(len(dataset), batch_size, world_size, rank, seed)
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_samples, num_workers=num_workers, pin_memory=True
    )
