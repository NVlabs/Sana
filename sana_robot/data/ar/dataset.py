# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.data.ar.packing.sampler import PackedWindowSampler
from sana_robot.data.ar.teacher_forcing import with_state_rows
from sana_robot.data.ar.windows import build_windows, supervised_chunks
from sana_robot.data.base.dataset import (
    collate_samples,
    normalized_robot_rows,
    read_window_frames,
    view_layout,
    window_prompts,
)
from sana_robot.data.base.windows import window_frame_rows, window_robot_rows
from sana_robot.data.format.episode import task_ids

CHUNK_FIELDS = ("chunk_states", "chunk_states_mask", "chunk_supervised")
ROW_FIELDS = ("action", "action_mask", "action_loss_mask")


def pad_after(values: torch.Tensor, count: int, dim: int = 0) -> torch.Tensor:
    """Return a tensor with ``count`` zero entries appended along ``dim``."""
    shape = list(values.shape)
    shape[dim] = int(count)
    return torch.cat((values, values.new_zeros(shape)), dim=dim)


def pad_window(sample: dict, chunks: int, chunk_size: int) -> dict:
    """Return a window right-padded to ``chunks`` chunks: its frames repeat the last one, its latents and robot rows are zero, and the padded chunks feed no slot and carry no loss."""
    extra = int(chunks) - int(sample["teacher_forcing_num_chunks"])
    if extra == 0:
        return sample
    padded = dict(sample)
    padded["teacher_forcing_num_chunks"] = int(chunks)
    for name in CHUNK_FIELDS:
        padded[name] = pad_after(sample[name], extra)
    for name in ROW_FIELDS:
        padded[name] = pad_after(sample[name], extra * (int(chunk_size) + 1))
    if "frames" in sample:
        frames_per_chunk = (sample["frames"][0].shape[0] - 1) // int(sample["teacher_forcing_num_chunks"])
        padded["frames"] = tuple(
            torch.cat((view, view[-1:].expand(extra * frames_per_chunk, -1, -1, -1))) for view in sample["frames"]
        )
        return padded
    padded["window_latents"] = pad_after(sample["window_latents"], extra, dim=1)
    padded["observation_latents"] = pad_after(sample["observation_latents"], extra, dim=1)
    return padded


class EpisodeWindowDataset(torch.utils.data.Dataset):
    """Whole-episode windows of a dataset for SANA-Robot-AR.

    A window is an episode from its first frame to its end, cut into chunks of ``chunk_size`` rows. Every episode
    gives one window per phase: with h = 0 .. ``chunk_size`` - 1 head rows in front of its first row, so that
    every row starts a chunk in one window. The head rows repeat the first frame and hold the first state; the
    first chunk of a window with head rows only fills the memory and is not trained. The rows after the last row
    repeat the last frame and the last action and are left out of the action loss.

    A sample is a dict with ``frames`` (one uint8 tensor [1 + L * frames per chunk, 3, H, W] per view) or, with a
    latent store, ``window_latents`` [C, 1 + L, tokens] and ``observation_latents`` [C, L - 1, tokens];
    ``prompts`` and ``unconditional_prompts``; ``initial_state`` and ``initial_state_condition_mask`` [80];
    ``chunk_states`` and ``chunk_states_mask`` [L - 1, 80]; ``action``, ``action_mask`` and ``action_loss_mask``
    [L * (chunk_size + 1) - 1, 80], the robot rows after the first state; ``chunk_supervised`` [L];
    ``teacher_forcing_num_chunks`` (L), ``view_latent_shape``, ``view_slot_ids``, ``model_fps`` and
    ``video_frame_stride``; and the ``window`` index.

    ``window_chunks`` and ``window_tasks`` give the length and the task of every window to the sampler.

    Args:
        storage: The dataset reader, a ``ZipArchiveStorage`` or a ``LeRobotStorage``.
        normalization: The normalization of the robot rows, from ``load_normalization``.
        view_sizes: The pixel size (height, width) of each view.
        chunk_size: The control rows of one action chunk.
        video_frame_stride: The control rows between two video frames.
        latent_store: A ``LatentStore`` to read precomputed latents from instead of decoding frames, or None.
        augment: A function that brings the stored frames of a window to the view sizes with random image
            changes, called as ``augment(frames, info, view_sizes)``; None for the plain resize.
    """

    def __init__(
        self,
        storage,
        normalization,
        view_sizes,
        chunk_size=32,
        video_frame_stride=4,
        latent_store=None,
        augment=None,
    ):
        self.storage = storage
        self.augment = augment
        self.normalization = normalization
        self.view_sizes = [tuple(size) for size in view_sizes]
        self.chunk_size = int(chunk_size)
        self.video_frame_stride = int(video_frame_stride)
        self.latent_store = latent_store
        episode_rows = [record.rows for record in storage.episodes]
        self.window_episode, self.window_head_rows, self.window_chunks = build_windows(
            episode_rows, self.chunk_size, self.video_frame_stride
        )
        self.window_tasks = task_ids(storage.episodes)[self.window_episode]

    def __len__(self) -> int:
        return len(self.window_chunks)

    def __getitem__(self, index: int) -> dict:
        episode, head_rows = int(self.window_episode[index]), int(self.window_head_rows[index])
        chunks = int(self.window_chunks[index])
        info, record = self.storage.info, self.storage.episodes[episode]
        state, action = self.storage.read_rows(record)
        states, actions, real_rows = window_robot_rows(state, action, -head_rows, chunks, self.chunk_size)
        robot = normalized_robot_rows(states, actions, real_rows, info, self.normalization)
        supervised = torch.from_numpy(supervised_chunks(head_rows, chunks))
        loss_mask = robot["real_action_mask"] & supervised.repeat_interleave(self.chunk_size)[:, None]
        sample = {
            "initial_state": robot["states"][0],
            "initial_state_condition_mask": robot["state_mask"][0],
            "chunk_states": robot["states"][1:],
            "chunk_states_mask": robot["state_mask"][1:],
            "action": with_state_rows(robot["actions"], chunks, self.chunk_size),
            "action_mask": with_state_rows(robot["action_mask"], chunks, self.chunk_size),
            "action_loss_mask": with_state_rows(loss_mask, chunks, self.chunk_size),
            "chunk_supervised": supervised,
            "teacher_forcing_num_chunks": chunks,
            "model_fps": float(info.fps),
            "video_frame_stride": self.video_frame_stride,
            "window": int(index),
            **view_layout(info, self.view_sizes),
            **window_prompts(info, record),
        }
        if self.latent_store is not None:
            sample["window_latents"], sample["observation_latents"] = self.latent_store.window_latents(
                record.name, record.rows, head_rows, chunks
            )
            return sample
        frame_rows = window_frame_rows(-head_rows, chunks, self.chunk_size, self.video_frame_stride, record.rows)
        sample["frames"] = read_window_frames(self.storage, record, frame_rows, self.view_sizes, self.augment)
        return sample

    def collate(self, samples) -> dict:
        """Return a batch of windows right-padded to its longest window, with ``chunk_counts``, the chunks of every window before the padding."""
        counts = [int(sample["teacher_forcing_num_chunks"]) for sample in samples]
        batch = collate_samples([pad_window(sample, max(counts), self.chunk_size) for sample in samples])
        batch["teacher_forcing_num_chunks"] = max(counts)
        batch["chunk_counts"] = torch.tensor(counts, dtype=torch.long)
        return batch


def window_dataloader(
    dataset, chunk_budget: int, world_size: int = 1, rank: int = 0, seed: int = 0, num_workers: int = 4
):
    """Build the training loader of SANA-Robot-AR: whole-episode windows in the task-balanced order, packed to a chunk budget per rank.

    Args:
        dataset: An ``EpisodeWindowDataset``.
        chunk_budget: The most chunks one rank holds in a step; a longer window takes a rank alone.
        world_size: The number of ranks.
        rank: The rank this loader serves.
        seed: The seed of the orders of all epochs.
        num_workers: The worker processes that read the windows.

    Returns:
        The loader; its ``batch_sampler`` is a ``PackedWindowSampler``, whose ``set_epoch`` selects the epoch and the
        step to resume from.
    """
    sampler = PackedWindowSampler(dataset.window_chunks, dataset.window_tasks, chunk_budget, world_size, rank, seed)
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=dataset.collate, num_workers=num_workers, pin_memory=True
    )
