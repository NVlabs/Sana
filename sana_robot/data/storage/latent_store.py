# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from sana_robot.data.ar.windows import window_chunk_count

LATENT_FORMAT = "sana_robot_latents_v1"
INFO_FILE = "meta.json"
OBSERVATIONS_FILE = "observations.npy"
BASE_TARGETS_FILE = "base_targets.npy"
AR_TARGETS_FILE = "ar_targets.npy"


def write_store_info(root, view_sizes, chunk_size: int, video_frame_stride: int) -> None:
    """Write the description of a latent store: the pixel size of every view and the chunk it was encoded for."""
    content = {
        "format": LATENT_FORMAT,
        "view_sizes": [[int(height), int(width)] for height, width in view_sizes],
        "chunk_rows": int(chunk_size),
        "video_frame_stride": int(video_frame_stride),
    }
    Path(root).mkdir(parents=True, exist_ok=True)
    Path(root, INFO_FILE).write_text(json.dumps(content, indent=2) + "\n")


def write_episode_latents(root, episode_name: str, file_name: str, latents: torch.Tensor) -> None:
    """Store the latents [frames, C, tokens] of one episode as float16 under one of the store's file names."""
    directory = Path(root, episode_name)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / file_name, latents.to(torch.float16).cpu().numpy())


def ar_target_offsets(episode_rows: int, chunk_size: int, video_frame_stride: int) -> np.ndarray:
    """Return where the targets of each phase's window start in an episode's AR targets, which hold the windows of phases 0 to C - 1 one after another, one target per chunk."""
    chunks = [
        window_chunk_count(episode_rows, head_rows, chunk_size, video_frame_stride) for head_rows in range(chunk_size)
    ]
    return np.concatenate(([0], np.cumsum(chunks)))


class LatentStore:
    """Read the video latents that ``sana_robot/data/utils/build_latents.py`` encoded for a dataset.

    Every episode has a directory of float16 arrays [frames, C, tokens], a latent frame being the concatenation of
    its flattened views: ``observations.npy`` (every row that starts a chunk, encoded alone), ``base_targets.npy``
    (per one-chunk window, the target latent of the clip that starts at its row) and ``ar_targets.npy`` (per phase,
    the target latents of the whole-episode window).

    Args:
        root: The store directory.
    """

    def __init__(self, root):
        self.root = Path(root)
        content = json.loads((self.root / INFO_FILE).read_text())
        self.view_sizes = [tuple(size) for size in content["view_sizes"]]
        self.chunk_size = int(content["chunk_rows"])
        self.video_frame_stride = int(content["video_frame_stride"])

    def read(self, episode_name: str, file_name: str, frames) -> torch.Tensor:
        """Return the stored latents of an episode at the given frames as [C, n, tokens]."""
        stored = np.load(self.root / episode_name / file_name, mmap_mode="r")
        return torch.from_numpy(np.ascontiguousarray(stored[np.asarray(frames, dtype=np.int64)])).transpose(0, 1)

    def base_latents(self, episode_name: str, first_row: int) -> torch.Tensor:
        """Return the two latents [C, 2, tokens] of the one-chunk window that starts at ``first_row``: its observation, then its target."""
        observation = self.read(episode_name, OBSERVATIONS_FILE, [first_row])
        return torch.cat((observation, self.read(episode_name, BASE_TARGETS_FILE, [first_row])), dim=1)

    def window_latents(self, episode_name: str, episode_rows: int, head_rows: int, chunks: int) -> tuple:
        """Return the latents of the whole-episode window of a phase: the sink followed by the targets [C, 1 + L, tokens] and the observations of chunks 1 to L - 1 [C, L - 1, tokens]."""
        first = int(ar_target_offsets(episode_rows, self.chunk_size, self.video_frame_stride)[head_rows])
        targets = self.read(episode_name, AR_TARGETS_FILE, first + np.arange(chunks))
        sink = self.read(episode_name, OBSERVATIONS_FILE, [0])
        observation_rows = np.arange(1, chunks) * self.chunk_size - head_rows
        observations = self.read(episode_name, OBSERVATIONS_FILE, observation_rows)
        return torch.cat((sink, targets), dim=1), observations
