# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np


def minimum_real_rows(video_frame_stride: int) -> int:
    """Return the real rows a chunk needs from its first row on: enough for its first target frame, one video stride after its observation."""
    return 1 + int(video_frame_stride)


def build_chunk_windows(episode_rows, video_frame_stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the (episode, first row) of every one-chunk window: a chunk starts at every row of an episode that leaves it a real first target frame, so the chunks at the end of an episode run past its last row."""
    episodes, first_rows = [], []
    for episode, rows in enumerate(episode_rows):
        count = max(0, int(rows) - minimum_real_rows(video_frame_stride) + 1)
        episodes.append(np.full(count, episode, dtype=np.int64))
        first_rows.append(np.arange(count, dtype=np.int64))
    return np.concatenate(episodes), np.concatenate(first_rows)


def window_frame_rows(
    first_row: int, chunks: int, chunk_size: int, video_frame_stride: int, episode_rows: int
) -> np.ndarray:
    """Return the episode row shown by every frame of a window of chunks that starts at ``first_row``: one frame per video stride, the first row repeated before the episode starts and the last row after it ends."""
    rows = int(first_row) + np.arange(0, int(chunks) * int(chunk_size) + 1, int(video_frame_stride))
    return np.clip(rows, 0, int(episode_rows) - 1)


def window_robot_rows(state: np.ndarray, action: np.ndarray, first_row: int, chunks: int, chunk_size: int) -> tuple:
    """Return the robot rows of a window of chunks that starts at ``first_row`` of an episode: the state at the first row of every chunk [L, 80], the action of every row [L * C, 80] and which of these rows are real. A row before the episode starts holds the robot at its first state and a row after the episode ends repeats the last action; neither is real."""
    rows = int(first_row) + np.arange(int(chunks) * int(chunk_size))
    inside = np.clip(rows, 0, len(state) - 1)
    actions = np.where((rows < 0)[:, None], state[0], action[inside])
    real = (rows >= 0) & (rows < len(state))
    return state[inside[:: int(chunk_size)]], actions, real
