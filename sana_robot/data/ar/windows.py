# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from sana_robot.data.base.windows import minimum_real_rows


def window_chunk_count(episode_rows: int, head_rows: int, chunk_size: int, video_frame_stride: int) -> int:
    """Return the chunks of the window that opens ``head_rows`` rows before an episode and runs to its end, a chunk being kept while it has a real first target frame; 0 when the episode is too short. The first chunk of a window with head rows only fills the memory, so such a window needs a second chunk."""
    reach = int(head_rows) + int(episode_rows) - minimum_real_rows(video_frame_stride)
    if reach < 0:
        return 0
    chunks = 1 + reach // int(chunk_size)
    return chunks if chunks >= (2 if head_rows else 1) else 0


def build_windows(episode_rows, chunk_size: int, video_frame_stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the (episode, head rows, chunks) of every window: each episode once per phase, with 0 to ``chunk_size`` - 1 head rows in front of its first row, so that every row starts a chunk in one of its windows."""
    episodes, heads, lengths = [], [], []
    for episode, rows in enumerate(episode_rows):
        for head_rows in range(int(chunk_size)):
            chunks = window_chunk_count(rows, head_rows, chunk_size, video_frame_stride)
            if chunks:
                episodes.append(episode)
                heads.append(head_rows)
                lengths.append(chunks)
    return (
        np.asarray(episodes, dtype=np.int64),
        np.asarray(heads, dtype=np.int64),
        np.asarray(lengths, dtype=np.int64),
    )


def supervised_chunks(head_rows: int, chunks: int) -> np.ndarray:
    """Return which chunks of a window are trained: all of them, except the first chunk of a window with head rows."""
    supervised = np.ones(int(chunks), dtype=bool)
    supervised[0] = int(head_rows) == 0
    return supervised
