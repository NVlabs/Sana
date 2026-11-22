# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import numpy as np
import torch

from sana_robot.data.ar.teacher_forcing import bucket_padding_chunks, strip_view_latents
from sana_robot.data.ar.windows import window_chunk_count
from sana_robot.data.base.dataset import read_window_frames
from sana_robot.data.base.windows import minimum_real_rows, window_frame_rows
from sana_robot.data.format.video import VAE_TEMPORAL_STRIDE, frames_to_model_range
from sana_robot.data.storage import open_storage
from sana_robot.data.storage.latent_store import (
    AR_TARGETS_FILE,
    BASE_TARGETS_FILE,
    OBSERVATIONS_FILE,
    write_episode_latents,
    write_store_info,
)
from sana_robot.utils.vae import encode_in_batches, load_video_encoder


def encode_clips(encode, view_frames, clip_rows: np.ndarray, batch_size: int, device) -> torch.Tensor:
    """Return the latents [clips, C, F, tokens] of clips of an episode, ``clip_rows`` [clips, T] naming the rows of every clip: each view is encoded in batches and the views are joined into one strip."""
    view_latents = []
    for frames in view_frames:
        batches = []
        for start in range(0, len(clip_rows), batch_size):
            clips = frames[torch.from_numpy(clip_rows[start : start + batch_size])]
            clips = frames_to_model_range(clips.to(device)).transpose(1, 2)
            batches.append(encode_in_batches(encode, clips, batch_size).float().cpu())
        view_latents.append(torch.cat(batches))
    return strip_view_latents(view_latents)


def encode_episode(
    encode, view_frames, episode_rows, chunk_size, video_frame_stride, modes, batch_size, device
) -> dict:
    """Return the latents of one episode by file name of the latent store, every array [frames, C, tokens]: the observations, and for each mode in ``modes`` its targets."""
    starts = np.arange(max(0, episode_rows - minimum_real_rows(video_frame_stride) + 1))
    latents = {OBSERVATIONS_FILE: encode_clips(encode, view_frames, starts[:, None], batch_size, device)[:, :, 0]}
    if "base" in modes:
        clip_rows = np.stack(
            [window_frame_rows(start, 1, chunk_size, video_frame_stride, episode_rows) for start in starts]
        )
        latents[BASE_TARGETS_FILE] = encode_clips(encode, view_frames, clip_rows, batch_size, device)[:, :, 1]
    if "ar" in modes:
        frames_per_chunk = chunk_size // video_frame_stride
        targets = []
        for head_rows in range(chunk_size):
            chunks = window_chunk_count(episode_rows, head_rows, chunk_size, video_frame_stride)
            if chunks:
                clip_rows = window_frame_rows(-head_rows, chunks, chunk_size, video_frame_stride, episode_rows)
                filler = np.repeat(clip_rows[-1:], bucket_padding_chunks(chunks) * frames_per_chunk)
                window = encode_clips(encode, view_frames, np.concatenate((clip_rows, filler))[None], 1, device)[0]
                targets.append(window[:, 1 : 1 + chunks * (frames_per_chunk // VAE_TEMPORAL_STRIDE)].transpose(0, 1))
        latents[AR_TARGETS_FILE] = torch.cat(targets)
    return latents


def build_latents(
    dataset,
    output,
    vae,
    view_sizes,
    modes=("base", "ar"),
    lerobot_mapping=None,
    chunk_size: int = 32,
    video_frame_stride: int = 4,
    first_episode: int = 0,
    episode_count: int | None = None,
    batch_size: int = 64,
    device: str = "cuda",
) -> None:
    """Encode the video latents of a dataset into a latent store, so that training reads latents instead of decoding and encoding frames.

    Per episode the store keeps the observation latent of every row that starts a chunk (the row's frame encoded
    alone), for ``base`` the target latent of every one-chunk window and for ``ar`` the target latents of the
    whole-episode window of every phase.

    Args:
        dataset: The dataset directory, zip archives or LeRobot.
        output: The latent store directory to create.
        vae: The directory of the video VAE weights.
        view_sizes: The pixel size (height, width) of each view.
        modes: Which targets to encode: ``base``, ``ar`` or both.
        lerobot_mapping: The column mapping of a LeRobot dataset, or its JSON file.
        chunk_size: The control rows of one action chunk.
        video_frame_stride: The control rows between two video frames.
        first_episode: Index of the first episode to encode.
        episode_count: Number of episodes to encode, all the remaining ones by default.
        batch_size: Clips encoded at once.
        device: The device of the VAE.
    """
    storage = open_storage(dataset, lerobot_mapping)
    encode = load_video_encoder(vae, device)
    write_store_info(output, view_sizes, chunk_size, video_frame_stride)
    last = None if episode_count is None else first_episode + episode_count
    for record in storage.episodes[first_episode:last]:
        view_frames = read_window_frames(storage, record, np.arange(record.rows), view_sizes)
        with torch.no_grad():
            latents = encode_episode(
                encode, view_frames, record.rows, chunk_size, video_frame_stride, modes, batch_size, device
            )
        for file_name, values in latents.items():
            write_episode_latents(output, record.name, file_name, values)
        print(f"{record.name}: {record.rows} rows encoded", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--view-sizes", required=True, help="one HEIGHTxWIDTH per view, separated by commas")
    parser.add_argument("--modes", default="base,ar")
    parser.add_argument("--lerobot-mapping", default=None)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--video-frame-stride", type=int, default=4)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--episode-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    build_latents(
        arguments.dataset,
        arguments.output,
        arguments.vae,
        [tuple(int(side) for side in size.split("x")) for size in arguments.view_sizes.split(",")],
        tuple(arguments.modes.split(",")),
        arguments.lerobot_mapping,
        arguments.chunk_size,
        arguments.video_frame_stride,
        arguments.first_episode,
        arguments.episode_count,
        arguments.batch_size,
        arguments.device,
    )
