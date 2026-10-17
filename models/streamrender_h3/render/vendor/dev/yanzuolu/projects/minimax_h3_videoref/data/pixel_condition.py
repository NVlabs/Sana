# SPDX-License-Identifier: Apache-2.0
"""A pixel video aligned frame by frame with the target as per-latent input condition maps.

The condition is any RGB video on the target's frame grid and resolution,
frame t beside target frame t. ``decode_condition_frames`` reads its frames
``[start_frame, start_frame + num_frames)`` in decode order as uint8
``[F, H, W, 3]``, the form a clip keeps in host and device memory.

``pixel_condition_maps`` turns the frames of chosen target latents into the
DiT's input condition maps. Pixels are scaled to [-1, 1] like the VAE input
and each frame is pixel-unshuffled by ``unshuffle`` u, losslessly, into
``3 u^2`` channels at ``1/u`` of the frame grid, channel ``c u^2 + i u + j``
holding colour c of pixel ``(u y + i, u x + j)``. Frames are packed on the
video codec's decode timeline exactly like the flow maps of ``flow_geometry``:
latent k owns ``FLOW_FRAME_SLOTS`` slots holding its frames in order, unused
slots zero, followed by one presence channel per slot. With the native H3
timeline every 17 frames form latents of 1, 4, 4, 4 and 4 frames, and at
u = 8 a latent carries ``4 * 192 + 4 = 772`` channels.
"""

from __future__ import annotations

from itertools import accumulate
from pathlib import Path

import torch
from torch.nn import functional as F

from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping
from dev.yanzuolu.projects.minimax_h3_videoref.data.flow_geometry import FLOW_FRAME_SLOTS, pack_frame_features


def pixel_condition_channels(unshuffle: int) -> int:
    """Input condition channels of one latent: every slot's unshuffled RGB, then the slot presence."""
    return FLOW_FRAME_SLOTS * (3 * unshuffle * unshuffle + 1)


def decode_condition_frames(
    path: str | Path, *, start_frame: int, num_frames: int, height: int, width: int, fps: int,
) -> torch.Tensor:
    """Frames ``[start_frame, start_frame + num_frames)`` of a constant-rate video -> uint8 ``[F, H, W, 3]``."""
    import av

    frames = torch.empty((num_frames, height, width, 3), dtype=torch.uint8)
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != fps:
            raise ValueError(f"{path}: condition videos must run at {fps} FPS, got {stream.average_rate}")
        if (stream.height, stream.width) != (height, width):
            raise ValueError(f"{path}: frames are {stream.width}x{stream.height}, the target is {width}x{height}")
        for index, frame in enumerate(container.decode(stream)):
            if index < start_frame:
                continue
            if count == num_frames:
                break
            frames[count] = torch.from_numpy(frame.to_ndarray(format="rgb24"))
            count += 1
    if count != num_frames:
        raise ValueError(f"{path}: holds {start_frame + count} frames, frames [{start_frame}, "
                         f"{start_frame + num_frames}) are needed")
    return frames


def pixel_condition_maps(
    frames: torch.Tensor, latents: torch.Tensor, *, video_temporal_mapping: VideoTemporalMapping,
    unshuffle: int, dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """The chosen latents' maps from a complete target's frames -> ``[len(latents), channels, H/u, W/u]``.

    ``frames`` [F, H, W, 3] uint8 hold every frame of the target, whose latent
    count the decode timeline derives from F. ``latents`` are indices into
    that target in any order, and only their frames are converted.
    """
    total = video_temporal_mapping.latent_t(frames.shape[0])
    spans = video_temporal_mapping.decode_timeline.spans(total)
    height, width = frames.shape[1] // unshuffle, frames.shape[2] // unshuffle
    chosen = [int(index) for index in latents.tolist()]
    if not chosen:
        return frames.new_zeros((0, pixel_condition_channels(unshuffle), height, width), dtype=dtype)
    starts = [0, *accumulate(spans)]
    indices = torch.cat([torch.arange(starts[index], starts[index + 1]) for index in chosen]).to(frames.device)
    pixels = frames.index_select(0, indices).permute(0, 3, 1, 2).float().div_(127.5).sub_(1).to(dtype)
    return pack_frame_features(F.pixel_unshuffle(pixels, unshuffle), tuple(spans[index] for index in chosen))


__all__ = ["decode_condition_frames", "pixel_condition_channels", "pixel_condition_maps"]
