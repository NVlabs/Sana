# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import av
import numpy as np
import torch

VIDEO_FRAME_STRIDE = 4
VAE_TEMPORAL_STRIDE = 8
VAE_SPATIAL_STRIDE = 32
RESIZE_STRETCH = "stretch"
RESIZE_CROP = "crop"


def decode_frames(source, frame_indices, fps: float, start_time: float = 0.0) -> np.ndarray:
    """Return the uint8 frames [n, H, W, 3] at ``frame_indices`` (repeats allowed) of a video file or its bytes, frame k being the one shown at ``start_time + k / fps`` seconds; an index past the last frame of the video returns that last frame."""
    wanted = np.asarray(frame_indices, dtype=np.int64)
    frames = {}
    latest, position, seek_ahead = None, None, True
    with av.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        seconds_per_tick = float(stream.time_base)
        decoded = container.decode(stream)
        for target in sorted(set(wanted.tolist())):
            jumped = seek_ahead and (position is None or target > position + 1)
            if jumped:
                container.seek(int((start_time + target / fps) / seconds_per_tick), stream=stream, backward=True)
                decoded = container.decode(stream)
            for frame in decoded:
                index = int(round((frame.pts * seconds_per_tick - start_time) * fps))
                if jumped and position is not None and index <= position:
                    seek_ahead = False
                jumped = False
                latest, position = frame, index
                if index >= target:
                    break
            if position == target:
                frames[target] = latest.to_ndarray(format="rgb24")
    final = latest.to_ndarray(format="rgb24") if len(frames) < len(set(wanted.tolist())) else None
    return np.stack([frames.get(int(index), final) for index in wanted])


def unit_frames(frames: np.ndarray) -> torch.Tensor:
    """Return uint8 frames [T, H, W, 3] as float32 frames [T, 3, H, W] in [0, 1]."""
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)


def byte_frames(video: torch.Tensor) -> torch.Tensor:
    """Return float frames in [0, 1] as uint8 frames, rounded to the nearest value."""
    return video.mul_(255).round_().clamp_(0, 255).to(torch.uint8)


def resize_frames(video: torch.Tensor, size, resize: str) -> torch.Tensor:
    """Return float frames [T, 3, H, W] at ``size`` = (h, w) by bilinear interpolation: stretched to it (``stretch``), or scaled until they cover it and cut at the centre (``crop``)."""
    height, width = video.shape[-2:]
    target_height, target_width = (int(side) for side in size)
    if (height, width) == (target_height, target_width):
        return video
    if resize == RESIZE_STRETCH:
        scaled = (target_height, target_width)
    elif target_height / height > target_width / width:
        scaled = (target_height, round(width * (target_height / height)))
    else:
        scaled = (round(height * (target_width / width)), target_width)
    video = torch.nn.functional.interpolate(video, size=scaled, mode="bilinear", align_corners=False)
    top = int(round(scaled[0] - target_height) / 2.0)
    left = int(round(scaled[1] - target_width) / 2.0)
    return video[..., top : top + target_height, left : left + target_width]


def model_frames(frames: np.ndarray, size, resize: str) -> torch.Tensor:
    """Return uint8 frames [T, H, W, 3] as a uint8 tensor [T, 3, h, w] at ``size`` = (h, w) under a resize rule."""
    return byte_frames(resize_frames(unit_frames(frames), size, resize))


def frames_to_model_range(frames: torch.Tensor) -> torch.Tensor:
    """Return uint8 frames as float32 values in [-1, 1], the input range of the video VAE."""
    return frames.float().div_(127.5).sub_(1.0)
