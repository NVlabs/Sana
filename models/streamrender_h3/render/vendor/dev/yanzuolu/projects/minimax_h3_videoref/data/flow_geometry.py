# SPDX-License-Identifier: Apache-2.0
"""Backward optical flow of a clip as per-latent AdaLN condition maps.

A clip's ``geometry.safetensors`` holds per RGB frame t, on the timeline of the
clip's reference video and on a grid 8x coarser than the frames:

* ``flow_bwd`` float16 [T, 2, H, W], the flow from frame t to t-1 in
  full-resolution pixels, x right and y down;
* ``flow_bwd_valid`` uint8 [T], 0 where frame t-1 does not exist;
* ``occlusion_bwd`` uint8 [T, H, W], how many of the cell's 64 pixels are
  occluded or out of the frame at t-1.

Only these game-agnostic signals are read. Each frame becomes four channels:
the flow divided by 32, the pixels per video token, then compressed as
``sign(v) log1p(|v|)`` and zeroed where it is invalid; the occluded fraction of
the cell; and the validity. Frames are packed on the video codec's decode
timeline: latent k owns ``FLOW_FRAME_SLOTS`` slots of four channels holding its
frames in order, unused slots zero, followed by one channel per slot marking the
slots in use. With the native H3 timeline every 17 frames form latents of
1, 4, 4, 4 and 4 frames, so a latent carries 20 channels.

The grid is 8x coarser than the frames, which is four times the 32-pixel video
token grid of a 16x VAE with 2x2 patches, as the DiT's condition encoder expects.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping

FLOW_FRAME_SLOTS = 4
FLOW_FRAME_CHANNELS = 4
FLOW_CONDITION_CHANNELS = FLOW_FRAME_SLOTS * (FLOW_FRAME_CHANNELS + 1)
FLOW_PIXELS_PER_TOKEN = 32.0
OCCLUSION_CELL_PIXELS = 64


def flow_frame_features(flow: torch.Tensor, valid: torch.Tensor, occlusion: torch.Tensor) -> torch.Tensor:
    """``[F, 2, H, W]`` flow, ``[F]`` validity and ``[F, H, W]`` occlusion counts -> ``[F, 4, H, W]`` float32."""
    valid = valid.to(torch.float32).view(-1, 1, 1, 1)
    flow = flow.to(torch.float32) / FLOW_PIXELS_PER_TOKEN * valid
    flow = flow.sign() * flow.abs().log1p()
    occluded = occlusion.to(torch.float32).unsqueeze(1) / OCCLUSION_CELL_PIXELS
    return torch.cat((flow, occluded, valid.expand_as(occluded)), dim=1)


def pack_frame_features(features: torch.Tensor, spans: tuple[int, ...]) -> torch.Tensor:
    """Place each latent's frames in its slots, then the slot-presence channels.

    ``features`` [F, c, H, W] holds the frames of consecutive latents whose
    frame counts are ``spans``. Returns ``[len(spans), FLOW_FRAME_SLOTS * (c + 1), H, W]``.
    """
    frames, channels, height, width = features.shape
    if sum(spans) != frames or max(spans) > FLOW_FRAME_SLOTS:
        raise ValueError(f"cannot pack {frames} frames into latents of {spans} frames with {FLOW_FRAME_SLOTS} slots")
    slots = features.new_zeros((len(spans), FLOW_FRAME_SLOTS, channels, height, width))
    present = features.new_zeros((len(spans), FLOW_FRAME_SLOTS, height, width))
    cursor = 0
    for latent, span in enumerate(spans):
        slots[latent, :span] = features[cursor:cursor + span]
        present[latent, :span] = 1
        cursor += span
    return torch.cat((slots.flatten(1, 2), present), dim=1)


def load_flow_condition_maps(
    path: str | Path, *, start_frame: int, latent_count: int, video_temporal_mapping: VideoTemporalMapping,
) -> torch.Tensor:
    """Read the frames of ``latent_count`` latents from ``start_frame`` on -> ``[L, 20, H, W]`` float16.

    ``start_frame`` is the geometry frame of the first RGB frame of latent 0,
    and latent 0 opens a cycle of the decode timeline, as in every native
    crop. Only the needed frames are read.
    """
    spans = video_temporal_mapping.decode_timeline.spans(latent_count)
    stop = start_frame + sum(spans)
    with safe_open(str(path), framework="pt") as handle:
        frames = handle.get_slice("flow_bwd_valid").get_shape()[0]
        if not 0 <= start_frame < stop <= frames:
            raise ValueError(f"{path}: frames [{start_frame}, {stop}) exceed its {frames} geometry frames")
        features = flow_frame_features(
            handle.get_slice("flow_bwd")[start_frame:stop],
            handle.get_slice("flow_bwd_valid")[start_frame:stop],
            handle.get_slice("occlusion_bwd")[start_frame:stop],
        )
    return pack_frame_features(features, spans).to(torch.float16)


__all__ = [
    "FLOW_CONDITION_CHANNELS", "FLOW_FRAME_CHANNELS", "FLOW_FRAME_SLOTS", "FLOW_PIXELS_PER_TOKEN",
    "OCCLUSION_CELL_PIXELS", "flow_frame_features", "load_flow_condition_maps", "pack_frame_features",
]
