# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WindowDataConfig:
    """The dataset of whole-episode windows and its loader.

    ``augmentation`` holds the random image changes of a window (``train/utils/augmentation.py``), each optional and
    drawn once per window: ``camera`` (``views``, ``probability``, ``scale``: a crop of the named cameras at a
    random scale and place), ``lighting`` (``gain_probability``, ``gain``, ``shadow_probability``, ``shadow_width``,
    ``shadow_depth``: one gain per colour channel and one soft shadow per view), ``color`` (``brightness``,
    ``contrast``, ``saturation``, ``gamma``: the largest relative change of each, shared by the views) and
    ``view_changes`` (``clean_probability``: the share of windows it leaves as they are; ``crop_area``, ``rotation``
    in degrees, ``translation`` as a share of the size, ``brightness``, ``contrast``, ``saturation``, ``hue``,
    ``gamma``: a change of every view on its own; ``camera_move`` with ``views``, ``probability``, ``yaw``,
    ``pitch``, ``roll`` in degrees, ``focal``, ``shift_x``, ``shift_y`` in pixels and ``radial``: a small move of
    the named cameras). It needs frames, so no latent store.
    """

    path: str = ""
    lerobot_mapping: Optional[str] = None
    normalization: str = ""
    view_sizes: List[List[int]] = field(default_factory=lambda: [[256, 320], [256, 320], [256, 320]])
    latent_store: Optional[str] = None
    augmentation: Optional[Dict[str, Any]] = None
    chunk_size: int = 32
    video_frame_stride: int = 4
    chunk_budget: int = 160
    num_workers: int = 6


@dataclass
class ChunkDataConfig:
    """The dataset of one-chunk windows and its loader."""

    path: str = ""
    lerobot_mapping: Optional[str] = None
    normalization: str = ""
    view_sizes: List[List[int]] = field(default_factory=lambda: [[256, 320], [256, 320], [256, 320]])
    latent_store: Optional[str] = None
    chunk_size: int = 32
    video_frame_stride: int = 4
    batch_size: int = 8
    num_workers: int = 6
