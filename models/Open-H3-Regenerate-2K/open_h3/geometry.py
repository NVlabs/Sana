"""MiniMax-H3 T2VA inference contract and DMD bridge.

MiniMax-H3 predicts a data-ward velocity over one packed text/audio/video
sequence. This module keeps the released Diffusers layout and scheduler
semantics intact while exposing the small set of operations a DMD trainer
needs. In particular, H3 uses ``t = 1 - sigma`` and therefore
``x0 = x_t + sigma * velocity``. The plus sign is a checkpoint contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)


H3_VIDEO_TAG = 0
H3_TEXT_TAG = 1
H3_AUDIO_TAG = 2
H3_PATCH_SIZE = (1, 2, 2)
H3_AUDIO_CHANNELS = 2
H3_VIDEO_SHIFT = 12.0
H3_AUDIO_SHIFT = 3.0


@dataclass(frozen=True)
class MiniMaxH3Geometry:
    """Latent geometry shared by every item in one H3 training microbatch."""

    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    video_latent_channels: int = 24
    audio_latent_channels: int = 32
    audio_channels: int = H3_AUDIO_CHANNELS
    patch_size: tuple[int, int, int] = H3_PATCH_SIZE

    def validate(self) -> None:
        if min(
            self.num_latent_frames,
            self.latent_height,
            self.latent_width,
            self.num_audio_latents,
        ) <= 0:
            raise ValueError(f"All MiniMax-H3 latent dimensions must be positive: {self}")
        patch_t, patch_h, patch_w = self.patch_size
        if (
            self.num_latent_frames % patch_t
            or self.latent_height % patch_h
            or self.latent_width % patch_w
        ):
            raise ValueError(f"MiniMax-H3 geometry {self} is not divisible by patch {self.patch_size}")
        if self.audio_channels != H3_AUDIO_CHANNELS:
            raise ValueError(f"Released MiniMax-H3 is stereo, got audio_channels={self.audio_channels}")

    @property
    def rows_per_video_frame(self) -> int:
        _, patch_h, patch_w = self.patch_size
        return (self.latent_height // patch_h) * (self.latent_width // patch_w)

    @property
    def num_video_rows(self) -> int:
        patch_t, _, _ = self.patch_size
        return (self.num_latent_frames // patch_t) * self.rows_per_video_frame

    @property
    def num_audio_rows(self) -> int:
        return self.audio_channels * self.num_audio_latents

    @property
    def video_row_dim(self) -> int:
        patch_t, patch_h, patch_w = self.patch_size
        return self.video_latent_channels * patch_t * patch_h * patch_w


@dataclass(frozen=True)
class MiniMaxH3PackedLayout:
    """Structural tensors describing one packed H3 attention document."""

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int = 0
    num_condition_audio_rows: int = 0

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    def to(self, device: torch.device | str) -> "MiniMaxH3PackedLayout":
        return MiniMaxH3PackedLayout(
            position_ids=self.position_ids.to(device),
            token_tags=self.token_tags.to(device),
            video_indices=self.video_indices.to(device),
            audio_indices=self.audio_indices.to(device),
            text_indices=self.text_indices.to(device),
            num_condition_video_rows=self.num_condition_video_rows,
            num_condition_audio_rows=self.num_condition_audio_rows,
        )


def build_t2va_layout(text_token_tags: torch.Tensor, geometry: MiniMaxH3Geometry) -> MiniMaxH3PackedLayout:
    """Build the official ``[text | target audio | target video]`` T2VA layout."""
    geometry.validate()
    tags = torch.as_tensor(text_token_tags, dtype=torch.long).flatten().cpu()
    if tags.numel() == 0:
        raise ValueError("MiniMax-H3 requires at least one text token")
    if not bool((tags == H3_TEXT_TAG).all()):
        raise ValueError("T2VA text_token_tags must all be the released text modality tag 1")
    values = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=tags,
        num_latent_frames=geometry.num_latent_frames,
        latent_height=geometry.latent_height,
        latent_width=geometry.latent_width,
        num_audio_latents=geometry.num_audio_latents,
        patch_size=geometry.patch_size,
        audio_channels=geometry.audio_channels,
        audio_tag=H3_AUDIO_TAG,
        video_tag=H3_VIDEO_TAG,
        keyframe_anchors=(),
    )
    layout = MiniMaxH3PackedLayout(*values)
    expected = tags.numel() + geometry.num_audio_rows + geometry.num_video_rows
    if layout.sequence_length != expected:
        raise RuntimeError(f"Packed H3 sequence has {layout.sequence_length} rows, expected {expected}")
    return layout


def pack_audio_latents(audio_latents: torch.Tensor) -> torch.Tensor:
    """Pack ``[B, 2, C, A]`` stereo latents into channel-major H3 rows."""
    if audio_latents.ndim != 4 or audio_latents.shape[1] != H3_AUDIO_CHANNELS:
        raise ValueError(f"Expected stereo audio latents [B,2,C,A], got {tuple(audio_latents.shape)}")
    batch, channels, latent_channels, num_latents = audio_latents.shape
    return audio_latents.permute(0, 1, 3, 2).reshape(batch, channels * num_latents, latent_channels).contiguous()


def unpack_audio_rows(audio_rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    """Inverse of :func:`pack_audio_latents`, returning ``[B, 2, C, A]``."""
    if audio_rows.ndim != 3 or audio_rows.shape[1] != H3_AUDIO_CHANNELS * num_audio_latents:
        raise ValueError(
            f"Expected audio rows [B,{H3_AUDIO_CHANNELS * num_audio_latents},C], got {tuple(audio_rows.shape)}"
        )
    batch, _, latent_channels = audio_rows.shape
    return (
        audio_rows.reshape(batch, H3_AUDIO_CHANNELS, num_audio_latents, latent_channels)
        .permute(0, 1, 3, 2)
        .contiguous()
    )


def unpatchify_video_rows(video_rows: torch.Tensor, geometry: MiniMaxH3Geometry) -> torch.Tensor:
    """Inverse of the official H3 video patchifier, returning ``[B,C,T,H,W]``."""
    geometry.validate()
    if video_rows.ndim != 3 or video_rows.shape[1:] != (geometry.num_video_rows, geometry.video_row_dim):
        raise ValueError(
            "Video row shape does not match MiniMax-H3 geometry: "
            f"rows={tuple(video_rows.shape)} geometry={geometry}"
        )
    patch_t, patch_h, patch_w = geometry.patch_size
    batch = video_rows.shape[0]
    rows = video_rows.reshape(
        batch,
        geometry.num_latent_frames // patch_t,
        geometry.latent_height // patch_h,
        geometry.latent_width // patch_w,
        geometry.video_latent_channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(
        batch,
        geometry.video_latent_channels,
        geometry.num_latent_frames,
        geometry.latent_height,
        geometry.latent_width,
    ).contiguous()


def _scalar_sigma(value: torch.Tensor | float, name: str) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() != 1:
        raise ValueError(
            f"{name} must be scalar/shared across a microbatch because H3 structural timesteps have no batch axis"
        )
    result = float(tensor.item())
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1], got {result}")
    return result


def build_row_timestep_plan(
    layout: MiniMaxH3PackedLayout,
    *,
    video_sigma: torch.Tensor | float,
    audio_sigma: torch.Tensor | float,
    keyframe_noise_aug: float = 0.999,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign H3 timesteps to all rows for one joint transformer call."""
    video_sigma_value = _scalar_sigma(video_sigma, "video_sigma")
    audio_sigma_value = _scalar_sigma(audio_sigma, "audio_sigma")
    video_t = 1.0 - video_sigma_value
    audio_t = 1.0 - audio_sigma_value
    unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices=layout.video_indices.cpu(),
        audio_indices=layout.audio_indices.cpu(),
        num_condition_video_rows=layout.num_condition_video_rows,
        num_condition_audio_rows=layout.num_condition_audio_rows,
        num_text_tokens=int(layout.text_indices.numel()),
        video_timestep=video_t,
        audio_timestep=audio_t,
        condition_video_timestep=max(video_t, keyframe_noise_aug),
        condition_audio_timestep=1.0,
    )
    return unique.to(layout.position_ids.device), inverse.to(layout.position_ids.device)
