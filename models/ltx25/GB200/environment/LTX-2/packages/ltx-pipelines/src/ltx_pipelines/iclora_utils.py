"""Shared IC-LoRA helpers: LoRA metadata, mask downsampling, reference-video conditioning.
Used by ``ic_lora`` and ``lipdub`` (video reference path only). LipDub audio helpers live in ``lipdub.py``.
"""

from __future__ import annotations

import logging

import torch
from einops import rearrange
from safetensors import safe_open

from ltx_core.conditioning import (
    ConditioningItem,
    ConditioningItemAttentionStrengthWrapper,
    VideoConditionByReferenceLatent,
)
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.media_io import decode_video_by_frame, video_preprocess


def read_lora_reference_downscale_factor(lora_path: str) -> int:
    """Read ``reference_downscale_factor`` from LoRA safetensors metadata (default 1)."""
    try:
        with safe_open(lora_path, framework="pt") as f:
            metadata = f.metadata() or {}
            return int(metadata.get("reference_downscale_factor", 1))
    except Exception as e:
        logging.warning("Failed to read metadata from LoRA file '%s': %s", lora_path, e)
        return 1


def read_lora_reference_temporal_scale_factor(lora_path: str) -> int:
    """Read ``reference_temporal_scale_factor`` from LoRA safetensors metadata (default 1)."""
    try:
        with safe_open(lora_path, framework="pt") as f:
            metadata = f.metadata() or {}
            return int(metadata.get("reference_temporal_scale_factor", 1))
    except Exception as e:
        logging.warning("Failed to read metadata from LoRA file '%s': %s", lora_path, e)
        return 1


def downsample_mask_video_to_latent(
    mask: torch.Tensor,
    target_latent_shape: VideoLatentShape,
) -> torch.Tensor:
    """Downsample a pixel-space mask video to flattened latent token weights."""
    b = mask.shape[0]
    f_lat = target_latent_shape.frames
    h_lat = target_latent_shape.height
    w_lat = target_latent_shape.width

    f_pix = mask.shape[2]
    spatial_down = torch.nn.functional.interpolate(
        rearrange(mask, "b 1 f h w -> (b f) 1 h w"),
        size=(h_lat, w_lat),
        mode="area",
    )
    spatial_down = rearrange(spatial_down, "(b f) 1 h w -> b 1 f h w", b=b)

    first_frame = spatial_down[:, :, :1, :, :]

    if f_pix > 1 and f_lat > 1:
        t = (f_pix - 1) // (f_lat - 1)
        assert (f_pix - 1) % (f_lat - 1) == 0, (
            f"Pixel frames ({f_pix}) not compatible with latent frames ({f_lat}): "
            f"(f_pix - 1) must be divisible by (f_lat - 1)"
        )
        rest = rearrange(spatial_down[:, :, 1:, :, :], "b 1 (f t) h w -> b 1 f t h w", t=t)
        rest = rest.mean(dim=3)
        latent_mask = torch.cat([first_frame, rest], dim=2)
    else:
        latent_mask = first_frame

    return rearrange(latent_mask, "b 1 f h w -> b (f h w)")


def temporal_subsample(video: torch.Tensor, temporal_scale_factor: int) -> torch.Tensor:
    """VAE-aligned temporal subsampling: keep frame 0, then every Nth frame."""
    indices = [0, *list(range(1, video.shape[2], temporal_scale_factor))]
    return video[:, :, indices]


def append_ic_lora_reference_video_conditionings(  # noqa: PLR0913
    conditionings: list[ConditioningItem],
    video_conditioning: list[tuple[str, float]],
    *,
    height: int,
    width: int,
    num_frames: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    reference_downscale_factor: int,
    reference_temporal_scale_factor: int = 1,
    conditioning_attention_strength: float,
    conditioning_attention_mask: torch.Tensor | None,
    tiling_config: TilingConfig | None = None,
) -> None:
    """Append :class:`VideoConditionByReferenceLatent` items for each reference path."""
    scale = reference_downscale_factor
    if scale != 1 and (height % scale != 0 or width % scale != 0):
        raise ValueError(
            f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor ({scale})"
        )
    ref_height = height // scale
    ref_width = width // scale

    for video_path, strength in video_conditioning:
        frame_gen = decode_video_by_frame(path=video_path, frame_cap=num_frames, device=device)
        video = video_preprocess(frame_gen, ref_height, ref_width, dtype, device)
        if reference_temporal_scale_factor > 1:
            video = temporal_subsample(video, reference_temporal_scale_factor)
        if tiling_config is not None:
            encoded_video = video_encoder.tiled_encode(video, tiling_config)
        else:
            encoded_video = video_encoder(video)
        reference_video_shape = VideoLatentShape.from_torch_shape(encoded_video.shape)

        if conditioning_attention_mask is not None:
            latent_mask = downsample_mask_video_to_latent(
                mask=conditioning_attention_mask,
                target_latent_shape=reference_video_shape,
            )
            attn_mask = latent_mask * conditioning_attention_strength
        elif conditioning_attention_strength < 1.0:
            attn_mask = conditioning_attention_strength
        else:
            attn_mask = None

        cond = VideoConditionByReferenceLatent(
            latent=encoded_video,
            downscale_factor=scale,
            temporal_scale_factor=reference_temporal_scale_factor,
            strength=strength,
        )
        if attn_mask is not None:
            cond = ConditioningItemAttentionStrengthWrapper(cond, attention_mask=attn_mask)
        conditionings.append(cond)
