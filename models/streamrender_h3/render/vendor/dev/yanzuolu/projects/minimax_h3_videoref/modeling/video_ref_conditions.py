# SPDX-License-Identifier: Apache-2.0
"""Native reference encoding and score conditions for synchronous depth windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    MiniMaxH3Ref2VAPresentationProcessor,
    Ref2VAPresentation,
    encode_ref2va_presentations,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_reference import (
    DEFAULT_VISUAL_ANCHOR,
    REFERENCE_FPS,
    EncodedReferencePlan,
    RefBlockSpec,
    RefMediaProbe,
    _encode_visual_latent,
    encode_ref_block_plan,
    video_latent_t_from_frame_count,
)


@dataclass(frozen=True)
class VideoRefScoreConditions:
    """Aligned native reference plans, Qwen presentations, and encoded rows."""

    reference_plans: list[EncodedReferencePlan]
    presentations: list[Ref2VAPresentation]
    prompt_embeds: list[torch.Tensor]
    negative: VideoRefScoreConditions | None = None


def video_ref_corpus_metadata(
    reference: torch.Tensor, *, path: str, num_frames: int, height: int, width: int,
    video_temporal_mapping: VideoTemporalMapping, start_frame: int = 0,
    start_time_seconds: float | None = None,
) -> dict[str, Any]:
    """Keep normalized clean reference latents and their original RGB timeline."""
    return {
        "reference_video": reference.to(dtype=torch.bfloat16, device="cpu"),
        "reference_video_path": str(Path(path).expanduser().resolve()),
        **({"reference_start_frame": start_frame} if start_time_seconds is None
           else {"reference_start_time_seconds": start_time_seconds}),
        "reference_height": height,
        "reference_width": width,
        "num_frames": num_frames,
        "fps": REFERENCE_FPS,
        "video_temporal_mapping": video_temporal_mapping.to_dict(),
    }


def _decoded_video(
    pixels: torch.Tensor, *,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> tuple[np.ndarray, RefBlockSpec]:
    """Recover RGB24 from the dataset's CTHW float32 window without rereading it."""
    assert pixels.ndim == 4 and pixels.shape[0] == 3
    frames, height, width = map(int, pixels.shape[1:])
    media = (
        pixels.detach().cpu().permute(1, 2, 3, 0)
        .mul(255).round().to(torch.uint8).contiguous().numpy()
    )
    spec = RefBlockSpec(
        condition_index=0,
        kind="video",
        path="decoded-video",
        start_time_seconds=0.0,
        probe=RefMediaProbe(
            width=width, height=height, has_audio=False,
            duration_seconds=frames / REFERENCE_FPS,
        ),
        resolved_width=width,
        resolved_height=height,
        resolved_frame_count=frames,
        video_temporal_mapping=video_temporal_mapping,
    )
    return media, spec


@torch.no_grad()
def encode_video_ref_targets(
    video_vae: Any,
    pixels: Sequence[torch.Tensor],
    *,
    seeds: Sequence[int],
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
    latent_indices: Sequence[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Encode corpus targets into native normalized CPU CTHW score latents.

    This encodes real corpus pixels only. Generated student latents already use
    H3 diffusion coordinates and do not pass through a codec conversion. Each
    target draws its posterior noise from its own entry of ``seeds``. Frame
    geometry must match the configured codec. The default full VAE uses 17n+5
    frames, including the 107-frame score window.

    Optional global latent indices preserve their supplied order. Native VAE
    encoding skips unused independent temporal blocks while retaining the full
    posterior draw. Other codecs encode normally before selecting their output.
    """
    if latent_indices is not None and len(latent_indices) != len(pixels):
        raise ValueError("one latent index tensor is required for each source video")
    outputs = []
    for index, (video, seed) in enumerate(zip(pixels, seeds, strict=True)):
        video_temporal_mapping.target_latent_t(int(video.shape[1]))
        media, spec = _decoded_video(video, video_temporal_mapping=video_temporal_mapping)
        outputs.append(
            _encode_visual_latent(
                video_vae, media, spec, seed=int(seed),
                **({"latent_indices": latent_indices[index]} if latent_indices is not None else {}),
            )[0].contiguous()
        )
    return outputs


@torch.no_grad()
def encode_video_ref_references(
    *,
    reference_pixels: Sequence[torch.Tensor],
    video_vae: Any,
    encode_seeds: Sequence[int],
    noise_seed: int,
    visual_anchor: float = DEFAULT_VISUAL_ANCHOR,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> list[EncodedReferencePlan]:
    """Encode synchronous depth windows with native reference anchors and seeds."""
    plans = []
    for pixels, encode_seed in zip(reference_pixels, encode_seeds, strict=True):
        media, spec = _decoded_video(pixels, video_temporal_mapping=video_temporal_mapping)
        plans.append(encode_ref_block_plan(
            [spec],
            video_vae=video_vae,
            audio_vae=None,
            target_latent_t=video_latent_t_from_frame_count(
                spec.resolved_frame_count, video_temporal_mapping=video_temporal_mapping
            ),
            encode_seed=int(encode_seed),
            noise_seed=noise_seed,
            visual_anchor=visual_anchor,
            decoded_visual_media={0: media},
        ))
    return plans


@torch.no_grad()
def prepare_video_ref_score_conditions(
    *,
    reference_pixels: Sequence[torch.Tensor],
    prompts: Sequence[str],
    video_vae: Any,
    text_encoder: Any,
    processor: MiniMaxH3Ref2VAPresentationProcessor,
    encode_seeds: Sequence[int],
    noise_seed: int,
    visual_anchor: float = DEFAULT_VISUAL_ANCHOR,
    include_empty_language: bool = False,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> VideoRefScoreConditions:
    """Prepare depth reference latents and multimodal Qwen rows once per batch.

    The same RGB window supplies both representations. Each sample has its own
    posterior seed from ``encode_seeds``. Condition augmentation uses the separate
    ``noise_seed`` and is fixed in the plans shared by fake and teacher. Qwen
    performs one forward regardless of the number of samples packed on a rank.
    Every rank must supply a non-empty depth-video batch so its visual-tower
    collectives match. The window must suit the configured codec, which is 107
    frames for the default full VAE score path. Reference audio is omitted.
    Empty language prompts retain the reference vision presentation. Optional
    empty-language conditions share the encoded reference plans and join the
    same Qwen batch as independent samples.
    """
    plans = encode_video_ref_references(
        reference_pixels=reference_pixels,
        video_vae=video_vae,
        encode_seeds=encode_seeds,
        noise_seed=noise_seed,
        visual_anchor=visual_anchor,
        video_temporal_mapping=video_temporal_mapping,
    )
    presentations = [
        processor.build(prompt, plan.qwen_media)
        for prompt, plan in zip(prompts, plans, strict=True)
    ]
    if include_empty_language:
        negative_presentations = [processor.build("", plan.qwen_media) for plan in plans]
        embeddings = encode_ref2va_presentations(
            text_encoder, presentations + negative_presentations
        )
        count = len(presentations)
        return VideoRefScoreConditions(
            reference_plans=plans,
            presentations=presentations,
            prompt_embeds=embeddings[:count],
            negative=VideoRefScoreConditions(
                reference_plans=plans,
                presentations=negative_presentations,
                prompt_embeds=embeddings[count:],
            ),
        )
    return VideoRefScoreConditions(
        reference_plans=plans,
        presentations=presentations,
        prompt_embeds=encode_ref2va_presentations(text_encoder, presentations),
    )


__all__ = [
    "VideoRefScoreConditions",
    "encode_video_ref_references",
    "encode_video_ref_targets",
    "prepare_video_ref_score_conditions",
    "video_ref_corpus_metadata",
]
