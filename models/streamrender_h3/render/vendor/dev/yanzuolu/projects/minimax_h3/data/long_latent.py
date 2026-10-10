# SPDX-License-Identifier: Apache-2.0
"""Native H3 mother-clip crops on a shared video and PCM timeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dev.yanzuolu.projects.minimax_h3.modeling.time_request import H3_VIDEO_TEMPORAL_MAPPING


LONG_LATENT_FORMAT = "minimax_h3_long_gt_latent_v1"
# The same mother-clip metadata with only ``reference_video``, for trainers that never read target values.
# Its ``audio_valid_samples`` admits every native crop.
LONG_REFERENCE_LATENT_FORMAT = "minimax_h3_long_reference_latent_v1"
VIDEO_FPS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_HOP_LENGTH = 800
AUDIO_PHASE_OFFSETS_SAMPLES = (0, 267, 533)
AUDIO_LATENT_CHANNELS = 32


def native_encode_num_frames(num_frames: int) -> int:
    """Truncate a mother clip to the native encoder's 17n+5 frame boundary."""
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames < 22:
        raise ValueError("native mother video encoding requires at least 22 RGB frames")
    return ((num_frames - 5) // 17) * 17 + 5


def native_crop_frames_to_latents(num_frames: int) -> int:
    """Validate the configured crop length and return its native latent count."""
    if (not isinstance(num_frames, int) or isinstance(num_frames, bool)
            or num_frames < 5 or (num_frames - 5) % 17):
        raise ValueError("native crops must contain 17n+5 RGB frames")
    return ((num_frames - 5) // 17) * 5 + 2


def audio_crop_phase(start_latent: int) -> tuple[int, int, int]:
    """Return phase, audio-latent start and PCM start for a video-period start.

    Five native video latents span seventeen RGB frames. At 24 FPS this is
    68000/3 samples of 32 kHz PCM, so three separately encoded PCM origins
    let every crop use an integer 800-sample audio hop.
    """
    if not isinstance(start_latent, int) or isinstance(start_latent, bool) or start_latent < 0 or start_latent % 5:
        raise ValueError("video crops must start at native latent 0, 5, 10, ...")
    period = start_latent // 5
    start_sample = (68000 * period + 1) // 3
    phase = period % 3
    start_audio = (start_sample - AUDIO_PHASE_OFFSETS_SAMPLES[phase]) // AUDIO_HOP_LENGTH
    return phase, start_audio, start_sample


def crop_audio_latent_count(num_frames: int) -> int:
    """Return the existing 40 Hz AV target length without floating-point drift."""
    native_crop_frames_to_latents(num_frames)
    return (5 * num_frames + 1) // 3


def aligned_crop_starts(
    total_frames: int, crop_frames: int, *, audio_valid_samples: int | None = None,
) -> list[int]:
    """List native latent starts whose complete AV crop fits genuine data."""
    native_crop_frames_to_latents(total_frames)
    native_crop_frames_to_latents(crop_frames)
    if total_frames < crop_frames:
        return []
    starts = range(0, ((total_frames - crop_frames) // 17 + 1) * 5, 5)
    if audio_valid_samples is None:
        return list(starts)
    if not isinstance(audio_valid_samples, int) or audio_valid_samples < 0:
        raise ValueError("audio_valid_samples must be a nonnegative integer")
    crop_samples = crop_audio_latent_count(crop_frames) * AUDIO_HOP_LENGTH
    return [start for start in starts if audio_crop_phase(start)[2] + crop_samples <= audio_valid_samples]


def crop_long_latent_entry(
    entry: Mapping[str, Any], start_latent: int, num_frames: int, *, reference_only: bool = False,
) -> dict[str, Any]:
    """Return one clip with local captions and the matching retained audio phase.

    Only the returned tensors are cropped. The source entry and its complete
    mother-clip metadata remain unchanged. Audio padding beyond the genuine
    PCM extent is never eligible for supervision.

    ``reference_only`` crops a ``LONG_REFERENCE_LATENT_FORMAT`` entry instead.
    Its target ``video`` and ``audio`` are zeros with the shapes and dtypes of
    the same crop of a ``LONG_LATENT_FORMAT`` entry.
    """
    from dev.yanzuolu.projects.minimax_h3.data.caption_timeline import crop_caption_segments

    if entry["format"] != (LONG_REFERENCE_LATENT_FORMAT if reference_only else LONG_LATENT_FORMAT):
        raise ValueError("unsupported long latent corpus format")
    if int(entry["fps"]) != VIDEO_FPS or int(entry["audio_sample_rate"]) != AUDIO_SAMPLE_RATE:
        raise ValueError("native long latents require 24 FPS video and 32 kHz PCM")
    if tuple(entry["audio_phase_offsets_samples"]) != AUDIO_PHASE_OFFSETS_SAMPLES:
        raise ValueError("audio phase origins disagree with native video-period crops")
    crop_t = native_crop_frames_to_latents(num_frames)
    total_frames = int(entry["num_frames"])
    starts = aligned_crop_starts(total_frames, num_frames, audio_valid_samples=int(entry["audio_valid_samples"]))
    if start_latent not in starts:
        raise ValueError("the selected crop exceeds the complete real video/audio extent")
    phase, audio_start, pcm_start = audio_crop_phase(start_latent)
    audio_t = crop_audio_latent_count(num_frames)
    reference = entry["reference_video"][:, start_latent:start_latent + crop_t].clone()
    if reference_only:
        # Complete entries store the target video like the reference and each audio phase as stereo bf16.
        video = reference.new_zeros(reference.shape)
        audio = reference.new_zeros(2, AUDIO_LATENT_CHANNELS, audio_t)
    else:
        phases = entry["audio_phases"]
        if len(phases) != len(AUDIO_PHASE_OFFSETS_SAMPLES):
            raise ValueError("long latent corpora require all three separately encoded audio phases")
        audio = phases[phase][..., audio_start:audio_start + audio_t].clone()
        video = entry["video"][:, start_latent:start_latent + crop_t].clone()
    if audio.shape[-1] != audio_t or video.shape[1] != crop_t or reference.shape[1] != crop_t:
        raise ValueError("stored mother latents do not cover the selected AV crop")
    start_frame = start_latent // 5 * 17
    result = {
        key: value for key, value in entry.items()
        if key not in {"video", "audio", "reference_video", "audio_phases"}
    }
    result.update(
        format="minimax_h3_gt_latent_crop_v1",
        video=video, audio=audio, reference_video=reference,
        num_frames=num_frames, video_latent_num_frames=crop_t,
        video_temporal_mapping=H3_VIDEO_TEMPORAL_MAPPING.to_dict(),
        reference_start_frame=int(entry.get("reference_start_frame", 0)) + start_frame,
        caption_segments=crop_caption_segments(entry["caption_segments"], start_frame, num_frames),
        source_crop_start_frame=start_frame,
        source_crop_start_latent=start_latent,
        source_audio_phase=phase,
        source_audio_start_sample=pcm_start,
        audio_valid_samples=audio_t * AUDIO_HOP_LENGTH,
    )
    if "reference_start_time_seconds" in result:
        result["reference_start_time_seconds"] = float(result["reference_start_time_seconds"]) + start_frame / VIDEO_FPS
    result["prompt"] = result["caption_segments"][0]["prompt"]
    return result


__all__ = [
    "LONG_LATENT_FORMAT", "LONG_REFERENCE_LATENT_FORMAT", "VIDEO_FPS", "AUDIO_SAMPLE_RATE", "AUDIO_HOP_LENGTH",
    "AUDIO_PHASE_OFFSETS_SAMPLES", "AUDIO_LATENT_CHANNELS", "native_encode_num_frames",
    "native_crop_frames_to_latents", "audio_crop_phase", "crop_audio_latent_count",
    "aligned_crop_starts", "crop_long_latent_entry",
]
