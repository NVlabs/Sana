# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate
from typing import Any


@dataclass(frozen=True)
class FrameTimeline:
    """Frame spans of an initial latent prefix and a repeating latent cycle."""

    prefix_spans: tuple[int, ...]
    repeating_spans: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.repeating_spans or any(
            not isinstance(span, int) or isinstance(span, bool) or span <= 0
            for span in (*self.prefix_spans, *self.repeating_spans)
        ):
            raise ValueError("timeline spans must be positive integers with a nonempty cycle")

    def _parts(self, count: int) -> tuple[int, int]:
        count = int(count)
        if count < 0:
            raise ValueError("latent boundary index must be nonnegative")
        prefix_count = min(count, len(self.prefix_spans))
        cycles, tail_count = divmod(count - prefix_count, len(self.repeating_spans))
        partial_frames = sum(self.prefix_spans[:prefix_count]) + sum(self.repeating_spans[:tail_count])
        return cycles, partial_frames

    def boundary(self, index: int) -> int:
        """Return the cumulative RGB-frame boundary before latent index."""
        cycles, partial_frames = self._parts(index)
        return cycles * sum(self.repeating_spans) + partial_frames

    def count_to_cover(self, frames: int) -> int:
        """Return the fewest latent spans covering the requested frame count."""
        frames = int(frames)
        if frames < 0:
            raise ValueError("frame count must be nonnegative")
        prefix_boundaries = (0, *accumulate(self.prefix_spans))
        if frames <= prefix_boundaries[-1]:
            return bisect_left(prefix_boundaries, frames)
        cycles, remaining = divmod(frames - prefix_boundaries[-1], sum(self.repeating_spans))
        cycle_boundaries = (0, *accumulate(self.repeating_spans))
        return (
            len(self.prefix_spans)
            + cycles * len(self.repeating_spans)
            + bisect_left(cycle_boundaries, remaining)
        )

    def spans(self, count: int) -> tuple[int, ...]:
        """Return one RGB-frame span for each latent."""
        count = int(count)
        if count < 0:
            raise ValueError("latent count must be nonnegative")
        prefix = self.prefix_spans[:count]
        cycles, tail_count = divmod(count - len(prefix), len(self.repeating_spans))
        return prefix + self.repeating_spans * cycles + self.repeating_spans[:tail_count]

    def clock_boundary(
        self, index: int, origin: float = 0.0, *, rate_num: int = 5, rate_den: int = 3,
    ) -> float:
        """Map a latent boundary onto a rationally scaled frame clock."""
        cycles, partial_frames = self._parts(index)
        cycle_span = sum(self.repeating_spans) * rate_num / rate_den
        return float(origin) + cycles * cycle_span + (rate_num / rate_den) * partial_frames

    def clock_boundary_ceil(
        self, index: int, *, rate_num: int = 5, rate_den: int = 3,
    ) -> int:
        """Round a frame boundary up on the rational clock using integer arithmetic."""
        return (self.boundary(index) * rate_num + rate_den - 1) // rate_den

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_spans": list(self.prefix_spans),
            "repeating_spans": list(self.repeating_spans),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FrameTimeline:
        return cls(tuple(value["prefix_spans"]), tuple(value["repeating_spans"]))


@dataclass(frozen=True)
class VideoTemporalMapping:
    """Codec encoding counts, decoding time spans and valid video frame counts."""

    encode_timeline: FrameTimeline
    decode_timeline: FrameTimeline
    valid_frame_period: int = 1
    valid_frame_remainders: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.valid_frame_period <= 0 or not self.valid_frame_remainders or any(
            remainder < 0 or remainder >= self.valid_frame_period
            for remainder in self.valid_frame_remainders
        ):
            raise ValueError("valid frame remainders must belong to a positive period")

    def latent_t(self, frame_count: int) -> int:
        """Return the encoded latent count for a valid positive input length."""
        frame_count = int(frame_count)
        if frame_count <= 0 or frame_count % self.valid_frame_period not in self.valid_frame_remainders:
            raise ValueError(
                f"video frame count must be positive with remainder in {self.valid_frame_remainders} "
                f"modulo {self.valid_frame_period}, got {frame_count}"
            )
        return self.encode_timeline.count_to_cover(frame_count)

    def frame_count(self, latent_t: int) -> int:
        """Return the frame count produced by decoding positive latent_t."""
        if latent_t <= 0:
            raise ValueError("video latent count must be positive")
        return self.decode_timeline.boundary(latent_t)

    def target_latent_t(self, frame_count: int) -> int:
        """Validate equal target input/output frame counts and return latent_t."""
        latent_t = self.latent_t(frame_count)
        decoded_frames = self.frame_count(latent_t)
        if decoded_frames != frame_count:
            raise ValueError(
                f"target video with {frame_count} frames encodes to {latent_t} latents "
                f"but decodes to {decoded_frames} frames"
            )
        return latent_t

    def position_start(self, index: int, origin: float = 0.0) -> float:
        """Return a decoded latent's position on the shared 40 Hz audio clock."""
        return self.decode_timeline.clock_boundary(index, origin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "encode_timeline": self.encode_timeline.to_dict(),
            "decode_timeline": self.decode_timeline.to_dict(),
            "valid_frame_period": self.valid_frame_period,
            "valid_frame_remainders": list(self.valid_frame_remainders),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VideoTemporalMapping:
        return cls(
            encode_timeline=FrameTimeline.from_dict(value["encode_timeline"]),
            decode_timeline=FrameTimeline.from_dict(value["decode_timeline"]),
            valid_frame_period=int(value["valid_frame_period"]),
            valid_frame_remainders=tuple(value["valid_frame_remainders"]),
        )


H3_VIDEO_TEMPORAL_MAPPING = VideoTemporalMapping(
    encode_timeline=FrameTimeline((), (1, 4, 4, 4, 4)),
    decode_timeline=FrameTimeline((), (1, 4, 4, 4, 4)),
    valid_frame_period=17,
    valid_frame_remainders=(0, 5),
)
CONTINUOUS_VIDEO_TEMPORAL_MAPPING = VideoTemporalMapping(
    encode_timeline=FrameTimeline((), (4,)),
    decode_timeline=FrameTimeline((1,), (4,)),
)


def video_temporal_mapping_from_model_config(config: Mapping[str, Any]) -> VideoTemporalMapping:
    """Read a codec class's temporal descriptor without constructing its model."""
    codec_class = getattr(importlib.import_module(config["module"]), config["class_name"])
    mapping = codec_class.video_temporal_mapping
    if not isinstance(mapping, VideoTemporalMapping):
        raise TypeError("video codec must declare a VideoTemporalMapping descriptor")
    return mapping


def minimax_h3_align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the next 17n or 17n+5 boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    quotient, remainder = divmod(current, 17)
    if remainder == 0:
        return current
    if remainder <= 5:
        return quotient * 17 + 5
    return (quotient + 1) * 17


def minimax_h3_video_latent_t(frame_count: int) -> int:
    current = int(frame_count)
    quotient, remainder = divmod(current, 17)
    if quotient >= 1 and remainder == 0:
        return quotient * 5
    if current >= 5 and remainder == 5:
        return quotient * 5 + 2
    raise ValueError(
        f"MiniMax H3 video frame count must match 17n or 17n+5, got {current}"
    )


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    current = int(out_t)
    if current == 1:
        return 1
    quotient, remainder = divmod(current, 5)
    if quotient >= 1 and remainder == 0:
        return quotient * 17
    if current >= 2 and remainder == 2:
        return quotient * 17 + 5
    raise ValueError(
        f"MiniMax H3 video latent T must be 1 or match 5n or 5n+2, got {current}"
    )


def minimax_h3_audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
) -> list[float]:
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    import torch

    # The rectified-flow sigma range is fixed at [1.0, 0.0].
    base = torch.linspace(
        1.0,
        0.0,
        int(num_steps),
        device="cpu",
        dtype=torch.float32,
    )
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    # A one-point request is still exactly one point.  Normal serving uses
    # multiple points, but preserving the requested cardinality keeps
    # ``num_inference_steps`` the sole schedule-size control.
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]
