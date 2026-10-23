# Copyright 2026 MiniMax
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-media Ref2VA preparation for MiniMax-H3.

This module prepares request-ordered image, audio, video, and video-with-audio
references. It deliberately stops at Qwen-ready RGB media and fixed, normalized
condition rows. Qwen tokenization, DiT forward, sampling, and decode belong to
other RAVEN components.
"""

from __future__ import annotations

import contextlib
import functools
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from dev.yanzuolu.common.seed import combine_seed
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
)

RefKind = Literal["image", "audio", "video", "video_audio"]

REFERENCE_FPS = 24
REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_IMAGE_MULTIPLE = 32
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_RATE = 32000
VIDEO_SOUNDTRACK_SAMPLE_RATE = 44100
AUDIO_HOP_LENGTH = 800
VISUAL_PATCH_SIZE = (1, 2, 2)
DEFAULT_VISUAL_ANCHOR = 0.999
DEFAULT_AUDIO_ANCHOR = 1.0


@dataclass(frozen=True)
class RefMediaProbe:
    """Frozen media facts produced before reference preparation."""

    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    audio_sample_rate: int | None = None
    duration_seconds: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], *, path: str) -> RefMediaProbe:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping")

        def optional_positive_int(key: str) -> int | None:
            raw = value.get(key)
            if raw is None:
                return None
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{path}.{key} must be a positive integer")
            parsed = int(raw)
            if parsed <= 0 or float(raw) != float(parsed):
                raise ValueError(f"{path}.{key} must be a positive integer")
            return parsed

        has_audio = value.get("has_audio")
        if has_audio is not None and not isinstance(has_audio, bool):
            raise ValueError(f"{path}.has_audio must be a bool when present")
        duration = value.get("duration_seconds")
        if duration is not None:
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError(f"{path}.duration_seconds must be positive and finite")
        return cls(
            width=optional_positive_int("width"),
            height=optional_positive_int("height"),
            has_audio=has_audio,
            audio_sample_rate=optional_positive_int("audio_sample_rate"),
            duration_seconds=duration,
        )


@dataclass(frozen=True)
class RefBlockSpec:
    """One validated reference block in original request order."""

    condition_index: int
    kind: RefKind
    path: str
    start_time_seconds: float
    probe: RefMediaProbe
    resolved_width: int | None = None
    resolved_height: int | None = None
    resolved_frame_count: int | None = None
    resolved_duration_seconds: float | None = None
    video_temporal_mapping: VideoTemporalMapping = field(
        default=H3_VIDEO_TEMPORAL_MAPPING, kw_only=True
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.condition_index, bool)
            or not isinstance(self.condition_index, int)
            or self.condition_index < 0
        ):
            raise ValueError("condition_index must be a non-negative integer")
        if self.kind not in ("image", "audio", "video", "video_audio"):
            raise ValueError(f"unsupported reference kind {self.kind!r}")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("reference path must be a non-empty string")
        start = float(self.start_time_seconds)
        if not math.isfinite(start) or start < 0.0:
            raise ValueError("start_time_seconds must be non-negative and finite")
        if self.kind == "image":
            if self.resolved_width is None or self.resolved_height is None:
                raise ValueError("image reference requires resolved width and height")
        if self.kind in ("video", "video_audio"):
            if self.resolved_width is None or self.resolved_height is None:
                raise ValueError("video reference requires resolved width and height")
            if self.resolved_frame_count is None:
                raise ValueError("video reference requires a resolved frame count")
            video_latent_t_from_frame_count(
                self.resolved_frame_count, video_temporal_mapping=self.video_temporal_mapping
            )
            if not isinstance(self.probe.has_audio, bool):
                raise ValueError("video probe must provide an explicit has_audio bool")
            if self.kind == "video_audio" and not self.probe.has_audio:
                raise ValueError("video_audio requires an audio stream in the media probe")
        if self.kind == "audio" and self.probe.audio_sample_rate is None:
            raise ValueError("audio probe must provide its native audio_sample_rate")
        for name, value in (
            ("resolved_width", self.resolved_width),
            ("resolved_height", self.resolved_height),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value % REFERENCE_IMAGE_MULTIPLE
            ):
                raise ValueError(
                    f"{name} must be positive and aligned to {REFERENCE_IMAGE_MULTIPLE}"
                )
        if self.resolved_duration_seconds is not None:
            duration = float(self.resolved_duration_seconds)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("resolved_duration_seconds must be positive and finite")


@dataclass(frozen=True)
class EncodedRefBlock:
    """One encoded block with its fixed rows and Qwen visual medium."""

    spec: RefBlockSpec
    visual_media: Any | None
    visual_rows: torch.Tensor
    audio_rows: torch.Tensor
    latent_t: int | None
    latent_h: int | None
    latent_w: int | None
    audio_t: int
    visual_anchor: float | None
    audio_anchor: float | None
    row_anchors: torch.Tensor
    row_timesteps: torch.Tensor
    clean_visual_rows: torch.Tensor | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_rows(self.visual_rows, 96, "visual_rows")
        if self.clean_visual_rows is not None:
            _validate_rows(self.clean_visual_rows, 96, "clean_visual_rows")
            if self.clean_visual_rows.shape != self.visual_rows.shape:
                raise ValueError("clean and conditioned visual rows must have matching shapes")
        _validate_rows(self.audio_rows, AUDIO_LATENT_CHANNELS, "audio_rows")
        if int(self.audio_rows.shape[0]) != AUDIO_CHANNELS * int(self.audio_t):
            raise ValueError(
                f"audio rows {int(self.audio_rows.shape[0])} do not match "
                f"audio_t={self.audio_t} and stereo channel-major packing"
            )
        visual_shape = (self.latent_t, self.latent_h, self.latent_w)
        if self.visual_rows.numel():
            if any(value is None or int(value) <= 0 for value in visual_shape):
                raise ValueError("visual rows require positive latent geometry")
            assert self.latent_t is not None
            assert self.latent_h is not None
            assert self.latent_w is not None
            if self.latent_h % 2 or self.latent_w % 2:
                raise ValueError("visual latent height and width must be divisible by 2")
            expected = self.latent_t * (self.latent_h // 2) * (self.latent_w // 2)
            if int(self.visual_rows.shape[0]) != expected:
                raise ValueError(
                    f"visual rows {int(self.visual_rows.shape[0])} do not match "
                    f"latent geometry {visual_shape}"
                )
        elif any(value is not None for value in visual_shape):
            raise ValueError("non-visual block must not carry visual latent geometry")
        expected_metadata = int(self.audio_rows.shape[0]) + int(self.visual_rows.shape[0])
        for name, tensor in (
            ("row_anchors", self.row_anchors),
            ("row_timesteps", self.row_timesteps),
        ):
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or not tensor.is_contiguous()
                or list(tensor.shape) != [expected_metadata]
            ):
                raise ValueError(
                    f"{name} must be contiguous CPU float32 [{expected_metadata}]"
                )

    @property
    def ordered_rows(self) -> tuple[torch.Tensor, ...]:
        """Return block rows in packed-layout order, with video audio first."""

        rows: list[torch.Tensor] = []
        if self.audio_rows.numel():
            rows.append(self.audio_rows)
        if self.visual_rows.numel():
            rows.append(self.visual_rows)
        return tuple(rows)


@dataclass(frozen=True)
class EncodedReferencePlan:
    """All outputs needed by Qwen presentation and DiT reference packing."""

    blocks: tuple[EncodedRefBlock, ...]
    qwen_media: tuple[Any, ...]
    visual_rows: torch.Tensor
    audio_rows: torch.Tensor
    visual_row_anchors: torch.Tensor
    audio_row_anchors: torch.Tensor
    visual_row_timesteps: torch.Tensor
    audio_row_timesteps: torch.Tensor
    packed_row_anchors: torch.Tensor
    packed_row_timesteps: torch.Tensor
    ref_blocks: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        _validate_rows(self.visual_rows, 96, "visual_rows")
        _validate_rows(self.audio_rows, AUDIO_LATENT_CHANNELS, "audio_rows")
        if len(self.blocks) != len(self.qwen_media) or len(self.blocks) != len(
            self.ref_blocks
        ):
            raise ValueError("blocks, qwen_media, and ref_blocks must stay aligned")


def _validate_rows(rows: torch.Tensor, width: int, name: str) -> None:
    if not isinstance(rows, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if list(rows.shape)[1:] != [width] or rows.ndim != 2:
        raise ValueError(f"{name} must have shape [n, {width}], got {list(rows.shape)}")
    if rows.device.type != "cpu" or rows.dtype != torch.float32 or not rows.is_contiguous():
        raise ValueError(f"{name} must be contiguous CPU float32")


def video_latent_t_from_frame_count(
    frame_count: int, *,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> int:
    """Return the encoded latent length declared by the visual codec."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("reference video frame_count must be an integer")
    return video_temporal_mapping.latent_t(frame_count)


def resolve_reference_image_shape(width: int | float, height: int | float) -> tuple[int, int]:
    """Resolve the independent 2048-short-edge reference image geometry."""

    try:
        source_width = float(width)
        source_height = float(height)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference image dimensions must be positive and finite") from exc
    if (
        not math.isfinite(source_width)
        or not math.isfinite(source_height)
        or source_width <= 0.0
        or source_height <= 0.0
    ):
        raise ValueError("reference image dimensions must be positive and finite")
    if source_width > 4.0 * source_height or source_height > 4.0 * source_width:
        raise ValueError(
            "reference image ratio must be within the inclusive range 1:4 to 4:1, "
            f"got {source_width:g}x{source_height:g}"
        )
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(source_width, source_height)

    def nearest_32(value: float) -> int:
        return max(
            REFERENCE_IMAGE_MULTIPLE,
            int(round(value / REFERENCE_IMAGE_MULTIPLE)) * REFERENCE_IMAGE_MULTIPLE,
        )

    return nearest_32(source_width * scale), nearest_32(source_height * scale)


def _lookup_indexed(
    values: Sequence[Mapping[str, object]] | Mapping[int, Mapping[str, object]],
    index: int,
    *,
    name: str,
) -> Mapping[str, object]:
    if isinstance(values, Mapping):
        value = values.get(index)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        value = values[index] if index < len(values) else None
    else:
        raise TypeError(f"{name} must be an index-keyed mapping or sequence")
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}[{index}] must be a mapping")
    return value


def parse_ref_block_plan(
    conditions: Sequence[Mapping[str, object]],
    media_probes: Sequence[Mapping[str, object]] | Mapping[int, Mapping[str, object]],
    *,
    resolved_shapes: Sequence[Mapping[str, object]]
    | Mapping[int, Mapping[str, object]]
    | None = None,
    target_frame_count: int | None = None,
    target_duration_seconds: float | None = None,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> tuple[RefBlockSpec, ...]:
    """Parse conditions without regrouping them and freeze all probe decisions.

    Video dimensions must come from ``resolved_shapes`` or explicit
    ``resolved_width`` and ``resolved_height`` probe fields. They are never
    guessed from the target canvas. Image dimensions default to the native
    2048-short-edge geometry, and ``resolved_shapes`` may override them with
    any 32-aligned size. Plain video soundtrack inclusion comes only from the
    probe's explicit ``has_audio`` value.
    """

    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        raise TypeError("conditions must be a sequence")
    if target_frame_count is not None:
        video_latent_t_from_frame_count(
            target_frame_count, video_temporal_mapping=video_temporal_mapping
        )
    if target_duration_seconds is not None:
        target_duration_seconds = float(target_duration_seconds)
        if not math.isfinite(target_duration_seconds) or target_duration_seconds <= 0:
            raise ValueError("target_duration_seconds must be positive and finite")
        duration_frames = int(round(target_duration_seconds * REFERENCE_FPS))
        video_latent_t_from_frame_count(
            duration_frames, video_temporal_mapping=video_temporal_mapping
        )
        if target_frame_count is not None and duration_frames != target_frame_count:
            raise ValueError(
                "target duration and frame count disagree at 24 FPS, got "
                f"{duration_frames} and {target_frame_count}"
            )
        target_frame_count = duration_frames

    parsed: list[RefBlockSpec] = []
    for index, raw in enumerate(conditions):
        if not isinstance(raw, Mapping):
            raise ValueError(f"conditions[{index}] must be a mapping")
        kind = raw.get("kind", raw.get("type"))
        if kind not in ("image", "audio", "video", "video_audio"):
            raise ValueError(f"conditions[{index}] has unsupported kind {kind!r}")
        path_value = raw.get("path", raw.get("uri"))
        if not isinstance(path_value, (str, Path)):
            raise ValueError(f"conditions[{index}] requires path or uri")
        start = float(raw.get("start_time_seconds", 0.0))
        probe_raw = _lookup_indexed(media_probes, index, name="media_probes")
        probe = RefMediaProbe.from_mapping(probe_raw, path=f"media_probes[{index}]")

        resolved = (
            _lookup_indexed(resolved_shapes, index, name="resolved_shapes")
            if resolved_shapes is not None
            else probe_raw
        )
        resolved_width: int | None = None
        resolved_height: int | None = None
        frame_count: int | None = None
        if kind == "image":
            if probe.width is None or probe.height is None:
                raise ValueError(f"image media_probes[{index}] requires width and height")
            resolved_width, resolved_height = resolve_reference_image_shape(
                probe.width, probe.height
            )
            if resolved_shapes is not None:
                # An explicit shape overrides the native 2048-short-edge geometry.
                resolved_width = int(resolved.get("resolved_width", resolved.get("width")))
                resolved_height = int(resolved.get("resolved_height", resolved.get("height")))
        elif kind in ("video", "video_audio"):
            width_value = resolved.get("resolved_width", resolved.get("width"))
            height_value = resolved.get("resolved_height", resolved.get("height"))
            if isinstance(width_value, bool) or not isinstance(width_value, int):
                raise ValueError(f"resolved video width missing for conditions[{index}]")
            if isinstance(height_value, bool) or not isinstance(height_value, int):
                raise ValueError(f"resolved video height missing for conditions[{index}]")
            resolved_width, resolved_height = width_value, height_value
            local_frame_count = resolved.get("frame_count", target_frame_count)
            if isinstance(local_frame_count, bool) or not isinstance(
                local_frame_count, int
            ):
                raise ValueError(
                    f"resolved video frame_count missing for conditions[{index}]"
                )
            frame_count = local_frame_count
            video_latent_t_from_frame_count(
                frame_count, video_temporal_mapping=video_temporal_mapping
            )
            if target_frame_count is not None and frame_count != target_frame_count:
                raise ValueError(
                    f"conditions[{index}] frame_count {frame_count} disagrees with "
                    f"target {target_frame_count}"
                )

        parsed.append(
            RefBlockSpec(
                condition_index=index,
                kind=kind,
                path=str(path_value),
                start_time_seconds=start,
                probe=probe,
                resolved_width=resolved_width,
                resolved_height=resolved_height,
                resolved_frame_count=frame_count,
                video_temporal_mapping=video_temporal_mapping,
                resolved_duration_seconds=(
                    float(target_frame_count) / REFERENCE_FPS
                    if target_frame_count is not None
                    else None
                ),
            )
        )
    if not parsed:
        raise ValueError("Ref2VA requires at least one condition")
    return tuple(parsed)


@contextlib.contextmanager
def _scoped_encode_rng(seed: int, device: torch.device):
    devices = [device] if device.type == "cuda" and torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(int(seed))
        for forked_device in devices:
            with torch.cuda.device(forked_device):
                torch.cuda.manual_seed(int(seed))
        yield


@contextlib.contextmanager
def _video_vae_fp32(video_vae: Any):
    parameter = next(video_vae.parameters())
    previous_dtype = parameter.dtype
    if previous_dtype != torch.float32:
        video_vae.to(torch.float32)
    try:
        yield parameter.device
    finally:
        if previous_dtype != torch.float32:
            video_vae.to(previous_dtype)


class _AudioVAEDeterminismContext:
    _depth = 0
    _saved: tuple[Any, ...] | None = None

    def __enter__(self) -> _AudioVAEDeterminismContext:
        if type(self)._depth == 0:
            backends = torch.backends
            type(self)._saved = (
                backends.cuda.matmul.allow_tf32,
                backends.cudnn.allow_tf32,
                backends.cudnn.benchmark,
                backends.cudnn.deterministic,
                backends.cudnn.enabled,
                backends.cuda.flash_sdp_enabled(),
                backends.cuda.mem_efficient_sdp_enabled(),
                backends.cuda.math_sdp_enabled(),
            )
            backends.cuda.matmul.allow_tf32 = False
            backends.cudnn.allow_tf32 = False
            backends.cudnn.benchmark = False
            backends.cudnn.deterministic = True
            backends.cudnn.enabled = False
            backends.cuda.enable_flash_sdp(False)
            backends.cuda.enable_mem_efficient_sdp(False)
            backends.cuda.enable_math_sdp(True)
        type(self)._depth += 1
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        type(self)._depth -= 1
        if type(self)._depth == 0:
            saved = type(self)._saved
            if saved is None:
                raise RuntimeError("audio VAE determinism context lost its saved state")
            backends = torch.backends
            (
                backends.cuda.matmul.allow_tf32,
                backends.cudnn.allow_tf32,
                backends.cudnn.benchmark,
                backends.cudnn.deterministic,
                backends.cudnn.enabled,
                flash,
                memory_efficient,
                math_sdp,
            ) = saved
            backends.cuda.enable_flash_sdp(flash)
            backends.cuda.enable_mem_efficient_sdp(memory_efficient)
            backends.cuda.enable_math_sdp(math_sdp)
            type(self)._saved = None


def _module_stats(module: Any, channels: int, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(module, "latents_mean") or not hasattr(module, "latents_std"):
        raise AttributeError("VAE must expose loader-injected latents_mean and latents_std")
    mean = torch.as_tensor(module.latents_mean).detach().cpu().float()
    std = torch.as_tensor(module.latents_std).detach().cpu().float()
    if mean.numel() != channels or std.numel() != channels:
        raise ValueError(
            f"VAE latent stats must each contain {channels} values, got "
            f"{mean.numel()} and {std.numel()}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std == 0).any():
        raise ValueError("VAE latent stats must be finite and std must be non-zero")
    return mean.view(shape), std.view(shape)


def _patchify_visual(latent: torch.Tensor) -> torch.Tensor:
    from dev.yanzuolu.projects.minimax_h3.modeling.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )

    rows = minimax_h3_patchify_video_latent(
        latent, patch_size=list(VISUAL_PATCH_SIZE)
    )
    return rows.detach().cpu().float().contiguous()


def _prepare_image(spec: RefBlockSpec) -> Any:
    from PIL import Image, ImageOps

    assert spec.resolved_width is not None
    assert spec.resolved_height is not None
    with Image.open(spec.path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    target = (spec.resolved_width, spec.resolved_height)
    if image.size != target:
        image = image.resize(target, Image.Resampling.LANCZOS)
    if image.mode != "RGB" or image.size != target:
        raise ValueError("prepared reference image is not the resolved RGB geometry")
    return image


def _decode_video(spec: RefBlockSpec) -> Any:
    import numpy as np

    assert spec.resolved_width is not None
    assert spec.resolved_height is not None
    assert spec.resolved_frame_count is not None
    video_latent_t_from_frame_count(
        spec.resolved_frame_count, video_temporal_mapping=spec.video_temporal_mapping
    )
    filters = (
        f"fps={REFERENCE_FPS},"
        f"scale={spec.resolved_width}:{spec.resolved_height}:flags=lanczos,setsar=1"
    )
    command = ["ffmpeg", "-v", "error"]
    if spec.start_time_seconds > 0:
        command += ["-ss", f"{spec.start_time_seconds:.9g}"]
    command += [
        "-i",
        spec.path,
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        filters,
        "-frames:v",
        str(spec.resolved_frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    decoded = subprocess.run(command, check=True, capture_output=True)
    payload = decoded.stdout
    frame_bytes = spec.resolved_width * spec.resolved_height * 3
    if len(payload) % frame_bytes:
        raise ValueError(
            "ffmpeg returned a partial reference-video frame, got "
            f"{len(payload)} bytes for {spec.resolved_width}x{spec.resolved_height}"
        )
    actual_frames = len(payload) // frame_bytes
    if actual_frames != spec.resolved_frame_count:
        raise ValueError(
            "reference video ended before the resolved frame count or was silently "
            f"truncated, expected {spec.resolved_frame_count}, got {actual_frames}"
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(
        actual_frames, spec.resolved_height, spec.resolved_width, 3
    )


def _encode_visual_latent(
    video_vae: Any,
    media: Any,
    spec: RefBlockSpec,
    *,
    seed: int,
    latent_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode RGB media to normalized NCTHW using the native reference policy."""
    expected_t = (
        1
        if spec.kind == "image"
        else video_latent_t_from_frame_count(
            int(spec.resolved_frame_count), video_temporal_mapping=spec.video_temporal_mapping
        )
    )
    selected_encoder = None
    if latent_indices is not None:
        if latent_indices.ndim != 1 or latent_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("latent indices must be one-dimensional integer tensors")
        latent_indices = latent_indices.detach().cpu().to(torch.long)
        if bool(torch.any(latent_indices < 0)) or bool(torch.any(latent_indices >= expected_t)):
            raise ValueError("latent indices must lie inside the complete source video")
        if spec.kind != "image":
            selected_encoder = getattr(video_vae, "encode_videos_selected", None)
        if selected_encoder is not None:
            full_t = video_vae.processor.get_latent_length(int(spec.resolved_frame_count))
            if full_t != expected_t:
                raise ValueError(
                    f"visual VAE source geometry requires {expected_t} latents but codec produces {full_t}"
                )
    with _video_vae_fp32(video_vae) as device:
        with _scoped_encode_rng(seed, device):
            if spec.kind == "image":
                latent = video_vae.encode_images(media, use_fp16_latent=True)[0]
            elif selected_encoder is not None:
                latent = selected_encoder(media, [latent_indices], use_fp16_latent=True)[0]
            else:
                latent = video_vae.encode_videos(media, use_fp16_latent=True)[0]
    latent = latent.detach().cpu().float()
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if latent.ndim != 5 or int(latent.shape[0]) != 1:
        raise ValueError(f"unexpected visual VAE latent shape {list(latent.shape)}")
    if int(latent.shape[1]) != VIDEO_LATENT_CHANNELS:
        raise ValueError(
            f"visual VAE latent channels must be {VIDEO_LATENT_CHANNELS}, got "
            f"{int(latent.shape[1])}"
        )
    latent_t, latent_h, latent_w = map(int, latent.shape[2:])
    assert spec.resolved_width is not None
    assert spec.resolved_height is not None
    output_t = expected_t if selected_encoder is None else latent_indices.numel()
    expected_h = spec.resolved_height // 16
    expected_w = spec.resolved_width // 16
    if (latent_t, latent_h, latent_w) != (output_t, expected_h, expected_w):
        raise ValueError(
            "visual VAE latent geometry disagrees with resolved media, expected "
            f"{(output_t, expected_h, expected_w)}, got "
            f"{(latent_t, latent_h, latent_w)}"
        )
    if latent_indices is not None and selected_encoder is None:
        latent = latent.index_select(2, latent_indices)
    mean, std = _module_stats(
        video_vae,
        VIDEO_LATENT_CHANNELS,
        (1, VIDEO_LATENT_CHANNELS, 1, 1, 1),
    )
    return latent.sub_(mean).div_(std)


def _encode_visual(
    video_vae: Any,
    media: Any,
    spec: RefBlockSpec,
    *,
    seed: int,
) -> tuple[torch.Tensor, int, int, int]:
    latent = _encode_visual_latent(video_vae, media, spec, seed=seed)
    latent_t, latent_h, latent_w = map(int, latent.shape[2:])
    rows = _patchify_visual(latent)
    expected_rows = latent_t * (latent_h // 2) * (latent_w // 2)
    if list(rows.shape) != [expected_rows, 96]:
        raise ValueError(
            f"patchified visual rows must be [{expected_rows}, 96], got "
            f"{list(rows.shape)}"
        )
    return rows, latent_t, latent_h, latent_w


def _image_spec(path: str, *, width: int, height: int) -> RefBlockSpec:
    return RefBlockSpec(
        condition_index=0, kind="image", path=path, start_time_seconds=0.0,
        probe=RefMediaProbe(), resolved_width=width, resolved_height=height,
    )


def prepare_reference_image(path: str | Path, *, width: int, height: int) -> Any:
    """Open an image file as the RGB image a native image reference presents at this geometry."""
    return _prepare_image(_image_spec(str(path), width=width, height=height))


@torch.no_grad()
def encode_reference_image(video_vae: Any, image: Any, *, seed: int) -> torch.Tensor:
    """Encode one prepared RGB image to a normalized ``[C, 1, H, W]`` native image latent."""
    spec = _image_spec("image", width=image.width, height=image.height)
    return _encode_visual_latent(video_vae, image, spec, seed=seed)[0]


def _load_waveform(spec: RefBlockSpec) -> tuple[torch.Tensor, int]:
    import numpy as np

    if spec.kind == "audio":
        source_rate = spec.probe.audio_sample_rate
        if source_rate is None or source_rate <= 0:
            raise ValueError("pure audio requires its probed native sample rate")
    elif spec.kind in ("video", "video_audio"):
        if not spec.probe.has_audio:
            raise ValueError("cannot extract a soundtrack when probe has_audio is false")
        source_rate = VIDEO_SOUNDTRACK_SAMPLE_RATE
    else:
        raise ValueError(f"reference kind {spec.kind!r} has no audio chain")

    command = ["ffmpeg", "-v", "error"]
    if spec.start_time_seconds > 0:
        command += ["-ss", f"{spec.start_time_seconds:.9g}"]
    command += [
        "-i",
        spec.path,
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(AUDIO_CHANNELS),
    ]
    if spec.kind != "audio":
        command += ["-ar", str(source_rate)]
    if spec.resolved_duration_seconds is not None:
        command += ["-t", f"{spec.resolved_duration_seconds:.9g}"]
    command += ["-f", "f32le", "pipe:1"]
    decoded = subprocess.run(command, check=True, capture_output=True)
    payload = decoded.stdout
    frame_bytes = AUDIO_CHANNELS * torch.float32.itemsize
    if len(payload) % frame_bytes:
        raise ValueError(
            f"ffmpeg returned a partial stereo float32 sample frame, got {len(payload)} bytes"
        )
    waveform = torch.from_numpy(
        np.frombuffer(payload, dtype=np.float32).reshape(-1, AUDIO_CHANNELS).T.copy()
    )
    if waveform.ndim != 2 or list(waveform.shape[:1]) != [AUDIO_CHANNELS]:
        raise ValueError(f"unexpected decoded waveform shape {list(waveform.shape)}")
    if waveform.numel() == 0:
        raise ValueError(f"reference audio is empty after start/duration bounds: {spec.path}")
    return waveform, int(source_rate)


@functools.lru_cache(maxsize=8)
def _audio_resampler(source_rate: int) -> Any:
    import torchaudio

    return torchaudio.transforms.Resample(int(source_rate), AUDIO_SAMPLE_RATE)


def _encode_audio(audio_vae: Any, spec: RefBlockSpec) -> tuple[torch.Tensor, int, float]:
    waveform, source_rate = _load_waveform(spec)
    if source_rate != AUDIO_SAMPLE_RATE:
        waveform = _audio_resampler(source_rate)(waveform)
    if waveform.ndim != 2 or int(waveform.shape[0]) != AUDIO_CHANNELS:
        raise ValueError(f"resampled waveform must be stereo, got {list(waveform.shape)}")
    sample_count = int(waveform.shape[-1])
    if sample_count <= 0:
        raise ValueError("resampled waveform is empty")
    device = next(audio_vae.parameters()).device
    waveform = waveform.to(device)

    with _AudioVAEDeterminismContext():
        audio_data = audio_vae.preprocess(waveform.unsqueeze(1), AUDIO_SAMPLE_RATE)
        expected_padded = math.ceil(sample_count / AUDIO_HOP_LENGTH) * AUDIO_HOP_LENGTH
        if list(audio_data.shape[:2]) != [AUDIO_CHANNELS, 1]:
            raise ValueError(
                f"audio VAE preprocess must preserve [2,1,T], got {list(audio_data.shape)}"
            )
        if int(audio_data.shape[-1]) != expected_padded:
            raise ValueError(
                "audio VAE preprocess must only right-pad to an 800-sample boundary, "
                f"expected {expected_padded}, got {int(audio_data.shape[-1])}"
            )
        encoded = audio_vae.encoder(audio_data)
        if bool(getattr(audio_vae, "attn_proj", False)):
            encoded = audio_vae.pre_block(encoded.transpose(1, 2)).transpose(1, 2)
        if not hasattr(audio_vae, "mean_proj"):
            raise AttributeError("audio VAE must expose mean_proj")
        latent = audio_vae.mean_proj(encoded).float()

    if latent.ndim != 3:
        raise ValueError(f"audio VAE mean_proj must return rank 3, got {list(latent.shape)}")
    if int(latent.shape[0]) != AUDIO_CHANNELS:
        raise ValueError(
            f"audio VAE mean_proj batch must be {AUDIO_CHANNELS}, got {list(latent.shape)}"
        )
    if int(latent.shape[1]) != AUDIO_LATENT_CHANNELS:
        raise ValueError(
            f"audio VAE mean_proj channels must be {AUDIO_LATENT_CHANNELS}, "
            f"got {list(latent.shape)}"
        )
    latent = latent.transpose(1, 2).contiguous()
    latent = latent.detach().cpu().float().contiguous()
    audio_t = int(latent.shape[1])
    expected_audio_t = expected_padded // AUDIO_HOP_LENGTH
    if audio_t != expected_audio_t:
        raise ValueError(
            f"audio latent T {audio_t} does not match 800-sample padding "
            f"derived T {expected_audio_t}"
        )
    mean, std = _module_stats(
        audio_vae,
        AUDIO_LATENT_CHANNELS,
        (1, 1, AUDIO_LATENT_CHANNELS),
    )
    latent.sub_(mean).div_(std)
    rows = latent.reshape(AUDIO_CHANNELS * audio_t, AUDIO_LATENT_CHANNELS).contiguous()
    return rows, audio_t, sample_count / float(AUDIO_SAMPLE_RATE)


def _noise_visual_rows(
    clean_rows: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    target_latent_t: int,
    condition_count: int,
    seed: int,
    anchor: float,
) -> torch.Tensor:
    latent_t, latent_h, latent_w = shape
    full_t = int(target_latent_t) + int(condition_count)
    if full_t < latent_t:
        raise ValueError(
            f"visual condition latent_t {latent_t} exceeds noise draw length {full_t}"
        )
    if anchor == 1.0:
        return clean_rows.cpu().float().contiguous()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        1,
        VIDEO_LATENT_CHANNELS,
        full_t,
        latent_h,
        latent_w,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )[:, :, :latent_t]
    noise_rows = _patchify_visual(noise)
    if noise_rows.shape != clean_rows.shape:
        raise ValueError("visual condition noise rows do not match clean rows")
    return (anchor * clean_rows + (1.0 - anchor) * noise_rows).float().contiguous()


def _noise_audio_rows(clean_rows: torch.Tensor, *, seed: int, anchor: float) -> torch.Tensor:
    if anchor == 1.0:
        return clean_rows.cpu().float().contiguous()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    noise = torch.randn(
        clean_rows.shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return (anchor * clean_rows.cpu().float() + (1.0 - anchor) * noise).contiguous()


def _anchor(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    return value


def builder_ref_blocks(
    blocks: Sequence[EncodedRefBlock],
) -> tuple[dict[str, object], ...]:
    """Build the exact dictionaries accepted by packed_sequence_ref2va_blocks."""

    result: list[dict[str, object]] = []
    previous_index = -1
    for block in blocks:
        index = block.spec.condition_index
        if index <= previous_index:
            raise ValueError("encoded blocks must remain in strictly increasing request order")
        previous_index = index
        if block.spec.kind == "image":
            assert block.latent_h is not None and block.latent_w is not None
            result.append(
                {
                    "kind": "image",
                    "latent_h": int(block.latent_h),
                    "latent_w": int(block.latent_w),
                }
            )
        elif block.spec.kind == "audio":
            result.append({"kind": "audio", "ref_audio_t": int(block.audio_t)})
        else:
            assert block.latent_t is not None
            assert block.latent_h is not None
            assert block.latent_w is not None
            result.append(
                {
                    "kind": block.spec.kind,
                    "ref_audio_t": int(block.audio_t),
                    "latent_t": int(block.latent_t),
                    "latent_h": int(block.latent_h),
                    "latent_w": int(block.latent_w),
                }
            )
            if block.spec.video_temporal_mapping != H3_VIDEO_TEMPORAL_MAPPING:
                result[-1]["video_temporal_mapping"] = block.spec.video_temporal_mapping.to_dict()
    return tuple(result)


def _cat_rows(parts: Sequence[torch.Tensor], width: int) -> torch.Tensor:
    nonempty = [part for part in parts if part.numel()]
    if not nonempty:
        return torch.empty((0, width), dtype=torch.float32)
    return torch.cat(nonempty, dim=0).cpu().float().contiguous()


def _cat_metadata(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    nonempty = [part for part in parts if part.numel()]
    if not nonempty:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(nonempty, dim=0).cpu().float().contiguous()


@torch.inference_mode()
def encode_ref_block_plan(
    specs: Sequence[RefBlockSpec],
    *,
    video_vae: Any,
    audio_vae: Any,
    target_latent_t: int,
    encode_seed: int,
    noise_seed: int,
    visual_anchor: float = DEFAULT_VISUAL_ANCHOR,
    audio_anchor: float = DEFAULT_AUDIO_ANCHOR,
    decoded_visual_media: Mapping[int, Any] | None = None,
    retain_clean_visual: bool = False,
) -> EncodedReferencePlan:
    """Encode fixed condition rows from paths or request-indexed decoded RGB.

    Decoded media bypasses file loading without changing normalization, posterior
    sampling, condition noise, or the RGB presented to Qwen. Each visual block
    derives its posterior seed from ``encode_seed`` and its condition index.
    ``noise_seed`` controls condition augmentation independently of the posterior.
    """

    specs = tuple(specs)
    if not specs:
        raise ValueError("specs must not be empty")
    if isinstance(target_latent_t, bool) or not isinstance(target_latent_t, int):
        raise ValueError("target_latent_t must be a positive integer")
    if target_latent_t <= 0:
        raise ValueError("target_latent_t must be a positive integer")
    visual_anchor = _anchor(visual_anchor, "visual_anchor")
    audio_anchor = _anchor(audio_anchor, "audio_anchor")
    visual_condition_count = sum(
        spec.kind in ("image", "video", "video_audio") for spec in specs
    )

    blocks: list[EncodedRefBlock] = []
    qwen_media: list[Any] = []
    for expected_index, spec in enumerate(specs):
        if not isinstance(spec, RefBlockSpec):
            raise TypeError("specs entries must be RefBlockSpec instances")
        if spec.condition_index != expected_index:
            raise ValueError(
                "specs must preserve complete zero-based request order, expected "
                f"condition_index {expected_index}, got {spec.condition_index}"
            )

        visual_media = None
        clean_visual = torch.empty((0, 96), dtype=torch.float32)
        clean_audio = torch.empty((0, AUDIO_LATENT_CHANNELS), dtype=torch.float32)
        latent_t = latent_h = latent_w = None
        audio_t = 0
        block_visual_anchor: float | None = None
        block_audio_anchor: float | None = None

        if spec.kind == "image":
            visual_media = (
                _prepare_image(spec)
                if decoded_visual_media is None
                else decoded_visual_media[spec.condition_index]
            )
            clean_visual, latent_t, latent_h, latent_w = _encode_visual(
                video_vae, visual_media, spec,
                seed=combine_seed(encode_seed, "visual", spec.condition_index),
            )
            block_visual_anchor = visual_anchor
        elif spec.kind == "audio":
            clean_audio, audio_t, _ = _encode_audio(audio_vae, spec)
            block_audio_anchor = audio_anchor
        else:
            visual_media = (
                _decode_video(spec)
                if decoded_visual_media is None
                else decoded_visual_media[spec.condition_index]
            )
            clean_visual, latent_t, latent_h, latent_w = _encode_visual(
                video_vae, visual_media, spec,
                seed=combine_seed(encode_seed, "visual", spec.condition_index),
            )
            block_visual_anchor = visual_anchor
            block_audio_anchor = audio_anchor
            if spec.probe.has_audio:
                clean_audio, audio_t, _ = _encode_audio(audio_vae, spec)
            elif spec.kind == "video_audio":
                raise ValueError("video_audio probe requires a soundtrack")

        visual_rows = (
            _noise_visual_rows(
                clean_visual,
                shape=(int(latent_t), int(latent_h), int(latent_w)),
                target_latent_t=target_latent_t,
                condition_count=visual_condition_count,
                seed=noise_seed,
                anchor=visual_anchor,
            )
            if clean_visual.numel()
            else clean_visual
        )
        audio_rows = (
            _noise_audio_rows(clean_audio, seed=noise_seed, anchor=audio_anchor)
            if clean_audio.numel()
            else clean_audio
        )
        ordered_anchor_parts: list[torch.Tensor] = []
        if audio_rows.numel():
            ordered_anchor_parts.append(
                torch.full(
                    (int(audio_rows.shape[0]),), audio_anchor, dtype=torch.float32
                )
            )
        if visual_rows.numel():
            ordered_anchor_parts.append(
                torch.full(
                    (int(visual_rows.shape[0]),), visual_anchor, dtype=torch.float32
                )
            )
        row_anchors = _cat_metadata(ordered_anchor_parts)
        block = EncodedRefBlock(
            spec=spec,
            visual_media=visual_media,
            visual_rows=visual_rows.cpu().float().contiguous(),
            audio_rows=audio_rows.cpu().float().contiguous(),
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            visual_anchor=block_visual_anchor,
            audio_anchor=block_audio_anchor,
            row_anchors=row_anchors,
            row_timesteps=row_anchors.clone().contiguous(),
            clean_visual_rows=clean_visual if retain_clean_visual else None,
        )
        blocks.append(block)

        from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
            Ref2VAPresentationMedia,
        )

        qwen_media.append(
            Ref2VAPresentationMedia(
                kind=spec.kind,
                media=visual_media,
                has_audio=(
                    spec.probe.has_audio
                    if spec.kind == "video"
                    else True if spec.kind == "video_audio" else None
                ),
            )
        )

    visual_rows = _cat_rows([block.visual_rows for block in blocks], 96)
    audio_rows = _cat_rows(
        [block.audio_rows for block in blocks], AUDIO_LATENT_CHANNELS
    )
    visual_anchors = _cat_metadata(
        [
            torch.full(
                (int(block.visual_rows.shape[0]),),
                float(block.visual_anchor),
                dtype=torch.float32,
            )
            for block in blocks
            if block.visual_rows.numel()
        ]
    )
    audio_anchors = _cat_metadata(
        [
            torch.full(
                (int(block.audio_rows.shape[0]),),
                float(block.audio_anchor),
                dtype=torch.float32,
            )
            for block in blocks
            if block.audio_rows.numel()
        ]
    )
    packed_anchors = _cat_metadata([block.row_anchors for block in blocks])
    return EncodedReferencePlan(
        blocks=tuple(blocks),
        qwen_media=tuple(qwen_media),
        visual_rows=visual_rows,
        audio_rows=audio_rows,
        visual_row_anchors=visual_anchors,
        audio_row_anchors=audio_anchors,
        visual_row_timesteps=visual_anchors.clone().contiguous(),
        audio_row_timesteps=audio_anchors.clone().contiguous(),
        packed_row_anchors=packed_anchors,
        packed_row_timesteps=packed_anchors.clone().contiguous(),
        ref_blocks=builder_ref_blocks(blocks),
    )


__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_SAMPLE_RATE",
    "DEFAULT_AUDIO_ANCHOR",
    "DEFAULT_VISUAL_ANCHOR",
    "EncodedRefBlock",
    "EncodedReferencePlan",
    "REFERENCE_FPS",
    "RefBlockSpec",
    "RefKind",
    "RefMediaProbe",
    "builder_ref_blocks",
    "encode_ref_block_plan",
    "encode_reference_image",
    "parse_ref_block_plan",
    "prepare_reference_image",
    "resolve_reference_image_shape",
    "video_latent_t_from_frame_count",
]
