# SPDX-License-Identifier: Apache-2.0
"""Streaming VideoRef training from normalized generated AV and reference latents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dev.yanzuolu.common.data import WorkerResumeContext
from dev.yanzuolu.projects.minimax_h3.data.streaming import StreamingLatentT2AVDataset
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping
from dev.yanzuolu.projects.minimax_h3_videoref.data.causal_video_ref_raw import decode_reference_video
from dev.yanzuolu.projects.minimax_h3_videoref.data.flow_geometry import load_flow_condition_maps
from dev.yanzuolu.projects.minimax_h3_videoref.data.pixel_condition import decode_condition_frames
from dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming import _VideoRefStreamingLayout
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_reference import (
    RefBlockSpec, RefMediaProbe, _decode_video,
)


class VideoRefStreamingLatentT2AVDataset(_VideoRefStreamingLayout, StreamingLatentT2AVDataset):
    """Read paired latent clips and retain original reference pixels for Qwen.

    The meta model selects windows and prepares visual conditions before SP
    synchronization when Qwen visual context is enabled. Without
    ``qwen_reference_video`` Qwen does not see the reference, so its pixels are not decoded.
    ``reference_video_rows`` false keeps the reference latents in every sample
    and budgets windows without reference rows.
    The inherited latent reader owns packing, dropout and worker replay. Point
    latent_dir at one output variant to avoid mixing backbone and EMA corpora.
    ``geometry_root`` adds each sample's backward-flow AdaLN condition maps as
    ``adaln_condition_maps`` [T, 20, H/8, W/8] float16, one per video latent.
    They are read from ``<geometry_root>/<artifact_id>/geometry.safetensors``,
    whose frames follow the reference video, from the entry's
    ``reference_start_frame`` on.
    ``pixel_condition`` ``{root, filename}`` adds each sample's condition video
    as ``condition_frames`` uint8 [F, H, W, 3], the frames of
    ``<root>/<artifact_id>/<filename>`` on the reference video's timeline from
    the entry's ``reference_start_frame`` on, at the target resolution. Only
    the samples a pack keeps are decoded, once per pack.
    """

    _STATE_SCHEMA = "minimax_h3_videoref_streaming_latent_worker"
    _STATE_VERSION = 1

    def __init__(
        self, seed: int, resume_context: WorkerResumeContext, *,
        reference_height: int | None = None, reference_width: int | None = None,
        fps: int = MINIMAX_H3_SUPPORTED_FPS,
        qwen_visual_context: bool = False, qwen_processor_path: str | None = None,
        fixed_window_rope: bool = False, keep_sink_reference: bool = True, qwen_reference_video: bool = True,
        reference_video_rows: bool = True, teacher_streaming_config: dict[str, Any] | None = None,
        teacher_reference_video_rows: bool | None = None,
        geometry_root: str | None = None, pixel_condition: dict[str, str] | None = None, **kwargs: Any,
    ) -> None:
        if fps != MINIMAX_H3_SUPPORTED_FPS:
            raise ValueError("streaming latent corpora require 24 FPS video")
        self.teacher_streaming_config = None if teacher_streaming_config is None else dict(teacher_streaming_config)
        self.teacher_reference_video_rows = teacher_reference_video_rows
        self._configure_streaming_conditioning(
            qwen_visual_context=qwen_visual_context, qwen_processor_path=qwen_processor_path,
            fixed_window_rope=fixed_window_rope, keep_sink_reference=keep_sink_reference,
            qwen_reference_video=qwen_reference_video, reference_video_rows=reference_video_rows,
        )
        super().__init__(seed, resume_context, **kwargs)
        self.fps = int(fps)
        self.reference_height = self.height if reference_height is None else int(reference_height)
        self.reference_width = self.width if reference_width is None else int(reference_width)
        alignment = self.spatial_vae_stride * 2
        if any(value <= 0 or value % alignment for value in (self.reference_height, self.reference_width)):
            raise ValueError("reference dimensions must be positive and aligned to video latent patches")
        self.reference_latent_shape = (
            self.video_latent_channels, self.latent_t,
            self.reference_height // self.spatial_vae_stride,
            self.reference_width // self.spatial_vae_stride,
        )
        self.geometry_root = None if geometry_root is None else Path(geometry_root).expanduser().resolve()
        self.pixel_condition = None if pixel_condition is None else (
            Path(pixel_condition["root"]).expanduser().resolve(), str(pixel_condition["filename"]))
        self._set_streaming_window_budget()

    def _pack_sample(self, *, prompt: str) -> dict[str, Any]:
        sample = self._pack_video_ref_sample(prompt=prompt)
        rows = int(sample["packing_rows"])
        if rows > self.max_seqlen or (
            self.max_seqlen_per_sample is not None and rows > self.max_seqlen_per_sample
        ):
            raise ValueError(f"video-reference latent sample needs {rows} rows, exceeding its packing budget")
        sample["seqlens"] = rows
        return sample

    def _build_pack(self, *state: Any) -> tuple[list[dict[str, Any]], Any]:
        """Decode the condition video of every sample the pack keeps."""
        samples, state = super()._build_pack(*state)
        if self.pixel_condition is not None:
            for sample in samples:
                path, start_frame = sample.pop("condition_video")
                sample["condition_frames"] = decode_condition_frames(
                    path, start_frame=start_frame, num_frames=self.num_frames,
                    height=self.height, width=self.width, fps=self.fps,
                )
        return samples, state

    def _finalize_latent_sample(
        self, sample: dict[str, Any], entry: dict[str, Any], path: Path,
    ) -> dict[str, Any]:
        reference = entry["reference_video"]
        if tuple(reference.shape) != self.reference_latent_shape:
            raise ValueError(f"{path}: reference latents disagree with the configured geometry")
        if "video_temporal_mapping" in entry and (
            VideoTemporalMapping.from_dict(entry["video_temporal_mapping"]) != self.video_temporal_mapping
        ):
            raise ValueError(f"{path}: corpus and training video temporal mappings differ")
        for key in ("num_frames", "fps", "reference_height", "reference_width"):
            if key in entry and entry[key] != getattr(self, key):
                raise ValueError(f"{path}: corpus {key} disagrees with the training geometry")
        sample["reference_video_latents"] = reference
        if self.geometry_root is not None:
            maps = load_flow_condition_maps(
                self.geometry_root / entry["artifact_id"] / "geometry.safetensors",
                start_frame=int(entry.get("reference_start_frame", 0)), latent_count=self.latent_t,
                video_temporal_mapping=self.video_temporal_mapping,
            )
            cell = self.spatial_vae_stride // 2
            if tuple(maps.shape[2:]) != (self.height // cell, self.width // cell):
                raise ValueError(f"{path}: geometry grid {tuple(maps.shape[2:])} is not the frames at 1/{cell}")
            sample["adaln_condition_maps"] = maps
        if self.pixel_condition is not None:
            root, filename = self.pixel_condition
            sample["condition_video"] = (str(root / entry["artifact_id"] / filename),
                                         int(entry.get("reference_start_frame", 0)))
        if self.qwen_visual_context and self.qwen_reference_video:
            reference_path = Path(entry["reference_video_path"]).expanduser()
            if not reference_path.is_absolute():
                reference_path = path.parent / reference_path
            if "reference_start_time_seconds" in entry:
                # Native generation seeks before resampling. Preserve that
                # source timeline, including offsets between frame-grid ticks.
                spec = RefBlockSpec(
                    condition_index=0, kind="video", path=str(reference_path),
                    start_time_seconds=float(entry["reference_start_time_seconds"]),
                    probe=RefMediaProbe(has_audio=False),
                    resolved_width=self.reference_width, resolved_height=self.reference_height,
                    resolved_frame_count=self.num_frames, video_temporal_mapping=self.video_temporal_mapping,
                )
                media = _decode_video(spec)
                sample["reference_video_pixels"] = torch.from_numpy(media.copy()).permute(3, 0, 1, 2).float().div_(255)
            else:
                sample["reference_video_pixels"] = decode_reference_video(
                    reference_path, start_frame=int(entry.get("reference_start_frame", 0)),
                    num_frames=self.num_frames, fps=self.fps,
                    height=self.reference_height, width=self.reference_width,
                )
        return sample


EntryClass = VideoRefStreamingLatentT2AVDataset

__all__ = ["VideoRefStreamingLatentT2AVDataset", "EntryClass"]
