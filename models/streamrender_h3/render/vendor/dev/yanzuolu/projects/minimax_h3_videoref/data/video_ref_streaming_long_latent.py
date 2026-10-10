# SPDX-License-Identifier: Apache-2.0
"""Online, cycle-aligned crops from complete normalized GT mother clips."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dev.yanzuolu.common.data import WorkerResumeContext
from dev.yanzuolu.projects.minimax_h3.data.long_latent import (
    LONG_LATENT_FORMAT,
    LONG_REFERENCE_LATENT_FORMAT,
    aligned_crop_starts,
    crop_long_latent_entry,
    native_crop_frames_to_latents,
)
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import H3_VIDEO_TEMPORAL_MAPPING
from dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_latent import (
    VideoRefStreamingLatentT2AVDataset,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    MiniMaxH3Ref2VAPresentationProcessor,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_reference import prepare_reference_image


class LongLatentCatalogMixin:
    """Choose a mother clip uniformly, then choose one of its legal starts uniformly.

    ``latent_index_path`` is the offline encoder's JSONL catalog. ``num_frames``
    controls each online crop, must be ``17n + 5``, and is independent of the
    mother's length. Starts are native latent indices ``5k``. The host's
    iterator owns packing and worker replay. Its RNG also owns both crop
    draws and one language-dropout draw for the complete crop's caption
    timeline.

    Keep the catalog and corpus fixed while resuming a run. Entries shorter
    than the requested crop are excluded before sampling, without changing
    the probability of the remaining mother clips according to their length.

    ``reference_only`` reads a catalog of ``LONG_REFERENCE_LATENT_FORMAT``
    mother clips instead, for trainers that never read target values. Their
    crops carry zero target video and audio latents of the configured shapes.

    ``picture_index_path`` optionally names a JSONL listing every catalog entry
    as ``{latent_path, picture}`` plus an optional ``prompt``, with paths
    relative to its own directory. Each picture depicts its mother clip's
    frame 0, so every crop then starts there. Each sample carries ``picture``,
    the image prepared like a native image reference as ``[H, W, 3]`` uint8
    RGB, and its budget covers that leading picture in every window. A
    ``prompt`` replaces the mother's caption timeline with one caption for the
    whole crop. Text dropout keeps the picture. Pictures require Qwen visual
    context.
    """

    def __init__(
        self, seed: int, resume_context: WorkerResumeContext, *,
        latent_index_path: str, picture_index_path: str | None = None, reference_only: bool = False,
        **kwargs: Any,
    ) -> None:
        if picture_index_path is not None and not kwargs.get("qwen_visual_context", False):
            raise ValueError("picture conditions require qwen_visual_context")
        index_path = Path(latent_index_path).expanduser().resolve()
        self._latent_index_path = index_path
        self.reference_only = bool(reference_only)
        native_crop_frames_to_latents(kwargs["num_frames"])
        kwargs.setdefault("latent_dir", str(index_path.parent))
        super().__init__(seed, resume_context, **kwargs)
        if self.video_temporal_mapping != H3_VIDEO_TEMPORAL_MAPPING:
            raise ValueError("long GT latent crops require the native H3 temporal mapping")
        self._mother_starts: dict[Path, list[int]] = {}
        corpus_format = LONG_REFERENCE_LATENT_FORMAT if self.reference_only else LONG_LATENT_FORMAT
        for record in self._mother_catalog:
            if record["format"] != corpus_format:
                raise ValueError(f"{index_path}: unsupported long latent corpus format")
            starts = aligned_crop_starts(
                record["num_frames"], self.num_frames, audio_valid_samples=record["audio_valid_samples"],
            )
            if picture_index_path is not None:
                starts = [start for start in starts if start == 0]
            if not starts:
                continue
            for field in ("height", "width", "fps"):
                if record[field] != getattr(self, field):
                    raise ValueError(f"{index_path}: corpus {field} differs from the configured crop")
            if record["audio_sample_rate"] != 32000:
                raise ValueError("long GT audio must use the native 32000 Hz clock")
            path = Path(record["latent_path"])
            if not path.is_absolute():
                path = index_path.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path in self._mother_starts:
                raise ValueError(f"duplicate mother latent in {index_path}: {path}")
            self._mother_starts[path] = starts
        self.latent_paths = sorted(self._mother_starts)
        if not self.latent_paths:
            raise ValueError(f"{index_path}: no mother clip can provide {self.num_frames} frames")
        self._mother_pictures: dict[Path, dict[str, Any]] | None = None
        if picture_index_path is not None:
            sidecar = Path(picture_index_path).expanduser().resolve()
            with sidecar.open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            pictures = {(sidecar.parent / row["latent_path"]).resolve(): dict(row, picture=sidecar.parent / row["picture"])
                        for row in rows}
            missing = [path for path in self.latent_paths if path not in pictures]
            if missing:
                raise ValueError(f"{sidecar}: {len(missing)} catalog entries have no picture, e.g. {missing[0]}")
            self._mother_pictures = {path: pictures[path] for path in self.latent_paths}

    def _picture_prefix_rows(self, processor: MiniMaxH3Ref2VAPresentationProcessor) -> int:
        if self._mother_pictures is None:
            return 0
        latent_h, latent_w = self.height // self.spatial_vae_stride, self.width // self.spatial_vae_stride
        return (latent_h // 2) * (latent_w // 2) + processor.image_presentation_length(height=self.height, width=self.width)

    def _discover_latent_paths(self, latent_dir: str) -> list[Path]:
        """The explicit catalog owns filenames and locations independently of legacy globbing."""
        with self._latent_index_path.open() as handle:
            self._mother_catalog = [json.loads(line) for line in handle if line.strip()]
        return sorted((self._latent_index_path.parent / record["latent_path"]).resolve()
                      for record in self._mother_catalog)

    def _draw_latent_sample(self, rng: random.Random) -> dict[str, Any]:
        path = self.latent_paths[rng.randrange(len(self.latent_paths))]
        starts = self._mother_starts[path]
        start = starts[rng.randrange(len(starts))]
        mother = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        entry = crop_long_latent_entry(mother, start, self.num_frames, reference_only=self.reference_only)
        if tuple(entry["video"].shape) != self.latent_shape or tuple(entry["audio"].shape) != self.audio_shape:
            raise ValueError(f"{path}: cropped AV latents disagree with the configured geometry")
        segments = entry["caption_segments"]
        picture = None if self._mother_pictures is None else self._mother_pictures[path]
        if picture is not None and "prompt" in picture:
            segments = [dict(start_frame=0, end_frame=self.num_frames, prompt=picture["prompt"])]
        if self.text_dropout > 0.0 and rng.random() < self.text_dropout:
            segments = [dict(segment, prompt="") for segment in segments]
        layouts = [self._pack_sample(prompt=segment["prompt"]) for segment in segments]
        candidate = dict(max(layouts, key=lambda item: item["packing_rows"]))
        candidate.update(
            video_latents=entry["video"], audio_latents=entry["audio"], caption_segments=segments,
            caption_text_input_ids=[item["text_input_ids"] for item in layouts],
            caption_text_lens=[item["text_lens"] for item in layouts],
            source_crop_start_frame=entry["source_crop_start_frame"],
        )
        if picture is not None:
            candidate["picture"] = torch.from_numpy(np.array(prepare_reference_image(
                picture["picture"], width=self.width, height=self.height,
            )))
        return self._finalize_latent_sample(candidate, entry, path)



class VideoRefStreamingLongLatentT2AVDataset(LongLatentCatalogMixin, VideoRefStreamingLatentT2AVDataset):
    """Emit one pack of complete crops per iteration, whose windows the meta chooses."""

    _STATE_SCHEMA = "minimax_h3_videoref_streaming_long_latent_worker"
    _STATE_VERSION = 1


EntryClass = VideoRefStreamingLongLatentT2AVDataset

__all__ = ["LongLatentCatalogMixin", "VideoRefStreamingLongLatentT2AVDataset", "EntryClass"]
