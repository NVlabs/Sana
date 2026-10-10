# SPDX-License-Identifier: Apache-2.0
"""Native Ref2VA validation of a long clip as a chain of bidirectional segments.

A request carries ``segments``, each a normal Ref2VA request. In segments after
the first, an image condition ``{type: image, previous_frame: last}`` stands for
the last decoded frame of the previous segment, so every segment opens on the
instant where the previous one ended. Every segment samples
``validation.num_frames`` frames. The stitched clip drops each later segment's
first frame and one frame of its audio, then is written as one mp4 per request.
Sampling seeds follow the prompt index unless a request sets ``seed_key``, which
keeps a request's result independent of its position in the request list.
``engine.early_stop_hours`` stops starting a request that could not finish within
that many hours, judged by the longest request so far::

    validation:
      requests:
        - prompt: caption for W&B      # optional, defaults to segment 0's prompt
          seed_key: clip-0001          # optional, replaces the prompt index in seeds
          segments:
            - {prompt, conditions, media_probes, resolved_shapes}
            - prompt: ...
              conditions:
                - {type: image, previous_frame: last}
                - {type: video, path: ..., start_time_seconds: ...}
              media_probes: [{width: 1344, height: 768}, ...]
              resolved_shapes: [{resolved_width: 1344, resolved_height: 768}, ...]
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from dev.yanzuolu.common import media
from dev.yanzuolu.common.distributed import ops
from dev.yanzuolu.common.logging import get_logger
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import RandomState, combine_seed
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_ref2va import (
    MiniMaxH3Ref2VABase,
)

logger = get_logger()


def _chain_prompt(request: Mapping[str, Any]) -> str:
    prompt = request.get("prompt", None)
    return prompt if prompt is not None else request["segments"][0]["prompt"]


def _segment_request(
    segment: Mapping[str, Any], previous_frame: Path | None, *, ref_noise_seed: int
) -> dict[str, Any]:
    """Resolve ``previous_frame`` conditions to ``previous_frame``'s path.

    ``ref_noise_seed`` applies unless the segment sets its own.
    """
    conditions = []
    for condition in segment["conditions"]:
        condition = dict(condition)
        if "previous_frame" in condition:
            assert condition.pop("previous_frame") == "last", (
                "previous_frame supports only 'last'"
            )
            assert previous_frame is not None, "the first segment has no previous frame"
            condition["path"] = str(previous_frame)
        conditions.append(condition)
    return {"ref_noise_seed": ref_noise_seed, **segment, "conditions": conditions}


def _save_last_frame(frames: torch.Tensor, path: Path) -> None:
    """Write the last frame of ``[1, 3, F, H, W]`` frames in [0, 1] as an RGB PNG."""
    image = frames[0, :, -1].float().mul(255).round().clamp(0, 255).to(torch.uint8)
    Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(path)


def _early_stop_due(started: float, hours: float | None, longest: float) -> bool:
    """Whether a request lasting ``longest`` seconds on any rank would end past ``hours``."""
    if hours is None:
        return False
    due = torch.tensor(
        int(time.monotonic() - started + longest >= float(hours) * 3600), dtype=torch.int32
    )
    ops.all_reduce_max(due)
    return bool(due.item())


def _stitch_segments(
    frames: Sequence[torch.Tensor],
    waveforms: Sequence[torch.Tensor],
    *,
    samples_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate segments in time, dropping the frame each later one shares.

    Segment ``k > 0`` opens on the instant the previous segment closed, so its
    first frame and the audio samples of that frame are dropped.
    """
    video = torch.cat([frames[0], *(clip[:, :, 1:] for clip in frames[1:])], dim=2)
    audio = torch.cat(
        [waveforms[0], *(wave[..., samples_per_frame:] for wave in waveforms[1:])], dim=-1
    )
    return video, audio


class MiniMaxH3Ref2VAChain(MiniMaxH3Ref2VABase):
    """Ref2VA validation that chains segments through their last decoded frame."""

    def __init__(self, config: Any) -> None:
        self._chain_started = time.monotonic()
        super().__init__(config)

    @execution_phase(ExecutionPhase.VALIDATION)
    @torch.no_grad()
    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Sample every segment of a request in order, then save one stitched mp4.

        Each segment is collective across the SP group. Its main rank decodes
        the segment and writes the last frame as a PNG, and every rank reads it
        after a barrier to build the next segment. All requests have the same
        segment count, so every group reaches the same barriers. An empty
        request list samples nothing.
        """
        config = ctx["config"]
        models = ctx["models"]
        step = int(ctx["step"])
        validation = config.validation
        requests = self._validation_requests(validation)
        segment_counts = {len(request["segments"]) for request in requests}
        assert len(segment_counts) <= 1, (
            f"every chain request must have the same segment count, got {segment_counts}"
        )
        assert int(validation.get("batch_size", 1)) == 1, "chain validation requires batch_size 1"
        assert not validation.get("save_latent", False), "chain validation does not save latents"

        rank = ops.get_rank()
        sp_size = self._sp_size()
        group_index = rank // sp_size
        num_groups = ops.get_world_size() // sp_size
        is_group_main = rank % sp_size == 0

        variants = self._validation_model_variants(models, validation)
        media_dir = logger.iter_dir("media", step)
        completed = ops.all_gather_object(
            self._completed_prompt_ids(media_dir, [name for name, _ in variants])
        )[0]
        pending = [
            (index, request) for index, request in enumerate(requests) if index not in completed
        ]
        entries = self._split_validation_prompts(pending, group_index, num_groups, _chain_prompt)

        original_training = [(model, model.training) for _, model in variants]
        saved_items: list[tuple[str, str, str, str]] = []
        fps = int(validation.get("fps", MINIMAX_H3_SUPPORTED_FPS))
        shift_seed = bool(validation.get("shift_seed", True))
        ref_noise_seed = int(validation.get("ref_noise_seed", validation.seed))
        sample_rate = int(models["audio_vae"].sample_rate)
        try:
            for _, model in variants:
                model.eval()
            longest = 0.0
            for entry in entries:
                if _early_stop_due(
                    self._chain_started, config.engine.get("early_stop_hours", None), longest
                ):
                    logger.info("chain validation stops early before prompt %d", entry["prompt_idx"])
                    break
                entry_started = time.monotonic()
                prompt_idx = entry["prompt_idx"]
                seed_key = entry["request"].get("seed_key", prompt_idx if shift_seed else 0)
                for variant_name, model in variants:
                    stem = f"validation/{variant_name}/prompt{prompt_idx:04d}"
                    if not entry["should_save"]:
                        # A padded duplicate must not share its frame files with
                        # the group that saves this prompt.
                        stem = f"{stem}_pad{group_index}"
                    previous_frame = None
                    segment_frames: list[torch.Tensor] = []
                    segment_waveforms: list[torch.Tensor] = []
                    seam_frames: list[Path] = []
                    record: dict[str, list[Any]] = {"seeds": [], "segment_seconds": []}
                    for segment_index, segment in enumerate(entry["request"]["segments"]):
                        segment_started = time.monotonic()
                        inputs = self._validation_inputs_for_requests(
                            config,
                            models,
                            [
                                _segment_request(
                                    segment,
                                    previous_frame,
                                    ref_noise_seed=combine_seed(
                                        ref_noise_seed, "segment", segment_index
                                    ),
                                )
                            ],
                        )
                        seed = combine_seed(
                            int(validation.seed),
                            "validation",
                            seed_key,
                            variant_name,
                            "segment",
                            segment_index,
                        )
                        rng = RandomState(seed)
                        latents = self._sample_latents(model, inputs, rng, rngs=[rng])
                        previous_frame = media_dir / f"{stem}_seg{segment_index}_last.png"
                        if is_group_main:
                            frames, waveform = self._decode_latents(models, latents)
                            previous_frame.parent.mkdir(parents=True, exist_ok=True)
                            _save_last_frame(frames, previous_frame)
                            seam_frames.append(previous_frame)
                            if entry["should_save"]:
                                segment_frames.append(
                                    self._chain_segment_frames(frames, inputs).cpu()
                                )
                                segment_waveforms.append(waveform.cpu())
                            del frames, waveform
                        del latents, inputs
                        ops.barrier()
                        record["seeds"].append(seed)
                        record["segment_seconds"].append(time.monotonic() - segment_started)

                    # Every rank has read the seam frames once past the last barrier.
                    for path in seam_frames:
                        path.unlink()
                    if not segment_frames:
                        continue
                    frames, waveform = _stitch_segments(
                        segment_frames, segment_waveforms,
                        samples_per_frame=round(sample_rate / fps),
                    )
                    del segment_frames, segment_waveforms
                    item = self._save_chain(
                        config, entry, variant_name, frames, waveform,
                        media_dir=media_dir, stem=stem, fps=fps, sample_rate=sample_rate,
                        record=record,
                    )
                    if item is not None:
                        saved_items.append(item)
                    del frames, waveform
                longest = max(longest, time.monotonic() - entry_started)

            gathered = ops.all_gather_object(saved_items)
            try:
                if rank == 0:
                    self._log_validation_videos(
                        [item for rank_items in gathered for item in rank_items],
                        variants, media_dir, validation, step=step, fps=fps,
                    )
            finally:
                ops.barrier()
        finally:
            for model, training in original_training:
                model.train(training)
        return ctx

    def _chain_segment_frames(self, frames: torch.Tensor, inputs: Any) -> torch.Tensor:
        """Frames of one decoded segment as stitched, with its reference panels."""
        return self._validation_frames(frames, inputs, index=0)

    def _save_chain(
        self,
        config: Any,
        entry: Mapping[str, Any],
        variant_name: str,
        frames: torch.Tensor,
        waveform: torch.Tensor,
        *,
        media_dir: Path,
        stem: str,
        fps: int,
        sample_rate: int,
        record: Mapping[str, Sequence[Any]],
    ) -> tuple[str, str, str, str] | None:
        """Write one stitched chain and return its W&B item, or None to log nothing.

        ``record`` holds each segment's sampling seed and wall-clock seconds.
        """
        validation = config.validation
        name = f"{stem}.mp4"
        path = media_dir / name
        media.save_video(
            frames,
            path,
            fps=fps,
            nrow=int(validation.get("nrow", 1)),
            crf=int(validation.get("crf", 18)),
            audio_tensor=waveform,
            audio_sample_rate=sample_rate,
            normalize=True,
            value_range=(0, 1),
        )
        return variant_name, name, str(path), f"[{variant_name}] {entry['prompt']}"

    def _log_validation_videos(
        self,
        items: list[tuple[str, str, str, str]],
        variants: list[tuple[str, Any]],
        media_dir: Path,
        validation: Any,
        *,
        step: int,
        fps: int,
    ) -> None:
        """Log the per-prompt mp4s and their grid preview per variant."""
        for variant_name, _ in variants:
            variant_items = [item for item in items if item[0] == variant_name]
            if not variant_items:
                continue
            logger.log_video(
                {name: Path(path) for _, name, path, _ in variant_items},
                step=step,
                collection=f"validation/{variant_name}/videos",
                captions={name: caption for _, name, _, caption in variant_items},
                fps=fps,
            )
            if not validation.get("save_grid", True):
                continue
            grid_name = f"validation/{variant_name}/grid.mp4"
            grid_path = media_dir / grid_name
            try:
                self._save_grid(
                    [Path(path) for _, _, path, _ in variant_items],
                    grid_path,
                    fps=fps,
                    nrow=int(validation.get("nrow", 4)),
                    crf=int(validation.get("crf", 18)),
                )
                logger.log_video(
                    {grid_name: grid_path},
                    step=step,
                    collection=f"validation/{variant_name}/grid",
                    fps=fps,
                )
            except Exception as exc:
                # The grid is a preview; the per-prompt files are the real artifact.
                logger.warning("validation grid build failed: %s", exc)


__all__ = ["MiniMaxH3Ref2VAChain"]
