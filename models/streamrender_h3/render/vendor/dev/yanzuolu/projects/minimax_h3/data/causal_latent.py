"""Stateful causal dataset over a precomputed MiniMax H3 latent corpus."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import get_worker_info

from dev.yanzuolu.common.data import WorkerResumeContext, WorkerStateEnvelope
from dev.yanzuolu.common.seed import yield_seed
from dev.yanzuolu.projects.minimax_h3.data.causal_text_only import CausalTextOnlyT2AVDataset


class CausalLatentT2AVDataset(CausalTextOnlyT2AVDataset):
    """Infinite worker-local stream backed by a precomputed MiniMax H3 corpus.

    Prompts come from the ``.pt`` entries themselves; this subclass never reads
    the parent's ``prompts`` field.

    Latents are already in normalized space -- the sampling side applied the
    shift/scale -- so consumers take ``video_latents`` / ``audio_latents`` as
    x0 directly. A corpus of raw VAE output would pass every check here and
    train against the wrong distribution.

    Resume replays the stream from ``(offset, avg_seqlen, cnt)``, which only
    reproduces the same clips while ``latent_paths`` is stable -- adding,
    removing or moving one ``prompt*.pt`` silently remaps every offset onto a
    different file. Nothing but the corpus may be written under ``latent_dir``.
    """

    _STATE_SCHEMA = "jarvis_minimax_h3_causal_latent_worker"
    _STATE_VERSION = 1

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        latent_dir: str,
        **kwargs: Any,
    ) -> None:
        self.latent_paths = self._discover_latent_paths(latent_dir)
        if not self.latent_paths:
            raise FileNotFoundError(f"No latent files found in: {latent_dir}")

        super().__init__(seed, resume_context, **kwargs)

    def _discover_latent_paths(self, latent_dir: str) -> list[Path]:
        """Return the stable sampling pool; indexed corpora may override discovery."""
        return sorted(Path(latent_dir).glob("**/prompt*.pt"))

    def _finalize_latent_sample(
        self, sample: dict[str, Any], entry: dict[str, Any], path: Path,
    ) -> dict[str, Any]:
        """Attach corpus-specific conditions before committing a packed sample."""
        return sample

    def _draw_latent_sample(self, rng: random.Random) -> dict[str, Any]:
        """Draw one normalized clip; subclasses may crop a longer corpus entry.

        Overriding _pack_sample instead is not available: the DMD meta model's
        _validation_packer instantiates the configured training dataset as a
        layout packer and calls _pack_sample with real prompt text, so
        reinterpreting that argument would break validation.
        """
        path = self.latent_paths[rng.randrange(len(self.latent_paths))]
        entry = torch.load(path, map_location="cpu", weights_only=True)
        # The layout the packer builds is derived from height/width/
        # num_frames, the tensors come off disk, and nothing downstream
        # compares them -- a mismatch surfaces as a reshape error deep
        # inside the model. The inherited width default (1280, so
        # latent_w 80) is already wrong for a 1376-wide corpus, so a
        # config that forgets the override lands here.
        if (
            entry["video"].shape != self.latent_shape
            or entry["audio"].shape != self.audio_shape
        ):
            raise ValueError(
                f"{path}: corpus latents {tuple(entry['video'].shape)} / "
                f"{tuple(entry['audio'].shape)} disagree with the configured "
                f"layout {self.latent_shape} / {self.audio_shape}"
            )
        prompt = entry["prompt"]
        if self.text_dropout > 0.0 and rng.random() < self.text_dropout:
            prompt = ""
        candidate = self._pack_sample(prompt=prompt)
        candidate["video_latents"] = entry["video"]
        candidate["audio_latents"] = entry["audio"]
        return self._finalize_latent_sample(candidate, entry, path)

    def _build_pack(
        self, offset: int, avg_seqlen: float, cnt: int,
    ) -> tuple[list[dict[str, Any]], tuple[int, float, int]]:
        """Draw one pack from ``offset`` and return the advanced draw state."""
        rng = random.Random(offset)
        samples: list[dict[str, Any]] = []
        cur_seqlen = 0
        num_retries = 0
        while len(samples) == 0 or cur_seqlen + avg_seqlen <= self.max_seqlen:
            candidate = self._draw_latent_sample(rng)
            candidate_seqlen = candidate["seqlens"]
            if (
                (
                    self.max_seqlen_per_sample is not None
                    and candidate_seqlen > self.max_seqlen_per_sample
                )
                or cur_seqlen + candidate_seqlen > self.max_seqlen
            ):
                if cur_seqlen + candidate_seqlen > self.max_seqlen:
                    num_retries += 1
                    if num_retries >= self.max_retries:
                        break
                continue
            avg_seqlen = avg_seqlen * cnt / (cnt + 1) + candidate_seqlen / (cnt + 1)
            cnt += 1
            cur_seqlen += candidate_seqlen
            num_retries = 0
            samples.append(candidate)
        return samples, (yield_seed(offset), avg_seqlen, cnt)

    def __iter__(self):
        worker_info = get_worker_info()
        physical_worker_id = worker_info.id if worker_info else 0
        physical_worker_count = worker_info.num_workers if worker_info else 1
        effective_workers = self.resume_context.num_workers or 1
        if physical_worker_count != effective_workers:
            raise ValueError(
                "worker topology mismatch: context expects "
                f"{effective_workers}, runtime has {physical_worker_count}"
            )
        logical_worker_id = (
            physical_worker_id + self.resume_context.next_logical_worker_id
        ) % physical_worker_count
        if logical_worker_id in self._decoded_worker_states:
            pack_state = self._decoded_worker_states[logical_worker_id]
        else:
            pack_state = self._initial_worker_state(logical_worker_id)

        while True:
            samples, pack_state = self._build_pack(*pack_state)
            batch = {key: [sample[key] for sample in samples] for key in samples[0]}
            state_after = self._encode_worker_state(logical_worker_id, *pack_state)
            yield WorkerStateEnvelope(batch, logical_worker_id, state_after)

class CausalLatentDiffusionForcingT2AVDataset(CausalLatentT2AVDataset):
    """Single-slot diffusion forcing over the MiniMax H3 latent corpus."""

    _STATE_SCHEMA = "jarvis_minimax_h3_causal_latent_df_worker"
    forcing = "diffusion"


__all__ = [
    "CausalLatentDiffusionForcingT2AVDataset",
    "CausalLatentT2AVDataset",
]
