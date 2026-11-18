"""Bidirectional multi-step sampling for MiniMax-H3: the released checkpoint,
in this framework, with no training logic attached.

``MiniMaxH3Base`` subclasses ``CausalMiniMaxH3DMD`` to reuse the shared
primitives -- packed layout derivation (``_build_inputs``), text encoding
(``_encode_prompts``), the dense bidirectional forward (``_bidirectional_forward``,
the very call the DMD critics/teacher use), VAE decoding (``_decode_latents``)
and the validation bookkeeping (``_validation_*``, resume, collective split,
mp4/grid logging). It deliberately does NOT call the parent ``__init__``: the
DMD constructor asserts training knobs (``fake_loss_type``, ``dmd_loss``, ...)
that an inference config has no business carrying.

Sampling is the global, non-causal loop: every step is one full-sequence
dense forward over ``[text | all audio | all video]``. There is no NaiveCache,
no chunk loop and no clean/noise cache roles -- causality is a generation-time
constraint of the *student*, not of the bidirectional model. This is exactly
what ``_bidirectional_forward`` already implements.

The sampling grid is the trial's own: ``diffusion.sampling_timesteps`` and
``diffusion.audio_sampling_timesteps`` (same step count, video ``shift`` and
audio ``shift`` of the released pipeline -- the official serving config is 50
steps with video flow_shift 12.0 and audio flow_shift 3.0). There is no CFG:
H3's released checkpoint is guidance-distilled, one positive denoise branch.

An inference trial needs only the sampling-side config::

    diffusion:
      sampling_timesteps:      # video grid (e.g. trailing, 50 steps, shift 12.0)
      audio_sampling_timesteps:# audio grid (same steps, shift 3.0)
      schedule:                # lerp, T 1.0, pred_type x_0
      sampler:                 # euler eta=0

    models:
      backbone:      bidirectional MiniMaxH3X0DiT + MiniMaxH3WeightLoader
      text_encoder:  MiniMaxH3TextEncoder
      video_vae:     MiniMaxH3VideoVAE
      audio_vae:     MiniMaxH3AudioVAE

    validation:
      prompts / prompt_path, num_frames, height, width, fps, seed, ...

    data:            # only the dataset class is instantiated, as the packer
      module/class_name/args (tokenizer_path, height, width, num_frames, ...)

The first consumer is FVD reference generation: fix prompts and seeds, sample
with the bidirectional model, extract I3D/VideoMAE features offline. The same
meta model doubles as the student's standalone evaluation entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from dev.yanzuolu.common import media
from dev.yanzuolu.common.diffusion import build_diffusion
from dev.yanzuolu.common.diffusion.schedule import PredictionType
from dev.yanzuolu.common.distributed import ops
from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.logging import get_logger
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import RandomState, combine_seed

from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd import (
    CausalMiniMaxH3DMD,
    _RolloutX0s,
)
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS

logger = get_logger()


class MiniMaxH3Base(CausalMiniMaxH3DMD):
    """Bidirectional multi-step T2AV sampling; no training primitives.

    Reuses the parent's layout/forward/decode/validation helpers but owns its
    own ``__init__`` (sampling diffusion nodes only) and ``validate`` (global
    dense sampling instead of the chunk-causal rollout).
    """

    def __init__(self, config: Any) -> None:
        # Deliberately not super().__init__: the DMD constructor requires
        # training-only knobs this config does not carry.
        self.config = config
        diffusion = build_diffusion(config.diffusion)
        self.sampling_timesteps = diffusion["sampling_timesteps"]
        self.audio_sampling_timesteps = diffusion["audio_sampling_timesteps"]
        self.schedule = diffusion["schedule"]
        self.sampler = diffusion["sampler"]
        # Every node is a MiniMaxH3X0Model: the prediction already IS x0, and
        # the sampler converts through this schedule on every step.
        assert self.schedule.pred_type == PredictionType.x_0, (
            f"diffusion.schedule.pred_type must be x_0 because every MiniMax-H3 "
            f"model node returns x0, got {self.schedule.pred_type}"
        )
        # Built lazily on the first validation, exactly like the DMD parent:
        # the dataset owns the packed layout, so validation packs its prompts
        # through the very same class rather than re-deriving it here.
        self._validation_packer_cache: Any = None

    # ------------------------------------------------------------ sampling --
    @torch.no_grad()
    def _sample_latents(
        self, model: Any, inputs: Any, rng: Any,
        *, rngs: Sequence[Any] | None = None,
    ) -> _RolloutX0s:
        """Global multi-step flow sampling; every step is one dense forward.

        One shared step index drives both grids, each with its own shift --
        H3's native "steps aligned, sigmas not" pairing, so the pair the model
        sees at every step is one its own sampler produces. The final step
        lands on the clean timestep 0: the grid's last ``s`` is the bound, and
        the eta=0 step collapses to the predicted x0.

        ``rngs`` (optional, one stream per sample) gives every sample its own
        deterministic stream inside ONE batched sampling pass -- the
        validation pattern (per-prompt seeds, batch is just a shared forward,
        so changing batch_size never perturbs a prompt's noise).
        """
        device = get_device()
        sample_rngs = rngs if rngs is not None else [rng] * inputs.batch_size
        assert len(sample_rngs) == inputs.batch_size, (
            f"rngs must have one stream per sample: {len(sample_rngs)} vs "
            f"batch_size {inputs.batch_size}"
        )
        video_noises = [
            torch.empty(layout.latent_shape, device=device).normal_(
                generator=self._generator(sample_rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]
        audio_noises = [
            torch.empty(layout.audio_shape, device=device).normal_(
                generator=self._generator(sample_rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]

        self.sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        self.audio_sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        video_grid = self.sampling_timesteps.timesteps
        audio_grid = self.audio_sampling_timesteps.timesteps
        assert video_grid.dim() == 1 and audio_grid.dim() == 1, (
            "dynamic-shift sampling grids are not supported: sampling advances "
            "the video and audio grids by a shared step index"
        )
        assert video_grid.numel() == audio_grid.numel(), (
            "sampling_timesteps and audio_sampling_timesteps must have the same "
            f"number of steps, got {video_grid.numel()} and {audio_grid.numel()}"
        )
        num_steps = int(video_grid.numel())
        # The sampler natively accepts a per-sample rng list.
        step_rng = rngs if rngs is not None else rng

        video_xts, audio_xts = video_noises, audio_noises
        for step in range(num_steps):
            video_t = video_grid[step].expand(inputs.batch_size).to(device)
            audio_t = audio_grid[step].expand(inputs.batch_size).to(device)
            video_s = self.sampling_timesteps.get_next_timesteps(video_t)
            audio_s = self.audio_sampling_timesteps.get_next_timesteps(audio_t)

            video_pred, audio_pred = self._bidirectional_forward(
                model,
                inputs,
                video_xts=video_xts,
                audio_xts=audio_xts,
                video_timesteps=video_t,
                audio_timesteps=audio_t,
            )
            video_xts = self.sampler.step_to(
                pred=video_pred, x_t=video_xts, t=video_t, s=video_s,
                rng=step_rng, seqlens=inputs.seqlens,
            )
            audio_xts = self.sampler.step_to(
                pred=audio_pred, x_t=audio_xts, t=audio_t, s=audio_s,
                rng=step_rng, seqlens=inputs.seqlens,
            )

        return _RolloutX0s(
            video=video_xts, audio=audio_xts, video_eps=None, audio_eps=None
        )

    # ---------------------------------------------------------- validation --
    def _validation_frames(
        self, frames: torch.Tensor, inputs: Any, *, index: int
    ) -> torch.Tensor:
        """Frames written to one validation mp4; subclasses may add panels beside the sample."""
        del inputs, index
        return frames

    @execution_phase(ExecutionPhase.VALIDATION)
    @torch.no_grad()
    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """One bidirectional sample per prompt, then both VAEs and an muxed mp4.

        Same contract as the DMD parent's validate -- every rank walks the same
        number of entries (the split pads by repeating the last one with
        ``should_save=False``), because the sampling is collective under FSDP
        and sequence parallelism; the only difference is ``_sample_latents``
        replacing the chunk-causal rollout.
        """
        config = ctx["config"]
        models = ctx["models"]
        step = int(ctx["step"])
        validation = config.validation
        requests = self._validation_requests(validation)
        assert requests, "validation requires at least one prompt"

        rank = ops.get_rank()
        sp_size = self._sp_size()
        group_index = rank // sp_size
        num_groups = ops.get_world_size() // sp_size
        is_group_main = rank % sp_size == 0

        variants = self._validation_model_variants(models, validation)
        media_dir = logger.iter_dir("media", step)
        # Latent mode writes the sampled x0 straight to disk instead of decoding
        # it. That is what a distillation corpus consumes, and it drops both VAEs
        # and the mp4 encode from the per-sample cost.
        save_latent = bool(validation.get("save_latent", False))
        # A corpus does not belong under runs/<exp>/media/<step>/: that path
        # carries the step, so a dump resumed at a different one would scan an
        # empty directory and re-sample all of it. latent_dir pins the location.
        out_dir = media_dir
        if save_latent and validation.get("latent_dir", None):
            out_dir = Path(validation.latent_dir).expanduser()

        # Resume: drop prompts already on disk, and adopt rank 0's view of which
        # those are. A divergent view gives ranks different per-rank counts and
        # desynchronises the collective forwards.
        completed = ops.all_gather_object(
            self._completed_prompt_ids(
                out_dir,
                [name for name, _ in variants],
                suffix="pt" if save_latent else "mp4",
            )
        )[0]
        pending = [
            (index, request)
            for index, request in enumerate(requests)
            if index not in completed
        ]
        entries = self._split_validation_prompts(
            pending,
            group_index,
            num_groups,
            self._validation_request_prompt,
        )

        original_training = [(model, model.training) for _, model in variants]
        saved_items: list[tuple[str, str, str, str]] = []
        fps = int(validation.get("fps", MINIMAX_H3_SUPPORTED_FPS))
        try:
            for _, model in variants:
                model.eval()
            # Validation batch_size packs that many prompts into ONE sampling
            # forward (training parity: the dataset packs the same way under
            # its max_seqlen budget). 1 keeps the per-prompt path, including
            # the exact per-prompt rng seed.
            batch_size = int(validation.get("batch_size", 1))
            assert batch_size >= 1, f"validation.batch_size must be >= 1, got {batch_size}"
            for batch_start in range(0, len(entries), batch_size):
                batch_entries = entries[batch_start : batch_start + batch_size]
                batch_requests = [entry["request"] for entry in batch_entries]
                inputs = self._validation_inputs_for_requests(
                    config, models, batch_requests
                )
                for variant_name, model in variants:
                    # Per-prompt seeds (the wan/crafter pattern): each prompt
                    # keeps its own deterministic stream inside the batched
                    # sampling, so changing batch_size never perturbs a
                    # prompt's noise. With batch_size 1 this is the historical
                    # per-prompt rng. shift_seed: false pins every prompt to
                    # the same seed (ltx2's option); true (default) shifts by
                    # prompt index.
                    shift_seed = bool(validation.get("shift_seed", True))
                    sample_rngs = [
                        RandomState(
                            combine_seed(
                                int(validation.seed),
                                "validation",
                                entry["prompt_idx"] if shift_seed else 0,
                                variant_name,
                            )
                        )
                        for entry in batch_entries
                    ]
                    latents = self._sample_latents(
                        model, inputs, sample_rngs[0], rngs=sample_rngs
                    )
                    for index, entry in enumerate(batch_entries):
                        if not (is_group_main and entry["should_save"]):
                            continue
                        stem = f"validation/{variant_name}/prompt{entry['prompt_idx']:04d}"
                        if save_latent:
                            name = f"{stem}.pt"
                            path = out_dir / name
                            self._save_validation_latents(
                                path, latents, index=index,
                                prompt=entry["prompt"], prompt_idx=entry["prompt_idx"],
                                metadata=self._validation_latent_metadata(
                                    config, inputs, entry["request"], index=index,
                                ),
                            )
                            saved_items.append((variant_name, name, str(path), ""))
                            continue
                        frames, waveform = self._decode_latents(models, latents, index=index)
                        frames = self._validation_frames(frames, inputs, index=index)
                        name = f"{stem}.mp4"
                        path = media_dir / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        media.save_video(
                            frames,
                            path,
                            fps=fps,
                            nrow=int(validation.get("nrow", 1)),
                            crf=int(validation.get("crf", 18)),
                            audio_tensor=waveform,
                            audio_sample_rate=int(models["audio_vae"].sample_rate),
                            normalize=True,
                            # revert_tensor already clamped the decoded pixels.
                            value_range=(0, 1),
                        )
                        saved_items.append(
                            (variant_name, name, str(path), f"[{variant_name}] {entry['prompt']}")
                        )
                        del frames, waveform
                    del latents
                del inputs

            gathered = ops.all_gather_object(saved_items)
            try:
                if rank == 0 and save_latent:
                    # Nothing to preview: no frames were decoded, and a grid of
                    # latents is not a thing. The count is the useful signal.
                    total = sum(len(rank_items) for rank_items in gathered)
                    logger.info(
                        "saved %d latents under %s", total, out_dir / "validation"
                    )
                elif rank == 0:
                    all_items = [item for rank_items in gathered for item in rank_items]
                    for variant_name, _ in variants:
                        variant_items = [item for item in all_items if item[0] == variant_name]
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
                            # The grid is a preview; the per-prompt files are the
                            # real artifact. Losing a run at step N because a
                            # convenience encode hiccuped is the worse failure.
                            logger.warning("validation grid build failed: %s", exc)
            finally:
                ops.barrier()
        finally:
            for model, training in original_training:
                model.train(training)
        return ctx
