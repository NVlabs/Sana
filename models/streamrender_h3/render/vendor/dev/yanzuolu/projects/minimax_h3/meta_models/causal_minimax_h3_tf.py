# SPDX-License-Identifier: Apache-2.0
"""Teacher forcing a chunk-causal MiniMax-H3 T2AV fit onto a corpus of x0.

Every chunk carries a noisy copy and, except for the last, a clean one; the
mask lets a noisy chunk see only earlier clean copies, never its own. So one
packed forward supervises all of a sample's chunks at once, each an independent
"denoise this chunk given clean history" task -- which is why timesteps are
drawn per chunk rather than per sample.

ctx contract (engines/diffusion_finetuning.py owns the calling sequence)::

    prepare_inputs    reads batch, models   writes inputs, clean_latents
    sync_inputs       reads batch, inputs   yields synchronized ctx copies
    sample_timesteps  reads inputs, rng     writes train_timesteps (per chunk)
    add_noise         reads inputs, clean_latents, train_timesteps, rng
                                            writes noisy_latents, context_eps
    forward           reads models, inputs, noisy_latents, clean_latents,
                      context_eps, train_timesteps          writes pred
    compute_loss      reads pred, clean_latents, inputs     writes loss

``diffusion`` supplies five components: ``training_timesteps`` for the per-chunk
draw, ``schedule`` for both add_noise and the sampler it is injected into, and
``sampling_timesteps``, ``audio_sampling_timesteps`` and ``sampler`` for the
inherited validation rollout.

``training_timesteps`` must be one of the paired classes in
``projects/minimax_h3/modeling/training_timesteps.py``: it carries both shifts
and returns the (video, audio) pair from a single draw, so there is no separate
``audio_training_timesteps`` node here. The two shifts are the same 12 and 3 the
sampling grids use, because the pair has to sit on the (sigma_v, sigma_a) curve
the corpus was sampled along.

``models`` needs ``backbone`` and ``text_encoder``, plus ``video_vae`` and
``audio_vae`` once validation runs. No fake_model, no tea_model. ``data`` must
be the latent corpus dataset: the inherited validation instantiates
``config.data.class_name`` again as its layout packer.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch

from dev.yanzuolu.common.diffusion import build_diffusion
from dev.yanzuolu.common.diffusion.schedule import PredictionType
from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.distributed.unified_parallel import (
    SPDistForward,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import local_seed, yield_seed
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd import CausalMiniMaxH3DMD


class CausalMiniMaxH3TF(CausalMiniMaxH3DMD):
    """Fit chunk-causal denoising against corpus x0 context and targets."""

    def __init__(self, config: Any) -> None:
        # Deliberately not super().__init__: the DMD constructor requires
        # algorithm-specific training knobs this fit does not use.
        self.config = config
        self.audio_loss_weight = float(config.meta_model.audio_loss_weight)

        diffusion = build_diffusion(config.diffusion)
        # ONE node, not two: it emits the (video, audio) pair itself. See
        # projects/minimax_h3/modeling/training_timesteps.py for why the pairing
        # lives there.
        self.training_timesteps = diffusion["training_timesteps"]
        self.sampling_timesteps = diffusion["sampling_timesteps"]
        self.audio_sampling_timesteps = diffusion["audio_sampling_timesteps"]
        self.schedule = diffusion["schedule"]
        self.sampler = diffusion["sampler"]

        # The inherited reduction supports DMD's optional temporal weighting;
        # this fit uses the unweighted per-modality MSE path.
        self.chunk_wise_weighting = None
        # Built lazily on the first validation, exactly like the DMD parent:
        # the dataset owns the packed layout, so validation packs its prompts
        # through the very same class rather than re-deriving it here.
        self._validation_packer_cache: Any = None

        # compute_loss compares the model's output directly against the corpus
        # x0, and add_noise builds x_t through this same schedule. Any other
        # pred_type would make the target wrong while the loss curve stays
        # entirely plausible.
        assert self.schedule.pred_type == PredictionType.x_0, (
            f"diffusion.schedule.pred_type must be x_0 because every MiniMax-H3 "
            f"model node returns x0, got {self.schedule.pred_type}"
        )

    @execution_phase(ExecutionPhase.PREPARE)
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: batch, models. Writes: inputs, clean_latents."""
        batch = ctx["batch"]
        prompt_embeds = self._encode_prompts(
            ctx["models"], batch["text_input_ids"], [int(v) for v in batch["text_lens"]]
        )
        ctx["inputs"] = self._build_inputs(batch, prompt_embeds)
        ctx["clean_latents"] = (
            [tensor.to(get_device()) for tensor in batch["video_latents"]],
            [tensor.to(get_device()) for tensor in batch["audio_latents"]],
        )
        return ctx

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Reads: batch, inputs. Yields exactly one ctx per SP-group source rank."""
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            yield ctx
            return

        # The dataset hands out CPU tensors and only _build_inputs moves them, but
        # the exchange is an NCCL broadcast: every payload leaf must be on device.
        payload = (self._to_device(ctx["batch"]), ctx["inputs"].prompt_embeds)
        sync = SPDistForward(name="causal_tf_inputs", comm_shape=True, device=get_device())
        for batch, prompt_embeds in sync(payload):
            sub_ctx = dict(ctx)
            sub_ctx["inputs"] = self._build_inputs(batch, prompt_embeds)
            sub_ctx["clean_latents"] = (batch["video_latents"], batch["audio_latents"])
            yield sub_ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def sample_timesteps(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: inputs, rng. Writes: train_timesteps."""
        inputs = ctx["inputs"]
        chunk_counts = [len(layout.chunks) for layout in inputs.layouts]
        # postprocess_sample's dynamic_shift branch asserts one seqlen per drawn
        # timestep, and a chunk inherits its sample's. Inert while the shift is a
        # constant, which is exactly why it needs saying.
        seqlens = torch.cat(
            [inputs.seqlens[index].expand(count) for index, count in enumerate(chunk_counts)]
        )
        positions = torch.cat(
            [
                self._chunk_masses_and_participation(layout, torch.device("cpu"))[1]
                for layout in inputs.layouts
            ]
        )

        # The node owns the video/audio pairing; this owns only the rng, exactly
        # as it would around a plain `sample`.
        rng = ctx["rng"]
        with local_seed(rng.seed % 2**31):
            drawn = self.training_timesteps.sample_pair(
                (sum(chunk_counts),),
                seqlens,
                get_device(),
                positions=positions,
            )
        rng.seed = yield_seed(rng.seed)

        # A continuous (T: float) training-timestep node samples in float64, and
        # A(t)*x_0 would promote the whole noisy pack with it -- silently, since
        # _embed casts to fp32 anyway. Pin it where it meets the latents.
        video_timesteps, audio_timesteps = (value.to(torch.float32) for value in drawn)
        ctx["train_timesteps"] = (
            list(video_timesteps.split(chunk_counts)),
            list(audio_timesteps.split(chunk_counts)),
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: inputs, clean_latents, train_timesteps, rng. Writes:
        noisy_latents, context_eps."""
        inputs = ctx["inputs"]
        clean_video, clean_audio = ctx["clean_latents"]
        video_timesteps, audio_timesteps = ctx["train_timesteps"]
        video_noises, audio_noises = self._sample_noises(inputs, ctx["rng"])

        ctx["noisy_latents"] = (
            [
                self._noise_chunks(clean, noise, timesteps, layout, "video")
                for clean, noise, timesteps, layout in zip(
                    clean_video, video_noises, video_timesteps, inputs.layouts
                )
            ],
            [
                self._noise_chunks(clean, noise, timesteps, layout, "audio")
                for clean, noise, timesteps, layout in zip(
                    clean_audio, audio_noises, audio_timesteps, inputs.layouts
                )
            ],
        )
        # Clean rows are perturbed inside MiniMaxH3X0Model at the checkpoint's
        # attested clean timestep, so their eps must be independently sampled.
        ctx["context_eps"] = self._sample_noises(inputs, ctx["rng"])
        return ctx

    def _noise_chunks(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        layout: Any,
        modality: str,
    ) -> torch.Tensor:
        """One sample's per-chunk noising, reassembled along the time axis."""
        return torch.cat(
            [
                self.schedule.forward(
                    x_0=clean[self._chunk_slice(chunk, modality)],
                    x_T=noise[self._chunk_slice(chunk, modality)],
                    t=timestep,
                )
                for chunk, timestep in zip(layout.chunks, timesteps, strict=True)
            ],
            dim=1 if modality == "video" else 2,
        )

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, inputs, noisy_latents, clean_latents, context_eps,
        train_timesteps. Writes: pred."""
        video_xts, audio_xts = ctx["noisy_latents"]
        clean_video, clean_audio = ctx["clean_latents"]
        video_eps, audio_eps = ctx["context_eps"]
        video_t, audio_t = ctx["train_timesteps"]
        ctx["pred"] = self._causal_packed_forward(
            ctx["models"]["backbone"],
            ctx["inputs"],
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_context=clean_video,
            audio_context=clean_audio,
            video_eps=video_eps,
            audio_eps=audio_eps,
            video_timesteps=video_t,
            audio_timesteps=audio_t,
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: pred, clean_latents, inputs. Writes: loss."""
        video_pred, audio_pred = ctx["pred"]
        clean_video, clean_audio = ctx["clean_latents"]
        ctx["loss"] = self._loss_fn(
            video_pred,
            audio_pred,
            inputs=ctx["inputs"],
            video_targets=clean_video,
            audio_targets=clean_audio,
            key_prefix="train_losses",
        )
        return ctx


__all__ = ["CausalMiniMaxH3TF"]
