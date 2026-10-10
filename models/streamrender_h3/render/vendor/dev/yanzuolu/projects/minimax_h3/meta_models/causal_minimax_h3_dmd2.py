# SPDX-License-Identifier: Apache-2.0
"""DMD2 adversarial training for chunk-causal MiniMax-H3."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.nn import functional as F

from dev.yanzuolu.common.diffusion import build_diffusion
from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.meter import get_running_average_meter
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import local_seed, yield_seed
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd import (
    CausalMiniMaxH3DMD,
    ForwardInput,
    _RolloutX0s,
)
from dev.yanzuolu.projects.minimax_h3.modeling.discriminator_output import (
    MiniMaxH3DiscriminatorOutput,
)


class CausalMiniMaxH3DMD2(CausalMiniMaxH3DMD):
    """Add a tapped bidirectional discriminator objective to MiniMax-H3 DMD.

    The discriminator sees corrupted x_t by default, or clean x with a
    pretended random timestep when ``gan_disc_clean_input`` is enabled.
    """

    _carry_clean_latents = True

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        mm = config.meta_model
        self.gan_lambda_disc = float(mm.get("gan_lambda_disc", 1.0e-2))
        self.gan_lambda_gen = float(mm.get("gan_lambda_gen", 5.0e-3))
        self.gan_global_weight = float(mm.get("gan_global_weight", 0.2))
        if not 0.0 <= self.gan_global_weight <= 1.0:
            raise ValueError("meta_model.gan_global_weight must be in [0, 1]")
        self.gan_disc_clean_input = bool(mm.get("gan_disc_clean_input", False))
        # Seaweed-APT uses lambda=100 and sigma=0.1 for video. Under a
        # first-order Taylor expansion, E[||D(x+sigma*eps)-D(x)||^2] =
        # sigma^2||grad_x D||^2; unlike true R1, it avoids the double backward
        # that is unavailable with FSDP/FlashAttention. APT's discriminator
        # loss carries an implicit lambda_D of 1, while ours pre-scales by
        # gan_lambda_disc (1e-2), so matching the paper's ratio needs
        # gan_r1_lambda ~ 100 * gan_lambda_disc ~ 1.0.
        self.gan_r1_lambda = float(mm.get("gan_r1_lambda", 0.0))
        self.gan_r1_sigma = float(mm.get("gan_r1_sigma", 0.1))

        gan_diffusion = build_diffusion(
            {"gan_training_timesteps": config.diffusion.gan_training_timesteps}
        )
        self.gan_training_timesteps = gan_diffusion["gan_training_timesteps"]
        if float(self.gan_training_timesteps.T) != float(self.fake_schedule.T):
            raise ValueError("gan_training_timesteps.T must match fake_schedule.T")

    @contextmanager
    def gen_model_context(self, models: dict[str, Any]) -> Iterator[None]:
        """Freeze the fake model and its discriminator while preserving input gradients."""
        frozen = (models["fake_model"],)
        requires_grad = [
            [parameter.requires_grad for parameter in model.parameters()]
            for model in frozen
        ]
        for model in frozen:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        try:
            with super().gen_model_context(models):
                yield
        finally:
            for model, states in zip(frozen, requires_grad, strict=True):
                for parameter, state in zip(model.parameters(), states, strict=True):
                    parameter.requires_grad_(state)

    def _prepare_gan_inputs(
        self,
        inputs: ForwardInput,
        latents: tuple[list[Tensor], list[Tensor]],
        rng: Any,
    ) -> tuple[tuple[list[Tensor], list[Tensor]], Tensor, Tensor]:
        with local_seed(rng.seed % 2**31):
            drawn = self.gan_training_timesteps.sample_pair(
                (inputs.batch_size,), inputs.seqlens, get_device()
            )
        rng.seed = yield_seed(rng.seed)
        video_timesteps, audio_timesteps = (
            value.to(torch.float32) for value in drawn
        )
        if self.gan_disc_clean_input:
            return latents, video_timesteps, audio_timesteps

        video_noises, audio_noises = self._sample_noises(inputs, rng)
        noisy_latents = (
            self.fake_schedule.forward(
                latents[0], video_noises, video_timesteps
            ),
            self.fake_schedule.forward(
                latents[1], audio_noises, audio_timesteps
            ),
        )
        return noisy_latents, video_timesteps, audio_timesteps

    def _pred_gan_logits(
        self,
        fake_model: Any,
        inputs: ForwardInput,
        *,
        noisy_latents: tuple[list[Tensor], list[Tensor]],
        video_timesteps: Tensor,
        audio_timesteps: Tensor,
        gan_chunk_keep_mask: Tensor | None = None,
    ) -> Tensor | MiniMaxH3DiscriminatorOutput:
        """Run the native bidirectional pack and classify its internal taps."""
        device = inputs.token_tags.device
        kwargs, row_timesteps, img_pos, audio_pos = self._bidirectional_kwargs(
            fake_model,
            inputs,
            video_xts=noisy_latents[0],
            audio_xts=noisy_latents[1],
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
        )
        assert bool((row_timesteps[img_pos] != 0).all()) and bool(
            (row_timesteps[audio_pos] != 0).all()
        ), "a bidirectional GAN row carries the clean timestep 0"

        document_lens = torch.stack(
            [
                native["cu_seqlens"][1:] - native["cu_seqlens"][:-1]
                for native in inputs.native
            ]
        ).reshape(-1).to(dtype=torch.int32, device=device)
        live_documents = torch.arange(
            document_lens.numel(), device=device
        ).remainder(2).eq(0)
        nonempty_documents = document_lens > 0
        k_lens = document_lens[nonempty_documents]
        live_documents = live_documents[nonempty_documents]
        head_options = (
            {"gan_chunk_keep_mask": gan_chunk_keep_mask}
            if gan_chunk_keep_mask is not None
            else {}
        )
        return fake_model(
            **kwargs,
            classify_mode=True,
            gan_video_timesteps=video_timesteps,
            gan_k_lens=k_lens,
            gan_live_documents=live_documents,
            **head_options,
        )

    def _gan_reduce(
        self,
        logits: Tensor | MiniMaxH3DiscriminatorOutput,
        function: Callable[..., Tensor],
        *paired_logits: Tensor | MiniMaxH3DiscriminatorOutput,
        metric_prefix: str | None = None,
    ) -> Tensor:
        """Reduce V3 heads per sample while preserving the V1/V2 tensor reduction.

        ``gan_global_weight`` mixes the V3 sample loss with its chunk loss.
        Each sample first averages its own chunks, so clip length cannot change
        its weight within the batch. Paired outputs support the same reduction
        for aR1 without separating its graph from the real GAN objective.
        """
        if isinstance(logits, Tensor):
            return function(logits, *paired_logits).mean(dim=1).mean()

        assert isinstance(logits, MiniMaxH3DiscriminatorOutput)
        for paired in paired_logits:
            assert isinstance(paired, MiniMaxH3DiscriminatorOutput)
            assert paired.chunk_logits.shape == logits.chunk_logits.shape
            assert paired.sample_logits.shape == logits.sample_logits.shape
            assert torch.equal(paired.chunk_sample_indices, logits.chunk_sample_indices)
        chunk_values = function(
            logits.chunk_logits.float(),
            *(paired.chunk_logits.float() for paired in paired_logits),
        ).mean(dim=1)
        sample_values = function(
            logits.sample_logits.float(),
            *(paired.sample_logits.float() for paired in paired_logits),
        ).mean(dim=1)
        sample_count = sample_values.shape[0]
        indices = logits.chunk_sample_indices
        counts = torch.bincount(indices, minlength=sample_count)
        assert counts.shape == (sample_count,) and bool((counts > 0).all()), (
            "V3 chunk logits must cover every sample"
        )
        chunk_sums = chunk_values.new_zeros(sample_count).index_add(0, indices, chunk_values)
        chunk_loss = (chunk_sums / counts.to(chunk_sums)).mean()
        sample_loss = sample_values.mean()
        if metric_prefix is not None:
            meter = get_running_average_meter()
            meter.put_scalar(f"{metric_prefix}/chunk", float(chunk_loss.detach()))
            meter.put_scalar(f"{metric_prefix}/global", float(sample_loss.detach()))
        return (1.0 - self.gan_global_weight) * chunk_loss + self.gan_global_weight * sample_loss

    @torch.no_grad()
    def _log_gan_logits(
        self, logits: Tensor | MiniMaxH3DiscriminatorOutput, metric_prefix: str
    ) -> None:
        if isinstance(logits, Tensor):
            meter = get_running_average_meter()
            for tap_index, value in enumerate(logits.mean(dim=0).detach().tolist()):
                meter.put_scalar(f"{metric_prefix}/tap{tap_index}", value)
        else:
            self._gan_reduce(logits, lambda value: value, metric_prefix=metric_prefix)

    def _gan_inputs(self, ctx: dict[str, Any], *, role: str) -> ForwardInput:
        """Select the conditioning view for one GAN objective."""
        key = {"fake": "fake_inputs", "real": "fake_inputs", "gen": "gen_inputs"}[role]
        return ctx[key]

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_fake(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs = ctx["inputs"]
        if inputs.clean_latents is None:
            raise ValueError(
                "MiniMax-H3 DMD2 requires corpus video/audio latents for "
                "discriminator real samples"
            )
        ctx = super().prepare_fake(ctx)
        rollout: _RolloutX0s = ctx["rollout_x0s"]

        (
            ctx["gan_fake_noisy_latents"],
            ctx["gan_fake_timesteps"],
            ctx["gan_fake_audio_timesteps"],
        ) = self._prepare_gan_inputs(
            inputs, (rollout.video, rollout.audio), ctx["rng"]
        )
        (
            ctx["gan_real_noisy_latents"],
            ctx["gan_real_timesteps"],
            ctx["gan_real_audio_timesteps"],
        ) = self._prepare_gan_inputs(inputs, inputs.clean_latents, ctx["rng"])
        if self.gan_r1_lambda > 0:
            video_noises, audio_noises = self._sample_noises(inputs, ctx["rng"])
            real_video, real_audio = ctx["gan_real_noisy_latents"]
            ctx["gan_real_perturbed_latents"] = (
                [
                    latent + self.gan_r1_sigma * noise
                    for latent, noise in zip(real_video, video_noises, strict=True)
                ],
                [
                    latent + self.gan_r1_sigma * noise
                    for latent, noise in zip(real_audio, audio_noises, strict=True)
                ],
            )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = super().fake_loss(ctx)
        fake_model = ctx["models"]["fake_model"]
        fake_inputs = self._gan_inputs(ctx, role="fake")
        real_inputs = self._gan_inputs(ctx, role="real")
        fake_logits = self._pred_gan_logits(
            fake_model,
            fake_inputs,
            noisy_latents=ctx["gan_fake_noisy_latents"],
            video_timesteps=ctx["gan_fake_timesteps"],
            audio_timesteps=ctx["gan_fake_audio_timesteps"],
        )
        real_logits = self._pred_gan_logits(
            fake_model,
            real_inputs,
            noisy_latents=ctx["gan_real_noisy_latents"],
            video_timesteps=ctx["gan_real_timesteps"],
            audio_timesteps=ctx["gan_real_audio_timesteps"],
        )
        fake_term = self._gan_reduce(
            fake_logits, F.softplus, metric_prefix="fake_losses/gan_fake"
        )
        real_term = self._gan_reduce(
            real_logits, lambda value: F.softplus(-value), metric_prefix="fake_losses/gan_real"
        )
        gan_loss = fake_term + real_term
        real_graph_term = self.gan_lambda_disc * real_term
        if self.gan_r1_lambda > 0:
            real_perturbed_logits = self._pred_gan_logits(
                fake_model,
                real_inputs,
                noisy_latents=ctx["gan_real_perturbed_latents"],
                video_timesteps=ctx["gan_real_timesteps"],
                audio_timesteps=ctx["gan_real_audio_timesteps"],
            )
            ar1_term = self._gan_reduce(
                real_logits, lambda value, perturbed: (value - perturbed).pow(2),
                real_perturbed_logits, metric_prefix="fake_losses/gan_ar1",
            )
            real_graph_term = real_graph_term + self.gan_r1_lambda * ar1_term

        meter = get_running_average_meter()
        self._log_gan_logits(fake_logits, "fake_losses/gan_logits_fake_mean")
        self._log_gan_logits(real_logits, "fake_losses/gan_logits_real_mean")
        meter.put_scalar("fake_losses/gan_loss", float(gan_loss.detach()))
        if self.gan_r1_lambda > 0:
            meter.put_scalar("fake_losses/gan_ar1", float(ar1_term.detach()))
        # THREE terms rather than their sum: score, fake D, and real D. The
        # engine backwards each on its own GraphTask, which keeps the FAKE
        # step's memory bounded -- summing them makes every parameter's
        # AccumulateGrad depend on all subgraphs, and the autograd engine then
        # parks each block's gradient in an InputBuffer until the last arrives.
        # See engines/dmd.py::_fake_backward for the measurements. aR1 cannot
        # become a fourth GraphTask: it shares real_logits with the real term
        # and the engine calls backward without retain_graph, so splitting
        # would raise backward-through-graph a second time. The cost is a
        # second D forward inside the real element (AccumulateGrad dep count
        # 2, the same InputBuffer parking that motivated the split) and one
        # more concurrently-live forward graph.
        #
        # The sum is unchanged, so the parameter update is too: FSDP2 adds into
        # sharded_param.grad rather than overwriting it.
        ctx["fake_loss"] = (
            ctx["fake_loss"],
            self.gan_lambda_disc * fake_term,
            real_graph_term,
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = super().gen_loss(ctx)
        inputs = self._gan_inputs(ctx, role="gen")
        noisy_latents, video_timesteps, audio_timesteps = self._prepare_gan_inputs(
            inputs, ctx["gen_x0s"], ctx["rng"]
        )
        logits = self._pred_gan_logits(
            ctx["models"]["fake_model"],
            inputs,
            noisy_latents=noisy_latents,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
        )
        gan_loss = self._gan_reduce(
            logits, lambda value: F.softplus(-value), metric_prefix="dmd_losses/gan_loss"
        )

        meter = get_running_average_meter()
        self._log_gan_logits(logits, "dmd_losses/gan_logits_mean")
        meter.put_scalar("dmd_losses/gan_loss", float(gan_loss.detach()))
        ctx["gen_loss"] = ctx["gen_loss"] + self.gan_lambda_gen * gan_loss
        return ctx


__all__ = ["CausalMiniMaxH3DMD2"]
