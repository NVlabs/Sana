"""Constructor adapters between the repo's YAML contract and upstream's signatures.

Not vendored code -- this file is ours.

``common.model.instantiate.DefaultModelInstantiate`` builds a model as
``model_cls(**config.args)``, i.e. flat keyword arguments straight out of the
YAML. Two of the three vendored MiniMax-H3 models do not accept that shape:

* the DiT takes sglang's ``BaseDiT`` signature, ``(config, hf_config, ...)``,
  where ``config`` is a nested dataclass;
* the video VAE takes flat kwargs but reads its runtime knobs (clip length,
  token drop, tiling) from a separate ``setup_forward(**kwargs)`` call that the
  instantiate stage has no way to make;
* neither VAE accepts the per-channel latent statistics that ship in its own
  released ``config.json`` -- ``AutoencoderKLLegacy`` has a ``**kwargs`` that
  would swallow them silently, and ``DacAudioVAE`` has an explicit signature
  that would raise on them. Both wrappers below take them and register them as
  non-persistent buffers, so the stats travel with the module the way wan's VAE
  carries its own ``mean``/``std``.

These subclasses add no *parameters*, and the latent statistics are registered
non-persistently, so ``state_dict()`` is unchanged and the checkpoint contract
is untouched.
"""

from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .audio_vae.audio_vae import (
    DacAudioVAE,
)
from .discriminator import (
    MiniMaxH3DMD2Discriminator,
    MiniMaxH3DMD2DiscriminatorV2,
    MiniMaxH3DMD2DiscriminatorV3,
)
from .discriminator_output import MiniMaxH3DiscriminatorOutput
from .transformer.config import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from .transformer.causal_model import (
    CausalMiniMaxH3DiTModel,
)
from .transformer.causal_model_sp import (
    CausalMiniMaxH3DiTModelSP,
)
from .transformer.model import (
    MiniMaxH3DiTModel,
)
from .transformer.model_sp import (
    MiniMaxH3DiTModelSP,
)
from .transformer.x0_model import (
    MiniMaxH3X0Model,
)
from .time_request import H3_VIDEO_TEMPORAL_MAPPING
from .video_vae.klvae import (
    AutoencoderKLLegacy,
)


def _register_latent_stats(
    module: torch.nn.Module,
    latents_mean: Sequence[float] | None,
    latents_std: Sequence[float] | None,
) -> None:
    """Attach per-channel latent statistics as non-persistent buffers.

    Both are optional: a VAE used only for its architecture (a smoke test, a
    shape check) does not need them, and registering ``None`` would break
    ``.to(device)``. When present they are consumed outside the VAE forward --
    upstream normalizes in its pipeline layer, not inside the module -- so they
    are exposed as attributes rather than folded into ``encode``.
    """
    if latents_mean is not None:
        module.register_buffer("latents_mean", torch.tensor(latents_mean), persistent=False)
    if latents_std is not None:
        module.register_buffer("latents_std", torch.tensor(latents_std), persistent=False)


class MiniMaxH3DiT(MiniMaxH3DiTModel):
    """``MiniMaxH3DiTModel`` built from flat architecture kwargs."""

    def __init__(self, **arch_kwargs: Any) -> None:
        arch = MiniMaxH3DiTArchConfig(**arch_kwargs)
        super().__init__(MiniMaxH3DiTConfig(arch_config=arch), hf_config={})


class MiniMaxH3X0DiT(MiniMaxH3X0Model):
    """``MiniMaxH3X0Model`` over the bidirectional DiT, from flat arch kwargs.

    ``MiniMaxH3X0Model`` takes its inner DiT as a ``_class_name``-tagged tree,
    which cannot survive the YAML round trip: ``build_nested_module`` rebuilds
    every mapping as a plain dict, so the ``MiniMaxH3DiTConfig`` the DiT expects
    arrives as a dict and ``config.arch_config`` raises. Same reason
    ``MiniMaxH3DiT`` above exists, one layer out.

    Every DMD model node wants the x0 wrapper -- the raw DiT returns H3's
    data-ward velocity, which differs from the repo's by a sign.
    """

    def __init__(self, *, timestep_conditioning_scale: float = 1.0, **arch_kwargs: Any) -> None:
        arch = MiniMaxH3DiTArchConfig(**arch_kwargs)
        super().__init__(
            MiniMaxH3DiTModel(MiniMaxH3DiTConfig(arch_config=arch), hf_config={}),
            timestep_conditioning_scale=timestep_conditioning_scale,
        )


class MiniMaxH3X0DiTSP(MiniMaxH3X0Model):
    """``MiniMaxH3X0DiT`` with Ulysses sequence parallelism.

    Same parameters and the same checkpoint as the dense bidirectional model;
    each rank computes over its own row shard. Inert when unified parallel is
    off or its world size is 1, and it is the only way to reach the
    bidirectional SP model at all -- ``MiniMaxH3X0DiT`` hardcodes the dense
    one. DMD's ``fake_model`` and ``tea_model`` are bidirectional, so this is
    what a sequence-parallel DMD trial configures for them.
    """

    _dit_cls = MiniMaxH3DiTModelSP

    def __init__(self, *, timestep_conditioning_scale: float = 1.0, **arch_kwargs: Any) -> None:
        arch = MiniMaxH3DiTArchConfig(**arch_kwargs)
        super().__init__(
            self._dit_cls(MiniMaxH3DiTConfig(arch_config=arch), hf_config={}),
            timestep_conditioning_scale=timestep_conditioning_scale,
        )


class MiniMaxH3GANX0DiTSP(MiniMaxH3X0DiTSP):
    """Bidirectional SP x0 model with the original sample-wise DMD2 head."""

    def __init__(
        self,
        *,
        discriminator_tap_blocks: list[int],
        discriminator_num_queries: int,
        timestep_conditioning_scale: float = 1.0,
        **arch_kwargs: Any,
    ) -> None:
        super().__init__(timestep_conditioning_scale=timestep_conditioning_scale, **arch_kwargs)
        if not discriminator_tap_blocks:
            raise ValueError("discriminator_tap_blocks must contain at least one tap")
        if any(
            not isinstance(block, int) or isinstance(block, bool)
            for block in discriminator_tap_blocks
        ):
            raise TypeError("discriminator_tap_blocks must contain only integers")
        if any(
            left >= right
            for left, right in zip(discriminator_tap_blocks, discriminator_tap_blocks[1:])
        ):
            raise ValueError("discriminator_tap_blocks must be strictly increasing")
        if (
            discriminator_tap_blocks[0] < 1
            or discriminator_tap_blocks[-1] > len(self.dit.blocks)
        ):
            raise ValueError(
                "Each discriminator_tap_blocks value is a count of leading transformer blocks and must be in "
                f"[1, {len(self.dit.blocks)}], got {discriminator_tap_blocks}"
            )

        self.discriminator_tap_blocks = tuple(discriminator_tap_blocks)
        self.discriminator = self._build_discriminator(
            arch=MiniMaxH3DiTArchConfig(**arch_kwargs),
            num_queries=discriminator_num_queries,
            num_taps=len(discriminator_tap_blocks),
        )

    def _build_discriminator(
        self,
        *,
        arch: MiniMaxH3DiTArchConfig,
        num_queries: int,
        num_taps: int,
    ) -> MiniMaxH3DMD2Discriminator:
        return MiniMaxH3DMD2Discriminator(
            arch=arch,
            num_queries=num_queries,
            num_taps=num_taps,
        )

    def _dispatch_discriminator(
        self,
        taps: Sequence[torch.Tensor],
        *,
        adaln_input: torch.Tensor,
        k_lens: torch.Tensor,
        live_documents: torch.Tensor,
        forward_kwargs: dict[str, Any],
        chunk_ranges: torch.Tensor | None = None,
        chunk_sample_indices: torch.Tensor | None = None,
        chunk_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if chunk_ranges is not None or chunk_sample_indices is not None:
            raise ValueError("explicit GAN chunk groups require the V2 discriminator")
        if chunk_keep_mask is not None:
            raise ValueError("GAN chunk filtering requires the V3 discriminator")
        return self.discriminator(
            taps,
            adaln_input=adaln_input,
            k_lens=k_lens,
            live_documents=live_documents,
        )

    def set_gradient_checkpointing(
        self,
        enabled: bool = True,
        gc_start_idx: int = 0,
        gc_step: int = 1,
    ) -> None:
        super().set_gradient_checkpointing(
            enabled,
            gc_start_idx=gc_start_idx,
            gc_step=gc_step,
        )
        self.discriminator.set_gradient_checkpointing(enabled)

    def forward(
        self,
        *,
        gan_video_timesteps: torch.Tensor | None = None,
        gan_k_lens: torch.Tensor | None = None,
        gan_live_documents: torch.Tensor | None = None,
        gan_chunk_ranges: torch.Tensor | None = None,
        gan_chunk_sample_indices: torch.Tensor | None = None,
        gan_chunk_keep_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor | MiniMaxH3DiscriminatorOutput:
        if not kwargs.get("classify_mode"):
            return super().forward(**kwargs)

        taps, t_emb = super().forward(
            **kwargs,
            tap_blocks=self.discriminator_tap_blocks,
            tap_outputs=[],
        )
        repo_row_timesteps = kwargs["unique_timesteps"].float()[
            kwargs["inverse_indices"]
        ]
        clean_anchors = kwargs.get("clean_timesteps_by_tag")
        h3_unique_timesteps = torch.unique(self._conditioning_timesteps(
            repo_row_timesteps, kwargs["token_tags"], clean_anchors,
            kwargs.get("timestep_conditioning_mask"),
        ))
        video_h3_timesteps = self._conditioning_timesteps(
            gan_video_timesteps,
            torch.zeros_like(gan_video_timesteps, dtype=torch.long),
            clean_anchors,
        )
        video_time_indices = torch.searchsorted(
            h3_unique_timesteps, video_h3_timesteps
        )
        assert bool(
            (
                h3_unique_timesteps.index_select(0, video_time_indices)
                == video_h3_timesteps
            ).all()
        )
        adaln_input = F.silu(t_emb.index_select(0, video_time_indices)).to(
            dtype=taps[0].dtype
        )
        return self._dispatch_discriminator(
            taps,
            adaln_input=adaln_input,
            k_lens=gan_k_lens,
            live_documents=gan_live_documents,
            forward_kwargs=kwargs,
            chunk_ranges=gan_chunk_ranges,
            chunk_sample_indices=gan_chunk_sample_indices,
            chunk_keep_mask=gan_chunk_keep_mask,
        )


class MiniMaxH3GANX0DiTSPV2(MiniMaxH3GANX0DiTSP):
    """Bidirectional SP x0 model with the chunk-wise DMD2 V2 head."""

    def __init__(
        self,
        *,
        discriminator_tap_blocks: list[int],
        discriminator_num_queries: int,
        discriminator_video_chunk_size: int = 5,
        **arch_kwargs: Any,
    ) -> None:
        object.__setattr__(
            self,
            "_discriminator_video_chunk_size",
            discriminator_video_chunk_size,
        )
        super().__init__(
            discriminator_tap_blocks=discriminator_tap_blocks,
            discriminator_num_queries=discriminator_num_queries,
            **arch_kwargs,
        )
        del self._discriminator_video_chunk_size

    def _build_discriminator(
        self,
        *,
        arch: MiniMaxH3DiTArchConfig,
        num_queries: int,
        num_taps: int,
    ) -> MiniMaxH3DMD2DiscriminatorV2:
        return MiniMaxH3DMD2DiscriminatorV2(
            arch=arch,
            num_queries=num_queries,
            num_taps=num_taps,
            video_chunk_size=self._discriminator_video_chunk_size,
        )

    def _dispatch_discriminator(
        self,
        taps: Sequence[torch.Tensor],
        *,
        adaln_input: torch.Tensor,
        k_lens: torch.Tensor,
        live_documents: torch.Tensor,
        forward_kwargs: dict[str, Any],
        chunk_ranges: torch.Tensor | None = None,
        chunk_sample_indices: torch.Tensor | None = None,
        chunk_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | MiniMaxH3DiscriminatorOutput:
        target_audio_rows = MiniMaxH3DiTModel._pos_ids(
            forward_kwargs["audio_pos_info"], "audio_pos_info"
        )
        update_audio_mask = forward_kwargs.get("update_audio_mask")
        if update_audio_mask is not None:
            if not isinstance(update_audio_mask, torch.Tensor):
                raise TypeError("update_audio_mask must be a tensor when provided")
            update_audio_mask = update_audio_mask.view(-1)
            if update_audio_mask.dtype != torch.bool:
                raise ValueError("update_audio_mask must be a bool tensor")
            if update_audio_mask.numel() != target_audio_rows.numel():
                raise ValueError(
                    "update_audio_mask length must match audio_pos_info.position_ids"
                )
            target_audio_rows = target_audio_rows[
                update_audio_mask.to(target_audio_rows.device)
            ]
        extra_kwargs = {}
        if chunk_keep_mask is not None:
            extra_kwargs["chunk_keep_mask"] = chunk_keep_mask
        return self.discriminator(
            taps,
            adaln_input=adaln_input,
            k_lens=k_lens,
            live_documents=live_documents,
            token_tags=forward_kwargs["token_tags"],
            position_ids=forward_kwargs["img_position_ids"],
            target_audio_rows=target_audio_rows,
            chunk_ranges=chunk_ranges,
            chunk_sample_indices=chunk_sample_indices,
            **extra_kwargs,
        )


class MiniMaxH3CausalGANX0DiTV2(MiniMaxH3GANX0DiTSPV2):
    """Causal x0 trunk with the unchanged V2 head and checkpoint structure."""

    _dit_cls = CausalMiniMaxH3DiTModel


class MiniMaxH3CausalGANX0DiTSPV2(MiniMaxH3GANX0DiTSPV2):
    """Causal Ulysses x0 trunk with V2 logits over explicit noisy chunks."""

    _dit_cls = CausalMiniMaxH3DiTModelSP


class MiniMaxH3GANX0DiTSPV3(MiniMaxH3GANX0DiTSPV2):
    """Bidirectional SP x0 model with shared chunk and sample discriminator blocks."""

    def __init__(
        self,
        *,
        discriminator_tap_blocks: list[int],
        discriminator_num_queries: int,
        discriminator_video_chunk_size: int = 5,
        discriminator_num_global_queries: int = 1,
        **arch_kwargs: Any,
    ) -> None:
        object.__setattr__(
            self, "_discriminator_num_global_queries", discriminator_num_global_queries
        )
        super().__init__(
            discriminator_tap_blocks=discriminator_tap_blocks,
            discriminator_num_queries=discriminator_num_queries,
            discriminator_video_chunk_size=discriminator_video_chunk_size,
            **arch_kwargs,
        )
        del self._discriminator_num_global_queries

    def _build_discriminator(
        self,
        *,
        arch: MiniMaxH3DiTArchConfig,
        num_queries: int,
        num_taps: int,
    ) -> MiniMaxH3DMD2DiscriminatorV3:
        return MiniMaxH3DMD2DiscriminatorV3(
            arch=arch,
            num_queries=num_queries,
            num_taps=num_taps,
            video_chunk_size=self._discriminator_video_chunk_size,
            num_global_queries=self._discriminator_num_global_queries,
        )


class MiniMaxH3GANX0DiTV3(MiniMaxH3GANX0DiTSPV3):
    """Bidirectional x0 trunk with the V3 discriminator."""

    _dit_cls = MiniMaxH3DiTModel


class MiniMaxH3CausalGANX0DiTV3(MiniMaxH3GANX0DiTSPV3):
    """Causal x0 trunk with V3 logits over retained target chunks and samples."""

    _dit_cls = CausalMiniMaxH3DiTModel


class MiniMaxH3CausalGANX0DiTSPV3(MiniMaxH3GANX0DiTSPV3):
    """Causal Ulysses x0 trunk with the V3 discriminator."""

    _dit_cls = CausalMiniMaxH3DiTModelSP


class MiniMaxH3CausalX0DiT(MiniMaxH3X0Model):
    """The chunk-causal student under the same wrapper and the same kwargs.

    Identical parameters to the bidirectional model, so a bidirectional
    checkpoint loads into it unchanged.
    """

    def __init__(self, *, timestep_conditioning_scale: float = 1.0, **arch_kwargs: Any) -> None:
        arch = MiniMaxH3DiTArchConfig(**arch_kwargs)
        super().__init__(
            CausalMiniMaxH3DiTModel(MiniMaxH3DiTConfig(arch_config=arch), hf_config={}),
            timestep_conditioning_scale=timestep_conditioning_scale,
        )


class MiniMaxH3CausalX0DiTSP(MiniMaxH3X0Model):
    """``MiniMaxH3CausalX0DiT`` with Ulysses sequence parallelism.

    Same parameters and the same checkpoint; the only difference is that each
    rank computes over its own row shard. Inert when unified parallel is off or
    its world size is 1 -- the SP model falls straight through to the dense
    forward -- so this class is safe to configure before ``distributed.up_size``
    exists, and it is the only way to reach the SP model at all: the class above
    hardcodes the dense one.

    The row shards are gathered before the output rows are selected, so the SP
    forward returns the same global logits on every rank and nothing downstream
    (loss, rollout, VAE decode) changes.
    """

    def __init__(self, *, timestep_conditioning_scale: float = 1.0, **arch_kwargs: Any) -> None:
        arch = MiniMaxH3DiTArchConfig(**arch_kwargs)
        super().__init__(
            CausalMiniMaxH3DiTModelSP(MiniMaxH3DiTConfig(arch_config=arch), hf_config={}),
            timestep_conditioning_scale=timestep_conditioning_scale,
        )


class MiniMaxH3VideoVAE(AutoencoderKLLegacy):
    """``AutoencoderKLLegacy`` with ``setup_forward`` folded into construction.

    ``forward_kwargs`` carries what the released component ``config.json`` calls
    ``vae_*`` (``vae_clip_length`` -> ``clip_length`` and so on); the remaining
    kwargs are the architecture, from ``source/config.json``.
    """

    video_temporal_mapping = H3_VIDEO_TEMPORAL_MAPPING

    def __init__(
        self,
        *,
        forward_kwargs: dict[str, Any] | None = None,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
        **arch_kwargs: Any,
    ) -> None:
        super().__init__(**arch_kwargs)
        self.setup_forward(**(forward_kwargs or {}))
        _register_latent_stats(self, latents_mean, latents_std)

    @torch.no_grad()
    def encode_videos_selected(
        self, videos: Any, latent_indices: Sequence[torch.Tensor], *,
        transform_input: bool = False, use_fp16_latent: bool = False,
        verbose: bool = False,
    ) -> list[torch.Tensor]:
        """Encode selected global latent frames from complete RGB videos.

        Inputs follow encode_videos, with one video per list entry. Independent
        temporal blocks retain their original spatial tiles and posterior noise
        shape. Returned CTHW tensors follow each supplied index tensor's order.
        """
        return self._encode_videos_selected(
            videos, latent_indices, transform_input=transform_input,
            use_fp16_latent=use_fp16_latent, verbose=verbose,
        )


class MiniMaxH3AudioVAE(DacAudioVAE):
    """``DacAudioVAE`` that also carries its per-channel latent statistics.

    The architecture kwargs match upstream's signature exactly; the only reason
    this subclass exists is that ``DacAudioVAE.__init__`` has no ``**kwargs``,
    so the ``latents_mean`` / ``latents_std`` that ship in the same released
    ``config.json`` would raise a ``TypeError`` rather than being accepted.
    """

    def __init__(
        self,
        *,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
        **arch_kwargs: Any,
    ) -> None:
        super().__init__(**arch_kwargs)
        _register_latent_stats(self, latents_mean, latents_std)
