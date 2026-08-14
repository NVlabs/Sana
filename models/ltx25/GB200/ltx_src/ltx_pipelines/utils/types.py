from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import torch

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.conditioning import ConditioningItem
from ltx_core.model.transformer import X0Model
from ltx_core.types import LatentState
from ltx_pipelines.utils.constants import VIDEO_LATENT_CHANNELS, VIDEO_SCALE_FACTORS


class PipelineComponents:
    """
    Container class for pipeline components used throughout the LTX pipelines.
    Attributes:
        dtype (torch.dtype): Default torch dtype for tensors in the pipeline.
        device (torch.device): Target device to place tensors and modules on.
        video_scale_factors (SpatioTemporalScaleFactors): Scale factors (T, H, W) for VAE latent space.
        video_latent_channels (int): Number of channels in the video latent representation.
        video_patchifier (VideoLatentPatchifier): Patchifier instance for video latents.
        audio_patchifier (AudioPatchifier): Patchifier instance for audio latents.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.dtype = dtype
        self.device = device

        self.video_scale_factors = VIDEO_SCALE_FACTORS
        self.video_latent_channels = VIDEO_LATENT_CHANNELS

        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)


@dataclass(frozen=True)
class DenoisedLatentResult:
    """Output of one denoiser call for a single modality.
    ``denoised`` is the final blended prediction for this modality.
    The remaining fields carry the per-pass raw outputs from ``_guided_denoise``
    (all ``None`` for ``SimpleDenoiser``).  Denoisers return a
    ``(video_result, audio_result)`` tuple; either element may be ``None``
    for absent modalities.
    """

    denoised: torch.Tensor
    uncond: torch.Tensor | None = None
    cond: torch.Tensor | None = None
    ptb: torch.Tensor | None = None
    mod: torch.Tensor | None = None

    @classmethod
    def result_or_none(
        cls,
        denoised: torch.Tensor | None,
        uncond: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        ptb: torch.Tensor | None = None,
        mod: torch.Tensor | None = None,
    ) -> DenoisedLatentResult | None:
        if denoised is None:
            return None
        return cls(denoised=denoised, uncond=uncond, cond=cond, ptb=ptb, mod=mod)


class Denoiser(Protocol):
    """Protocol for a denoiser that receives the transformer at call time.
    The transformer is not stored — it is passed as the first argument so the
    caller (a denoising loop or a pipeline block) controls its lifecycle.
    Args:
        transformer: The diffusion model.
        video_state: Current video latent state, or ``None`` if absent.
        audio_state: Current audio latent state, or ``None`` if absent.
        sigmas: 1-D tensor of sigma values for each diffusion step.
        step_index: Index of the current denoising step.
    Returns:
        A ``(video_result, audio_result)`` tuple of :class:`DenoisedLatentResult`,
        either may be ``None`` for absent modalities.
    """

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]: ...


@dataclass(frozen=True)
class ModalitySpec:
    """Specification for one modality passed to a diffusion stage.
    Carries everything needed to build the initial noised latent state
    and run the denoising loop for a single modality (video or audio).
    Tools are created by ``DiffusionStage`` from pixel-space dimensions.
    ``frozen=True`` zeros the denoise mask and marks the resulting ``LatentState``
    so ``Modality.sigma`` is forced to 0 (not only per-token timesteps).
    """

    context: torch.Tensor | None = None
    conditionings: list[ConditioningItem] = field(default_factory=list)
    noise_scale: float = 1.0
    frozen: bool = False
    initial_latent: torch.Tensor | None = None


@dataclass(frozen=True)
class AutoDuration:
    """Request auto-predicted duration, clamped to ``[min_seconds, max_seconds]``.
    The predicted duration is converted to a frame count snapped to the VAE's causal
    temporal grid (``8k + 1``). Defaults: 1s and 20s.
    """

    min_seconds: float = 1.0
    max_seconds: float = 20.0


DEFAULT_AUTO_DURATION = AutoDuration()


class OffloadMode(Enum):
    """Weight offloading strategy.
    Controls where model weights reside during inference:
    - ``NONE``: All weights on GPU (no streaming). Fastest inference,
      requires enough VRAM for the full model (~28 GB for LTX-2).
    - ``CPU``: Weights pinned in CPU RAM, streamed layer-by-layer to a
      small GPU buffer. First pass reads from disk; subsequent passes
      reuse the CPU cache. Requires ~36 GB RAM + ~5 GB VRAM.
    - ``DISK``: Weights read from disk on demand through a small CPU
      buffer, then streamed to GPU. Every pass re-reads from disk.
      Lowest memory: ~5 GB RAM + ~5 GB VRAM.
    """

    NONE = "none"
    CPU = "cpu"
    DISK = "disk"
