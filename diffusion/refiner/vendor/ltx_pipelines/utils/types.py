from dataclasses import dataclass, field
from typing import Protocol

import torch

from diffusion.refiner.vendor.ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from diffusion.refiner.vendor.ltx_core.conditioning import ConditioningItem
from diffusion.refiner.vendor.ltx_core.model.transformer import X0Model
from diffusion.refiner.vendor.ltx_core.types import LatentState
from diffusion.refiner.vendor.ltx_pipelines.utils.constants import VIDEO_LATENT_CHANNELS, VIDEO_SCALE_FACTORS


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
        ``(denoised_video, denoised_audio)`` tensors (either may be ``None``).
    """

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...


@dataclass(frozen=True)
class ModalitySpec:
    """Specification for one modality passed to a diffusion stage.
    Carries everything needed to build the initial noised latent state
    and run the denoising loop for a single modality (video or audio).
    Tools are created by ``DiffusionStage`` from pixel-space dimensions.
    """

    context: torch.Tensor
    conditionings: list[ConditioningItem] = field(default_factory=list)
    noise_scale: float = 1.0
    frozen: bool = False
    initial_latent: torch.Tensor | None = None
