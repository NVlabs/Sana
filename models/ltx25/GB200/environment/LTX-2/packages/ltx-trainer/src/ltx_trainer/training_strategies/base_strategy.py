"""Base class for training strategies.
This module defines the abstract base class that all training strategies must implement,
along with the base configuration class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from ltx_core.components.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
    get_pixel_coords,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape
from ltx_trainer.timestep_samplers import TimestepSampler

# Default frames per second for video missing in the FPS metadata
DEFAULT_FPS = 24


class TrainingStrategyConfigBase(BaseModel):
    """Base configuration class for training strategies.
    All strategy-specific configuration classes should inherit from this.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["text_to_video", "video_to_video", "flexible"] = Field(
        description="Unique name identifying the training strategy type"
    )

    @abstractmethod
    def get_data_sources(self) -> dict[str, str]:
        """Get the required data sources for this strategy.
        Returns a mapping of directory name (relative to ``preprocessed_data_root``)
        to the dataset output key under which that directory's contents are exposed.
        This is the single source of truth for which directories the strategy needs:
        it drives both dataset wiring (in the trainer) and existence validation
        (in ``LtxTrainerConfig``).
        """


@dataclass
class ModelInputs:
    """Container for model inputs using the Modality-based interface."""

    video: Modality | None
    audio: Modality | None

    # Training targets (for loss computation)
    video_targets: Tensor | None
    audio_targets: Tensor | None

    # Masks for loss computation (True = compute loss for this token)
    video_loss_mask: Tensor | None
    audio_loss_mask: Tensor | None


class TrainingStrategy(ABC):
    """Abstract base class for training strategies.
    Each strategy encapsulates the logic for a specific training mode,
    handling input preparation and loss computation.
    """

    def __init__(self, config: TrainingStrategyConfigBase):
        """Initialize strategy with configuration.
        Args:
            config: Strategy-specific configuration
        """
        self.config = config
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)
        self.video_scale_factors = SpatioTemporalScaleFactors.default()

    @abstractmethod
    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare training inputs from a raw data batch.
        Args:
            batch: Raw batch data from the dataset. Contains:
                - "latents": Video latent data
                - "conditions": Text embeddings with keys:
                    - "video_prompt_embeds": Already processed by embedding connectors
                    - "audio_prompt_embeds": Already processed by embedding connectors
                    - "prompt_attention_mask": Attention mask
                - Additional keys depending on strategy (e.g., "ref_latents" for IC-LoRA)
            timestep_sampler: Sampler for generating timesteps and noise
        Returns:
            ModelInputs containing Modality objects and training targets
        """

    @abstractmethod
    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute the training loss.
        Args:
            video_pred: Video prediction from the transformer model
            audio_pred: Audio prediction from the transformer model (None for video-only)
            inputs: The prepared model inputs containing targets and masks
        Returns:
            Per-element loss tensor of shape [B,]. The trainer reduces to a scalar
            before backward(). Returning unreduced loss enables per-sigma-bucket tracking.
        """

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get strategy-specific metadata to include in checkpoint files.
        Override this method in subclasses to add custom metadata,
        e.g. any parameters that a downstream inference pipeline may need.
        Returns:
            Dictionary of metadata key-value pairs (values must be JSON-serializable)
        """
        return {}

    def _get_video_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        batch_size: int,
        fps: float,
        device: torch.device,
    ) -> Tensor:
        """Generate video position embeddings using ltx_core's native implementation.
        Args:
            num_frames: Number of latent frames
            height: Latent height
            width: Latent width
            batch_size: Batch size
            fps: Frames per second
            device: Target device
        Returns:
            Position tensor of shape [B, 3, seq_len, 2] (float32)
        """
        latent_coords = self._video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=num_frames,
                height=height,
                width=width,
                batch=batch_size,
                channels=128,  # Video latent channels
            ),
            device=device,
        )

        # Convert latent coords to pixel coords with causal fix
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.video_scale_factors,
            causal_fix=True,
        ).float()

        # Scale temporal dimension by 1/fps to get time in seconds
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

        return pixel_coords

    def _get_audio_positions(
        self,
        num_time_steps: int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Generate audio position embeddings using ltx_core's native implementation.
        Args:
            num_time_steps: Number of audio time steps (T, not T*mel_bins)
            batch_size: Batch size
            device: Target device
        Returns:
            Position tensor of shape [B, 1, num_time_steps, 2]
        Note:
            Audio latents should be in patchified format [B, T, C*F] = [B, T, 128]
            where T is the number of time steps, C=8 channels, F=16 mel bins.
            This matches the format produced by AudioPatchifier.patchify().
        """
        mel_bins = 16

        return self._audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(
                frames=num_time_steps,
                mel_bins=mel_bins,
                batch=batch_size,
                channels=8,  # Audio latent channels
            ),
            device=device,
        )

    @staticmethod
    def _create_per_token_timesteps(conditioning_mask: Tensor, sampled_sigma: Tensor) -> Tensor:
        """Create per-token timesteps based on conditioning mask.
        Args:
            conditioning_mask: Boolean mask of shape (batch_size, sequence_length),
                where True = conditioning token (timestep=0), False = target token (use sigma)
            sampled_sigma: Sampled sigma values of shape (batch_size,) or (batch_size, 1, 1)
        Returns:
            Timesteps tensor of shape [batch_size, sequence_length]
        """
        # Expand to match conditioning mask shape [B, seq_len]
        expanded_sigma = sampled_sigma.view(-1, 1).expand_as(conditioning_mask)

        # Conditioning tokens get 0, target tokens get the sampled sigma
        return torch.where(conditioning_mask, torch.zeros_like(expanded_sigma), expanded_sigma)

    @staticmethod
    def _create_first_frame_conditioning_mask(
        batch_size: int,
        sequence_length: int,
        height: int,
        width: int,
        device: torch.device,
        first_frame_conditioning_p: float = 0.0,
    ) -> Tensor:
        """Create conditioning mask for first frame conditioning.
        Args:
            batch_size: Batch size
            sequence_length: Total sequence length
            height: Latent height
            width: Latent width
            device: Target device
            first_frame_conditioning_p: Probability of conditioning on the first frame
        Returns:
            Boolean mask where True indicates first frame tokens (if conditioning is enabled).
            The conditioning decision is drawn independently per batch element so the training
            signal across samples in a batch is i.i.d.
        """
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        if first_frame_conditioning_p > 0:
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                # Per-sample Bernoulli draw so each batch element is independently conditioned.
                per_sample_condition = torch.rand(batch_size, device=device) < first_frame_conditioning_p
                conditioning_mask[per_sample_condition, :first_frame_end_idx] = True

        return conditioning_mask
