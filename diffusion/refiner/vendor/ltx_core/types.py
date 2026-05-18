from dataclasses import dataclass, replace
from typing import NamedTuple

import torch


class VideoPixelShape(NamedTuple):
    """
    Shape of the tensor representing the video pixel array. Assumes BGR channel format.
    """

    batch: int
    frames: int
    height: int
    width: int
    fps: float


class SpatioTemporalScaleFactors(NamedTuple):
    """
    Describes the spatiotemporal downscaling between decoded video space and
    the corresponding VAE latent grid.
    """

    time: int
    width: int
    height: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, width=32, height=32)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """
    Shape of the tensor representing video in VAE latent space.
    The latent representation is a 5D tensor with dimensions ordered as
    (batch, channels, frames, height, width). Spatial and temporal dimensions
    are downscaled relative to pixel space according to the VAE's scale factors.
    """

    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    def token_count(self) -> int:
        """Number of tokens after patchification with the default patch size of 1."""
        return self.frames * self.height * self.width

    def mask_shape(self) -> "VideoLatentShape":
        return self._replace(channels=1)

    @staticmethod
    def from_pixel_shape(
        shape: VideoPixelShape,
        latent_channels: int = 128,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    ) -> "VideoLatentShape":
        frames = (shape.frames - 1) // scale_factors[0] + 1
        height = shape.height // scale_factors[1]
        width = shape.width // scale_factors[2]

        return VideoLatentShape(
            batch=shape.batch,
            channels=latent_channels,
            frames=frames,
            height=height,
            width=width,
        )

    def upscale(self, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> "VideoLatentShape":
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * scale_factors.time + 1,
            height=self.height * scale_factors.height,
            width=self.width * scale_factors.width,
        )


class AudioLatentShape(NamedTuple):
    """
    Shape of audio in VAE latent space: (batch, channels, frames, mel_bins).
    mel_bins is the number of frequency bins from the mel-spectrogram encoding.
    """

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    def token_count(self) -> int:
        """Number of tokens after patchification."""
        return self.frames

    def mask_shape(self) -> "AudioLatentShape":
        return self._replace(channels=1, mel_bins=1)

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)

        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )

    @staticmethod
    def from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        return AudioLatentShape.from_duration(
            batch=shape.batch,
            duration=float(shape.frames) / float(shape.fps),
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
        )


@dataclass(frozen=True)
class Audio:
    """
    Container for decoded audio samples and metadata.
    Attributes:
        waveform: Audio waveform tensor.
        sampling_rate: Sampling rate (Hz) of the waveform.
    """

    waveform: torch.Tensor
    sampling_rate: int

    def to(self, **kwargs: object) -> "Audio":
        return replace(self, waveform=self.waveform.to(**kwargs))


@dataclass(frozen=True)
class LatentState:
    """
    State of latents during the diffusion denoising process.
    Attributes:
        latent: The current noisy latent tensor being denoised.
        denoise_mask: Mask encoding the denoising strength for each token (1 = full denoising, 0 = no denoising).
        positions: Positional indices for each latent element, used for positional embeddings.
        clean_latent: Initial state of the latent before denoising, may include conditioning latents.
        attention_mask: Optional 2D self-attention mask of shape (B, T, T). Values in [0, 1] where 1 = full attention,
            0 = no attention. None means full attention everywhere. Built incrementally by conditioning items.
    """

    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: torch.Tensor
    attention_mask: torch.Tensor | None = None

    def clone(self) -> "LatentState":
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
        )
