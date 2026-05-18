from dataclasses import replace
from typing import Protocol

import torch

from diffusion.refiner.vendor.ltx_core.types import LatentState


class Noiser(Protocol):
    """Protocol for adding noise to a latent state during diffusion."""

    def __call__(self, latent_state: LatentState, noise_scale: float) -> LatentState: ...


class GaussianNoiser(Noiser):
    """Adds Gaussian noise to a latent state, scaled by the denoise mask."""

    def __init__(self, generator: torch.Generator):
        super().__init__()

        self.generator = generator

    def __call__(self, latent_state: LatentState, noise_scale: float = 1.0) -> LatentState:
        noise = torch.randn(
            *latent_state.latent.shape,
            device=latent_state.latent.device,
            dtype=latent_state.latent.dtype,
            generator=self.generator,
        )
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
        return replace(
            latent_state,
            latent=latent.to(latent_state.latent.dtype),
        )
