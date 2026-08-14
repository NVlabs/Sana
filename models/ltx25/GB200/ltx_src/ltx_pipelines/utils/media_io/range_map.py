"""Map between display/working [0,1] and VAE [-1,1]."""

from __future__ import annotations

import torch


def normalize_images(images: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (images / 127.5 - 1.0).to(device=device, dtype=dtype)


def to_vae_range(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] working-space codes to [-1, 1] (VAE input convention).
    Raises:
        ValueError: If ``x`` is outside ``[0, 1]`` (callers must compress or
            already hold valid working-space codes — this does not clamp).
    """
    if bool((x < 0).any() or (x > 1).any()):
        raise ValueError(f"to_vae_range expects values in [0, 1]; got min={float(x.min())}, max={float(x.max())}")
    return x * 2.0 - 1.0


def from_vae_range(z: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] (VAE output convention) to [0, 1]."""
    return torch.clamp((z + 1.0) / 2.0, 0.0, 1.0)
