"""Latent upsampler model components."""

from ltx_core.model.upsampler.model import LatentUpsampler, upsample_video
from ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator

__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "upsample_video",
]
