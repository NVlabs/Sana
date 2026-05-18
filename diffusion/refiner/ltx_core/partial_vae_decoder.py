"""Partial VAE decoder for perceptual temporal feature extraction.

Loads only ``conv_in`` + the first N ``up_blocks`` of the full VAE decoder.
All parameters are frozen; gradients flow through the input tensor only
(for student path backprop).

Typical usage in distillation training::

    partial_dec = PartialVAEDecoder.from_checkpoint(
        checkpoint_path, num_up_blocks=1, device=device, dtype=dtype,
    )
    # teacher (no grad)
    with torch.no_grad():
        t_feat = partial_dec(teacher_latent_bcfhw)
    # student (grad flows through input)
    s_feat = partial_dec(student_latent_bcfhw)
    loss = F.mse_loss(s_feat, t_feat)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from diffusion.refiner.vendor.ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from diffusion.refiner.vendor.ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VideoDecoderConfigurator,
)
from diffusion.refiner.vendor.ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D

logger = logging.getLogger(__name__)


class PartialVAEDecoder(nn.Module):
    """Frozen partial VAE decoder: ``conv_in`` + first *num_up_blocks* blocks.

    Parameters are frozen (``requires_grad=False``), but the module runs in
    normal forward mode so that input gradients propagate back for the
    student loss path.
    """

    def __init__(self, conv_in: nn.Module, up_blocks: nn.ModuleList, per_channel_statistics: nn.Module, causal: bool):
        super().__init__()
        self.conv_in = conv_in
        self.up_blocks = up_blocks
        self.per_channel_statistics = per_channel_statistics
        self.causal = causal
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str,
        num_up_blocks: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.bfloat16,
    ) -> PartialVAEDecoder:
        """Load partial decoder from a fused checkpoint.

        Loads the full decoder, extracts the needed layers, then discards
        the rest to save memory.
        """
        builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        )
        full_decoder = builder.build(device="cpu", dtype=dtype)

        # Extract only the layers we need
        conv_in = full_decoder.conv_in
        up_blocks = nn.ModuleList(list(full_decoder.up_blocks)[:num_up_blocks])
        per_channel_statistics = full_decoder.per_channel_statistics
        causal = full_decoder.causal

        partial = PartialVAEDecoder(conv_in, up_blocks, per_channel_statistics, causal)
        partial = partial.to(device=device, dtype=dtype).eval()

        n_params = sum(p.numel() for p in partial.parameters())
        logger.info(
            "PartialVAEDecoder: %d up_blocks, %.1fM params, device=%s",
            num_up_blocks,
            n_params / 1e6,
            device,
        )

        # Help GC reclaim the rest of the full decoder
        del full_decoder

        return partial

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass: denormalize → conv_in → up_blocks.

        Parameters
        ----------
        latent : (B, 128, F, H, W) — raw latent in model space

        Returns
        -------
        features : (B, C', F', H', W') — intermediate decoder features
        """
        sample = self.per_channel_statistics.un_normalize(latent)
        sample = self.conv_in(sample, causal=self.causal)
        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                sample = up_block(sample, causal=self.causal)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal)
            else:
                sample = up_block(sample, causal=self.causal)
        return sample
