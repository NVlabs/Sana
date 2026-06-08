# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming decoder adapter for ``AutoencoderKLCausalLTX2Video``.

Wraps the causal LTX-2 VAE so it can be driven chunk-by-chunk in an interactive
inference loop. Each call to :meth:`CausalVaeStreamingDecoder.decode_chunk`
forwards one block of latent frames through the decoder while reusing the
persistent per-layer feature cache (``_decoder_cache.feat_map``). The output is
the pixel-space block for those latent frames only, ready to be written to the
progressive MP4 writer.

The adapter is intentionally minimal: it owns no state beyond the wrapped VAE
and a few scalar flags. State that persists across chunks lives on the VAE
itself (its ``DecoderCacheManager``), which is what allows the LTX-2 causal
decoder to keep temporal context without re-feeding earlier frames.
"""

from __future__ import annotations

import torch

from diffusion.model.ltx2.causal_vae import AutoencoderKLCausalLTX2Video


class CausalVaeStreamingDecoder:
    """Drive ``AutoencoderKLCausalLTX2Video`` one latent block at a time.

    Args:
        vae: The causal VAE instance (already on the target device + dtype).
        scaling_factor: Override for ``vae.config.scaling_factor``. Defaults to
            the VAE's own value.
    """

    def __init__(
        self,
        vae: AutoencoderKLCausalLTX2Video,
        *,
        scaling_factor: float | None = None,
    ) -> None:
        if not getattr(vae.decoder, "is_causal", False):
            raise ValueError(
                "CausalVaeStreamingDecoder requires a causal decoder "
                f"(got decoder.is_causal={getattr(vae.decoder, 'is_causal', None)})."
            )
        self._vae = vae
        self._scaling_factor = float(scaling_factor if scaling_factor is not None else vae.config.scaling_factor)
        self._first_chunk = True

    @property
    def vae(self) -> AutoencoderKLCausalLTX2Video:
        return self._vae

    def reset(self) -> None:
        """Clear the persistent decoder cache; the next call will be the first chunk."""
        self._vae.clear_decoder_cache()
        self._first_chunk = True

    @torch.no_grad()
    def decode_chunk(self, z_chunk: torch.Tensor) -> torch.Tensor:
        """Decode one latent block, mutating the VAE's persistent feature cache.

        Args:
            z_chunk: Normalized latent block ``(B, C, T_lat_block, H_lat, W_lat)``
                in the same convention as ``vae_encode`` returns (i.e. already
                shifted/scaled by ``(z - mean) * scaling_factor / std``).

        Returns:
            Pixel-space tensor ``(B, 3, T_pix_block, H_pix, W_pix)`` for this
            block only; range ``[-1, 1]``.
        """
        if z_chunk.dim() != 5:
            raise ValueError(f"z_chunk must be 5D (B,C,T,H,W); got shape {tuple(z_chunk.shape)}.")
        if z_chunk.shape[2] <= 0:
            raise ValueError("z_chunk must contain at least one latent frame.")

        vae = self._vae
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z_chunk.device, z_chunk.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(z_chunk.device, z_chunk.dtype)

        # Reverse the (z - mean) * sf / std normalization done at encode/sample time.
        z_unnormalized = z_chunk * latents_std / self._scaling_factor + latents_mean
        z_unnormalized = z_unnormalized.to(vae.dtype)

        # Frame-by-frame causal decode: reset the per-layer feature cache only on
        # the first chunk of a stream; later chunks carry temporal state via the
        # cache. Iterating decode_per_frame_with_cache chunk-by-chunk is
        # bit-identical to one big single-call decode_per_frame_with_cache.
        reset = self._first_chunk
        pixel_chunks = list(
            vae.decode_per_frame_with_cache(
                z_unnormalized,
                temb=None,
                causal=True,
                reset_cache=reset,
            )
        )
        self._first_chunk = False
        return torch.cat(pixel_chunks, dim=2)
