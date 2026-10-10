"""Causal ViT-24 video decoder as a drop-in `models.video_decoder` for the streaming H3 pipeline.

The released H3 VAE's ViT decoder pruned to 24 layers and distilled to chunk-causal decoding
(vae_prune/causal_vit_decoder.py). Decoding follows the stream: latents are buffered and every
2 consecutive latents (absolute indices 2k, 2k+1) are decoded together as soon as both exist,
with per-layer K/V of the previous 4 latents. Deployment kernels: vae_prune/causal_vit_stream.py
(FlashAttention-4 varlen windows + one CUDA graph per chunk).

Encoding stays with `models.video_vae` (the streaming TAE); this module only decodes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file
from torch import nn

VAE_PRUNE = str(Path(__file__).resolve().parent / "native")
if VAE_PRUNE not in sys.path:
    sys.path.insert(0, VAE_PRUNE)

from causal_vit_decoder import PATCH_T, CausalViTDecoder  # noqa: E402
from dev.yanzuolu.projects.minimax_h3.modeling.taeh3 import H3_VIDEO_TEMPORAL_MAPPING  # noqa: E402

KEEP_24 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 31, 32, 35]


class _CausalViTStream:
    """One sample's decoder state. `decode` takes denormalized NTCHW latents, returns NTCHW RGB or None."""

    requires_single_latent_dispatch = False

    def __init__(self, codec: "CausalViTVideoDecoder") -> None:
        self.codec = codec
        self.pending: list[torch.Tensor] = []
        self.fast = None
        self.t = 0

    def _decode_pair(self, z: torch.Tensor) -> torch.Tensor:
        c = self.codec
        if self.fast is None:
            self.fast = c.fast_for(z.shape[-2], z.shape[-1], z.device)
            self.fast.init_state()
        px = self.fast.decode_chunk(z)                                  # 1,3,F,h,w decode_base space
        self.t += c.chunk
        return c.processor.revert_tensor(px.float()).transpose(1, 2)    # 1,F,3,h,w

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor | None:
        c = self.codec
        z = (latents.float() - c.latents_mean.view(1, 1, -1, 1, 1)) / c.latents_std.view(1, 1, -1, 1, 1)
        self.pending.extend(z[:, i:i + 1] for i in range(z.shape[1]))
        outs = []
        while len(self.pending) >= c.chunk:
            pair = torch.cat(self.pending[:c.chunk], 1)
            del self.pending[:c.chunk]
            outs.append(self._decode_pair(pair))
        return torch.cat(outs, 1) if outs else None

    @torch.no_grad()
    def flush_decoder(self) -> torch.Tensor | None:
        """Odd final latent: pair it with a copy of itself and keep only its own frames."""
        if not self.pending:
            return None
        c = self.codec
        z = self.pending[0]
        n_own = PATCH_T - (c.frames_to_trim if self.t % c.block_latents == 0 else 0)
        self.pending.clear()
        return self._decode_pair(torch.cat([z, z], 1))[:, :n_own]


class CausalViTVideoDecoder(nn.Module):
    video_temporal_mapping = H3_VIDEO_TEMPORAL_MAPPING          # read on the class by the meta model

    def __init__(self, vae_config: str, student_weights: str, keep_layers: list[int] | None = None,
                 chunk: int = 2, cache_latents: int = 4, cuda_graph: bool = True) -> None:
        super().__init__()
        from dev.yanzuolu.common.config import CfgNode
        from dev.yanzuolu.common.model import build_model
        if isinstance(vae_config, dict):
            cfg = vae_config
        else:
            with open(vae_config) as handle:
                cfg = yaml.safe_load(handle)
        vae = build_model("video_vae", CfgNode(cfg["models"]["video_vae"]))["model"].eval()
        self.vit = CausalViTDecoder(vae, keep_layers or KEEP_24, chunk=chunk, cache_latents=cache_latents)
        missing = self.vit.load_state_dict(load_file(student_weights, device="cpu"), strict=False)
        assert not missing.missing_keys and not missing.unexpected_keys, missing
        self.processor = vae.processor
        self.vae_ratio, self.vae_ratio_t = vae.vae_ratio, vae.vae_ratio_t
        assert vae.video_temporal_mapping == self.video_temporal_mapping
        self.register_buffer("latents_mean", vae.latents_mean.detach().clone().float(), persistent=False)
        self.register_buffer("latents_std", vae.latents_std.detach().clone().float(), persistent=False)
        del vae
        self.chunk, self.block_latents, self.frames_to_trim = chunk, self.vit.block_latents, self.vit.frames_to_trim
        self.latent_chunk = chunk
        self.cuda_graph = cuda_graph
        self._fast: dict[Any, Any] = {}

    def fast_for(self, H: int, W: int, device: torch.device):
        """One deployment decoder per latent resolution; streams run one after another and reset it."""
        key = (H, W, str(device))
        if key not in self._fast:
            from causal_vit_stream import StreamingCausalViT
            self.vit.to(device)
            self._fast[key] = StreamingCausalViT(self.vit, H, W, cuda_graph=self.cuda_graph)
        return self._fast[key]

    def create_stream(self) -> _CausalViTStream:
        return _CausalViTStream(self)


__all__ = ["CausalViTVideoDecoder"]
