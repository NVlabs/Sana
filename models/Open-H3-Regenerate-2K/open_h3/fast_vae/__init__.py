"""Fast H3 VAE decoder for whole clips (non-causal, 24 layers).

From the fast H3 VAE work of StreamRender-H3 (https://nvlabs.github.io/Sana/Sol-Engine/StreamRender-H3/, Section 3.2:
the released 36-layer H3 VAE decoder distilled into a fast decoder with non-overlapping shifted windows). This is
its non-causal variant: the decoder is pruned to 24 layers, decodes native 5-latent blocks bidirectionally with
16x16-token windows and no spatial tiling or temporal overlap, and is distilled from the full H3 VAE. It replaces
only the video decoder; the encoder and latent statistics are the released H3 VAE's.

    vae = FastVAE(weights, base_model)        # base_model/vae/config.json supplies the latent statistics
    frames = vae.decode_frames(latents)       # normalized (24, T, H, W) -> (F, 16H, 16W, 3) uint8 on the GPU

engine="fast" (default) runs the deployment engine in noncausal_fast3.py (FlashAttention-4 varlen windows, fused
Triton kernels, quack GEMMs, one CUDA graph per (T, H, W)); engine="ref" runs the plain PyTorch decoder in
decoder.py (SDPA, any CUDA GPU). The first call for a new (T, H, W) builds the engine and records its graph.
"""
from collections import OrderedDict
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .decoder import KEEP, NonCausalViTDecoder

WEIGHTS_FILE = "vae_fast/fast_h3_vae_noncausal_24l.safetensors"
WEIGHTS_SHA256 = "8658f1e85b5997c4ba716406a2f148bfa36cf97d4d5fcd402c7bd341f54e77a1"
PIXEL_MEAN, PIXEL_STD = (.485, .456, .406), (.229, .224, .225)


class FastVAE:
    def __init__(self, weights, base_model, engine="fast", max_engines=4, device="cuda"):
        config = json.loads((Path(base_model) / "vae/config.json").read_text())
        model = NonCausalViTDecoder(config["latents_mean"], config["latents_std"], KEEP[24], time_norm=5,
                                    heads=config["decoder_num_attention_heads"], dim_head=config["decoder_attention_head_dim"],
                                    rope_theta=config["decoder_rope_theta"], rope_dim_ratio=config["decoder_rope_dim_ratio"],
                                    eps=config["decoder_norm_eps"], latent_channels=config["latent_channels"])
        model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        self.model = model.to(device).eval().requires_grad_(False)
        self.engine, self.max_engines = engine, max_engines
        self._engines = OrderedDict()
        self.mean = torch.tensor(PIXEL_MEAN, device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(PIXEL_STD, device=device).view(1, 3, 1, 1, 1)

    def _get(self, T, H, W):
        key = (T, H, W)
        if key in self._engines:
            self._engines.move_to_end(key)
            return self._engines[key]
        from .noncausal_fast3 import NonCausalFast3
        engine = NonCausalFast3(self.model, T, H, W, cuda_graph=True)
        self._engines[key] = engine
        while len(self._engines) > self.max_engines:
            self._engines.popitem(last=False)
            torch.cuda.empty_cache()
        return engine

    @torch.no_grad()
    def decode(self, z):
        """z: normalized latents (1, T, 24, H, W) -> ImageNet-normalized pixels (1, 3, F, 16H, 16W), fp32."""
        if self.engine == "ref":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(z.float()).float()
        _, T, _, H, W = z.shape
        return self._get(T, H, W).decode(z.float().contiguous())

    @torch.no_grad()
    def decode_frames(self, latents):
        """latents: normalized (24, T, H, W) -> (F, 16H, 16W, 3) uint8 on the GPU."""
        pixels = self.decode(latents.cuda().transpose(0, 1)[None])
        return (pixels * self.std + self.mean).clamp_(0, 1).mul_(255).round_().to(torch.uint8)[0].permute(1, 2, 3, 0)
