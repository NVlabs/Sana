"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com

Clean-room CD-FVD VideoMAE-v2 implementation, adapted from
OpenGVLab/VideoMAEv2 (MIT, https://github.com/OpenGVLab/VideoMAEv2). Protocol
constants follow cdfvd_videomae2_ssv2_prelogits_1408: 1408-dim pre-logits
features (mean-pool over all tokens + fc_norm), 224x224 bilinear stretch (no
aspect preservation, no crop), [0,1] input range (NO ImageNet normalization,
NO [-1,1] -- deliberately different from FVDI3D), 16-frame clips, tubelet
size 2. The original eval-time dropout and drop-path operations are no-ops and
are omitted here.

Weights: Hugging Face OpenGVLab/VideoMAE2 revision
97d119d270aca7a1217fff7f413c0c543eb325d9,
mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth, sha256
5a210a92f035dff30c53b46157b612e7a1a5d3c99700e1b2d71da5c399ca7e70. Load
the bare state_dict unwrapped from checkpoint["module"]. Known deviation:
Frechet computation is not implemented here; callers reuse fvd_i3d.py's N-1
covariance plus symmetric-eigh sandwich implementation, while the original
cd-fvd uses N plus scipy sqrtm(Sigma_1 Sigma_2) (mathematically equivalent).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Video tubelet embedding."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(
            3,
            1408,
            kernel_size=(2, 14, 14),
            stride=(2, 14, 14),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention with separate query and value biases."""

    def __init__(self):
        super().__init__()
        self.num_heads = 16
        self.head_dim = 88
        self.q_bias = nn.Parameter(torch.zeros(1408))
        self.v_bias = nn.Parameter(torch.zeros(1408))
        self.qkv = nn.Linear(1408, 4224, bias=False)
        self.proj = nn.Linear(1408, 1408, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class Mlp(nn.Module):
    """Transformer feed-forward network."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1408, 6144)
        self.fc2 = nn.Linear(6144, 1408)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-normalized VideoMAE transformer block."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(1408, eps=1e-6)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(1408, eps=1e-6)
        self.mlp = Mlp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def _sinusoid_encoding_table() -> torch.Tensor:
    positions = torch.arange(2048, dtype=torch.float64).unsqueeze(1)
    dimensions = torch.arange(1408, dtype=torch.float64).unsqueeze(0)
    angles = positions / torch.pow(10000.0, 2.0 * torch.floor(dimensions / 2.0) / 1408.0)
    table = torch.empty((2048, 1408), dtype=torch.float64)
    table[:, 0::2] = torch.sin(angles[:, 0::2])
    table[:, 1::2] = torch.cos(angles[:, 1::2])
    return table.to(dtype=torch.float32).unsqueeze(0)


class FVDVideoMAE(nn.Module):
    """VideoMAE-v2 ViT-giant/14 SSv2 pre-logits feature extractor."""

    feature_dim = 1408

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.register_buffer("pos_embed", _sinusoid_encoding_table(), persistent=False)
        self.blocks = nn.ModuleList([Block() for _ in range(40)])
        self.fc_norm = nn.LayerNorm(1408, eps=1e-6)
        # Retained only for strict loading of the SSv2 checkpoint; forward never uses it.
        self.head = nn.Linear(1408, 174)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.fc_norm(x.mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    @torch.no_grad()
    def extract(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract one pre-logits feature vector per uint8 BTHWC video."""
        # Feature extraction is an eval-only protocol. Keep the same guard as
        # FVDI3D so the model config cannot silently extract in training mode;
        # it must carry ``runtime: {training: false}``.
        assert not self.training, "FVDVideoMAE.extract requires eval mode (runtime: {training: false})"
        assert videos.dtype == torch.uint8
        assert videos.ndim == 5 and videos.shape[-1] == 3
        assert videos.shape[1] == 16, "FVDVideoMAE requires 16-frame clips (chunk_len must be 16)"

        batch, time, height, width, channels = videos.shape
        # Deliberately stop at [0,1]: no ImageNet or [-1,1] normalization.
        x = videos.to(dtype=torch.float32).div(255.0)
        x = x.reshape(batch * time, height, width, channels).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.reshape(batch, time, channels, 224, 224).permute(0, 2, 1, 3, 4)

        parameter_dtype = next(self.parameters()).dtype
        if x.dtype != parameter_dtype:
            x = x.to(dtype=parameter_dtype)
        return self.forward_features(x).to(dtype=torch.float32)
