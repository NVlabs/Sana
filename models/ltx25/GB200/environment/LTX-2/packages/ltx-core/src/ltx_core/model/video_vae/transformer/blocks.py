"""NABlock and DiffusionNABlock for the diffusion video VAE."""

from __future__ import annotations

import torch
from torch import nn

from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D
from ltx_core.model.video_vae.transformer.layers import AdaLNZero, modulate
from ltx_core.model.video_vae.transformer.swiglu import SwiGLU


class NABlock(nn.Module):
    """Pre-norm transformer block: NA -> SwiGLU MLP with residual adds."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Channels-last in/out: (B, T, H, W, C)."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionNABlock(nn.Module):
    """NA + SwiGLU MLP with shared AdaLN-Zero scale/shift (ungated residuals).
    The decoder owns one ``shared_adaln``; each block adds a ``scale_shift_table``
    residual. AdaLN stays 7-chunk for checkpoint shape compat; gate slots are
    discarded. Legacy ``gate_msa`` / ``gate_mlp`` / ``gate_ctx`` tensors are
    folded into ``attn.proj`` / ``mlp.w_down`` / ``context_proj`` at load time
    (see DiffVAE decoder SDOps).
    """

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.context_channels = context_channels
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)
        self.attn.proj.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]

        x = x + self.context_proj(latent_context)
        x = x + self.attn(modulate(self.norm1(x), scale_msa, shift_msa))
        x = x + self.mlp(modulate(self.norm2(x), scale_mlp, shift_mlp))
        return x

    def forward_combined(
        self,
        combined: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """``combined`` is ``cat([latent_context, x], dim=-1)`` — one tensor's T/H/W symbols."""
        latent_context = combined[..., : self.context_channels]
        x = combined[..., self.context_channels :]
        return self.forward(x, latent_context, modulation)
