"""3D Neighborhood Attention via NATTEN + absolute RoPE prelude."""

from __future__ import annotations

import torch
from torch import nn

from ltx_core.model.video_vae.transformer.qkv import QKVProjections
from ltx_core.model.video_vae.transformer.rope import (
    DEFAULT_ABS_ROPE_NUM_TILES,
    DEFAULT_ROPE_USE_CUSTOM_OP,
    default_rope_dim_split,
    qkv_rope,
)

try:
    import natten

    _NATTEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    natten = None  # type: ignore[assignment]
    _NATTEN_AVAILABLE = False


def natten_available() -> bool:
    return _NATTEN_AVAILABLE


class NeighborhoodAttention3D(nn.Module):
    """3D Neighborhood Attention with absolute RoPE + NATTEN ``na3d``.
    Q/K receive absolute RoPE (default fixed W-tile count); attention is
    ``natten.na3d``. Relative gather-based NA is not used on this branch.
    NATTEN shifts its window inward at grid boundaries instead of
    clamp-and-mask; interior positions match the gather reference closely,
    boundary positions may differ slightly.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_dim_split: tuple[int, int, int] | None = None,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, f"dim={dim} not divisible by head_dim={head_dim}"
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5

        if rope_dim_split is None:
            rope_dim_split = default_rope_dim_split(head_dim)
        assert sum(rope_dim_split) == head_dim, f"rope_dim_split={rope_dim_split} must sum to head_dim={head_dim}"
        self.rope_dim_split = rope_dim_split
        self.rope_base = rope_base
        self.rope_num_tiles = DEFAULT_ABS_ROPE_NUM_TILES
        self.rope_compute_dtype = torch.float32
        self.rope_use_custom_op = DEFAULT_ROPE_USE_CUSTOM_OP
        # Optional ``natten.na3d`` backend pin (e.g. ``"cutlass-fna"``). ``None``
        # leaves NATTEN's auto-pick (hopper-fna on H100, etc.).
        self.natten_backend: str | None = None

        self.qkv = QKVProjections(dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V as owned contiguous ``(B,T,H,W,NH,HD)`` tensors."""
        batch, t, h, w, _ = x.shape
        q, k, v = self.qkv(x)
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D NA. ``x``/output: (B, T, H, W, C) -- channels-last.
        RoPE positions inside are always local (0-based) -- no absolute tile
        origin is threaded through here; see ``rope.py``'s module docstring
        for why that's provably equivalent to using the true absolute origin,
        given every call here processes exactly one tile in isolation.
        """
        if not _NATTEN_AVAILABLE:
            raise ImportError(
                "natten is required for NeighborhoodAttention3D. "
                "Install with: uv sync --package ltx-core --extra natten "
                '(or: uv pip install "natten==0.21.5+torch290cu128" -f https://whl.natten.org)'
            )
        batch, t, h, w, _ = x.shape
        kt, kh, kw = self.kernel_size
        if t < kt or h < kh or w < kw:
            raise ValueError(
                f"natten.na3d requires spatial dims >= kernel_size; "
                f"got (T,H,W)=({t},{h},{w}) vs kernel={self.kernel_size}"
            )

        q, k, v = qkv_rope(
            self,
            x,
            num_tiles=self.rope_num_tiles,
            compute_dtype=self.rope_compute_dtype,
        )
        # natten's CUTLASS kernel silently produces wrong output if inputs are non-contiguous.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = natten.na3d(
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            scale=1.0,
            backend=self.natten_backend,
        )
        out = out.reshape(batch, t, h, w, self.dim)
        return self.proj(out)
