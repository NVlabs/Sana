"""3D Neighborhood Attention via NATTEN + absolute RoPE prelude.
Parameter shell shared by det ``NABlock`` and both diff-attn roles.
Diffusion AdaLN residuals live in pathway packages (each owns its RoPE);
det stages use ``det_attn_rope`` from :meth:`NeighborhoodAttention3D.forward`.
``attention_function`` selects the NA backend (NATTEN, Triton/eager fallback, or CuTe DSL).
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from ltx_core.model.video_vae.transformer.det_attn_rope import det_qkv_rope
from ltx_core.model.video_vae.transformer.qkv import QKVProjections
from ltx_core.model.video_vae.transformer.rope_math import (
    DEFAULT_ABS_ROPE_NUM_TILES,
    default_rope_dim_split,
    rope_inv_freqs,
)

try:
    import natten

    _NATTEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    natten = None  # type: ignore[assignment]
    _NATTEN_AVAILABLE = False


def natten_available() -> bool:
    return _NATTEN_AVAILABLE


class NAAttentionCallable(Protocol):
    """A windowed 3D neighborhood-attention backend.
    Q/K/V arrive as ``(B, T, H, W, NH, HD)``, already normed, scaled and RoPE'd;
    the return is ``(B, T, H, W, NH*HD)`` or anything reshapeable to it. The owning
    module is passed so a backend can read configuration (``kernel_size``, softmax
    bound, …). Backend-specific settings (NATTEN's kernel pin) live on the callable.
    """

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor: ...


class NattenAttention(NAAttentionCallable):
    """``natten.na3d``, the default backend.
    ``backend`` pins ``na3d``'s own kernel choice (e.g. ``"cutlass-fna"``); ``None``
    leaves NATTEN's auto-pick (hopper-fna on H100, etc.).
    """

    def __init__(self, backend: str | None = None) -> None:
        self._backend = backend

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if not _NATTEN_AVAILABLE:
            raise ImportError(
                "natten is required for NeighborhoodAttention3D. "
                "Install with: uv sync --package ltx-core --extra natten "
                '(or: uv pip install "natten==0.21.7+torch2130cu132" -f https://whl.natten.org; '
                "requires torch==2.13.0+cu132)"
            )
        # scale=1.0: callers already applied ``attn.scale`` to Q.
        # RMSNorm under bf16 autocast can leave Q/K in float32 while V stays bf16;
        # natten requires a uniform dtype (same cast pattern as flash-attn paths).
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        return natten.na3d(q, k, v, kernel_size=attn.kernel_size, scale=1.0, backend=self._backend)


class NeighborhoodAttention3D(nn.Module):
    """3D Neighborhood Attention with absolute RoPE + pluggable NA backend.
    Q/K receive absolute RoPE; attention is ``attention_function`` (NATTEN by
    default; Triton or eager SDPA when natten is missing; CuTe DSL via
    DiffVAE BLACKWELL_DSL install). Relative gather-based NA is not used as a
    production gather path on this branch.
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
        # Kept for the chunked opaque residual (string arg); callable is the swap surface.
        self.natten_backend: str | None = None
        self.attention_function: NAAttentionCallable = NattenAttention()

        self.register_buffer("rope_inv_t", rope_inv_freqs(rope_dim_split[0], rope_base), persistent=False)
        self.register_buffer("rope_inv_h", rope_inv_freqs(rope_dim_split[1], rope_base), persistent=False)
        self.register_buffer("rope_inv_w", rope_inv_freqs(rope_dim_split[2], rope_base), persistent=False)

        self.qkv = QKVProjections(dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

        # W-chunking configuration (consumed by ``chunked.attn``).
        self.w_chunks = 1  # 1 = no chunking

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V as owned contiguous ``(B,T,H,W,NH,HD)`` tensors."""
        batch, t, h, w, _ = x.shape
        q, k, v = self.qkv(x)
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Det-stage NA: opaque abs-RoPE via ``det_attn_rope`` + ``attention_function``.
        ``x``/output: (B, T, H, W, C) — channels-last. RoPE positions are local
        0-based (see ``det_attn_rope`` module docstring for why that is
        equivalent under tiled decode).
        """
        batch, t, h, w, _ = x.shape
        kt, kh, kw = self.kernel_size
        if t < kt or h < kh or w < kw:
            raise ValueError(
                f"3D neighborhood attention requires spatial dims >= kernel_size; "
                f"got (T,H,W)=({t},{h},{w}) vs kernel={self.kernel_size}"
            )

        q, k, v = det_qkv_rope(self, x)
        # natten's CUTLASS kernel silently produces wrong output if inputs are non-contiguous.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = self.attention_function(self, q, k, v)
        out = out.reshape(batch, t, h, w, self.dim)
        return self.proj(out)


def configure_w_chunks(module_root: nn.Module, w_chunks: int = 1) -> None:
    """Set W-chunking on ``NeighborhoodAttention3D`` under ``module_root``.
    When ``w_chunks > 1``, also sets ``rope_num_tiles=1`` so ``chunked.attn``
    owns the W axis (RoPE W-tiling would double-split). Pass only the diffusion
    residual subtree — det-stage attention must keep its default RoPE tiling.
    """
    for module in module_root.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.w_chunks = w_chunks
            if w_chunks > 1:
                module.rope_num_tiles = 1
