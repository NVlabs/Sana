"""SwiGLU MLP module for NA / DiffusionNA blocks."""

from __future__ import annotations

from enum import Enum

import torch
from torch import nn

from ltx_core.model.video_vae.transformer.swiglu_efficient import (
    DEFAULT_SWIGLU_TILE_SPEC,
    SwiGLUTileSpec,
    swiglu_tiled,
    swiglu_triton,
)


class SwiGLUMode(Enum):
    """``TRITON``: force Triton fuse. ``TILED``: chunked workspace path (Triton if CUDA)."""

    TRITON = "triton"
    TILED = "tiled"


class SwiGLU(nn.Module):
    """Gated MLP: ``w_down(silu(w_gate(x)) * w_up(x))``.
    - ``TRITON``: ``swiglu_triton`` (requires Triton/CUDA).
    - ``TILED``: ``swiglu_tiled`` — hard token cap + one reusable ``(chunk, hidden)``
      workspace (donor memory-efficient path; Triton fuse when available).
    Defaults: ``mode=TILED``, ``tile_size=16384``. ``compile_diffusion_decoder``
    forces ``TILED`` (custom_op so Dynamo does not unroll the chunk loop).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mode: SwiGLUMode = SwiGLUMode.TILED,
        tile: SwiGLUTileSpec = DEFAULT_SWIGLU_TILE_SPEC,
    ) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.mode = mode
        self.tile = tile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        weights = (self.w_gate.weight, self.w_up.weight, self.w_down.weight)
        if self.mode is SwiGLUMode.TRITON:
            return swiglu_triton(x, *weights, self.tile)
        # custom_op + workspace chunking (compile-safe; peak O(chunk·hidden))
        return swiglu_tiled(x, *weights, self.tile)


def configure_swiglu(
    module_root: nn.Module,
    mode: SwiGLUMode | str = SwiGLUMode.TILED,
    *,
    num_tiles: int | None = None,
    tile_size: int | None = None,
) -> None:
    """Set ``mode`` / tile spec on every ``SwiGLU`` under ``module_root``.
    Pass exactly one of ``num_tiles`` or ``tile_size`` to replace the tile spec;
    omit both to leave each module's existing ``tile`` unchanged.
    """
    resolved = SwiGLUMode(mode) if isinstance(mode, str) else mode
    tile: SwiGLUTileSpec | None
    tile = None if num_tiles is None and tile_size is None else SwiGLUTileSpec(num_tiles=num_tiles, tile_size=tile_size)

    for module in module_root.modules():
        if isinstance(module, SwiGLU):
            module.mode = resolved
            if tile is not None:
                module.tile = tile
