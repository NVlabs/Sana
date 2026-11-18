"""DiffVAE NA fallbacks when natten is unavailable: Triton then eager SDPA.
Vendored from comfy-kitchen (Apache-2.0). Selection order for NATTEN-kind recipes::
    natten → (loud warning) Triton → eager tiled SDPA
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from ltx_core.model.video_vae.transformer.fallback_na.eager import na3d as eager_na3d

if TYPE_CHECKING:
    from ltx_core.model.video_vae.transformer.attention import NAAttentionCallable, NeighborhoodAttention3D

logger = logging.getLogger(__name__)

_NO_NATTEN_WARNING = (
    "================================================================================\n"
    "DiffVAE: natten is NOT installed. Falling back to a slower neighborhood-attention\n"
    "backend (Triton if available, else pure-PyTorch tiled SDPA). This path is for\n"
    "compatibility only — install natten for production DiffVAE decode:\n"
    "  uv sync --package ltx-core --extra natten\n"
    "================================================================================"
)


def triton_na_available() -> bool:
    """True when CUDA is available and the ``triton`` package imports cleanly.
    Triton on Windows is supported via the community ``triton-windows`` builds
    (https://github.com/triton-lang/triton-windows); there is no platform ban here.
    Import failures beyond ``ImportError`` (e.g. ``OSError`` when libcuda is missing)
    are treated as unavailable so hosts fall back to eager SDPA.
    """
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401, PLC0415
    except (ImportError, OSError):
        return False
    return True


@torch.library.custom_op("ltx_core::na_attention_eager", mutates_args=())
def na_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
) -> torch.Tensor:
    """Opaque eager tiled-SDPA NA so Dynamo does not graph-break into the Python loop."""
    return eager_na3d(q, k, v, kernel_size=kernel_size, is_causal=None, scale=1.0)


@na_attention_eager.register_fake
def _na_attention_eager_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
) -> torch.Tensor:
    del k, v, kernel_size
    return torch.empty_like(q)


@torch.library.custom_op("ltx_core::na_attention_triton", mutates_args=())
def na_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
) -> torch.Tensor:
    """Opaque Triton NA launch so the surrounding block graph stays whole under compile."""
    from ltx_core.model.video_vae.transformer.fallback_na.triton_na import (  # noqa: PLC0415
        na3d as triton_na3d,
    )

    return triton_na3d(q, k, v, kernel_size=kernel_size, is_causal=None, scale=1.0)


@na_attention_triton.register_fake
def _na_attention_triton_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
) -> torch.Tensor:
    del k, v, kernel_size
    return torch.empty_like(q)


class EagerSdpaAttention:
    """Limited-workspace tiled SDPA NA (always available)."""

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        return na_attention_eager(q, k, v, list(attn.kernel_size))


class TritonNaAttention:
    """Triton flash-style NA (hosts with CUDA + a working Triton install)."""

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if not triton_na_available():
            raise ImportError(
                "Triton neighborhood attention requires CUDA and the triton package "
                "(on Windows: https://github.com/triton-lang/triton-windows)."
            )
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        return na_attention_triton(q, k, v, list(attn.kernel_size))


def warn_no_natten(*, backend: str) -> None:
    """Emit the loud no-natten banner and which fallback backend was chosen."""
    logger.warning(_NO_NATTEN_WARNING)
    logger.warning("DiffVAE NA fallback: using %s.", backend)


def fallback_na_attention() -> NAAttentionCallable:
    """Pick Triton if usable, else eager; emit the no-natten warning."""
    if triton_na_available():
        warn_no_natten(backend="Triton na3d")
        return TritonNaAttention()
    warn_no_natten(backend="eager tiled SDPA na3d")
    return EagerSdpaAttention()


__all__ = [
    "EagerSdpaAttention",
    "TritonNaAttention",
    "fallback_na_attention",
    "na_attention_eager",
    "na_attention_triton",
    "triton_na_available",
    "warn_no_natten",
]
