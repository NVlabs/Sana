"""Liger-Kernel RMSNorm wrapper for Sana video training.

Provides a drop-in replacement for ``diffusion.model.norms.RMSNorm`` using
Liger-Kernel's Triton-fused implementation.  Enabled by default when
liger-kernel is installed.  Disable via ``SANA_USE_LIGER=0``.

The Triton kernel fuses cast → square → mean → rsqrt → scale into a single
GPU pass, eliminating ~5-6 intermediate tensor allocations per call.
"""

import logging
import os
from typing import Type

import torch.nn as nn

logger = logging.getLogger(__name__)


def get_rmsnorm_class() -> Type[nn.Module]:
    """Return the RMSNorm class to use, controlled by ``SANA_USE_LIGER`` env var.

    Liger is **on by default** when the package is installed.
    Set ``SANA_USE_LIGER=0`` to force the original PyTorch implementation.

    Returns:
        ``SanaLigerRMSNorm`` when liger-kernel is available (and not disabled),
        otherwise the original ``diffusion.model.norms.RMSNorm``.
    """
    # Explicitly disabled
    if os.environ.get("SANA_USE_LIGER", "") in ("0", "false", "False"):
        from diffusion.model.norms import RMSNorm

        return RMSNorm

    # Auto-detect: try importing liger-kernel
    try:
        from liger_kernel.transformers import LigerRMSNorm
    except ImportError:
        from diffusion.model.norms import RMSNorm

        return RMSNorm

    class SanaLigerRMSNorm(LigerRMSNorm):
        """Drop-in replacement matching ``RMSNorm(dim, scale_factor, eps, norm_dim)`` signature.

        Uses ``casting_mode="gemma"`` so that both the normalisation *and*
        the weight multiplication happen in fp32 before casting back — this
        matches the original Sana RMSNorm behaviour exactly.
        """

        def __init__(
            self,
            dim: int,
            scale_factor: float = 1.0,
            eps: float = 1e-6,
            norm_dim: int = -1,
        ) -> None:
            if norm_dim != -1:
                raise ValueError(f"LigerRMSNorm only supports norm_dim=-1, got {norm_dim}")
            if scale_factor != 1.0:
                raise ValueError(f"LigerRMSNorm requires scale_factor=1.0, got {scale_factor}")
            super().__init__(
                hidden_size=dim,
                eps=eps,
                offset=0.0,
                casting_mode="gemma",
                init_fn="ones",
                in_place=True,
            )

    logger.info("Using Liger-Kernel RMSNorm (Triton-fused, casting_mode=gemma)")
    return SanaLigerRMSNorm
