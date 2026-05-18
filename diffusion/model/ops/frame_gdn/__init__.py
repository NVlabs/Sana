"""Triton-optimized frame-wise Gated Delta Net (GDN) kernels.

Provides drop-in replacements for the PyTorch GDN functions:
  - ``triton_chunk_sana_gdn``   : chunk-parallel with Triton scan (training, supports backward)
  - ``triton_recurrent_sana_gdn``: fused recurrent with Triton (inference, forward-only)

Triton is imported lazily so the package can be loaded on CPU-only nodes
(the kernels will fail at call time if no GPU is present).
"""

from diffusion.model.ops.frame_gdn.api import (
    triton_chunk_sana_gdn,
    triton_recurrent_sana_gdn,
)

__all__ = [
    "triton_chunk_sana_gdn",
    "triton_recurrent_sana_gdn",
]
