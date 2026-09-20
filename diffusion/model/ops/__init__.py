"""Triton-optimized operators for Sana video models."""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fused_gdn import _precompute_inv_rms, fused_bidi_merge, prepare_rope_tables
from .fused_gdn_chunkwise import fused_bidi_stateful_chunkwise_shared_phase_a


def _resolve_gdn_variant() -> str:
    """Pick the V2V GDN forward path from env vars."""
    return "chunkwise" if os.environ.get("USE_CHUNKWISE_GDN", "1") == "1" else "pytorch"


@dataclass(frozen=True)
class _FusedGDNPrep:
    B: int
    N: int
    C: int
    T: int
    H_s: int
    W_s: int
    S: int
    H: int
    D: int
    dtype_orig: torch.dtype
    qkv: torch.Tensor
    beta_p: torch.Tensor
    decay: torch.Tensor
    k_scale: float
    q_nw: torch.Tensor
    k_nw: torch.Tensor


def _prepare_fused_gdn_inputs(self, x: torch.Tensor, HW) -> _FusedGDNPrep:
    B, N, C = x.shape
    T, H_s, W_s = HW
    S = H_s * W_s
    H, D = self.heads, self.dim

    q_w = self.q.weight.squeeze(-1)
    k_w = self.k.weight.squeeze(-1)
    v_w = self.v.weight
    qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
    qkv = F.linear(x, qkv_w).reshape(B, N, 3, H, D)

    beta, decay = self._compute_frame_gates(x, HW)
    beta_p = beta.permute(0, 3, 1, 2).contiguous()
    k_scale = (D**-0.5) * (S**-0.5)

    if not isinstance(self.q_norm, nn.Identity):
        q_nw = self.q_norm.weight.float()
        k_nw = self.k_norm.weight.float()
    else:
        q_nw = torch.ones(C, device=x.device, dtype=torch.float32)
        k_nw = torch.ones(C, device=x.device, dtype=torch.float32)

    return _FusedGDNPrep(
        B=B,
        N=N,
        C=C,
        T=T,
        H_s=H_s,
        W_s=W_s,
        S=S,
        H=H,
        D=D,
        dtype_orig=x.dtype,
        qkv=qkv,
        beta_p=beta_p,
        decay=decay,
        k_scale=k_scale,
        q_nw=q_nw,
        k_nw=k_nw,
    )


__all__ = [
    "_precompute_inv_rms",
    "_prepare_fused_gdn_inputs",
    "_resolve_gdn_variant",
    "fused_bidi_merge",
    "fused_bidi_stateful_chunkwise_shared_phase_a",
    "prepare_rope_tables",
]
