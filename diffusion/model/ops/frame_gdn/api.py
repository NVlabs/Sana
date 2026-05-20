# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Public API for Triton-accelerated frame-wise GDN.

Exposes two drop-in replacements for the PyTorch GDN update-rule functions:

* ``triton_chunk_sana_gdn``    -- chunk-parallel (training, supports backward)
* ``triton_recurrent_sana_gdn`` -- fused recurrent (inference, forward only)

Both accept the same signature as ``torch_recurrent_sana_gdn`` /
``torch_chunk_sana_gdn`` in ``sana_gdn_blocks.py``.
"""

from __future__ import annotations

import os

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ---------------------------------------------------------------------------
# Compiled output projection (for CP path and standalone use)
# ---------------------------------------------------------------------------


@torch.compile(disable=os.environ.get("CP_DISABLE_COMPILE", "0") not in ("0", "false"))
def compiled_gdn_output_projection(
    S_kv_all: torch.Tensor,
    S_z_all: torch.Tensor,
    q_rot_f: torch.Tensor,
    q_f: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled output projection for GDN scan states.

    AOTAutograd jointly optimizes the forward+backward, applying min-cut
    tensor saving and operator fusion.  This reduces peak backward memory
    compared to eager autograd.

    Args:
        S_kv_all: Scan KV states, ``(B, H, T, D, D)``.
        S_z_all:  Scan Z states,  ``(B, H, T, D)``.
        q_rot_f:  Rotary-embedded queries, ``(B, H, T, D, S)``.
        q_f:      Queries (unrotated), ``(B, H, T, D, S)``.
        eps:      Stability constant.

    Returns:
        out_num: ``(B, H, T, D, S)`` -- numerator.
        out_den: ``(B, H, T, 1, S)`` -- denominator.
    """
    out_num = torch.matmul(S_kv_all, q_rot_f)
    out_den = torch.matmul(S_z_all.unsqueeze(-2), q_f)
    return out_num, out_den


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------


def _build_transition_matrices(
    k_f: torch.Tensor,
    v_f: torch.Tensor,
    k_rot_f: torch.Tensor,
    beta_f: torch.Tensor,
    decay_f: torch.Tensor,
    I: torch.Tensor,
    BH: int,
    T: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the D x D transition and input matrices for the state scan.

    Args:
        k_f:      Frame-reshaped keys,         ``(B, H, T, D, S)``.
        v_f:      Frame-reshaped values,        ``(B, H, T, D, S)``.
        k_rot_f:  Frame-reshaped rotated keys,  ``(B, H, T, D, S)``.
        beta_f:   Update gate, ``(B, H, T, 1, 1)`` or ``(B, H, T, 1, S)``.
        decay_f:  Decay gate,  ``(B, H, T, 1, 1)``.
        I:        Identity matrix, ``(1, 1, 1, D, D)``.
        BH:       ``B * H``.
        T:        Number of frames.
        D:        Head dimension.

    Returns:
        W_kv, U_kv: ``(B*H, T, D, D)``  -- KV transition / input matrices.
        W_z,  U_z:  ``(B*H, T, D, D)`` and ``(B*H, T, D)`` -- Z matrices.
    """
    k_rot_beta = k_rot_f * beta_f
    W_kv = decay_f * (I - torch.matmul(k_rot_beta, k_rot_f.transpose(-1, -2)))
    U_kv = torch.matmul(v_f * beta_f, k_rot_f.transpose(-1, -2))

    k_beta = k_f * beta_f
    W_z = decay_f * (I - torch.matmul(k_beta, k_f.transpose(-1, -2)))
    U_z = k_beta.sum(dim=-1)

    return (
        W_kv.reshape(BH, T, D, D).contiguous(),
        U_kv.reshape(BH, T, D, D).contiguous(),
        W_z.reshape(BH, T, D, D).contiguous(),
        U_z.reshape(BH, T, D).contiguous(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.compiler.disable
def triton_chunk_sana_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    recall_gate: torch.Tensor | None = None,
    chunk_size: int | None = 21,
    chunk_index: list[int] | None = None,
    chunk_split_strategy: str = "uniform",
    eps: float = 1e-6,
    return_components: bool = False,
    proj_dtype: torch.dtype | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated chunk-parallel frame-wise GDN (training).

    Same signature as ``torch_chunk_sana_gdn``.  The ``chunk_size`` /
    ``chunk_index`` parameters are accepted for API compatibility but are
    **ignored** -- the Triton scan processes all T frames in a single fused
    kernel, which is faster than chunked PyTorch loops for any chunk size.

    Architecture (two-phase pipeline):

    Phase 1 -- Preprocessing (gradient-checkpointed PyTorch ops):
        Build D x D transition / input matrices from k, v, beta, decay.
        Wrapped in ``torch.utils.checkpoint`` so the large (B,H,T,D,S)
        intermediates are freed after the forward and recomputed cheaply
        during backward (the D x D outputs are tiny).
    Phase 2+3 -- Fused scan + output projection
        (``FrameGDNScanAndProject`` autograd.Function):
        Triton forward kernel computes all states, then projects against
        q / q_rot to produce num / den.  Fusing eliminates intermediate
        autograd nodes from the output matmuls, avoids double-referencing
        S_kv_all across autograd boundaries, and provides explicit tensor
        lifetime control during backward -- reducing peak backward memory.

    Gradients flow through both phases:
        output -> Phase 2+3 backward (matmul grads + Triton scan bwd)
        -> dW/dU -> Phase 1 backward (recomputed) -> dk, dv, dbeta, ddecay.
    """
    del recall_gate, chunk_size, chunk_index, chunk_split_strategy

    from diffusion.model.ops.frame_gdn.scan_triton import FrameGDNScanAndProject

    B, H, D, N = q.shape
    T = beta.shape[2] if beta.ndim >= 3 else beta.shape[-1]
    S = N // T
    BH = B * H

    def to_frame(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_f = to_frame(q)
    k_f, v_f = to_frame(k), to_frame(v)
    q_rot_f, k_rot_f = to_frame(q_rot), to_frame(k_rot)

    if beta.ndim == 4:
        beta_f = beta.unsqueeze(3)
    else:
        beta_f = beta.view(B, H, T, 1, 1)
    decay_f = decay.view(B, H, T, 1, 1)
    I = torch.eye(D, device=q.device, dtype=q.dtype).reshape(1, 1, 1, D, D)

    # ---- Phase 1: Preprocessing (gradient-checkpointed) ----
    W_kv, U_kv, W_z, U_z = grad_checkpoint(
        _build_transition_matrices,
        k_f,
        v_f,
        k_rot_f,
        beta_f,
        decay_f,
        I,
        BH,
        T,
        D,
        use_reentrant=False,
    )

    # ---- Phase 2+3: Fused scan + output projection ----
    # When proj_dtype is set (e.g. bfloat16), q_rot_f and q_f are downcast
    # before being saved for backward.  These tensors are only used for the
    # output projection (one matmul per frame, no temporal accumulation),
    # so reduced precision is numerically safe while halving their memory.
    if proj_dtype is not None and q_f.dtype != proj_dtype:
        q_f = q_f.to(proj_dtype)
        q_rot_f = q_rot_f.to(proj_dtype)

    out_num, out_den = FrameGDNScanAndProject.apply(
        W_kv,
        U_kv,
        W_z,
        U_z,
        q_rot_f,
        q_f,
    )

    final_num = out_num.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)
    final_den = out_den.permute(0, 1, 3, 2, 4).reshape(B, H, 1, N)

    if return_components:
        return final_num, final_den

    return final_num / (final_den + eps)


def triton_recurrent_sana_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    recall_gate: torch.Tensor | None = None,
    eps: float = 1e-6,
    return_components: bool = False,
    state_kv_in: torch.Tensor | None = None,
    state_z_in: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Triton fused recurrent frame-wise GDN (forward only, for inference).

    Same core signature as ``torch_recurrent_sana_gdn``, with additional
    ``state_kv_in / state_z_in / output_final_state`` for persistent-state
    autoregressive generation.
    """
    del recall_gate

    B, H, D, N = q.shape
    T = beta.shape[2] if beta.ndim >= 3 else beta.shape[-1]
    S = N // T

    BH = B * H
    q_flat = q.reshape(BH, D, N).contiguous()
    k_flat = k.reshape(BH, D, N).contiguous()
    v_flat = v.reshape(BH, D, N).contiguous()
    q_rot_flat = q_rot.reshape(BH, D, N).contiguous()
    k_rot_flat = k_rot.reshape(BH, D, N).contiguous()

    if beta.ndim == 4:
        beta_flat = beta.reshape(BH, T, S).contiguous()
    else:
        beta_flat = beta.reshape(BH, T).contiguous()

    decay_flat = decay.reshape(BH, T).contiguous()

    if state_kv_in is not None:
        state_kv_in = state_kv_in.reshape(BH, D, D).contiguous()
    if state_z_in is not None:
        state_z_in = state_z_in.reshape(BH, D).contiguous()

    from diffusion.model.ops.frame_gdn.fused_recurrent_triton import (
        frame_gdn_fused_recurrent_fwd,
    )

    out_num, out_den, state_kv_out, state_z_out = frame_gdn_fused_recurrent_fwd(
        q_flat,
        k_flat,
        v_flat,
        q_rot_flat,
        k_rot_flat,
        beta_flat,
        decay_flat,
        T=T,
        S=S,
        state_kv_in=state_kv_in,
        state_z_in=state_z_in,
        output_final_state=output_final_state,
    )

    final_num = out_num.view(B, H, D, N)
    final_den = out_den.view(B, H, 1, N)

    if return_components:
        return final_num, final_den

    return final_num / (final_den + eps)
