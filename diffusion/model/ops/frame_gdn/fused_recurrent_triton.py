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

"""Triton fused recurrent kernel for frame-wise GDN (forward only).

Processes all T frames sequentially inside a single kernel instance, keeping
the D x D ``state_kv`` and D x 1 ``state_z`` entirely in registers.  For each
frame two passes over spatial blocks are performed:

  1. **Delta pass** -- compute ``v_pred``, ``delta_v``, accumulate state update.
  2. **Output pass** -- project the updated state to produce ``out_num / out_den``.

The kernel supports persistent state (initial state in, final state out) for
autoregressive inference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["state_kv_in"] is not None,
        "STORE_FINAL_STATE": lambda args: args["state_kv_out"] is not None,
        "IS_BETA_SPATIAL": lambda args: args["IS_BETA_SPATIAL"],
    }
)
@triton.jit(do_not_specialize=["T", "S"])
def frame_gdn_fused_recurrent_fwd_kernel(
    # Inputs:  (B, H, D, T*S) layout -- last dim contiguous
    q_ptr,
    k_ptr,
    v_ptr,
    q_rot_ptr,
    k_rot_ptr,
    # Gates
    beta_ptr,  # (B, H, T) or (B, H, T, S)
    decay_ptr,  # (B, H, T)
    # Persistent state I/O (optional)
    state_kv_in,  # (B, H, D, D) or None
    state_z_in,  # (B, H, D) or None
    state_kv_out,  # (B, H, D, D) or None
    state_z_out,  # (B, H, D) or None
    # Outputs:  (B, H, D, T*S) and (B, H, 1, T*S)
    out_num_ptr,
    out_den_ptr,
    # Dimensions
    T,
    S,
    D: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    # Heuristic flags
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_BETA_SPATIAL: tl.constexpr,
):
    i_bh = tl.program_id(0)
    N = T * S  # total tokens

    o_d = tl.arange(0, BD)
    o_s = tl.arange(0, BS)
    mask_d = o_d < D
    mask_dd = mask_d[:, None] & mask_d[None, :]

    # --- Load / initialize states ---
    state_kv = tl.zeros([BD, BD], dtype=tl.float32)
    state_z = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        skv_base = state_kv_in + i_bh.to(tl.int64) * D * D
        state_kv = tl.load(
            skv_base + o_d[:, None] * D + o_d[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        sz_base = state_z_in + i_bh.to(tl.int64) * D
        state_z = tl.load(sz_base + o_d, mask=mask_d, other=0.0).to(tl.float32)

    # Strides for (B, H, D, N) layout -- N is the last (contiguous) dim.
    # Element [bh, d, n] at offset  bh * D * N + d * N + n
    stride_bh_dn = D * N
    base_dn = i_bh.to(tl.int64) * stride_bh_dn  # start for this (b, h)

    # Strides for beta: (B*H, T) or (B*H, T, S)
    base_beta = i_bh.to(tl.int64) * (T * S if IS_BETA_SPATIAL else T)
    # Strides for decay: (B*H, T)
    base_decay = i_bh.to(tl.int64) * T

    for t in range(0, T):
        frame_start = t * S  # first spatial index of this frame

        # --- Decay ---
        gt = tl.load(decay_ptr + base_decay + t).to(tl.float32)
        state_kv = state_kv * gt
        state_z = state_z * gt

        # --- Load scalar beta (shared across spatial) ---
        if not IS_BETA_SPATIAL:
            beta_scalar = tl.load(beta_ptr + base_beta + t).to(tl.float32)

        # --- Pass 1: delta computation + state accumulation ---
        accum_kv = tl.zeros([BD, BD], dtype=tl.float32)
        accum_z = tl.zeros([BD], dtype=tl.float32)

        for s_start in range(0, S, BS):
            s_idx = s_start + o_s
            mask_s = s_idx < S
            n_idx = frame_start + s_idx  # global token index

            # Pointer pattern: base_dn + d * N + n
            p_krot = k_rot_ptr + base_dn + o_d[:, None] * N + n_idx[None, :]
            p_k = k_ptr + base_dn + o_d[:, None] * N + n_idx[None, :]
            p_v = v_ptr + base_dn + o_d[:, None] * N + n_idx[None, :]
            mask_ds = mask_d[:, None] & mask_s[None, :]

            krot_s = tl.load(p_krot, mask=mask_ds, other=0.0).to(tl.float32)  # (BD, BS)
            k_s = tl.load(p_k, mask=mask_ds, other=0.0).to(tl.float32)
            v_s = tl.load(p_v, mask=mask_ds, other=0.0).to(tl.float32)

            # v_pred = state_kv @ krot_s  -->  (BD, BD) @ (BD, BS)
            v_pred = tl.dot(state_kv, krot_s, allow_tf32=False)

            if IS_BETA_SPATIAL:
                p_beta_s = beta_ptr + base_beta + t * S + s_idx
                beta_s = tl.load(p_beta_s, mask=mask_s, other=0.0).to(tl.float32)  # (BS,)
                delta_v = (v_s - v_pred) * beta_s[None, :]
            else:
                delta_v = (v_s - v_pred) * beta_scalar

            # accum_kv += delta_v @ krot_s^T -->  (BD, BS) @ (BS, BD)
            accum_kv += tl.dot(delta_v, tl.trans(krot_s), allow_tf32=False)

            # Z stream
            # z_pred = state_z^T @ k_s  -->  (1, BD) @ (BD, BS) = (1, BS)
            z_pred = tl.sum(state_z[:, None] * k_s, axis=0)  # (BS,)
            if IS_BETA_SPATIAL:
                delta_z = (1.0 - z_pred) * beta_s
            else:
                delta_z = (1.0 - z_pred) * beta_scalar

            # accum_z += k_s @ delta_z^T  sum over S  -->  (BD,)
            accum_z += tl.sum(k_s * delta_z[None, :], axis=1)

        # Apply accumulated state update
        state_kv = state_kv + accum_kv
        state_z = state_z + accum_z

        # --- Pass 2: output computation ---
        for s_start in range(0, S, BS):
            s_idx = s_start + o_s
            mask_s = s_idx < S
            n_idx = frame_start + s_idx
            mask_ds = mask_d[:, None] & mask_s[None, :]

            p_qrot = q_rot_ptr + base_dn + o_d[:, None] * N + n_idx[None, :]
            p_q = q_ptr + base_dn + o_d[:, None] * N + n_idx[None, :]
            qrot_s = tl.load(p_qrot, mask=mask_ds, other=0.0).to(tl.float32)
            q_s = tl.load(p_q, mask=mask_ds, other=0.0).to(tl.float32)

            # out_num = state_kv @ qrot_s  -->  (BD, BD) @ (BD, BS)
            num_s = tl.dot(state_kv, qrot_s, allow_tf32=False)
            # out_den = state_z^T @ q_s  -->  (1, BS)
            den_s = tl.sum(state_z[:, None] * q_s, axis=0)  # (BS,)

            p_num = out_num_ptr + base_dn + o_d[:, None] * N + n_idx[None, :]
            tl.store(p_num, num_s.to(p_num.dtype.element_ty), mask=mask_ds)

            p_den = out_den_ptr + i_bh.to(tl.int64) * N + n_idx
            tl.store(p_den, den_s.to(p_den.dtype.element_ty), mask=mask_s)

    # --- Store final state ---
    if STORE_FINAL_STATE:
        skv_out = state_kv_out + i_bh.to(tl.int64) * D * D
        tl.store(
            skv_out + o_d[:, None] * D + o_d[None, :],
            state_kv.to(skv_out.dtype.element_ty),
            mask=mask_dd,
        )
        sz_out = state_z_out + i_bh.to(tl.int64) * D
        tl.store(sz_out + o_d, state_z.to(sz_out.dtype.element_ty), mask=mask_d)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def _select_bs(S: int) -> int:
    """Choose spatial block size for tiling."""
    if S <= 32:
        return max(16, triton.next_power_of_2(S))
    if S <= 64:
        return 32
    return 64


def frame_gdn_fused_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    T: int,
    S: int,
    state_kv_in: torch.Tensor | None = None,
    state_z_in: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Launch the fused recurrent forward kernel.

    Args:
        q, k, v, q_rot, k_rot: ``(B*H, D, T*S)`` contiguous float32.
        beta: ``(B*H, T)`` or ``(B*H, T, S)`` contiguous float32.
        decay: ``(B*H, T)`` contiguous float32.
        T: Number of frames.
        S: Spatial tokens per frame.
        state_kv_in: Optional initial KV state ``(B*H, D, D)``.
        state_z_in:  Optional initial Z state  ``(B*H, D)``.
        output_final_state: Whether to return the final states.

    Returns:
        out_num: ``(B*H, D, T*S)``
        out_den: ``(B*H, T*S)``
        state_kv_out: ``(B*H, D, D)`` if *output_final_state* else ``None``.
        state_z_out:  ``(B*H, D)``   if *output_final_state* else ``None``.
    """
    BH, D, N = q.shape
    assert N == T * S

    from diffusion.model.ops.frame_gdn.scan_triton import _select_bd

    BD = _select_bd(D)
    BS = _select_bs(S)
    is_beta_spatial = beta.ndim == 3

    out_num = torch.empty_like(q)
    out_den = q.new_empty(BH, N)

    state_kv_out = q.new_empty(BH, D, D, dtype=torch.float32) if output_final_state else None
    state_z_out = q.new_empty(BH, D, dtype=torch.float32) if output_final_state else None

    grid = (BH,)
    frame_gdn_fused_recurrent_fwd_kernel[grid](
        q,
        k,
        v,
        q_rot,
        k_rot,
        beta,
        decay,
        state_kv_in,
        state_z_in,
        state_kv_out,
        state_z_out,
        out_num,
        out_den,
        T=T,
        S=S,
        D=D,
        BD=BD,
        BS=BS,
        IS_BETA_SPATIAL=is_beta_spatial,
        num_warps=4,
        num_stages=1,
    )
    return out_num, out_den, state_kv_out, state_z_out
