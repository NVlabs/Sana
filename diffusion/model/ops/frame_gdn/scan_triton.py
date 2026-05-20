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

"""Triton kernels for the D x D state scan in frame-wise GDN.

Forward scan:
    S_kv[t] = S_kv[t-1] @ W_kv[t] + U_kv[t]     (KV state, D x D)
    S_z[t]  = W_z[t] @ S_z[t-1] + U_z[t]          (Z state,  D x 1)

Backward scan (reverse):
    ds_kv[t] = dS_kv_all[t] + ds_kv[t+1] @ W_kv[t+1]^T
    dW_kv[t] = S_kv[t-1]^T @ ds_kv[t]
    dU_kv[t] = ds_kv[t]
    (analogous for Z state with left-multiply convention)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def frame_gdn_scan_fwd_kernel(
    W_kv_ptr,
    U_kv_ptr,
    W_z_ptr,
    U_z_ptr,
    S_kv_all_ptr,
    S_z_all_ptr,
    H: tl.constexpr,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    """Scan forward: one program per (batch, head) pair."""
    i_bh = tl.program_id(0)

    stride_dd = D * D
    base_dd = i_bh.to(tl.int64) * T * stride_dd
    base_d = i_bh.to(tl.int64) * T * D

    o_d = tl.arange(0, BD)
    mask_d = o_d < D
    mask_dd = mask_d[:, None] & mask_d[None, :]

    state_kv = tl.zeros([BD, BD], dtype=tl.float32)
    state_z = tl.zeros([BD], dtype=tl.float32)

    for t in range(0, T):
        t_off_dd = base_dd + t * stride_dd
        t_off_d = base_d + t * D

        # --- KV state: S_kv = S_kv @ W_kv[t] + U_kv[t] ---
        p_W = W_kv_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        p_U = U_kv_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        W_t = tl.load(p_W, mask=mask_dd, other=0.0).to(tl.float32)
        U_t = tl.load(p_U, mask=mask_dd, other=0.0).to(tl.float32)

        state_kv = tl.dot(state_kv, W_t, allow_tf32=False) + U_t

        p_S = S_kv_all_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        tl.store(p_S, state_kv.to(p_S.dtype.element_ty), mask=mask_dd)

        # --- Z state: S_z = W_z[t] @ S_z + U_z[t] ---
        p_Wz = W_z_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        Wz_t = tl.load(p_Wz, mask=mask_dd, other=0.0).to(tl.float32)
        p_Uz = U_z_ptr + t_off_d + o_d
        Uz_t = tl.load(p_Uz, mask=mask_d, other=0.0).to(tl.float32)

        # Matrix-vector: result[i] = sum_j Wz[i,j] * sz[j]
        state_z = tl.sum(Wz_t * state_z[None, :], axis=1) + Uz_t

        p_Sz = S_z_all_ptr + t_off_d + o_d
        tl.store(p_Sz, state_z.to(p_Sz.dtype.element_ty), mask=mask_d)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def frame_gdn_scan_bwd_kernel(
    # Saved from forward
    W_kv_ptr,
    S_kv_all_ptr,
    W_z_ptr,
    S_z_all_ptr,
    # Upstream gradients
    dS_kv_all_ptr,
    dS_z_all_ptr,
    # Output gradients
    dW_kv_ptr,
    dU_kv_ptr,
    dW_z_ptr,
    dU_z_ptr,
    H: tl.constexpr,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    """Scan backward (reverse): one program per (batch, head) pair."""
    i_bh = tl.program_id(0)

    stride_dd = D * D
    base_dd = i_bh.to(tl.int64) * T * stride_dd
    base_d = i_bh.to(tl.int64) * T * D

    o_d = tl.arange(0, BD)
    mask_d = o_d < D
    mask_dd = mask_d[:, None] & mask_d[None, :]

    ds_kv = tl.zeros([BD, BD], dtype=tl.float32)
    ds_z = tl.zeros([BD], dtype=tl.float32)

    for t_idx in range(0, T):
        t = T - 1 - t_idx
        t_off_dd = base_dd + t * stride_dd
        t_off_d = base_d + t * D

        # --- KV backward ---
        # Accumulate: ds_kv += dS_kv_all[t]
        p_dS = dS_kv_all_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        dS_t = tl.load(p_dS, mask=mask_dd, other=0.0).to(tl.float32)
        ds_kv = ds_kv + dS_t

        # dU_kv[t] = ds_kv
        p_dU = dU_kv_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        tl.store(p_dU, ds_kv.to(p_dU.dtype.element_ty), mask=mask_dd)

        # dW_kv[t] = S_kv[t-1]^T @ ds_kv   (zero when t == 0)
        prev_off_dd = base_dd + tl.maximum(t - 1, 0) * stride_dd
        p_Sp = S_kv_all_ptr + prev_off_dd + o_d[:, None] * D + o_d[None, :]
        s_prev = tl.load(p_Sp, mask=mask_dd & (t > 0), other=0.0).to(tl.float32)
        dW = tl.dot(tl.trans(s_prev), ds_kv, allow_tf32=False)
        p_dW = dW_kv_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        tl.store(p_dW, dW.to(p_dW.dtype.element_ty), mask=mask_dd)

        # Propagate: ds_kv = ds_kv @ W_kv[t]^T
        p_W = W_kv_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        W_t = tl.load(p_W, mask=mask_dd, other=0.0).to(tl.float32)
        ds_kv = tl.dot(ds_kv, tl.trans(W_t), allow_tf32=False)

        # --- Z backward ---
        # Accumulate: ds_z += dS_z_all[t]
        p_dSz = dS_z_all_ptr + t_off_d + o_d
        dSz_t = tl.load(p_dSz, mask=mask_d, other=0.0).to(tl.float32)
        ds_z = ds_z + dSz_t

        # dU_z[t] = ds_z
        p_dUz = dU_z_ptr + t_off_d + o_d
        tl.store(p_dUz, ds_z.to(p_dUz.dtype.element_ty), mask=mask_d)

        # dW_z[t] = ds_z @ S_z[t-1]^T   (outer product, zero when t == 0)
        prev_off_d = base_d + tl.maximum(t - 1, 0) * D
        p_SzP = S_z_all_ptr + prev_off_d + o_d
        sz_prev = tl.load(p_SzP, mask=mask_d & (t > 0), other=0.0).to(tl.float32)
        dWz = ds_z[:, None] * sz_prev[None, :]  # outer product (BD, BD)
        p_dWz = dW_z_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        tl.store(p_dWz, dWz.to(p_dWz.dtype.element_ty), mask=mask_dd)

        # Propagate: ds_z = W_z[t]^T @ ds_z
        p_Wz = W_z_ptr + t_off_dd + o_d[:, None] * D + o_d[None, :]
        Wz_t = tl.load(p_Wz, mask=mask_dd, other=0.0).to(tl.float32)
        # result[i] = sum_j W_z[j,i] * ds_z[j]
        ds_z = tl.sum(Wz_t * ds_z[:, None], axis=0)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _select_bd(D: int) -> int:
    bd = max(16, triton.next_power_of_2(D))
    if bd > 128:
        raise ValueError(
            f"Head dim D={D} (BD={bd}) exceeds the max supported size 128. " "Fall back to the PyTorch implementation."
        )
    return bd


def _select_num_warps(BD: int) -> int:
    if BD <= 16:
        return 1
    if BD <= 32:
        return 2
    return 4


def frame_gdn_scan_fwd(
    W_kv: torch.Tensor,
    U_kv: torch.Tensor,
    W_z: torch.Tensor,
    U_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the forward scan producing all intermediate states.

    Args:
        W_kv: Transition matrices, shape ``(B*H, T, D, D)``.
        U_kv: Input matrices,     shape ``(B*H, T, D, D)``.
        W_z:  Z transition,       shape ``(B*H, T, D, D)``.
        U_z:  Z input,            shape ``(B*H, T, D)``.

    Returns:
        S_kv_all: ``(B*H, T, D, D)`` -- all intermediate KV states.
        S_z_all:  ``(B*H, T, D)``    -- all intermediate Z states.
    """
    BH, T, D, _ = W_kv.shape
    BD = _select_bd(D)
    H = 1  # BH already flat

    S_kv_all = torch.empty_like(W_kv)
    S_z_all = torch.empty_like(U_z)

    grid = (BH,)
    frame_gdn_scan_fwd_kernel[grid](
        W_kv,
        U_kv,
        W_z,
        U_z,
        S_kv_all,
        S_z_all,
        H=H,
        T=T,
        D=D,
        BD=BD,
        num_warps=_select_num_warps(BD),
        num_stages=1,
    )
    return S_kv_all, S_z_all


def frame_gdn_scan_bwd(
    W_kv: torch.Tensor,
    S_kv_all: torch.Tensor,
    dS_kv_all: torch.Tensor,
    W_z: torch.Tensor,
    S_z_all: torch.Tensor,
    dS_z_all: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the backward scan computing gradients for W and U.

    Returns:
        dW_kv, dU_kv: ``(B*H, T, D, D)`` each.
        dW_z, dU_z:   ``(B*H, T, D, D)`` and ``(B*H, T, D)``.
    """
    BH, T, D, _ = W_kv.shape
    BD = _select_bd(D)
    H = 1

    dW_kv = torch.empty_like(W_kv)
    dU_kv = torch.empty_like(W_kv)
    dW_z = torch.empty_like(W_z)
    dU_z = torch.empty_like(S_z_all)

    grid = (BH,)
    frame_gdn_scan_bwd_kernel[grid](
        W_kv,
        S_kv_all,
        W_z,
        S_z_all,
        dS_kv_all,
        dS_z_all,
        dW_kv,
        dU_kv,
        dW_z,
        dU_z,
        H=H,
        T=T,
        D=D,
        BD=BD,
        num_warps=_select_num_warps(BD),
        num_stages=1,
    )
    return dW_kv, dU_kv, dW_z, dU_z


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------


class FrameGDNScan(torch.autograd.Function):
    """Differentiable wrapper around the Triton forward/backward scan kernels.

    Saves only the transition matrices and computed states for the backward
    pass -- the D x D tensors are tiny relative to the full q/k/v inputs.
    """

    @staticmethod
    def forward(
        ctx,
        W_kv: torch.Tensor,
        U_kv: torch.Tensor,
        W_z: torch.Tensor,
        U_z: torch.Tensor,
        S_init_kv: torch.Tensor | None = None,
        S_init_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if S_init_kv is not None or S_init_z is not None:
            raise NotImplementedError("Triton scan with S_init is not implemented yet. Use torch backend.")

        S_kv_all, S_z_all = frame_gdn_scan_fwd(
            W_kv.detach(),
            U_kv.detach(),
            W_z.detach(),
            U_z.detach(),
        )
        ctx.save_for_backward(W_kv, S_kv_all, W_z, S_z_all)
        return S_kv_all, S_z_all

    @staticmethod
    def backward(
        ctx,
        dS_kv_all: torch.Tensor,
        dS_z_all: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        W_kv, S_kv_all, W_z, S_z_all = ctx.saved_tensors
        dW_kv, dU_kv, dW_z, dU_z = frame_gdn_scan_bwd(
            W_kv,
            S_kv_all,
            dS_kv_all.contiguous(),
            W_z,
            S_z_all,
            dS_z_all.contiguous(),
        )
        return dW_kv, dU_kv, dW_z, dU_z


class FrameGDNScanAndProject(torch.autograd.Function):
    """Fused scan + output projection to reduce peak backward memory.

    Combines Phase 2 (Triton scan) and Phase 3 (output projection) into a
    single autograd Function.  This eliminates the intermediate autograd
    graph nodes for the matmul ops, avoids the double-reference of
    ``S_kv_all`` across two autograd nodes, and gives explicit control
    over tensor lifetimes during backward.

    Inputs:
        W_kv, U_kv: ``(B*H, T, D, D)`` -- transition / input matrices.
        W_z,  U_z:  ``(B*H, T, D, D)`` and ``(B*H, T, D)`` -- Z matrices.
        q_rot_f:    ``(B, H, T, D, S)`` -- rotary-embedded queries.
        q_f:        ``(B, H, T, D, S)`` -- queries (unrotated, for Z).

    Outputs:
        out_num: ``(B, H, T, D, S)`` -- numerator of the attention output.
        out_den: ``(B, H, T, 1, S)`` -- denominator of the attention output.
    """

    @staticmethod
    def forward(
        ctx,
        W_kv: torch.Tensor,
        U_kv: torch.Tensor,
        W_z: torch.Tensor,
        U_z: torch.Tensor,
        q_rot_f: torch.Tensor,
        q_f: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H = q_rot_f.shape[:2]

        S_kv_all, S_z_all = frame_gdn_scan_fwd(
            W_kv.detach(),
            U_kv.detach(),
            W_z.detach(),
            U_z.detach(),
        )

        S_kv_view = S_kv_all.view(B, H, -1, S_kv_all.shape[-2], S_kv_all.shape[-1])
        S_z_view = S_z_all.view(B, H, -1, S_z_all.shape[-1])

        # q_rot_f / q_f may be BF16 while S_kv/S_z are FP32.  Upcast
        # queries temporarily for the matmul; the BF16 originals are kept
        # in save_for_backward to halve their memory footprint.
        scan_dtype = S_kv_all.dtype
        out_num = torch.matmul(S_kv_view, q_rot_f.to(scan_dtype))
        out_den = torch.matmul(S_z_view.unsqueeze(-2), q_f.to(scan_dtype))

        ctx.save_for_backward(W_kv, S_kv_all, W_z, S_z_all, q_rot_f, q_f)
        ctx.shape_BH = (B, H)
        ctx.q_dtype = q_rot_f.dtype

        return out_num, out_den

    @staticmethod
    def backward(
        ctx,
        d_out_num: torch.Tensor,
        d_out_den: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        W_kv, S_kv_all, W_z, S_z_all, q_rot_f, q_f = ctx.saved_tensors
        B, H = ctx.shape_BH
        q_dtype = ctx.q_dtype

        scan_dtype = S_kv_all.dtype
        S_kv_view = S_kv_all.view(B, H, -1, S_kv_all.shape[-2], S_kv_all.shape[-1])
        S_z_view = S_z_all.view(B, H, -1, S_z_all.shape[-1])

        # Upstream gradients may arrive in BF16 (e.g. from FSDP mixed
        # precision).  Upcast to match the scan state dtype (FP32) so all
        # matmuls in this backward are numerically stable.
        if d_out_num.dtype != scan_dtype:
            d_out_num = d_out_num.to(scan_dtype)
            d_out_den = d_out_den.to(scan_dtype)

        # -- Gradients for q_rot_f and q_f (output projection backward) --
        # out_num = S_kv @ q_rot_f  =>  dq_rot_f = S_kv^T @ d_out_num
        dq_rot_f = torch.matmul(S_kv_view.transpose(-1, -2), d_out_num)
        # out_den = S_z.unsqueeze(-2) @ q_f  =>  dq_f = S_z.unsqueeze(-1) @ d_out_den
        dq_f = torch.matmul(S_z_view.unsqueeze(-1), d_out_den)

        # Match gradient dtype to the (possibly lower-precision) input dtype.
        if dq_rot_f.dtype != q_dtype:
            dq_rot_f = dq_rot_f.to(q_dtype)
            dq_f = dq_f.to(q_dtype)

        # -- Gradients for S_kv_all and S_z_all --
        # Upcast BF16 queries to FP32 for the matmul; result feeds the
        # Triton backward scan which operates in FP32.
        q_rot_f_fp = q_rot_f if q_rot_f.dtype == scan_dtype else q_rot_f.to(scan_dtype)
        q_f_fp = q_f if q_f.dtype == scan_dtype else q_f.to(scan_dtype)

        dS_kv_all = torch.matmul(d_out_num, q_rot_f_fp.transpose(-1, -2))
        dS_z_all = torch.matmul(d_out_den, q_f_fp.transpose(-1, -2)).squeeze(-2)

        # Free query tensors and views no longer needed.
        del d_out_num, d_out_den, q_rot_f, q_f, S_kv_view, S_z_view

        # Reshape for Triton backward (ensure FP32 for scan stability).
        BH = B * H
        dS_kv_flat = dS_kv_all.reshape(BH, -1, dS_kv_all.shape[-2], dS_kv_all.shape[-1])
        dS_z_flat = dS_z_all.reshape(BH, -1, dS_z_all.shape[-1])
        if dS_kv_flat.dtype != W_kv.dtype:
            dS_kv_flat = dS_kv_flat.to(W_kv.dtype)
            dS_z_flat = dS_z_flat.to(W_kv.dtype)
        dS_kv_flat = dS_kv_flat.contiguous()
        dS_z_flat = dS_z_flat.contiguous()
        del dS_kv_all, dS_z_all

        # -- Triton backward scan --
        dW_kv, dU_kv, dW_z, dU_z = frame_gdn_scan_bwd(
            W_kv,
            S_kv_all,
            dS_kv_flat,
            W_z,
            S_z_all,
            dS_z_flat,
        )

        return dW_kv, dU_kv, dW_z, dU_z, dq_rot_f, dq_f
