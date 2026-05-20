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

"""Chunkwise backward kernels + autograd wrapper for BiGDN.

This module provides:
  - phase_c_bwd:  Phase C̄ Triton kernel (per-frame parallel dQ + dM_C)
  - phase_b_bidi_bwd: Phase B̄ serial reverse scan; Triton kernel on Blackwell-DC,
    PyTorch fallback elsewhere (A100 cuBLAS bmm beats Triton on small (D,D) matmuls)
  - phase_a_kv_bwd: Phase Ā KV Triton kernel (per-frame parallel dK, dV, dβ from dA, dP)
  - phase_a_z_bwd:  Phase Ā Z Triton kernel (per-frame parallel dK, dβ from dB_z, dP_z)
  - FusedBiGDNChunkwiseFunction: autograd Function combining the existing chunkwise
    forward kernels with the new chunkwise backward kernels.

The backward math is non-causal within a frame (matches production forward semantics).

Derivation and math are documented in I7 of fused_improve_plan.md (T12).
Validated at cos_sim ≥ 0.999 across P0/P2 at H100 layer-level bench.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from diffusion.model.ops.fused_gdn_chunkwise import (
    _arch_key,
    phase_a,
    phase_b_triton,
    phase_c,
)

# ──────────────────────────────────────────────────────────────────
# Per-arch BWD launch params. Consumer Blackwell (5090 sm_120, GB10 sm_121)
# has only ~102 KB SRAM/SM. Default BLOCK_S=64 + num_stages=2 needs
# ~114-120 KB (BLOCK_D=128 fp32 accumulator dominates: 64 KB; bf16 M_f
# 32 KB; plus per-stage Q/dO tiles). Drop to BS=16 + ns=1 there.
# (NOTE: phase_a_kv_bwd still OOMs at BS=16+ns=1 on consumer Blackwell —
# BLOCK_D × BLOCK_D bf16 dA+dP buffers alone exceed SRAM. Real fix is a
# D-tile rewrite, deferred. H100/A100/GB200 work fine.)
# ──────────────────────────────────────────────────────────────────
_BWD_LAUNCH_PARAMS: dict[str, dict] = {
    "ampere": {"BLOCK_S": 64, "phase_c_ns": 2, "phase_a_ns": 1},
    "hopper": {"BLOCK_S": 64, "phase_c_ns": 2, "phase_a_ns": 1},
    "blackwell_dc": {"BLOCK_S": 64, "phase_c_ns": 2, "phase_a_ns": 1},
    "blackwell_spark": {"BLOCK_S": 16, "phase_c_ns": 1, "phase_a_ns": 1},
}


def _resolve_bwd_params() -> dict:
    """Return arch-appropriate launch params for the bwd Triton kernels."""
    if not torch.cuda.is_available():
        return _BWD_LAUNCH_PARAMS["ampere"]
    cap = torch.cuda.get_device_capability(0)
    return _BWD_LAUNCH_PARAMS.get(_arch_key(cap), _BWD_LAUNCH_PARAMS["ampere"])


def _resolve_bwd_block_s(default: int = 64) -> int:
    """Return arch-appropriate BLOCK_S for the chunkwise bwd kernels."""
    return _resolve_bwd_params().get("BLOCK_S", default)


# ======================================================================
# Phase C̄ — backward through O_f = Q_f @ M_f  (M_f = post-frame state)
# ======================================================================
@triton.jit
def _phase_c_bwd_kernel(
    Q_ptr,
    M_ptr,
    dO_ptr,
    dQ_ptr,
    dM_C_ptr,
    B: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """One block per (b, f). Loops over S tiles, writes dQ per-tile, accumulates dM_C."""
    # Backward kernels use bf16 TC (with fp32 accumulate) — enough precision for gradients
    # while avoiding the 3× Markidis fp32 IEEE penalty that dominates at P0.
    # cos_sim bar is 0.999; measured cos_dx stays at 0.999+.
    dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "tf32"

    pid = tl.program_id(0)
    b = pid // F
    f = pid % F

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    M_bf = M_ptr + (b * F + f) * BLOCK_D * BLOCK_D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]
    M_f = tl.load(M_bf + offs_dd, mask=mask_dd, other=0.0)

    dM_C_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)

    qkv_stride_bn = F * S * D
    qkv_stride_n = D
    Q_bf_base = Q_ptr + b * qkv_stride_bn + f * S * D
    dO_bf_base = dO_ptr + b * qkv_stride_bn + f * S * D
    dQ_bf_base = dQ_ptr + b * qkv_stride_bn + f * S * D

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]

        q_ptrs = Q_bf_base + offs_s[:, None] * qkv_stride_n + offs_d[None, :]
        do_ptrs = dO_bf_base + offs_s[:, None] * qkv_stride_n + offs_d[None, :]
        Q_tile = tl.load(q_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        dO_tile = tl.load(do_ptrs, mask=mask_sd, other=0.0).to(tl.float32)

        dQ_tile = tl.dot(
            dO_tile.to(dot_dtype), tl.trans(M_f).to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip
        )
        dq_ptrs = dQ_bf_base + offs_s[:, None] * qkv_stride_n + offs_d[None, :]
        tl.store(dq_ptrs, dQ_tile.to(tl.float32), mask=mask_sd)

        dM_C_acc += tl.dot(
            tl.trans(Q_tile).to(dot_dtype), dO_tile.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip
        )

    dMC_bf = dM_C_ptr + (b * F + f) * BLOCK_D * BLOCK_D
    tl.store(dMC_bf + offs_dd, dM_C_acc, mask=mask_dd)


def phase_c_bwd(Q, M_post, dO, D, BLOCK_S=None, dot_precision=0):
    """Phase C̄ driver. Q, dO: (B, F, S, D); M_post: (B, F, D, D).
    Returns dQ: (B, F, S, D), dM_C: (B, F, D, D)."""
    p = _resolve_bwd_params()
    if BLOCK_S is None:
        BLOCK_S = p["BLOCK_S"]
    ns = p["phase_c_ns"]
    B, F, S, D_in = Q.shape
    assert D_in == D
    BLOCK_D = triton.next_power_of_2(D)
    dQ = torch.empty_like(Q)
    dM_C = torch.empty(B, F, BLOCK_D, BLOCK_D, device=Q.device, dtype=torch.float32)
    if D != BLOCK_D:
        pad = BLOCK_D - D
        M_post_p = torch.nn.functional.pad(M_post, (0, pad, 0, pad)).contiguous()
    else:
        M_post_p = M_post.contiguous()
    _phase_c_bwd_kernel[(B * F,)](
        Q,
        M_post_p,
        dO,
        dQ,
        dM_C,
        B=B,
        F=F,
        S=S,
        D=D,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        DOT_PRECISION=dot_precision,
        num_warps=8,
        num_stages=ns,
    )
    return dQ, dM_C[:, :, :D, :D].contiguous()


# ======================================================================
# Phase B̄ — serial reverse scan
#  - PyTorch fallback for H100/A100/Spark (cuBLAS bmm fuses BH×F efficiently)
#  - Triton kernel for Blackwell-DC (cuBLAS launch latency dominates there)
# ======================================================================
def _phase_b_bidi_bwd_pytorch(dM_C_fwd, dM_C_rev, P_all, g, dM_final_fwd):
    """PyTorch fallback (used on H100/A100 + consumer Blackwell — A100 cuBLAS
    bmm fuses (BH×F) D×D matmuls efficiently and beats our Triton kernel)."""
    BH, F, D, _ = dM_C_fwd.shape
    I_D = torch.eye(D, device=dM_C_fwd.device, dtype=dM_C_fwd.dtype)

    total_dM_fwd = torch.empty_like(dM_C_fwd)
    total_dM_fwd[:, F - 1] = dM_final_fwd + dM_C_fwd[:, F - 1]
    for f in range(F - 2, -1, -1):
        g_next = g[:, f + 1].view(BH, 1, 1)
        I_minus_P_next = I_D - P_all[:, f + 1]
        total_dM_fwd[:, f] = dM_C_fwd[:, f] + g_next * (I_minus_P_next.transpose(-2, -1) @ total_dM_fwd[:, f + 1])
    g0 = g[:, 0].view(BH, 1, 1)
    I_minus_P0 = I_D - P_all[:, 0]
    dM_init_fwd = g0 * (I_minus_P0.transpose(-2, -1) @ total_dM_fwd[:, 0])

    total_dM_rev = torch.empty_like(dM_C_rev)
    total_dM_rev[:, 0] = dM_C_rev[:, 0]
    for f in range(F - 1):
        g_next = g[:, f + 1].view(BH, 1, 1)
        I_minus_P_next = I_D - P_all[:, f + 1]
        total_dM_rev[:, f + 1] = dM_C_rev[:, f + 1] + g_next * (I_minus_P_next.transpose(-2, -1) @ total_dM_rev[:, f])

    return total_dM_fwd, total_dM_rev, dM_init_fwd


@triton.jit
def _phase_b_bidi_bwd_kernel(
    dM_C_fwd_ptr,
    dM_C_rev_ptr,
    P_all_ptr,
    g_ptr,
    dM_final_ptr,
    total_dM_fwd_ptr,
    total_dM_rev_ptr,
    dM_init_ptr,
    F: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """One program per (B*H). Loops F-1 times in-kernel for each direction.
    Replaces a PyTorch for-loop of small (D,D) matmuls — eliminates the
    cuBLAS launch latency that dominates on Blackwell-DC.
    """
    bh = tl.program_id(0)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    # Identity matrix (BLOCK_D, BLOCK_D), masked to D x D — used to compute (I - P)
    I_eye = tl.where(offs_d[:, None] == offs_d[None, :], 1.0, 0.0).to(tl.float32)
    I_eye = tl.where(mask_dd, I_eye, 0.0)

    bh_F_DD = bh * F * BLOCK_D * BLOCK_D
    bh_DD = bh * BLOCK_D * BLOCK_D

    # ──────────── Forward direction (reverse scan in time: F-1 → 0) ────────────
    accum = tl.load(dM_C_fwd_ptr + bh_F_DD + (F - 1) * BLOCK_D * BLOCK_D + offs_dd, mask=mask_dd, other=0.0).to(
        tl.float32
    )
    accum += tl.load(dM_final_ptr + bh_DD + offs_dd, mask=mask_dd, other=0.0).to(tl.float32)
    tl.store(total_dM_fwd_ptr + bh_F_DD + (F - 1) * BLOCK_D * BLOCK_D + offs_dd, accum, mask=mask_dd)

    for f_off in range(1, F):
        f = F - 1 - f_off
        P_next = tl.load(P_all_ptr + bh_F_DD + (f + 1) * BLOCK_D * BLOCK_D + offs_dd, mask=mask_dd, other=0.0).to(
            tl.float32
        )
        I_minus_P_T = tl.trans(I_eye - P_next)
        g_val = tl.load(g_ptr + bh * F + (f + 1)).to(tl.float32)
        new_accum = tl.dot(
            I_minus_P_T.to(tl.bfloat16),
            accum.to(tl.bfloat16),
            out_dtype=tl.float32,
            input_precision="ieee" if DOT_PRECISION == 2 else "tf32",
        )
        new_accum = g_val * new_accum
        dMC_f = tl.load(dM_C_fwd_ptr + bh_F_DD + f * BLOCK_D * BLOCK_D + offs_dd, mask=mask_dd, other=0.0).to(
            tl.float32
        )
        new_accum += dMC_f
        tl.store(total_dM_fwd_ptr + bh_F_DD + f * BLOCK_D * BLOCK_D + offs_dd, new_accum, mask=mask_dd)
        accum = new_accum

    P0 = tl.load(P_all_ptr + bh_F_DD + 0 + offs_dd, mask=mask_dd, other=0.0).to(tl.float32)
    I_minus_P0_T = tl.trans(I_eye - P0)
    g0 = tl.load(g_ptr + bh * F + 0).to(tl.float32)
    dM_init = g0 * tl.dot(
        I_minus_P0_T.to(tl.bfloat16),
        accum.to(tl.bfloat16),
        out_dtype=tl.float32,
        input_precision="ieee" if DOT_PRECISION == 2 else "tf32",
    )
    tl.store(dM_init_ptr + bh_DD + offs_dd, dM_init, mask=mask_dd)

    # ──────────── Reverse direction (forward scan: 0 → F-1) ────────────
    accum = tl.load(dM_C_rev_ptr + bh_F_DD + 0 + offs_dd, mask=mask_dd, other=0.0).to(tl.float32)
    tl.store(total_dM_rev_ptr + bh_F_DD + 0 + offs_dd, accum, mask=mask_dd)

    for f in range(F - 1):
        P_next = tl.load(P_all_ptr + bh_F_DD + (f + 1) * BLOCK_D * BLOCK_D + offs_dd, mask=mask_dd, other=0.0).to(
            tl.float32
        )
        I_minus_P_T = tl.trans(I_eye - P_next)
        g_val = tl.load(g_ptr + bh * F + (f + 1)).to(tl.float32)
        new_accum = tl.dot(
            I_minus_P_T.to(tl.bfloat16),
            accum.to(tl.bfloat16),
            out_dtype=tl.float32,
            input_precision="ieee" if DOT_PRECISION == 2 else "tf32",
        )
        new_accum = g_val * new_accum
        dMC_f1 = tl.load(dM_C_rev_ptr + bh_F_DD + (f + 1) * BLOCK_D * BLOCK_D + offs_dd, mask=mask_dd, other=0.0).to(
            tl.float32
        )
        new_accum += dMC_f1
        tl.store(total_dM_rev_ptr + bh_F_DD + (f + 1) * BLOCK_D * BLOCK_D + offs_dd, new_accum, mask=mask_dd)
        accum = new_accum


def phase_b_bidi_bwd(dM_C_fwd, dM_C_rev, P_all, g, dM_final_fwd):
    """Forward direction reverse scan + reverse direction forward scan.

    Args:
      dM_C_fwd, dM_C_rev: (BH, F, D, D) — Phase C̄ injections per frame, per direction.
      P_all:              (BH, F, D, D)
      g:                  (BH, F)
      dM_final_fwd:       (BH, D, D) — grad on final forward state (0 if not exposed)

    Returns:
      total_dM_fwd, total_dM_rev: (BH, F, D, D)
      dM_init_fwd: (BH, D, D)
    """
    BH, F, D, _ = dM_C_fwd.shape
    BLOCK_D = triton.next_power_of_2(D)
    fp32 = torch.float32

    # Triton wins everywhere when properly tuned (lesson #4 from
    # fused_improve_plan.md "Lessons learned"): persistent-state kernels
    # need num_warps swept up to 32, not capped at 4. nw=4 was register-
    # pressure-bound; nw=8 ns=1 hits ~peak across H100/A100/GB200 (8-10×
    # over pytorch at F=11). Consumer Blackwell still needs PyTorch
    # fallback (SRAM OOM on the kernel due to BLOCK_D × BLOCK_D buffers).
    use_triton = True
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 10:
            props = torch.cuda.get_device_properties(0)
            smem = getattr(props, "shared_memory_per_multiprocessor", 0)
            use_triton = smem >= 150 * 1024  # only false on consumer Blackwell
    if not use_triton:
        return _phase_b_bidi_bwd_pytorch(dM_C_fwd, dM_C_rev, P_all, g, dM_final_fwd)

    def pad_DD(x):
        if x.shape[-1] == BLOCK_D:
            return x.contiguous()
        pad = BLOCK_D - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad, 0, pad)).contiguous()

    dM_C_fwd_p = pad_DD(dM_C_fwd)
    dM_C_rev_p = dM_C_fwd_p if dM_C_fwd is dM_C_rev else pad_DD(dM_C_rev)
    P_all_p = pad_DD(P_all.float() if P_all.dtype != fp32 else P_all)
    dM_final_p = (
        pad_DD(dM_final_fwd.float() if dM_final_fwd.dtype != fp32 else dM_final_fwd)
        .reshape(BH, BLOCK_D, BLOCK_D)
        .contiguous()
        if dM_final_fwd.shape[-1] != BLOCK_D
        else dM_final_fwd.contiguous()
    )
    # Simpler: pad as 3D
    if dM_final_fwd.shape[-1] != BLOCK_D:
        pad = BLOCK_D - dM_final_fwd.shape[-1]
        dM_final_p = torch.nn.functional.pad(
            dM_final_fwd.float() if dM_final_fwd.dtype != fp32 else dM_final_fwd, (0, pad, 0, pad)
        ).contiguous()
    else:
        dM_final_p = (dM_final_fwd.float() if dM_final_fwd.dtype != fp32 else dM_final_fwd).contiguous()
    g_c = g.float().contiguous() if g.dtype != fp32 else g.contiguous()

    total_dM_fwd_p = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=dM_C_fwd.device, dtype=fp32)
    total_dM_rev_p = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=dM_C_fwd.device, dtype=fp32)
    dM_init_p = torch.empty(BH, BLOCK_D, BLOCK_D, device=dM_C_fwd.device, dtype=fp32)

    if dM_C_fwd_p.dtype != fp32:
        dM_C_fwd_p = dM_C_fwd_p.float()
    if dM_C_rev_p.dtype != fp32:
        dM_C_rev_p = dM_C_rev_p.float()
    if P_all_p.dtype != fp32:
        P_all_p = P_all_p.float()

    _phase_b_bidi_bwd_kernel[(BH,)](
        dM_C_fwd_p,
        dM_C_rev_p,
        P_all_p,
        g_c,
        dM_final_p,
        total_dM_fwd_p,
        total_dM_rev_p,
        dM_init_p,
        F=F,
        D=D,
        BLOCK_D=BLOCK_D,
        DOT_PRECISION=0,
        num_warps=8,
        num_stages=1,  # nw=8 ns=1 wins on H100/A100/GB200 (8-30× pt)
    )

    total_dM_fwd = total_dM_fwd_p[:, :, :D, :D].contiguous()
    total_dM_rev = total_dM_rev_p[:, :, :D, :D].contiguous()
    dM_init_fwd = dM_init_p[:, :D, :D].contiguous()

    return total_dM_fwd, total_dM_rev, dM_init_fwd


def combine_bidi_dA_dP_dg(total_dM_fwd, total_dM_rev, M_fwd_prev, M_rev_at, P_all, g):
    """Combine forward + reverse direction's contributions into per-frame (dA, dP, dg).
    Forward scan at f uses (P_f, A_f, g_f) with state M_fwd_prev[f].
    Reverse scan producing M_rev[f-1] uses (P_f, A_f, g_f) with state M_rev_at[f]  (for f >= 1)."""
    BH, F, D, _ = total_dM_fwd.shape
    I_D = torch.eye(D, device=total_dM_fwd.device, dtype=total_dM_fwd.dtype)
    g_per = g.view(BH, F, 1, 1)
    I_minus_P = I_D - P_all

    dA_total = total_dM_fwd.clone()
    dA_total[:, 1:] += total_dM_rev[:, : F - 1]

    dP_fwd = -g_per * (total_dM_fwd @ M_fwd_prev.transpose(-2, -1))
    dP_rev = torch.zeros_like(dP_fwd)
    dP_rev[:, 1:] = -g_per[:, 1:] * (total_dM_rev[:, : F - 1] @ M_rev_at[:, 1:].transpose(-2, -1))
    dP_total = dP_fwd + dP_rev

    I_minus_P_M_fwd = I_minus_P @ M_fwd_prev
    dg_fwd = (total_dM_fwd * I_minus_P_M_fwd).sum(dim=(-2, -1))
    dg_rev = torch.zeros_like(dg_fwd)
    dg_rev[:, 1:] = (total_dM_rev[:, : F - 1] * (I_minus_P[:, 1:] @ M_rev_at[:, 1:])).sum(dim=(-2, -1))
    dg_total = dg_fwd + dg_rev

    return dA_total, dP_total, dg_total


# ======================================================================
# Phase Ā KV — backward through P_f = K^T diag(β) K and A_f = K^T diag(β) V
# ======================================================================
@triton.jit
def _phase_a_kv_bwd_kernel(
    K_ptr,
    V_ptr,
    beta_ptr,
    dA_ptr,
    dP_ptr,
    dK_ptr,
    dV_ptr,
    dbeta_ptr,
    B: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """One block per (b, f). Loops over S tiles producing dK, dV, dβ from dA, dP constants."""
    # Backward kernels use bf16 TC (with fp32 accumulate) — enough precision for gradients
    # while avoiding the 3× Markidis fp32 IEEE penalty that dominates at P0.
    # cos_sim bar is 0.999; measured cos_dx stays at 0.999+.
    tl.bfloat16
    dot_ip: tl.constexpr = "tf32"

    pid = tl.program_id(0)
    b = pid // F
    f = pid % F

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    dA = tl.load(dA_ptr + (b * F + f) * BLOCK_D * BLOCK_D + offs_dd, mask=mask_dd, other=0.0).to(tl.bfloat16)
    dP = tl.load(dP_ptr + (b * F + f) * BLOCK_D * BLOCK_D + offs_dd, mask=mask_dd, other=0.0).to(tl.bfloat16)

    qkv_stride_bn = F * S * D
    qkv_stride_n = D
    K_bf_base = K_ptr + b * qkv_stride_bn + f * S * D
    V_bf_base = V_ptr + b * qkv_stride_bn + f * S * D
    beta_bf_base = beta_ptr + b * F * S + f * S
    dK_bf_base = dK_ptr + b * qkv_stride_bn + f * S * D
    dV_bf_base = dV_ptr + b * qkv_stride_bn + f * S * D
    dbeta_bf_base = dbeta_ptr + b * F * S + f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]

        k_ptrs = K_bf_base + offs_s[:, None] * qkv_stride_n + offs_d[None, :]
        v_ptrs = V_bf_base + offs_s[:, None] * qkv_stride_n + offs_d[None, :]
        K_tile = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        V_tile = tl.load(v_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_tile = tl.load(beta_bf_base + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        K_dP = tl.dot(K_tile.to(tl.bfloat16), dP, out_dtype=tl.float32, input_precision=dot_ip)
        K_dPT = tl.dot(K_tile.to(tl.bfloat16), tl.trans(dP), out_dtype=tl.float32, input_precision=dot_ip)
        dK_from_P = beta_tile[:, None] * (K_dP + K_dPT)

        V_dAT = tl.dot(V_tile.to(tl.bfloat16), tl.trans(dA), out_dtype=tl.float32, input_precision=dot_ip)
        dK_from_A = beta_tile[:, None] * V_dAT
        dK_tile = dK_from_P + dK_from_A

        K_dA = tl.dot(K_tile.to(tl.bfloat16), dA, out_dtype=tl.float32, input_precision=dot_ip)
        dV_tile = beta_tile[:, None] * K_dA

        dbeta_tile = tl.sum(K_dP * K_tile, axis=1) + tl.sum(K_dA * V_tile, axis=1)

        dk_ptrs = dK_bf_base + offs_s[:, None] * qkv_stride_n + offs_d[None, :]
        dv_ptrs = dV_bf_base + offs_s[:, None] * qkv_stride_n + offs_d[None, :]
        tl.store(dk_ptrs, dK_tile, mask=mask_sd)
        tl.store(dv_ptrs, dV_tile, mask=mask_sd)
        tl.store(dbeta_bf_base + offs_s, dbeta_tile, mask=mask_s)


def phase_a_kv_bwd(K, V, beta, dA, dP, D, BLOCK_S=None, dot_precision=0):
    """Phase Ā KV driver. K, V: (B, F, S, D); dA, dP: (B, F, D, D); beta: (B, F, S).
    Returns dK, dV, dbeta."""
    p = _resolve_bwd_params()
    if BLOCK_S is None:
        BLOCK_S = p["BLOCK_S"]
    ns = p["phase_a_ns"]
    B, F, S, D_in = K.shape
    BLOCK_D = triton.next_power_of_2(D)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    dbeta = torch.empty_like(beta)

    def pad_DxD(x):
        if x.shape[-1] == BLOCK_D:
            return x.contiguous()
        pad = BLOCK_D - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad, 0, pad)).contiguous()

    dA_p = pad_DxD(dA)
    dP_p = pad_DxD(dP)

    _phase_a_kv_bwd_kernel[(B * F,)](
        K,
        V,
        beta,
        dA_p,
        dP_p,
        dK,
        dV,
        dbeta,
        B=B,
        F=F,
        S=S,
        D=D,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        DOT_PRECISION=dot_precision,
        num_warps=8,
        num_stages=ns,
    )
    return dK, dV, dbeta


# ======================================================================
# Phase Ā Z — backward through P_z = K^T diag(β) K, B_z = K^T β
# ======================================================================
@triton.jit
def _phase_a_z_bwd_kernel(
    K_ptr,
    beta_ptr,
    dB_z_ptr,
    dP_z_ptr,
    dK_ptr,
    dbeta_ptr,
    B: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Phase Ā for Z-stream. One block per (b, f). dB_z (D-vector) broadcasts across S tiles."""
    # Backward kernels use bf16 TC (with fp32 accumulate) — enough precision for gradients
    # while avoiding the 3× Markidis fp32 IEEE penalty that dominates at P0.
    # cos_sim bar is 0.999; measured cos_dx stays at 0.999+.
    tl.bfloat16
    dot_ip: tl.constexpr = "tf32"

    pid = tl.program_id(0)
    b = pid // F
    f = pid % F

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    dB_z = tl.load(dB_z_ptr + (b * F + f) * BLOCK_D + offs_d, mask=mask_d, other=0.0)
    dP_z = tl.load(dP_z_ptr + (b * F + f) * BLOCK_D * BLOCK_D + offs_dd, mask=mask_dd, other=0.0).to(tl.bfloat16)

    qkv_stride_bn = F * S * D
    qkv_stride_n = D
    K_bf_base = K_ptr + b * qkv_stride_bn + f * S * D
    beta_bf_base = beta_ptr + b * F * S + f * S
    dK_bf_base = dK_ptr + b * qkv_stride_bn + f * S * D
    dbeta_bf_base = dbeta_ptr + b * F * S + f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]

        k_ptrs = K_bf_base + offs_s[:, None] * qkv_stride_n + offs_d[None, :]
        K_tile = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_tile = tl.load(beta_bf_base + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        K_dPz = tl.dot(K_tile.to(tl.bfloat16), dP_z, out_dtype=tl.float32, input_precision=dot_ip)
        K_dPzT = tl.dot(K_tile.to(tl.bfloat16), tl.trans(dP_z), out_dtype=tl.float32, input_precision=dot_ip)
        dK_from_Pz = beta_tile[:, None] * (K_dPz + K_dPzT)
        dK_from_Bz = beta_tile[:, None] * dB_z[None, :]
        dK_tile = dK_from_Pz + dK_from_Bz

        dbeta_from_Pz = tl.sum(K_dPz * K_tile, axis=1)
        dbeta_from_Bz = tl.sum(K_tile * dB_z[None, :], axis=1)
        dbeta_tile = dbeta_from_Pz + dbeta_from_Bz

        dk_ptrs = dK_bf_base + offs_s[:, None] * qkv_stride_n + offs_d[None, :]
        tl.store(dk_ptrs, dK_tile, mask=mask_sd)
        tl.store(dbeta_bf_base + offs_s, dbeta_tile, mask=mask_s)


def phase_a_z_bwd(K, beta, dB_z, dP_z, D, BLOCK_S=None, dot_precision=0):
    """Phase Ā_z driver."""
    p = _resolve_bwd_params()
    if BLOCK_S is None:
        BLOCK_S = p["BLOCK_S"]
    ns = p["phase_a_ns"]
    B, F, S, _ = K.shape
    BLOCK_D = triton.next_power_of_2(D)
    dK = torch.empty_like(K)
    dbeta = torch.empty_like(beta)

    def pad_D(x):
        if x.shape[-1] == BLOCK_D:
            return x.contiguous()
        pad = BLOCK_D - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad)).contiguous()

    def pad_DxD(x):
        if x.shape[-1] == BLOCK_D:
            return x.contiguous()
        pad = BLOCK_D - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad, 0, pad)).contiguous()

    dB_z_p = pad_D(dB_z)
    dP_z_p = pad_DxD(dP_z)

    _phase_a_z_bwd_kernel[(B * F,)](
        K,
        beta,
        dB_z_p,
        dP_z_p,
        dK,
        dbeta,
        B=B,
        F=F,
        S=S,
        D=D,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        DOT_PRECISION=dot_precision,
        num_warps=8,
        num_stages=ns,
    )
    return dK, dbeta


# ======================================================================
# Output normalization divide VJP: out = num / (den + eps)
# ======================================================================
def output_divide_bwd(dout, num, den, eps=1e-6, out_dtype=None):
    """dnum = dout / (den+eps); dden = -sum_D(dout*num) / (den+eps)^2.

    Computes in fp32 for numerical stability, casts outputs to `out_dtype`
    (defaults to dout dtype) to keep memory low.
    """
    if out_dtype is None:
        out_dtype = dout.dtype
    den_broadcast = den.float().permute(0, 2, 1).unsqueeze(-1) + eps
    dout_f = dout.float()
    num_f = num.float()
    dnum = (dout_f / den_broadcast).to(out_dtype)
    dden_per = -dout_f * num_f / (den_broadcast**2)
    dden = dden_per.sum(dim=-1).permute(0, 2, 1).contiguous().to(out_dtype)
    return dnum, dden


# ======================================================================
# Full autograd Function
# ======================================================================
def _rope_pair_flip(X):
    """Pair-flip along last dim: swap (d, d^1) pairs."""
    D = X.shape[-1]
    return X.reshape(*X.shape[:-1], D // 2, 2).flip(-1).reshape(*X.shape)


def _apply_rope(X, cos, sin):
    """RoPE: X_rot = X * cos + pair_flip(X) * sin."""
    return X * cos[None, :, None, :] + _rope_pair_flip(X) * sin[None, :, None, :]


def _unrope(dY, cos, sin):
    """VJP of _apply_rope: d/dX = d/dY * cos + pair_flip(d/dY * sin)."""
    return dY * cos[None, :, None, :] + _rope_pair_flip(dY * sin[None, :, None, :])


# ======================================================================
# Fused rope+relu (fwd, op #5) and unrope+add+relu_mask (bwd, op #13).
# Forward Phase A Triton kernel already does rope+relu inline; mirroring
# that on the bwd side fuses 2 PyTorch chains (each ~13-17% of bwd at
# F=11 H100/A100) into 2 Triton kernels.
# ======================================================================
@triton.jit
def _rope_relu_fwd_kernel(
    Q_in_ptr,
    K_in_ptr,  # (BHFS, D) bf16, contiguous
    rope_cos_ptr,
    rope_sin_ptr,  # (FS, D) fp32
    Q_relu_ptr,
    K_relu_ptr,  # outputs, bf16 (post-relu, used by Phase C̄ den)
    Q_rope_ptr,
    K_rope_ptr,  # outputs, bf16 (post-rope, used by Phase C̄ KV)
    k_scale,
    FS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (BH, F, S) row — element-wise relu + paired-flip rope."""
    pid = tl.program_id(0)
    fs_idx = pid % FS

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    pair_mask = offs_d_pair < D

    base_in = pid * D
    base_rope = fs_idx * D

    Q = tl.load(Q_in_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    Q_pair = tl.load(Q_in_ptr + base_in + offs_d_pair, mask=pair_mask, other=0.0).to(tl.float32)
    K = tl.load(K_in_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    K_pair = tl.load(K_in_ptr + base_in + offs_d_pair, mask=pair_mask, other=0.0).to(tl.float32)
    cos = tl.load(rope_cos_ptr + base_rope + offs_d, mask=mask_d, other=1.0).to(tl.float32)
    sin = tl.load(rope_sin_ptr + base_rope + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    Q_relu = tl.maximum(Q, 0.0)
    Q_pair_relu = tl.maximum(Q_pair, 0.0)
    Q_rope = Q_relu * cos + Q_pair_relu * sin

    K_relu = tl.maximum(K, 0.0) * k_scale
    K_pair_relu = tl.maximum(K_pair, 0.0) * k_scale
    K_rope = K_relu * cos + K_pair_relu * sin

    tl.store(Q_relu_ptr + base_in + offs_d, Q_relu.to(tl.bfloat16), mask=mask_d)
    tl.store(K_relu_ptr + base_in + offs_d, K_relu.to(tl.bfloat16), mask=mask_d)
    tl.store(Q_rope_ptr + base_in + offs_d, Q_rope.to(tl.bfloat16), mask=mask_d)
    tl.store(K_rope_ptr + base_in + offs_d, K_rope.to(tl.bfloat16), mask=mask_d)


def fused_rope_relu_fwd(Q_normed, K_normed, rope_cos, rope_sin, k_scale, F, S):
    """Fused rope + relu forward. Inputs are (BH, F, S, D) bf16; outputs same shape.
    Returns (Q_post_relu, K_post_relu, Q_for_num, K_kv).
    Equivalent PyTorch:
        Q_relu = clamp(Q_normed, min=0)
        K_relu = clamp(K_normed, min=0) * k_scale
        Q_rope = apply_rope(Q_relu)
        K_rope = apply_rope(K_relu)
    """
    BH, F_in, S_in, D = Q_normed.shape
    assert F_in == F and S_in == S
    BLOCK_D = triton.next_power_of_2(D)
    FS = F * S

    Q_relu = torch.empty_like(Q_normed)
    K_relu = torch.empty_like(K_normed)
    Q_rope = torch.empty_like(Q_normed)
    K_rope = torch.empty_like(K_normed)

    Q_in_c = Q_normed.contiguous()
    K_in_c = K_normed.contiguous()
    cos_c = (
        rope_cos.reshape(FS, D).float().contiguous()
        if rope_cos.dtype != torch.float32
        else rope_cos.reshape(FS, D).contiguous()
    )
    sin_c = (
        rope_sin.reshape(FS, D).float().contiguous()
        if rope_sin.dtype != torch.float32
        else rope_sin.reshape(FS, D).contiguous()
    )

    _rope_relu_fwd_kernel[(BH * FS,)](
        Q_in_c,
        K_in_c,
        cos_c,
        sin_c,
        Q_relu,
        K_relu,
        Q_rope,
        K_rope,
        float(k_scale),
        FS=FS,
        D=D,
        BLOCK_D=BLOCK_D,
        num_warps=2,
        num_stages=1,
    )
    return Q_relu, K_relu, Q_rope, K_rope


@triton.jit
def _rope_unrope_bwd_kernel(
    dQ_kv_ptr,
    dK_kv_ptr,  # (BHFS, D) bf16
    dQ_z_ptr,
    dK_z_ptr,  # (BHFS, D) bf16 — extra grad to add
    Q_relu_ptr,
    K_relu_ptr,  # (BHFS, D) bf16 — for relu mask
    rope_cos_ptr,
    rope_sin_ptr,  # (FS, D) fp32
    dQ_normed_ptr,
    dK_normed_ptr,  # outputs, bf16
    k_scale,
    FS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (BH, F, S) row — fused unrope + add dz + relu mask."""
    pid = tl.program_id(0)
    fs_idx = pid % FS

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    pair_mask = offs_d_pair < D

    base_in = pid * D
    base_rope = fs_idx * D

    dQ_kv = tl.load(dQ_kv_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    dQ_kv_pair = tl.load(dQ_kv_ptr + base_in + offs_d_pair, mask=pair_mask, other=0.0).to(tl.float32)
    dK_kv = tl.load(dK_kv_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    dK_kv_pair = tl.load(dK_kv_ptr + base_in + offs_d_pair, mask=pair_mask, other=0.0).to(tl.float32)
    dQ_z = tl.load(dQ_z_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    dK_z = tl.load(dK_z_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    Q_relu = tl.load(Q_relu_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    K_relu = tl.load(K_relu_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    cos = tl.load(rope_cos_ptr + base_rope + offs_d, mask=mask_d, other=1.0).to(tl.float32)
    sin_pair = tl.load(rope_sin_ptr + base_rope + offs_d_pair, mask=pair_mask, other=0.0).to(tl.float32)

    # unrope: dY_pre[d] = dY[d]*cos[d] + dY[d^1]*sin[d^1]
    dQ_post_relu = dQ_kv * cos + dQ_kv_pair * sin_pair
    dQ_post_relu = dQ_post_relu + dQ_z

    dK_post_relu = dK_kv * cos + dK_kv_pair * sin_pair
    dK_post_relu = dK_post_relu + dK_z

    Q_mask_f = (Q_relu > 0.0).to(tl.float32)
    K_mask_f = (K_relu > 0.0).to(tl.float32)
    dQ_normed = dQ_post_relu * Q_mask_f
    dK_normed = dK_post_relu * K_mask_f * k_scale

    tl.store(dQ_normed_ptr + base_in + offs_d, dQ_normed.to(tl.bfloat16), mask=mask_d)
    tl.store(dK_normed_ptr + base_in + offs_d, dK_normed.to(tl.bfloat16), mask=mask_d)


def fused_rope_unrope_bwd(dQ_kv, dK_kv, dQ_z, dK_z, Q_relu, K_relu, rope_cos, rope_sin, k_scale, F, S):
    """Fused unrope + add dz + relu mask backward. All BHFSD bf16, returns (dQ_normed, dK_normed).
    Equivalent PyTorch:
        dQ_post = unrope(dQ_kv) + dQ_z
        dK_post = unrope(dK_kv) + dK_z
        dQ_normed = dQ_post * (Q_relu > 0)
        dK_normed = dK_post * (K_relu > 0) * k_scale
    """
    BH, F_in, S_in, D = dQ_kv.shape
    assert F_in == F and S_in == S
    BLOCK_D = triton.next_power_of_2(D)
    FS = F * S

    dQ_normed = torch.empty_like(dQ_kv)
    dK_normed = torch.empty_like(dK_kv)

    cos_c = (
        rope_cos.reshape(FS, D).float().contiguous()
        if rope_cos.dtype != torch.float32
        else rope_cos.reshape(FS, D).contiguous()
    )
    sin_c = (
        rope_sin.reshape(FS, D).float().contiguous()
        if rope_sin.dtype != torch.float32
        else rope_sin.reshape(FS, D).contiguous()
    )

    _rope_unrope_bwd_kernel[(BH * FS,)](
        dQ_kv.contiguous(),
        dK_kv.contiguous(),
        dQ_z.contiguous(),
        dK_z.contiguous(),
        Q_relu.contiguous(),
        K_relu.contiguous(),
        cos_c,
        sin_c,
        dQ_normed,
        dK_normed,
        float(k_scale),
        FS=FS,
        D=D,
        BLOCK_D=BLOCK_D,
        num_warps=2,
        num_stages=1,
    )
    return dQ_normed, dK_normed


class FusedBiGDNChunkwiseFunction(torch.autograd.Function):
    """BiGDN autograd with chunkwise forward + chunkwise backward.

    Forward: full-channel RMSNorm → chunkwise phase_a/b/c → output-divide.
    Backward: output-divide VJP → chunkwise Phase C̄/B̄/Ā (KV + Z) → ReLU + RoPE VJPs
              → full-channel RMSNorm backward.

    Drop-in for FusedBiGDNFunction when gradients are needed.
    """

    @staticmethod
    def forward(
        ctx,
        qkv,
        beta,
        decay,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        F,
        S,
        k_scale=1.0,
        norm_eps=1e-5,
        eps=1e-6,
        dot_precision=0,
        BLOCK_S=None,
    ):
        # Resolve BLOCK_S per arch — consumer Blackwell needs smaller tiles to
        # fit ~102 KB SRAM (default 64 → OOM, drop to 16).
        if BLOCK_S is None:
            BLOCK_S = _resolve_bwd_block_s()
        B, N, three, H, D = qkv.shape
        C = H * D
        assert three == 3 and N == F * S

        device = qkv.device
        fp32 = torch.float32
        if q_norm_weight is None:
            q_norm_weight = torch.ones(C, device=device, dtype=fp32)
        if k_norm_weight is None:
            k_norm_weight = torch.ones(C, device=device, dtype=fp32)

        # Full-channel RMSNorm — keep q_raw/k_raw as VIEWS into qkv, don't upcast to fp32.
        # Only the sum-of-squares needs fp32 accumulation; the per-element multiply can stay bf16.
        q_raw_v = qkv[:, :, 0]  # view, same dtype as qkv
        k_raw_v = qkv[:, :, 1]
        q_inv_rms = torch.rsqrt((q_raw_v.float().pow(2)).sum(dim=(-2, -1)) / C + norm_eps)
        k_inv_rms = torch.rsqrt((k_raw_v.float().pow(2)).sum(dim=(-2, -1)) / C + norm_eps)
        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)
        qkv_normed = qkv.clone()
        qkv_normed[:, :, 0] = (q_raw_v.float() * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(qkv.dtype)
        qkv_normed[:, :, 1] = (k_raw_v.float() * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(qkv.dtype)

        # Chunkwise forward (identity norm inside kernel; norm already done above)
        dummy_inv = torch.ones(B, N, device=device, dtype=fp32)
        dummy_nw = torch.ones(C, device=device, dtype=fp32)
        I_P_kv, A, I_P_z, B_z = phase_a(
            qkv_normed,
            beta,
            dummy_inv,
            dummy_inv,
            dummy_nw,
            dummy_nw,
            rope_cos,
            rope_sin,
            F=F,
            S=S,
            k_scale=k_scale,
            norm_eps=norm_eps,
            dot_precision=dot_precision,
        )
        M_fwd, z_fwd, _, _ = phase_b_triton(I_P_kv, A, I_P_z, B_z, decay, F=F, dot_precision=dot_precision, direction=1)
        num_out, den_out = phase_c(
            qkv_normed,
            dummy_inv,
            dummy_nw,
            rope_cos,
            rope_sin,
            M_fwd,
            z_fwd,
            F=F,
            S=S,
            dot_precision=dot_precision,
            accumulate=False,
        )
        _, _, M_rev, z_rev = phase_b_triton(I_P_kv, A, I_P_z, B_z, decay, F=F, dot_precision=dot_precision, direction=2)
        phase_c(
            qkv_normed,
            dummy_inv,
            dummy_nw,
            rope_cos,
            rope_sin,
            M_rev,
            z_rev,
            F=F,
            S=S,
            dot_precision=dot_precision,
            num_out=num_out,
            den_out=den_out,
            accumulate=True,
        )

        total_den = den_out.float().permute(0, 2, 1).unsqueeze(-1)
        out = (num_out.float() / (total_den + eps)).to(qkv.dtype)

        # 2026-04-30 PM: Save Phase A+B intermediates instead of recomputing —
        # closes the 17.1% (F=11 H100) recompute share at the cost of ~360 MB
        # at B=8 (trivial vs model state). qkv_normed is recomputed cheap (~8% phase).
        del qkv_normed

        ctx.save_for_backward(
            qkv,
            beta,
            decay,
            q_norm_weight,
            k_norm_weight,
            q_inv_rms,
            k_inv_rms,
            rope_cos,
            rope_sin,
            num_out,
            den_out,
            I_P_kv,
            A,
            I_P_z,
            B_z,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
        )
        ctx.shape = (B, N, H, D, F, S, C)
        ctx.k_scale = k_scale
        ctx.norm_eps = norm_eps
        ctx.eps = eps
        ctx.dot_precision = dot_precision
        ctx.BLOCK_S = BLOCK_S
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            qkv,
            beta,
            decay,
            q_norm_weight,
            k_norm_weight,
            q_inv_rms,
            k_inv_rms,
            rope_cos,
            rope_sin,
            num_out,
            den_out,
            I_P_kv,
            A,
            I_P_z,
            B_z,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
        ) = ctx.saved_tensors
        B, N, H, D, F, S, C = ctx.shape
        k_scale, eps = ctx.k_scale, ctx.eps
        ctx.norm_eps
        dot_precision, BLOCK_S = ctx.dot_precision, ctx.BLOCK_S
        device = qkv.device
        fp32 = torch.float32
        dtype = qkv.dtype  # bf16 typically
        BH = B * H

        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)

        # ──── 1. Output divide VJP — keep dnum/dden in bf16 to save ~725MB at B=8 ────
        dnum, dden = output_divide_bwd(dout, num_out, den_out, eps=eps, out_dtype=dtype)
        del num_out, den_out

        # ──── 2. Reconstruct qkv_normed (bf16, same as forward) ────
        q_raw_v = qkv[:, :, 0]
        k_raw_v = qkv[:, :, 1]
        qkv_normed = qkv.clone()
        qkv_normed[:, :, 0] = (q_raw_v.float() * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(dtype)
        qkv_normed[:, :, 1] = (k_raw_v.float() * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(dtype)

        # ──── 3. Phase A + B intermediates loaded from ctx (saved during forward) ────
        # Adapt state to (BH, F, D, D) / (BH, F, D) — fp32 for scan math precision
        I_P_kv.shape[-1]
        I_D = torch.eye(D, device=device, dtype=fp32)
        P_kv_all = I_D[None, None] - I_P_kv[:, :, :D, :D].float()
        P_z_all = I_D[None, None] - I_P_z[:, :, :D, :D].float()
        del I_P_kv, A, I_P_z, B_z  # free padded versions — we have D×D unpadded now

        M_fwd_d = M_fwd[:, :, :D, :D].float()
        del M_fwd
        M_rev_d = M_rev[:, :, :D, :D].float()
        del M_rev
        z_fwd_d = z_fwd[:, :, :D].float()
        del z_fwd
        z_rev_d = z_rev[:, :, :D].float()
        del z_rev
        zero_DD = torch.zeros(BH, 1, D, D, device=device, dtype=fp32)
        zero_D = torch.zeros(BH, 1, D, device=device, dtype=fp32)
        M_fwd_full = torch.cat([zero_DD, M_fwd_d], dim=1)
        del M_fwd_d
        M_rev_full = torch.cat([M_rev_d, zero_DD], dim=1)
        del M_rev_d
        z_fwd_full = torch.cat([zero_D, z_fwd_d], dim=1)
        del z_fwd_d
        z_rev_full = torch.cat([z_rev_d, zero_D], dim=1)
        del z_rev_d

        # ──── 4. Post-relu + post-rope directly in BHFSD format (skip BNHD intermediates) ────
        # V is never normalized/relu'd — use qkv directly (not qkv_normed).
        def bnhd_to_bhfsd(x):
            return x.permute(0, 2, 1, 3).reshape(B, H, F, S, D).reshape(BH, F, S, D).contiguous()

        def bhfsd_to_bnhd(x):
            return x.reshape(BH, F * S, D).reshape(B, H, N, D).permute(0, 2, 1, 3).contiguous()

        # Go direct: Q_normed (BHFSD) → Q_post_relu (BHFSD) → Q_for_num (BHFSD).
        # Avoids holding BNHD duplicates of 363 MB each at B=8.
        Q_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 0])
        K_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 1])
        V_bhfsd = bnhd_to_bhfsd(qkv[:, :, 2])  # V: use raw qkv (no norm applied)
        del qkv_normed  # 1.09 GB freed

        # Fused rope+relu: combines clamp(Q_normed) + apply_rope (4 PyTorch ops)
        # into 1 Triton kernel. Closes ~13% of bwd time on H100 F=11 (op #5).
        Q_post_relu_bhfsd, K_post_relu_bhfsd, Q_for_num_bhfsd, K_kv_bhfsd = fused_rope_relu_fwd(
            Q_normed_bhfsd,
            K_normed_bhfsd,
            rope_cos,
            rope_sin,
            k_scale,
            F,
            S,
        )
        del Q_normed_bhfsd, K_normed_bhfsd

        # For Phase C den, we use Q_post_relu (no rope). Reuse.
        Q_for_den_bhfsd = Q_post_relu_bhfsd
        K_z_bhfsd = K_post_relu_bhfsd
        # rope_cos_fs / rope_sin_fs no longer needed by Phase C̄ KV path here,
        # but kept for the bwd unrope below.
        rope_cos.reshape(F, S, D)
        rope_sin.reshape(F, S, D)

        beta_bhfs = beta.reshape(BH, F, S).float()
        decay_bhf = decay.reshape(BH, F).float()
        dO_bhfsd = bnhd_to_bhfsd(dnum)
        dden_bhfs = dden.reshape(BH, F, S).contiguous()
        del dnum

        # 4. KV-chain: Phase C̄ → B̄ → Ā
        M_combined = (M_fwd_full[:, 1:] + M_rev_full[:, :F]).contiguous()
        dQ_kv, dM_C = phase_c_bwd(
            Q_for_num_bhfsd.contiguous(), M_combined, dO_bhfsd, D, BLOCK_S=BLOCK_S, dot_precision=dot_precision
        )
        dM_final_fwd = torch.zeros(BH, D, D, device=device, dtype=fp32)
        total_dM_fwd, total_dM_rev, dM_init_kv = phase_b_bidi_bwd(dM_C, dM_C, P_kv_all, decay_bhf, dM_final_fwd)
        dA_total, dP_kv_total, dg_kv_total = combine_bidi_dA_dP_dg(
            total_dM_fwd,
            total_dM_rev,
            M_fwd_full[:, :-1].contiguous(),
            M_rev_full[:, :F].contiguous(),
            P_kv_all,
            decay_bhf,
        )
        dK_kv, dV, dbeta_kv = phase_a_kv_bwd(
            K_kv_bhfsd.contiguous(),
            V_bhfsd.contiguous(),
            beta_bhfs,
            dA_total,
            dP_kv_total,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )

        # 5. Z-chain: Phase C̄ → B̄ → Ā
        # Note: z_fwd_full is fp32 (kernel output); dden_bhfs is bf16 now (memory-opt).
        # dQ_z output should be bf16 (matches Phase C̄ dQ_kv). dz_C must be fp32 for
        # the B̄_z scan which uses fp32 P_z.
        z_combined = z_fwd_full[:, 1:] + z_rev_full[:, :F]  # fp32
        dQ_z = (dden_bhfs.unsqueeze(-1) * z_combined.unsqueeze(2)).to(dtype)  # bf16
        dz_C = (Q_for_den_bhfsd.float() * dden_bhfs.unsqueeze(-1).float()).sum(dim=2)  # fp32

        # Phase B̄ for Z (serial scan in PyTorch — cheap)
        total_dz_fwd = torch.empty_like(dz_C)
        total_dz_fwd[:, F - 1] = dz_C[:, F - 1]
        for f in range(F - 2, -1, -1):
            gnext = decay_bhf[:, f + 1].view(BH, 1)
            I_minus_P_next = I_D - P_z_all[:, f + 1]
            total_dz_fwd[:, f] = dz_C[:, f] + gnext * (
                I_minus_P_next.transpose(-2, -1) @ total_dz_fwd[:, f + 1].unsqueeze(-1)
            ).squeeze(-1)
        total_dz_rev = torch.empty_like(dz_C)
        total_dz_rev[:, 0] = dz_C[:, 0]
        for f in range(F - 1):
            gnext = decay_bhf[:, f + 1].view(BH, 1)
            I_minus_P_next = I_D - P_z_all[:, f + 1]
            total_dz_rev[:, f + 1] = dz_C[:, f + 1] + gnext * (
                I_minus_P_next.transpose(-2, -1) @ total_dz_rev[:, f].unsqueeze(-1)
            ).squeeze(-1)

        dB_z = total_dz_fwd.clone()
        dB_z[:, 1:] += total_dz_rev[:, : F - 1]
        z_fwd_prev = z_fwd_full[:, :-1].contiguous()
        z_rev_at = z_rev_full[:, :F].contiguous()
        g_per = decay_bhf.view(BH, F, 1, 1)
        dP_z_fwd = -g_per * (total_dz_fwd.unsqueeze(-1) @ z_fwd_prev.unsqueeze(-2))
        dP_z_rev = torch.zeros_like(dP_z_fwd)
        dP_z_rev[:, 1:] = -g_per[:, 1:] * (total_dz_rev[:, : F - 1].unsqueeze(-1) @ z_rev_at[:, 1:].unsqueeze(-2))
        dP_z_total = dP_z_fwd + dP_z_rev

        I_minus_P_z = I_D - P_z_all
        dg_z = (total_dz_fwd * (I_minus_P_z @ z_fwd_prev.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
        dg_z_rev_part = torch.zeros_like(dg_z)
        dg_z_rev_part[:, 1:] = (
            total_dz_rev[:, : F - 1] * (I_minus_P_z[:, 1:] @ z_rev_at[:, 1:].unsqueeze(-1)).squeeze(-1)
        ).sum(dim=-1)
        dg_z_total = dg_z + dg_z_rev_part

        dK_z, dbeta_z = phase_a_z_bwd(
            K_z_bhfsd.contiguous(), beta_bhfs, dB_z, dP_z_total, D, BLOCK_S=BLOCK_S, dot_precision=dot_precision
        )

        # ──── 6. Combine KV + Z, undo RoPE, undo ReLU, reshape to BNHD for RMSNorm ────
        # Work in BHFSD throughout; reshape to BNHD only at the end for RMSNorm backward.
        def _unrope_bhfsd(dY, cos_fs, sin_fs):
            Dd = dY.shape[-1]
            sin_scaled = dY * sin_fs[None, :, :, :]
            sin_scaled_pair = sin_scaled.reshape(*sin_scaled.shape[:-1], Dd // 2, 2).flip(-1).reshape(*sin_scaled.shape)
            return dY * cos_fs[None, :, :, :] + sin_scaled_pair

        # Fused unrope + add dz + relu mask (op #13): replaces 6 PyTorch ops
        # per direction. Closes ~17% of bwd time on H100 F=11.
        dQ_normed_bhfsd, dK_normed_bhfsd = fused_rope_unrope_bwd(
            dQ_kv,
            dK_kv,
            dQ_z,
            dK_z,
            Q_post_relu_bhfsd,
            K_post_relu_bhfsd,
            rope_cos,
            rope_sin,
            k_scale,
            F,
            S,
        )
        del dQ_kv, dK_kv, dQ_z, dK_z, Q_post_relu_bhfsd, K_post_relu_bhfsd

        # Reshape BHFSD → BNHD once at the end.
        dQ_normed_bnhd = bhfsd_to_bnhd(dQ_normed_bhfsd)
        del dQ_normed_bhfsd
        dK_normed_bnhd = bhfsd_to_bnhd(dK_normed_bhfsd)
        del dK_normed_bhfsd
        dV_bnhd = bhfsd_to_bnhd(dV)
        del dV
        # Match input beta's shape (B, H, F, S) — earlier `B, H, F*S` flattened
        # the last two dims and tripped autograd's gradient-shape check.
        dbeta_total = (dbeta_kv + dbeta_z).reshape(B, H, F, S)
        del dbeta_kv, dbeta_z
        ddecay_total = (dg_kv_total + dg_z_total).reshape(B, H, F)
        del dg_kv_total, dg_z_total

        # RMSNorm backward: d/dx = inv_rms*w*d/dy - (inv_rms^3/C) * x * Σ(w*d/dy*x)
        # Use fp32 for math; q_raw_v is kept as original (bf16) and upcasted inline.
        q_raw_f = q_raw_v.float()
        q_irms = q_inv_rms[:, :, None, None]
        gw_q = dQ_normed_bnhd * q_nw_hd[None, None]
        dq_nw = (dQ_normed_bnhd * q_raw_f * q_irms).sum(dim=(0, 1)).reshape(-1)
        corr_q = (gw_q * q_raw_f).sum(dim=(-2, -1), keepdim=True)
        dQ_raw = q_irms * gw_q - (q_irms**3) / C * q_raw_f * corr_q
        del dQ_normed_bnhd, gw_q, corr_q, q_raw_f

        k_raw_f = k_raw_v.float()
        k_irms = k_inv_rms[:, :, None, None]
        gw_k = dK_normed_bnhd * k_nw_hd[None, None]
        dk_nw = (dK_normed_bnhd * k_raw_f * k_irms).sum(dim=(0, 1)).reshape(-1)
        corr_k = (gw_k * k_raw_f).sum(dim=(-2, -1), keepdim=True)
        dK_raw = k_irms * gw_k - (k_irms**3) / C * k_raw_f * corr_k
        del dK_normed_bnhd, gw_k, corr_k, k_raw_f

        dqkv = torch.stack([dQ_raw.to(dtype), dK_raw.to(dtype), dV_bnhd.to(dtype)], dim=2)

        return (
            dqkv,
            dbeta_total.to(beta.dtype),
            ddecay_total.to(decay.dtype),
            dq_nw.to(q_norm_weight.dtype),
            dk_nw.to(k_norm_weight.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_bigdn_chunkwise_autograd(
    qkv,
    beta,
    decay,
    q_norm_weight,
    k_norm_weight,
    rope_cos,
    rope_sin,
    F,
    S,
    k_scale=1.0,
    norm_eps=1e-5,
    eps=1e-6,
    dot_precision=0,
    BLOCK_S=64,
):
    """BiGDN chunkwise forward + chunkwise backward with full autograd support."""
    return FusedBiGDNChunkwiseFunction.apply(
        qkv,
        beta,
        decay,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        F,
        S,
        k_scale,
        norm_eps,
        eps,
        dot_precision,
        BLOCK_S,
    )
