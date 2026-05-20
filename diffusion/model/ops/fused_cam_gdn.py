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

"""Triton-fused camera-branch UCPE single-path delta-rule prep.

Companion to :mod:`diffusion.model.ops.fused_gdn` (main GDN branch). This
module hosts the inference-path prep pipeline for the *camera* branch of
:class:`ChunkCausalGDNUCPESinglePathLiteLA`.

``_cam_prep_kernel`` fuses, per ``(batch, token, head)``, the RMSNorm over
the full ``C`` channels, ReLU on Q/K, K-scale on K, the UCPE 4x4
block-diagonal projection matrix on the first ``D/2`` dims, and the
interleaved-pair complex RoPE on the second ``D/2`` dims.  Q, K, V are
processed in one pass.  The kernel also emits per-token pre-UCPE and
post-UCPE ``||k||^2`` so the caller can compute the inflation-squared
factor used for Dynamic Beta Discounting.

V skips RMSNorm / ReLU / K-scale but receives the same UCPE 4x4 + RoPE
transforms as K (``apply_fn_kv`` in production). The short convolution on
K and the inverse UCPE output transform (``apply_fn_o``) stay in PyTorch.

The camera-branch *scan* runs through ``cam_scan_bidi_chunkwise`` in
``fused_gdn_chunkwise``.
"""

# ruff: noqa: E501

from __future__ import annotations

import torch
import triton
import triton.language as tl

from diffusion.model.nets.sana_camctrl_blocks import (
    compute_fov_from_fx_xi,
    ucm_unproject_grid_fov,
    world_to_ray_mats,
)

# =============================================================================
# Scalar helpers
# =============================================================================


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix batch (closed-form).

    Mirrors the production ``_invert_SE3`` in ``sana_camctrl_blocks.py``;
    inlined to keep this module dependency-light.
    """
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _process_camera_conditions_raymats_only(
    camera_conditions: torch.Tensor,
    B: int,
    HW: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Lightweight variant of ``_process_camera_conditions_ucpe`` — raymats only.

    Computes *only* the per-ray ``world -> ray_local`` SE(3) transforms used
    by UCPE single-path.  Skips the ``compute_up_lat_map`` path (absmap) that
    the cam branch never consumes — that saves ~1 ms per block on H100.

    Args:
        camera_conditions: ``(B, F, 20)`` — ``[c2w_16 | fx | fy | cx | cy]``.
        B: Batch size (redundant with ``camera_conditions.shape[0]``; kept
            for parity with the production signature).
        HW: ``(T_latent, H_latent, W_latent)`` from the caller.
        patch_size: ``(pt, ph, pw)`` patch embedding stride.

    Returns:
        ``raymats`` of shape ``(B, F, H_latent, W_latent, 4, 4)``.
    """
    F_dim = camera_conditions.shape[1]
    c2w_flat = camera_conditions[..., :16]
    C_to_W = c2w_flat.view(B, F_dim, 4, 4)

    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]
    H_dim, W_dim = HW[1], HW[2]
    image_width = W_dim * patch_size[2]
    image_height = H_dim * patch_size[1]

    xi = torch.zeros(
        (B, F_dim),
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    )
    x_fov = compute_fov_from_fx_xi(
        fx,
        xi,
        image_width,
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    ).view(B, F_dim)
    y_fov = compute_fov_from_fx_xi(
        fy,
        xi,
        image_height,
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    ).view(B, F_dim)

    d_cam = ucm_unproject_grid_fov(
        x_fov,
        y_fov,
        xi,
        H_dim,
        W_dim,
        cx / patch_size[2],
        cy / patch_size[1],
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    )
    if d_cam.ndim == 4 and d_cam.shape[0] == B * F_dim:
        d_cam = d_cam.view(B, F_dim, H_dim, W_dim, 3)

    return world_to_ray_mats(d_cam, C_to_W)  # (B, F, H, W, 4, 4)


def _precompute_cam_inv_rms(raw: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute ``1/RMS`` per ``(b, n)`` over full-``C`` channels.

    Args:
        raw: ``(B, N, H, D)`` raw QKV projection output (typically fp32).
        eps: RMSNorm epsilon.

    Returns:
        ``inv_rms`` of shape ``(B, N)`` in fp32, contiguous.
    """
    B, N, H, D = raw.shape
    C = H * D
    sq_sum = (raw.float() * raw.float()).sum(dim=(-1, -2))  # (B, N)
    return torch.rsqrt(sq_sum / C + eps).contiguous()


def _prepare_ucpe_rope_tables(
    rotary_emb_cam: torch.Tensor,
    N: int,
    D_half: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert complex RoPE ``(1, 1, N, D_half//2)`` to interleaved ``(N, D_half)`` cos/sin.

    Uses the interleaved-pair convention:
        y[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        y[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
    encoded as  ``y[d] = x[d]*cos_exp[d] + x[d^1]*sin_exp[d]`` with
        sin_exp[2i] = -sin[i], sin_exp[2i+1] = +sin[i].
    """
    del device  # all outputs inherit device from freqs
    freqs = rotary_emb_cam.squeeze(0).squeeze(0)  # (N, D_half//2) complex
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1).contiguous()
    rope_sin = torch.stack([-sin_half, sin_half], dim=-1).reshape(N, D_half).contiguous()
    return rope_cos, rope_sin


# =============================================================================
# Triton kernels — lifted verbatim from cam_gdn_playground.py::TritonCamBranch
# =============================================================================


_DEFAULT_BLOCK_S = 64


@triton.jit
def _cam_prep_kernel(
    q_raw_ptr,  # (B, N, H, D) contiguous, any fp dtype
    k_raw_ptr,  # (B, N, H, D) contiguous (post short-conv on K)
    v_raw_ptr,  # (B, N, H, D) contiguous
    q_inv_rms_ptr,  # (B, N) float32 — precomputed over full C channels
    k_inv_rms_ptr,  # (B, N) float32
    q_norm_w_ptr,  # (C,) = (H*D,) float32
    k_norm_w_ptr,  # (C,) float32
    proj_q_ptr,  # (B, N, 4, 4) — applied to Q first D/2 dims (P_T)
    proj_kv_ptr,  # (B, N, 4, 4) — applied to K,V first D/2 dims (P_inv)
    rope_cos_ptr,  # (N, D_rope) float32, D_rope = D//2
    rope_sin_ptr,  # (N, D_rope) float32
    # --- outputs in (B, H, D, N) layout, same strides pattern ---
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    k_pre_norm_sq_ptr,  # (B, H, N) float32 — ||k_pre_ucpe||^2
    k_post_norm_sq_ptr,  # (B, H, N) float32 — ||k_post_ucpe||^2
    # --- dims ---
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,  # head dim
    D_HALF: tl.constexpr,  # D // 2
    N_GROUPS: tl.constexpr,  # D_HALF // 4
    K_SCALE,
    # --- tile sizes ---
    BLOCK_D_ROPE: tl.constexpr,  # next pow2 of D_HALF (rope block)
    BLOCK_GROUPS: tl.constexpr,  # next pow2 of N_GROUPS
):
    """One program per (b, n, h) — processes a single (Q, K, V) head slice.

    Loads the first D_HALF dims as a (N_GROUPS, 4) tile (for the UCPE
    block-diagonal 4x4 projmat), and the second D_HALF dims as a
    (D_HALF,) vector (for RoPE). No redundant loads.
    """
    pid = tl.program_id(0)
    h_idx = pid % H
    bn_idx = pid // H
    b_idx = bn_idx // N
    n_idx = bn_idx % N

    # layout (B, N, H, D) contiguous
    row_base = b_idx * (N * H * D) + n_idx * (H * D) + h_idx * D
    nw_off = h_idx * D

    # ---- load inv-RMS (scalar, shared across heads for this token) ----
    q_inv_rms = tl.load(q_inv_rms_ptr + bn_idx).to(tl.float32)
    k_inv_rms = tl.load(k_inv_rms_ptr + bn_idx).to(tl.float32)

    # ---- load per-token P matrices (4,4) shared across heads ----
    proj_base = (b_idx * N + n_idx) * 16
    offs_i = tl.arange(0, 4)
    offs_j = tl.arange(0, 4)
    P_q = tl.load(proj_q_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]).to(tl.float32)
    P_kv = tl.load(proj_kv_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]).to(tl.float32)

    # ==================================================================
    # Pass 1 — UCPE block-diagonal projmat on first D_HALF dims
    # ==================================================================
    offs_g = tl.arange(0, BLOCK_GROUPS)
    mask_g = offs_g < N_GROUPS
    offs_gj = offs_g[:, None] * 4 + offs_j[None, :]  # (BLOCK_GROUPS, 4)
    mask_gj = mask_g[:, None]

    q_half = tl.load(q_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)
    k_half = tl.load(k_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)
    v_half = tl.load(v_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)

    q_nw_half = tl.load(q_norm_w_ptr + nw_off + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)
    k_nw_half = tl.load(k_norm_w_ptr + nw_off + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)

    q_half = q_half * q_inv_rms * q_nw_half
    q_half = tl.where(q_half > 0, q_half, 0.0)

    k_half = k_half * k_inv_rms * k_nw_half
    k_half = tl.where(k_half > 0, k_half, 0.0) * K_SCALE

    # Pre-UCPE ||k||^2 contribution from first half
    k_half_masked = tl.where(mask_gj, k_half, 0.0)
    k_pre_half_sq = tl.sum(k_half_masked * k_half_masked)

    # Apply 4x4 projmat: out[g, i] = sum_j P[i, j] * in[g, j]
    # (BLOCK_GROUPS, 1, 4) * (1, 4, 4) -> (BLOCK_GROUPS, 4, 4), sum axis=-1
    q_half_out = tl.sum(q_half[:, None, :] * P_q[None, :, :], axis=-1)
    k_half_out = tl.sum(k_half[:, None, :] * P_kv[None, :, :], axis=-1)
    v_half_out = tl.sum(v_half[:, None, :] * P_kv[None, :, :], axis=-1)

    # Post-UCPE ||k||^2 contribution from first half
    k_half_out_masked = tl.where(mask_gj, k_half_out, 0.0)
    k_post_half_sq = tl.sum(k_half_out_masked * k_half_out_masked)

    # ==================================================================
    # Pass 2 — RoPE on second D_HALF dims
    # ==================================================================
    offs_r = tl.arange(0, BLOCK_D_ROPE)
    mask_r = offs_r < D_HALF
    offs_r_pair = offs_r ^ 1
    mask_r_pair = offs_r_pair < D_HALF

    rope_row = n_idx * D_HALF
    cos_v = tl.load(rope_cos_ptr + rope_row + offs_r, mask=mask_r, other=1.0).to(tl.float32)
    sin_v = tl.load(rope_sin_ptr + rope_row + offs_r, mask=mask_r, other=0.0).to(tl.float32)

    # Load second-half raw values and their pair partners
    rope_base = row_base + D_HALF
    q_r = tl.load(q_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    k_r = tl.load(k_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    v_r = tl.load(v_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    q_r_pair = tl.load(q_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)
    k_r_pair = tl.load(k_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)
    v_r_pair = tl.load(v_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)

    q_nw_r = tl.load(q_norm_w_ptr + nw_off + D_HALF + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    k_nw_r = tl.load(k_norm_w_ptr + nw_off + D_HALF + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    q_nw_r_pair = tl.load(q_norm_w_ptr + nw_off + D_HALF + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)
    k_nw_r_pair = tl.load(k_norm_w_ptr + nw_off + D_HALF + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)

    q_r_n = q_r * q_inv_rms * q_nw_r
    q_r_n = tl.where(q_r_n > 0, q_r_n, 0.0)
    q_r_pair_n = q_r_pair * q_inv_rms * q_nw_r_pair
    q_r_pair_n = tl.where(q_r_pair_n > 0, q_r_pair_n, 0.0)

    k_r_n = k_r * k_inv_rms * k_nw_r
    k_r_n = tl.where(k_r_n > 0, k_r_n, 0.0) * K_SCALE
    k_r_pair_n = k_r_pair * k_inv_rms * k_nw_r_pair
    k_r_pair_n = tl.where(k_r_pair_n > 0, k_r_pair_n, 0.0) * K_SCALE

    # Pre-UCPE ||k||^2 contribution from second half (using post-ReLU/scale k_r_n)
    k_r_n_masked = tl.where(mask_r, k_r_n, 0.0)
    k_pre_rope_sq = tl.sum(k_r_n_masked * k_r_n_masked)

    q_rope_out = q_r_n * cos_v + q_r_pair_n * sin_v
    k_rope_out = k_r_n * cos_v + k_r_pair_n * sin_v
    v_rope_out = v_r * cos_v + v_r_pair * sin_v

    # Post-UCPE ||k||^2 contribution from second half
    k_rope_masked = tl.where(mask_r, k_rope_out, 0.0)
    k_post_rope_sq = tl.sum(k_rope_masked * k_rope_masked)

    # Store scalar per-token norm squares
    norm_out_idx = (b_idx * H + h_idx) * N + n_idx
    tl.store(k_pre_norm_sq_ptr + norm_out_idx, k_pre_half_sq + k_pre_rope_sq)
    tl.store(k_post_norm_sq_ptr + norm_out_idx, k_post_half_sq + k_post_rope_sq)

    # ==================================================================
    # Store outputs in (B, H, D, N) layout: ptr[b, h, d, n] = base_bh + d*N + n
    # ==================================================================
    out_base = b_idx * (H * D * N) + h_idx * (D * N) + n_idx

    # First half: d = g*4 + i, write at out_base + d*N (strided by N).
    offs_d_half = offs_g[:, None] * 4 + offs_i[None, :]  # (BLOCK_GROUPS, 4)
    mask_d_half = mask_g[:, None]
    tl.store(q_out_ptr + out_base + offs_d_half * N, q_half_out, mask=mask_d_half)
    tl.store(k_out_ptr + out_base + offs_d_half * N, k_half_out, mask=mask_d_half)
    tl.store(v_out_ptr + out_base + offs_d_half * N, v_half_out, mask=mask_d_half)

    # Second half (RoPE region): d = D_HALF + r
    offs_d_r = D_HALF + offs_r  # (BLOCK_D_ROPE,)
    tl.store(q_out_ptr + out_base + offs_d_r * N, q_rope_out, mask=mask_r)
    tl.store(k_out_ptr + out_base + offs_d_r * N, k_rope_out, mask=mask_r)
    tl.store(v_out_ptr + out_base + offs_d_r * N, v_rope_out, mask=mask_r)


def cam_prep_func(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,  # (B, N, 4, 4)
    proj_kv: torch.Tensor,  # (B, N, 4, 4)
    rope_cos: torch.Tensor,  # (N, D//2)
    rope_sin: torch.Tensor,  # (N, D//2)
    k_scale: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + ReLU + (K-scale on K) + UCPE 4x4 + RoPE for the cam branch.

    Args:
        q_raw, k_raw, v_raw: ``(B, N, H, D)`` contiguous (any fp dtype).
            ``K`` must already have the short convolution applied.
        q_norm_weight, k_norm_weight: ``(C,) = (H*D,)`` fp32.
        proj_q, proj_kv: ``(B, N, 4, 4)`` fp32 (``P_T`` and ``P_inv`` in UCPE).
        rope_cos, rope_sin: ``(N, D//2)`` fp32 interleaved-pair tables.
        k_scale: ``(D^-0.5) * (S^-0.5)``.
        norm_eps: RMSNorm epsilon.

    Returns:
        q_trans, k_trans, v_trans: ``(B, H, D, N)`` same dtype as ``q_raw``.
        inflation_sq: ``(B, H, N)`` fp32, ratio
            ``(||k_post_ucpe|| / ||k_pre_ucpe||)^2`` per token/head.
    """
    B, N, H, D = q_raw.shape
    assert k_raw.shape == q_raw.shape and v_raw.shape == q_raw.shape
    assert D % 2 == 0 and (D // 2) % 4 == 0, f"D={D} must be 2x and (D/2) % 4 == 0"
    D_half = D // 2
    N_groups = D_half // 4

    assert q_raw.is_contiguous() and k_raw.is_contiguous() and v_raw.is_contiguous()
    assert proj_q.shape == (B, N, 4, 4) and proj_q.is_contiguous()
    assert proj_kv.shape == (B, N, 4, 4) and proj_kv.is_contiguous()
    assert rope_cos.shape == (N, D_half) and rope_cos.is_contiguous()
    assert rope_sin.shape == (N, D_half) and rope_sin.is_contiguous()
    assert q_norm_weight.numel() == H * D and q_norm_weight.dtype == torch.float32
    assert k_norm_weight.numel() == H * D and k_norm_weight.dtype == torch.float32

    # Precompute inv-RMS over full C channels (shared across heads per token).
    q_inv_rms = _precompute_cam_inv_rms(q_raw, norm_eps)
    k_inv_rms = _precompute_cam_inv_rms(k_raw, norm_eps)

    out_dtype = q_raw.dtype
    q_out = torch.empty(B, H, D, N, dtype=out_dtype, device=q_raw.device)
    k_out = torch.empty(B, H, D, N, dtype=out_dtype, device=q_raw.device)
    v_out = torch.empty(B, H, D, N, dtype=out_dtype, device=q_raw.device)
    k_pre_sq = torch.empty(B, H, N, dtype=torch.float32, device=q_raw.device)
    k_post_sq = torch.empty(B, H, N, dtype=torch.float32, device=q_raw.device)

    BLOCK_D_ROPE = triton.next_power_of_2(D_half)
    BLOCK_GROUPS = triton.next_power_of_2(N_groups)

    grid = (B * N * H,)
    _cam_prep_kernel[grid](
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        proj_q,
        proj_kv,
        rope_cos,
        rope_sin,
        q_out,
        k_out,
        v_out,
        k_pre_sq,
        k_post_sq,
        H=H,
        N=N,
        D=D,
        D_HALF=D_half,
        N_GROUPS=N_groups,
        K_SCALE=k_scale,
        BLOCK_D_ROPE=BLOCK_D_ROPE,
        BLOCK_GROUPS=BLOCK_GROUPS,
        num_warps=1,
    )
    # inflation_sq = (clamp(sqrt(post), 1e-6) / clamp(sqrt(pre), 1e-6))^2
    #              = clamp(post, 1e-12) / clamp(pre, 1e-12)  (equivalent).
    inflation_sq = k_post_sq.clamp_min(1e-12) / k_pre_sq.clamp_min(1e-12)
    return q_out, k_out, v_out, inflation_sq
