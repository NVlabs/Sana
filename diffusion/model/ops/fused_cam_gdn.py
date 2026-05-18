"""Triton-fused camera-branch UCPE single-path delta rule.

Companion to :mod:`diffusion.model.ops.fused_gdn` (main GDN
branch).  This module fuses the *camera* branch of
:class:`ChunkCausalGDNUCPESinglePathLiteLA` through two Triton kernels:

1. ``_cam_prep_kernel`` — fuses, per ``(batch, token, head)``, the RMSNorm
   over the full ``C`` channels, ReLU on Q/K, K-scale on K, the UCPE 4x4
   block-diagonal projection matrix on the first ``D/2`` dims, and the
   interleaved-pair complex RoPE on the second ``D/2`` dims.  Q, K, V are
   processed in one pass.  The kernel also emits per-token pre-UCPE and
   post-UCPE ``||k||^2`` so the caller can compute the inflation-squared
   factor used for Dynamic Beta Discounting.

2. ``_cam_scan_kernel`` — fuses the numerator-only single-path delta-rule
   scan per ``(batch, head)``.  ``REVERSE=1`` implements the
   ``flip_and_shift`` backward pass semantics directly, avoiding the
   torch-side flips in the per-chunk backward loop.

Notes:
    - V skips RMSNorm / ReLU / K-scale but receives the same UCPE 4x4 +
      RoPE transforms as K (apply_fn_kv in production).
    - The short convolution on K and the inverse UCPE output transform
      (``apply_fn_o``) stay in PyTorch — they are single lightweight ops
      not on the critical path.
"""

# ruff: noqa: E501

from __future__ import annotations

import os

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


@triton.jit
def _cam_prep_bwd_kernel(
    # --- forward inputs (replayed for ReLU mask + k_post_kscale recompute) ---
    q_raw_ptr,  # (B, N, H, D) contiguous, any fp dtype
    k_raw_ptr,  # (B, N, H, D) contiguous (post short-conv on K)
    q_norm_w_ptr,  # (C,) = (H*D,) float32
    k_norm_w_ptr,  # (C,) float32
    q_inv_rms_ptr,  # (B, N) float32 — saved from forward
    k_inv_rms_ptr,  # (B, N) float32
    proj_q_ptr,  # (B, N, 4, 4) — applied to Q first D/2 dims (P_T)
    proj_kv_ptr,  # (B, N, 4, 4) — applied to K,V first D/2 dims (P_inv)
    rope_cos_ptr,  # (N, D_rope) float32, D_rope = D//2
    rope_sin_ptr,  # (N, D_rope) float32
    # --- upstream gradients (B, H, D, N) layout matching forward outputs ---
    d_q_out_ptr,  # grad of q_out (any dtype, cast to fp32 on load)
    eff_d_k_out_ptr,  # grad of k_out + inflation_sq contribution through k_out (fp32)
    d_v_out_ptr,  # grad of v_out
    # --- inflation_sq direct grad to k_post_kscale^2 sum (B, H, N) fp32 ---
    d_pre_k_sq_ptr,
    # --- outputs ---
    d_q_post_norm_ptr,  # (B, N, H, D) fp32 — grad after RoPE^T+UCPE^T+Kscale+ReLU; consumed by torch RMSNorm bwd
    d_k_post_norm_ptr,  # (B, N, H, D) fp32 — same for K
    dv_raw_ptr,  # (B, N, H, D) fp32 — final dv_raw (V skips norm/ReLU/Kscale)
    # --- dims ---
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    D_HALF: tl.constexpr,
    N_GROUPS: tl.constexpr,
    K_SCALE,
    # --- tile sizes ---
    BLOCK_D_ROPE: tl.constexpr,  # next pow2 of D_HALF (rope block)
    BLOCK_GROUPS: tl.constexpr,  # next pow2 of N_GROUPS
):
    """One program per (b, n, h) — matches the forward kernel's parallelism.

    Implements the bwd of the fused fwd:
        forward order:  RMSNorm -> ReLU -> [K-scale on K] -> UCPE (first D/2)
                        -> RoPE (second D/2) -> output (B, H, D, N).
        backward order: RoPE^T (second D/2) -> UCPE^T (first D/2)
                        -> K-scale (only K) -> ReLU mask -> emit d_post_norm
                        intermediates for the cross-head RMSNorm bwd handled
                        outside (in :func:`_cam_prep_bwd_dispatch`).

    The full-channel RMSNorm bwd's outer-product term and per-channel weight
    grad both require a sum over ``H*D`` (the full ``C``) per token, which
    couples heads. We deliberately leave that step in PyTorch (a couple of
    fused element-wise + reduction ops) — see :func:`_cam_prep_bwd_dispatch`.

    Inflation handling: the kernel takes an *effective* ``dO_k`` that already
    includes the contribution from ``grad_inflation_sq`` flowing through
    ``k_out`` (i.e. ``eff_dO_k = grad_k + 2 * k_out * d_post_k_sq``), plus a
    per-(b, h, n) scalar ``d_pre_k_sq`` that is the chain-rule contribution
    of ``grad_inflation_sq`` into the pre-UCPE ``||k_post_kscale||^2`` sum.
    Inside the kernel we recompute ``k_post_kscale`` (= post-norm * ReLU *
    K_SCALE) and add ``2 * k_post_kscale[d] * d_pre_k_sq`` as a direct
    contribution to ``d_k_post_kscale``.
    """
    pid = tl.program_id(0)
    h_idx = pid % H
    bn_idx = pid // H
    b_idx = bn_idx // N
    n_idx = bn_idx % N

    # ---- load saved scalars: inv-RMS (per b, n) and d_pre_k_sq (per b, h, n) ----
    q_inv_rms = tl.load(q_inv_rms_ptr + bn_idx).to(tl.float32)
    k_inv_rms = tl.load(k_inv_rms_ptr + bn_idx).to(tl.float32)
    bhn_idx = (b_idx * H + h_idx) * N + n_idx
    d_pre_k_sq = tl.load(d_pre_k_sq_ptr + bhn_idx).to(tl.float32)

    # ---- load per-token P matrices (shared across heads) ----
    proj_base = (b_idx * N + n_idx) * 16
    offs_i = tl.arange(0, 4)
    offs_j = tl.arange(0, 4)
    P_q = tl.load(proj_q_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]).to(tl.float32)
    P_kv = tl.load(proj_kv_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]).to(tl.float32)

    # ---- layout offsets ----
    row_base = b_idx * (N * H * D) + n_idx * (H * D) + h_idx * D  # (B, N, H, D)
    nw_off = h_idx * D
    out_base_BHDN = b_idx * (H * D * N) + h_idx * (D * N) + n_idx  # (B, H, D, N) for dO_*
    norm_base = row_base  # d_*_post_norm and dv_raw share the (B, N, H, D) layout

    # ============================================================
    # First half — UCPE region
    # ============================================================
    offs_g = tl.arange(0, BLOCK_GROUPS)
    mask_g = offs_g < N_GROUPS
    offs_gj = offs_g[:, None] * 4 + offs_j[None, :]  # (BLOCK_GROUPS, 4)
    mask_gj = mask_g[:, None]

    # Recompute post-norm (pre-ReLU) Q/K for the ReLU mask + k_post_kscale.
    q_half_raw = tl.load(q_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)
    k_half_raw = tl.load(k_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)
    q_nw_half = tl.load(q_norm_w_ptr + nw_off + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)
    k_nw_half = tl.load(k_norm_w_ptr + nw_off + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)

    q_post_norm_half = q_half_raw * q_inv_rms * q_nw_half
    k_post_norm_half = k_half_raw * k_inv_rms * k_nw_half
    q_relu_mask_half = q_post_norm_half > 0
    k_relu_mask_half = k_post_norm_half > 0

    # k_post_kscale = relu(k_post_norm) * K_SCALE  (used for direct inflation contribution)
    k_post_relu_half = tl.where(k_relu_mask_half, k_post_norm_half, 0.0)
    k_post_kscale_half = k_post_relu_half * K_SCALE

    # Load upstream gradients for first half.
    # In (B, H, D, N): ptr[b, h, d, n] = out_base_BHDN + d * N. d = g*4 + i.
    offs_d_half = offs_g[:, None] * 4 + offs_i[None, :]  # (BLOCK_GROUPS, 4)
    mask_d_half = mask_g[:, None]
    dO_q_half = tl.load(d_q_out_ptr + out_base_BHDN + offs_d_half * N, mask=mask_d_half, other=0.0).to(tl.float32)
    dO_k_eff_half = tl.load(eff_d_k_out_ptr + out_base_BHDN + offs_d_half * N, mask=mask_d_half, other=0.0).to(
        tl.float32
    )
    dO_v_half = tl.load(d_v_out_ptr + out_base_BHDN + offs_d_half * N, mask=mask_d_half, other=0.0).to(tl.float32)

    # UCPE^T: din[g, j] = sum_i P[i, j] * dout[g, i]
    # forward: out[g, i] = sum_j q[g, j] * P[i, j]  -- so bwd sums over i
    d_q_post_relu_half = tl.sum(dO_q_half[:, :, None] * P_q[None, :, :], axis=1)
    d_k_post_kscale_via_ucpe_half = tl.sum(dO_k_eff_half[:, :, None] * P_kv[None, :, :], axis=1)
    d_v_first_half = tl.sum(dO_v_half[:, :, None] * P_kv[None, :, :], axis=1)

    # K direct inflation contribution: 2 * k_post_kscale * d_pre_k_sq
    d_k_post_kscale_half = d_k_post_kscale_via_ucpe_half + 2.0 * k_post_kscale_half * d_pre_k_sq

    # K-scale bwd (multiply by K_SCALE)
    d_k_post_relu_half = d_k_post_kscale_half * K_SCALE

    # ReLU mask
    d_q_post_norm_half = tl.where(q_relu_mask_half, d_q_post_relu_half, 0.0)
    d_k_post_norm_half = tl.where(k_relu_mask_half, d_k_post_relu_half, 0.0)

    # Mask out-of-bounds groups to 0 explicitly (for safety on uneven N_GROUPS).
    d_q_post_norm_half = tl.where(mask_d_half, d_q_post_norm_half, 0.0)
    d_k_post_norm_half = tl.where(mask_d_half, d_k_post_norm_half, 0.0)
    d_v_first_half = tl.where(mask_d_half, d_v_first_half, 0.0)

    # Store at (B, N, H, D), d = g*4+i
    tl.store(d_q_post_norm_ptr + norm_base + offs_d_half, d_q_post_norm_half, mask=mask_d_half)
    tl.store(d_k_post_norm_ptr + norm_base + offs_d_half, d_k_post_norm_half, mask=mask_d_half)
    tl.store(dv_raw_ptr + norm_base + offs_d_half, d_v_first_half, mask=mask_d_half)

    # ============================================================
    # Second half — RoPE region
    # ============================================================
    offs_r = tl.arange(0, BLOCK_D_ROPE)
    mask_r = offs_r < D_HALF
    offs_r_pair = offs_r ^ 1
    mask_r_pair = offs_r_pair < D_HALF

    rope_row = n_idx * D_HALF
    cos_v = tl.load(rope_cos_ptr + rope_row + offs_r, mask=mask_r, other=1.0).to(tl.float32)
    sin_v_pair = tl.load(rope_sin_ptr + rope_row + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)

    # Recompute post-norm (pre-ReLU) Q/K for the ReLU mask.
    rope_base = row_base + D_HALF
    q_r_raw = tl.load(q_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    k_r_raw = tl.load(k_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    q_nw_r = tl.load(q_norm_w_ptr + nw_off + D_HALF + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    k_nw_r = tl.load(k_norm_w_ptr + nw_off + D_HALF + offs_r, mask=mask_r, other=0.0).to(tl.float32)

    q_post_norm_r = q_r_raw * q_inv_rms * q_nw_r
    k_post_norm_r = k_r_raw * k_inv_rms * k_nw_r
    q_relu_mask_r = q_post_norm_r > 0
    k_relu_mask_r = k_post_norm_r > 0

    k_post_relu_r = tl.where(k_relu_mask_r, k_post_norm_r, 0.0)
    k_post_kscale_r = k_post_relu_r * K_SCALE

    # Load upstream gradients (second half) — direct + pair.
    offs_d_r = D_HALF + offs_r
    offs_d_r_pair = D_HALF + offs_r_pair
    dO_q_r = tl.load(d_q_out_ptr + out_base_BHDN + offs_d_r * N, mask=mask_r, other=0.0).to(tl.float32)
    dO_k_eff_r = tl.load(eff_d_k_out_ptr + out_base_BHDN + offs_d_r * N, mask=mask_r, other=0.0).to(tl.float32)
    dO_v_r = tl.load(d_v_out_ptr + out_base_BHDN + offs_d_r * N, mask=mask_r, other=0.0).to(tl.float32)
    dO_q_r_pair = tl.load(d_q_out_ptr + out_base_BHDN + offs_d_r_pair * N, mask=mask_r_pair, other=0.0).to(tl.float32)
    dO_k_eff_r_pair = tl.load(eff_d_k_out_ptr + out_base_BHDN + offs_d_r_pair * N, mask=mask_r_pair, other=0.0).to(
        tl.float32
    )
    dO_v_r_pair = tl.load(d_v_out_ptr + out_base_BHDN + offs_d_r_pair * N, mask=mask_r_pair, other=0.0).to(tl.float32)

    # RoPE^T:  forward y[r] = x[r]*cos[r] + x[r^1]*sin[r]
    #          bwd     dx[r] = dy[r]*cos[r] + dy[r^1]*sin[r^1]
    d_q_post_relu_r = dO_q_r * cos_v + dO_q_r_pair * sin_v_pair
    d_k_post_kscale_via_rope_r = dO_k_eff_r * cos_v + dO_k_eff_r_pair * sin_v_pair
    d_v_second_r = dO_v_r * cos_v + dO_v_r_pair * sin_v_pair

    # K direct inflation contribution
    d_k_post_kscale_r = d_k_post_kscale_via_rope_r + 2.0 * k_post_kscale_r * d_pre_k_sq

    # K-scale bwd
    d_k_post_relu_r = d_k_post_kscale_r * K_SCALE

    # ReLU mask
    d_q_post_norm_r = tl.where(q_relu_mask_r, d_q_post_relu_r, 0.0)
    d_k_post_norm_r = tl.where(k_relu_mask_r, d_k_post_relu_r, 0.0)

    # Out-of-bound mask
    d_q_post_norm_r = tl.where(mask_r, d_q_post_norm_r, 0.0)
    d_k_post_norm_r = tl.where(mask_r, d_k_post_norm_r, 0.0)
    d_v_second_r = tl.where(mask_r, d_v_second_r, 0.0)

    norm_offs_r = D_HALF + offs_r
    tl.store(d_q_post_norm_ptr + norm_base + norm_offs_r, d_q_post_norm_r, mask=mask_r)
    tl.store(d_k_post_norm_ptr + norm_base + norm_offs_r, d_k_post_norm_r, mask=mask_r)
    tl.store(dv_raw_ptr + norm_base + norm_offs_r, d_v_second_r, mask=mask_r)


@triton.jit
def _cam_scan_kernel(
    # --- inputs (B, H, D, N) contiguous, fp32 ---
    q_ptr,
    k_ptr,
    v_ptr,
    # --- gates ---
    beta_ptr,  # (B, H, F, S) contiguous
    decay_ptr,  # (B, H, F) contiguous
    # --- output (B, H, D, N) fp32 ---
    out_ptr,
    # --- saved state snapshots (used when SAVE_STATES=1) ---
    state_pre_ptr,  # (B, H, F, BLOCK_D, BLOCK_D) fp32 — state after decay, before update
    state_post_ptr,  # (B, H, F, BLOCK_D, BLOCK_D) fp32 — state after update
    # --- forward-direction cache state (used when LOAD_INIT_STATE / SAVE_FINAL_STATE) ---
    init_state_ptr,  # (B*H, BLOCK_D, BLOCK_D) fp32 — state at end of prefix
    final_state_ptr,  # (B*H, BLOCK_D, BLOCK_D) fp32 — state after last frame's update
    # --- dims ---
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,  # F * S
    REVERSE: tl.constexpr,
    SAVE_STATES: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,
    SAVE_FINAL_STATE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """One program per (b, h) — runs the full numerator-only delta-rule scan.

    When ``SAVE_STATES=1`` the kernel additionally writes per-frame snapshots
    of ``state_prev`` (state after applying ``decay``, before the K@delta
    update) to ``state_pre_ptr`` and ``state_curr`` (state after the update)
    to ``state_post_ptr``, both indexed by ``q_frame``. The bwd kernel
    (``_cam_scan_bwd_kernel``) consumes these snapshots for both
    ``REVERSE=0`` and ``REVERSE=1``. In ``REVERSE=1`` the slot at
    ``q_frame=F-1`` always holds the all-zero state (skip-update, decay=1
    on the zero initial state), and the bwd kernel reads exactly that —
    no special-case load is needed.

    When ``LOAD_INIT_STATE=1`` (forward direction only — wrapper enforces
    ``REVERSE=0``) the per-program ``state_curr`` is initialized from
    ``init_state_ptr`` instead of zero. The convention is: the loaded value
    is the state AT THE END of a prefix sequence (i.e., AFTER the prefix's
    last update, BEFORE any further decay applied here). On the very first
    frame, the kernel's own ``state_curr *= g`` then applies ``decay[0]``
    to this loaded state — which is exactly the decay that the global
    sequence's f=K-th frame would have applied. This matches the AR-cache
    invariant that splitting at K and resuming gives bit-identical state
    trajectories from frame K onwards.

    When ``SAVE_FINAL_STATE=1`` (forward direction only) the final
    ``state_curr`` (after the last frame's update) is written to
    ``final_state_ptr``. This is the state to be loaded with
    ``LOAD_INIT_STATE`` for a downstream segment.
    """
    pid = tl.program_id(0)
    pid_b = pid // H
    pid_h = pid % H
    bh = pid_b * H + pid_h

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    base_bh = pid_b * (H * D * N) + pid_h * (D * N)
    q_bh = q_ptr + base_bh
    k_bh = k_ptr + base_bh
    v_bh = v_ptr + base_bh
    out_bh = out_ptr + base_bh
    beta_bh = beta_ptr + bh * F * S
    decay_bh = decay_ptr + bh * F
    if SAVE_STATES:
        spre_bh = state_pre_ptr + bh * F * BLOCK_D * BLOCK_D
        spost_bh = state_post_ptr + bh * F * BLOCK_D * BLOCK_D

    # State: (D_k, D_v) in the upstream convention. Here we call rows "k-dim"
    # (input dim of state) and cols "v-dim" (output dim of state).
    if LOAD_INIT_STATE:
        init_bh = init_state_ptr + bh * BLOCK_D * BLOCK_D
        state_curr = tl.load(init_bh + offs_dd, mask=mask_dd, other=0.0).to(tl.float32)
    else:
        state_curr = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)

    for f_iter in range(F):
        if REVERSE:
            q_frame = F - 1 - f_iter
            kv_frame = F - f_iter if f_iter > 0 else 0
            skip_update = f_iter == 0
        else:
            q_frame = f_iter
            kv_frame = f_iter
            skip_update = False

        if REVERSE and f_iter == 0:
            g = 1.0
        else:
            g = tl.load(decay_bh + kv_frame).to(tl.float32)
        state_curr = state_curr * g
        state_prev = state_curr  # fp32 snapshot (same tensor, kept for clarity)

        if SAVE_STATES:
            tl.store(
                spre_bh + q_frame * BLOCK_D * BLOCK_D + offs_dd,
                state_prev,
                mask=mask_dd,
            )

        if skip_update == 0:
            kv_n_base = kv_frame * S
            f_beta = beta_bh + kv_frame * S

            for s0 in range(0, S, BLOCK_S):
                offs_s = s0 + tl.arange(0, BLOCK_S)
                mask_s = offs_s < S
                mask_sd = mask_s[:, None] & mask_d[None, :]
                n_idx = kv_n_base + offs_s

                # Load K, V tiles (BLOCK_S, BLOCK_D) from (B, H, D, N) layout:
                #   ptr[s, d] = base_bh + offs_d[d] * N + n_idx[s]
                k_ptrs = k_bh + offs_d[None, :] * N + n_idx[:, None]
                v_ptrs = v_bh + offs_d[None, :] * N + n_idx[:, None]
                K = tl.load(k_ptrs, mask=mask_sd, other=0.0)
                V = tl.load(v_ptrs, mask=mask_sd, other=0.0)

                bt = tl.load(f_beta + offs_s, mask=mask_s, other=0.0).to(tl.float32)

                # V_pred = K @ state_prev  :  (BLOCK_S, BLOCK_D)
                V_pred = tl.dot(
                    K,
                    state_prev,
                    out_dtype=tl.float32,
                    input_precision="tf32",
                )
                dv = (V - V_pred) * bt[:, None]
                state_curr += tl.dot(
                    tl.trans(K),
                    dv,
                    out_dtype=tl.float32,
                    input_precision="tf32",
                )

        if SAVE_STATES:
            tl.store(
                spost_bh + q_frame * BLOCK_D * BLOCK_D + offs_dd,
                state_curr,
                mask=mask_dd,
            )

        # --- Pass 2: output ---
        state_out = state_curr
        q_n_base = q_frame * S
        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            mask_s = offs_s < S
            mask_sd = mask_s[:, None] & mask_d[None, :]
            n_idx = q_n_base + offs_s

            q_ptrs = q_bh + offs_d[None, :] * N + n_idx[:, None]
            Q = tl.load(q_ptrs, mask=mask_sd, other=0.0)

            # num = Q @ state_out  :  (BLOCK_S, BLOCK_D), rows=S, cols=D_v
            num = tl.dot(
                Q,
                state_out,
                out_dtype=tl.float32,
                input_precision="tf32",
            )

            # Store transposed into (B, H, D, N):
            #   ptr[d, s] = out_bh + offs_d[d] * N + n_idx[s]
            out_ptrs = out_bh + offs_d[:, None] * N + n_idx[None, :]
            mask_ds = mask_d[:, None] & mask_s[None, :]
            tl.store(out_ptrs, tl.trans(num), mask=mask_ds)

    if SAVE_FINAL_STATE:
        final_bh = final_state_ptr + bh * BLOCK_D * BLOCK_D
        tl.store(final_bh + offs_dd, state_curr, mask=mask_dd)


# =============================================================================
# Backward Triton kernel
# =============================================================================


@triton.jit
def _cam_scan_bwd_kernel(
    # --- forward inputs (B, H, D, N) fp32 contiguous ---
    q_ptr,
    k_ptr,
    v_ptr,
    # --- gates ---
    beta_ptr,  # (B, H, F, S) fp32
    decay_ptr,  # (B, H, F) fp32
    # --- saved state snapshots (B*H, F, BLOCK_D, BLOCK_D) fp32, indexed by q_frame ---
    state_pre_ptr,  # state after decay, before update
    state_post_ptr,  # state after update
    # --- upstream gradient (B, H, D, N) fp32 ---
    grad_out_ptr,
    # --- output gradients ---
    dq_ptr,  # (B, H, D, N) fp32
    dk_ptr,
    dv_ptr,
    dbeta_ptr,  # (B, H, F, S) fp32
    ddecay_ptr,  # (B, H, F) fp32
    # --- dims ---
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,  # F * S
    REVERSE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Reverse-time backward of the numerator-only delta-rule scan.

    One program per ``(b, h)``. Walks the same ``f_iter`` index space as the
    forward kernel — but in reverse time order — replaying the recurrence
    using the per-``q_frame`` state snapshots saved by the forward pass with
    ``SAVE_STATES=1``. Accumulates gradients into ``dq``, ``dk``, ``dv``,
    ``dbeta``, ``ddecay``.

    Forward indexing (matched here exactly)::

        REVERSE=False:  q_frame = kv_frame = f_iter,    skip_update = False
        REVERSE=True :  q_frame = F-1-f_iter,
                        kv_frame = F-f_iter (skip when f_iter==0)
                        g = decay[kv_frame] (1.0 when f_iter==0)

    Per-iteration backward derivation (when ``not skip_update``):

        ds_post += Q.T @ d_out                         # accumulate output grad
        dQ[q]   = d_out @ s_post.T

        ddelta       = K @ ds_post                     # via K.T @ delta term
        dV[kv]       = ddelta * beta[kv,:]
        dbeta[kv,:]  = sum_d (ddelta * (V - K @ s_pre))
        dV_pred      = -ddelta * beta[kv,:]
        dK[kv]       = delta @ ds_post.T + dV_pred @ s_pre.T
        ds_pre       = ds_post + K.T @ dV_pred         # direct + V_pred path

        ddecay[kv]   = sum(ds_pre * s_pre[q]) / g    (= 0 when state_in is 0)
        ds_post[next-bwd-iter] = ds_pre * g            # propagate through decay

    For the ``skip_update`` branch (``REVERSE=True`` and ``f_iter==0``) the
    update / decay path is bypassed: ``state_post == state_pre == 0`` so
    ``dQ == 0``, ``ds_post`` is left unchanged across the iter, and
    no writes to ``dK``, ``dV``, ``dbeta`` or ``ddecay`` happen at
    ``kv_frame == 0``. Caller MUST pre-zero those output buffers (the
    dispatch wrapper uses ``torch.zeros_like``).

    The ``ddecay`` formula uses ``state_pre / g``; for ``REVERSE=False`` at
    fwd frame 0 we hardcode ``ddecay[0] = 0`` (state_in is exactly 0, but
    the division would amplify any rounding noise). For very small ``g``
    the existing ``+ 1e-12`` epsilon is matched verbatim from
    ``_fused_gdn_bwd_kernel``.
    """
    pid = tl.program_id(0)
    pid_b = pid // H
    pid_h = pid % H
    bh = pid_b * H + pid_h

    base_bh = pid_b * (H * D * N) + pid_h * (D * N)
    q_bh = q_ptr + base_bh
    k_bh = k_ptr + base_bh
    v_bh = v_ptr + base_bh
    do_bh = grad_out_ptr + base_bh
    dq_bh = dq_ptr + base_bh
    dk_bh = dk_ptr + base_bh
    dv_bh = dv_ptr + base_bh

    beta_bh = beta_ptr + bh * F * S
    decay_bh = decay_ptr + bh * F
    dbeta_bh = dbeta_ptr + bh * F * S
    ddecay_bh = ddecay_ptr + bh * F

    spre_bh = state_pre_ptr + bh * F * BLOCK_D * BLOCK_D
    spost_bh = state_post_ptr + bh * F * BLOCK_D * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    # bf16 operands for tl.dot keep shared-memory pressure manageable for large
    # BLOCK_D (e.g., 128 in production). fp32 accumulators preserve precision.
    grad_dtype = tl.bfloat16
    grad_ip: tl.constexpr = "tf32"

    # Reverse-time accumulator: gradient w.r.t. ``state_post`` for the iter
    # currently being processed.
    ds_post = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)

    for f_rev in range(F):
        # Walk fwd iters in reverse: F-1, F-2, ..., 0.
        f_iter = F - 1 - f_rev

        if REVERSE:
            q_frame = F - 1 - f_iter
            kv_frame = F - f_iter if f_iter > 0 else 0
            skip_update = f_iter == 0
        else:
            q_frame = f_iter
            kv_frame = f_iter
            skip_update = 0

        if REVERSE and f_iter == 0:
            g = 1.0
        else:
            g = tl.load(decay_bh + kv_frame).to(tl.float32)

        # ---- Pass 2: dQ + ds_post += Q.T @ d_out ----------------------
        # Load state_post[q_frame] (zero in the REVERSE skip-update slot —
        # the fwd save still writes the all-zero state at that index).
        state = tl.load(
            spost_bh + q_frame * BLOCK_D * BLOCK_D + offs_dd,
            mask=mask_dd,
            other=0.0,
        )

        q_n_base = q_frame * S
        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            mask_s = offs_s < S
            mask_sd = mask_s[:, None] & mask_d[None, :]
            n_idx = q_n_base + offs_s

            q_ptrs = q_bh + offs_d[None, :] * N + n_idx[:, None]
            Q = tl.load(q_ptrs, mask=mask_sd, other=0.0)

            do_ptrs = do_bh + offs_d[None, :] * N + n_idx[:, None]
            dO = tl.load(do_ptrs, mask=mask_sd, other=0.0)

            ds_post += tl.dot(
                tl.trans(Q.to(grad_dtype)),
                dO.to(grad_dtype),
                out_dtype=tl.float32,
                input_precision=grad_ip,
            )

            dQ = tl.dot(
                dO.to(grad_dtype),
                tl.trans(state.to(grad_dtype)),
                out_dtype=tl.float32,
                input_precision=grad_ip,
            )

            dq_ptrs = dq_bh + offs_d[:, None] * N + n_idx[None, :]
            mask_ds = mask_d[:, None] & mask_s[None, :]
            tl.store(dq_ptrs, tl.trans(dQ), mask=mask_ds)

        if skip_update == 0:
            # ---- Reload state with state_pre[q_frame] for Pass 1 ----
            state = tl.load(
                spre_bh + q_frame * BLOCK_D * BLOCK_D + offs_dd,
                mask=mask_dd,
                other=0.0,
            )

            # ds_pre starts equal to ds_post (direct pass-through term).
            ds_pre = ds_post

            kv_n_base = kv_frame * S
            for s0 in range(0, S, BLOCK_S):
                offs_s = s0 + tl.arange(0, BLOCK_S)
                mask_s = offs_s < S
                mask_sd = mask_s[:, None] & mask_d[None, :]
                n_idx = kv_n_base + offs_s

                k_ptrs = k_bh + offs_d[None, :] * N + n_idx[:, None]
                v_ptrs = v_bh + offs_d[None, :] * N + n_idx[:, None]
                K = tl.load(k_ptrs, mask=mask_sd, other=0.0)
                V = tl.load(v_ptrs, mask=mask_sd, other=0.0)

                bt = tl.load(beta_bh + kv_frame * S + offs_s, mask=mask_s, other=0.0).to(tl.float32)

                V_pred = tl.dot(
                    K.to(grad_dtype),
                    state.to(grad_dtype),
                    out_dtype=tl.float32,
                    input_precision=grad_ip,
                )
                r = V - V_pred
                delta = r * bt[:, None]

                ddelta = tl.dot(
                    K.to(grad_dtype),
                    ds_post.to(grad_dtype),
                    out_dtype=tl.float32,
                    input_precision=grad_ip,
                )

                dV = ddelta * bt[:, None]
                dv_ptrs = dv_bh + offs_d[:, None] * N + n_idx[None, :]
                mask_ds = mask_d[:, None] & mask_s[None, :]
                tl.store(dv_ptrs, tl.trans(dV), mask=mask_ds)

                dbeta_st = tl.sum(ddelta * r, axis=1)
                tl.store(dbeta_bh + kv_frame * S + offs_s, dbeta_st, mask=mask_s)

                dV_pred = -ddelta * bt[:, None]

                dK_part1 = tl.dot(
                    delta.to(grad_dtype),
                    tl.trans(ds_post.to(grad_dtype)),
                    out_dtype=tl.float32,
                    input_precision=grad_ip,
                )
                dK_part2 = tl.dot(
                    dV_pred.to(grad_dtype),
                    tl.trans(state.to(grad_dtype)),
                    out_dtype=tl.float32,
                    input_precision=grad_ip,
                )
                dK = dK_part1 + dK_part2
                dk_ptrs = dk_bh + offs_d[:, None] * N + n_idx[None, :]
                tl.store(dk_ptrs, tl.trans(dK), mask=mask_ds)

                ds_pre += tl.dot(
                    tl.trans(K.to(grad_dtype)),
                    dV_pred.to(grad_dtype),
                    out_dtype=tl.float32,
                    input_precision=grad_ip,
                )

            # ---- ddecay[kv_frame] ----
            # state_pre = state_in * g, so state_in = state_pre / g.
            # ddecay = sum(ds_pre * state_in) = sum(ds_pre * state_pre) / g.
            # For REVERSE=False fwd frame 0, state_in is exactly 0 — hardcode
            # 0 to avoid amplifying rounding noise via 1/g.
            if (REVERSE == 0) and (f_iter == 0):
                ddecay_f = 0.0
            else:
                inv_g = 1.0 / (g + 1e-12)
                ddecay_f = tl.sum(ds_pre * state) * inv_g
            tl.store(ddecay_bh + kv_frame, ddecay_f)

            # Propagate to next bwd iter (which is fwd's previous iter).
            ds_post = ds_pre * g
        # else (skip_update branch): state_post == state_pre == 0, no
        # ddecay write, no kv-side writes; ds_post passes through unchanged
        # since ∂state_post/∂state_in = I when the update is skipped and g=1.
        # In REVERSE=True this is the LAST bwd iter (f_iter=0) so the
        # carried-over ds_post is discarded.


# =============================================================================
# Python wrappers
# =============================================================================


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


def _run_cam_prep_fwd_save(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_scale: float,
    norm_eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run :func:`cam_prep_func` while exposing intermediates needed by the bwd kernel.

    Mirrors :func:`cam_prep_func` exactly (same kernel launch, same outputs)
    but additionally returns the per-token ``inv_rms`` for both Q and K, plus
    the raw ``||k_pre_ucpe||^2`` and ``||k_post_ucpe||^2`` per ``(B, H, N)``.
    These are required by :func:`_cam_prep_bwd_dispatch` to (a) replay the
    ReLU/RMSNorm chain in fp32 without re-summing over the full ``C``
    channels and (b) chain ``grad_inflation_sq`` back through ``k_out`` and
    the pre-UCPE ``||k||^2`` term with the correct ``clamp_min`` indicators.

    Returns:
        ``(q_out, k_out, v_out, inflation_sq, q_inv_rms, k_inv_rms,
        k_pre_sq, k_post_sq)``. The first four match :func:`cam_prep_func`;
        the rest are fp32 contiguous saved-state tensors for the backward.
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

    _cam_prep_kernel[(B * N * H,)](
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
    inflation_sq = k_post_sq.clamp_min(1e-12) / k_pre_sq.clamp_min(1e-12)
    return q_out, k_out, v_out, inflation_sq, q_inv_rms, k_inv_rms, k_pre_sq, k_post_sq


def _cam_prep_bwd_dispatch(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    k_pre_sq: torch.Tensor,
    k_post_sq: torch.Tensor,
    k_out: torch.Tensor,
    *,
    grad_q: torch.Tensor | None,
    grad_k: torch.Tensor | None,
    grad_v: torch.Tensor | None,
    grad_inflation_sq: torch.Tensor | None,
    k_scale: float,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Hybrid Triton + torch backward for :func:`cam_prep_func`.

    Pipeline:

    1. **(torch)** Chain ``grad_inflation_sq`` through ``k_post_sq`` and
       ``k_pre_sq`` to produce
       (a) ``eff_d_k_out = grad_k + 2 * k_out * d_post_k_sq``  and
       (b) ``d_pre_k_sq``  — a per-(B, H, N) scalar fed into the kernel as a
       direct contribution to ``d_k_post_kscale``. Both ``clamp_min(1e-12)``
       indicators are honored so the gradient is exactly 0 in the (rare)
       saturating regime, matching :func:`_torch_cam_prep_reference`.

    2. **(Triton)** Launch :func:`_cam_prep_bwd_kernel` per ``(b, n, h)``
       to apply RoPE^T, UCPE^T, K-scale^T and the ReLU mask. Emits
       ``d_q_post_norm`` / ``d_k_post_norm`` ``(B, N, H, D)`` fp32
       intermediates (the post-norm pre-RMSNorm grad slots) and writes
       ``dv_raw`` directly (V skips RMSNorm/ReLU/K-scale).

    3. **(torch)** Apply the full-channel RMSNorm bwd. The cross-head
       coupling means ``S_q[b, n] = sum_{h,d} d_q_post_norm * q_raw *
       q_norm_w`` is reduced over ``H*D``; we then form
       ``dq_raw = d_q_post_norm * inv_rms_q * q_norm_w
                 - inv_rms_q^3 / C * q_raw * S_q``
       elementwise, and ``dq_norm_weight[c] = sum_{b,n}
       d_q_post_norm[b,n,c] * q_raw[b,n,c] * inv_rms_q[b,n]``.

    Returns:
        ``(dq_raw, dk_raw, dv_raw, dq_norm_weight, dk_norm_weight)``. Each
        slot is ``None`` if upstream did not request that grad — handled
        by the caller via ``ctx.needs_input_grad``. ``dq_raw`` / ``dk_raw``
        / ``dv_raw`` are returned in the same dtype as ``q_raw``;
        norm-weight grads are fp32 (matching the input dtype).
    """
    B, N, H, D = q_raw.shape
    assert k_raw.shape == q_raw.shape
    assert q_raw.is_contiguous() and k_raw.is_contiguous()
    assert q_inv_rms.shape == (B, N) and k_inv_rms.shape == (B, N)
    assert k_pre_sq.shape == (B, H, N) and k_post_sq.shape == (B, H, N)
    D_half = D // 2
    N_groups = D_half // 4
    C = H * D

    # ---- prepare inflation_sq grad chain (in fp32) ----
    eps_floor = 1e-12
    pre_clamped = k_pre_sq.clamp_min(eps_floor)
    if grad_inflation_sq is not None:
        gis = grad_inflation_sq.to(torch.float32)
        post_clamped = k_post_sq.clamp_min(eps_floor)
        pre_indicator = (k_pre_sq >= eps_floor).to(torch.float32)
        post_indicator = (k_post_sq >= eps_floor).to(torch.float32)
        # d(inflation_sq)/d(post_k_sq) = (1 / pre_clamped) * post_indicator
        # d(inflation_sq)/d(pre_k_sq)  = -post_clamped / pre_clamped^2 * pre_indicator
        d_post_k_sq = (post_indicator / pre_clamped) * gis  # (B, H, N)
        d_pre_k_sq = (-post_clamped / (pre_clamped * pre_clamped) * pre_indicator) * gis  # (B, H, N)
    else:
        d_post_k_sq = torch.zeros_like(k_pre_sq)
        d_pre_k_sq = torch.zeros_like(k_pre_sq)

    # eff_d_k_out: (B, H, D, N) fp32, contiguous
    if grad_k is None:
        grad_k_f32 = torch.zeros((B, H, D, N), dtype=torch.float32, device=q_raw.device)
    else:
        grad_k_f32 = grad_k.to(torch.float32)
    if grad_inflation_sq is not None:
        # k_out: (B, H, D, N), d_post_k_sq: (B, H, N) → broadcast over D dim.
        eff_d_k_out = (grad_k_f32 + 2.0 * k_out.to(torch.float32) * d_post_k_sq.unsqueeze(2)).contiguous()
    else:
        eff_d_k_out = grad_k_f32.contiguous()
    d_pre_k_sq = d_pre_k_sq.contiguous()

    # grad_q / grad_v as fp32 (B, H, D, N) contiguous (zero-fill if absent)
    if grad_q is None:
        grad_q_f32 = torch.zeros((B, H, D, N), dtype=torch.float32, device=q_raw.device)
    else:
        grad_q_f32 = grad_q.to(torch.float32).contiguous()
    if grad_v is None:
        grad_v_f32 = torch.zeros((B, H, D, N), dtype=torch.float32, device=q_raw.device)
    else:
        grad_v_f32 = grad_v.to(torch.float32).contiguous()

    # ---- allocate outputs / intermediates ----
    d_q_post_norm = torch.empty((B, N, H, D), dtype=torch.float32, device=q_raw.device)
    d_k_post_norm = torch.empty((B, N, H, D), dtype=torch.float32, device=q_raw.device)
    dv_raw_f32 = torch.empty((B, N, H, D), dtype=torch.float32, device=q_raw.device)

    BLOCK_D_ROPE = triton.next_power_of_2(D_half)
    BLOCK_GROUPS = triton.next_power_of_2(N_groups)

    _cam_prep_bwd_kernel[(B * N * H,)](
        q_raw,
        k_raw,
        q_norm_weight,
        k_norm_weight,
        q_inv_rms,
        k_inv_rms,
        proj_q,
        proj_kv,
        rope_cos,
        rope_sin,
        grad_q_f32,
        eff_d_k_out,
        grad_v_f32,
        d_pre_k_sq,
        d_q_post_norm,
        d_k_post_norm,
        dv_raw_f32,
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

    # ---- torch RMSNorm bwd over the saved post-norm grads ----
    # Cast raw inputs to fp32 for the cross-head reduction (matches kernel
    # numerics — the kernel uses fp32 internally as well).
    q_raw_f32 = q_raw.to(torch.float32)
    k_raw_f32 = k_raw.to(torch.float32)
    q_inv_rms_view = q_inv_rms.view(B, N, 1, 1)
    k_inv_rms_view = k_inv_rms.view(B, N, 1, 1)
    q_nw_view = q_norm_weight.view(1, 1, H, D)
    k_nw_view = k_norm_weight.view(1, 1, H, D)

    # S_q[b, n] = sum_{h, d} d_q_post_norm[b, n, h, d] * q_raw[b, n, h, d] * q_norm_w[h, d]
    weighted_q = d_q_post_norm * q_raw_f32  # reused for dq_norm_weight reduction
    weighted_k = d_k_post_norm * k_raw_f32
    S_q = (weighted_q * q_nw_view).sum(dim=(2, 3))  # (B, N)
    S_k = (weighted_k * k_nw_view).sum(dim=(2, 3))

    # dq_raw[b, n, h, d] = d_q_post_norm * inv_rms_q * q_norm_w
    #                   - inv_rms_q^3 / C * q_raw * S_q
    inv_q3 = (q_inv_rms**3).view(B, N, 1, 1)
    inv_k3 = (k_inv_rms**3).view(B, N, 1, 1)
    inv_C = 1.0 / float(C)
    dq_raw_f32 = d_q_post_norm * q_inv_rms_view * q_nw_view - inv_q3 * inv_C * q_raw_f32 * S_q.view(B, N, 1, 1)
    dk_raw_f32 = d_k_post_norm * k_inv_rms_view * k_nw_view - inv_k3 * inv_C * k_raw_f32 * S_k.view(B, N, 1, 1)

    # dq_norm_weight[h, d] = sum_{b, n} d_q_post_norm[b, n, h, d] * q_raw[b, n, h, d] * inv_rms_q[b, n]
    dq_norm_weight = (weighted_q * q_inv_rms_view).sum(dim=(0, 1)).reshape(-1).contiguous()
    dk_norm_weight = (weighted_k * k_inv_rms_view).sum(dim=(0, 1)).reshape(-1).contiguous()

    # Cast Q/K/V grads back to input dtype to match torch.autograd convention.
    dq_raw = dq_raw_f32.to(q_raw.dtype)
    dk_raw = dk_raw_f32.to(q_raw.dtype)
    dv_raw = dv_raw_f32.to(q_raw.dtype)

    return dq_raw, dk_raw, dv_raw, dq_norm_weight, dk_norm_weight


def cam_scan_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    reverse: bool = False,
    init_state: torch.Tensor | None = None,
    save_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused numerator-only single-path delta-rule scan for the cam branch.

    Args:
        q, k, v: ``(B, H, D, N)`` fp32 contiguous tensors.
        beta: ``(B, H, F, S)`` fp32 contiguous.
        decay: ``(B, H, F)`` fp32 contiguous.
        reverse: If ``True``, run the scan as the backward pass (equivalent
            to ``flip_and_shift``-ing the inputs along the frame axis and
            running forward).
        init_state: optional ``(B*H, BLOCK_D, BLOCK_D)`` fp32 contiguous
            tensor holding the forward-scan KV state at the END of a prefix
            sequence (i.e., AFTER the prefix's last update, BEFORE any
            further decay applied by this call). When provided, the kernel
            resumes the scan from this state instead of zero. ``BLOCK_D =
            next_pow2(D)`` and only the top-left ``D x D`` submatrix is
            read. Forward direction only — raises ``NotImplementedError``
            if combined with ``reverse=True``.
        save_final_state: when True, allocate a fresh fp32 zero buffer for
            the final KV state (after the last frame's update) and pass it
            to the kernel for write-out. Returned as the second tuple slot.
            Forward direction only.

    Returns:
        ``out`` of shape ``(B, H, D, N)`` fp32 matching
        ``torch_chunk_cam_single_path_delta_rule`` with ``chunk_size >= T``.

        When ``save_final_state=True``, returns ``(out, final_state)`` where
        ``final_state`` is fp32 ``(B*H, BLOCK_D, BLOCK_D)``.

    Raises:
        NotImplementedError: if ``reverse=True`` is combined with state
            passing. The cam branch's anti-causal scan resets per chunk in
            the production block, so there is no global cross-prefix state
            to cache for the reverse direction.
    """
    # Chunkwise integration (2026-05-06): dispatch all paths (fwd + reverse)
    # to `cam_scan_chunkwise`. Reverse uses chunkwise's existing direction=2
    # mode in phase_b_triton, which has the same flip-and-shift semantics as
    # cam's REVERSE=1 path. Bypass via FUSED_GDN_FORCE_LEGACY=1.
    if os.environ.get("FUSED_GDN_FORCE_LEGACY", "0") != "1":
        from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_chunkwise

        return cam_scan_chunkwise(
            q,
            k,
            v,
            beta,
            decay,
            reverse=reverse,
            init_state=init_state,
            save_final_state=save_final_state,
        )

    assert q.shape == k.shape == v.shape
    B, H, D, N = q.shape
    assert beta.shape[0] == B and beta.shape[1] == H
    F_frames = beta.shape[2]
    assert N % F_frames == 0
    S = N // F_frames
    assert beta.shape == (B, H, F_frames, S), f"beta shape {beta.shape}"
    assert decay.shape == (B, H, F_frames), f"decay shape {decay.shape}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert beta.is_contiguous() and decay.is_contiguous()
    assert q.dtype == torch.float32

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_S = _DEFAULT_BLOCK_S
    num_warps = 4
    num_stages = 1

    if reverse and (init_state is not None or save_final_state):
        raise NotImplementedError(
            "cam_scan_func: state passing (init_state / save_final_state) is "
            "only supported for the forward direction (reverse=False). The "
            "cam branch's anti-causal pass resets per chunk; there is no "
            "global cross-prefix state to cache for the reverse direction."
        )

    if init_state is not None:
        expected_shape = (B * H, BLOCK_D, BLOCK_D)
        if tuple(init_state.shape) != expected_shape:
            raise ValueError(
                f"cam_scan_func: init_state shape {tuple(init_state.shape)} "
                f"does not match expected {expected_shape} (BLOCK_D=next_pow2(D)={BLOCK_D})."
            )
        if init_state.dtype != torch.float32:
            raise ValueError(f"cam_scan_func: init_state must be fp32 (got {init_state.dtype}).")
        if not init_state.is_contiguous():
            raise ValueError("cam_scan_func: init_state must be contiguous.")
        if init_state.device != q.device:
            raise ValueError("cam_scan_func: init_state must be on the same device as q.")
        load_init = 1
    else:
        load_init = 0

    if save_final_state:
        final_state = torch.zeros(B * H, BLOCK_D, BLOCK_D, device=q.device, dtype=torch.float32)
        save_final = 1
    else:
        final_state = None
        save_final = 0

    out = torch.empty_like(q)

    dummy_state = torch.empty(1, device=q.device, dtype=torch.float32)
    init_state_ptr = init_state if load_init else dummy_state
    final_state_ptr = final_state if save_final else dummy_state

    _cam_scan_kernel[(B * H,)](
        q,
        k,
        v,
        beta,
        decay,
        out,
        dummy_state,
        dummy_state,
        init_state_ptr,
        final_state_ptr,
        H=H,
        F=F_frames,
        S=S,
        D=D,
        N=N,
        REVERSE=1 if reverse else 0,
        SAVE_STATES=0,
        LOAD_INIT_STATE=load_init,
        SAVE_FINAL_STATE=save_final,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if save_final_state:
        return out, final_state
    return out


def _run_cam_scan_fwd_save(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the forward scan with per-frame state snapshots saved.

    Used by :class:`CamScanFunction` to preserve the ``(state_pre, state_post)``
    snapshots that the Triton bwd kernel consumes. Snapshots are indexed by
    ``q_frame`` (matching the existing fwd-kernel save logic), so the bwd
    kernel can load them with the same ``q_frame`` derived in its
    ``REVERSE``-aware iteration.

    Returns:
        (out, state_pre, state_post). ``state_pre`` and ``state_post`` are
        ``(B, H, F, BLOCK_D, BLOCK_D)`` fp32 with ``BLOCK_D = next_pow2(D)``.
        Padding columns/rows past ``D`` are zero-masked on store.
    """
    assert q.shape == k.shape == v.shape
    B, H, D, N = q.shape
    F_frames = beta.shape[2]
    S = N // F_frames
    assert beta.shape == (B, H, F_frames, S)
    assert decay.shape == (B, H, F_frames)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert beta.is_contiguous() and decay.is_contiguous()
    assert q.dtype == torch.float32

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_S = _DEFAULT_BLOCK_S
    num_warps = 4
    num_stages = 1

    out = torch.empty_like(q)
    state_pre = torch.zeros(B * H, F_frames, BLOCK_D, BLOCK_D, device=q.device, dtype=torch.float32)
    state_post = torch.zeros(B * H, F_frames, BLOCK_D, BLOCK_D, device=q.device, dtype=torch.float32)
    dummy_state = torch.empty(1, device=q.device, dtype=torch.float32)

    _cam_scan_kernel[(B * H,)](
        q,
        k,
        v,
        beta,
        decay,
        out,
        state_pre,
        state_post,
        dummy_state,
        dummy_state,
        H=H,
        F=F_frames,
        S=S,
        D=D,
        N=N,
        REVERSE=1 if reverse else 0,
        SAVE_STATES=1,
        LOAD_INIT_STATE=0,
        SAVE_FINAL_STATE=0,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, state_pre, state_post


def _cam_scan_bwd_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    state_pre: torch.Tensor,
    state_post: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch ``_cam_scan_bwd_kernel`` and return ``(dq, dk, dv, dbeta, ddecay)``.

    All gradient outputs are fp32 contiguous, matching the dtype of the
    forward inputs. ``grad_out`` is cast to fp32 before launching.

    When ``reverse=True``, ``kv_frame=0`` is never visited (only used in the
    skipped first iter), so ``dk[..., 0, :]``, ``dv[..., 0, :]``,
    ``dbeta[..., 0, :]`` and ``ddecay[..., 0]`` must remain zero. We pre-zero
    every output buffer here so the kernel only needs to write the live slots.
    """
    assert q.shape == k.shape == v.shape
    B, H, D, N = q.shape
    F_frames = beta.shape[2]
    S = N // F_frames

    grad_out_f32 = grad_out.to(torch.float32).contiguous()
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)
    ddecay = torch.zeros_like(decay)

    BLOCK_D = triton.next_power_of_2(D)
    # For small S, ``next_pow2(S) < _DEFAULT_BLOCK_S`` — using the smaller value
    # avoids zero-padding huge unused tiles into shared memory.
    BLOCK_S = min(_DEFAULT_BLOCK_S, max(triton.next_power_of_2(S), 16))
    num_stages = 1
    REVERSE = 1 if reverse else 0

    last_err: Exception | None = None
    for num_warps in (4, 2, 1):
        try:
            _cam_scan_bwd_kernel[(B * H,)](
                q,
                k,
                v,
                beta,
                decay,
                state_pre,
                state_post,
                grad_out_f32,
                dq,
                dk,
                dv,
                dbeta,
                ddecay,
                H=H,
                F=F_frames,
                S=S,
                D=D,
                N=N,
                REVERSE=REVERSE,
                BLOCK_D=BLOCK_D,
                BLOCK_S=BLOCK_S,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return dq, dk, dv, dbeta, ddecay
        except triton.runtime.errors.OutOfResources as exc:
            last_err = exc
            continue
    raise RuntimeError("_cam_scan_bwd_kernel exhausted all num_warps choices: " + str(last_err))


# =============================================================================
# Section: Torch reference implementations used by autograd backward fallback
# =============================================================================
#
# The cam-branch backward uses a torch fallback (forward in Triton, backward
# through a torch reference via autograd). The references replicate the
# exact math of the Triton kernels (full-channel RMSNorm + ReLU + K-scale +
# 4x4 UCPE projmat + interleaved-pair real-valued RoPE, then numerator-only
# single-path delta-rule scan). They run in fp32 internally and cast outputs
# back to the input dtype, matching the kernels.


def _flip_and_shift(x: torch.Tensor, dim: int, shift_val: float) -> torch.Tensor:
    """Flip ``x`` along ``dim`` and right-shift by one (pad with ``shift_val``).

    Mirrors ``flip_and_shift`` in ``cam_gdn_playground.py`` and the production
    ``sana_gdn_blocks.flip_and_shift``.
    """
    x_flip = torch.flip(x, dims=[dim])
    x_shifted = x_flip.narrow(dim, 0, x.shape[dim] - 1)
    pad_shape = list(x.shape)
    pad_shape[dim] = 1
    padding = torch.full(pad_shape, shift_val, device=x.device, dtype=x.dtype)
    return torch.cat([padding, x_shifted], dim=dim)


def _torch_cam_scan_single_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch single-chunk delta-rule scan (numerator-only).

    Algebraically equivalent to ``torch_chunk_cam_single_path_delta_rule`` with
    ``chunk_size >= T``; matches the Triton ``_cam_scan_kernel`` math exactly
    (which also runs as a single chunk over all F frames).

    Args:
        q, k, v: ``(B, H, D, N)`` fp32 contiguous.
        beta:    ``(B, H, F, S)`` or ``(B, H, F)`` fp32 contiguous.
        decay:   ``(B, H, F)`` fp32 contiguous.

    Returns:
        out: ``(B, H, D, N)`` fp32 contiguous, ``N = F * S``.
    """
    B, H, D, N = q.shape
    if beta.ndim == 4:
        T = beta.shape[2]
    elif beta.ndim == 3:
        T = beta.shape[2]
    else:
        raise ValueError(f"beta must be (B,H,F[,S]); got ndim={beta.ndim}")
    if N % T != 0:
        raise ValueError(f"N ({N}) must be divisible by T ({T}).")
    S = N // T

    def to_frame_seq(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)  # (B, H, T, D, S)

    q_t = to_frame_seq(q)
    k_t = to_frame_seq(k)
    v_t = to_frame_seq(v)

    if beta.ndim == 4:
        beta_view = beta.unsqueeze(3)  # (B, H, T, 1, S)
    else:
        beta_view = beta.view(B, H, T, 1, 1)
    decay_view = decay.view(B, H, T, 1, 1)

    eye = torch.eye(D, device=q.device, dtype=q.dtype).view(1, 1, 1, D, D)

    k_beta = k_t * beta_view
    W = decay_view * (eye - torch.matmul(k_beta, k_t.transpose(-1, -2)))
    U = torch.matmul(v_t * beta_view, k_t.transpose(-1, -2))

    state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    s_kv_list: list[torch.Tensor] = []
    for t in range(T):
        state = torch.matmul(state, W[:, :, t]) + U[:, :, t]
        s_kv_list.append(state)
    s_all = torch.stack(s_kv_list, dim=2)  # (B, H, T, D, D)

    out_t = torch.matmul(s_all, q_t)  # (B, H, T, D, S)
    return out_t.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)


def _torch_cam_scan_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """Pure-torch reference for ``cam_scan_func`` supporting ``reverse=True``.

    For ``reverse=False`` this is the standard forward delta-rule scan.

    For ``reverse=True`` we emulate the Triton kernel's per-chunk
    ``flip_and_shift`` semantics (q is flipped only; k/v/beta are
    flip-and-shifted with pad value 0; decay is flip-and-shifted with pad
    value 1; output is then flipped back along the time axis).
    """
    if not reverse:
        return _torch_cam_scan_single_chunk(q, k, v, beta, decay)

    B, H, D, N = q.shape
    if beta.ndim == 4:
        T = beta.shape[2]
    elif beta.ndim == 3:
        T = beta.shape[2]
    else:
        raise ValueError(f"beta must be (B,H,F[,S]); got ndim={beta.ndim}")
    S = N // T

    def to_frame(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)  # (B, H, T, D, S)

    def from_frame(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)

    q_bwd = torch.flip(to_frame(q), dims=[2])
    k_bwd = _flip_and_shift(to_frame(k), dim=2, shift_val=0.0)
    v_bwd = _flip_and_shift(to_frame(v), dim=2, shift_val=0.0)
    beta_bwd = _flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_bwd = _flip_and_shift(decay, dim=2, shift_val=1.0)

    out_bwd = _torch_cam_scan_single_chunk(
        from_frame(q_bwd),
        from_frame(k_bwd),
        from_frame(v_bwd),
        beta_bwd,
        decay_bwd,
    )
    out_bwd_t = out_bwd.view(B, H, D, T, S)  # already in (B, H, D, T, S)
    return torch.flip(out_bwd_t, dims=[3]).reshape(B, H, D, N)


def _torch_cam_prep_reference(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_scale: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch reference for ``cam_prep_func`` matching ``_cam_prep_kernel``.

    Replicates exactly:
        - Full-channel (over ``H*D``) RMSNorm + per-channel weight on Q, K.
        - ReLU on Q, K.
        - K-scale on K.
        - 4x4 UCPE projmat on first ``D/2`` dims (Q via ``proj_q``,
          K and V via ``proj_kv``).
        - Interleaved-pair real-valued RoPE on second ``D/2`` dims using
          ``rope_cos`` / ``rope_sin`` (the same tables passed to the kernel).
        - ``inflation_sq = ||k_post_ucpe||^2 / ||k_pre_ucpe||^2`` per (B, H, N),
          with the same ``clamp_min(1e-12)`` floor as ``cam_prep_func``.

    All math runs in fp32 internally; outputs ``(q, k, v)`` are cast back to
    ``q_raw.dtype`` and ``inflation_sq`` is fp32.
    """
    B, N, H, D = q_raw.shape
    if D % 2 != 0:
        raise ValueError(f"D ({D}) must be even.")
    if (D // 2) % 4 != 0:
        raise ValueError(f"D/2 ({D // 2}) must be divisible by 4 (UCPE projmat).")
    C = H * D
    D_half = D // 2
    n_groups = D_half // 4

    q32 = q_raw.float()
    k32 = k_raw.float()
    v32 = v_raw.float()

    # ---- Full-channel RMSNorm + per-channel weight (Q, K only) ----
    q_inv_rms = torch.rsqrt((q32 * q32).sum(dim=(-1, -2)) / C + norm_eps)  # (B, N)
    k_inv_rms = torch.rsqrt((k32 * k32).sum(dim=(-1, -2)) / C + norm_eps)
    q_nw = q_norm_weight.float().view(1, 1, H, D)
    k_nw = k_norm_weight.float().view(1, 1, H, D)
    q_normed = q32 * q_inv_rms.view(B, N, 1, 1) * q_nw
    k_normed = k32 * k_inv_rms.view(B, N, 1, 1) * k_nw

    # ---- ReLU + K-scale ----
    q_normed = torch.relu(q_normed)
    k_normed = torch.relu(k_normed) * k_scale

    # ---- Pre-UCPE ||k||^2 over the full D dim ----
    pre_k_sq_BNH = (k_normed * k_normed).sum(dim=-1)  # (B, N, H)

    # ---- UCPE 4x4 projmat on first half ----
    q_first = q_normed[..., :D_half].reshape(B, N, H, n_groups, 4)
    k_first = k_normed[..., :D_half].reshape(B, N, H, n_groups, 4)
    v_first = v32[..., :D_half].reshape(B, N, H, n_groups, 4)

    # out[b,n,h,g,i] = sum_j P[b,n,i,j] * x[b,n,h,g,j]
    # einsum: 'bnij,bnhgj->bnhgi'
    proj_q_f = proj_q.float()
    proj_kv_f = proj_kv.float()
    q_first_proj = torch.einsum("bnij,bnhgj->bnhgi", proj_q_f, q_first).reshape(B, N, H, D_half)
    k_first_proj = torch.einsum("bnij,bnhgj->bnhgi", proj_kv_f, k_first).reshape(B, N, H, D_half)
    v_first_proj = torch.einsum("bnij,bnhgj->bnhgi", proj_kv_f, v_first).reshape(B, N, H, D_half)

    # ---- Interleaved-pair real-valued RoPE on second half ----
    # Kernel form: y[d] = x[d]*rope_cos[d] + x[d^1]*rope_sin[d]
    # where rope_cos/rope_sin come from _prepare_ucpe_rope_tables.
    q_second = q_normed[..., D_half:]
    k_second = k_normed[..., D_half:]
    v_second = v32[..., D_half:]

    def _pair_swap(x: torch.Tensor) -> torch.Tensor:
        # Swap consecutive pairs along the last dim: (..., D_half) where D_half is even.
        # x[..., 2i] <-> x[..., 2i+1].
        x_pairs = x.unflatten(-1, (D_half // 2, 2))
        x_swapped = x_pairs.flip(-1)
        return x_swapped.flatten(-2)

    cos_b = rope_cos.float().view(1, N, 1, D_half)
    sin_b = rope_sin.float().view(1, N, 1, D_half)
    q_rope = q_second * cos_b + _pair_swap(q_second) * sin_b
    k_rope = k_second * cos_b + _pair_swap(k_second) * sin_b
    v_rope = v_second * cos_b + _pair_swap(v_second) * sin_b

    # ---- Reassemble (B, N, H, D) and post-UCPE k norm ----
    q_out_BNHD = torch.cat([q_first_proj, q_rope], dim=-1)
    k_out_BNHD = torch.cat([k_first_proj, k_rope], dim=-1)
    v_out_BNHD = torch.cat([v_first_proj, v_rope], dim=-1)

    post_k_sq_BNH = (k_out_BNHD * k_out_BNHD).sum(dim=-1)  # (B, N, H)

    out_dtype = q_raw.dtype
    q_out = q_out_BNHD.to(out_dtype).permute(0, 2, 3, 1).contiguous()
    k_out = k_out_BNHD.to(out_dtype).permute(0, 2, 3, 1).contiguous()
    v_out = v_out_BNHD.to(out_dtype).permute(0, 2, 3, 1).contiguous()

    pre_k_sq = pre_k_sq_BNH.permute(0, 2, 1).contiguous()  # (B, H, N)
    post_k_sq = post_k_sq_BNH.permute(0, 2, 1).contiguous()
    inflation_sq = post_k_sq.clamp_min(1e-12) / pre_k_sq.clamp_min(1e-12)
    return q_out, k_out, v_out, inflation_sq


# =============================================================================
# Section: Autograd-enabled wrappers (torch-fallback backward)
# =============================================================================


def _cam_scan_torch_fallback_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    needs: tuple[bool, bool, bool, bool, bool],
    grad_out: torch.Tensor,
    reverse: bool,
) -> list[torch.Tensor | None]:
    """Recompute the cam-branch scan via the torch reference and return grads.

    Used (a) for ``reverse=True`` (the Triton bwd kernel only handles
    ``reverse=False``), and (b) as an opt-in debugging fallback when the
    environment variable ``CAM_SCAN_BWD_FALLBACK=1`` is set.
    """
    detached = []
    for tensor, need in zip((q, k, v, beta, decay), needs):
        t = tensor.detach()
        if need:
            t = t.requires_grad_(True)
        detached.append(t)
    active = [t for t in detached if t.requires_grad]

    with torch.enable_grad():
        q_d, k_d, v_d, beta_d, decay_d = detached
        ref_out = _torch_cam_scan_reference(q_d, k_d, v_d, beta_d, decay_d, reverse=reverse)
        if active:
            active_grads = torch.autograd.grad(
                outputs=ref_out,
                inputs=tuple(active),
                grad_outputs=grad_out.to(ref_out.dtype),
                allow_unused=True,
            )
        else:
            active_grads = []

    grads: list[torch.Tensor | None] = []
    active_iter = iter(active_grads)
    for tensor in detached:
        grads.append(next(active_iter) if tensor.requires_grad else None)
    return grads


class CamScanFunction(torch.autograd.Function):
    """Autograd ``Function`` wrapping ``cam_scan_func``.

    Forward calls the Triton ``_cam_scan_kernel`` with ``SAVE_STATES=1`` so
    per-frame state snapshots (``state_pre[q_frame]``, ``state_post[q_frame]``)
    are kept for the backward pass. Backward runs the true Triton bwd
    kernel (``_cam_scan_bwd_kernel``) for both ``reverse=False`` and
    ``reverse=True``, replaying the recurrence in reverse time using the
    saved snapshots.

    Set the environment variable ``CAM_SCAN_BWD_FALLBACK=1`` to opt back
    into the torch-recompute backward (debugging / bisection only — the
    Triton bwd kernel is the production path for both reverse values).
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        reverse: bool,
    ) -> torch.Tensor:
        ctx.set_materialize_grads(False)
        ctx.reverse = bool(reverse)

        force_torch_fallback = os.environ.get("CAM_SCAN_BWD_FALLBACK", "0") == "1"
        ctx.use_triton_bwd = not force_torch_fallback

        if ctx.use_triton_bwd:
            out, state_pre, state_post = _run_cam_scan_fwd_save(q, k, v, beta, decay, reverse=ctx.reverse)
            ctx.save_for_backward(q, k, v, beta, decay, state_pre, state_post)
            return out

        # Torch-fallback backward path: don't bother saving state snapshots.
        ctx.save_for_backward(q, k, v, beta, decay)
        return cam_scan_func(q, k, v, beta, decay, reverse=reverse)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        if grad_out is None:
            return (None, None, None, None, None, None)

        if ctx.use_triton_bwd:
            q, k, v, beta, decay, state_pre, state_post = ctx.saved_tensors
            needs = ctx.needs_input_grad[:5]  # q, k, v, beta, decay
            dq, dk, dv, dbeta, ddecay = _cam_scan_bwd_dispatch(
                q,
                k,
                v,
                beta,
                decay,
                state_pre,
                state_post,
                grad_out,
                reverse=ctx.reverse,
            )
            grads: list[torch.Tensor | None] = [
                dq if needs[0] else None,
                dk if needs[1] else None,
                dv if needs[2] else None,
                dbeta if needs[3] else None,
                ddecay if needs[4] else None,
            ]
            return (*grads, None)

        # Env-var torch-recompute backward (debug bisection only).
        q, k, v, beta, decay = ctx.saved_tensors
        needs = ctx.needs_input_grad[:5]
        grads = _cam_scan_torch_fallback_backward(q, k, v, beta, decay, needs, grad_out, ctx.reverse)
        return (*grads, None)


def cam_scan_func_with_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """Autograd-enabled wrapper around :func:`cam_scan_func`.

    Forward is identical to :func:`cam_scan_func`; backward is computed via
    a torch reference (``_torch_cam_scan_reference``). Use this in training
    paths where any of ``q, k, v, beta, decay`` may require gradients.

    Inference paths can keep calling :func:`cam_scan_func` directly to avoid
    the small autograd bookkeeping overhead.
    """
    return CamScanFunction.apply(q, k, v, beta, decay, reverse)


def _cam_prep_torch_fallback_backward(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    needs: tuple[bool, ...],
    grad_q: torch.Tensor | None,
    grad_k: torch.Tensor | None,
    grad_v: torch.Tensor | None,
    grad_inflation_sq: torch.Tensor | None,
    k_scale: float,
    norm_eps: float,
) -> list[torch.Tensor | None]:
    """Recompute the cam-branch prep via the torch reference and return grads.

    This is the legacy path: re-runs ``_torch_cam_prep_reference`` under
    ``torch.enable_grad()`` and lets PyTorch autograd compute every gradient.
    Used (a) when any of ``proj_q / proj_kv / rope_cos / rope_sin`` is itself
    requesting a gradient (the Triton kernel does not produce those grads),
    and (b) as the opt-in debug path triggered by
    ``CAM_PREP_BWD_FALLBACK=1``.

    Args:
        q_raw, k_raw, ..., rope_sin: the nine tensor inputs of
            :func:`cam_prep_func` (in the same order as
            :class:`CamPrepFunction.forward`'s arg list).
        needs: ``ctx.needs_input_grad[:9]`` — boolean per-input flags.
        grad_q, grad_k, grad_v, grad_inflation_sq: upstream gradients.
        k_scale, norm_eps: scalar fwd args.

    Returns:
        A 9-element list of ``Optional[torch.Tensor]`` aligned with the
        ``saved`` tuple. Entries that didn't request a gradient are ``None``.
    """
    saved = (
        q_raw,
        k_raw,
        v_raw,
        q_norm_weight,
        k_norm_weight,
        proj_q,
        proj_kv,
        rope_cos,
        rope_sin,
    )
    detached: list[torch.Tensor] = []
    for tensor, need in zip(saved, needs):
        t = tensor.detach()
        if need:
            t = t.requires_grad_(True)
        detached.append(t)
    active = [t for t in detached if t.requires_grad]

    with torch.enable_grad():
        (q_d, k_d, v_d, qnw_d, knw_d, pq_d, pkv_d, rc_d, rs_d) = detached
        ref_q, ref_k, ref_v, ref_inf = _torch_cam_prep_reference(
            q_d,
            k_d,
            v_d,
            q_norm_weight=qnw_d,
            k_norm_weight=knw_d,
            proj_q=pq_d,
            proj_kv=pkv_d,
            rope_cos=rc_d,
            rope_sin=rs_d,
            k_scale=k_scale,
            norm_eps=norm_eps,
        )

        outputs = []
        grad_outputs = []
        if grad_q is not None:
            outputs.append(ref_q)
            grad_outputs.append(grad_q.to(ref_q.dtype))
        if grad_k is not None:
            outputs.append(ref_k)
            grad_outputs.append(grad_k.to(ref_k.dtype))
        if grad_v is not None:
            outputs.append(ref_v)
            grad_outputs.append(grad_v.to(ref_v.dtype))
        if grad_inflation_sq is not None:
            outputs.append(ref_inf)
            grad_outputs.append(grad_inflation_sq.to(ref_inf.dtype))

        if active and outputs:
            active_grads = torch.autograd.grad(
                outputs=tuple(outputs),
                inputs=tuple(active),
                grad_outputs=tuple(grad_outputs),
                allow_unused=True,
            )
        else:
            active_grads = []

    grads: list[torch.Tensor | None] = []
    active_iter = iter(active_grads)
    for tensor in detached:
        grads.append(next(active_iter) if tensor.requires_grad else None)
    return grads


class CamPrepFunction(torch.autograd.Function):
    """Autograd ``Function`` wrapping ``cam_prep_func``.

    Forward calls the fused Triton ``_cam_prep_kernel`` via
    :func:`_run_cam_prep_fwd_save` so the per-token ``inv_rms`` /
    ``k_pre_sq`` / ``k_post_sq`` snapshots required by the bwd kernel are
    preserved alongside the standard outputs.

    Backward runs the true Triton bwd kernel via
    :func:`_cam_prep_bwd_dispatch` for the standard training path
    (``q_raw``, ``k_raw``, ``v_raw``, ``q_norm_weight``, ``k_norm_weight``
    only request grads). The Triton path implements:

    * RoPE^T, UCPE^T, K-scale^T, and ReLU mask in a single fused kernel
      (one program per ``(b, n, h)``);
    * ``grad_inflation_sq`` chain through ``k_post_sq`` (added into
      ``eff_dO_k``) and ``k_pre_sq`` (direct contribution to
      ``d_k_post_kscale``), with ``clamp_min(1e-12)`` indicators honored;
    * The full-channel RMSNorm bwd (per-token cross-head reduction) is
      done in PyTorch on the kernel's ``d_q_post_norm`` /
      ``d_k_post_norm`` intermediates — see
      :func:`_cam_prep_bwd_dispatch` for details.

    The legacy torch-recompute fallback (running
    :func:`_torch_cam_prep_reference` under autograd) is automatically
    selected when any of ``proj_q / proj_kv / rope_cos / rope_sin`` requests
    a gradient (the Triton path emits ``None`` for those slots) or when the
    environment variable ``CAM_PREP_BWD_FALLBACK=1`` is set (debug /
    bisection only).
    """

    @staticmethod
    def forward(
        ctx,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        proj_q: torch.Tensor,
        proj_kv: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        k_scale: float,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        ctx.k_scale = float(k_scale)
        ctx.norm_eps = float(norm_eps)

        force_torch_fallback = os.environ.get("CAM_PREP_BWD_FALLBACK", "0") == "1"
        ctx.use_triton_bwd = not force_torch_fallback

        if ctx.use_triton_bwd:
            (
                q_out,
                k_out,
                v_out,
                inflation_sq,
                q_inv_rms,
                k_inv_rms,
                k_pre_sq,
                k_post_sq,
            ) = _run_cam_prep_fwd_save(
                q_raw,
                k_raw,
                v_raw,
                q_norm_weight=q_norm_weight,
                k_norm_weight=k_norm_weight,
                proj_q=proj_q,
                proj_kv=proj_kv,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                k_scale=k_scale,
                norm_eps=norm_eps,
            )
            ctx.save_for_backward(
                q_raw,
                k_raw,
                v_raw,
                q_norm_weight,
                k_norm_weight,
                proj_q,
                proj_kv,
                rope_cos,
                rope_sin,
                q_inv_rms,
                k_inv_rms,
                k_pre_sq,
                k_post_sq,
                k_out,
            )
        else:
            q_out, k_out, v_out, inflation_sq = cam_prep_func(
                q_raw,
                k_raw,
                v_raw,
                q_norm_weight=q_norm_weight,
                k_norm_weight=k_norm_weight,
                proj_q=proj_q,
                proj_kv=proj_kv,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                k_scale=k_scale,
                norm_eps=norm_eps,
            )
            ctx.save_for_backward(
                q_raw,
                k_raw,
                v_raw,
                q_norm_weight,
                k_norm_weight,
                proj_q,
                proj_kv,
                rope_cos,
                rope_sin,
            )
        return q_out, k_out, v_out, inflation_sq

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v, grad_inflation_sq):  # type: ignore[override]
        if grad_q is None and grad_k is None and grad_v is None and grad_inflation_sq is None:
            return tuple([None] * 11)

        needs = ctx.needs_input_grad[:9]  # nine tensor inputs
        # If anyone outside (q_raw, k_raw, v_raw, q_norm_weight, k_norm_weight)
        # requests a grad, the Triton bwd cannot handle it — fall back to the
        # torch reference.
        triton_bwd_supported_needs = needs[:5]
        proj_or_rope_needs_grad = any(needs[5:])

        if ctx.use_triton_bwd and not proj_or_rope_needs_grad:
            (
                q_raw,
                k_raw,
                v_raw,
                q_norm_weight,
                k_norm_weight,
                proj_q,
                proj_kv,
                rope_cos,
                rope_sin,
                q_inv_rms,
                k_inv_rms,
                k_pre_sq,
                k_post_sq,
                k_out,
            ) = ctx.saved_tensors

            (
                dq_raw,
                dk_raw,
                dv_raw,
                dq_norm_weight,
                dk_norm_weight,
            ) = _cam_prep_bwd_dispatch(
                q_raw,
                k_raw,
                q_norm_weight,
                k_norm_weight,
                proj_q,
                proj_kv,
                rope_cos,
                rope_sin,
                q_inv_rms,
                k_inv_rms,
                k_pre_sq,
                k_post_sq,
                k_out,
                grad_q=grad_q,
                grad_k=grad_k,
                grad_v=grad_v,
                grad_inflation_sq=grad_inflation_sq,
                k_scale=ctx.k_scale,
            )
            grads: list[torch.Tensor | None] = [
                dq_raw if triton_bwd_supported_needs[0] else None,
                dk_raw if triton_bwd_supported_needs[1] else None,
                dv_raw if triton_bwd_supported_needs[2] else None,
                dq_norm_weight if triton_bwd_supported_needs[3] else None,
                dk_norm_weight if triton_bwd_supported_needs[4] else None,
                None,  # proj_q
                None,  # proj_kv
                None,  # rope_cos
                None,  # rope_sin
            ]
            return (*grads, None, None)

        # Torch fallback path. ``ctx.saved_tensors`` holds either 9 (legacy
        # forward) or 14 (Triton fwd save) tensors — slice the leading nine.
        saved = ctx.saved_tensors[:9]
        (
            q_raw,
            k_raw,
            v_raw,
            q_norm_weight,
            k_norm_weight,
            proj_q,
            proj_kv,
            rope_cos,
            rope_sin,
        ) = saved
        grads = _cam_prep_torch_fallback_backward(
            q_raw,
            k_raw,
            v_raw,
            q_norm_weight,
            k_norm_weight,
            proj_q,
            proj_kv,
            rope_cos,
            rope_sin,
            needs=needs,
            grad_q=grad_q,
            grad_k=grad_k,
            grad_v=grad_v,
            grad_inflation_sq=grad_inflation_sq,
            k_scale=ctx.k_scale,
            norm_eps=ctx.norm_eps,
        )
        # Two trailing None for non-tensor scalars (k_scale, norm_eps).
        return (*grads, None, None)


def cam_prep_func_with_grad(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_scale: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd-enabled wrapper around :func:`cam_prep_func`.

    Forward is identical to :func:`cam_prep_func`; backward is computed via a
    torch reference (``_torch_cam_prep_reference``). Use this in training
    paths so gradients flow back through Q/K/V projection inputs and through
    the RMSNorm weights.

    Inference paths can keep calling :func:`cam_prep_func` directly.
    """
    return CamPrepFunction.apply(
        q_raw,
        k_raw,
        v_raw,
        q_norm_weight,
        k_norm_weight,
        proj_q,
        proj_kv,
        rope_cos,
        rope_sin,
        k_scale,
        norm_eps,
    )


__all__ = [
    "CamPrepFunction",
    "CamScanFunction",
    "_cam_prep_bwd_dispatch",
    "_cam_prep_bwd_kernel",
    "_cam_prep_kernel",
    "_cam_prep_torch_fallback_backward",
    "_cam_scan_bwd_dispatch",
    "_cam_scan_bwd_kernel",
    "_cam_scan_kernel",
    "_invert_SE3",
    "_precompute_cam_inv_rms",
    "_prepare_ucpe_rope_tables",
    "_process_camera_conditions_raymats_only",
    "_run_cam_prep_fwd_save",
    "_run_cam_scan_fwd_save",
    "_torch_cam_prep_reference",
    "cam_prep_func",
    "cam_prep_func_with_grad",
    "cam_scan_func",
    "cam_scan_func_with_grad",
]
