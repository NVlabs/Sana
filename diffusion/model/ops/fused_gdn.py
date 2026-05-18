"""Fused-BiGDN Triton mega-kernel.

What this file contains:
  1. `_fused_gdn_kernel` — the unified Triton jit kernel. It fuses
     RMSNorm + ReLU + k_scale + RoPE + BiGDN recurrence (forward OR reverse)
     in a single GPU program per (batch, head) with double-buffered state.
  2. Config helpers (`_get_kernel_config`, `_kcfg`, `_precision_params`,
     `_resolve_launch_config`, `_prepare_launch`).
  3. `prepare_rope_tables` — complex (1,1,N,D//2) RoPE freqs → (N, D) cos/sin
     tables with interleaved-pair layout.
  4. `_precompute_inv_rms` — cross-head (full-channel) 1/RMS per token.
  5. `fused_gdn_func` — one-direction wrapper (used for FWD and BWD separately).
  6. `fused_bigdn_func` — bidirectional wrapper: runs both directions and
     fuses `(num_fwd + num_bwd) / (den_fwd + den_bwd + eps)`.
  7. `FusedStateCachedForwardRef` — reference snippet (Python-only class)
     showing the intended call convention from
     `StateCachedGDNChunkCasualNATConvChunk._fused_statecached_forward`.
     This is a *reference skeleton* for how the upstream block wires up the
     kernel inside a trained GDN block, NOT a runnable class (it has no
     constructor).

Deliberately omitted (upstream code not needed for inference A/B testing):
  - SAVE_STATE / saved_state snapshots (training backward).
  - LOAD_INIT_STATE / SAVE_FINAL_STATE (state caching wrappers
    `fused_gdn_stateful`, `fused_bigdn_stateful`).
  - Sink-attention / softmax-attention variants.
  - `fused_gdn_forward` uni-directional wrapper (our model is bidirectional).
  - `BiGDNCasualNATConv` base class (ATTENTION_BLOCKS registry dependency).

Call convention (intended usage, see `FusedStateCachedForwardRef`):

    qkv  = F.linear(x, concat(q.weight, k.weight, v.weight))        # (B, N, 3C)
    qkv  = qkv.reshape(B, N, 3, H, D)
    q_inv_rms = _precompute_inv_rms(qkv, 0, C, norm_eps)            # (B, N)
    k_inv_rms = _precompute_inv_rms(qkv, 1, C, norm_eps)            # (B, N)
    rope_cos, rope_sin = prepare_rope_tables(rotary_emb, N, D, dev) # (N, D)
    out = fused_bigdn_func(
        qkv, q_inv_rms, k_inv_rms,
        q_norm_weight=self.q_norm.weight.float(),   # (C,)
        k_norm_weight=self.k_norm.weight.float(),   # (C,)
        rope_cos=rope_cos, rope_sin=rope_sin,
        beta=beta,                                   # (B, H, F, S)
        decay=decay,                                 # (B, H, F)
        F=T, S=H_s*W_s,
        k_scale=(D**-0.5) * (S**-0.5),
        eps=self.eps,
    )                                                # (B, N, H, D)

Precision knob: env var `FUSED_GDN_PRECISION` or `PRECISION_OVERRIDE`:
  0=IEEE fp32 dots, 1=TF32, 2=bf16 TC + fp32 state [default], 3=bf16 TC + bf16 state.
"""

# ruff: noqa: E501

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# =====================================================================
#  GPU-adaptive kernel config
# =====================================================================


def _get_kernel_config() -> dict:
    """Return optimal kernel parameters for the current GPU.

    STATE_FP32: use fp32 state_prev when SRAM is large enough.
      - bf16 state_prev: ~96KB total SRAM (fits GB10's 101KB).
      - fp32 state_prev: ~128KB total SRAM (needs H100's 228KB+).
    """
    if not torch.cuda.is_available():
        return {"BLOCK_S": 64, "num_stages": 1, "num_warps": 4, "STATE_FP32": False}
    smem = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
    state_fp32 = smem >= 150 * 1024  # H100 (228KB) yes, GB10 (101KB) no
    return {"BLOCK_S": 64, "num_stages": 1, "num_warps": 8, "STATE_FP32": state_fp32}


_KCFG = None


def _kcfg():
    global _KCFG
    if _KCFG is None:
        _KCFG = _get_kernel_config()
    return _KCFG


# precision=0 → IEEE fp32 dots + fp32 state  (DOT_PRECISION=2, STATE_FP32=1)
# precision=1 → TF32  dots   + fp32 state    (DOT_PRECISION=1, STATE_FP32=1)
# precision=2 → bf16  dots   + fp32 state    (DOT_PRECISION=0, STATE_FP32=1) [default]
# precision=3 → bf16  dots   + bf16 state    (DOT_PRECISION=0, STATE_FP32=0)
def _precision_params(precision: int) -> tuple:
    if precision == 0:
        return 2, True
    elif precision == 1:
        return 1, True
    elif precision == 3:
        return 0, False
    else:  # default
        return 0, True


_env_prec = os.environ.get("FUSED_GDN_PRECISION", None)
PRECISION_OVERRIDE: int | None = int(_env_prec) if _env_prec is not None else None


def _resolve_launch_config() -> tuple:
    """Returns (prec, dot_prec, state_fp32, num_warps).

    Uses ``PRECISION_OVERRIDE`` when set; otherwise falls back to ``_kcfg()``
    (which picks ``STATE_FP32`` based on per-GPU SRAM). ``num_warps`` is
    clamped to 4 when dots run on fp32 operands (more registers needed).
    """
    cfg = _kcfg()
    prec = PRECISION_OVERRIDE if PRECISION_OVERRIDE is not None else 2
    dot_prec, state_fp32 = _precision_params(prec)
    if PRECISION_OVERRIDE is None:
        state_fp32 = cfg["STATE_FP32"]
    nw = cfg["num_warps"]
    if dot_prec >= 1:
        nw = min(nw, 4)
    return prec, dot_prec, state_fp32, nw


def _prepare_launch(D: int, beta: torch.Tensor, decay: torch.Tensor) -> tuple:
    """Shared launcher preamble.

    Returns (BLOCK_D, BLOCK_S, dot_prec, state_fp32, nw, cfg, beta_c, decay_c).
    ``beta_c`` / ``decay_c`` are the contiguous copies the kernel needs.
    """
    BLOCK_D = triton.next_power_of_2(D)
    cfg = _kcfg()
    BLOCK_S = cfg["BLOCK_S"]
    _, dot_prec, state_fp32, nw = _resolve_launch_config()
    return BLOCK_D, BLOCK_S, dot_prec, state_fp32, nw, cfg, beta.contiguous(), decay.contiguous()


# =====================================================================
#  Unified forward Triton Mega-Kernel (inference-only variant)
# =====================================================================
# Fuses: RMSNorm + ReLU + k_scale + RoPE + BiGDN recurrence.
#
# Inputs:
#   qkv (B, N, 3, H, D) interleaved  — strides passed explicitly.
#   beta (B, H, F, S), decay (B, H, F)   contiguous.
#   q_norm_w, k_norm_w (H*D,) full-channel  — only read when QK_NORM=1.
#   rope_cos, rope_sin (N, D) contiguous.
#   q_inv_rms, k_inv_rms (B, N) full-channel  — only read when USE_PRECOMPUTED_RMS=1.
#
# Outputs:
#   out (B, N, H, D)  = num / (den + eps)   — unused by BiGDN wrappers.
#   num (B, N, H, D)  = numerator before divide (summed across directions).
#   den (B, H, N)     = denominator before divide (summed across directions).
#
# NOTE (inference-only build): upstream also supports SAVE_STATE,
# LOAD_INIT_STATE, SAVE_FINAL_STATE for training backward / state caching.
# Those constexpr branches are preserved in the kernel so the source stays
# 1-for-1 with upstream (they compile away when launched with flags=0).


@triton.jit
def _fused_gdn_kernel(
    # ---- interleaved QKV : (B, N, 3, H, D) ----
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    # ---- gates ----
    beta_ptr,
    decay_ptr,
    # ---- inv-RMS (B, N) — only read when USE_PRECOMPUTED_RMS=1 ----
    q_inv_rms_ptr,
    k_inv_rms_ptr,
    # ---- norm weights (H*D,) full-channel — only read when QK_NORM=1 ----
    q_norm_w_ptr,
    k_norm_w_ptr,
    # ---- RoPE tables (N, D) contiguous ----
    rope_cos_ptr,
    rope_sin_ptr,
    # ---- outputs ----
    out_ptr,  # (B, N, H, D)
    num_ptr,  # (B, N, H, D)
    den_ptr,  # (B, H, N)
    # ---- saved-state dummies (unused in this build but kept for signature parity) ----
    saved_state_ptr,
    saved_z_ptr,
    saved_state_curr_ptr,
    saved_z_curr_ptr,
    init_state_kv_ptr,
    init_state_z_ptr,
    final_state_kv_ptr,
    final_state_z_ptr,
    # ---- scalars / dims ----
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    EPS: tl.constexpr,
    QK_NORM: tl.constexpr,
    USE_PRECOMPUTED_RMS: tl.constexpr,
    STATE_FP32: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    REVERSE: tl.constexpr,
    SAVE_STATE: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,
    SAVE_FINAL_STATE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # ---- dot product precision / operand dtype ----
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    # ---- program → (batch, head) ----
    pid = tl.program_id(0)
    pid_b = pid // H
    pid_h = pid % H
    N = F * S
    bh = pid_b * H + pid_h

    # ---- base pointers ----
    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    out_bh = out_ptr + pid_b * (N * H * D) + pid_h * D
    num_bh = num_ptr + pid_b * (N * H * D) + pid_h * D
    den_bh = den_ptr + bh * N
    beta_bh = beta_ptr + bh * (F * S)
    decay_bh = decay_ptr + bh * F
    if SAVE_STATE:
        st_bh = saved_state_ptr + bh * F * BLOCK_D * BLOCK_D
        sz_bh = saved_z_ptr + bh * F * BLOCK_D
        stc_bh = saved_state_curr_ptr + bh * F * BLOCK_D * BLOCK_D
        szc_bh = saved_z_curr_ptr + bh * F * BLOCK_D

    # ---- D-index helpers ----
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D
    D_inv = 1.0 / D

    # ---- full-channel norm weights (only when QK_NORM=1) ----
    nw_offset = pid_h * D
    if QK_NORM:
        q_nw = tl.load(q_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        q_nw_pair = tl.load(q_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0).to(tl.float32)
        k_nw_pair = tl.load(k_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0).to(tl.float32)

    k_scale = K_SCALE
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    # ---- double-buffer state ----
    if LOAD_INIT_STATE:
        init_kv_bh = init_state_kv_ptr + bh * BLOCK_D * BLOCK_D
        state_curr = tl.load(init_kv_bh + offs_dd, mask=mask_dd, other=0.0).to(tl.float32)
        init_z_bh = init_state_z_ptr + bh * BLOCK_D
        state_z_curr = tl.load(init_z_bh + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    else:
        state_curr = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        state_z_curr = tl.zeros([BLOCK_D], dtype=tl.float32)
    if STATE_FP32:
        state_prev = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    else:
        state_prev = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.bfloat16)
    state_z_prev = tl.zeros([BLOCK_D], dtype=tl.float32)

    # ========================================================
    #  Temporal loop — serial over F
    # ========================================================
    for f_iter in range(F):
        if REVERSE:
            q_frame = F - 1 - f_iter
            kv_frame = F - f_iter if f_iter > 0 else 0  # unused at f=0
            skip_update = f_iter == 0
        else:
            q_frame = f_iter
            kv_frame = f_iter
            skip_update = False

        # ---- decay + state snapshot ----
        if REVERSE and f_iter == 0:
            g = 1.0
        else:
            g = tl.load(decay_bh + kv_frame).to(tl.float32)
        state_curr = state_curr * g
        state_z_curr = state_z_curr * g
        if STATE_FP32:
            state_prev = state_curr + 0.0
        else:
            state_prev = state_curr.to(tl.bfloat16)
        state_z_prev = state_z_curr

        if SAVE_STATE:
            st_f = st_bh + q_frame * BLOCK_D * BLOCK_D
            tl.store(st_f + offs_dd, state_prev, mask=mask_dd)
            tl.store(sz_bh + q_frame * BLOCK_D + offs_d, state_z_prev, mask=mask_d)

        # ------------------------------------------
        #  Pass 1 — State Accumulation
        # ------------------------------------------
        if skip_update == False:
            kv_n_base = kv_frame * S
            f_beta = beta_bh + kv_frame * S

            for s0 in range(0, S, BLOCK_S):
                offs_s = s0 + tl.arange(0, BLOCK_S)
                mask_s = offs_s < S
                mask_sd = mask_s[:, None] & mask_d[None, :]
                mask_sd_pair = mask_s[:, None] & mask_d_pair[None, :]
                n_idx = kv_n_base + offs_s

                k_ptrs = qkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d[None, :] * stride_d
                v_ptrs = qkv_bh + n_idx[:, None] * stride_n + 2 * stride_3 + offs_d[None, :] * stride_d
                K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
                V_raw = tl.load(v_ptrs, mask=mask_sd, other=0.0).to(tl.float32)

                if QK_NORM:
                    if USE_PRECOMPUTED_RMS:
                        k_inv_rms = tl.load(k_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0).to(tl.float32)
                    else:
                        k_var = tl.sum(K_raw * K_raw, axis=1) * D_inv
                        k_inv_rms = 1.0 / tl.sqrt(k_var + NORM_EPS)
                    K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
                else:
                    K_normed = K_raw
                K = tl.where(K_normed > 0, K_normed, 0.0) * k_scale

                k_pair_ptrs = qkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d_pair[None, :] * stride_d
                K_pair_raw = tl.load(k_pair_ptrs, mask=mask_sd_pair, other=0.0).to(tl.float32)
                if QK_NORM:
                    K_pair_normed = K_pair_raw * k_inv_rms[:, None] * k_nw_pair[None, :]
                else:
                    K_pair_normed = K_pair_raw
                K_pair = tl.where(K_pair_normed > 0, K_pair_normed, 0.0) * k_scale

                rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
                Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
                Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
                K_rot = K * Cos + K_pair * Sin

                bt = tl.load(f_beta + offs_s, mask=mask_s, other=0.0).to(tl.float32)

                K_rot_dc = K_rot.to(dot_dtype)
                V_pred = tl.dot(K_rot_dc, state_prev.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                dv = (V_raw - V_pred) * bt[:, None]
                state_curr += tl.dot(tl.trans(K_rot), dv, out_dtype=tl.float32, input_precision="tf32")

                z_hat = tl.sum(K * state_z_prev[None, :], axis=1)
                dz = (1.0 - z_hat) * bt
                state_z_curr += tl.sum(K * dz[:, None], axis=0)

        if SAVE_STATE:
            stc_f = stc_bh + q_frame * BLOCK_D * BLOCK_D
            tl.store(stc_f + offs_dd, state_curr, mask=mask_dd)
            tl.store(szc_bh + q_frame * BLOCK_D + offs_d, state_z_curr, mask=mask_d)

        # ------------------------------------------
        #  Pass 2 — Output (reads state_curr, inclusive)
        # ------------------------------------------
        state_out = state_curr.to(dot_dtype)
        state_z_out = state_z_curr
        q_n_base = q_frame * S

        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            mask_s = offs_s < S
            mask_sd = mask_s[:, None] & mask_d[None, :]
            mask_sd_pair = mask_s[:, None] & mask_d_pair[None, :]
            n_idx = q_n_base + offs_s

            q_ptrs = qkv_bh + n_idx[:, None] * stride_n + 0 * stride_3 + offs_d[None, :] * stride_d
            q_pair_ptrs = qkv_bh + n_idx[:, None] * stride_n + 0 * stride_3 + offs_d_pair[None, :] * stride_d
            Q_raw = tl.load(q_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
            Q_pair_raw = tl.load(q_pair_ptrs, mask=mask_sd_pair, other=0.0).to(tl.float32)

            if QK_NORM:
                if USE_PRECOMPUTED_RMS:
                    q_inv_rms = tl.load(q_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0).to(tl.float32)
                else:
                    q_var = tl.sum(Q_raw * Q_raw, axis=1) * D_inv
                    q_inv_rms = 1.0 / tl.sqrt(q_var + NORM_EPS)
                Q_normed = Q_raw * q_inv_rms[:, None] * q_nw[None, :]
                Q_pair_normed = Q_pair_raw * q_inv_rms[:, None] * q_nw_pair[None, :]
            else:
                Q_normed = Q_raw
                Q_pair_normed = Q_pair_raw
            Q = tl.where(Q_normed > 0, Q_normed, 0.0)
            Q_pair = tl.where(Q_pair_normed > 0, Q_pair_normed, 0.0)

            rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
            Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
            Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
            Q_rot = Q * Cos + Q_pair * Sin

            num = tl.dot(Q_rot.to(dot_dtype), state_out, out_dtype=tl.float32, input_precision=dot_ip)
            den = tl.sum(Q * state_z_out[None, :], axis=1)

            result = num / (den[:, None] + EPS)
            out_ptrs = out_bh + n_idx[:, None] * (H * D) + offs_d[None, :]
            num_ptrs = num_bh + n_idx[:, None] * (H * D) + offs_d[None, :]
            tl.store(out_ptrs, result.to(tl.bfloat16), mask=mask_sd)
            tl.store(num_ptrs, num.to(tl.bfloat16), mask=mask_sd)
            tl.store(den_bh + n_idx, den.to(tl.bfloat16), mask=mask_s)

    if SAVE_FINAL_STATE:
        final_kv_bh = final_state_kv_ptr + bh * BLOCK_D * BLOCK_D
        tl.store(final_kv_bh + offs_dd, state_curr, mask=mask_dd)
        final_z_bh = final_state_z_ptr + bh * BLOCK_D
        tl.store(final_z_bh + offs_d, state_z_curr, mask=mask_d)


# =====================================================================
#  Python wrappers
# =====================================================================


def prepare_rope_tables(rotary_emb, N: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex rotary_emb `(1, 1, N, D//2)` → expanded (N, D) cos/sin tables.

    Encodes the interleaved-pair rotation
        y[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        y[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
    as  y[d] = x[d]*cos_exp[d] + x[d^1]*sin_exp[d]
    where sin_exp[2i] = -sin[i], sin_exp[2i+1] = +sin[i].

    Returns (cos_exp, sin_exp) both (N, D) float32, contiguous.
    """
    if rotary_emb is None:
        return (
            torch.ones(N, D, device=device, dtype=torch.float32),
            torch.zeros(N, D, device=device, dtype=torch.float32),
        )
    freqs = rotary_emb.squeeze(0).squeeze(0)  # (N, D//2) complex
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1)
    rope_sin = torch.stack([-sin_half, sin_half], dim=-1).reshape(N, D)
    return rope_cos.contiguous(), rope_sin.contiguous()


def _precompute_inv_rms(qkv: torch.Tensor, idx: int, C: int, eps: float = 1e-5) -> torch.Tensor:
    """Compute 1/RMS for one component of QKV over the full C = H*D channel dim.

    Args:
      qkv:   (B, N, 3, H, D)
      idx:   0 for Q, 1 for K, 2 for V
      C:     H*D (channel count)
      eps:   RMSNorm epsilon

    Returns:
      inv_rms: (B, N) float32
    """
    raw = qkv[:, :, idx].float()  # (B, N, H, D)
    sq_sum = (raw * raw).sum(dim=(-2, -1))  # (B, N)
    return torch.rsqrt(sq_sum / C + eps)


# =====================================================================
#  Fused single-pass Q+K inverse-RMS Triton kernel
# =====================================================================
# Single Triton launch that reads each `(b, n)` row of `qkv` once and emits
# both `q_inv_rms[b, n]` and `k_inv_rms[b, n]`. Replaces two separate PyTorch
# scans (cast→square→sum→rsqrt) over `qkv[:, :, 0]` and `qkv[:, :, 1]`.
#
# Layout assumed: `qkv` is (B, N, 3, H, D) contiguous, so the C = H*D channels
# for a given (b, n, qkv_idx) live in a contiguous memory span.


@triton.jit
def _fused_qk_inv_rms_kernel(
    qkv_ptr,  # *T_in     (B, N, 3, H, D), contiguous
    q_inv_rms_ptr,  # *float32  (B, N)
    k_inv_rms_ptr,  # *float32  (B, N)
    N: tl.constexpr,
    C: tl.constexpr,  # H * D
    eps,
    BLOCK_C: tl.constexpr,
):
    bn_id = tl.program_id(0)
    qkv_row_stride = 3 * C
    row_base = bn_id * qkv_row_stride
    q_base = row_base
    k_base = row_base + C

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    q_vals = tl.load(qkv_ptr + q_base + offs, mask=mask, other=0.0).to(tl.float32)
    k_vals = tl.load(qkv_ptr + k_base + offs, mask=mask, other=0.0).to(tl.float32)

    q_sq = tl.sum(q_vals * q_vals, axis=0)
    k_sq = tl.sum(k_vals * k_vals, axis=0)

    inv_c = 1.0 / C
    q_inv = tl.rsqrt(q_sq * inv_c + eps)
    k_inv = tl.rsqrt(k_sq * inv_c + eps)

    tl.store(q_inv_rms_ptr + bn_id, q_inv)
    tl.store(k_inv_rms_ptr + bn_id, k_inv)


def fused_qk_inv_rms(
    qkv: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass Triton fused Q+K inverse-RMS.

    Replaces ``(_precompute_inv_rms(qkv, 0, C, eps), _precompute_inv_rms(qkv, 1, C, eps))``
    with one launch that reads each ``(b, n)`` row of ``qkv`` exactly once.

    Args:
      qkv: (B, N, 3, H, D) contiguous tensor, any fp dtype.
      eps: RMSNorm epsilon.

    Returns:
      (q_inv_rms, k_inv_rms), each (B, N) float32 contiguous.
    """
    assert qkv.is_contiguous(), "qkv must be contiguous (B, N, 3, H, D)"
    assert qkv.dim() == 5 and qkv.shape[2] == 3, f"expected (B, N, 3, H, D), got {tuple(qkv.shape)}"
    B, N, _, H, D = qkv.shape
    C = H * D
    q_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    k_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    BLOCK_C = triton.next_power_of_2(C)
    _fused_qk_inv_rms_kernel[(B * N,)](
        qkv,
        q_inv_rms,
        k_inv_rms,
        N=N,
        C=C,
        eps=eps,
        BLOCK_C=BLOCK_C,
    )
    return q_inv_rms, k_inv_rms


def fused_gdn_func(
    qkv: torch.Tensor,  # (B, N, 3, H, D)
    q_inv_rms: torch.Tensor,  # (B, N) float32
    k_inv_rms: torch.Tensor,  # (B, N) float32
    q_norm_weight: torch.Tensor,  # (C,) = (H*D,) float32
    k_norm_weight: torch.Tensor,  # (C,) float32
    rope_cos: torch.Tensor,  # (N, D) float32
    rope_sin: torch.Tensor,  # (N, D) float32
    beta: torch.Tensor,  # (B, H, F, S)
    decay: torch.Tensor,  # (B, H, F)
    F: int,
    S: int,
    k_scale: float,
    eps: float = 1e-6,
    reverse: bool = False,
    init_state_kv: torch.Tensor | None = None,
    init_state_z: torch.Tensor | None = None,
    save_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One direction of fused BiGDN via the unified kernel.

    Args:
      qkv .. eps: see kernel signature.
      reverse: forward (False) or anti-causal (True) scan.
      init_state_kv: optional ``(B*H, BLOCK_D, BLOCK_D)`` fp32 contiguous
        tensor holding the forward-scan KV state at the END of a prefix
        sequence (i.e., AFTER the prefix's last update, BEFORE any further
        decay applied by this call). When provided, the kernel resumes the
        scan from this state instead of zero. ``BLOCK_D = next_pow2(D)``.
        Only the top-left ``D x D`` submatrix of the tile is read.
      init_state_z: optional ``(B*H, BLOCK_D)`` fp32 contiguous companion
        for the Z denominator state. Must be provided iff ``init_state_kv``
        is provided.
      save_final_state: when True, allocate fresh fp32 zero buffers for the
        final KV / Z state (after the last frame's update) and pass them to
        the kernel for write-out. Returns the buffers as additional outputs.

    Returns:
      ``(num, den)`` — bf16 numerator ``(B, N, H, D)`` and denominator
      ``(B, H, N)`` before divide.

      When ``save_final_state=True``, also returns
      ``(final_state_kv, final_state_z)`` fp32 with shapes
      ``(B*H, BLOCK_D, BLOCK_D)`` and ``(B*H, BLOCK_D)``.

    Raises:
      NotImplementedError: if any state I/O argument is set together with
        ``reverse=True``. The kernel supports state passing in both
        directions but the AR-cache use case only needs the forward
        direction; reverse-direction semantics are intentionally undefined
        here to avoid silent misuse.
    """
    # Chunkwise integration (2026-05-06): dispatch ALL paths to chunkwise — both
    # the stateless bidi case and the AR cached-state path. Routing both to
    # chunkwise preserves split-equivalence (out_full ≡ out_prefix + out_suffix
    # with state save/load); previously a mixed dispatch (chunkwise non-state
    # + legacy fused stateful) failed split-equivalence at bf16 precision because
    # chunkwise and legacy produce slightly different numerics on the same math.
    # Bypass via env: FUSED_GDN_FORCE_LEGACY=1.
    if os.environ.get("FUSED_GDN_FORCE_LEGACY", "0") != "1":
        from diffusion.model.ops.fused_gdn_chunkwise import (
            fused_gdn_func_chunkwise,
            fused_gdn_stateful_chunkwise,
        )

        # Validate state I/O args upfront — preserves the legacy fused_gdn_func's
        # validation contract (callers depend on these specific ValueError /
        # NotImplementedError signatures, e.g., test_state_validation).
        if (init_state_kv is None) != (init_state_z is None):
            raise ValueError(
                "fused_gdn_func: init_state_kv and init_state_z must be provided together "
                "(both None or both fp32 tensors)."
            )
        if reverse and (init_state_kv is not None or save_final_state):
            raise NotImplementedError(
                "fused_gdn_func: state passing (init_state_kv / init_state_z / "
                "save_final_state) is only supported for the forward direction "
                "(reverse=False)."
            )
        if init_state_kv is not None:
            B_q, _N, _three, H_q, D_q = qkv.shape
            BLOCK_D_q = triton.next_power_of_2(D_q)
            expected_kv = (B_q * H_q, BLOCK_D_q, BLOCK_D_q)
            expected_z = (B_q * H_q, BLOCK_D_q)
            if tuple(init_state_kv.shape) != expected_kv:
                raise ValueError(
                    f"fused_gdn_func: init_state_kv shape {tuple(init_state_kv.shape)} != " f"expected {expected_kv}."
                )
            if tuple(init_state_z.shape) != expected_z:
                raise ValueError(
                    f"fused_gdn_func: init_state_z shape {tuple(init_state_z.shape)} != " f"expected {expected_z}."
                )
            if init_state_kv.dtype != torch.float32 or init_state_z.dtype != torch.float32:
                raise ValueError(
                    f"fused_gdn_func: init_state_kv/init_state_z must be fp32 "
                    f"(got {init_state_kv.dtype}, {init_state_z.dtype})."
                )
            if not init_state_kv.is_contiguous() or not init_state_z.is_contiguous():
                raise ValueError("fused_gdn_func: init_state_kv / init_state_z must be contiguous.")

        # Stateless path
        if init_state_kv is None and init_state_z is None and not save_final_state:
            return fused_gdn_func_chunkwise(
                qkv,
                q_inv_rms,
                k_inv_rms,
                q_norm_weight,
                k_norm_weight,
                rope_cos,
                rope_sin,
                beta,
                decay,
                F=F,
                S=S,
                k_scale=k_scale,
                eps=eps,
                reverse=reverse,
            )

        # Stateful path: shape-adapt state I/O between the public
        # state layout and the chunkwise-stateful kernel's storage.
        # Public state_kv: (B*H, BLOCK_D, BLOCK_D) row-major as
        #   M[K_feat, V_feat].
        # Chunkwise stateful: takes user-facing (B, H, D_in, D_out) and
        #   transposes internally to (B*H, D_out, D_in) for kernel storage.
        # Public state_z:    (B*H, BLOCK_D).
        # Chunkwise stateful: (B, H, D, 1) or (B, H, D).
        B, N, _three, H, D = qkv.shape
        BLOCK_D = triton.next_power_of_2(D)

        ck_init_kv = None
        ck_init_z = None
        if init_state_kv is not None:
            # (B*H, BLOCK_D, BLOCK_D) → (B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D]
            # then transpose so chunkwise's internal `.transpose(-1, -2)` undoes it.
            ck_init_kv = init_state_kv.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].transpose(-1, -2).contiguous()
        if init_state_z is not None:
            # (B*H, BLOCK_D) → (B, H, BLOCK_D)[:, :, :D] → (B, H, D, 1)
            ck_init_z = init_state_z.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()

        result = fused_gdn_stateful_chunkwise(
            qkv,
            q_inv_rms,
            k_inv_rms,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            beta,
            decay,
            F=F,
            S=S,
            k_scale=k_scale,
            eps=eps,
            reverse=reverse,
            init_state_kv=ck_init_kv,
            init_state_z=ck_init_z,
            return_final_state=save_final_state,
        )

        if not save_final_state:
            return result  # (num, den)

        num, den, ck_state_kv, ck_state_z = result
        # chunkwise returns state_kv as (B, H, D, D), [K_feat, V_feat] (post
        # its internal back-transpose). Convert to the public layout
        # (B*H, BLOCK_D, BLOCK_D) by transposing back to internal storage
        # and padding to BLOCK_D.
        out_state_kv = torch.zeros(B * H, BLOCK_D, BLOCK_D, device=qkv.device, dtype=torch.float32)
        out_state_kv[:, :D, :D] = ck_state_kv.transpose(-1, -2).reshape(B * H, D, D)
        out_state_z = torch.zeros(B * H, BLOCK_D, device=qkv.device, dtype=torch.float32)
        out_state_z[:, :D] = ck_state_z.squeeze(-1).reshape(B * H, D)
        return num, den, out_state_kv, out_state_z

    B, N, three, H, D = qkv.shape
    assert three == 3

    BLOCK_D, BLOCK_S, dot_prec, state_fp32, nw, cfg, beta, decay = _prepare_launch(D, beta, decay)

    has_init_state = init_state_kv is not None or init_state_z is not None
    if reverse and (has_init_state or save_final_state):
        raise NotImplementedError(
            "fused_gdn_func: state passing (init_state_kv / init_state_z / "
            "save_final_state) is only supported for the forward direction "
            "(reverse=False). The chunk-causal anti-causal pass resets state "
            "per chunk and has no global cross-prefix state to cache."
        )

    if has_init_state:
        if init_state_kv is None or init_state_z is None:
            raise ValueError(
                "fused_gdn_func: init_state_kv and init_state_z must be "
                "provided together (got "
                f"init_state_kv={'set' if init_state_kv is not None else 'None'}, "
                f"init_state_z={'set' if init_state_z is not None else 'None'})."
            )
        expected_kv_shape = (B * H, BLOCK_D, BLOCK_D)
        expected_z_shape = (B * H, BLOCK_D)
        if tuple(init_state_kv.shape) != expected_kv_shape:
            raise ValueError(
                f"fused_gdn_func: init_state_kv shape {tuple(init_state_kv.shape)} "
                f"does not match expected {expected_kv_shape} (BLOCK_D=next_pow2(D)={BLOCK_D})."
            )
        if tuple(init_state_z.shape) != expected_z_shape:
            raise ValueError(
                f"fused_gdn_func: init_state_z shape {tuple(init_state_z.shape)} "
                f"does not match expected {expected_z_shape}."
            )
        if init_state_kv.dtype != torch.float32 or init_state_z.dtype != torch.float32:
            raise ValueError(
                "fused_gdn_func: init_state_kv and init_state_z must be fp32 "
                f"(got {init_state_kv.dtype}, {init_state_z.dtype})."
            )
        if not init_state_kv.is_contiguous() or not init_state_z.is_contiguous():
            raise ValueError("fused_gdn_func: init_state_kv and init_state_z must be contiguous.")
        if init_state_kv.device != qkv.device or init_state_z.device != qkv.device:
            raise ValueError("fused_gdn_func: init_state_* must live on the same device as qkv.")
        load_init = 1
        init_kv_arg = init_state_kv
        init_z_arg = init_state_z
    else:
        load_init = 0
        init_kv_arg = None  # placeholder set below

    if save_final_state:
        final_state_kv = torch.zeros(B * H, BLOCK_D, BLOCK_D, device=qkv.device, dtype=torch.float32)
        final_state_z = torch.zeros(B * H, BLOCK_D, device=qkv.device, dtype=torch.float32)
        save_final = 1
    else:
        final_state_kv = None
        final_state_z = None
        save_final = 0

    num = torch.empty(B, N, H, D, device=qkv.device, dtype=qkv.dtype)
    den = torch.empty(B, H, N, device=qkv.device, dtype=qkv.dtype)
    dummy = torch.empty(1, device=qkv.device, dtype=torch.float32)

    # Resolve pointer args for the unused slots to a shared scratch tensor;
    # the kernel compiles the corresponding load/store away when the
    # constexpr flag is 0.
    init_kv_ptr = init_kv_arg if load_init else dummy
    init_z_ptr = init_z_arg if load_init else dummy
    final_kv_ptr = final_state_kv if save_final else dummy
    final_z_ptr = final_state_z if save_final else dummy

    _fused_gdn_kernel[(B * H,)](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta,
        decay,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        num,  # `out_ptr` reuses `num` buffer (result immediately overwritten below)
        num,
        den,
        dummy,
        dummy,
        dummy,
        dummy,  # saved-state dummies (SAVE_STATE=0)
        init_kv_ptr,
        init_z_ptr,
        final_kv_ptr,
        final_z_ptr,
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=1e-5,  # unused with USE_PRECOMPUTED_RMS=1
        EPS=eps,
        QK_NORM=1,
        USE_PRECOMPUTED_RMS=1,
        STATE_FP32=1 if state_fp32 else 0,
        DOT_PRECISION=dot_prec,
        REVERSE=1 if reverse else 0,
        SAVE_STATE=0,
        LOAD_INIT_STATE=load_init,
        SAVE_FINAL_STATE=save_final,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        num_stages=cfg["num_stages"],
        num_warps=nw,
    )
    if save_final_state:
        return num, den, final_state_kv, final_state_z
    return num, den


def fused_bigdn_func(
    qkv: torch.Tensor,  # (B, N, 3, H, D)
    q_inv_rms: torch.Tensor,  # (B, N) — pre-computed via `_precompute_inv_rms`
    k_inv_rms: torch.Tensor,  # (B, N)
    q_norm_weight: torch.Tensor,  # (C,) float32
    k_norm_weight: torch.Tensor,  # (C,)
    rope_cos: torch.Tensor,  # (N, D)
    rope_sin: torch.Tensor,  # (N, D)
    beta: torch.Tensor,  # (B, H, F, S)
    decay: torch.Tensor,  # (B, H, F)
    F: int,
    S: int,
    k_scale: float,
    eps: float = 1e-6,
    # -- chunk-causal extensions (not in upstream; see adapter notes below) --
    qkv_bwd: torch.Tensor | None = None,
    beta_bwd: torch.Tensor | None = None,
    decay_bwd: torch.Tensor | None = None,
    q_inv_rms_bwd: torch.Tensor | None = None,
    k_inv_rms_bwd: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full bidirectional fused GDN.

    Returns: out (B, N, H, D) bf16 = (num_fwd + num_bwd) / (den_fwd + den_bwd + eps).

    Chunk-causal extensions (optional):
      For chunk-causal GDN we need to zero state at chunk boundaries in the
      BACKWARD direction only. Pass separately pre-processed backward tensors
      (decay_bwd with zeros at boundary frames, and optionally qkv_bwd /
      beta_bwd with K/V or beta zeroed at boundary frames). If any `*_bwd`
      argument is None, the forward tensor is reused.
    """
    if (
        os.environ.get("FUSED_GDN_FORCE_LEGACY", "0") != "1"
        and qkv_bwd is None
        and beta_bwd is None
        and decay_bwd is None
        and q_inv_rms_bwd is None
        and k_inv_rms_bwd is None
    ):
        from diffusion.model.ops.fused_gdn_chunkwise import fused_bigdn_bidi_chunkwise

        return fused_bigdn_bidi_chunkwise(
            qkv,
            q_inv_rms,
            k_inv_rms,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            beta,
            decay,
            F=F,
            S=S,
            k_scale=k_scale,
            eps=eps,
        )

    num_fwd, den_fwd = fused_gdn_func(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        beta,
        decay,
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
        reverse=False,
    )
    num_bwd, den_bwd = fused_gdn_func(
        qkv if qkv_bwd is None else qkv_bwd,
        q_inv_rms if q_inv_rms_bwd is None else q_inv_rms_bwd,
        k_inv_rms if k_inv_rms_bwd is None else k_inv_rms_bwd,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        beta if beta_bwd is None else beta_bwd,
        decay if decay_bwd is None else decay_bwd,
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
        reverse=True,
    )
    # num: (B, N, H, D), den: (B, H, N). Fuse then divide.
    total_num = num_fwd + num_bwd
    total_den = (den_fwd + den_bwd).permute(0, 2, 1).unsqueeze(-1)  # (B, N, H, 1)
    return total_num / (total_den + eps)


# =====================================================================
#  Reference skeleton: how the upstream block wires up the kernel
# =====================================================================
# This class is NOT a runnable module — it has no __init__. It documents the
# expected call convention inside a trained BiGDN inference block, extracted
# verbatim from `StateCachedGDNChunkCasualNATConvChunk._fused_statecached_forward`.
# When adapting into the playground, make sure to:
#
#   * Use the *same* QKV linear as the torch path (Conv1d k=1 == Linear; weight
#     is `(3C, C, 1)` squeeze to `(3C, C)` then `F.linear(x, qkv_w)`).
#   * Pre-compute `q_inv_rms`, `k_inv_rms` from RAW qkv over the full C dim.
#   * Full-channel RMSNorm weights: concatenate `self.q_norm.weight` (length C)
#     and `self.k_norm.weight` across the kernel's per-head slice pid_h*D:pid_h*D+D.
#   * `k_scale = (D**-0.5) * (S**-0.5)` — note S = H_s * W_s (spatial count),
#     distinct from channel dim C.
#   * Short-conv: upstream state-cached block SKIPS short-conv. Our SANA-WM
#     ChunkCausalGDN *does* apply short-conv on K. The adapter must do the
#     short-conv in PyTorch and write convolved K back into qkv[:,:,1] before
#     calling the kernel.
#   * Chunk-causal backward: zero `decay_bwd` (and optionally k/v in `qkv_bwd`)
#     at interior chunk boundaries — see `fused_bigdn_func`'s `*_bwd` args.


class FusedStateCachedForwardRef:  # noqa: D401
    """Reference forward method from upstream block (skeleton, not runnable).

    Call graph for a real block:
        qkv (B,N,3,H,D)  ←  F.linear(x, concat(q.W, k.W, v.W))
        beta, decay      ←  self._compute_frame_gates(x, HW)
        q_inv_rms, k_inv_rms  ←  _precompute_inv_rms(qkv, 0/1, C, 1e-5)
        rope_cos, rope_sin    ←  prepare_rope_tables(rotary_emb[:, :, -N:], N, D)
        qkv_out (B,N,H,D)     ←  fused_bigdn_func(...)
        out (B,N,C)           ←  qkv_out.reshape(B, N, C)
        + output-gate + proj as usual.
    """

    def _fused_statecached_forward(self, x, HW, rotary_emb, kv_cache=None, save_kv_cache=False):
        """Fused BiGDN forward with state caching (Triton mega-kernel path).

        THIS IS A REFERENCE SNIPPET — do not instantiate this class. It exists
        only to document the intended wiring of the mega-kernel inside a
        trained block.
        """
        import torch.nn as nn  # local import to silence lint in skeleton

        B, N, C = x.shape
        T, H_s, W_s = HW
        S = H_s * W_s
        H, D = self.heads, self.dim
        dtype_orig = x.dtype

        # Combined QKV matmul (Conv1d(k=1) == Linear)
        q_w = self.q.weight.squeeze(-1)
        k_w = self.k.weight.squeeze(-1)
        v_w = self.v.weight
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv = F.linear(x, qkv_w).reshape(B, N, 3, H, D)

        beta, decay = self._compute_frame_gates(x, HW)
        beta = beta.permute(0, 3, 1, 2).contiguous()  # (B, H, T, S)

        # RoPE — only current chunk positions (state-cached path slices tail N)
        rotary_emb_local = rotary_emb[:, :, -N:] if rotary_emb is not None else None
        rope_cos, rope_sin = prepare_rope_tables(rotary_emb_local, N, D, x.device)
        k_scale = (D**-0.5) * (S**-0.5)

        # Norm weights: full-channel (C,)
        qk_norm = not isinstance(self.q_norm, nn.Identity)
        if qk_norm:
            q_nw = self.q_norm.weight.float()
            k_nw = self.k_norm.weight.float()
        else:
            q_nw = torch.ones(C, device=x.device, dtype=torch.float32)
            k_nw = torch.ones(C, device=x.device, dtype=torch.float32)

        q_inv_rms = _precompute_inv_rms(qkv, 0, C, 1e-5)
        k_inv_rms = _precompute_inv_rms(qkv, 1, C, 1e-5)

        # Forward (no state init in this playground adapter — training state
        # caching stripped). Inference call: see fused_bigdn_func.
        num_fwd, den_fwd = fused_gdn_func(
            qkv,
            q_inv_rms,
            k_inv_rms,
            q_nw,
            k_nw,
            rope_cos,
            rope_sin,
            beta,
            decay,
            F=T,
            S=S,
            k_scale=k_scale,
            eps=self.eps,
            reverse=False,
        )
        # Backward
        num_bwd, den_bwd = fused_gdn_func(
            qkv,
            q_inv_rms,
            k_inv_rms,
            q_nw,
            k_nw,
            rope_cos,
            rope_sin,
            beta,
            decay,
            F=T,
            S=S,
            k_scale=k_scale,
            eps=self.eps,
            reverse=True,
        )

        # Fuse forward + backward
        total_num = num_fwd + num_bwd
        total_den = (den_fwd + den_bwd).permute(0, 2, 1).unsqueeze(-1)  # (B, N, H, 1)
        out = total_num / (total_den + self.eps)
        out = out.reshape(B, N, C)

        if self.use_output_gate and self.output_gate is not None:
            out = out * F.silu(self.output_gate(x).to(torch.float32)).to(dtype_orig)

        return self.proj(out)


# =====================================================================
#  Backward / autograd Functions
# =====================================================================
# Adds:
#   1. ``_fused_gdn_bwd_kernel`` -- Triton jit kernel that replays the
#      forward recurrence in reverse time using per-frame state snapshots
#      written by the forward kernel under ``SAVE_STATE=1``.
#   2. ``_run_fwd_save`` -- helper that runs the existing forward
#      ``_fused_gdn_kernel`` with ``SAVE_STATE=1``. Adapted to pass our
#      extra ``init_state_kv_ptr / init_state_z_ptr / final_state_kv_ptr /
#      final_state_z_ptr`` pointers + ``LOAD_INIT_STATE / SAVE_FINAL_STATE``
#      constexpr flags (all unused on the autograd path -> dummy / 0).
#   3. ``FusedGDNFunction`` -- autograd Function for unidirectional GDN
#      with ``QK_NORM=1`` (in-kernel per-head RMSNorm).
#   4. ``FusedBiGDNFunction`` -- autograd Function for bidirectional BiGDN.
#      Pre-normalizes Q/K in PyTorch with full-channel RMSNorm, runs the
#      forward kernel twice with ``QK_NORM=0 + SAVE_STATE=1``, fuses
#      ``(num_fwd + num_bwd) / (den_fwd + den_bwd + eps)``. Backward
#      computes ``dnum / dden`` from upstream ``dout`` and runs the bwd
#      kernel twice with ``BIDI_MODE=1``.
#   5. Python wrappers ``fused_gdn_forward_with_grad`` /
#      ``fused_bigdn_forward_with_grad`` -- drop-in autograd-enabled
#      replacements for ``fused_gdn_func`` / ``fused_bigdn_func``.
#
# Chunk-causal autograd support: ``FusedBiGDNFunction`` (and the public
# wrapper ``fused_bigdn_forward_with_grad``) accepts optional
# ``beta_bwd`` / ``decay_bwd`` overrides for the reverse-direction
# kernel call -- exactly the same masking convention used by the
# inference path ``fused_bigdn_func``.  When provided, the reverse
# direction's forward and backward kernels both run on these masked
# tensors, and the backward returns separate gradient tensors
# (``dbeta_bwd`` / ``ddecay_bwd``) so autograd can route them back
# through any ``clone() + index = 0`` masking the caller applied.


@triton.jit
def _fused_gdn_bwd_kernel(
    # ---- original inputs ----
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    beta_ptr,
    decay_ptr,
    q_norm_w_ptr,
    k_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    # ---- saved from forward ----
    saved_state_ptr,  # (B*H, F, BLOCK_D, BLOCK_D) -- state_prev snapshots
    saved_z_ptr,  # (B*H, F, BLOCK_D)
    saved_state_curr_ptr,  # (B*H, F, BLOCK_D, BLOCK_D) -- state_curr (after update)
    saved_z_curr_ptr,  # (B*H, F, BLOCK_D)
    # ---- upstream gradient / pre-computed dnum ----
    dout_ptr,  # GDN mode: (B, N, H, D) upstream grad. BiDI mode: pre-computed dnum
    # ---- BiDI mode: external dden ----
    dden_ext_ptr,  # BiDI mode: (B, H, N) pre-computed dden. GDN mode: unused
    # ---- output gradients ----
    dqkv_ptr,  # (B, N, 3, H, D) -- same layout as qkv
    dbeta_ptr,  # (B, H, F, S)
    ddecay_ptr,  # (B, H, F)
    # ---- dims ----
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    EPS: tl.constexpr,
    QK_NORM: tl.constexpr,
    STATE_FP32: tl.constexpr,
    REVERSE_BWD: tl.constexpr,  # 0=backward of forward GDN, 1=backward of reversed GDN
    BIDI_MODE: tl.constexpr,  # 0=GDN (compute dnum/dden), 1=BiGDN (use provided)
    DOT_PRECISION: tl.constexpr,  # 0=bf16 TC, 1=TF32 TC, 2=IEEE fp32
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = pid // H
    pid_h = pid % H
    N: tl.constexpr = F * S
    bh = pid_b * H + pid_h

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    dqkv_bh = dqkv_ptr + pid_b * stride_b + pid_h * stride_h
    dout_bh = dout_ptr + pid_b * (N * H * D) + pid_h * D
    beta_bh = beta_ptr + bh * (F * S)
    decay_bh = decay_ptr + bh * F
    dbeta_bh = dbeta_ptr + bh * (F * S)
    ddecay_bh = ddecay_ptr + bh * F
    st_bh = saved_state_ptr + bh * F * BLOCK_D * BLOCK_D
    sz_bh = saved_z_ptr + bh * F * BLOCK_D
    stc_bh = saved_state_curr_ptr + bh * F * BLOCK_D * BLOCK_D
    szc_bh = saved_z_curr_ptr + bh * F * BLOCK_D
    if BIDI_MODE:
        dden_ext_bh = dden_ext_ptr + bh * N

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D

    nw_offset = pid_h * D
    if QK_NORM:
        q_nw = tl.load(q_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        q_nw_pair = tl.load(q_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0).to(tl.float32)
        k_nw_pair = tl.load(k_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0).to(tl.float32)

    D_inv = 1.0 / D
    k_scale = K_SCALE

    # Dot precision: mirror forward kernel
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    # Gradient matmuls: always use bf16 TC + TF32 input precision (matching PyTorch backward)
    grad_dtype = tl.bfloat16
    grad_ip: tl.constexpr = "tf32"

    # ---- Gradient state accumulators (reverse time) ----
    dstate = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    dstate_z = tl.zeros([BLOCK_D], dtype=tl.float32)

    for f_rev in range(F):
        # Backward iterates in reverse of forward direction.
        if REVERSE_BWD:
            f = f_rev  # backward of reversed GDN: iterate 0..F-1
            # In fwd_save REVERSE, q_frame=f had kv_frame=f+1 (or skip at f=F-1).
            kv_frame_bwd = f + 1 if f < F - 1 else f
            skip_bwd = f == F - 1  # f=F-1 was dummy step (f_iter=0 in fwd)
        else:
            f = F - 1 - f_rev  # backward of forward GDN: iterate F-1..0
            kv_frame_bwd = f
            skip_bwd = False
        q_n_base = f * S
        kv_n_base = kv_frame_bwd * S
        f_beta = beta_bh + kv_frame_bwd * S

        # ---- Load state_curr for Pass 2 output (both directions use inclusive) ----
        st_f = st_bh + f * BLOCK_D * BLOCK_D
        offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
        mask_dd = mask_d[:, None] & mask_d[None, :]

        stc_f = stc_bh + f * BLOCK_D * BLOCK_D
        P_state = tl.load(stc_f + offs_dd, mask=mask_dd, other=0.0)
        Pz_state = tl.load(szc_bh + f * BLOCK_D + offs_d, mask=mask_d, other=0.0)
        if STATE_FP32 == 0:
            P_state = P_state.to(tl.float32)

        # Decay: for REVERSE_BWD, use decay[kv_frame] matching fwd_save.
        if REVERSE_BWD and skip_bwd:
            g = 1.0
        elif REVERSE_BWD:
            g = tl.load(decay_bh + kv_frame_bwd).to(tl.float32)
        else:
            g = tl.load(decay_bh + f).to(tl.float32)

        # ========================================================
        # Pass 2 backward: Output gradients -> dQ, dstate, dstate_z
        # ========================================================
        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            mask_s = offs_s < S
            mask_sd = mask_s[:, None] & mask_d[None, :]
            mask_sd_pair = mask_s[:, None] & mask_d_pair[None, :]
            n_idx = q_n_base + offs_s  # Q data from q_frame

            # Load dout; recompute Q, Q_pair, Q_rot, num, den from saved P_f/Pz_f.
            dout_ptrs = dout_bh + n_idx[:, None] * (H * D) + offs_d[None, :]
            d_out = tl.load(dout_ptrs, mask=mask_sd, other=0.0).to(tl.float32)

            # Recompute Q, Q_pair, Q_rot (same as forward).
            q_ptrs = qkv_bh + n_idx[:, None] * stride_n + 0 * stride_3 + offs_d[None, :] * stride_d
            Q_raw = tl.load(q_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
            q_pair_ptrs = qkv_bh + n_idx[:, None] * stride_n + 0 * stride_3 + offs_d_pair[None, :] * stride_d
            Q_pair_raw = tl.load(q_pair_ptrs, mask=mask_sd_pair, other=0.0).to(tl.float32)

            if QK_NORM:
                q_var = tl.sum(Q_raw * Q_raw, axis=1) * D_inv
                q_inv_rms = 1.0 / tl.sqrt(q_var + NORM_EPS)
                Q_normed = Q_raw * q_inv_rms[:, None] * q_nw[None, :]
                Q_pair_normed = Q_pair_raw * q_inv_rms[:, None] * q_nw_pair[None, :]
            else:
                Q_normed = Q_raw
                Q_pair_normed = Q_pair_raw
            Q = tl.where(Q_normed > 0, Q_normed, 0.0)
            Q_pair = tl.where(Q_pair_normed > 0, Q_pair_normed, 0.0)

            rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
            Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
            Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
            Q_rot = Q * Cos + Q_pair * Sin

            # Compute dnum and dden.
            if BIDI_MODE:
                # BiGDN: dnum and dden pre-computed externally from total num/den.
                dnum = d_out  # dout_ptr already contains pre-computed dnum
                dden = tl.load(dden_ext_bh + n_idx, mask=mask_s, other=0.0).to(tl.float32)
            else:
                # GDN: recompute num/den using direction-appropriate state.
                num_tile = tl.dot(
                    Q_rot.to(dot_dtype), P_state.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip
                )
                den_tile = tl.sum(Q * Pz_state[None, :], axis=1)
                inv_den = 1.0 / (den_tile + EPS)
                dnum = d_out * inv_den[:, None]
                dden = -tl.sum(d_out * num_tile, axis=1) * inv_den * inv_den

            # dstate += Q_rot^T @ dnum  (state contribution from num = Q_rot @ P_state).
            dstate = dstate + tl.dot(
                tl.trans(Q_rot.to(grad_dtype)),
                dnum.to(grad_dtype),
                out_dtype=tl.float32,
                input_precision=grad_ip,
            )

            # dstate_z += sum(dden * Q, axis=0)  (Pz contribution from den = Q . Pz).
            dstate_z += tl.sum(dden[:, None] * Q, axis=0)

            # dQ_rot = dnum @ P_state^T  (uses state that forward's output read).
            dQ_rot = tl.dot(
                dnum.to(grad_dtype),
                tl.trans(P_state.to(grad_dtype)),
                out_dtype=tl.float32,
                input_precision=grad_ip,
            )

            # dQ_from_den = dden * Pz_state.
            dQ_from_den = dden[:, None] * Pz_state[None, :]

            # RoPE inverse for Q: store dQ_rot, reload at paired indices.
            # Store dQ_rot temporarily to dqkv[Q] at normal d positions.
            dq_ptrs = dqkv_bh + n_idx[:, None] * stride_n + 0 * stride_3 + offs_d[None, :] * stride_d
            tl.store(dq_ptrs, dQ_rot.to(tl.bfloat16), mask=mask_sd)

            # Load dQ_rot at paired positions.
            dq_pair_ptrs = dqkv_bh + n_idx[:, None] * stride_n + 0 * stride_3 + offs_d_pair[None, :] * stride_d
            dQ_rot_pair = tl.load(dq_pair_ptrs, mask=mask_sd_pair, other=0.0).to(tl.float32)

            # RoPE inverse: dQ = dQ_rot * Cos - dQ_rot_pair * Sin.
            dQ = dQ_rot * Cos - dQ_rot_pair * Sin + dQ_from_den

            # ReLU backward.
            relu_mask_q = (Q_normed > 0).to(tl.float32)
            dQ_normed = dQ * relu_mask_q

            # Norm backward (QK_NORM) or direct (no norm).
            if QK_NORM:
                gw = dQ_normed * q_nw[None, :]
                corr = tl.sum(gw * Q_raw, axis=1) * D_inv * q_inv_rms * q_inv_rms
                dQ_raw = q_inv_rms[:, None] * (gw - Q_raw * corr[:, None])
            else:
                dQ_raw = dQ_normed

            # Store final dQ_raw to dqkv[Q].
            tl.store(dq_ptrs, dQ_raw.to(tl.bfloat16), mask=mask_sd)

        # Both directions use inclusive output (state_curr), so capture dDelta AFTER Pass 2.
        dDelta = dstate
        dDelta_z = dstate_z

        # ========================================================
        # Reload state_prev for Pass 1 backward (reuse P_state variable)
        # ========================================================
        P_state = tl.load(st_f + offs_dd, mask=mask_dd, other=0.0)
        if STATE_FP32 == 0:
            P_state = P_state.to(tl.float32)
        Pz_state = tl.load(sz_bh + f * BLOCK_D + offs_d, mask=mask_d, other=0.0)

        # ========================================================
        # Pass 1 backward: State update gradients -> dK, dV, dbeta, dstate
        # Skip for REVERSE_BWD dummy frame (skip_bwd=True) to avoid clobbering.
        # ========================================================
        if skip_bwd == False:
            for s0 in range(0, S, BLOCK_S):
                offs_s = s0 + tl.arange(0, BLOCK_S)
                mask_s = offs_s < S
                mask_sd = mask_s[:, None] & mask_d[None, :]
                mask_sd_pair = mask_s[:, None] & mask_d_pair[None, :]
                n_idx = kv_n_base + offs_s  # K/V from kv_frame

                # Recompute K, K_pair, K_rot, V.
                k_ptrs = qkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d[None, :] * stride_d
                v_ptrs = qkv_bh + n_idx[:, None] * stride_n + 2 * stride_3 + offs_d[None, :] * stride_d
                K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
                V_raw = tl.load(v_ptrs, mask=mask_sd, other=0.0).to(tl.float32)

                k_pair_ptrs = qkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d_pair[None, :] * stride_d
                K_pair_raw = tl.load(k_pair_ptrs, mask=mask_sd_pair, other=0.0).to(tl.float32)

                if QK_NORM:
                    k_var = tl.sum(K_raw * K_raw, axis=1) * D_inv
                    k_inv_rms = 1.0 / tl.sqrt(k_var + NORM_EPS)
                    K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
                    K_pair_normed = K_pair_raw * k_inv_rms[:, None] * k_nw_pair[None, :]
                else:
                    K_normed = K_raw
                    K_pair_normed = K_pair_raw
                K = tl.where(K_normed > 0, K_normed, 0.0) * k_scale
                K_pair = tl.where(K_pair_normed > 0, K_pair_normed, 0.0) * k_scale

                rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
                Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
                Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
                K_rot = K * Cos + K_pair * Sin

                bt = tl.load(f_beta + offs_s, mask=mask_s, other=0.0).to(tl.float32)

                # Recompute V_pred and delta_v.
                K_rot_dc = K_rot.to(dot_dtype)
                V_pred = tl.dot(K_rot_dc, P_state.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                delta_v = (V_raw - V_pred) * bt[:, None]

                # ---- KV stream backward ----
                ddelta_v = tl.dot(
                    K_rot.to(grad_dtype),
                    dDelta.to(grad_dtype),
                    out_dtype=tl.float32,
                    input_precision=grad_ip,
                )

                dK_rot_from_delta = tl.dot(
                    delta_v.to(grad_dtype),
                    tl.trans(dDelta.to(grad_dtype)),
                    out_dtype=tl.float32,
                    input_precision=grad_ip,
                )

                dV = ddelta_v * bt[:, None]
                dbeta_kv = tl.sum(ddelta_v * (V_raw - V_pred), axis=1)

                dV_pred = -ddelta_v * bt[:, None]
                dK_rot_from_vpred = tl.dot(
                    dV_pred.to(grad_dtype),
                    tl.trans(P_state.to(grad_dtype)),
                    out_dtype=tl.float32,
                    input_precision=grad_ip,
                )

                dstate = dstate + tl.dot(
                    tl.trans(K_rot.to(grad_dtype)),
                    dV_pred.to(grad_dtype),
                    out_dtype=tl.float32,
                    input_precision=grad_ip,
                )

                dK_rot = dK_rot_from_delta + dK_rot_from_vpred

                # ---- Z stream backward ----
                z_hat = tl.sum(K * Pz_state[None, :], axis=1)
                dz = (1.0 - z_hat) * bt

                ddz = tl.sum(K * dDelta_z[None, :], axis=1)
                dz_hat = -ddz * bt
                dK_z = dDelta_z[None, :] * dz[:, None] + dz_hat[:, None] * Pz_state[None, :]
                dstate_z = dstate_z + tl.sum(dz_hat[:, None] * K, axis=0)

                dbeta_z = ddz * (1.0 - z_hat)
                dbeta_total = dbeta_kv + dbeta_z
                tl.store(dbeta_bh + kv_frame_bwd * S + offs_s, dbeta_total.to(tl.bfloat16), mask=mask_s)

                # ---- RoPE inverse for K ----
                dk_ptrs = dqkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d[None, :] * stride_d
                tl.store(dk_ptrs, dK_rot.to(tl.bfloat16), mask=mask_sd)
                dk_pair_ptrs = dqkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d_pair[None, :] * stride_d
                dK_rot_pair = tl.load(dk_pair_ptrs, mask=mask_sd_pair, other=0.0).to(tl.float32)

                dK_from_kv = dK_rot * Cos - dK_rot_pair * Sin
                dK_total = dK_from_kv + dK_z

                relu_mask_k = (K_normed > 0).to(tl.float32)
                dK_normed = dK_total * k_scale * relu_mask_k

                if QK_NORM:
                    gw_k = dK_normed * k_nw[None, :]
                    corr_k = tl.sum(gw_k * K_raw, axis=1) * D_inv * k_inv_rms * k_inv_rms
                    dK_raw = k_inv_rms[:, None] * (gw_k - K_raw * corr_k[:, None])
                else:
                    dK_raw = dK_normed

                tl.store(dk_ptrs, dK_raw.to(tl.bfloat16), mask=mask_sd)
                dv_ptrs = dqkv_bh + n_idx[:, None] * stride_n + 2 * stride_3 + offs_d[None, :] * stride_d
                tl.store(dv_ptrs, dV.to(tl.bfloat16), mask=mask_sd)

            # ========================================================
            # Decay backward (inside skip_bwd guard)
            # ========================================================
            is_first_frame = f_rev == F - 1
            if is_first_frame:
                ddecay_f = 0.0
            else:
                inv_g = 1.0 / (g + 1e-12)
                ddecay_kv = tl.sum(dstate * P_state) * inv_g
                ddecay_z_val = tl.sum(dstate_z * Pz_state) * inv_g
                ddecay_f = ddecay_kv + ddecay_z_val
            tl.store(ddecay_bh + kv_frame_bwd, ddecay_f)

        # Propagate gradient through decay: dS_{f-1} = g[f] * dP_f.
        dstate = dstate * g
        dstate_z = dstate_z * g


# =====================================================================
#  Forward-with-state-save helper (for autograd Functions)
# =====================================================================


def _run_fwd_save(
    qkv,
    beta,
    decay,
    q_norm_weight,
    k_norm_weight,
    rope_cos,
    rope_sin,
    F: int,
    S: int,
    k_scale: float,
    norm_eps: float,
    eps: float,
    qk_norm: bool,
    reverse: bool,
    cfg,
):
    """Run forward kernel for one direction with ``SAVE_STATE=1``.

    Returns ``(num, den, saved_state, saved_z, saved_state_curr, saved_z_curr)``.
    The forward kernel writes ``out = num/(den+eps)`` first and then overwrites
    the same buffer with raw ``num``, so the returned ``num`` tensor holds raw
    numerator values (matching the BiGDN combine-then-divide convention).
    """
    B, N, three, H, D = qkv.shape
    BLOCK_D, BLOCK_S, dot_prec, state_fp32, nw, _, beta, decay = _prepare_launch(D, beta, decay)

    num_out = torch.empty(B, N, H, D, device=qkv.device, dtype=qkv.dtype)
    den_out = torch.empty(B, H, N, device=qkv.device, dtype=qkv.dtype)
    state_dtype = torch.float32 if state_fp32 else torch.bfloat16
    saved_state = torch.empty(B * H, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=state_dtype)
    saved_z = torch.empty(B * H, F, BLOCK_D, device=qkv.device, dtype=torch.float32)
    saved_state_curr = torch.empty(B * H, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=torch.float32)
    saved_z_curr = torch.empty(B * H, F, BLOCK_D, device=qkv.device, dtype=torch.float32)
    # The kernel writes ``out = num/(den+eps)`` first then overwrites with raw num
    # in the same buffer. Reuse num_out as the (discarded) ``out`` slot so the
    # final contents end up being raw num.
    out_discard = num_out
    dummy_inv = torch.empty(1, device=qkv.device, dtype=torch.float32)

    _fused_gdn_kernel[(B * H,)](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta,
        decay,
        dummy_inv,
        dummy_inv,  # unused inv_rms ptrs (USE_PRECOMPUTED_RMS=0)
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        out_discard,
        num_out,
        den_out,
        saved_state,
        saved_z,
        saved_state_curr,
        saved_z_curr,
        dummy_inv,
        dummy_inv,
        dummy_inv,
        dummy_inv,  # init/final-state dummies
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=norm_eps,
        EPS=eps,
        QK_NORM=1 if qk_norm else 0,
        USE_PRECOMPUTED_RMS=0,
        STATE_FP32=1 if state_fp32 else 0,
        DOT_PRECISION=dot_prec,
        REVERSE=1 if reverse else 0,
        SAVE_STATE=1,
        LOAD_INIT_STATE=0,
        SAVE_FINAL_STATE=0,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        num_stages=cfg["num_stages"],
        num_warps=nw,
    )
    return num_out, den_out, saved_state, saved_z, saved_state_curr, saved_z_curr


# =====================================================================
#  Unidirectional GDN autograd Function
# =====================================================================


class FusedGDNFunction(torch.autograd.Function):
    """Autograd Function for unidirectional fused GDN with in-kernel RMSNorm.

    Forward runs ``_fused_gdn_kernel`` with ``QK_NORM=1`` and ``SAVE_STATE=1``,
    saving per-frame state snapshots for backward. Backward runs
    ``_fused_gdn_bwd_kernel`` with ``BIDI_MODE=0`` (kernel computes
    ``dnum``/``dden`` from upstream ``dout``).
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
        F: int,
        S: int,
        k_scale: float = 1.0,
        norm_eps: float = 1e-6,
        eps: float = 1e-6,
        qk_norm: bool = True,
    ):
        B, N, three, H, D = qkv.shape
        assert three == 3 and N == F * S

        BLOCK_D, BLOCK_S, dot_prec, state_fp32, nw, cfg, beta, decay = _prepare_launch(D, beta, decay)

        if q_norm_weight is None:
            q_norm_weight = torch.ones(D, device=qkv.device, dtype=torch.float32)
        if k_norm_weight is None:
            k_norm_weight = torch.ones(D, device=qkv.device, dtype=torch.float32)

        out = torch.empty(B, N, H, D, device=qkv.device, dtype=qkv.dtype)

        # Saved states for backward.
        state_dtype = torch.float32 if state_fp32 else torch.bfloat16
        saved_state = torch.empty(B * H, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=state_dtype)
        saved_z = torch.empty(B * H, F, BLOCK_D, device=qkv.device, dtype=torch.float32)
        saved_state_curr = torch.empty(B * H, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=torch.float32)
        saved_z_curr = torch.empty(B * H, F, BLOCK_D, device=qkv.device, dtype=torch.float32)

        # Dummy num/den for forward kernel (still writes them but we discard).
        num_out = torch.empty(B, N, H, D, device=qkv.device, dtype=qkv.dtype)
        den_out = torch.empty(B, H, N, device=qkv.device, dtype=qkv.dtype)
        dummy_inv = torch.empty(1, device=qkv.device, dtype=torch.float32)

        _fused_gdn_kernel[(B * H,)](
            qkv,
            qkv.stride(0),
            qkv.stride(1),
            qkv.stride(2),
            qkv.stride(3),
            qkv.stride(4),
            beta,
            decay,
            dummy_inv,
            dummy_inv,  # unused inv_rms ptrs (USE_PRECOMPUTED_RMS=0)
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            out,
            num_out,
            den_out,
            saved_state,
            saved_z,
            saved_state_curr,
            saved_z_curr,
            dummy_inv,
            dummy_inv,
            dummy_inv,
            dummy_inv,  # init/final-state dummies
            H=H,
            F=F,
            S=S,
            D=D,
            K_SCALE=k_scale,
            NORM_EPS=norm_eps,
            EPS=eps,
            QK_NORM=1 if qk_norm else 0,
            USE_PRECOMPUTED_RMS=0,
            STATE_FP32=1 if state_fp32 else 0,
            DOT_PRECISION=dot_prec,
            REVERSE=0,
            SAVE_STATE=1,
            LOAD_INIT_STATE=0,
            SAVE_FINAL_STATE=0,
            BLOCK_D=BLOCK_D,
            BLOCK_S=BLOCK_S,
            num_stages=cfg["num_stages"],
            num_warps=nw,
        )
        del num_out, den_out

        ctx.save_for_backward(
            qkv,
            beta,
            decay,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            saved_state,
            saved_z,
            saved_state_curr,
            saved_z_curr,
        )
        ctx.F = F
        ctx.S = S
        ctx.k_scale = k_scale
        ctx.norm_eps = norm_eps
        ctx.eps = eps
        ctx.qk_norm = qk_norm
        ctx.dot_prec = dot_prec
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            qkv,
            beta,
            decay,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            saved_state,
            saved_z,
            saved_state_curr,
            saved_z_curr,
        ) = ctx.saved_tensors

        B, N, three, H, D = qkv.shape
        F_val = ctx.F
        S = ctx.S

        BLOCK_D, BLOCK_S_BWD, _, _, _, cfg, beta, decay = _prepare_launch(D, beta, decay)
        dqkv = torch.zeros_like(qkv)
        dbeta = torch.zeros_like(beta)
        ddecay = torch.zeros_like(decay)

        # Dummy dden_ext (unused in GDN mode).
        dden_ext = torch.empty(1, device=qkv.device, dtype=torch.float32)

        # Progressive num_warps reduction on tmem overflow.
        nw = cfg["num_warps"]
        if ctx.dot_prec >= 1:
            nw = min(nw, 4)
        while nw >= 1:
            try:
                _fused_gdn_bwd_kernel[(B * H,)](
                    qkv,
                    qkv.stride(0),
                    qkv.stride(1),
                    qkv.stride(2),
                    qkv.stride(3),
                    qkv.stride(4),
                    beta,
                    decay,
                    q_norm_weight,
                    k_norm_weight,
                    rope_cos,
                    rope_sin,
                    saved_state,
                    saved_z,
                    saved_state_curr,
                    saved_z_curr,
                    dout.contiguous(),
                    dden_ext,
                    dqkv,
                    dbeta,
                    ddecay,
                    H=H,
                    F=F_val,
                    S=S,
                    D=D,
                    K_SCALE=ctx.k_scale,
                    NORM_EPS=ctx.norm_eps,
                    EPS=ctx.eps,
                    QK_NORM=1 if ctx.qk_norm else 0,
                    STATE_FP32=1 if cfg["STATE_FP32"] else 0,
                    REVERSE_BWD=0,
                    BIDI_MODE=0,
                    DOT_PRECISION=ctx.dot_prec,
                    BLOCK_D=BLOCK_D,
                    BLOCK_S=BLOCK_S_BWD,
                    num_stages=cfg["num_stages"],
                    num_warps=nw,
                )
                break
            except Exception as e:
                if "OutOfResources" in str(type(e).__name__) or "out of resource" in str(e).lower():
                    nw = nw // 2
                    if nw < 1:
                        raise RuntimeError(
                            "FusedGDN backward: Triton kernel OutOfResources at all warp "
                            f"counts (8, 4, 2, 1). Most recent error: {e}"
                        ) from e
                else:
                    raise

        return dqkv, dbeta, ddecay, None, None, None, None, None, None, None, None, None, None


def fused_gdn_forward_with_grad(
    qkv: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    q_norm_weight: torch.Tensor | None,
    k_norm_weight: torch.Tensor | None,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    F: int,
    S: int,
    k_scale: float = 1.0,
    norm_eps: float = 1e-6,
    eps: float = 1e-6,
    qk_norm: bool = True,
) -> torch.Tensor:
    """Drop-in autograd-enabled replacement for the unidirectional GDN path.

    Unlike ``fused_gdn_func`` (which expects pre-computed ``q_inv_rms``/
    ``k_inv_rms``), this wrapper computes per-head RMSNorm inside the
    Triton kernel (``QK_NORM=1`` / ``USE_PRECOMPUTED_RMS=0``) so the
    backward kernel can reproduce the exact same normed Q/K when
    replaying the recurrence.
    """
    return FusedGDNFunction.apply(
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
        qk_norm,
    )


# =====================================================================
#  Bidirectional BiGDN autograd Function (full-channel RMSNorm in Python)
# =====================================================================


class FusedBiGDNFunction(torch.autograd.Function):
    """Autograd Function for bidirectional fused BiGDN.

    Full-channel RMSNorm is applied in Python (so the norm backward can
    couple all heads correctly), then the kernel runs with ``QK_NORM=0``
    on the pre-normed QKV. Forward and reverse directions are run
    separately with ``SAVE_STATE=1``, and combined as
    ``out = (num_fwd + num_bwd) / (den_fwd + den_bwd + eps)``.

    Backward computes ``dnum`` and ``dden`` from upstream ``dout`` and
    runs the bwd kernel twice (forward + reverse) with ``BIDI_MODE=1``.
    Norm backward is computed in Python (full-channel RMSNorm couples
    all heads).

    Chunk-causal masking (optional):
        Pass ``beta_bwd`` and/or ``decay_bwd`` to override the beta/decay
        tensors used by the **reverse-direction** kernel calls (forward
        save + backward).  The forward direction always uses the
        unmasked ``beta`` / ``decay``.  This mirrors the inference path
        in :func:`fused_bigdn_func` and unlocks chunk-causal autograd
        training: callers typically build ``beta_bwd`` / ``decay_bwd``
        as ``beta.clone()`` / ``decay.clone()`` with interior chunk
        boundaries zeroed, so the anti-causal scan resets state at
        every chunk boundary.

        When ``beta_bwd`` is ``None``, the kernel-emitted reverse-
        direction beta gradient is summed into the forward-direction
        gradient (returned via the ``beta`` slot) and the ``beta_bwd``
        gradient slot returns ``None``.  When ``beta_bwd`` is provided,
        the two gradient streams are kept separate: the forward-
        direction gradient flows through the ``beta`` slot and the
        reverse-direction gradient flows through the ``beta_bwd`` slot
        so autograd can route them through any ``clone() + index = 0``
        masking applied by the caller.  ``decay_bwd`` is handled
        identically.
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
        F: int,
        S: int,
        k_scale: float = 1.0,
        norm_eps: float = 1e-5,
        eps: float = 1e-6,
        beta_bwd: torch.Tensor | None = None,
        decay_bwd: torch.Tensor | None = None,
    ):
        B, N, three, H, D = qkv.shape
        C = H * D
        assert three == 3 and N == F * S
        cfg = _kcfg()

        if q_norm_weight is None:
            q_norm_weight = torch.ones(C, device=qkv.device, dtype=torch.float32)
        if k_norm_weight is None:
            k_norm_weight = torch.ones(C, device=qkv.device, dtype=torch.float32)

        # Full-channel RMSNorm: inv_rms over all H*D dims.
        q_raw = qkv[:, :, 0].float()  # (B, N, H, D)
        k_raw = qkv[:, :, 1].float()
        q_inv_rms = torch.rsqrt((q_raw * q_raw).sum(dim=(-2, -1)) / C + norm_eps)  # (B, N)
        k_inv_rms = torch.rsqrt((k_raw * k_raw).sum(dim=(-2, -1)) / C + norm_eps)

        # Apply norm to Q and K: Q_normed = Q_raw * inv_rms * weight.
        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)
        qkv_normed = qkv.clone()
        qkv_normed[:, :, 0] = (q_raw * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(qkv.dtype)
        qkv_normed[:, :, 1] = (k_raw * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(qkv.dtype)

        # Reverse-direction beta/decay overrides for chunk-causal masking.
        # When the caller supplies ``beta_bwd`` / ``decay_bwd`` (typically
        # ``beta.clone()`` / ``decay.clone()`` with interior chunk-boundary
        # frames zeroed), the reverse-direction kernel reads them instead of
        # the unmasked tensors so the anti-causal scan resets state at chunk
        # boundaries.  The forward (causal) direction always uses the
        # unmasked ``beta`` / ``decay``.
        beta_for_bwd_dir = beta_bwd if beta_bwd is not None else beta
        decay_for_bwd_dir = decay_bwd if decay_bwd is not None else decay

        # Run forward-save with QK_NORM=0 on pre-normed data.
        dummy_nw = torch.ones(D, device=qkv.device, dtype=torch.float32)
        num_fwd, den_fwd, sv_fwd, sz_fwd, svc_fwd, szc_fwd = _run_fwd_save(
            qkv_normed,
            beta,
            decay,
            dummy_nw,
            dummy_nw,
            rope_cos,
            rope_sin,
            F,
            S,
            k_scale,
            norm_eps,
            eps,
            False,
            False,
            cfg,
        )
        num_bwd, den_bwd, sv_bwd, sz_bwd, svc_bwd, szc_bwd = _run_fwd_save(
            qkv_normed,
            beta_for_bwd_dir,
            decay_for_bwd_dir,
            dummy_nw,
            dummy_nw,
            rope_cos,
            rope_sin,
            F,
            S,
            k_scale,
            norm_eps,
            eps,
            False,
            True,
            cfg,
        )

        # Combine: out = (num_fwd + num_bwd) / (den_fwd + den_bwd + eps).
        total_num = num_fwd.float() + num_bwd.float()
        total_den = den_fwd.float() + den_bwd.float()
        total_den_exp = total_den.permute(0, 2, 1).unsqueeze(-1)  # (B, N, H, 1)
        out = (total_num / (total_den_exp + eps)).to(qkv.dtype)

        # Save ``beta_bwd`` / ``decay_bwd`` (possibly ``None``) so the
        # backward pass can (a) replay the reverse-direction kernel against
        # the same masked inputs, and (b) decide whether to keep the
        # reverse-direction beta/decay gradients separate (caller-supplied
        # override) or fold them into the forward-direction gradient
        # (no override, legacy behaviour).
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
            sv_fwd,
            sz_fwd,
            svc_fwd,
            szc_fwd,
            sv_bwd,
            sz_bwd,
            svc_bwd,
            szc_bwd,
            out,
            total_den.to(qkv.dtype),
            beta_bwd,
            decay_bwd,
        )
        _, _dot_prec, _, _ = _resolve_launch_config()
        ctx.dot_prec = _dot_prec
        ctx.F = F
        ctx.S = S
        ctx.k_scale = k_scale
        ctx.norm_eps = norm_eps
        ctx.eps = eps
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
            sv_fwd,
            sz_fwd,
            svc_fwd,
            szc_fwd,
            sv_bwd,
            sz_bwd,
            svc_bwd,
            szc_bwd,
            out,
            total_den_saved,
            beta_bwd_saved,
            decay_bwd_saved,
        ) = ctx.saved_tensors

        # Track whether the caller supplied separate ``beta_bwd`` /
        # ``decay_bwd`` overrides; this controls whether the reverse-
        # direction kernel gradients are summed into the forward-direction
        # slot (legacy behaviour) or routed back through dedicated grad
        # slots so autograd can flow through the caller's masking ops
        # (``clone() + index = 0``).
        has_beta_bwd = beta_bwd_saved is not None
        has_decay_bwd = decay_bwd_saved is not None

        B, N, three, H, D = qkv.shape
        C = H * D

        # Recompute qkv_normed (avoid saving B*N*3*H*D extra tensor).
        q_raw = qkv[:, :, 0].float()
        k_raw = qkv[:, :, 1].float()
        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)
        qkv_normed = qkv.clone()
        qkv_normed[:, :, 0] = (q_raw * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(qkv.dtype)
        qkv_normed[:, :, 1] = (k_raw * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(qkv.dtype)
        F_val = ctx.F
        S = ctx.S
        eps = ctx.eps
        BLOCK_D, _, _, _, _, cfg, beta, decay = _prepare_launch(D, beta, decay)

        # Reverse-direction beta/decay actually fed to the reverse kernel.
        # When the caller supplied an override, we replay against the
        # masked tensor; otherwise we reuse the unmasked beta/decay so the
        # legacy summing path is bit-identical to the pre-extension
        # behaviour.
        if has_beta_bwd:
            beta_for_bwd_dir = beta_bwd_saved.contiguous()
        else:
            beta_for_bwd_dir = beta
        if has_decay_bwd:
            decay_for_bwd_dir = decay_bwd_saved.contiguous()
        else:
            decay_for_bwd_dir = decay

        # ---- Pre-compute dnum and dden ----
        total_den_exp = total_den_saved.float().permute(0, 2, 1).unsqueeze(-1)
        inv_total_den = 1.0 / (total_den_exp + eps)
        dnum = (dout.float() * inv_total_den).to(qkv.dtype).contiguous()
        dden = (
            (-(dout.float() * out.float()).sum(dim=-1) * inv_total_den.squeeze(-1))
            .permute(0, 2, 1)
            .to(qkv.dtype)
            .contiguous()
        )
        del out, total_den_saved, total_den_exp, inv_total_den

        dummy_nw = torch.ones(D, device=qkv.device, dtype=torch.float32)

        # ---- Backward for forward direction (QK_NORM=0, operates on normed QKV) ----
        dqkv_fwd = torch.zeros_like(qkv)
        dbeta_fwd = torch.zeros_like(beta)
        ddecay_fwd = torch.zeros_like(decay)

        def _run_triton_bwd(sv, sz, svc, szc, dqkv_out, dbeta_out, ddecay_out, reverse_bwd, beta_kernel, decay_kernel):
            """Try backward kernel with progressively fewer warps on tmem overflow.

            ``beta_kernel`` / ``decay_kernel`` are passed as explicit
            arguments (instead of closing over the outer-scope ``beta`` /
            ``decay``) so the reverse-direction call can replay against the
            chunk-causal-masked tensors (``beta_bwd`` / ``decay_bwd``) when
            present, while the forward-direction call always uses the
            unmasked ``beta`` / ``decay``.
            """
            nw = cfg["num_warps"]
            if ctx.dot_prec >= 1:
                nw = min(nw, 4)
            while nw >= 1:
                try:
                    _fused_gdn_bwd_kernel[(B * H,)](
                        qkv_normed,
                        qkv_normed.stride(0),
                        qkv_normed.stride(1),
                        qkv_normed.stride(2),
                        qkv_normed.stride(3),
                        qkv_normed.stride(4),
                        beta_kernel,
                        decay_kernel,
                        dummy_nw,
                        dummy_nw,
                        rope_cos,
                        rope_sin,
                        sv,
                        sz,
                        svc,
                        szc,
                        dnum,
                        dden,
                        dqkv_out,
                        dbeta_out,
                        ddecay_out,
                        H=H,
                        F=F_val,
                        S=S,
                        D=D,
                        K_SCALE=ctx.k_scale,
                        NORM_EPS=ctx.norm_eps,
                        EPS=eps,
                        QK_NORM=0,
                        STATE_FP32=1 if cfg["STATE_FP32"] else 0,
                        REVERSE_BWD=reverse_bwd,
                        BIDI_MODE=1,
                        DOT_PRECISION=ctx.dot_prec,
                        BLOCK_D=BLOCK_D,
                        BLOCK_S=cfg["BLOCK_S"],
                        num_stages=cfg["num_stages"],
                        num_warps=nw,
                    )
                    return  # success
                except Exception as e:
                    if "OutOfResources" in str(type(e).__name__) or "out of resource" in str(e).lower():
                        nw = nw // 2
                        if nw >= 1:
                            continue
                        raise RuntimeError(
                            "FusedBiGDN backward: Triton kernel OutOfResources at all warp counts "
                            f"(8, 4, 2, 1). Most recent error: {e}"
                        ) from e
                    else:
                        raise

        _run_triton_bwd(sv_fwd, sz_fwd, svc_fwd, szc_fwd, dqkv_fwd, dbeta_fwd, ddecay_fwd, 0, beta, decay)
        del sv_fwd, sz_fwd, svc_fwd, szc_fwd

        # ---- Backward for reversed direction (replays against masked beta/decay if any) ----
        # Allocate kernel-output gradients with the exact shape the kernel
        # writes — these always match the input ``beta_for_bwd_dir`` /
        # ``decay_for_bwd_dir`` shapes (override or fall-back).
        dqkv_bwd = torch.zeros_like(qkv)
        dbeta_bwd_kernel = torch.zeros_like(beta_for_bwd_dir)
        ddecay_bwd_kernel = torch.zeros_like(decay_for_bwd_dir)

        _run_triton_bwd(
            sv_bwd,
            sz_bwd,
            svc_bwd,
            szc_bwd,
            dqkv_bwd,
            dbeta_bwd_kernel,
            ddecay_bwd_kernel,
            1,
            beta_for_bwd_dir,
            decay_for_bwd_dir,
        )
        del sv_bwd, sz_bwd, svc_bwd, szc_bwd
        del qkv_normed, dnum, dden

        # Q/K/V gradient is always summed: qkv is shared by both directions.
        dqkv_fwd += dqkv_bwd
        del dqkv_bwd

        # Beta gradient: route depends on whether the caller supplied an
        # override. With override -> keep separate (so autograd routes the
        # reverse-direction grad through the caller's clone+mask op).
        # Without override -> sum into the forward-direction grad
        # (legacy behaviour, bit-identical to pre-extension code).
        if has_beta_bwd:
            dbeta = dbeta_fwd
            dbeta_bwd_out: torch.Tensor | None = dbeta_bwd_kernel
        else:
            dbeta_fwd += dbeta_bwd_kernel
            dbeta = dbeta_fwd
            dbeta_bwd_out = None
        del dbeta_bwd_kernel

        # Decay gradient: same routing logic, independent of beta override.
        if has_decay_bwd:
            ddecay = ddecay_fwd
            ddecay_bwd_out: torch.Tensor | None = ddecay_bwd_kernel
        else:
            ddecay_fwd += ddecay_bwd_kernel
            ddecay = ddecay_fwd
            ddecay_bwd_out = None
        del ddecay_bwd_kernel

        dqkv_normed = dqkv_fwd

        # ---- Full-channel RMSNorm backward for Q and K ----
        # y = x * inv_rms * w  ->  dL/dx = inv_rms*w*dL/dy - inv_rms^3/C * x * sum(w*dL/dy*x)
        # Process Q and K sequentially to reduce peak fp32 memory.

        # Q norm backward.
        q_irms = q_inv_rms[:, :, None, None]
        dq_normed = dqkv_normed[:, :, 0].float()
        gw_q = dq_normed * q_nw_hd[None, None]
        dq_nw = (dq_normed * q_raw * q_irms).sum(dim=(0, 1)).reshape(-1)
        corr_q = (gw_q * q_raw).sum(dim=(-2, -1), keepdim=True)
        dqkv_normed[:, :, 0] = (q_irms * gw_q - (q_irms**3) / C * q_raw * corr_q).to(qkv.dtype)
        del dq_normed, gw_q, corr_q, q_raw, q_irms

        # K norm backward.
        k_irms = k_inv_rms[:, :, None, None]
        dk_normed = dqkv_normed[:, :, 1].float()
        gw_k = dk_normed * k_nw_hd[None, None]
        dk_nw = (dk_normed * k_raw * k_irms).sum(dim=(0, 1)).reshape(-1)
        corr_k = (gw_k * k_raw).sum(dim=(-2, -1), keepdim=True)
        dqkv_normed[:, :, 1] = (k_irms * gw_k - (k_irms**3) / C * k_raw * corr_k).to(qkv.dtype)
        del dk_normed, gw_k, corr_k, k_raw, k_irms

        return (
            dqkv_normed,
            dbeta,
            ddecay,
            dq_nw.to(q_norm_weight.dtype),
            dk_nw.to(k_norm_weight.dtype),
            None,  # rope_cos
            None,  # rope_sin
            None,  # F
            None,  # S
            None,  # k_scale
            None,  # norm_eps
            None,  # eps
            dbeta_bwd_out,  # beta_bwd
            ddecay_bwd_out,  # decay_bwd
        )


def fused_bigdn_forward_with_grad(
    qkv: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    q_norm_weight: torch.Tensor | None,
    k_norm_weight: torch.Tensor | None,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    F: int,
    S: int,
    k_scale: float = 1.0,
    norm_eps: float = 1e-5,
    eps: float = 1e-6,
    beta_bwd: torch.Tensor | None = None,
    decay_bwd: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bidirectional fused BiGDN with autograd support (full-channel RMSNorm).

    Unlike ``fused_bigdn_func`` (which expects pre-computed ``q_inv_rms`` /
    ``k_inv_rms``), this wrapper computes the full-channel inv-RMS in Python
    so the norm backward can flow through the autograd graph naturally.

    Chunk-causal masking (optional):
        Pass ``beta_bwd`` and/or ``decay_bwd`` to override the beta/decay
        tensors used by the **reverse-direction** kernel only.  These are
        typically built by the caller as ``beta.clone()`` / ``decay.clone()``
        with interior chunk-boundary frames zeroed, so the anti-causal scan
        resets state at chunk boundaries while the causal scan keeps full
        context.  The reverse-direction beta/decay gradients are routed
        back through the ``beta_bwd`` / ``decay_bwd`` slots (instead of
        being summed into the forward-direction grad), which lets autograd
        flow the reverse-direction gradient through the caller's
        ``clone() + index = 0`` masking op.

        When ``beta_bwd`` / ``decay_bwd`` is ``None`` (default), behaviour
        is bit-identical to the pre-extension full-sequence-bidirectional
        path: the reverse-direction kernel uses the unmasked ``beta`` /
        ``decay`` and its kernel-emitted gradient is summed into the
        forward-direction gradient before being returned.
    """
    return FusedBiGDNFunction.apply(
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
        beta_bwd,
        decay_bwd,
    )
