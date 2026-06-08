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

"""Triton-kernel dispatch wrappers for streaming chunk-causal inference.

These are pure helpers that take an attention-block ``layer`` instance and the
current chunk's tensors and run the fused kernels in
:mod:`diffusion.model.ops.fused_gdn`,
:mod:`diffusion.model.ops.fused_gdn_chunkwise`, and
:mod:`diffusion.model.ops.fused_cam_gdn`.  The ``nn.Module`` wrappers live in
:mod:`diffusion.model.nets.sana_gdn_blocks` (``CachedChunkCausalGDN`` /
``CachedChunkCausalSoftmaxAttn``) and
:mod:`diffusion.model.nets.sana_gdn_camctrl_blocks`
(``CachedChunkCausalGDNUCPESinglePathLiteLA`` /
``CachedSoftmaxUCPESinglePathLiteLA``) and call into here.

Cache slot layout (10 slots per attention block, shared with the
scheduler).  Slot 6 distinguishes GDN (state-based) from softmax
(concat-based) blocks.

.. list-table::
   :header-rows: 1

   * - Slot
     - GDN blocks
     - Softmax blocks
   * - 0
     - S_kv state (B, H, D, D)
     - k post-RoPE (B, H, N, D)
   * - 1
     - S_z state (B, H, D, 1)
     - v (B, H, N, D)
   * - 2
     - cam_S_kv state (B, H_c, D_c, D_c)
     - cam_k post-UCPE (B, H_c, N, D_c)
   * - 3
     - None
     - cam_v post-UCPE (B, H_c, N, D_c)
   * - 4
     - ShortConv K state (B*S, K-1, C)
     - None (no conv for softmax)
   * - 5
     - tconv state (handled by CachedGLUMBConvTemp)
     - tconv state
   * - 6
     - type flag: 1.0
     - type flag: 0.0
   * - 7-9
     - reserved
     - reserved
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import ShortConvolution

# ---------------------------------------------------------------------------
# Cache slot indices (must match scheduler constants)
# ---------------------------------------------------------------------------

_SLOT_FWD_KV = 0
_SLOT_FWD_Z = 1
_SLOT_CAM = 2
_SLOT_CAM_AUX = 3
_SLOT_SHORTCONV = 4
_SLOT_TCONV = 5  # NOTE: CachedGLUMBConvTemp actually writes to kv_cache[-1] (slot 9), not slot 5!
_SLOT_TYPE_FLAG = 6

_TYPE_STATE = 1.0  # GDN: state-based cache
_TYPE_CONCAT = 0.0  # Softmax: concat-based cache


def _slice_rope_to_current_chunk(rotary_emb: torch.Tensor, current_n: int) -> torch.Tensor:
    """Slice rotary embedding freqs to the trailing ``current_n`` token positions.

    When ``sink_token=true``, upstream rope is built for sink + current chunk
    positions (covers ``frame_index.numel()`` frames). But q/k inside the
    cached chunk-causal attention only cover the current chunk — sink K is
    either pre-rotated in S_kv (linear attn) or pre-rotated in kv_cache K
    (softmax attn). Slicing the trailing portion of ``rotary_emb`` aligns it
    with current-chunk q/k. If sizes already match (e.g. rolling_rope path
    that generates rope only for the current chunk's frame range), this is a
    no-op.
    """
    rope_n = rotary_emb.shape[-2]
    if rope_n == current_n:
        return rotary_emb
    if rope_n < current_n:
        raise RuntimeError(
            f"rotary_emb has {rope_n} positions, smaller than current chunk's " f"{current_n}; cannot slice."
        )
    return rotary_emb[..., -current_n:, :]


# ---------------------------------------------------------------------------
# Cached temporal short convolution
# ---------------------------------------------------------------------------


def _cached_temporal_short_conv(
    x: torch.Tensor,
    conv: ShortConvolution,
    HW: tuple[int, int, int],
    conv_cache: torch.Tensor | None,
    save_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Short conv with cached left context: forward-cached + backward-isolated.

    Mirrors ``ChunkCausalGDN._apply_temporal_short_conv`` but replaces the
    global forward causal conv with a cache-aware version.

    Uses the same ``ShortConvolution.forward()`` (Triton/CUDA backend) as the
    training path for bit-exact numerical parity.  The only difference is that
    the forward causal pass prepends cached left context instead of starting
    from zeros.

    Args:
        x: ``(B, N, C)`` where ``N = T * S``.
        conv: FLA ``ShortConvolution`` (depthwise causal Conv1d).
        HW: ``(T, H, W)``.
        conv_cache: ``(B*S, K-1, C)`` from previous chunk, or None.
        save_cache: Whether to return a new cache for the next chunk.

    Returns:
        (output, new_cache): output ``(B, N, C)``, new_cache ``(B*S, K-1, C)`` or None.
    """
    T, H, W = HW
    S = H * W
    B_orig, N, C = x.shape
    dtype_in = x.dtype
    K = conv.weight.shape[-1]

    # Reshape to temporal: (B*S, T, C).
    x_t = x.reshape(B_orig, T, S, C).permute(0, 2, 1, 3).contiguous().reshape(B_orig * S, T, C)

    # --- Forward causal conv with cache ---
    # Use ShortConvolution.forward() (Triton/CUDA kernel) for exact numerical
    # parity with ChunkCausalGDN._apply_temporal_short_conv.
    if conv_cache is not None:
        # Prepend cached left context and run full causal conv, then slice.
        x_fwd_in = torch.cat([conv_cache.to(x_t.dtype), x_t], dim=1)
        y_fwd_full, _ = conv(x_fwd_in)
        y_fwd = y_fwd_full[:, K - 1 :, :]  # drop positions from cached prefix
    else:
        y_fwd, _ = conv(x_t)

    # --- Backward conv (isolated within current chunk) ---
    # Same as ChunkCausalGDN._backward_causal_conv_per_chunk for a single chunk:
    # flip → causal conv → flip back.
    y_bwd_flipped, _ = conv(x_t.flip(1))
    y_bwd = y_bwd_flipped.flip(1)  # (B*S, T, C)

    # --- Center tap ---
    w_center = conv.weight[:, 0, -1]  # (C,)
    center_term = x_t * w_center.unsqueeze(0).unsqueeze(0)

    y = y_fwd + y_bwd - center_term

    # Save cache: last K-1 timesteps of the conv INPUT (for next chunk's left context).
    new_cache: torch.Tensor | None = None
    if save_cache and K > 1:
        new_cache = x_t[:, -(K - 1) :, :].detach().clone()

    # Reshape back to (B, N, C).
    y = y.reshape(B_orig, S, T, C).permute(0, 2, 1, 3).reshape(B_orig, N, C)
    if y.dtype != dtype_in:
        y = y.to(dtype_in)
    return y, new_cache


# ---------------------------------------------------------------------------
# Fused Triton scan (chunk-causal main GDN branch)
# ---------------------------------------------------------------------------


def _gdn_main_triton(
    layer,
    qkv: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    rotary_emb: torch.Tensor | None,
    HW: tuple[int, int, int],
    S_kv_prev: torch.Tensor | None,
    S_z_prev: torch.Tensor | None,
    save_kv_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run the chunk-causal main GDN scan through the fused Triton chunkwise
    kernels.

    Uses two fused Triton calls (forward-with-state + per-chunk reverse) on
    a shared ``qkv`` prep.

    The chunk-causal layout (forward seeded from cached state, backward
    isolated per chunk) is NOT what the bidir convenience wrapper does
    (``fused_bigdn_bidi_chunkwise`` sums both directions in-kernel and
    assumes neither has state).  We call ``phase_a`` once, ``phase_b_triton``
    twice (direction=1 stateful, direction=2 stateless), accumulate in
    ``phase_c`` via ``accumulate=True``, then divide.

    Args:
        layer: A :class:`CachedChunkCausalGDN` instance — for q_norm /
            k_norm weights, eps, kernel_func, etc.
        qkv: ``(B, N, 3, H, D)`` raw QKV (post short-conv K).  Triton applies
            RMSNorm + ReLU + K-scale + RoPE inside the kernel.
        beta: ``(B, H, F)`` or ``(B, H, F, S)`` per-frame gates (input dtype).
        decay: ``(B, H, F)`` per-frame gates.
        rotary_emb: complex rotary frequencies for the current chunk.
        HW: ``(T=F, H, W)``; with ``S = H * W``.
        S_kv_prev: cached forward-scan ``(B, H, D, D)`` state from the prior
            chunk, or ``None`` for the first chunk.
        S_z_prev: cached forward-scan ``(B, H, D, 1)`` state, or ``None``.
        save_kv_cache: when ``True``, return ``(out, S_kv_new, S_z_new)``;
            otherwise return ``(out, None, None)``.

    Returns:
        ``(out, S_kv_new, S_z_new)`` where ``out`` is ``(B, N, H, D)``
        post-divide in the kernel's dot_precision dtype (fp32 or bf16).
    """
    # Imports are local so the module loads on CPU-only environments.
    from diffusion.model.ops.fused_gdn import fused_qk_inv_rms, prepare_rope_tables
    from diffusion.model.ops.fused_gdn_chunkwise import _default_dot_prec, phase_a, phase_b_triton, phase_c

    B, N, three, H, D = qkv.shape
    assert three == 3, f"qkv last-3 dim must be 3 (q,k,v); got shape {qkv.shape}"
    T, H_sp, W_sp = HW
    S = H_sp * W_sp
    assert N == T * S, f"N={N} != T*S={T * S} for HW={HW}"
    C = H * D

    # ---- RMS norm parameters ----
    if isinstance(layer.q_norm, nn.Identity):
        q_nw = torch.ones(C, device=qkv.device, dtype=torch.float32)
        k_nw = torch.ones(C, device=qkv.device, dtype=torch.float32)
        norm_eps = 1e-5
    else:
        q_nw = layer.q_norm.weight.float().contiguous()
        k_nw = layer.k_norm.weight.float().contiguous()
        norm_eps = float(getattr(layer.q_norm, "eps", 1e-5))

    # ---- inv RMS (single fused launch over Q and K halves of qkv) ----
    q_inv_rms, k_inv_rms = fused_qk_inv_rms(qkv, eps=norm_eps)

    # ---- RoPE tables for the current chunk ----
    if rotary_emb is None:
        rope_cos = torch.ones(N, D, device=qkv.device, dtype=torch.float32)
        rope_sin = torch.zeros(N, D, device=qkv.device, dtype=torch.float32)
    else:
        rope_cur = _slice_rope_to_current_chunk(rotary_emb, N)
        rope_cos, rope_sin = prepare_rope_tables(rope_cur, N, D, qkv.device)

    # ---- K scale (same convention as torch path: D^-1/2 * S^-1/2) ----
    k_scale = (D**-0.5) * (S**-0.5)

    # ---- Beta broadcast convention: kernels accept (B,H,F) or (B,H,F,S) ----
    beta_c = beta.contiguous()
    decay_c = decay.contiguous()

    dot_prec = _default_dot_prec()

    # ---- Phase A: shared prep for both directions ----
    I_P_kv, A_buf, I_P_z, B_z = phase_a(
        qkv,
        beta_c,
        q_inv_rms,
        k_inv_rms,
        q_nw,
        k_nw,
        rope_cos,
        rope_sin,
        F=T,
        S=S,
        k_scale=k_scale,
        norm_eps=norm_eps,
        dot_precision=dot_prec,
    )

    # ---- Pad caller-supplied (B,H,D,D)/(B,H,D,1) state to padded layout ----
    BLOCK_D = I_P_kv.shape[-1]
    init_kv_padded = None
    init_z_padded = None
    if S_kv_prev is not None:
        sk = S_kv_prev
        sk = sk.to(torch.float32) if sk.dtype != torch.float32 else sk
        B_s, H_s, D_in, D_out = sk.shape
        if D_in != BLOCK_D or D_out != BLOCK_D:
            init_kv_padded = F.pad(
                sk.transpose(-1, -2).reshape(B_s * H_s, D_out, D_in),
                (0, BLOCK_D - D_in, 0, BLOCK_D - D_out),
            ).contiguous()
        else:
            init_kv_padded = sk.transpose(-1, -2).reshape(B_s * H_s, BLOCK_D, BLOCK_D).contiguous()
        sz = S_z_prev.squeeze(-1) if S_z_prev.dim() == 4 else S_z_prev
        sz = sz.to(torch.float32) if sz.dtype != torch.float32 else sz
        Bz, Hz, Dz = sz.shape
        if Dz != BLOCK_D:
            init_z_padded = F.pad(sz.reshape(Bz * Hz, Dz), (0, BLOCK_D - Dz)).contiguous()
        else:
            init_z_padded = sz.reshape(Bz * Hz, BLOCK_D).contiguous()

    # ---- Phase B forward (direction=1) with state ----
    if save_kv_cache:
        M_fwd, z_fwd, _, _, final_kv, final_z = phase_b_triton(
            I_P_kv,
            A_buf,
            I_P_z,
            B_z,
            decay_c,
            F=T,
            dot_precision=dot_prec,
            direction=1,
            init_state_kv=init_kv_padded,
            init_state_z=init_z_padded,
            return_final_state=True,
        )
    else:
        M_fwd, z_fwd, _, _ = phase_b_triton(
            I_P_kv,
            A_buf,
            I_P_z,
            B_z,
            decay_c,
            F=T,
            dot_precision=dot_prec,
            direction=1,
            init_state_kv=init_kv_padded,
            init_state_z=init_z_padded,
        )

    # ---- Phase B reverse (direction=2) — per-chunk, no state ----
    _, _, M_rev, z_rev = phase_b_triton(
        I_P_kv,
        A_buf,
        I_P_z,
        B_z,
        decay_c,
        F=T,
        dot_precision=dot_prec,
        direction=2,
    )
    del I_P_kv, A_buf, I_P_z, B_z

    # ---- Phase C: fwd output, then accumulate rev output into same buffers ----
    num_out, den_out = phase_c(
        qkv,
        q_inv_rms,
        q_nw,
        rope_cos,
        rope_sin,
        M_fwd,
        z_fwd,
        F=T,
        S=S,
        dot_precision=dot_prec,
    )
    phase_c(
        qkv,
        q_inv_rms,
        q_nw,
        rope_cos,
        rope_sin,
        M_rev,
        z_rev,
        F=T,
        S=S,
        dot_precision=dot_prec,
        num_out=num_out,
        den_out=den_out,
        accumulate=True,
    )
    del M_fwd, z_fwd, M_rev, z_rev

    # ---- Divide: (B, H, N) -> (B, N, H, 1) for broadcast over D ----
    eps = float(layer.eps)
    total_den = den_out.float().permute(0, 2, 1).unsqueeze(-1)  # (B, N, H, 1)
    out = (num_out.float() / (total_den + eps)).to(qkv.dtype)  # (B, N, H, D)
    del num_out, den_out, total_den

    # ---- Unpad final state back to (B, H, D, D) / (B, H, D, 1) ----
    if save_kv_cache:
        S_kv_new = final_kv.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].transpose(-1, -2).contiguous()
        S_z_new = final_z.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()
        return out, S_kv_new, S_z_new
    return out, None, None


def _cam_main_triton(
    q_cam_trans: torch.Tensor,
    k_cam_trans: torch.Tensor,
    v_cam_trans: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    cam_S_kv_prev: torch.Tensor | None,
    save_kv_cache: bool,
    T: int,
    S: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the camera-branch single-path delta-rule chunk-causal scan with
    two ``cam_scan_chunkwise`` calls (forward seeded from cached state +
    per-chunk reverse).

    The kernel expects fp32 q/k/v in ``(B, H, D, N)`` layout (already
    cam-prep'd: RMSNorm + ReLU + UCPE + RoPE).  State is stored as
    ``(B*H, BLOCK_D, BLOCK_D)`` fp32 internally; we accept/return the
    ``(B, H, D, D)`` torch-format used by the kv_cache slot.

    Args:
        q_cam_trans, k_cam_trans, v_cam_trans: ``(B, H, D, N)`` —
            cam-prep'd inputs.  Cast to fp32 if not already.
        beta: ``(B, H, F)`` or ``(B, H, F, S)``.
        decay: ``(B, H, F)``.
        cam_S_kv_prev: ``(B, H, D, D)`` fp32 cached state, or ``None``.
        save_kv_cache: when ``True``, return ``(out, cam_S_kv_new)``.
        T, S: frames and spatial-tokens-per-frame (``N = T * S``).

    Returns:
        ``(out, cam_S_kv_new)`` with ``out`` shaped ``(B, H, D, N)`` fp32
        (fwd + per-chunk bwd combined) and ``cam_S_kv_new`` shaped
        ``(B, H, D, D)`` fp32 (or ``None`` when not saving).
    """
    import os as _os

    import triton

    # CUDA cam scan on by default; set SANA_GDN_CUDA=0 to force Triton.
    if _os.environ.get("SANA_GDN_CUDA", "1") != "0":
        try:
            from diffusion.model.ops.fused_gdn_chunkwise_cuda import cam_scan_chunkwise_cuda as cam_scan_chunkwise
        except Exception:
            from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_chunkwise
    else:
        from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_chunkwise

    B, H, D, N = q_cam_trans.shape
    assert N == T * S, f"N={N} != T*S={T * S}"
    BLOCK_D = triton.next_power_of_2(D)

    # ---- Inputs: fp32 contiguous (kernel hard-requires this). ----
    q32 = q_cam_trans.float().contiguous() if q_cam_trans.dtype != torch.float32 else q_cam_trans.contiguous()
    k32 = k_cam_trans.float().contiguous() if k_cam_trans.dtype != torch.float32 else k_cam_trans.contiguous()
    v32 = v_cam_trans.float().contiguous() if v_cam_trans.dtype != torch.float32 else v_cam_trans.contiguous()

    # ---- Beta: kernel expects (B, H, F, S) per docstring. ----
    if beta.ndim == 3:
        beta_c = beta.unsqueeze(-1).expand(B, H, T, S).contiguous().float()
    else:
        beta_c = beta.contiguous().float()
    decay_c = decay.contiguous().float()

    # ---- Pad caller-supplied (B, H, D, D) to (B*H, BLOCK_D, BLOCK_D) fp32. ----
    init_state = None
    if cam_S_kv_prev is not None:
        sk = cam_S_kv_prev.to(torch.float32) if cam_S_kv_prev.dtype != torch.float32 else cam_S_kv_prev
        if D != BLOCK_D:
            init_state = F.pad(
                sk.reshape(B * H, D, D),
                (0, BLOCK_D - D, 0, BLOCK_D - D),
            ).contiguous()
        else:
            init_state = sk.reshape(B * H, BLOCK_D, BLOCK_D).contiguous()

    # ---- Forward scan with state ----
    if save_kv_cache:
        out_fwd, final_state = cam_scan_chunkwise(
            q32,
            k32,
            v32,
            beta_c,
            decay_c,
            reverse=False,
            init_state=init_state,
            save_final_state=True,
        )
    else:
        out_fwd = cam_scan_chunkwise(
            q32,
            k32,
            v32,
            beta_c,
            decay_c,
            reverse=False,
            init_state=init_state,
            save_final_state=False,
        )
        final_state = None

    # ---- Backward scan (per-chunk isolated; no state) ----
    out_bwd = cam_scan_chunkwise(
        q32,
        k32,
        v32,
        beta_c,
        decay_c,
        reverse=True,
        init_state=None,
        save_final_state=False,
    )

    out = out_fwd + out_bwd  # (B, H, D, N) fp32

    if final_state is None:
        return out, None
    # Cam state is stored as M[K_feat, V_feat] (row-major D_K, D_V) — NO
    # transpose unlike main GDN (which transposes on save/load).  Unpad to
    # the (B, H, D, D) shape callers expect for the kv_cache slot.
    cam_S_kv_new = final_state.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].contiguous()
    return out, cam_S_kv_new


# ---------------------------------------------------------------------------
# Fused Triton cam-prep (RMSNorm + ReLU + K-scale + UCPE 4x4 + RoPE)
# ---------------------------------------------------------------------------


def _cam_prep_triton(
    layer,
    x: torch.Tensor,
    HW: tuple[int, int, int],
    camera_conditions: torch.Tensor,
    rotary_emb: torch.Tensor | None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, callable]:
    """Streaming cam-branch QKV prep through the bidir's fused Triton kernel.

    Mirrors the prep section of
    :meth:`BidirectionalGDNUCPESinglePathLiteLABothTriton._forward_cam_branch`
    (QKV linear + cam-K short conv + ``cam_prep_func`` Triton kernel +
    ``inflation_sq`` reshape) but applies the K conv with a per-chunk
    isolated bidirectional pattern matching the torch streaming path
    (``BidirectionalGDN._apply_temporal_short_conv``).

    Args:
        layer: A :class:`CachedChunkCausalGDNUCPESinglePathLiteLA` instance.
        x: ``(B, N, C)`` input activations for the current chunk.
        HW: ``(T, H, W)`` token layout.
        camera_conditions: ``(B, T, ...)`` camera-pose tensor.
        rotary_emb: complex RoPE frequencies for the current chunk.

    Returns:
        ``(q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq, apply_fn_o)``
        with ``q/k/v_cam_trans`` shaped ``(B, H_cam, D_cam, N)`` in the input
        dtype, ``inflation_sq`` shaped ``(B, H_cam, 1, N)`` fp32, and
        ``apply_fn_o`` a torch closure that applies the inverse UCPE+RoPE to
        the scan output.
    """
    from diffusion.model.nets.sana_camctrl_blocks import _prepare_ray_apply_fns
    from diffusion.model.ops.fused_cam_gdn import (
        _invert_SE3,
        _prepare_ucpe_rope_tables,
        _process_camera_conditions_raymats_only,
        cam_prep_func,
    )

    if layer.conv_q_cam is not None or layer.conv_v_cam is not None:
        raise NotImplementedError(
            "Triton cam-prep requires k_conv_only=True on the camera branch " "(conv_q_cam / conv_v_cam must be None)."
        )

    B, N, _ = x.shape
    T, H_sp, W_sp = HW
    S = H_sp * W_sp
    H_heads = layer.cam_heads
    D_head = layer.cam_head_dim

    # ---- 1. QKV linear (fused via cat) + cam-K short conv ----
    qkv_w = torch.cat([layer.q_proj_cam.weight, layer.k_proj_cam.weight, layer.v_proj_cam.weight])
    qkv_b = torch.cat([layer.q_proj_cam.bias, layer.k_proj_cam.bias, layer.v_proj_cam.bias])
    qkv_cam = F.linear(x, qkv_w, qkv_b)
    q_raw, k_raw, v_raw = qkv_cam.chunk(3, dim=-1)

    if layer.conv_k_cam is not None:
        # Match the streaming torch path's K-conv routing:
        # ``_GDNUCPEBase._apply_temporal_short_conv`` dispatches to the
        # bidirectional in-chunk conv when ``chunk_size >= T`` (the streaming
        # cam K conv has no cross-chunk state).
        k_raw = layer._apply_temporal_short_conv(k_raw, layer.conv_k_cam, HW, **kwargs)

    q_raw = q_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
    k_raw = k_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
    v_raw = v_raw.contiguous().view(B, N, H_heads, D_head).contiguous()

    # ---- 2. UCPE projection matrices (P / P_T / P_inv) ----
    raymats = _process_camera_conditions_raymats_only(camera_conditions, B, HW, layer.patch_size)
    raymats = raymats.reshape(B, -1, 4, 4)
    P = raymats
    P_T = P.transpose(-1, -2).contiguous()
    P_inv = _invert_SE3(P).contiguous()

    # ---- 3. Sliced cam-branch RoPE (D/2 dims; T/H/W split halved) ----
    if rotary_emb is not None:
        # Mirror the WAN-RoPE slicing used by the bidir kernel call site.
        head_dim = D_head
        orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
        orig_h_size = head_dim // 6
        new_head_dim = head_dim // 2
        new_t_size = new_head_dim // 2 - 2 * (new_head_dim // 6)
        new_h_size = new_head_dim // 6
        new_w_size = new_head_dim // 6
        t_part = rotary_emb[..., :new_t_size]
        h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
        w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
        rotary_emb_cam = torch.cat([t_part, h_part, w_part], dim=-1)
        # Slice trailing N positions when upstream RoPE covers sink+current.
        rotary_emb_cam = _slice_rope_to_current_chunk(rotary_emb_cam, N)
        rope_cos, rope_sin = _prepare_ucpe_rope_tables(rotary_emb_cam, N, D_head // 2, x.device)
    else:
        rotary_emb_cam = None
        rope_cos = torch.ones(N, D_head // 2, device=x.device, dtype=torch.float32)
        rope_sin = torch.zeros(N, D_head // 2, device=x.device, dtype=torch.float32)

    # ---- 4. Fused Triton prep kernel ----
    q_norm_w = layer.q_norm_cam.weight.float().contiguous()
    k_norm_w = layer.k_norm_cam.weight.float().contiguous()
    k_scale = (D_head**-0.5) * (S**-0.5)
    norm_eps_val = float(getattr(layer.q_norm_cam, "eps", getattr(layer.q_norm_cam, "variance_epsilon", 1e-6)))
    q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq = cam_prep_func(
        q_raw,
        k_raw,
        v_raw,
        q_norm_weight=q_norm_w,
        k_norm_weight=k_norm_w,
        proj_q=P_T,
        proj_kv=P_inv,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_scale=k_scale,
        norm_eps=norm_eps_val,
    )
    inflation_sq = inflation_sq.view(B, H_heads, 1, N)

    # ---- 5. Inverse-UCPE closure for the scan output ----
    _, _, apply_fn_o = _prepare_ray_apply_fns(head_dim=D_head, P=P, P_T=P_T, P_inv=P_inv, rotary_emb=rotary_emb_cam)

    return q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq, apply_fn_o


def _cached_gdn_forward_triton(
    layer,
    x: torch.Tensor,
    HW: tuple[int, int, int] | None,
    rotary_emb: torch.Tensor | None,
    apply_output_gate: bool,
    **kwargs: object,
) -> tuple[torch.Tensor, list]:
    """Cached chunk-causal forward through the fused Triton scan.

    Same return contract as the torch ``CachedChunkCausalGDN.forward``
    cached path (``(out, kv_cache)``).  Recurrent state on slots
    ``[_SLOT_FWD_KV, _SLOT_FWD_Z]`` and the shortconv slot
    ``[_SLOT_SHORTCONV]`` are updated in place exactly like the torch
    path so the scheduler can swap between implementations chunk-by-chunk
    without seeing a difference.

    Takes ``layer`` as an explicit argument (not ``self``) so it works
    whether the dispatch comes from ``CachedChunkCausalGDN.forward`` called
    directly or from the camctrl wrapper's
    ``CachedChunkCausalGDNUCPESinglePathLiteLA.forward`` which invokes
    ``CachedChunkCausalGDN.forward(self, ...)`` against the wrapper
    instance (wrapper instances don't have this helper on themselves).

    Guards: ``conv_q``/``conv_v`` are unsupported by the fused kernel —
    the streaming production checkpoint uses ``k_conv_only=True`` so this
    is fine in practice, but raise here if anyone tries to load a
    non-k_conv_only configuration through this path.
    """
    if HW is None:
        raise ValueError("HW (T, H, W) must be provided.")
    if layer.conv_q is not None or layer.conv_v is not None:
        raise NotImplementedError(
            "Triton chunk-causal scan requires k_conv_only=True; " "got conv_q / conv_v not None."
        )

    kv_cache = kwargs["kv_cache"]
    save_kv_cache = kwargs.get("save_kv_cache", False)
    B, N, C = x.shape
    T, H_sp, W_sp = HW
    S = H_sp * W_sp
    if N != T * S:
        raise ValueError(f"N={N} != T*S={T * S} for HW={HW}")
    H, D = layer.heads, layer.dim

    # 1. QKV projection -> (B, N, 3, H, D), made contiguous so the fused
    #    kernel can stride-iterate over it.
    qkv = layer.qkv(x).reshape(B, N, 3, H, D)

    # 2. Short conv on K (with cache).  Write the post-conv K back into
    #    qkv[:, :, 1] so the kernel sees it as the K stream.
    if layer.conv_k is not None:
        k_flat = qkv[:, :, 1].reshape(B, N, C)
        k_flat, new_conv_cache = _cached_temporal_short_conv(
            k_flat, layer.conv_k, HW, kv_cache[_SLOT_SHORTCONV], save_kv_cache
        )
        qkv = qkv.contiguous()
        qkv[:, :, 1].copy_(k_flat.reshape(B, N, H, D))
        if save_kv_cache:
            kv_cache[_SLOT_SHORTCONV] = new_conv_cache
    else:
        qkv = qkv.contiguous()

    # 3. Frame gates — Triton accepts (B,H,F) or (B,H,F,S); same as torch.
    precomputed_gates = kwargs.get("precomputed_gates", None)
    if precomputed_gates is not None:
        beta, decay = precomputed_gates
    else:
        beta, decay = layer._compute_frame_gates(x, HW)

    # 4. Fused Triton fwd-with-state + per-chunk rev scan.
    S_kv_prev = kv_cache[_SLOT_FWD_KV]
    S_z_prev = kv_cache[_SLOT_FWD_Z]
    out_4d, S_kv_new, S_z_new = _gdn_main_triton(
        layer,
        qkv,
        beta,
        decay,
        rotary_emb,
        HW,
        S_kv_prev,
        S_z_prev,
        save_kv_cache,
    )

    if save_kv_cache:
        kv_cache[_SLOT_FWD_KV] = S_kv_new.detach().clone()
        kv_cache[_SLOT_FWD_Z] = S_z_new.detach().clone()
        kv_cache[_SLOT_TYPE_FLAG] = _TYPE_STATE

    # 5. Output gate + projection, matching the torch path's tail.
    out = out_4d.reshape(B, N, C)
    if apply_output_gate:
        out = layer._apply_output_gate(out, x)
        out = layer.proj(out.to(layer.proj.weight.dtype))
        return out, kv_cache
    return out, kv_cache
