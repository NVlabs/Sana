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

"""Triton-fused GDN attention blocks (single-GPU inference).

Drop-in replacements for the chunk-causal and bidirectional GDN attention
blocks whose main GDN scan (and, optionally, camera branch) is rewired
through a fused mega-kernel (:mod:`diffusion.model.ops.fused_gdn` for the
main branch and :mod:`diffusion.model.ops.fused_cam_gdn` for the camera
branch). All learnable parameters, sub-modules and state-dict keys are
inherited unchanged from the PyTorch baselines, so an existing checkpoint
loads cleanly with zero conversion.

The following features are not supported on the Triton path and will
raise ``NotImplementedError``:

* Per-frame validity masking (``frame_valid_mask``).
* Q/V short convolutions (only ``k_conv_only=True`` is honoured).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from diffusion.model.nets.sana_camctrl_blocks import (
    _maybe_drop_cam_branch,
    _prepare_ray_apply_fns,
)
from diffusion.model.nets.sana_gdn_blocks import BidirectionalGDN
from diffusion.model.nets.sana_gdn_camctrl_blocks import (
    BidirectionalGDNUCPESinglePathLiteLA,
)
from diffusion.model.ops.fused_cam_gdn import (
    _invert_SE3,
    _prepare_ucpe_rope_tables,
    _process_camera_conditions_raymats_only,
    cam_prep_func,
)
from diffusion.model.ops.fused_gdn import (
    fused_bigdn_func,
    fused_qk_inv_rms,
    prepare_rope_tables,
)
from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_bidi_chunkwise

# CUDA chunkwise cam scan (on by default; set SANA_GDN_CUDA=0 to disable): bf16
# CUDA kernels that read q/k/v [B,H,D,N] directly + write the transposed output
# directly (no packing/transpose glue) -> ~38x less cam transient VRAM. The
# drop-in falls back to Triton for unsupported shapes/dtype and, if the CUDA
# extension can't build (no nvcc/toolkit), latches off to Triton at runtime.
if os.environ.get("SANA_GDN_CUDA", "1") != "0":
    try:
        from diffusion.model.ops.fused_gdn_chunkwise_cuda import cam_scan_bidi_chunkwise_cuda as cam_scan_bidi_chunkwise
    except Exception as _e:
        print(f"[SANA_GDN_CUDA] CUDA cam path unavailable ({_e}); using Triton")
from diffusion.model.registry import ATTENTION_BLOCKS


@ATTENTION_BLOCKS.register_module()
class BidirectionalGDNTriton(BidirectionalGDN):
    """Bidirectional GDN with a fused Triton scan (inference + opt-in autograd).

    Subclasses :class:`BidirectionalGDN` and only overrides :meth:`__init__`
    (to accept ``use_autograd_kernel``) and :meth:`forward`.  Every learned
    sub-module (``qkv``, ``proj``, ``q_norm``, ``k_norm``, ``conv_k``,
    ``beta_proj``, ``gate_proj``, ``A_log``, ``dt_bias``, ``output_gate``)
    and helper (``_apply_temporal_short_conv``, ``_compute_frame_gates``,
    ``_apply_output_gate``) is inherited unchanged so existing checkpoints
    load with zero conversion.

    When ``use_autograd_kernel=True`` the fused-kernel call switches to
    :func:`fused_bigdn_forward_with_grad` (autograd-enabled, identical
    forward, real Triton backward kernel for the main branch).
    """

    def __init__(self, *args, use_autograd_kernel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_autograd_kernel = use_autograd_kernel

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        # ---- Guards: this path supports inference only. -------------------
        if HW is None:
            raise ValueError("BidirectionalGDNTriton requires HW=(T, H, W).")
        del mask, block_mask  # unused in the bidirectional Triton path
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "BidirectionalGDNTriton does not support frame_valid_mask (training-only feature)."
            )
        if self.conv_q is not None or self.conv_v is not None:
            raise NotImplementedError("BidirectionalGDNTriton requires k_conv_only=True; got conv_q or conv_v.")

        B, N, C = x.shape
        T, H_s, W_s = HW
        S = H_s * W_s
        H, D = self.heads, self.dim
        if N != T * S:
            raise ValueError(f"N={N} != T*S={T * S} for HW={HW}.")
        if C != H * D:
            raise ValueError(f"C={C} != heads*dim={H * D}.")

        # ---- 1. QKV projection -> (B, N, 3, H, D), kept contiguous. -------
        qkv = self.qkv(x).reshape(B, N, 3, H, D)

        # ---- 2. Bidirectional short conv on K (parent method).  ----------
        # ``BidirectionalGDN._apply_temporal_short_conv`` runs the causal
        # conv forward + backward then averages, giving a symmetric filter
        # with one set of weights.  Inherited unchanged.
        if self.conv_k is not None:
            k_raw = qkv[:, :, 1].contiguous().reshape(B, N, C)
            k_conv = self._apply_temporal_short_conv(k_raw, self.conv_k, HW)
            qkv[:, :, 1].copy_(k_conv.reshape(B, N, H, D))

        # ---- 3. Frame gates (precomputed when shared with cam branch). ----
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        beta = beta.contiguous()
        decay = decay.contiguous()

        # ---- 4. Full-channel RMSNorm weights. -----------------------------
        if not isinstance(self.q_norm, nn.Identity):
            q_nw = self.q_norm.weight.float().contiguous()
            k_nw = self.k_norm.weight.float().contiguous()
            norm_eps = float(getattr(self.q_norm, "eps", 1e-5))
        else:
            q_nw = torch.ones(C, device=x.device, dtype=torch.float32)
            k_nw = torch.ones(C, device=x.device, dtype=torch.float32)
            norm_eps = 1e-5

        # ---- 5. Fused Q+K inverse-RMS (single Triton launch). -------------
        q_inv_rms, k_inv_rms = fused_qk_inv_rms(qkv, eps=norm_eps)

        # ---- 6. Expanded RoPE cos/sin tables (N, D). ---------------------
        rope_cos, rope_sin = prepare_rope_tables(rotary_emb, N, D, x.device)

        # ---- 7. K scale absorbs Q/K^T variance + spatial mean-pool. -----
        k_scale = (D**-0.5) * (S**-0.5)

        # ---- 8. Fused bidirectional Triton scan over the full sequence. --
        # No ``*_bwd`` overrides: the kernel's ``reverse=True`` path already
        # implements the exclusive (t+1..T) reverse recurrence, matching the
        # torch ``flip_and_shift`` semantics used in ``BidirectionalGDN``.
        out = fused_bigdn_func(
            qkv,
            q_inv_rms,
            k_inv_rms,
            q_norm_weight=q_nw,
            k_norm_weight=k_nw,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            beta=beta,
            decay=decay,
            F=T,
            S=S,
            k_scale=k_scale,
            eps=self.eps,
        )  # (B, N, H, D)

        # ---- 9. Output gate + projection. --------------------------------
        out = out.reshape(B, N, C)
        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
        return out


@ATTENTION_BLOCKS.register_module()
class BidirectionalGDNUCPESinglePathLiteLABothTriton(BidirectionalGDNUCPESinglePathLiteLA):
    """Bidirectional UCPE camera-controlled GDN with **both** branches on Triton.

    Main branch: :class:`BidirectionalGDNTriton`-fused scan, dispatched via
    ``BidirectionalGDNTriton.forward(self, ...)`` from this class's
    :meth:`forward` because MRO would otherwise pick up the torch
    :class:`BidirectionalGDN` parent.

    Cam branch: :meth:`_forward_cam_branch` runs a fused Triton pipeline:

        1. Torch QKV linear + bidirectional short conv on K.
        2. UCPE ``P / P_T / P_inv`` from ``camera_conditions``.
        3. Sliced cam-branch RoPE → interleaved ``(N, D/2)`` cos/sin tables.
        4. Fused prep kernel (RMSNorm + ReLU + K-scale + UCPE 4x4 + RoPE),
           emitting ``inflation_sq`` for Dynamic Beta Discounting.
        5. Beta discounting via ``inflation_sq``.
        6. Fused forward scan (``reverse=False``) over the full sequence.
        7. Fused reverse scan (``reverse=True``) over the full sequence —
           the kernel applies flip-and-shift internally, so no per-chunk
           loop is needed.
        8. Inverse UCPE (``apply_fn_o``) in torch.

    State-dict keys are identical to
    :class:`BidirectionalGDNUCPESinglePathLiteLA`.

    Set ``use_autograd_kernel=True`` to enable autograd mode for both
    branches: the main branch goes through :func:`fused_bigdn_forward_with_grad`
    and the cam branch through :func:`cam_prep_func_with_grad` +
    :func:`cam_scan_func_with_grad` (torch-recompute backward fallback).
    Forward cost is unchanged.
    """

    def __init__(self, *args, use_autograd_kernel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_autograd_kernel = use_autograd_kernel

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # Pre-compute shared gates once for both branches.
        if HW is not None:
            precomputed_gates = self._compute_frame_gates(x, HW)
        else:
            precomputed_gates = None

        # Main branch — Triton-fused bidirectional scan.
        main_raw = BidirectionalGDNTriton.forward(
            self,
            x,
            mask=mask,
            HW=HW,
            rotary_emb=rotary_emb,
            block_mask=block_mask,
            apply_output_gate=False,
            chunk_size=chunk_size,
            precomputed_gates=precomputed_gates,
            **kwargs,
        )

        # Camera branch (inherited torch implementation).
        cam_contrib: torch.Tensor | int = 0
        camera_conditions = _maybe_drop_cam_branch(
            camera_conditions,
            kwargs.get("cam_branch_drop_prob", 0.0),
            self.training,
            x.device,
        )
        if camera_conditions is not None:
            if HW is None:
                raise ValueError("HW (T, H, W) must be provided for UCPE camera branch.")
            cam_raw = self._forward_cam_branch(
                x,
                HW,
                camera_conditions,
                rotary_emb,
                chunk_size=chunk_size,
                precomputed_gates=precomputed_gates,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        return self.proj(combined.to(self.proj.weight.dtype))

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        # ---- Guards: k_conv_only=True. ----
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "BidirectionalGDNUCPESinglePathLiteLABothTriton does not "
                "support frame_valid_mask (training-only feature)."
            )
        if self.conv_q_cam is not None or self.conv_v_cam is not None:
            raise NotImplementedError(
                "BidirectionalGDNUCPESinglePathLiteLABothTriton requires "
                "k_conv_only=True (conv_q_cam / conv_v_cam must be None)."
            )

        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        dtype_orig = x.dtype
        H_heads = self.cam_heads
        D_head = self.cam_head_dim

        # ---- 1. QKV linear + bidirectional short conv on K ---------------
        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        qkv_cam = torch.nn.functional.linear(x, qkv_w, qkv_b)
        q_raw, k_raw, v_raw = qkv_cam.chunk(3, dim=-1)

        if self.conv_k_cam is not None:
            # Parent routing (BidirectionalGDN) gives the bidirectional
            # forward+backward causal conv + average.
            k_raw = self._apply_temporal_short_conv(k_raw, self.conv_k_cam, HW)

        q_raw = q_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        k_raw = k_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        v_raw = v_raw.contiguous().view(B, N, H_heads, D_head).contiguous()

        # ---- 2. UCPE P, P_T, P_inv (inline; skip cached prope_fns). -----
        raymats = _process_camera_conditions_raymats_only(camera_conditions, B, HW, self.patch_size)
        raymats = raymats.reshape(B, -1, 4, 4)
        P = raymats
        P_T = P.transpose(-1, -2).contiguous()
        P_inv = _invert_SE3(P).contiguous()

        # ---- 3. Sliced cam-branch RoPE + interleaved tables. ------------
        if rotary_emb is not None:
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
            rope_cos, rope_sin = _prepare_ucpe_rope_tables(rotary_emb_cam, N, D_head // 2, x.device)
        else:
            rotary_emb_cam = None
            rope_cos = torch.ones(N, D_head // 2, device=x.device, dtype=torch.float32)
            rope_sin = torch.zeros(N, D_head // 2, device=x.device, dtype=torch.float32)

        # ---- 4. Fused Triton prep kernel --------------------------------
        q_norm_w = self.q_norm_cam.weight.float().contiguous()
        k_norm_w = self.k_norm_cam.weight.float().contiguous()
        k_scale = (D_head**-0.5) * (S**-0.5)
        norm_eps_val = float(
            getattr(
                self.q_norm_cam,
                "eps",
                getattr(self.q_norm_cam, "variance_epsilon", 1e-6),
            )
        )
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

        # ---- 5. Gates + beta discounting -------------------------------
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        inflation_sq_spatial = inflation_sq.view(B, H_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        # ---- 6. fp32 cast + broadcast beta to (B, H, F, S) -------------
        if getattr(self, "fp32_attention", True):
            q_cam_trans = q_cam_trans.float()
            k_cam_trans = k_cam_trans.float()
            v_cam_trans = v_cam_trans.float()
            beta = beta.float()
            decay = decay.float()
        if beta.ndim == 3:
            beta = beta.unsqueeze(-1).expand(B, H_heads, T, S).contiguous()
        else:
            assert beta.shape == (B, H_heads, T, S), f"beta shape {beta.shape}"
            beta = beta.contiguous()
        decay = decay.contiguous()

        q_cam_trans = q_cam_trans.contiguous()
        k_cam_trans = k_cam_trans.contiguous()
        v_cam_trans = v_cam_trans.contiguous()

        # ---- 7. Fused bidirectional chunkwise scan. --------------------
        out = cam_scan_bidi_chunkwise(q_cam_trans, k_cam_trans, v_cam_trans, beta, decay)

        # ---- 9. Cast back to input dtype, then inverse UCPE. -----------
        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        _, _, apply_fn_o = _prepare_ray_apply_fns(
            head_dim=D_head,
            P=P,
            P_T=P_T,
            P_inv=P_inv,
            rotary_emb=rotary_emb_cam,
        )
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        out = out.reshape(B, self.cam_dim, -1).permute(0, 2, 1)
        return out
