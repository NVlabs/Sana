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

"""Triton-fused GDN attention blocks (inference + opt-in autograd training).

Drop-in replacements for the chunk-causal and bidirectional GDN attention
blocks whose main GDN scan (and, optionally, camera branch) is rewired
through a fused mega-kernel (:mod:`diffusion.model.ops.fused_gdn` for the
main branch and :mod:`diffusion.model.ops.fused_cam_gdn` for the camera
branch). All learnable parameters, sub-modules and state-dict keys are
inherited unchanged from the PyTorch baselines, so an existing checkpoint
loads cleanly with zero conversion.

Inference is enabled by default; pass ``use_autograd_kernel=True`` (e.g.,
via config ``model.use_autograd_kernel``) to enable training-time gradient
flow through the same fused forward. The following features are not
supported on the Triton path and will raise ``NotImplementedError``:

* Context-parallel scan (``cp_enabled``).
* Per-frame validity masking (``frame_valid_mask``).
* Q/V short convolutions (only ``k_conv_only=True`` is honoured).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion.distributed.context_parallel.config import cp_enabled
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
from diffusion.model.registry import ATTENTION_BLOCKS
from diffusion.utils.chunk_utils import (
    is_chunk_causal_request,
    normalize_chunk_index,
    size1_chunk_position_indices,
)


def _mask_reverse_gates_for_chunk_boundaries(
    beta: torch.Tensor,
    decay: torch.Tensor,
    valid_chunk_index: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero reverse-scan gates where chunk-local anti-causal state must reset."""
    interior = [i for i in valid_chunk_index if 0 < i < beta.shape[2]]
    size1_positions = size1_chunk_position_indices(valid_chunk_index)
    bwd_zero_positions = sorted(set(interior) | set(size1_positions))
    if not bwd_zero_positions:
        return beta, decay

    beta_bwd = beta.clone()
    decay_bwd = decay.clone()
    beta_bwd[:, :, bwd_zero_positions, :] = 0.0
    decay_bwd[:, :, bwd_zero_positions] = 0.0
    return beta_bwd, decay_bwd


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

    def _forward_cp_scan_triton_ag(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int],
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        """Bidirectional CP-fused forward dispatch (forward-fused + backward-eager).

        Forward fusion routes through :func:`cp_fused_gdn_chunkwise_raw_autograd`.
        The backward (anti-causal) recurrence runs an eager
        :func:`_build_transition_matrices` + :func:`cp_frame_gdn_scan(reverse=True)`
        + :func:`compiled_gdn_output_projection` chain.

        Args:
            x: ``(B, N_local, C)`` CP-local input slice.
            mask: Unused (API symmetry).
            HW: ``(T_local, H, W)`` token layout for this rank.
            rotary_emb: CP-local RoPE complex frequencies.
            block_mask: Unused (API symmetry).
            apply_output_gate: When False, return raw attention output before
                gate + projection.

        Returns:
            ``(B, N_local, C)`` after attention + (optional) output gate + projection.
        """
        import torch.distributed as dist
        from torch.utils.checkpoint import checkpoint as grad_checkpoint

        from diffusion.distributed.context_parallel.config import get_cp_group
        from diffusion.distributed.context_parallel.distributed_scan import cp_frame_gdn_scan
        from diffusion.distributed.context_parallel.halo_exchange import cp_halo_exchange
        from diffusion.model.ops.frame_gdn.api import (
            _build_transition_matrices,
            compiled_gdn_output_projection,
        )
        from diffusion.model.ops.fused_gdn_cp import cp_fused_gdn_chunkwise_raw_autograd

        del mask, block_mask  # unused on this path

        # ---- Guards: training-only / AR-layout rejections. ----
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "BidirectionalGDNTriton CP-Triton path does not support " "frame_valid_mask (training-only feature)."
            )
        if self.conv_q is not None or self.conv_v is not None:
            raise NotImplementedError(
                "BidirectionalGDNTriton CP-Triton path supports k_conv_only="
                "True; got conv_q or conv_v which would require additional "
                "Triton paths."
            )
        if kwargs.get("cp_ar_layout", None) is not None:
            # Bidirectional teacher / critic blocks are not used under AR
            # layouts. Reject explicitly so an unexpected caller cannot
            # silently pick up alternate CP semantics through this path.
            raise NotImplementedError(
                "BidirectionalGDNTriton CP-Triton path does not support "
                "cp_ar_layout (teacher / critic blocks are not used under "
                "AR layouts). Disable CP_TRITON_BLOCK_FUSION (eager fallback) "
                "for this layout."
            )

        B, N, C = x.shape
        T, H_s, W_s = HW
        S = H_s * W_s
        H, D = self.heads, self.dim
        if N != T * S:
            raise ValueError(f"N={N} != T*S={T * S} for HW={HW}.")
        if C != H * D:
            raise ValueError(f"C={C} != heads*dim={H * D}.")

        cp_group = get_cp_group()
        if cp_group is None:
            raise RuntimeError(
                "BidirectionalGDNTriton._forward_cp_scan_triton_ag called but " "CP group is not initialized."
            )

        # ---- 1. QKV projection on the CP-local slice. ---------------------
        qkv = self.qkv(x).reshape(B, N, 3, H, D)

        # ---- 2. Bidirectional short conv on K (parent method). ----
        # ``BidirectionalGDN._apply_temporal_short_conv`` handles the CP
        # halo exchange internally (sana_gdn_blocks.py:1043-1050) so we
        # can reuse it verbatim.
        if self.conv_k is not None:
            k_raw = qkv[:, :, 1].contiguous().reshape(B, N, C)
            k_conv = self._apply_temporal_short_conv(k_raw, self.conv_k, HW)
            qkv = qkv.clone()
            qkv[:, :, 1] = k_conv.reshape(B, N, H, D)

        # ---- 3. Frame gates (precomputed when shared with cam branch). ----
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        beta = beta.contiguous()
        decay = decay.contiguous()

        # ---- 4. Full-channel RMSNorm weights + norm_eps. -----------------
        if not isinstance(self.q_norm, nn.Identity):
            q_nw = self.q_norm.weight.float().contiguous()
            k_nw = self.k_norm.weight.float().contiguous()
            norm_eps = float(getattr(self.q_norm, "eps", 1e-5))
        else:
            q_nw = None
            k_nw = None
            norm_eps = 1e-5

        # ---- 5. CP-local RoPE tables. ------------------------------------
        rope_cos, rope_sin = prepare_rope_tables(rotary_emb, N, D, x.device)

        # ---- 6. K scale via the model's documented mode. -----------------
        k_scale = self._key_scale(S)

        # ---- 7. dot_precision: fp32 inputs -> IEEE fp32 bridge (2);
        # bf16/fp16 -> TF32 bf16 bridge (0).
        dot_precision = 2 if x.dtype == torch.float32 else 0

        # ============================================================
        #  FORWARD BRANCH (fused Triton autograd entry)
        # ============================================================
        # The ``grad_checkpoint`` wrap around the fused-forward call is
        # handled internally by ``cp_fused_gdn_chunkwise_raw_autograd``
        # (see ``fused_gdn_cp.py``, ``use_checkpoint`` arg).
        res = cp_fused_gdn_chunkwise_raw_autograd(
            qkv,
            beta,
            decay,
            q_nw,
            k_nw,
            rope_cos,
            rope_sin,
            F=T,
            S=S,
            group=cp_group,
            k_scale=k_scale,
            norm_eps=norm_eps,
            eps=self.eps,
            dot_precision=dot_precision,
            reverse_rank_order=False,
            truncate_to_active=None,
            # ``use_checkpoint=None`` → auto-detect.
        )
        num_fwd, den_fwd = res.num, res.den

        # ============================================================
        #  BACKWARD BRANCH (eager, mirrors sana_gdn_blocks.py:1138-1209)
        # ============================================================
        # Recompute q, k, v, q_rot, k_rot from `qkv` (post-conv) so we can
        # hand them to `_build_transition_matrices` exactly as eager does.
        # FP32 for numerical stability (matches eager's `fp32_attention=True`
        # default at sana_gdn_blocks.py:1096-1099).
        BH = B * self.heads

        # Apply RMSNorm to Q and K (eager order: Q/K norm BEFORE ReLU +
        # k_scale BEFORE rope), mirroring sana_gdn_blocks.py:1291-1299.
        q_raw = qkv[:, :, 0].reshape(B, N, C)
        k_raw_bwd = qkv[:, :, 1].reshape(B, N, C)
        v_raw_bwd = qkv[:, :, 2].reshape(B, N, C)
        q_normed = self.q_norm(q_raw)
        k_normed = self.k_norm(k_raw_bwd)
        q_bnhd = q_normed.reshape(B, N, H, D)
        k_bnhd = k_normed.reshape(B, N, H, D)
        v_bnhd = v_raw_bwd.reshape(B, N, H, D)
        # ReLU kernel + k_scale (eager: lines 1295-1299).
        q_bnhd = self.kernel_func(q_bnhd)
        k_bnhd = self.kernel_func(k_bnhd)
        k_bnhd = k_bnhd * k_scale
        # Permute to (B, H, D, N).
        # Do NOT call ``.contiguous()`` before rotary: ``_apply_rotary_emb``
        # requires the non-contiguous stride layout produced by
        # ``permute(0, 2, 3, 1)`` so its internal ``permute+to+unflatten``
        # chain ends with a stride-1 final dim.
        q_perm = q_bnhd.permute(0, 2, 3, 1)
        k_perm = k_bnhd.permute(0, 2, 3, 1)
        v_perm = v_bnhd.permute(0, 2, 3, 1)
        if rotary_emb is not None:
            q_rot_perm = self._apply_rotary_emb(q_perm, rotary_emb)
            k_rot_perm = self._apply_rotary_emb(k_perm, rotary_emb)
        else:
            q_rot_perm = q_perm
            k_rot_perm = k_perm

        # Promote to fp32 (matches eager `fp32_attention=True`).
        if getattr(self, "fp32_attention", True):
            q_perm = q_perm.float()
            k_perm = k_perm.float()
            v_perm = v_perm.float()
            q_rot_perm = q_rot_perm.float()
            k_rot_perm = k_rot_perm.float()
            beta_bwd_src = beta.float()
            decay_bwd_src = decay.float()
        else:
            # Re-establish contiguous layout for downstream ``_to_frame``
            # ``.view`` when fp32 promotion is disabled (mirrors the
            # ChunkCausal counterpart's bf16 branch).
            q_perm = q_perm.contiguous()
            k_perm = k_perm.contiguous()
            v_perm = v_perm.contiguous()
            q_rot_perm = q_rot_perm.contiguous()
            k_rot_perm = k_rot_perm.contiguous()
            beta_bwd_src = beta
            decay_bwd_src = decay

        # Reshape to (B, H, T, D, S) frame layout via eager `to_frame`
        # semantics.
        def _to_frame(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, self.heads, D, T, S).permute(0, 1, 3, 2, 4).contiguous()

        q_f = _to_frame(q_perm)
        k_f = _to_frame(k_perm)
        v_f = _to_frame(v_perm)
        q_rot_f = _to_frame(q_rot_perm)
        k_rot_f = _to_frame(k_rot_perm)
        if beta_bwd_src.ndim == 4:
            beta_f = beta_bwd_src.unsqueeze(3)
        else:
            beta_f = beta_bwd_src.view(B, self.heads, T, 1, 1)
        decay_f = decay_bwd_src.view(B, self.heads, T, 1, 1)
        I = torch.eye(D, device=q_perm.device, dtype=q_perm.dtype).reshape(1, 1, 1, D, D)

        # Distributed flip+shift (mirrors the bidirectional path in
        # sana_gdn_blocks.py). No chunk-boundary masking is applied on
        # the flip+shift inputs.
        cp_world = dist.get_world_size(cp_group)
        cp_rank_local = dist.get_rank(cp_group)

        def _cp_flip_and_shift(tensors: list[torch.Tensor], shift_vals: list[float]) -> list[torch.Tensor]:
            is_last = cp_rank_local == cp_world - 1
            results = []
            for tensor, sv in zip(tensors, shift_vals):
                first_frame = tensor[:, :, :1, ...].contiguous()
                haloed = cp_halo_exchange(first_frame, left_size=0, right_size=1, dim=2, group=cp_group)
                boundary = haloed[:, :, 1:2, ...]
                if is_last and sv != 0.0:
                    boundary = boundary.mul(0.0).add(sv)
                T_loc = tensor.shape[2]
                flipped = torch.flip(tensor, dims=[2])
                body = flipped[:, :, : T_loc - 1, ...]
                results.append(torch.cat([boundary, body], dim=2))
            return results

        q_bwd_f = torch.flip(q_f, dims=[2])
        q_rot_bwd_f = torch.flip(q_rot_f, dims=[2])

        k_bwd_f, v_bwd_f, k_rot_bwd_f, beta_bwd_f, decay_bwd_f = _cp_flip_and_shift(
            [k_f, v_f, k_rot_f, beta_f, decay_f],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        )

        W_kv_bwd, U_kv_bwd, W_z_bwd, U_z_bwd = grad_checkpoint(
            _build_transition_matrices,
            k_bwd_f,
            v_bwd_f,
            k_rot_bwd_f,
            beta_bwd_f,
            decay_bwd_f,
            I,
            BH,
            T,
            D,
            use_reentrant=False,
        )
        S_kv_bwd, S_z_bwd = cp_frame_gdn_scan(W_kv_bwd, U_kv_bwd, W_z_bwd, U_z_bwd, cp_group, reverse=True)
        S_kv_bwd = S_kv_bwd.view(B, self.heads, T, D, D)
        S_z_bwd = S_z_bwd.view(B, self.heads, T, D)
        num_bwd_flipped, den_bwd_flipped = compiled_gdn_output_projection(
            S_kv_bwd,
            S_z_bwd,
            q_rot_bwd_f,
            q_bwd_f,
            eps=self.eps,
        )
        num_bwd_eager = torch.flip(num_bwd_flipped, dims=[2])  # (B, H, T, D, S)
        den_bwd_eager = torch.flip(den_bwd_flipped, dims=[2])  # (B, H, T, 1, S)

        # ============================================================
        #  COMBINE & FINAL DIVIDE
        # ============================================================
        # num_fwd is (B, N=T*S, H, D); re-layout to (B, H, T, D, S).
        num_fwd_5d = num_fwd.reshape(B, T, S, H, D).permute(0, 3, 1, 4, 2).contiguous()
        # den_fwd is (B, H, N=T*S); re-layout to (B, H, T, 1, S).
        den_fwd_5d = den_fwd.reshape(B, H, T, S).unsqueeze(3).contiguous()

        # Promote to fp32 for the sum + divide (mirrors eager line 1206-1209).
        total_num = num_fwd_5d.float() + num_bwd_eager.float()
        total_den = den_fwd_5d.float() + den_bwd_eager.float()

        out = total_num / (total_den + self.eps)  # (B, H, T, D, S)
        if getattr(self, "fp32_attention", True) and x.dtype != torch.float32:
            out = out.to(x.dtype)

        # (B, H, T, D, S) -> (B, H, D, T, S) -> (B, H, D, N) -> (B, N, C).
        out = out.permute(0, 1, 3, 2, 4).reshape(B, self.heads, D, N)
        out = out.permute(0, 3, 1, 2).reshape(B, N, C)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
        return out

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
        if cp_enabled():
            # Gate-controlled dispatch.
            # Default (CP_TRITON_BLOCK_FUSION=False): fall back to the eager
            # non-Triton parent's ``_forward_cp_scan``.
            # Opt-in (CP_TRITON_BLOCK_FUSION=True): route through
            # ``_forward_cp_scan_triton_ag`` which uses fused-forward
            # (``cp_fused_gdn_chunkwise_raw_autograd``) + eager-backward
            # (``cp_frame_gdn_scan(reverse=True)``).
            from diffusion.distributed.context_parallel.config import (
                get_cp_triton_block_fusion,
            )

            if not get_cp_triton_block_fusion():
                return BidirectionalGDN.forward(
                    self,
                    x,
                    mask=mask,
                    HW=HW,
                    rotary_emb=rotary_emb,
                    block_mask=block_mask,
                    apply_output_gate=apply_output_gate,
                    **kwargs,
                )
            return self._forward_cp_scan_triton_ag(
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                apply_output_gate=apply_output_gate,
                **kwargs,
            )
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
class BidirectionalGDNUCPESinglePathLiteLATriton(BidirectionalGDNUCPESinglePathLiteLA):
    """Bidirectional UCPE camera-controlled GDN with a Triton main branch.

    Inherits the entire camera branch (``_forward_cam_branch``),
    ``_prepare_cam_qkv``, every sub-module and every checkpoint key from
    :class:`BidirectionalGDNUCPESinglePathLiteLA`.  The **only** behavioural
    delta is that the main-branch GDN scan dispatches through
    :class:`BidirectionalGDNTriton.forward` instead of the inherited
    :class:`BidirectionalGDN.forward`.

    Because ``_GDNUCPEBase.forward`` routes the main branch via
    ``super().forward(...)`` — which MRO-resolves to
    :class:`BidirectionalGDN`, not our Triton variant — we re-implement the
    dual-branch forward here to explicitly call
    ``BidirectionalGDNTriton.forward(self, ...)``.  The body is otherwise
    bit-identical to the parent's ``forward``.

    The ``use_autograd_kernel`` flag is stored on this instance and consulted
    inside :meth:`BidirectionalGDNTriton.forward` (the dispatch passes
    ``self``, so the flag is visible to the main-branch forward).  The cam
    branch is the inherited torch path; use
    :class:`BidirectionalGDNUCPESinglePathLiteLABothTriton` for a fully
    Triton + autograd-aware cam branch.
    """

    # The Both-Triton class hierarchy does not include
    # :class:`BidirectionalGDNTriton` in its MRO. ``forward`` calls
    # ``BidirectionalGDNTriton.forward(self, ...)`` as an unbound method
    # which then dispatches to ``self._forward_cp_scan_triton_ag(...)``
    # when ``CP_TRITON_BLOCK_FUSION=True``. Re-export the method here so
    # subclasses (LATriton and BothTriton) resolve the lookup correctly.
    _forward_cp_scan_triton_ag = BidirectionalGDNTriton._forward_cp_scan_triton_ag

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
        if self.cam_debug_ratios:
            self.reset_cam_debug_stats()
        if self.training:
            self._cam_debug_step_counter += 1

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


@ATTENTION_BLOCKS.register_module()
class BidirectionalGDNUCPESinglePathLiteLABothTriton(BidirectionalGDNUCPESinglePathLiteLATriton):
    """Bidirectional UCPE camera-controlled GDN with **both** branches on Triton.

    Subclasses :class:`BidirectionalGDNUCPESinglePathLiteLATriton` (which
    already rewires the main GDN scan) and replaces
    :meth:`_forward_cam_branch` with a fused Triton camera pipeline:

        1. Torch QKV linear + bidirectional short conv on K.
        2. UCPE ``P / P_T / P_inv`` from ``camera_conditions``.
        3. Sliced cam-branch RoPE → interleaved ``(N, D/2)`` cos/sin tables.
        4. Fused prep kernel (RMSNorm + ReLU + K-scale + UCPE 4x4 + RoPE),
           emitting ``inflation_sq`` for Dynamic Beta Discounting.
        5. Beta discounting via ``inflation_sq`` (mirrors torch path).
        6. Fused forward scan (``reverse=False``) over the full sequence.
        7. Fused reverse scan (``reverse=True``) over the full sequence —
           the kernel applies flip-and-shift internally, so no per-chunk
           loop is needed.
        8. Inverse UCPE (``apply_fn_o``) in torch.

    State-dict keys are identical to
    :class:`BidirectionalGDNUCPESinglePathLiteLA`.

    Set ``use_autograd_kernel=True`` (inherited from
    :class:`BidirectionalGDNUCPESinglePathLiteLATriton`) to enable autograd
    mode for both branches: the main branch goes through
    :func:`fused_bigdn_forward_with_grad` and the cam branch through
    :func:`cam_prep_func_with_grad` + :func:`cam_scan_func_with_grad`
    (torch-recompute backward fallback).  Forward cost is unchanged.
    """

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        # ---- Guards: no CP, k_conv_only=True (apply in either mode). ----
        if cp_enabled():
            # Gate-controlled dispatch (default OFF).
            # Default (CP_TRITON_BLOCK_FUSION=False): fall back to the eager
            # non-Triton parent's ``_forward_cam_branch``.
            # Opt-in (CP_TRITON_BLOCK_FUSION=True): route through
            # ``_forward_cam_branch_cp_triton_ag`` (bidirectional camera
            # variant, full-sequence reverse via
            # ``cp_fused_cam_gdn_num_autograd(reverse_rank_order=True)``).
            from diffusion.distributed.context_parallel.config import (
                get_cp_triton_block_fusion,
            )

            if not get_cp_triton_block_fusion():
                return BidirectionalGDNUCPESinglePathLiteLA._forward_cam_branch(
                    self, x, HW, camera_conditions, rotary_emb, **kwargs
                )
            return self._forward_cam_branch_cp_triton_ag(x, HW, camera_conditions, rotary_emb, **kwargs)
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

    def _forward_cam_branch_cp_triton_ag(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Bidirectional CP-aware Triton-fused cam branch.

        Activated when ``CP_TRITON_BLOCK_FUSION=True``:

        * Forward branch uses Triton-fused :func:`cam_prep_func_with_grad`
          (RMSNorm + ReLU + K-scale + UCPE-projmat + RoPE) followed by
          :func:`cp_fused_cam_gdn_num_autograd`.
        * Backward branch uses an eager ``_cp_flip_and_shift`` +
          :func:`_build_transition_matrices` +
          :func:`cp_frame_gdn_scan(reverse=True)` + matmul output, full
          sequence with no chunk-boundary masking.
        """
        import torch.distributed as dist

        from diffusion.distributed.context_parallel.config import get_cp_group
        from diffusion.distributed.context_parallel.distributed_scan import cp_frame_gdn_scan
        from diffusion.distributed.context_parallel.halo_exchange import cp_halo_exchange
        from diffusion.model.ops.frame_gdn.api import _build_transition_matrices
        from diffusion.model.ops.fused_gdn_cp import cp_fused_cam_gdn_num_autograd

        # ---- Guards: training-only / AR-layout / Q-V conv rejections. ----
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "BidirectionalGDNUCPESinglePathLiteLABothTriton CP-Triton "
                "cam branch does not support frame_valid_mask (training-only feature)."
            )
        if self.conv_q_cam is not None or self.conv_v_cam is not None:
            raise NotImplementedError(
                "BidirectionalGDNUCPESinglePathLiteLABothTriton CP-Triton "
                "cam branch requires k_conv_only=True (conv_q_cam / "
                "conv_v_cam must be None)."
            )
        if kwargs.get("cp_ar_layout", None) is not None:
            # Bidirectional camera is not used under AR layouts. Reject
            # explicitly so an unexpected caller cannot silently pick up
            # alternate CP semantics through this path.
            raise NotImplementedError(
                "BidirectionalGDNUCPESinglePathLiteLABothTriton CP-Triton "
                "cam branch does not support cp_ar_layout (bidi teacher / "
                "critic are not used under AR layouts). Disable "
                "CP_TRITON_BLOCK_FUSION for this layout."
            )

        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        dtype_orig = x.dtype
        H_heads = self.cam_heads
        D_head = self.cam_head_dim

        cp_group = get_cp_group()
        if cp_group is None:
            raise RuntimeError("_forward_cam_branch_cp_triton_ag (bidi) called but CP group is not initialized.")

        # ---- 1. QKV linear + bidirectional short conv on K (CP-local). ----
        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        qkv_cam = torch.nn.functional.linear(x, qkv_w, qkv_b)
        q_raw, k_raw, v_raw = qkv_cam.chunk(3, dim=-1)

        if self.conv_k_cam is not None:
            # Parent (BidirectionalGDN._apply_temporal_short_conv) handles
            # CP halo exchange internally.
            k_raw = self._apply_temporal_short_conv(k_raw, self.conv_k_cam, HW)

        q_raw = q_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        k_raw = k_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        v_raw = v_raw.contiguous().view(B, N, H_heads, D_head).contiguous()

        # ---- 2. UCPE P, P_T, P_inv (CP-local segment). ----
        raymats = _process_camera_conditions_raymats_only(camera_conditions, B, HW, self.patch_size)
        raymats = raymats.reshape(B, -1, 4, 4)
        P = raymats
        P_T = P.transpose(-1, -2).contiguous()
        P_inv = _invert_SE3(P).contiguous()

        # ---- 3. Sliced cam-branch RoPE + interleaved tables (CP-local). ----
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

        # ---- 4. Fused Triton prep kernel (RMSNorm+ReLU+K-scale+UCPE+RoPE). ----
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

        # ---- 5. Gates + beta discounting (camera inflation-sq). ----
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

        # ---- 6. fp32 cast + broadcast beta to (B, H, F, S). ----
        if getattr(self, "fp32_attention", True):
            q_cam_trans = q_cam_trans.float()
            k_cam_trans = k_cam_trans.float()
            v_cam_trans = v_cam_trans.float()
            beta = beta.float()
            decay = decay.float()
        if beta.ndim == 3:
            beta_bhfs = beta.unsqueeze(-1).expand(B, H_heads, T, S).contiguous()
        else:
            assert beta.shape == (B, H_heads, T, S), f"beta shape {beta.shape}"
            beta_bhfs = beta.contiguous()
        decay = decay.contiguous()

        q_cam_trans = q_cam_trans.contiguous()
        k_cam_trans = k_cam_trans.contiguous()
        v_cam_trans = v_cam_trans.contiguous()

        # ============================================================
        #  FORWARD BRANCH (fused via cp_fused_cam_gdn_num_autograd).
        # ============================================================
        out_fwd, _ = cp_fused_cam_gdn_num_autograd(
            q_cam_trans,
            k_cam_trans,
            v_cam_trans,
            beta_bhfs,
            decay,
            F=T,
            S=S,
            group=cp_group,
            reverse_rank_order=False,
            truncate_to_active=None,
        )  # (B, H, D, N_local)

        # ============================================================
        #  BACKWARD BRANCH (eager, mirrors sana_gdn_camctrl_blocks.py:1414-1462)
        # ============================================================
        # Reshape (B, H, D, N) -> frame layout (B, H, T, D, S) so we can
        # reuse _build_transition_matrices.
        BH = B * H_heads

        def _to_frame(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, H_heads, D_head, T, S).permute(0, 1, 3, 2, 4).contiguous()

        q_rot_f = _to_frame(q_cam_trans)
        k_rot_f = _to_frame(k_cam_trans)
        v_f = _to_frame(v_cam_trans)
        beta_f = beta_bhfs.unsqueeze(3)  # (B, H, T, 1, S)
        decay_f = decay.view(B, H_heads, T, 1, 1)
        I = torch.eye(D_head, device=x.device, dtype=q_rot_f.dtype).reshape(1, 1, 1, D_head, D_head)

        # Distributed flip+shift across CP ranks (no chunk-boundary
        # masking; bidi cam is full-sequence).
        cp_world = dist.get_world_size(cp_group)
        cp_rank_local = dist.get_rank(cp_group)

        def _cp_flip_and_shift(tensors: list[torch.Tensor], shift_vals: list[float]) -> list[torch.Tensor]:
            is_last = cp_rank_local == cp_world - 1
            results = []
            for tensor, sv in zip(tensors, shift_vals):
                first_frame = tensor[:, :, :1, ...].contiguous()
                haloed = cp_halo_exchange(first_frame, left_size=0, right_size=1, dim=2, group=cp_group)
                boundary = haloed[:, :, 1:2, ...]
                if is_last and sv != 0.0:
                    boundary = boundary.mul(0.0).add(sv)
                T_loc = tensor.shape[2]
                flipped = torch.flip(tensor, dims=[2])
                body = flipped[:, :, : T_loc - 1, ...]
                results.append(torch.cat([boundary, body], dim=2))
            return results

        q_rot_bwd_f = torch.flip(q_rot_f, dims=[2])
        k_rot_bwd_f, v_bwd_f, beta_bwd_f, decay_bwd_f = _cp_flip_and_shift(
            [k_rot_f, v_f, beta_f, decay_f],
            [0.0, 0.0, 0.0, 1.0],
        )

        # Camera single-path passes k_rot in BOTH k_f and k_rot_f slots
        # (mirrors sana_gdn_camctrl_blocks.py:1442-1452 -- single-path
        # uses rotated keys only).
        W_kv_bwd, U_kv_bwd, W_z_bwd, U_z_bwd = _build_transition_matrices(
            k_rot_bwd_f,
            v_bwd_f,
            k_rot_bwd_f,
            beta_bwd_f,
            decay_bwd_f,
            I,
            BH,
            T,
            D_head,
        )
        # Zero Z component -- single-path numerator-only has no
        # denominator. Matches sana_gdn_camctrl_blocks.py:1453-1454.
        W_z_bwd = torch.zeros_like(W_z_bwd)
        U_z_bwd = torch.zeros_like(U_z_bwd)

        S_kv_bwd, _ = cp_frame_gdn_scan(W_kv_bwd, U_kv_bwd, W_z_bwd, U_z_bwd, cp_group, reverse=True)
        S_kv_bwd = S_kv_bwd.view(B, H_heads, T, D_head, D_head)
        # Num-only output projection: out = S_kv_bwd @ q_rot_bwd_f.
        out_bwd_flipped_5d = torch.matmul(S_kv_bwd, q_rot_bwd_f)  # (B, H, T, D, S)
        out_bwd_5d = torch.flip(out_bwd_flipped_5d, dims=[2])  # back to original frame order
        # (B, H, T, D, S) -> (B, H, D, T, S) -> (B, H, D, N).
        out_bwd = out_bwd_5d.permute(0, 1, 3, 2, 4).reshape(B, H_heads, D_head, N).contiguous()

        # ============================================================
        #  COMBINE: out = out_fwd + out_bwd (num-only, no divide).
        # ============================================================
        out = out_fwd + out_bwd

        # ---- 9. Cast back, then inverse UCPE + projection. ----
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
