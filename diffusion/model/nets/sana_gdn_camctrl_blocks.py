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

"""GDN-based UCPE camera-control attention blocks.

Each block follows a dual-branch design:
  - **Main branch**: Inherited from the corresponding GDN variant
    (GDN / BidirectionalGDN / ChunkCausalGDN).  ``super().forward()``
    is called with ``apply_output_gate=False`` to get raw attention.
  - **Camera branch**: Separate QKV projections with UCPE per-ray
    transforms.  Camera QK normalization uses branch-specific RMSNorm
    copies (initialized from main branch) to avoid cross-branch
    distribution coupling.

The two raw outputs are combined, then the shared output gate and
projection are applied once.  At init the camera branch contributes
zero (``out_proj_cam`` is zero-initialized), so the model starts
identical to the base GDN.

When the GDN kernels (torch / triton) are upgraded, both branches
pick up the improvement automatically.
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import ShortConvolution

from diffusion.utils.chunk_utils import is_chunk_causal_request, normalize_chunk_index

from .sana_camctrl_blocks import _maybe_drop_cam_branch, prepare_prope_fns
from .sana_gdn_blocks import (
    GDN,
    BidirectionalGDN,
    _forward_softmax_attn,
    flip_and_shift,
)

# ---------------------------------------------------------------------------
# Softmax-block KV cache helpers.
#
# Project Q/K/V for a softmax-attention block, apply RoPE (main branch) or
# UCPE per-position transforms (cam branch), and return the post-transform
# tensors without running SDPA. The AR KV-cache uses these to stash K and V
# in a per-block cache and replay them across AR sub-steps.
# ---------------------------------------------------------------------------


def _prepare_softmax_main_qkv_post_rope(
    block: GDN,
    x: torch.Tensor,
    HW: tuple[int, int, int],
    rotary_emb: torch.Tensor | None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    """Project Q/K/V for the softmax main branch, apply norm and RoPE.

    Returns post-norm, post-RoPE, post-bf16 cast tensors without running
    SDPA, so the caller can either run SDPA itself or stash K/V in a cache.

    Args:
        block: A :class:`GDN` (or subclass) that owns the softmax-attn
            params (``qkv``, ``q_norm``, ``k_norm``).
        x: Input tokens of shape ``(B, N, C)``.
        HW: ``(T, H, W)`` token layout.
        rotary_emb: Optional RoPE table; ``None`` skips RoPE.

    Returns:
        ``(q, k, v, dtype_orig)`` where Q/K/V are shape ``(B, H, N, D)``
        and ``dtype_orig`` is the original ``x.dtype``.
    """
    B, N, C = x.shape
    T, H_sp, W_sp = HW
    S = H_sp * W_sp

    frame_valid_mask = kwargs.get("frame_valid_mask", None)
    token_valid_mask, _, _ = GDN._prepare_frame_valid_masks(
        frame_valid_mask,
        B=B,
        T=T,
        S=S,
        device=x.device,
        dtype=x.dtype,
    )
    if token_valid_mask is not None:
        x = x * token_valid_mask.view(B, N, 1)

    qkv = block.qkv(x).reshape(B, N, 3, block.heads, block.dim)
    q, k, v = qkv.unbind(2)
    if token_valid_mask is not None:
        m = token_valid_mask.view(B, N, 1, 1)
        q, k, v = q * m, k * m, v * m

    q = block.q_norm(q.reshape(B, N, C)).reshape(B, N, block.heads, block.dim)
    k = block.k_norm(k.reshape(B, N, C)).reshape(B, N, block.heads, block.dim)

    if rotary_emb is not None:
        q_perm = q.permute(0, 2, 3, 1)
        k_perm = k.permute(0, 2, 3, 1)
        q_perm = GDN._apply_rotary_emb(q_perm, rotary_emb)
        k_perm = GDN._apply_rotary_emb(k_perm, rotary_emb)
        q = q_perm.permute(0, 3, 1, 2)
        k = k_perm.permute(0, 3, 1, 2)

    if token_valid_mask is not None:
        m = token_valid_mask.view(B, N, 1, 1)
        q, k, v = q * m, k * m, v * m

    q = q.transpose(1, 2)  # (B, H, N, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    dtype_orig = x.dtype
    if q.dtype == torch.float32:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    return q, k, v, dtype_orig


def _sdpa_unmasked_with_pad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Run ``F.scaled_dot_product_attention(q, k, v)`` with FA-friendly head_dim padding.

    FlashAttention-2 only supports head_dim in {32, 64, 128, 256}.
    Other head_dims (e.g. 112) fall back to the math backend. We pad
    head_dim up to the next supported size, run SDPA, then slice back
    to the original head_dim. Mirrors the no-mask path in
    :func:`_forward_softmax_attn` (lines ~3034-3061).

    Args:
        q, k, v: ``(B, H, N_q, D)``, ``(B, H, N_kv, D)``, ``(B, H, N_kv, D)``.

    Returns:
        ``(B, H, N_q, D)`` attention output.
    """
    D = q.shape[-1]
    _need_pad = D not in (32, 64, 128, 256) and D < 256
    if _need_pad:
        _pad_to = 128 if D <= 128 else 256
        _pad_size = _pad_to - D
        q = F.pad(q, (0, _pad_size))
        k = F.pad(k, (0, _pad_size))
        v = F.pad(v, (0, _pad_size))
    out = F.scaled_dot_product_attention(q, k, v)
    if _need_pad:
        out = out[..., :D]
    return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def torch_recurrent_cam_single_path_delta_rule(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """Numerator-only delta-rule recurrence for experimental camera ablations."""
    B, H, D, N = q_rot.shape
    T = beta.shape[2]
    S = N // T

    def to_frame_seq(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_rot_f = to_frame_seq(q_rot)
    k_rot_f = to_frame_seq(k_rot)
    v_f = to_frame_seq(v)

    if beta.ndim == 4:
        beta = beta.unsqueeze(3)
    else:
        beta = beta.view(B, H, T, 1, 1)
    decay = decay.view(B, H, T, 1, 1)

    state_kv = torch.zeros(B, H, D, D, device=q_rot.device, dtype=q_rot.dtype)
    out_list: list[torch.Tensor] = []
    for t in range(T):
        qrt = q_rot_f[:, :, t]
        krt = k_rot_f[:, :, t]
        vt = v_f[:, :, t]
        bt = beta[:, :, t]
        gt = decay[:, :, t]

        state_kv = state_kv * gt
        v_pred = torch.matmul(state_kv, krt)
        delta_v = (vt - v_pred) * bt
        state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))
        out_list.append(torch.matmul(state_kv, qrt))

    out = torch.stack(out_list, dim=2)
    return out.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)


@torch.compile(dynamic=True, disable=os.environ.get("GDN_DISABLE_COMPILE", "0") not in ("0", "false"))
def torch_chunk_cam_single_path_delta_rule(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    chunk_size: int | None = 21,
) -> torch.Tensor:
    """Parallel chunk-scan version of the single-path delta-rule recurrence.

    Algebraically equivalent to ``torch_recurrent_cam_single_path_delta_rule``
    but restructured as a linear recurrence in D x D state space so that
    Phases 1 (transition-matrix construction) and 3 (output projection) are
    fully parallel over T, while Phase 2 (the D x D state scan) is chunked
    and benefits from ``@torch.compile``.

    The recurrence:
        state[t] = state[t-1] * g[t] + delta_v[t] @ k_rot[t]^T
    where delta_v[t] = (v[t] - state[t-1]*g[t] @ k_rot[t]) * beta[t]

    is equivalent to:
        state[t] = state[t-1] @ W[t] + U[t]
    with:
        W[t] = g[t] * (I - beta[t] * k_rot[t] @ k_rot[t]^T)
        U[t] = beta[t] * v[t] @ k_rot[t]^T
    """
    B, H, D, N = q_rot.shape
    if beta.ndim not in (3, 4):
        raise ValueError(f"Expected beta.ndim in (3, 4), got {beta.ndim}.")
    T = beta.shape[2]
    if T <= 0:
        raise ValueError(f"Expected T > 0, got T={T}.")
    if N % T != 0:
        raise ValueError(f"Expected N divisible by T, got N={N}, T={T}.")
    S = N // T

    def to_frame_seq(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_rot = to_frame_seq(q_rot)
    k_rot = to_frame_seq(k_rot)
    v = to_frame_seq(v)

    if beta.ndim == 4:
        beta = beta.unsqueeze(3)
    else:
        beta = beta.view(B, H, T, 1, 1)
    decay = decay.view(B, H, T, 1, 1)

    # =========================================================================
    # Phase 1: PARALLEL PRE-PROCESSING  (fully parallel over T)
    # =========================================================================
    I = torch.eye(D, device=q_rot.device, dtype=q_rot.dtype).view(1, 1, 1, D, D)

    k_rot_beta = k_rot * beta
    W_kv = decay * (I - torch.matmul(k_rot_beta, k_rot.transpose(-1, -2)))
    U_kv = torch.matmul(v * beta, k_rot.transpose(-1, -2))

    # =========================================================================
    # Phase 2: CHUNKED SCAN over D x D state space
    # =========================================================================
    valid_chunk_index, _ = normalize_chunk_index(None, T, chunk_size)
    split_sizes = [valid_chunk_index[i + 1] - valid_chunk_index[i] for i in range(len(valid_chunk_index) - 1)]

    W_kv_c = W_kv.split(split_sizes, dim=2)
    U_kv_c = U_kv.split(split_sizes, dim=2)

    S_kv = torch.zeros(B, H, D, D, device=q_rot.device, dtype=q_rot.dtype)
    out_S_kv: list[torch.Tensor] = []

    def _chunk_scan_kv(w_kv: torch.Tensor, u_kv: torch.Tensor, s_kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_len = w_kv.shape[2]
        s_kv_list: list[torch.Tensor] = []
        for t in range(c_len):
            s_kv = torch.matmul(s_kv, w_kv[:, :, t]) + u_kv[:, :, t]
            s_kv_list.append(s_kv)
        return torch.stack(s_kv_list, dim=2), s_kv

    for i in range(len(split_sizes)):
        s_kv_all, S_kv = _chunk_scan_kv(W_kv_c[i], U_kv_c[i], S_kv)
        out_S_kv.append(s_kv_all)

    S_kv_all = torch.cat(out_S_kv, dim=2)

    # =========================================================================
    # Phase 3: PARALLEL OUTPUT PROJECTION  (no denominator)
    # =========================================================================
    out = torch.matmul(S_kv_all, q_rot)  # (B, H, T, D, S)

    return out.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)


class _GDNUCPEBase(GDN):
    """Shared camera-branch logic for all GDN + UCPE variants.

    Adds a second attention branch whose positional encoding comes from
    UCPE per-ray camera transforms instead of the standard RoPE used by
    the main branch.

    **Camera-specific parameters** (4 Linear layers per block):
        ``q_proj_cam``, ``k_proj_cam``, ``v_proj_cam``, ``out_proj_cam``

    **Shared with main branch** (no duplication):
        QK norms, GDN gates (beta/gate/dt_bias/A_log/recall_gate),
        output gate, output projection.

    Requires ``cam_dim == in_dim`` and ``cam_heads == heads`` so that
    all shared parameters have matching dimensions.

    Subclasses only need to override ``_forward_cam_branch`` when the
    camera branch requires a different recurrence pattern (e.g.
    bidirectional or chunk-causal).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        cam_dim: int,
        cam_heads: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        **kwargs: object,
    ) -> None:
        cam_debug_ratios = bool(kwargs.pop("cam_debug_ratios", False))
        cam_debug_log_per_block = bool(kwargs.pop("cam_debug_log_per_block", False))
        cam_update_rule_func: str = str(kwargs.pop("cam_update_rule_func", "torch_chunk"))
        super().__init__(in_dim, out_dim, **kwargs)

        self.patch_size = patch_size
        self.cam_dim = cam_dim
        self.cam_heads = cam_heads
        self.cam_head_dim = cam_dim // cam_heads
        self.cam_debug_ratios = cam_debug_ratios
        self.cam_debug_log_per_block = cam_debug_log_per_block
        self._cam_debug_stats: dict[str, float] = {}
        self._cam_debug_step_counter: int = 0
        self._cam_debug_log_interval: int = 50

        from functools import partial

        chunk_gdn_chunk_size = kwargs.get("chunk_gdn_chunk_size", 21)
        if cam_update_rule_func == "torch_recurrent":
            self._cam_single_path_fn = torch_recurrent_cam_single_path_delta_rule
        elif cam_update_rule_func == "torch_chunk":
            self._cam_single_path_fn = partial(
                torch_chunk_cam_single_path_delta_rule,
                chunk_size=chunk_gdn_chunk_size,
            )
        else:
            raise ValueError(f"Unsupported cam_update_rule_func: {cam_update_rule_func}")

        if cam_dim != in_dim:
            raise ValueError(
                f"Parameter sharing requires cam_dim == in_dim, " f"got cam_dim={cam_dim}, in_dim={in_dim}."
            )
        if cam_heads != self.heads:
            raise ValueError(
                f"Parameter sharing requires cam_heads == heads, " f"got cam_heads={cam_heads}, heads={self.heads}."
            )
        if self.cam_head_dim % 4 != 0:
            raise ValueError(
                "UCPE camera branch requires cam_head_dim divisible by 4, "
                f"got {self.cam_head_dim} (cam_dim={cam_dim}, cam_heads={cam_heads})."
            )

        # ---- Camera-specific: QKV + output projections only ----
        self.q_proj_cam = nn.Linear(in_dim, cam_dim, bias=True)
        self.k_proj_cam = nn.Linear(in_dim, cam_dim, bias=True)
        self.v_proj_cam = nn.Linear(in_dim, cam_dim, bias=True)
        self.out_proj_cam = nn.Linear(cam_dim, out_dim, bias=True)

        # Keep branch-specific Q/K norms so camera statistics do not disturb the
        # main branch (and vice versa). Start from identical weights.
        self.q_norm_cam = deepcopy(self.q_norm)
        self.k_norm_cam = deepcopy(self.k_norm)

        nn.init.constant_(self.out_proj_cam.weight, 0)
        nn.init.constant_(self.out_proj_cam.bias, 0)

        # Short convolutions for camera branch (matching base GDN variant).
        if self.conv_kernel_size > 0:
            self.conv_k_cam = ShortConvolution(
                hidden_size=cam_dim,
                kernel_size=self.conv_kernel_size,
                activation=None,
            )
            if self.k_conv_only:
                self.conv_q_cam = None
                self.conv_v_cam = None
            else:
                self.conv_q_cam = ShortConvolution(
                    hidden_size=cam_dim,
                    kernel_size=self.conv_kernel_size,
                    activation=None,
                )
                self.conv_v_cam = ShortConvolution(
                    hidden_size=cam_dim,
                    kernel_size=self.conv_kernel_size,
                    activation=None,
                )
            self._init_cam_short_conv_for_linear_equiv()
        else:
            self.conv_q_cam = None
            self.conv_k_cam = None
            self.conv_v_cam = None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_cam_short_conv_for_linear_equiv(self) -> None:
        """Initialize camera short convs as identity to match base at step 0."""
        if self.conv_k_cam is None:
            return
        for conv in (self.conv_q_cam, self.conv_k_cam, self.conv_v_cam):
            if conv is None:
                continue
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, 0, -1] = 1.0
                if getattr(conv, "bias", None) is not None:
                    conv.bias.zero_()

    def init_cam_branch_weights(self) -> None:
        """Copy main-branch QKV weights into the camera branch for transfer learning."""
        if self.cam_dim != self.dim * self.heads:
            print(
                f"Warning: Skipping init_cam_branch_weights because "
                f"cam_dim ({self.cam_dim}) != dim ({self.dim}) * heads ({self.heads})"
            )
            return

        print(f"Initializing camera branch QKV from base model QKV for {self.__class__.__name__}")
        w = self.qkv.weight
        b = self.qkv.bias
        dim = self.cam_dim

        self.q_proj_cam.weight.data.copy_(w[:dim])
        self.k_proj_cam.weight.data.copy_(w[dim : 2 * dim])
        self.v_proj_cam.weight.data.copy_(w[2 * dim :])
        if b is not None:
            self.q_proj_cam.bias.data.copy_(b[:dim])
            self.k_proj_cam.bias.data.copy_(b[dim : 2 * dim])
            self.v_proj_cam.bias.data.copy_(b[2 * dim :])

        # Mirror main-branch Q/K norm initialization into camera-specific norms.
        if hasattr(self.q_norm, "state_dict") and hasattr(self.q_norm_cam, "load_state_dict"):
            self.q_norm_cam.load_state_dict(self.q_norm.state_dict(), strict=False)
        if hasattr(self.k_norm, "state_dict") and hasattr(self.k_norm_cam, "load_state_dict"):
            self.k_norm_cam.load_state_dict(self.k_norm.state_dict(), strict=False)

        # Copy short conv weights from base to camera branch.
        if self.conv_k_cam is not None and self.conv_k is not None:
            self.conv_k_cam.load_state_dict(self.conv_k.state_dict())
        if self.conv_q_cam is not None and self.conv_q is not None:
            self.conv_q_cam.load_state_dict(self.conv_q.state_dict())
        if self.conv_v_cam is not None and self.conv_v is not None:
            self.conv_v_cam.load_state_dict(self.conv_v.state_dict())

    @staticmethod
    def _downscale_to_reference_rms(
        ref: torch.Tensor,
        transformed: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Downscale transformed tensor if its channel RMS exceeds reference.

        Args:
            ref: Reference tensor with target magnitude, shape (B, H, D, N).
            transformed: Tensor to stabilize, shape (B, H, D, N).
            eps: Numerical epsilon for RMS.

        Returns:
            Stabilized tensor with per-(B,H,N) channel RMS not larger than ref.
        """
        ref_rms = ref.square().mean(dim=2, keepdim=True).add(eps).sqrt()
        tr_rms = transformed.square().mean(dim=2, keepdim=True).add(eps).sqrt()
        scale = (ref_rms / tr_rms.clamp_min(eps)).clamp(max=1.0)
        return transformed * scale

    def reset_cam_debug_stats(self) -> None:
        """Clear debug-only camera branch ratio summaries."""
        self._cam_debug_stats = {}

    def pop_cam_debug_stats(self) -> dict[str, float]:
        """Return and clear debug-only camera branch ratio summaries."""
        stats = dict(self._cam_debug_stats)
        self._cam_debug_stats = {}
        return stats

    def _record_cam_debug_stat(self, name: str, value: float) -> None:
        """Store one debug scalar when camera ratio logging is enabled."""
        if not self.cam_debug_ratios:
            return
        self._cam_debug_stats[name] = float(value)

    @staticmethod
    def _compute_cam_ratio_summary(
        ref: torch.Tensor,
        transformed: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> tuple[float, float]:
        """Compute mean/max channel-norm amplification ratios."""
        ref_norm = torch.linalg.vector_norm(ref.float(), dim=2).clamp_min(eps)
        transformed_norm = torch.linalg.vector_norm(transformed.float(), dim=2)
        ratio = (transformed_norm / ref_norm).detach()
        if token_valid_mask is not None:
            valid = token_valid_mask.to(torch.bool).unsqueeze(1).expand_as(ratio)
            ratio = ratio.masked_select(valid)
            if ratio.numel() == 0:
                return 0.0, 0.0
        return float(ratio.mean().item()), float(ratio.max().item())

    @staticmethod
    def _compute_cam_norm_summary(
        tensor: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """Compute mean/max channel norms for debug-only logging."""
        norms = torch.linalg.vector_norm(tensor.float(), dim=2).detach()
        if token_valid_mask is not None:
            valid = token_valid_mask.to(torch.bool).unsqueeze(1).expand_as(norms)
            norms = norms.masked_select(valid)
            if norms.numel() == 0:
                return 0.0, 0.0
        return float(norms.mean().item()), float(norms.max().item())

    def _record_cam_inflation_stats(
        self,
        prefix: str,
        k_cam: torch.Tensor,
        k_cam_trans: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Record squared key inflation statistics for one transform stage."""
        k_ratio_sq = (
            (
                torch.linalg.vector_norm(k_cam_trans.float(), dim=2).clamp_min(1e-6)
                / torch.linalg.vector_norm(k_cam.float(), dim=2).clamp_min(1e-6)
            )
            .pow(2)
            .detach()
        )
        if token_valid_mask is not None:
            valid = token_valid_mask.to(torch.bool).unsqueeze(1).expand_as(k_ratio_sq)
            k_ratio_sq = k_ratio_sq.masked_select(valid)
            if k_ratio_sq.numel() == 0:
                self._record_cam_debug_stat(f"{prefix}_inflation_sq_mean", 0.0)
                self._record_cam_debug_stat(f"{prefix}_inflation_sq_max", 0.0)
                return
        self._record_cam_debug_stat(f"{prefix}_inflation_sq_mean", float(k_ratio_sq.mean().item()))
        self._record_cam_debug_stat(f"{prefix}_inflation_sq_max", float(k_ratio_sq.max().item()))

    def _should_log_cam_debug(self) -> bool:
        """Check whether cam debug stats should be recorded this step."""
        if not self.cam_debug_ratios:
            return False
        return self._cam_debug_step_counter % self._cam_debug_log_interval == 0

    def _record_cam_transform_stats(
        self,
        stage_prefix: str,
        q_cam: torch.Tensor,
        k_cam: torch.Tensor,
        v_cam: torch.Tensor,
        q_cam_trans: torch.Tensor,
        k_cam_trans: torch.Tensor,
        v_cam_trans: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Record debug-only camera transform ratios for one transform stage."""
        if not self._should_log_cam_debug():
            return

        for tensor_prefix, ref, transformed in (
            ("q_cam", q_cam, q_cam_trans),
            ("k_cam", k_cam, k_cam_trans),
            ("v_cam", v_cam, v_cam_trans),
        ):
            ratio_mean, ratio_max = self._compute_cam_ratio_summary(
                ref,
                transformed,
                token_valid_mask=token_valid_mask,
            )
            self._record_cam_debug_stat(f"{stage_prefix}_{tensor_prefix}_ratio_mean", ratio_mean)
            self._record_cam_debug_stat(f"{stage_prefix}_{tensor_prefix}_ratio_max", ratio_max)

        self._record_cam_inflation_stats(
            stage_prefix,
            k_cam,
            k_cam_trans,
            token_valid_mask=token_valid_mask,
        )

    def _maybe_record_cam_output_stats(
        self,
        pre_output_transform: torch.Tensor,
        post_output_transform: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Record inverse-UCPE output transform amplification ratios."""
        if not self._should_log_cam_debug():
            return

        ratio_mean, ratio_max = self._compute_cam_ratio_summary(
            pre_output_transform,
            post_output_transform,
            token_valid_mask=token_valid_mask,
        )
        self._record_cam_debug_stat("o_cam_ratio_mean", ratio_mean)
        self._record_cam_debug_stat("o_cam_ratio_max", ratio_max)
        pre_norm_mean, pre_norm_max = self._compute_cam_norm_summary(
            pre_output_transform,
            token_valid_mask=token_valid_mask,
        )
        post_norm_mean, post_norm_max = self._compute_cam_norm_summary(
            post_output_transform,
            token_valid_mask=token_valid_mask,
        )
        self._record_cam_debug_stat("o_cam_pre_norm_mean", pre_norm_mean)
        self._record_cam_debug_stat("o_cam_pre_norm_max", pre_norm_max)
        self._record_cam_debug_stat("o_cam_post_norm_mean", post_norm_mean)
        self._record_cam_debug_stat("o_cam_post_norm_max", post_norm_max)

    def _stabilize_cam_transforms(
        self,
        q_cam: torch.Tensor,
        k_cam: torch.Tensor,
        v_cam: torch.Tensor,
        q_cam_trans: torch.Tensor,
        k_cam_trans: torch.Tensor,
        v_cam_trans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optional post-UCPE stabilization hook for experimental variants."""
        del q_cam, k_cam, v_cam
        return q_cam_trans, k_cam_trans, v_cam_trans

    # ------------------------------------------------------------------
    # Camera-branch building blocks
    # ------------------------------------------------------------------

    def _prepare_cam_qkv(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        *,
        token_valid_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple:
        """Project camera QKV, apply short conv + QK norm + kernel + scaling + UCPE.

        The processing order mirrors the base GDN branch:
          project -> mask -> short_conv -> QK_norm -> kernel -> scale -> permute -> UCPE

        Args:
            token_valid_mask: Pre-computed mask of shape ``(B, N)`` from the
                caller.  Avoids redundant ``_prepare_frame_valid_masks`` calls.

        Returns:
            (q_cam, k_cam, v_cam_trans, q_cam_trans, k_cam_trans, apply_fn_o, inflation_sq)

        All tensors are shaped ``(B, cam_heads, cam_head_dim, N)``.
        ``apply_fn_o`` is the UCPE inverse-output transform closure.
        ``inflation_sq`` is the energy inflation factor of shape ``(B, cam_heads, 1, N)``.
        """
        B, N, C = x.shape
        T, H, W = HW
        S = H * W

        # Pre-projection token masking (matching base branch).
        if token_valid_mask is not None:
            x = x * token_valid_mask.view(B, N, 1)

        # Fused camera QKV projection (1 GEMM instead of 3 kernel launches).
        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        qkv_cam = F.linear(x, qkv_w, qkv_b)
        q_cam, k_cam, v_cam = qkv_cam.chunk(3, dim=-1)

        # Post-projection token masking (before conv, matching base branch).
        if token_valid_mask is not None:
            token_mask = token_valid_mask.view(B, N, 1)
            q_cam = q_cam * token_mask
            k_cam = k_cam * token_mask
            v_cam = v_cam * token_mask

        # Short convolution along T (before norm / kernel activation).
        if self.conv_q_cam is not None:
            q_cam = self._apply_temporal_short_conv(q_cam, self.conv_q_cam, HW, **kwargs)
        if self.conv_k_cam is not None:
            k_cam = self._apply_temporal_short_conv(k_cam, self.conv_k_cam, HW, **kwargs)
        if self.conv_v_cam is not None:
            v_cam = self._apply_temporal_short_conv(v_cam, self.conv_v_cam, HW, **kwargs)

        # Camera-specific QK normalization.
        q_cam = self.q_norm_cam(q_cam).reshape(B, N, self.cam_heads, self.cam_head_dim)
        k_cam = self.k_norm_cam(k_cam).reshape(B, N, self.cam_heads, self.cam_head_dim)
        v_cam = v_cam.reshape(B, N, self.cam_heads, self.cam_head_dim)

        # ReLU kernel (shared).
        q_cam = self.kernel_func(q_cam)
        k_cam = self.kernel_func(k_cam)

        # FIXED: K scaling -- explicitly use ** for exponentiation!
        k_scale = (self.cam_head_dim**-0.5) * (S**-0.5)
        k_cam = k_cam * k_scale

        # Permute to (B, H, D, N) for GDN processing.
        q_cam = q_cam.permute(0, 2, 3, 1).contiguous()
        k_cam = k_cam.permute(0, 2, 3, 1).contiguous()
        v_cam = v_cam.permute(0, 2, 3, 1).contiguous()

        # Measure safe geometric norm before UCPE applies translations
        pre_ucpe_k_norm = torch.linalg.vector_norm(k_cam, dim=2, keepdim=True).clamp_min(1e-6)

        # UCPE per-ray transforms — reuse model-level cache when available
        # to avoid recomputing _process_camera_conditions_ucpe per block.
        cached_fns = kwargs.get("prope_fns", None)
        if cached_fns is not None:
            apply_fn_q, apply_fn_kv, apply_fn_o = cached_fns
        else:
            apply_fn_q, apply_fn_kv, apply_fn_o = prepare_prope_fns(
                camctrl_type="UCPE",
                head_dim=self.cam_head_dim,
                camera_conditions=camera_conditions,
                HW=HW,
                patch_size=self.patch_size,
                rotary_emb=rotary_emb,
            )

        # UCPE expects (B, h, N, d); our tensors are (B, h, d, N).
        # Avoid eager contiguous copies before transforms, and fuse K/V transform
        # into one call (same apply_fn_kv), then split back.
        q_cam_trans = apply_fn_q(q_cam.transpose(-1, -2)).transpose(-1, -2).contiguous()
        kv_cam = torch.cat([k_cam, v_cam], dim=1)
        kv_cam_trans = apply_fn_kv(kv_cam.transpose(-1, -2)).transpose(-1, -2).contiguous()
        k_cam_trans, v_cam_trans = torch.chunk(kv_cam_trans, chunks=2, dim=1)

        self._record_cam_transform_stats(
            stage_prefix="raw",
            q_cam=q_cam,
            k_cam=k_cam,
            v_cam=v_cam,
            q_cam_trans=q_cam_trans,
            k_cam_trans=k_cam_trans,
            v_cam_trans=v_cam_trans,
            token_valid_mask=token_valid_mask,
        )
        q_cam_trans, k_cam_trans, v_cam_trans = self._stabilize_cam_transforms(
            q_cam=q_cam,
            k_cam=k_cam,
            v_cam=v_cam,
            q_cam_trans=q_cam_trans,
            k_cam_trans=k_cam_trans,
            v_cam_trans=v_cam_trans,
        )
        self._record_cam_transform_stats(
            stage_prefix="post_stab",
            q_cam=q_cam,
            k_cam=k_cam,
            v_cam=v_cam,
            q_cam_trans=q_cam_trans,
            k_cam_trans=k_cam_trans,
            v_cam_trans=v_cam_trans,
            token_valid_mask=token_valid_mask,
        )

        # Measure inflated geometric norm after UCPE
        post_ucpe_k_norm = torch.linalg.vector_norm(k_cam_trans, dim=2, keepdim=True).clamp_min(1e-6)

        # Calculate the squared inflation factor for beta discounting
        inflation_sq = (post_ucpe_k_norm / pre_ucpe_k_norm) ** 2

        return q_cam, k_cam, v_cam_trans, q_cam_trans, k_cam_trans, apply_fn_o, inflation_sq

    def _run_cam_gdn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        """Run the shared GDN kernel on camera-branch tensors.

        Uses shared ``self.recall_gate``.  Handles FP32 casting.
        Returns ``num / (den + eps)`` shaped ``(B, H, D, N)``.
        """
        recall_gate = self.recall_gate
        if getattr(self, "fp32_attention", True):
            q = q.float()
            k = k.float()
            v = v.float()
            q_rot = q_rot.float()
            k_rot = k_rot.float()
            beta = beta.float()
            decay = decay.float()
            recall_gate = recall_gate.float()

        return self.update_rule_func(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            recall_gate=recall_gate,
            eps=self.eps,
        )

    def _run_cam_gdn_components(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like ``_run_cam_gdn`` but returns ``(num, den)`` components."""
        recall_gate = self.recall_gate
        if getattr(self, "fp32_attention", True):
            q = q.float()
            k = k.float()
            v = v.float()
            q_rot = q_rot.float()
            k_rot = k_rot.float()
            beta = beta.float()
            decay = decay.float()
            recall_gate = recall_gate.float()

        return self.update_rule_func(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            recall_gate=recall_gate,
            eps=self.eps,
            return_components=True,
        )

    def _run_cam_single_path(
        self,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        """Run the numerator-only camera delta-rule recurrence.

        Dispatches to either the recurrent reference or the parallel chunk
        scan depending on ``cam_update_rule_func`` set at init time.
        """
        if getattr(self, "fp32_attention", True):
            q_rot = q_rot.float()
            k_rot = k_rot.float()
            v = v.float()
            beta = beta.float()
            decay = decay.float()
        return self._cam_single_path_fn(q_rot, k_rot, v, beta, decay)

    # ------------------------------------------------------------------
    # Camera-branch forward (forward-only causal -- default)
    # ------------------------------------------------------------------

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward-only causal GDN camera branch with UCPE transforms.

        Subclasses override this for bidirectional / chunk-causal variants.

        Returns raw attention output ``(B, N, C)`` -- no output gate or
        projection applied (those are shared and applied in ``forward()``).
        """
        B, N, _ = x.shape
        T, H, W = HW
        S = H * W
        dtype_orig = x.dtype

        # Compute masks once; pass token_valid_mask to _prepare_cam_qkv for
        # pre-conv masking and reuse here for post-UCPE masking + gate masking.
        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            kwargs.get("frame_valid_mask", None),
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )

        q_cam, k_cam, v_cam_trans, q_cam_trans, k_cam_trans, apply_fn_o, inflation_sq = self._prepare_cam_qkv(
            x,
            HW,
            camera_conditions,
            rotary_emb,
            token_valid_mask=token_valid_mask,
            **kwargs,
        )

        # Re-mask after UCPE transforms (which can reintroduce non-zero values).
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_cam = q_cam * token_mask_qkv
            k_cam = k_cam * token_mask_qkv
            v_cam_trans = v_cam_trans * token_mask_qkv
            q_cam_trans = q_cam_trans * token_mask_qkv
            k_cam_trans = k_cam_trans * token_mask_qkv

        # Shared GDN gates (use pre-computed when available).
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        # Dynamic Beta Discounting: scale beta by UCPE inflation factor.
        inflation_sq_spatial = inflation_sq.view(B, self.cam_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        out = self._run_cam_gdn(
            q_cam,
            k_cam,
            v_cam_trans,
            q_cam_trans,
            k_cam_trans,
            beta,
            decay,
        )

        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)

        # Inverse UCPE transform on output.
        out_before_apply_fn_o = out
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        self._maybe_record_cam_output_stats(out_before_apply_fn_o, out, token_valid_mask=token_valid_mask)
        out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

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
        """Dual-branch forward: GDN main + UCPE camera.

        Flow:
            1. main_raw = GDN attention (no gate/proj)
            2. cam_raw  = GDN+UCPE attention (no gate/proj)
            3. combined = main_raw + out_proj_cam(cam_raw)   [zero at init]
            4. output   = proj(output_gate(combined))        [shared, once]
        """
        if self.cam_debug_ratios:
            self.reset_cam_debug_stats()
        if self.training:
            self._cam_debug_step_counter += 1

        # Pre-compute shared gates once for both branches.
        if HW is not None:
            precomputed_gates = self._compute_frame_gates(x, HW)
        else:
            precomputed_gates = None

        # Main branch -- raw attention without gate/proj.
        main_raw = super().forward(
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

        # Camera branch.
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

        # Combine, then shared gate + projection (applied once).
        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        return self.proj(combined.to(self.proj.weight.dtype))


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class BidirectionalGDNUCPELiteLA(_GDNUCPEBase, BidirectionalGDN):
    """Bidirectional GDN with UCPE camera conditioning.

    Main branch: bidirectional GDN (inherited from ``BidirectionalGDN``).
    Camera branch: bidirectional GDN with UCPE transforms.
    """

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        dtype_orig = x.dtype

        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            kwargs.get("frame_valid_mask", None),
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )

        q_cam, k_cam, v_cam_trans, q_cam_trans, k_cam_trans, apply_fn_o, inflation_sq = self._prepare_cam_qkv(
            x,
            HW,
            camera_conditions,
            rotary_emb,
            token_valid_mask=token_valid_mask,
            **kwargs,
        )
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_cam = q_cam * token_mask_qkv
            k_cam = k_cam * token_mask_qkv
            v_cam_trans = v_cam_trans * token_mask_qkv
            q_cam_trans = q_cam_trans * token_mask_qkv
            k_cam_trans = k_cam_trans * token_mask_qkv

        # Shared GDN gates (use pre-computed when available).
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        # Dynamic Beta Discounting: scale beta by UCPE inflation factor.
        inflation_sq_spatial = inflation_sq.view(B, self.cam_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        H_heads = self.cam_heads
        D_head = self.cam_head_dim

        # -- Forward pass (inclusive 1..t) --
        num_fwd, den_fwd = self._run_cam_gdn_components(
            q_cam,
            k_cam,
            v_cam_trans,
            q_cam_trans,
            k_cam_trans,
            beta,
            decay,
        )

        # -- Backward pass (exclusive t+1..T) --
        def to_time(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, H_heads, D_head, T, S).permute(0, 1, 3, 2, 4)

        def from_time(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 1, 3, 2, 4).reshape(B, H_heads, D_head, N)

        q_T = to_time(q_cam)
        k_T = to_time(k_cam)
        v_T = to_time(v_cam_trans)
        q_rot_T = to_time(q_cam_trans)
        k_rot_T = to_time(k_cam_trans)

        q_bwd = torch.flip(q_T, dims=[2])
        q_rot_bwd = torch.flip(q_rot_T, dims=[2])
        k_bwd = flip_and_shift(k_T, dim=2, shift_val=0.0)
        v_bwd = flip_and_shift(v_T, dim=2, shift_val=0.0)
        k_rot_bwd = flip_and_shift(k_rot_T, dim=2, shift_val=0.0)
        beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        num_bwd_f, den_bwd_f = self._run_cam_gdn_components(
            from_time(q_bwd),
            from_time(k_bwd),
            from_time(v_bwd),
            from_time(q_rot_bwd),
            from_time(k_rot_bwd),
            beta_bwd,
            decay_bwd,
        )

        def flip_back(tensor: torch.Tensor) -> torch.Tensor:
            d = tensor.shape[2]
            return torch.flip(
                tensor.view(B, H_heads, d, T, S),
                dims=[3],
            ).reshape(B, H_heads, d, N)

        num_bwd = flip_back(num_bwd_f)
        den_bwd = flip_back(den_bwd_f)
        out = (num_fwd + num_bwd) / (den_fwd + den_bwd + self.eps)

        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)

        out_before_apply_fn_o = out
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        self._maybe_record_cam_output_stats(out_before_apply_fn_o, out, token_valid_mask=token_valid_mask)
        out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
        return out


class BidirectionalGDNUCPELiteLAPostUCPERenorm(BidirectionalGDNUCPELiteLA):
    """Bidirectional GDNUCPE with post-UCPE RMS downscaling.

    The raw UCPE transforms are still measured for debug logging, but the
    transformed camera tensors are downscaled back to their pre-UCPE RMS
    envelope before they enter the recurrence.
    """

    def _stabilize_cam_transforms(
        self,
        q_cam: torch.Tensor,
        k_cam: torch.Tensor,
        v_cam: torch.Tensor,
        q_cam_trans: torch.Tensor,
        k_cam_trans: torch.Tensor,
        v_cam_trans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_cam_trans = self._downscale_to_reference_rms(q_cam, q_cam_trans)
        k_cam_trans = self._downscale_to_reference_rms(k_cam, k_cam_trans)
        v_cam_trans = self._downscale_to_reference_rms(v_cam, v_cam_trans)
        return q_cam_trans, k_cam_trans, v_cam_trans


class BidirectionalGDNUCPESinglePathLiteLA(BidirectionalGDNUCPELiteLAPostUCPERenorm):
    """Bidirectional UCPE camera branch with numerator-only delta-rule updates.

    This is an experimental ablation that keeps the main branch unchanged,
    applies UCPE plus post-UCPE RMS downscaling on the camera tensors, and
    replaces the camera branch's ``num / den`` recurrence with a single-path
    delta rule over the transformed camera stream only.
    """

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        T, H, W = HW
        S = H * W
        dtype_orig = x.dtype

        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            kwargs.get("frame_valid_mask", None),
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )

        q_cam, _, v_cam_trans, q_cam_trans, k_cam_trans, apply_fn_o, inflation_sq = self._prepare_cam_qkv(
            x,
            HW,
            camera_conditions,
            rotary_emb,
            token_valid_mask=token_valid_mask,
            **kwargs,
        )
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_cam = q_cam * token_mask_qkv
            v_cam_trans = v_cam_trans * token_mask_qkv
            q_cam_trans = q_cam_trans * token_mask_qkv
            k_cam_trans = k_cam_trans * token_mask_qkv

        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        inflation_sq_spatial = inflation_sq.view(B, self.cam_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        H_heads = self.cam_heads
        D_head = self.cam_head_dim
        out_fwd = self._run_cam_single_path(
            q_cam_trans,
            k_cam_trans,
            v_cam_trans,
            beta,
            decay,
        )

        def to_time(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, H_heads, D_head, T, S).permute(0, 1, 3, 2, 4)

        def from_time(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 1, 3, 2, 4).reshape(B, H_heads, D_head, N)

        q_rot_T = to_time(q_cam_trans)
        k_rot_T = to_time(k_cam_trans)
        v_T = to_time(v_cam_trans)

        q_rot_bwd = torch.flip(q_rot_T, dims=[2])
        k_rot_bwd = flip_and_shift(k_rot_T, dim=2, shift_val=0.0)
        v_bwd = flip_and_shift(v_T, dim=2, shift_val=0.0)
        beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        out_bwd_f = self._run_cam_single_path(
            from_time(q_rot_bwd),
            from_time(k_rot_bwd),
            from_time(v_bwd),
            beta_bwd,
            decay_bwd,
        )

        out_bwd = torch.flip(
            out_bwd_f.view(B, H_heads, D_head, T, S),
            dims=[3],
        ).reshape(B, H_heads, D_head, N)
        out = out_fwd + out_bwd

        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)

        out_before_apply_fn_o = out
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        self._maybe_record_cam_output_stats(out_before_apply_fn_o, out, token_valid_mask=token_valid_mask)
        out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
        return out


def _prepare_cam_qkv_softmax(
    self,
    x: torch.Tensor,
    HW: tuple,
    camera_conditions: torch.Tensor,
    rotary_emb: torch.Tensor | None,
    *,
    token_valid_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple:
    """Camera branch Q/K/V for softmax attention.

    Mirrors ``_GDNUCPEBase._prepare_cam_qkv`` but skips the ReLU kernel and
    GDN key scaling — standard softmax SDPA provides its own 1/sqrt(d_k).
    Returns ``(q, k, v, apply_fn_o)`` shaped ``(B, cam_heads, cam_head_dim, N)``.
    """
    B, N, C = x.shape

    if token_valid_mask is not None:
        x = x * token_valid_mask.view(B, N, 1)

    qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
    qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
    qkv_cam = F.linear(x, qkv_w, qkv_b)
    q_cam, k_cam, v_cam = qkv_cam.chunk(3, dim=-1)

    if token_valid_mask is not None:
        m = token_valid_mask.view(B, N, 1)
        q_cam, k_cam, v_cam = q_cam * m, k_cam * m, v_cam * m

    if self.conv_q_cam is not None:
        q_cam = self._apply_temporal_short_conv(q_cam, self.conv_q_cam, HW, **kwargs)
    if self.conv_k_cam is not None:
        k_cam = self._apply_temporal_short_conv(k_cam, self.conv_k_cam, HW, **kwargs)
    if self.conv_v_cam is not None:
        v_cam = self._apply_temporal_short_conv(v_cam, self.conv_v_cam, HW, **kwargs)

    q_cam = self.q_norm_cam(q_cam).reshape(B, N, self.cam_heads, self.cam_head_dim)
    k_cam = self.k_norm_cam(k_cam).reshape(B, N, self.cam_heads, self.cam_head_dim)
    v_cam = v_cam.reshape(B, N, self.cam_heads, self.cam_head_dim)

    q_cam = q_cam.permute(0, 2, 3, 1).contiguous()
    k_cam = k_cam.permute(0, 2, 3, 1).contiguous()
    v_cam = v_cam.permute(0, 2, 3, 1).contiguous()

    cached_fns = kwargs.get("prope_fns", None)
    if cached_fns is not None:
        apply_fn_q, apply_fn_kv, apply_fn_o = cached_fns
    else:
        apply_fn_q, apply_fn_kv, apply_fn_o = prepare_prope_fns(
            camctrl_type="UCPE",
            head_dim=self.cam_head_dim,
            camera_conditions=camera_conditions,
            HW=HW,
            patch_size=self.patch_size,
            rotary_emb=rotary_emb,
        )

    q_cam_trans = apply_fn_q(q_cam.transpose(-1, -2)).transpose(-1, -2).contiguous()
    kv_cam = torch.cat([k_cam, v_cam], dim=1)
    kv_cam_trans = apply_fn_kv(kv_cam.transpose(-1, -2)).transpose(-1, -2).contiguous()
    k_cam_trans, v_cam_trans = torch.chunk(kv_cam_trans, chunks=2, dim=1)

    q_cam_trans, k_cam_trans, v_cam_trans = self._stabilize_cam_transforms(
        q_cam=q_cam,
        k_cam=k_cam,
        v_cam=v_cam,
        q_cam_trans=q_cam_trans,
        k_cam_trans=k_cam_trans,
        v_cam_trans=v_cam_trans,
    )
    return q_cam_trans, k_cam_trans, v_cam_trans, apply_fn_o


def _forward_cam_branch_softmax(
    self,
    x: torch.Tensor,
    HW: tuple,
    camera_conditions: torch.Tensor,
    rotary_emb: torch.Tensor | None,
    frame_causal: bool,
    **kwargs,
) -> torch.Tensor:
    """Bidirectional softmax camera branch (with UCPE transforms).

    Uses ``F.scaled_dot_product_attention`` with optional invalid-key masking.
    """
    B, N, _ = x.shape
    T, H, W = HW
    S = H * W

    token_valid_mask, _, _ = self._prepare_frame_valid_masks(
        kwargs.get("frame_valid_mask", None),
        B=B,
        T=T,
        S=S,
        device=x.device,
        dtype=x.dtype,
    )

    q_cam_trans, k_cam_trans, v_cam_trans, apply_fn_o = _prepare_cam_qkv_softmax(
        self,
        x,
        HW,
        camera_conditions,
        rotary_emb,
        token_valid_mask=token_valid_mask,
        **kwargs,
    )

    if token_valid_mask is not None:
        m = token_valid_mask.view(B, 1, 1, N)
        q_cam_trans, v_cam_trans = q_cam_trans * m, v_cam_trans * m

    q_sdpa = q_cam_trans.transpose(-1, -2)
    k_sdpa = k_cam_trans.transpose(-1, -2)
    v_sdpa = v_cam_trans.transpose(-1, -2)

    dtype_orig = x.dtype
    if getattr(self, "fp32_attention", True):
        q_sdpa, k_sdpa, v_sdpa = q_sdpa.float(), k_sdpa.float(), v_sdpa.float()
    # SDPA / FlashAttention only supports bf16/fp16; fp32 falls back to math backend.
    if q_sdpa.dtype == torch.float32:
        q_sdpa, k_sdpa, v_sdpa = q_sdpa.bfloat16(), k_sdpa.bfloat16(), v_sdpa.bfloat16()

    invalid_kv_logit_bias = None
    if token_valid_mask is not None and not bool(token_valid_mask.all()):
        invalid_kv_logit_bias = torch.where(
            token_valid_mask.bool().view(B, 1, 1, -1),
            torch.zeros((), dtype=q_sdpa.dtype, device=q_sdpa.device),
            torch.full((), -1e9, dtype=q_sdpa.dtype, device=q_sdpa.device),
        )

    # FlashAttention-2 only supports head_dim in {32, 64, 128, 256}.
    D = q_sdpa.shape[-1]
    _need_pad = D not in (32, 64, 128, 256) and D < 256
    if _need_pad:
        _pad_to = 128 if D <= 128 else 256
        _pad_size = _pad_to - D
        q_sdpa = F.pad(q_sdpa, (0, _pad_size))
        k_sdpa = F.pad(k_sdpa, (0, _pad_size))
        v_sdpa = F.pad(v_sdpa, (0, _pad_size))
    out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=invalid_kv_logit_bias)
    if _need_pad:
        out = out[..., :D]

    out = out.transpose(-1, -2)
    if out.dtype != dtype_orig:
        out = out.to(dtype_orig)
    if token_valid_mask is not None:
        out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)
    out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
    out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
    if token_valid_mask is not None:
        out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
    return out


class _SoftmaxUCPESinglePathLiteLA(
    BidirectionalGDNUCPESinglePathLiteLA,
):
    """Softmax attention with UCPE camera conditioning (single-path).

    Replaces GDN recurrence with ``F.scaled_dot_product_attention``.
    Automatically selects the correct masking mode based on ``chunk_size``:

    - ``chunk_size is None`` or ``chunk_size >= T``: full bidirectional (no mask)
    - ``chunk_size < T``: chunk-causal (full within chunks, causal across)

    All parameters match the GDN variants for checkpoint compatibility.
    GDN-specific parameters are present but unused in forward.
    """

    def __init__(self, *args, conv_kernel_size: int = 0, **kwargs):
        super().__init__(*args, conv_kernel_size=0, **kwargs)

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

        main_raw = _forward_softmax_attn(
            self,
            x,
            HW,
            rotary_emb,
            frame_causal=False,
            apply_output_gate=False,
            chunk_size=chunk_size,
            **kwargs,
        )

        cam_contrib: torch.Tensor | int = 0
        camera_conditions = _maybe_drop_cam_branch(
            camera_conditions,
            kwargs.get("cam_branch_drop_prob", 0.0),
            self.training,
            x.device,
        )
        if camera_conditions is not None:
            if HW is None:
                raise ValueError("HW must be provided for UCPE camera branch.")
            cam_raw = _forward_cam_branch_softmax(
                self,
                x,
                HW,
                camera_conditions,
                rotary_emb,
                frame_causal=False,
                chunk_size=chunk_size,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        return self.proj(combined.to(x.dtype))


# Aliases for backward compatibility and clear intent in mappings.
BidirectionalSoftmaxUCPESinglePathLiteLA = _SoftmaxUCPESinglePathLiteLA
ChunkCausalSoftmaxUCPESinglePathLiteLA = _SoftmaxUCPESinglePathLiteLA
