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

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import ShortConvolution

from diffusion.model.registry import ATTENTION_BLOCKS

from .sana_camctrl_blocks import _maybe_drop_cam_branch, prepare_prope_fns
from .sana_gdn_blocks import (
    GDN,
    BidirectionalGDN,
    ChunkCausalGDN,
    _forward_softmax_attn,
    _sdpa_needs_head_pad,
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
    _need_pad = _sdpa_needs_head_pad(D)
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

    Subclasses provide their own ``_forward_cam_branch`` (e.g. the fused
    Triton pipeline in :class:`BidirectionalGDNUCPESinglePathLiteLABothTriton`
    or the cached streaming pipeline in
    :class:`CachedChunkCausalGDNUCPESinglePathLiteLA`).
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
        # Silently swallow legacy debug-stat kwargs so older configs/checkpoints
        # don't blow up on construction.
        kwargs.pop("cam_debug_ratios", None)
        kwargs.pop("cam_debug_log_per_block", None)
        super().__init__(in_dim, out_dim, **kwargs)

        self.patch_size = patch_size
        self.cam_dim = cam_dim
        self.cam_heads = cam_heads
        self.cam_head_dim = cam_dim // cam_heads

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


class BidirectionalGDNUCPESinglePathLiteLA(_GDNUCPEBase, BidirectionalGDN):
    """Bidirectional GDN with UCPE camera conditioning (single-path delta rule).

    Main branch: bidirectional GDN (inherited from :class:`BidirectionalGDN`).
    Camera branch: numerator-only delta-rule recurrence over UCPE-transformed
    camera tensors (RMSNorm + ReLU + UCPE 4x4 + RoPE, then a single-path scan).

    This is the production base for both the bidir Triton variant
    (:class:`BidirectionalGDNUCPESinglePathLiteLABothTriton`) and the streaming
    chunk-causal variant (:class:`ChunkCausalGDNUCPESinglePathLiteLA`).
    """


class ChunkCausalGDNUCPESinglePathLiteLA(BidirectionalGDNUCPESinglePathLiteLA, ChunkCausalGDN):
    """Chunk-causal variant of ``BidirectionalGDNUCPESinglePathLiteLA``.

    Main branch: chunk-causal GDN (inherited via MRO from
    :class:`ChunkCausalGDN`).  Camera branch: single-path (numerator-only)
    delta rule with chunk-boundary isolation in the backward pass.

    All parameter names match ``BidirectionalGDNUCPESinglePathLiteLA``
    exactly, so checkpoints from bidirectional training load directly.
    The only behavioral difference is that the backward recurrence in the
    camera branch is isolated at chunk boundaries (decay and inputs
    zeroed), preventing future chunk information from leaking into past
    chunks.
    """

    def _apply_temporal_short_conv(
        self,
        x: torch.Tensor,
        conv: ShortConvolution,
        HW: tuple[int, int, int],
        **kwargs: object,
    ) -> torch.Tensor:
        """Route short conv: chunk-causal when chunk boundaries exist, else bidirectional.

        For single-chunk (``chunk_size >= T``) or no chunk_size, use the
        bidirectional short conv (identical forward AND backward to the
        parent class).  For multi-chunk, use chunk-causal short conv to
        enforce causality.
        """
        chunk_size = kwargs.get("chunk_size", None)
        T = HW[0]
        if chunk_size is not None and chunk_size < T:
            return ChunkCausalGDN._apply_temporal_short_conv(self, x, conv, HW, **kwargs)
        return BidirectionalGDN._apply_temporal_short_conv(self, x, conv, HW, **kwargs)


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


# ===========================================================================
# Cached streaming variants (camera-wrapped)
# ===========================================================================
#
# Streaming-inference subclasses of the chunk-causal cam classes above.
# GDN cam-wrapper dispatches main + cam branches through the fused Triton
# kernels in :mod:`diffusion.model.ops.fused_streaming`; softmax cam-wrapper
# prepends cached cam K, V to the current chunk and runs SDPA via the
# unified ``_sdpa_maybe_chunk_causal`` helper in
# :mod:`diffusion.model.nets.sana_gdn_blocks`.


@ATTENTION_BLOCKS.register_module()
class CachedChunkCausalGDNUCPESinglePathLiteLA(ChunkCausalGDNUCPESinglePathLiteLA):
    """Cached variant of :class:`ChunkCausalGDNUCPESinglePathLiteLA`.

    Main branch: :class:`CachedChunkCausalGDN` (state-based cache via
    ``_cached_gdn_forward_triton``).
    Camera branch: ``_cam_prep_triton`` + ``_cam_main_triton`` with
    ``cam_S_kv`` state cached in slot 2.
    """

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
    ) -> tuple[torch.Tensor, list]:
        from .sana_gdn_blocks import CachedChunkCausalGDN

        kv_cache = kwargs.pop("kv_cache", None)
        save_kv_cache = kwargs.pop("save_kv_cache", False)
        if kv_cache is None:
            raise RuntimeError(
                "CachedChunkCausalGDNUCPESinglePathLiteLA requires kv_cache "
                "to be provided (streaming inference only)."
            )
        if HW is None:
            raise ValueError("HW (T, H, W) must be provided.")

        # Pre-compute shared gates once for main + cam branches.
        precomputed_gates = self._compute_frame_gates(x, HW)

        main_raw, kv_cache = CachedChunkCausalGDN.forward(
            self,
            x,
            mask=mask,
            HW=HW,
            rotary_emb=rotary_emb,
            block_mask=block_mask,
            apply_output_gate=False,
            chunk_size=chunk_size,
            precomputed_gates=precomputed_gates,
            kv_cache=kv_cache,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

        cam_contrib: torch.Tensor | int = 0
        if camera_conditions is not None:
            cam_raw = self._cached_cam_branch(
                x,
                HW,
                camera_conditions,
                rotary_emb,
                kv_cache,
                save_kv_cache,
                precomputed_gates,
                chunk_size=chunk_size,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        output = self.proj(combined.to(self.proj.weight.dtype))
        return output, kv_cache

    def _cached_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        kv_cache: list,
        save_kv_cache: bool,
        precomputed_gates: tuple | None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Camera branch with cached delta-rule state."""
        from diffusion.model.ops.fused_streaming import (
            _SLOT_CAM,
            _cam_main_triton,
            _cam_prep_triton,
        )

        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        dtype_orig = x.dtype

        # Fused Triton cam prep: RMSNorm + ReLU + K-scale + UCPE 4x4 + RoPE.
        q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq, apply_fn_o = _cam_prep_triton(
            self, x, HW, camera_conditions, rotary_emb, **kwargs
        )

        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        # Dynamic beta discounting (UCPE inflation factor).
        inflation_sq_spatial = inflation_sq.view(B, self.cam_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        out, cam_S_kv_final = _cam_main_triton(
            q_cam_trans,
            k_cam_trans,
            v_cam_trans,
            beta,
            decay,
            kv_cache[_SLOT_CAM],
            save_kv_cache,
            T,
            S,
        )
        if save_kv_cache:
            kv_cache[_SLOT_CAM] = cam_S_kv_final.detach().clone()

        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        return out.reshape(B, self.cam_dim, N).permute(0, 2, 1)


@ATTENTION_BLOCKS.register_module()
class CachedSoftmaxUCPESinglePathLiteLA(_SoftmaxUCPESinglePathLiteLA):
    """Cached softmax + UCPE camera attention for streaming inference.

    Main branch: :class:`CachedChunkCausalSoftmaxAttn` (concatenate cached
    post-RoPE K, V; SDPA).
    Camera branch: concatenate cached post-UCPE cam K, V; SDPA on the unioned
    sequence.  Falls back to the parent's non-cached forward when
    ``kv_cache`` is absent.
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
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        from .sana_gdn_blocks import CachedChunkCausalSoftmaxAttn

        kv_cache = kwargs.pop("kv_cache", None)
        save_kv_cache = kwargs.pop("save_kv_cache", False)
        if kv_cache is None:
            return super().forward(
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                camera_conditions=camera_conditions,
                chunk_size=chunk_size,
                **kwargs,
            )
        if HW is None:
            raise ValueError("HW must be provided.")

        main_raw, kv_cache = CachedChunkCausalSoftmaxAttn.forward(
            self,
            x,
            mask=mask,
            HW=HW,
            rotary_emb=rotary_emb,
            block_mask=block_mask,
            apply_output_gate=False,
            chunk_size=chunk_size,
            kv_cache=kv_cache,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

        cam_contrib: torch.Tensor | int = 0
        if camera_conditions is not None:
            cam_raw = self._cached_cam_branch_softmax(
                x,
                HW,
                camera_conditions,
                rotary_emb,
                kv_cache,
                save_kv_cache,
                chunk_size=chunk_size,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        return self.proj(combined.to(x.dtype)), kv_cache

    def _cached_cam_branch_softmax(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        kv_cache: list,
        save_kv_cache: bool,
        **kwargs: object,
    ) -> torch.Tensor:
        """Camera branch: SDPA with cached UCPE-transformed K, V."""
        from diffusion.model.ops.fused_streaming import _SLOT_CAM, _SLOT_CAM_AUX

        from .sana_gdn_blocks import _sdpa_maybe_chunk_causal

        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp

        q_cam_trans, k_cam_trans, v_cam_trans, apply_fn_o = _prepare_cam_qkv_softmax(
            self, x, HW, camera_conditions, rotary_emb, **kwargs
        )

        # (B, H, D, N) -> (B, H, N, D) for SDPA.
        q_sdpa = q_cam_trans.transpose(-1, -2)
        k_sdpa = k_cam_trans.transpose(-1, -2)
        v_sdpa = v_cam_trans.transpose(-1, -2)

        dtype_orig = x.dtype
        if q_sdpa.dtype == torch.float32:
            q_sdpa, k_sdpa, v_sdpa = q_sdpa.bfloat16(), k_sdpa.bfloat16(), v_sdpa.bfloat16()

        cached_cam_k = kv_cache[_SLOT_CAM]
        cached_cam_v = kv_cache[_SLOT_CAM_AUX]
        if save_kv_cache:
            kv_cache[_SLOT_CAM] = k_sdpa.detach().clone()
            kv_cache[_SLOT_CAM_AUX] = v_sdpa.detach().clone()
        if cached_cam_k is not None:
            k_sdpa = torch.cat([cached_cam_k.to(k_sdpa.dtype), k_sdpa], dim=2)
            v_sdpa = torch.cat([cached_cam_v.to(v_sdpa.dtype), v_sdpa], dim=2)

        out = _sdpa_maybe_chunk_causal(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            need_chunk_mask=False,
            T=T,
            S=S,
            chunk_size=kwargs.get("chunk_size", None),
            chunk_index=kwargs.get("chunk_index", None),
            chunk_split_strategy=kwargs.get("chunk_split_strategy", "uniform"),
            device=x.device,
        )

        out = out.transpose(-1, -2)
        if out.dtype != dtype_orig:
            out = out.to(dtype_orig)
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        return out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
