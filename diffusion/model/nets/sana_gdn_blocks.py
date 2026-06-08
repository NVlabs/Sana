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

"""Frame-wise Gated Delta Net (GDN) attention for Sana video."""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import ShortConvolution
from timm.models.vision_transformer import Attention as Attention_

from diffusion.model.liger_norms import get_rmsnorm_class
from diffusion.utils.chunk_utils import (
    chunk_index_from_chunk_size,
    is_chunk_causal_request,
    is_uniform_chunking,
    normalize_chunk_index,
    size1_chunk_position_indices,
)

RMSNorm = get_rmsnorm_class()
from diffusion.model.registry import ATTENTION_BLOCKS

# Gate ``@torch.compile`` on all GDN scan / helper functions via
# ``GDN_DISABLE_COMPILE``.  When set to anything other than ``"0"`` / ``"false"``,
# compile is disabled (useful for debugging / parity work).
_COMPILE_DISABLE = os.environ.get("GDN_DISABLE_COMPILE", "0") not in ("0", "false")

_HAS_FLEX_ATTENTION = bool(int(os.environ.get("SANA_USE_FLEX_ATTENTION", "0")))
_SDPA_D112_DIRECT = os.environ.get("SANA_WM_SDPA_D112_DIRECT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

OUTPUT_GATE_INIT_BIAS = 1.278464542761074  # silu(x)=1.0


def _sdpa_needs_head_pad(head_dim: int) -> bool:
    if head_dim == 112 and _SDPA_D112_DIRECT:
        return False
    return head_dim not in (32, 64, 128, 256) and head_dim < 256


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def flip_and_shift(x, dim=2, shift_val=0.0):
    """Flip a sequence and shift it right by one step.

    The operation reverses the sequence, drops the last element, and pads the
    front with ``shift_val``.

    Example:
        [x0, x1, x2, x3] -> flip [x3, x2, x1, x0] -> shift [v, x3, x2, x1]

    Args:
        x: Input tensor with a time dimension at ``dim``.
        dim: Dimension to flip and shift.
        shift_val: Value used for the padded step.

    Returns:
        Tensor with the same shape as ``x``.
    """
    x_flip = torch.flip(x, dims=[dim])
    x_shifted = x_flip.narrow(dim, 0, x.shape[dim] - 1)
    pad_shape = list(x.shape)
    pad_shape[dim] = 1
    padding = torch.full(pad_shape, shift_val, device=x.device, dtype=x.dtype)
    return torch.cat([padding, x_shifted], dim=dim)


class _IdentityForwardContiguousBackward(torch.autograd.Function):
    """Identity in forward; force contiguous grad tensor in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (grad_output.contiguous(),)


def _contiguous_backward(x: torch.Tensor) -> torch.Tensor:
    """Ensure downstream backward receives a contiguous gradient buffer."""
    return _IdentityForwardContiguousBackward.apply(x)


@torch.compile(disable=_COMPILE_DISABLE)
def _compute_frame_gates(
    x: torch.Tensor,
    T: int,
    S: int,
    heads: int,
    beta_weight: torch.Tensor,
    beta_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled frame gate computation (fuses sigmoid + softplus + exp chain)."""
    B, N, C = x.shape
    beta = F.linear(x, beta_weight, beta_bias).sigmoid().reshape(B, T, S, heads).permute(0, 3, 1, 2)
    x_frame = x.reshape(B, T, S, C).mean(dim=2)
    a_out = F.linear(x_frame, gate_weight, gate_bias).float()
    dt = dt_bias.float().view(1, 1, -1)
    A_val = A_log.float().exp().view(1, 1, -1)
    decay = (-A_val * F.softplus(a_out + dt)).exp().transpose(1, 2)
    return beta, decay


@torch.compile(disable=_COMPILE_DISABLE)
def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Compiled rotary embedding application (fuses view_as_complex + multiply chain)."""
    x_rotated = torch.view_as_complex(
        hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)),
    )
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
    return x_out.type_as(hidden_states)


@torch.compile(disable=_COMPILE_DISABLE)
def _apply_output_gate(
    out: torch.Tensor,
    gate_x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
) -> torch.Tensor:
    """Compiled output gate (fuses linear + silu + multiply)."""
    gate = F.silu(F.linear(gate_x, gate_weight, gate_bias).to(torch.float32))
    return out * gate


@ATTENTION_BLOCKS.register_module()
class GDN(Attention_):
    """Frame-wise Gated Delta Net attention for Sana video.

    This block follows Sana's vanilla linear attention strategy but upgrades it
    with a Gated Delta Network mechanism:
    - Apply ReLU kernel to q/k.
    - Apply RoPE only on the numerator (q_rot, k_rot).
    - Denominator (Z stream) uses unrotated q/k to maintain mass conservation.
    - Gated delta rule is applied across time (T). Gates are computed per-frame
      (shared spatially), but states are maintained per-pixel.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        eps: float = 1e-15,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
        use_output_gate: bool = True,
        conv_kernel_size: int = 4,
        k_conv_only: bool = True,
        **kwargs: object,
    ) -> None:
        heads = heads or int(out_dim // dim * heads_ratio)
        super().__init__(in_dim, num_heads=heads, qkv_bias=use_bias)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps
        self.k_conv_only = k_conv_only
        self.key_scale_mode = str(kwargs.pop("key_scale_mode", "dim_spatial"))

        self.kernel_func = nn.ReLU(inplace=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(self.in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Gate projections operate on pooled frame features (B, T, D) -> (B, T, H).
        self.beta_proj = nn.Linear(in_dim, heads, bias=True)
        self.gate_proj = nn.Linear(in_dim, heads, bias=True)

        A = torch.empty(self.heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Explicitly skip weight decay (biases are excluded in param grouping).
        self.dt_bias._no_weight_decay = True

        # recall_gate is unused (computation commented out) but kept as buffer
        # for checkpoint backward compatibility. Converted from Parameter to buffer
        # because FSDP2's set_optimizer_state_dict fails on scalar parameters.
        self.register_buffer("recall_gate", torch.zeros(1))

        self.use_output_gate = use_output_gate
        if use_output_gate:
            self.output_gate = nn.Linear(in_dim, out_dim, bias=True)
        else:
            self.output_gate = None

        self.qkv_store_buffer = None

        # Short Convolutions (FLA causal depthwise Conv1d along T)
        self.conv_kernel_size = conv_kernel_size
        if conv_kernel_size > 0:
            self.conv_k = ShortConvolution(
                hidden_size=out_dim,
                kernel_size=conv_kernel_size,
                activation=None,
            )
            if k_conv_only:
                self.conv_q = None
                self.conv_v = None
            else:
                self.conv_q = ShortConvolution(
                    hidden_size=out_dim,
                    kernel_size=conv_kernel_size,
                    activation=None,
                )
                self.conv_v = ShortConvolution(
                    hidden_size=out_dim,
                    kernel_size=conv_kernel_size,
                    activation=None,
                )
        else:
            self.conv_q = None
            self.conv_k = None
            self.conv_v = None

        self._init_gdn_gates_for_linear_equiv()

    def _key_scale(self, spatial_tokens: int) -> float:
        """Return the post-ReLU key scale used by frame-wise GDN."""
        if self.key_scale_mode == "dim_spatial":
            return (self.dim**-0.5) * (spatial_tokens**-0.5)
        if self.key_scale_mode == "dim":
            return self.dim**-0.5
        if self.key_scale_mode == "none":
            return 1.0
        raise ValueError(f"Unsupported GDN key_scale_mode: {self.key_scale_mode}")

    def _init_short_conv_for_linear_equiv(self) -> None:
        """Initialize short conv as identity to match no-conv behavior at step 0."""
        if self.conv_k is None:
            return

        for conv in (self.conv_q, self.conv_k, self.conv_v):
            if conv is None:
                continue
            with torch.no_grad():
                # FLA ShortConvolution uses causal kernels. The last tap is x[t].
                conv.weight.zero_()
                conv.weight[:, 0, -1] = 1.0
                if getattr(conv, "bias", None) is not None:
                    conv.bias.zero_()

    def _init_gdn_gates_for_linear_equiv(self) -> None:
        """Initialize gates near identity to mimic Linear Attention at start."""
        self.recall_gate.zero_()  # buffer, not parameter

        # Beta ≈ 1.0
        # Sigmoid(5.0) ≈ 0.993
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 5.0)

        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        with torch.no_grad():
            self.dt_bias.fill_(-5.0)
            self.A_log.fill_(math.log(1.0))

        if self.use_output_gate and self.output_gate is not None:
            nn.init.zeros_(self.output_gate.weight)
            nn.init.constant_(self.output_gate.bias, OUTPUT_GATE_INIT_BIAS)

        self._init_short_conv_for_linear_equiv()

    def _apply_output_gate(self, out: torch.Tensor, gate_x: torch.Tensor) -> torch.Tensor:
        if not (self.use_output_gate and self.output_gate is not None):
            return out
        return _apply_output_gate(out, gate_x, self.output_gate.weight, self.output_gate.bias)

    @staticmethod
    def _reshape_to_temporal(x: torch.Tensor, HW: tuple[int, int, int]) -> tuple[torch.Tensor, int, int, int]:
        """Reshape (B, T*S, C) to (B*S, T, C) for temporal conv.

        Returns:
            Reshaped tensor and (B, S, T) for later restoration.
        """
        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        # FLA ShortConvolution backward is not reliable on non-contiguous
        # strided layouts produced by this permutation path.
        x = x.reshape(B, T, S, C).permute(0, 2, 1, 3).contiguous().reshape(B * S, T, C)
        return x, B, S, T

    @staticmethod
    def _reshape_from_temporal(x: torch.Tensor, B: int, S: int, T: int) -> torch.Tensor:
        """Reshape (B*S, T, C) back to (B, T*S, C)."""
        x = _contiguous_backward(x)
        C = x.shape[-1]
        return x.reshape(B, S, T, C).permute(0, 2, 1, 3).reshape(B, T * S, C)

    @staticmethod
    def _causal_conv_1d(
        x: torch.Tensor,
        conv: ShortConvolution,
    ) -> torch.Tensor:
        """Run causal conv and preserve input dtype.

        Args:
            x: Tensor of shape (batch, seq_len, channels).
            conv: FLA ``ShortConvolution`` module.

        Returns:
            Tensor of same shape and dtype as ``x``.
        """
        dtype_in = x.dtype
        y, _ = conv(x)
        if y.dtype != dtype_in:
            y = y.to(dtype_in)
        return y

    @staticmethod
    def _bidirectional_causal_conv_1d(
        x: torch.Tensor,
        conv: ShortConvolution,
    ) -> torch.Tensor:
        """Simulate non-causal conv by combining forward + backward causal passes.

        A causal depthwise Conv1d with kernel ``[w_0, w_1, ..., w_{k-1}]``
        computes at time *t*:

            ``y_fwd[t] = w_0 * x[t-k+1] + ... + w_{k-1} * x[t]``

        Running the same kernel on the time-flipped input and flipping back
        gives:

            ``y_bwd[t] = w_{k-1} * x[t] + ... + w_0 * x[t+k-1]``

        Both passes include the current timestep ``x[t]`` with the center
        weight ``w_{k-1}``.  To avoid double-counting we subtract one copy
        of the center contribution:

            ``y = y_fwd + y_bwd - w_{k-1} * x``

        The result is a symmetric temporal filter where every position in
        the window ``[t-k+1, t+k-1]`` is counted exactly once.

        Args:
            x: Tensor of shape ``(batch, seq_len, channels)``.
            conv: FLA ``ShortConvolution`` module (depthwise causal Conv1d).

        Returns:
            Tensor of same shape and dtype as ``x``.
        """
        dtype_in = x.dtype

        y_fwd, _ = conv(x)
        y_bwd, _ = conv(x.flip(1))
        y_bwd = y_bwd.flip(1)

        # Subtract the shared center tap (last weight of the causal kernel).
        # ShortConvolution weight shape: (channels, 1, kernel_size).
        # The last element along dim=-1 is the weight applied to x[t].
        w_center = conv.weight[:, 0, -1]  # (channels,)
        center_term = x * w_center.unsqueeze(0).unsqueeze(0)  # broadcast over (B, T)

        y = y_fwd + y_bwd - center_term
        if y.dtype != dtype_in:
            y = y.to(dtype_in)
        return y

    def _apply_temporal_short_conv(
        self,
        x: torch.Tensor,
        conv: ShortConvolution,
        HW: tuple[int, int, int],
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply causal ShortConvolution along T, with S merged into batch.

        Under CP, a causal conv of kernel size K needs K-1 left-context
        frames from the previous rank at each boundary.  We use a halo
        exchange (O(K) communication) instead of a full gather (O(T)).

        Args:
            x: Input tensor of shape (B, N, C) where N = T * S.
            conv: FLA ``ShortConvolution`` module.
            HW: Tuple of (T, H, W) describing the token layout.
            **kwargs: Extra keyword arguments (unused in base; subclasses
                may consume ``chunk_size``, ``chunk_index``, etc.).

        Returns:
            Tensor of shape (B, N, C) after temporal convolution.
        """
        del kwargs  # unused in base class

        x, B, S, T = self._reshape_to_temporal(x, HW)
        x = self._causal_conv_1d(x, conv)
        return self._reshape_from_temporal(x, B, S, T)

    @staticmethod
    def _apply_rotary_emb(
        hidden_states: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embeddings (delegates to compiled ``_apply_rotary_emb``)."""
        return _apply_rotary_emb(hidden_states, freqs)

    def _compute_frame_gates(
        self,
        x: torch.Tensor,
        hw: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-frame gates shared across spatial positions.

        Delegates to the module-level compiled ``_compute_frame_gates``.
        """
        T, H, W = hw
        S = H * W
        return _compute_frame_gates(
            x,
            T,
            S,
            self.heads,
            self.beta_proj.weight,
            self.beta_proj.bias,
            self.gate_proj.weight,
            self.gate_proj.bias,
            self.dt_bias,
            self.A_log,
        )

    @staticmethod
    def _prepare_frame_valid_masks(
        frame_valid_mask: torch.Tensor | None,
        *,
        B: int,
        T: int,
        S: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Convert frame-valid mask to token/beta/decay masks used by GDN blocks."""
        if frame_valid_mask is None:
            return None, None, None

        m = frame_valid_mask
        if m.ndim == 5:
            # (B, 1, T, 1, 1)
            m = m[:, 0, :, 0, 0]
        elif m.ndim == 3 and m.shape[1] == 1:
            # (B, 1, T)
            m = m[:, 0, :]
        elif m.ndim != 2:
            raise ValueError(
                "frame_valid_mask must be shaped (B, 1, T, 1, 1), (B, 1, T), or (B, T); "
                f"got shape={list(frame_valid_mask.shape)}"
            )

        if m.shape[0] != B or m.shape[1] != T:
            raise ValueError(f"frame_valid_mask shape mismatch: expected (B={B}, T={T}), got {list(m.shape)}")

        m = m.to(device=device, dtype=dtype)
        token_valid_mask = m[:, :, None].expand(B, T, S).reshape(B, T * S)
        beta_valid_mask = m.view(B, 1, T, 1)
        decay_valid_mask = m.view(B, 1, T)
        return token_valid_mask, beta_valid_mask, decay_valid_mask


@ATTENTION_BLOCKS.register_module()
class BidirectionalGDN(GDN):
    """Bidirectional GDN attention with forward/backward fusion."""

    def _apply_temporal_short_conv(
        self,
        x: torch.Tensor,
        conv: ShortConvolution,
        HW: tuple[int, int, int],
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply bidirectional (non-causal) ShortConvolution along T.

        Uses the forward+backward causal trick: run the causal conv in
        both directions and average, yielding a symmetric temporal filter
        with a single set of weights.

        Args:
            x: Input tensor of shape (B, N, C) where N = T * S.
            conv: FLA ``ShortConvolution`` module.
            HW: Tuple of (T, H, W) describing the token layout.
            **kwargs: Unused.

        Returns:
            Tensor of shape (B, N, C) after bidirectional temporal conv.
        """
        del kwargs

        x, B, S, T = self._reshape_to_temporal(x, HW)
        x = self._bidirectional_causal_conv_1d(x, conv)
        return self._reshape_from_temporal(x, B, S, T)


@ATTENTION_BLOCKS.register_module()
class ChunkCausalGDN(GDN):
    """Chunk-causal GDN attention.

    Within each chunk the recurrence behaves bidirectionally (forward
    causal scan plus per-chunk backward scan); across chunks it remains
    strictly causal.  This matches the attention pattern of a frame-wise
    block-causal mask while retaining the linear-time GDN scan.

    Chunk boundaries are derived from ``chunk_size`` / ``chunk_index`` /
    ``chunk_split_strategy`` passed via ``forward``.  When ``chunk_size``
    is ``None`` or larger than ``T`` the block degenerates to the
    bidirectional GDN scan.
    """

    @staticmethod
    def _backward_causal_conv_per_chunk(
        x: torch.Tensor,
        conv: ShortConvolution,
        T: int,
        chunk_size: int | None,
        chunk_index: list[int] | None,
        chunk_split_strategy: str,
    ) -> torch.Tensor:
        """Run backward (anti-causal) conv isolated per chunk.

        Within each chunk the input is time-flipped, the causal conv is
        applied, and the output is flipped back.  Chunks do not share any
        context, preventing backward information leakage across boundaries.

        Args:
            x: Tensor of shape ``(B*S, T, C)``.
            conv: FLA ``ShortConvolution`` module.
            T: Number of temporal frames.
            chunk_size: Uniform chunk size (or ``None``).
            chunk_index: Explicit chunk boundaries (or ``None``).
            chunk_split_strategy: Strategy for deriving boundaries.

        Returns:
            Tensor of shape ``(B*S, T, C)`` — per-chunk backward conv.
        """
        BS = x.shape[0]

        if chunk_size is not None and T % chunk_size == 0:
            # Vectorized: reshape chunks into batch, flip, conv, flip back.
            num_chunks = T // chunk_size
            xc = x.reshape(BS, num_chunks, chunk_size, -1)
            xc = xc.reshape(BS * num_chunks, chunk_size, -1)
            yc, _ = conv(xc.flip(1))
            yc = yc.flip(1)
            return yc.reshape(BS, num_chunks, chunk_size, -1).reshape(BS, T, -1)

        # Resolve chunk boundaries for non-uniform patterns.
        valid_chunk_index, _ = normalize_chunk_index(chunk_index, T, chunk_size, chunk_split_strategy)
        chunk_sizes = [valid_chunk_index[i + 1] - valid_chunk_index[i] for i in range(len(valid_chunk_index) - 1)]

        # Fast path for first_plus_one pattern: first chunk is (chunk_size+1),
        # remaining chunks are all chunk_size.  This reduces ~N conv calls to 2-3.
        if (
            chunk_size is not None
            and len(chunk_sizes) >= 2
            and chunk_sizes[0] == chunk_size + 1
            and all(cs == chunk_size for cs in chunk_sizes[1:])
        ):
            first_chunk_size = chunk_size + 1

            # Process first chunk (size chunk_size+1) with one conv call.
            first_seg = x[:, :first_chunk_size, :]
            first_out, _ = conv(first_seg.flip(1))
            first_out = first_out.flip(1)

            # Vectorize the uniform tail into batched conv calls.
            # Cap batch size to avoid Triton kernel grid-dimension limits
            # (BS * num_tail can reach ~17k during inference, exceeding limits).
            _MAX_CONV_BATCH = 4096
            tail_x = x[:, first_chunk_size:, :]
            T_tail = T - first_chunk_size
            num_tail = T_tail // chunk_size
            if num_tail > 0:
                vectorizable_len = num_tail * chunk_size
                total_batch = BS * num_tail
                if total_batch <= _MAX_CONV_BATCH:
                    tail_batch = tail_x[:, :vectorizable_len, :].reshape(total_batch, chunk_size, -1)
                    tail_out, _ = conv(tail_batch.flip(1))
                    tail_out = tail_out.flip(1).reshape(BS, vectorizable_len, -1)
                else:
                    # Process in sub-batches to stay within kernel limits.
                    max_chunks_per_call = max(1, _MAX_CONV_BATCH // BS)
                    tail_parts: list[torch.Tensor] = []
                    for i in range(0, num_tail, max_chunks_per_call):
                        n = min(max_chunks_per_call, num_tail - i)
                        seg_len = n * chunk_size
                        seg = tail_x[:, i * chunk_size : i * chunk_size + seg_len, :]
                        seg_batch = seg.reshape(BS * n, chunk_size, -1)
                        seg_out, _ = conv(seg_batch.flip(1))
                        tail_parts.append(seg_out.flip(1).reshape(BS, seg_len, -1))
                    tail_out = torch.cat(tail_parts, dim=1)

                # Handle possible remainder chunk (if T_tail is not divisible by chunk_size).
                remainder = T_tail - vectorizable_len
                if remainder > 0:
                    rem_seg = tail_x[:, vectorizable_len:, :]
                    rem_out, _ = conv(rem_seg.flip(1))
                    rem_out = rem_out.flip(1)
                    return torch.cat([first_out, tail_out, rem_out], dim=1)
                return torch.cat([first_out, tail_out], dim=1)
            else:
                # Only the first chunk exists (edge case: T == chunk_size+1).
                return first_out

        # Generic fallback: loop over arbitrary chunk boundaries.
        bounds = list(zip(valid_chunk_index[:-1], valid_chunk_index[1:]))
        parts: list[torch.Tensor] = []
        for start_t, end_t in bounds:
            seg = x[:, start_t:end_t, :]
            seg_out, _ = conv(seg.flip(1))
            parts.append(seg_out.flip(1))
        return torch.cat(parts, dim=1)

    def _apply_temporal_short_conv(
        self,
        x: torch.Tensor,
        conv: ShortConvolution,
        HW: tuple[int, int, int],
        **kwargs: object,
    ) -> torch.Tensor:
        """Chunk-causal ShortConvolution: global forward + per-chunk backward.

        Mirrors the ChunkCausalGDN recurrence semantics:

        * **Forward (causal)** — runs over the full sequence so that later
          chunks receive temporal context from earlier chunks.
        * **Backward (anti-causal)** — runs independently inside each chunk
          so that no future information leaks across chunk boundaries.
        * **Center-tap correction** — the current timestep ``x[t]`` appears
          in both passes; one copy is subtracted so every position in the
          resulting symmetric window is counted exactly once.

        Args:
            x: Input tensor of shape ``(B, N, C)`` where ``N = T * S``.
            conv: FLA ``ShortConvolution`` module.
            HW: Tuple of ``(T, H, W)`` describing the token layout.
            **kwargs: Must contain ``chunk_size``, ``chunk_index``, and
                ``chunk_split_strategy``.

        Returns:
            Tensor of shape ``(B, N, C)``.
        """
        chunk_size = kwargs.get("chunk_size")
        chunk_index = kwargs.get("chunk_index")
        chunk_split_strategy = kwargs.get("chunk_split_strategy", "uniform")

        dtype_in = x.dtype
        x, B, S, T = self._reshape_to_temporal(x, HW)

        # NOTE: CP removed — single-GPU path only.
        # 1. Global forward causal conv (cross-chunk context flows forward).
        y_fwd, _ = conv(x)

        # 2. Per-chunk backward causal conv (isolated within each chunk).
        y_bwd = self._backward_causal_conv_per_chunk(
            x,
            conv,
            T,
            chunk_size,
            chunk_index,
            chunk_split_strategy,
        )

        # 3. Subtract the shared center tap to avoid double-counting x[t].
        w_center = conv.weight[:, 0, -1]  # (channels,)
        center_term = x * w_center.unsqueeze(0).unsqueeze(0)

        y = y_fwd + y_bwd - center_term
        if y.dtype != dtype_in:
            y = y.to(dtype_in)

        return self._reshape_from_temporal(y, B, S, T)


_frame_causal_mask_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _get_frame_causal_mask(T: int, S: int, device: torch.device) -> torch.Tensor:
    """Frame-wise block-causal mask: full attention within each frame,
    causal across frames.

    Returns a boolean tensor of shape ``(1, 1, T*S, T*S)`` where ``True``
    indicates positions that may attend.
    """
    key = (T, S, device)
    if key not in _frame_causal_mask_cache:
        frame_idx = torch.arange(T, device=device).repeat_interleave(S)
        mask = frame_idx.unsqueeze(1) >= frame_idx.unsqueeze(0)
        _frame_causal_mask_cache[key] = mask.unsqueeze(0).unsqueeze(0)
    return _frame_causal_mask_cache[key]


def _forward_softmax_attn(
    self,
    x: torch.Tensor,
    HW: tuple[int, int, int],
    rotary_emb: torch.Tensor | None,
    frame_causal: bool,
    apply_output_gate: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Softmax attention (SDPA) reusing GDN parameters.

    Used by the hybrid GDN+Softmax architecture: every Nth block runs
    softmax attention instead of the gated-delta recurrence. Reuses the
    parent block's QKV/q_norm/k_norm/proj for parameter compatibility.
    """
    import torch.nn.functional as F

    B, N, C = x.shape
    T, H, W = HW
    S = H * W

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

    qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
    q, k, v = qkv.unbind(2)
    if token_valid_mask is not None:
        m = token_valid_mask.view(B, N, 1, 1)
        q, k, v = q * m, k * m, v * m

    q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
    k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

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

    attn_mask = _get_frame_causal_mask(T, S, x.device) if frame_causal else None

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2).reshape(B, N, C).to(dtype_orig)

    if apply_output_gate:
        # Re-apply the parent's output projection w/ silu gate; some GDN
        # variants split projection into proj_o + proj_gate; match those.
        if hasattr(self, "proj_gate"):
            out = out * F.silu(self.proj_gate(x))
        out = self.proj(out)
    return out


# ---------------------------------------------------------------------------
# Chunk-causal softmax attention for hybrid GDN-Softmax architectures
# ---------------------------------------------------------------------------

# flex_attention for chunk-causal softmax (single-kernel, no O(N^2) mask).
try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention_raw

    _flex_attention = torch.compile(_flex_attention_raw, dynamic=False, mode="max-autotune-no-cudagraphs")
    _HAS_FLEX_ATTENTION_CHUNK = True
except ImportError:
    _HAS_FLEX_ATTENTION_CHUNK = False

_chunk_causal_block_mask_cache: dict[tuple, BlockMask] = {}


def _get_chunk_causal_block_mask(
    chunk_boundaries: list[int],
    S: int,
    q_len: int,
    kv_len: int,
    q_frame_offset: int,
    device: torch.device,
) -> BlockMask:
    """Build a flex_attention BlockMask for chunk-causal attention.

    Token ``q_idx`` can attend to ``kv_idx`` iff ``chunk(q) >= chunk(kv)``,
    i.e. ``kv_idx < chunk_end(q)``.  Uses the HiAR pattern: precompute an
    ``ends`` tensor mapping each token to its chunk's exclusive end token
    index.  Results are cached by
    ``(chunk_boundaries, S, q_len, kv_len, q_frame_offset, device)``.
    """
    cache_key = (tuple(chunk_boundaries), S, q_len, kv_len, q_frame_offset, device)
    if cache_key in _chunk_causal_block_mask_cache:
        return _chunk_causal_block_mask_cache[cache_key]

    # flex_attention requires Q_LEN and KV_LEN to be multiples of 128.
    q_pad = (128 - q_len % 128) % 128
    kv_pad = (128 - kv_len % 128) % 128
    Q_LEN = q_len + q_pad
    KV_LEN = kv_len + kv_pad

    # Build per-token ``ends`` array: ends[tok] = exclusive end token index
    # of the chunk that ``tok`` belongs to (in global KV token space).
    # Padded tokens map to KV_LEN so they attend to everything (masked later).
    ends_kv = torch.full((KV_LEN,), KV_LEN, device=device, dtype=torch.long)
    for ci in range(len(chunk_boundaries) - 1):
        tok_start = chunk_boundaries[ci] * S
        tok_end = chunk_boundaries[ci + 1] * S
        if tok_end > KV_LEN:
            tok_end = KV_LEN
        if tok_start < KV_LEN:
            ends_kv[tok_start:tok_end] = tok_end

    # For Q tokens: map local Q index -> global token, then look up chunk end.
    q_offset_tokens = q_frame_offset * S
    ends_q = torch.full((Q_LEN,), KV_LEN, device=device, dtype=torch.long)
    for qi in range(min(q_len, Q_LEN)):
        global_qi = qi + q_offset_tokens
        if global_qi < len(ends_kv):
            ends_q[qi] = ends_kv[global_qi]

    def mask_fn(b, h, q_idx, kv_idx):
        return kv_idx < ends_q[q_idx]

    block_mask = create_block_mask(
        mask_fn,
        B=None,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        _compile=False,
        device=device,
    )

    _chunk_causal_block_mask_cache[cache_key] = block_mask
    return block_mask


_chunk_causal_mask_cache: dict[tuple, torch.Tensor] = {}


def _get_chunk_causal_mask(
    T: int,
    S: int,
    chunk_boundaries: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Chunk-wise block-causal mask for video generation.

    Full attention within each chunk (all spatial tokens across all frames
    in the chunk attend to each other), causal across chunks (tokens in
    chunk C attend to all tokens in chunks 0..C only).

    Args:
        T: Number of temporal frames.
        S: Number of spatial tokens per frame (``H * W``).
        chunk_boundaries: Sorted chunk boundary list ``[0, c1, ..., T]``.
        device: Target device for the mask tensor.

    Returns:
        Boolean mask ``(1, 1, T*S, T*S)`` where ``True`` = allowed to attend.
    """
    key = (T, S, tuple(chunk_boundaries), device)
    if key not in _chunk_causal_mask_cache:
        frame_to_chunk = torch.zeros(T, device=device, dtype=torch.long)
        for i in range(len(chunk_boundaries) - 1):
            frame_to_chunk[chunk_boundaries[i] : chunk_boundaries[i + 1]] = i
        token_to_chunk = frame_to_chunk.repeat_interleave(S)
        mask = token_to_chunk.unsqueeze(1) >= token_to_chunk.unsqueeze(0)
        _chunk_causal_mask_cache[key] = mask.unsqueeze(0).unsqueeze(0)
    return _chunk_causal_mask_cache[key]


def _forward_softmax_attn_chunk_causal(
    self: GDN,
    x: torch.Tensor,
    HW: tuple[int, int, int],
    rotary_emb: torch.Tensor | None,
    chunk_size: int | None,
    chunk_split_strategy: str,
    chunk_index: list[int] | None,
    apply_output_gate: bool = True,
    **kwargs: object,
) -> torch.Tensor:
    """Chunk-causal softmax attention (SDPA / flex_attention) reusing GDN parameters.

    Used by ``ChunkCausalSoftmaxAttn``.  Reuses ``qkv``, ``q_norm``,
    ``k_norm``, ``proj``, and the output gate from the parent ``GDN``
    parameter set.  When ``chunk_size`` is ``None`` or ``>= T`` the
    attention degenerates to fully bidirectional softmax.
    """
    B, N, C = x.shape
    T, H, W = HW
    S = H * W

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

    qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
    q, k, v = qkv.unbind(2)
    if token_valid_mask is not None:
        m = token_valid_mask.view(B, N, 1, 1)
        q, k, v = q * m, k * m, v * m

    q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
    k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

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

    # NOTE: CP removed — single-GPU path only.
    _chunk_causal = chunk_size is not None and chunk_size < T

    if _chunk_causal:
        chunk_boundaries, _ = normalize_chunk_index(chunk_index, T, chunk_size, chunk_split_strategy)
        q_len = T * S
        kv_len = T * S
        q_frame_offset = 0

        if _HAS_FLEX_ATTENTION_CHUNK:
            # flex_attention: single compiled kernel with block-sparse mask.
            block_mask = _get_chunk_causal_block_mask(chunk_boundaries, S, q_len, kv_len, q_frame_offset, x.device)
            q_pad = (128 - q_len % 128) % 128
            kv_pad = (128 - kv_len % 128) % 128
            if q_pad > 0:
                q = F.pad(q, (0, 0, 0, q_pad))
            if kv_pad > 0:
                k = F.pad(k, (0, 0, 0, kv_pad))
                v = F.pad(v, (0, 0, 0, kv_pad))
            out = _flex_attention(q, k, v, block_mask=block_mask)
            if q_pad > 0:
                out = out[:, :, :q_len, :]
        else:
            # Fallback: per-chunk loop with head_dim padding for FlashAttention.
            D = q.shape[-1]
            _need_pad = _sdpa_needs_head_pad(D)
            if _need_pad:
                _pad_to = 128 if D <= 128 else 256
                _pad_size = _pad_to - D
                q = F.pad(q, (0, _pad_size))
                k = F.pad(k, (0, _pad_size))
                v = F.pad(v, (0, _pad_size))
            out_chunks: list[torch.Tensor] = []
            for ci in range(len(chunk_boundaries) - 1):
                c_start = chunk_boundaries[ci]
                c_end = chunk_boundaries[ci + 1]
                q_chunk = q[:, :, c_start * S : c_end * S, :]
                out_chunk = F.scaled_dot_product_attention(q_chunk, k[:, :, : c_end * S, :], v[:, :, : c_end * S, :])
                out_chunks.append(out_chunk)
            out = torch.cat(out_chunks, dim=2)
            if _need_pad:
                out = out[..., :D]
    else:
        # Fully bidirectional softmax (no chunking).
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

    if out.dtype != dtype_orig:
        out = out.to(dtype_orig)

    out = out.transpose(1, 2).reshape(B, N, C)
    if token_valid_mask is not None:
        out = out * token_valid_mask.view(B, N, 1).to(out.dtype)

    if apply_output_gate:
        out = self._apply_output_gate(out, x)
        out = self.proj(out.to(dtype_orig))
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
        return out
    return out


@ATTENTION_BLOCKS.register_module()
class ChunkCausalSoftmaxAttn(ChunkCausalGDN):
    """Chunk-causal softmax attention with GDN-compatible parameter layout.

    Inherits all parameters from ``ChunkCausalGDN`` for checkpoint
    compatibility.  GDN-specific parameters (``beta_proj``, ``gate_proj``,
    ``A_log``, ``dt_bias``, ``recall_gate``) are present but unused in
    forward.

    Uses ``F.scaled_dot_product_attention`` (or ``flex_attention`` when
    available) with a chunk-wise causal mask: full bidirectional attention
    within each chunk, causal across chunks.  This matches the attention
    pattern of ``ChunkCausalGDN`` while using exact softmax instead of the
    linear GDN recurrence.
    """

    def __init__(self, *args: object, conv_kernel_size: int = 0, **kwargs: object) -> None:
        del conv_kernel_size  # Softmax variant always uses conv_kernel_size=0.
        super().__init__(*args, conv_kernel_size=0, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        chunk_size: int | None = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: list[int] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply chunk-causal softmax attention to a token sequence."""
        del mask, block_mask
        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for ChunkCausalSoftmaxAttn.")
        return _forward_softmax_attn_chunk_causal(
            self,
            x,
            HW,
            rotary_emb,
            chunk_size=chunk_size,
            chunk_split_strategy=chunk_split_strategy,
            chunk_index=chunk_index,
            apply_output_gate=apply_output_gate,
            **kwargs,
        )


# ===========================================================================
# Cached streaming variants
# ===========================================================================
#
# These ``Cached*`` classes are streaming-inference subclasses of their
# non-cached parents above (``ChunkCausalGDN`` and ``ChunkCausalSoftmaxAttn``).
# Each ``forward()`` takes a per-block ``kv_cache`` (10-slot list) and a
# ``save_kv_cache`` flag; GDN classes dispatch to fused-Triton helpers in
# :mod:`diffusion.model.ops.fused_streaming`, softmax classes prepend cached
# K, V to the current chunk and run plain SDPA (cache enforces causality).
#
# Slot layout: see :mod:`diffusion.model.ops.fused_streaming` docstring.


def _sdpa_maybe_chunk_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    need_chunk_mask: bool,
    T: int,
    S: int,
    chunk_size: int | None,
    chunk_index: list[int] | None,
    chunk_split_strategy: str,
    device: torch.device,
) -> torch.Tensor:
    """Run SDPA with chunk-causal masking when needed, or plain SDPA otherwise.

    Replicates the masking logic from ``_forward_softmax_attn`` and
    ``_forward_cam_branch_softmax`` so that the cached softmax path produces
    bit-exact results for the first chunk (no cached state).  In the streaming
    inference loop ``need_chunk_mask`` is always ``False`` because the cache
    already enforces causality; we keep the masked branch as a defensive
    fallback.
    """
    if need_chunk_mask:
        chunk_boundaries, _ = normalize_chunk_index(chunk_index, T, chunk_size, chunk_split_strategy)
        q_len = T * S
        kv_len = T * S

        if _HAS_FLEX_ATTENTION:
            block_mask = _get_chunk_causal_block_mask(chunk_boundaries, S, q_len, kv_len, 0, device)
            q_pad = (128 - q_len % 128) % 128
            kv_pad = (128 - kv_len % 128) % 128
            if q_pad > 0:
                q = F.pad(q, (0, 0, 0, q_pad))
            if kv_pad > 0:
                k = F.pad(k, (0, 0, 0, kv_pad))
                v = F.pad(v, (0, 0, 0, kv_pad))
            out = _flex_attention(q, k, v, block_mask=block_mask)
            if q_pad > 0:
                out = out[:, :, :q_len, :]
            return out

        # Fallback: per-chunk loop with head_dim padding for FlashAttention.
        D = q.shape[-1]
        _need_pad = D not in (32, 64, 128, 256) and D < 256
        if _need_pad:
            _pad_to = 128 if D <= 128 else 256
            _pad_size = _pad_to - D
            q = F.pad(q, (0, _pad_size))
            k = F.pad(k, (0, _pad_size))
            v = F.pad(v, (0, _pad_size))
        out_chunks: list[torch.Tensor] = []
        for ci in range(len(chunk_boundaries) - 1):
            c_start = chunk_boundaries[ci]
            c_end = chunk_boundaries[ci + 1]
            q_chunk = q[:, :, c_start * S : c_end * S, :]
            out_chunk = F.scaled_dot_product_attention(
                q_chunk,
                k[:, :, : c_end * S, :],
                v[:, :, : c_end * S, :],
            )
            out_chunks.append(out_chunk)
        out = torch.cat(out_chunks, dim=2)
        if _need_pad:
            out = out[..., :D]
        return out

    # Standard path: full SDPA (all cached tokens are causally prior).
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


@ATTENTION_BLOCKS.register_module()
class CachedChunkCausalGDN(ChunkCausalGDN):
    """Cached chunk-causal GDN for streaming inference.

    Runs a state-based cached forward scan (fused Triton kernels) on each
    incoming chunk and updates ``kv_cache`` in place.  Streaming inference
    only — raises if ``kv_cache`` is not provided.
    """

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        chunk_size: int | None = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: list[int] | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, list]:
        from diffusion.model.ops.fused_streaming import _cached_gdn_forward_triton

        if kwargs.get("kv_cache", None) is None:
            raise RuntimeError("CachedChunkCausalGDN requires kv_cache to be provided " "(streaming inference only).")
        del mask, block_mask, chunk_split_strategy, chunk_index
        return _cached_gdn_forward_triton(
            self,
            x,
            HW=HW,
            rotary_emb=rotary_emb,
            apply_output_gate=apply_output_gate,
            **kwargs,
        )


@ATTENTION_BLOCKS.register_module()
class CachedChunkCausalSoftmaxAttn(ChunkCausalSoftmaxAttn):
    """Cached chunk-causal softmax attention for streaming inference.

    Caches post-RoPE K, V from past chunks; prepends cached K, V to the
    current chunk for full-history SDPA (cache enforces causality so no mask
    is required).  Falls back to the parent's non-cached forward when
    ``kv_cache`` is absent.
    """

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        chunk_size: int | None = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: list[int] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        from diffusion.model.ops.fused_streaming import (
            _SLOT_FWD_KV,
            _SLOT_FWD_Z,
            _SLOT_TYPE_FLAG,
            _TYPE_CONCAT,
            _slice_rope_to_current_chunk,
        )

        kv_cache = kwargs.get("kv_cache", None)
        save_kv_cache = kwargs.get("save_kv_cache", False)

        if kv_cache is None:
            return super().forward(
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                apply_output_gate=apply_output_gate,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
                **kwargs,
            )

        del mask, block_mask
        if HW is None:
            raise ValueError("HW (T, H, W) must be provided.")

        B, N, C = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp

        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        # RoPE: upstream rope may cover sink + current under sink_token=True;
        # cached K is already post-rope so only the current chunk needs rotation.
        if rotary_emb is not None:
            q_perm = q.permute(0, 2, 3, 1)  # (B, H, D, N)
            k_perm = k.permute(0, 2, 3, 1)
            rotary_emb_cur = _slice_rope_to_current_chunk(rotary_emb, q_perm.shape[-1])
            q_perm = GDN._apply_rotary_emb(q_perm, rotary_emb_cur)
            k_perm = GDN._apply_rotary_emb(k_perm, rotary_emb_cur)
            q = q_perm.permute(0, 3, 1, 2)
            k = k_perm.permute(0, 3, 1, 2)

        # (B, N, H, D) -> (B, H, N, D) for SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.reshape(B, N, self.heads, self.dim).transpose(1, 2)

        dtype_orig = x.dtype
        if q.dtype == torch.float32:
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        # Read cached K, V before overwriting; save current chunk's K, V.
        cached_k = kv_cache[_SLOT_FWD_KV]
        cached_v = kv_cache[_SLOT_FWD_Z]
        if save_kv_cache:
            kv_cache[_SLOT_FWD_KV] = k.detach().clone()
            kv_cache[_SLOT_FWD_Z] = v.detach().clone()
            kv_cache[_SLOT_TYPE_FLAG] = _TYPE_CONCAT
        if cached_k is not None:
            k = torch.cat([cached_k.to(k.dtype), k], dim=2)
            v = torch.cat([cached_v.to(v.dtype), v], dim=2)

        # Cache enforces chunk causality; no in-forward mask needed.
        out = _sdpa_maybe_chunk_causal(
            q,
            k,
            v,
            need_chunk_mask=False,
            T=T,
            S=S,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            chunk_split_strategy=chunk_split_strategy,
            device=x.device,
        )

        if out.dtype != dtype_orig:
            out = out.to(dtype_orig)
        out = out.transpose(1, 2).reshape(B, N, C)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(dtype_orig))
        return out, kv_cache
