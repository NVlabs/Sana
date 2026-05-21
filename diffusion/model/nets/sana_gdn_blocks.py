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

_HAS_FLEX_ATTENTION = bool(int(os.environ.get("SANA_USE_FLEX_ATTENTION", "0")))

OUTPUT_GATE_INIT_BIAS = 1.278464542761074  # silu(x)=1.0


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


def torch_recurrent_sana_gdn(q, k, v, q_rot, k_rot, beta, decay, recall_gate, eps=1e-6, return_components=False):
    """Apply the frame-wise Gated Delta Rule.

    The update uses full spatial frames per time step while maintaining
    recurrent KV and Z states.

    Args:
        q: Query tensor of shape (B, H, D, T*S).
        k: Key tensor of shape (B, H, D, T*S).
        v: Value tensor of shape (B, H, D, T*S).
        q_rot: Rotary-embedded queries, same shape as ``q``.
        k_rot: Rotary-embedded keys, same shape as ``k``.
        beta: Update gate of shape (B, H, T) or (B, H, T, S).
        decay: Decay gate of shape (B, H, T).
        recall_gate: Recall scale (broadcasted across batch/time).
        eps: Small constant for numerical stability.

    Returns:
        Output tensor of shape (B, H, D, T*S).
    """
    # Reshape inputs to (B, H, T, D, S).
    B, H, D, N = q.shape
    # beta has shape (B, H, T) or (B, H, T, S); T is always dim=2.
    T = beta.shape[2]
    S = N // T

    target_z = 1.0

    def to_frame_seq(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q = to_frame_seq(q)
    k = to_frame_seq(k)
    v = to_frame_seq(v)
    q_rot = to_frame_seq(q_rot)
    k_rot = to_frame_seq(k_rot)

    # beta: (B, H, T) -> (B, H, T, 1, 1) or (B, H, T, S) -> (B, H, T, 1, S)
    if beta.ndim == 4:
        beta = beta.unsqueeze(3)
    else:
        beta = beta.view(B, H, T, 1, 1)

    decay = decay.view(B, H, T, 1, 1)

    # Scale: (1,) -> (1, 1, 1, 1, 1)
    scale = 1  # recall_gate.view(1, 1, 1, 1)

    state_kv = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    state_z = torch.zeros(B, H, D, 1, device=q.device, dtype=q.dtype)

    num_list = []
    den_list = []

    for t in range(T):
        # Slice
        qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
        qrt, krt = q_rot[:, :, t], k_rot[:, :, t]
        bt, gt = beta[:, :, t], decay[:, :, t]

        # Decay
        state_kv = state_kv * gt
        state_z = state_z * gt

        # KV Update
        v_pred = torch.matmul(state_kv, krt)
        delta_v = (vt - scale * v_pred) * bt
        state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))

        # Z Update
        z_pred = torch.matmul(state_z.transpose(-1, -2), kt)
        delta_z = (target_z - scale * z_pred) * bt
        state_z = state_z + torch.matmul(kt, delta_z.transpose(-1, -2))

        # Output Components
        # num: (B, H, D, S)
        out_num = torch.matmul(state_kv, qrt)
        # den: (B, H, 1, S)
        out_den = torch.matmul(state_z.transpose(-1, -2), qt)

        num_list.append(out_num)
        den_list.append(out_den)

    # 4. Stack & Reshape
    # (B, H, T, D, S)
    num_stacked = torch.stack(num_list, dim=2)
    # (B, H, T, 1, S)
    den_stacked = torch.stack(den_list, dim=2)

    def restore_shape(tensor, target_d):
        # tensor: (B, H, T, d_in, S) -> (B, H, d_in, T*S)
        return tensor.permute(0, 1, 3, 2, 4).reshape(B, H, target_d, N)

    final_num = restore_shape(num_stacked, D)
    final_den = restore_shape(den_stacked, 1)

    if return_components:
        return final_num, final_den

    return final_num / (final_den + eps)


@torch.compile
def torch_chunk_sana_gdn(
    q,
    k,
    v,
    q_rot,
    k_rot,
    beta,
    decay,
    recall_gate=None,
    chunk_size: int | None = 21,
    eps: float = 1e-6,
    return_components: bool = False,
):
    del recall_gate  # Currently unused; kept for API parity.

    B, H, D, N = q.shape
    if beta.ndim not in (3, 4):
        raise ValueError(f"Expected beta.ndim in (3, 4), got {beta.ndim}.")
    T = beta.shape[2]
    if T <= 0:
        raise ValueError(f"Expected T > 0, got T={T}.")
    if N % T != 0:
        raise ValueError(f"Expected N divisible by T, got N={N}, T={T}.")
    S = N // T

    target_z = 1.0
    scale = 1.0

    def to_frame_seq(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q, k, v = to_frame_seq(q), to_frame_seq(k), to_frame_seq(v)
    q_rot, k_rot = to_frame_seq(q_rot), to_frame_seq(k_rot)

    if beta.ndim == 4:
        beta = beta.unsqueeze(3)
    else:
        beta = beta.view(B, H, T, 1, 1)

    decay = decay.view(B, H, T, 1, 1)

    # =========================================================================
    # 1. PARALLEL PRE-PROCESSING
    # =========================================================================

    I = torch.eye(D, device=q.device, dtype=q.dtype).view(1, 1, 1, D, D)

    # KV State Matrices: W = g * (I - c * K @ K^T)
    k_rot_beta = k_rot * beta
    W_kv = decay * (I - scale * torch.matmul(k_rot_beta, k_rot.transpose(-1, -2)))
    U_kv = torch.matmul(v * beta, k_rot.transpose(-1, -2))

    # Z State Matrices: W = g * (I - c * K @ K^T)
    k_beta = k * beta
    W_z = decay * (I - scale * torch.matmul(k_beta, k.transpose(-1, -2)))
    U_z = target_z * k_beta.sum(dim=-1, keepdim=True)  # Equivalent to Kt @ bt^T over spatial dim

    # =========================================================================
    # 2. CHUNKING LOGIC
    # =========================================================================

    valid_chunk_index, _ = normalize_chunk_index(None, T, chunk_size)
    split_sizes = [valid_chunk_index[i + 1] - valid_chunk_index[i] for i in range(len(valid_chunk_index) - 1)]

    W_kv_c = W_kv.split(split_sizes, dim=2)
    U_kv_c = U_kv.split(split_sizes, dim=2)
    W_z_c = W_z.split(split_sizes, dim=2)
    U_z_c = U_z.split(split_sizes, dim=2)

    # =========================================================================
    # 3. FAST INTRA-CHUNK SCAN OVER DxD SPACE
    # =========================================================================

    S_kv = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    S_z = torch.zeros(B, H, D, 1, device=q.device, dtype=q.dtype)

    out_S_kv = []
    out_S_z = []

    def _chunk_scan(w_kv, u_kv, w_z, u_z, s_kv, s_z):
        c_len = w_kv.shape[2]
        s_kv_list, s_z_list = [], []
        for t in range(c_len):
            s_kv = torch.matmul(s_kv, w_kv[:, :, t]) + u_kv[:, :, t]
            s_z = torch.matmul(w_z[:, :, t], s_z) + u_z[:, :, t]
            s_kv_list.append(s_kv)
            s_z_list.append(s_z)
        return torch.stack(s_kv_list, dim=2), s_kv, torch.stack(s_z_list, dim=2), s_z

    for i in range(len(split_sizes)):
        s_kv_all, S_kv, s_z_all, S_z = _chunk_scan(W_kv_c[i], U_kv_c[i], W_z_c[i], U_z_c[i], S_kv, S_z)
        out_S_kv.append(s_kv_all)
        out_S_z.append(s_z_all)

    S_kv_all = torch.cat(out_S_kv, dim=2)
    S_z_all = torch.cat(out_S_z, dim=2)

    # =========================================================================
    # 4. PARALLEL OUTPUT PROJECTION
    # =========================================================================

    out_num = torch.matmul(S_kv_all, q_rot)
    out_den = torch.matmul(S_z_all.transpose(-1, -2), q)

    def restore_shape(tensor, target_d):
        return tensor.permute(0, 1, 3, 2, 4).reshape(B, H, target_d, N)

    final_num = restore_shape(out_num, D)
    final_den = restore_shape(out_den, 1)

    if return_components:
        return final_num, final_den

    return final_num / (final_den + eps)


# ---------------------------------------------------------------------------
# Compiled helpers for hot-path operations (fuses elementwise chains)
# ---------------------------------------------------------------------------

_COMPILE_DISABLE = os.environ.get("GDN_DISABLE_COMPILE", "0") not in ("0", "false")


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
        update_rule_func: str = "torch_chunk_sana_gdn",
        chunk_gdn_chunk_size: int = 21,
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

        if update_rule_func == "torch_recurrent_sana_gdn":
            self.update_rule_func = torch_recurrent_sana_gdn
        elif update_rule_func == "torch_chunk_sana_gdn":
            from functools import partial

            self.update_rule_func = partial(torch_chunk_sana_gdn, chunk_size=chunk_gdn_chunk_size)
        else:
            raise ValueError(f"Unsupported update rule function: {update_rule_func}")

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
        """Apply GDN attention to a token sequence.

        Args:
            x: Input tensor of shape (B, N, C).
            mask: Unused attention mask (kept for API compatibility).
            HW: Tuple of (T, H, W) describing the token layout.
            rotary_emb: Optional rotary embeddings for q/k.
            block_mask: Unused block mask (kept for API compatibility).
            apply_output_gate: When False, return raw attention output
                before output gate and projection.
            **kwargs: Unused extra arguments.

        Returns:
            Tensor of shape (B, N, C) after attention and projection.
        """
        del mask, block_mask
        frame_valid_mask = kwargs.get("frame_valid_mask", None)

        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for GDN attention.")

        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            frame_valid_mask,
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )
        if token_valid_mask is not None:
            x = x * token_valid_mask.view(B, N, 1)

        # Projections.
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        if token_valid_mask is not None:
            token_mask_bnhd = token_valid_mask.view(B, N, 1, 1)
            q = q * token_mask_bnhd
            k = k * token_mask_bnhd
            v = v * token_mask_bnhd

        # Short convolution along T (before norm / kernel activation).
        if self.conv_k is not None:
            if self.conv_q is not None:
                q = self._apply_temporal_short_conv(q.reshape(B, N, C), self.conv_q, HW).reshape(
                    B, N, self.heads, self.dim
                )
            k = self._apply_temporal_short_conv(k.reshape(B, N, C), self.conv_k, HW).reshape(B, N, self.heads, self.dim)
            if self.conv_v is not None:
                v = self._apply_temporal_short_conv(v.reshape(B, N, C), self.conv_v, HW).reshape(
                    B, N, self.heads, self.dim
                )

        # Apply Q/K norm on flattened channels (B, N, C) then reshape to heads.
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        # ReLU kernel.
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        k_scale = self._key_scale(S)
        k = k * k_scale

        # Permute to (B, H, D, N) for processing.
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q = q * token_mask_qkv
            k = k * token_mask_qkv
            v = v * token_mask_qkv

        # RoPE preparation (numerator only).
        if rotary_emb is not None:
            q_rot = self._apply_rotary_emb(q, rotary_emb)
            k_rot = self._apply_rotary_emb(k, rotary_emb)
        else:
            q_rot = q
            k_rot = k
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_rot = q_rot * token_mask_qkv
            k_rot = k_rot * token_mask_qkv

        # Gate computation (use pre-computed gates when available to avoid
        # redundant work in dual-branch CamCtrl models).
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        # Run the frame-wise GDN update.
        # Force FP32 to preserve recurrent stability.
        dtype_orig = x.dtype
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

        out = self.update_rule_func(q, k, v, q_rot, k_rot, beta, decay, recall_gate=recall_gate, eps=self.eps)

        # Reshape and project output.
        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        out = out.permute(0, 3, 1, 2)
        N_out = out.shape[1]
        out = out.reshape(B, N_out, C)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N_out, 1).to(out.dtype)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
            if token_valid_mask is not None:
                out = out * token_valid_mask.view(B, N_out, 1).to(out.dtype)
            return out
        return out


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
        """Apply bidirectional GDN attention to a token sequence.

        Args:
            x: Input tensor of shape (B, N, C).
            mask: Unused attention mask (kept for API compatibility).
            HW: Tuple of (T, H, W) describing the token layout.
            rotary_emb: Optional rotary embeddings for q/k.
            block_mask: Unused block mask (kept for API compatibility).
            **kwargs: Unused extra arguments.

        Returns:
            Tensor of shape (B, N, C) after attention and projection.
        """
        del mask, block_mask
        frame_valid_mask = kwargs.get("frame_valid_mask", None)

        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for GDN attention.")

        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            frame_valid_mask,
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )
        if token_valid_mask is not None:
            x = x * token_valid_mask.view(B, N, 1)

        # Projections.
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        if token_valid_mask is not None:
            token_mask_bnhd = token_valid_mask.view(B, N, 1, 1)
            q = q * token_mask_bnhd
            k = k * token_mask_bnhd
            v = v * token_mask_bnhd

        # Short convolution along T (before norm / kernel activation).
        if self.conv_k is not None:
            if self.conv_q is not None:
                q = self._apply_temporal_short_conv(q.reshape(B, N, C), self.conv_q, HW).reshape(
                    B, N, self.heads, self.dim
                )
            k = self._apply_temporal_short_conv(k.reshape(B, N, C), self.conv_k, HW).reshape(B, N, self.heads, self.dim)
            if self.conv_v is not None:
                v = self._apply_temporal_short_conv(v.reshape(B, N, C), self.conv_v, HW).reshape(
                    B, N, self.heads, self.dim
                )

        # Apply Q/K norm on flattened channels (B, N, C) then reshape to heads.
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        # ReLU kernel.
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        k_scale = self._key_scale(S)
        k = k * k_scale

        # Permute to (B, H, D, N) for processing.
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q = q * token_mask_qkv
            k = k * token_mask_qkv
            v = v * token_mask_qkv

        # RoPE preparation (numerator only).
        if rotary_emb is not None:
            q_rot = self._apply_rotary_emb(q, rotary_emb)
            k_rot = self._apply_rotary_emb(k, rotary_emb)
        else:
            q_rot = q
            k_rot = k
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_rot = q_rot * token_mask_qkv
            k_rot = k_rot * token_mask_qkv

        # Gate computation (use pre-computed gates when available).
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        H_eff = q.shape[1]
        N_eff = q.shape[3]
        T_eff = N_eff // S

        # Run the frame-wise GDN update.
        # Force FP32 to preserve recurrent stability.
        dtype_orig = x.dtype
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

        # Forward pass (inclusive: 1..t).
        num_fwd, den_fwd = self.update_rule_func(
            q, k, v, q_rot, k_rot, beta, decay, recall_gate=recall_gate, eps=self.eps, return_components=True
        )

        # Backward pass (exclusive: t+1..T).
        def to_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(B, H_eff, self.dim, T_eff, S).permute(0, 1, 3, 2, 4)

        def from_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 1, 3, 2, 4).reshape(B, H_eff, self.dim, N_eff)

        q_T = to_time_structure(q)
        k_T = to_time_structure(k)
        v_T = to_time_structure(v)
        q_rot_T = to_time_structure(q_rot)
        k_rot_T = to_time_structure(k_rot)

        q_bwd = torch.flip(q_T, dims=[2])
        q_rot_bwd = torch.flip(q_rot_T, dims=[2])

        k_bwd = flip_and_shift(k_T, dim=2, shift_val=0.0)
        v_bwd = flip_and_shift(v_T, dim=2, shift_val=0.0)
        k_rot_bwd = flip_and_shift(k_rot_T, dim=2, shift_val=0.0)
        beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        k_bwd_flat = from_time_structure(k_bwd)
        v_bwd_flat = from_time_structure(v_bwd)
        q_bwd_flat = from_time_structure(q_bwd)
        q_rot_bwd_flat = from_time_structure(q_rot_bwd)
        k_rot_bwd_flat = from_time_structure(k_rot_bwd)

        num_bwd_flipped, den_bwd_flipped = self.update_rule_func(
            q_bwd_flat,
            k_bwd_flat,
            v_bwd_flat,
            q_rot_bwd_flat,
            k_rot_bwd_flat,
            beta_bwd,
            decay_bwd,
            recall_gate=recall_gate,
            eps=self.eps,
            return_components=True,
        )

        def flip_back(tensor: torch.Tensor) -> torch.Tensor:
            d_actual = tensor.shape[2]
            t_struct = tensor.view(B, H_eff, d_actual, T_eff, S)
            return torch.flip(t_struct, dims=[3]).reshape(B, H_eff, d_actual, N_eff)

        num_bwd = flip_back(num_bwd_flipped)
        den_bwd = flip_back(den_bwd_flipped)

        total_num = num_fwd + num_bwd
        total_den = den_fwd + den_bwd

        out = total_num / (total_den + self.eps)

        # Reshape and project output.
        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        out = out.permute(0, 3, 1, 2)
        N_out = out.shape[1]
        out = out.reshape(B, N_out, C)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N_out, 1).to(out.dtype)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
            if token_valid_mask is not None:
                out = out * token_valid_mask.view(B, N_out, 1).to(out.dtype)
            return out
        return out


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
