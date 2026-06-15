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

"""V2V attention blocks used by the active LTX2 configs.

This file keeps the active v2v attention modules self-contained. Parameter
names intentionally match the historical implementations so existing
checkpoints can be loaded unchanged.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusion.model.norms import RMSNorm
from diffusion.model.ops import (
    _precompute_inv_rms,
    _prepare_fused_gdn_inputs,
    _resolve_gdn_variant,
    fused_bidi_merge,
    fused_bidi_stateful_chunkwise_shared_phase_a,
    prepare_rope_tables,
)
from diffusion.model.registry import ATTENTION_BLOCKS
from diffusion.utils.import_utils import get_flash_attn_func, is_flash_attn_available, is_xformers_available

_xformers_available = False if os.environ.get("DISABLE_XFORMERS", "0") == "1" else is_xformers_available()
if _xformers_available:
    import xformers.ops

_flash_attn_available = False if os.environ.get("DISABLE_FLASH_ATTN", "0") == "1" else is_flash_attn_available()
flash_attn_func = get_flash_attn_func() if _flash_attn_available else None


def flip_and_shift(x: torch.Tensor, dim: int = 2, shift_val: float = 0.0) -> torch.Tensor:
    x_flip = torch.flip(x, dims=[dim])
    x_shifted = x_flip.narrow(dim, 0, x.shape[dim] - 1)
    pad_shape = list(x.shape)
    pad_shape[dim] = 1
    padding = torch.full(pad_shape, shift_val, device=x.device, dtype=x.dtype)
    return torch.cat([padding, x_shifted], dim=dim)


def _merge_gdn_num_den(
    num_fwd: torch.Tensor,
    num_bwd: torch.Tensor | None,
    den_fwd: torch.Tensor,
    den_bwd: torch.Tensor | None,
    eps: float,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    assert (num_bwd is None) == (den_bwd is None), "num_bwd/den_bwd must both be None or both provided"
    num = num_fwd if num_bwd is None else num_fwd + num_bwd
    den = den_fwd if den_bwd is None else den_fwd + den_bwd
    out = num.float() / (den.permute(0, 2, 1).unsqueeze(-1).float() + float(eps))
    if gate is not None:
        out = out * F.silu(gate.float())
    return out.to(torch.float32 if num_fwd.dtype == torch.float32 else num_fwd.dtype)


def _get_cache_frame_index(cache_frame_indices, *, T: int) -> int | None:
    if cache_frame_indices is None:
        return None
    assert isinstance(cache_frame_indices, torch.Tensor), "cache_frame_indices must be a tensor or None"
    assert (
        int(cache_frame_indices.numel()) == 1
    ), f"cache_frame_indices must contain exactly one frame, got {cache_frame_indices.tolist()}"
    frame_idx = int(cache_frame_indices.item())
    assert 0 <= frame_idx < int(T), f"cache_frame_indices={frame_idx} out of local frame range [0, {int(T)})"
    return frame_idx


def _select_token_frame(tokens: torch.Tensor, frame_idx: int | None, *, S: int, dim: int) -> torch.Tensor:
    if frame_idx is None:
        return tokens
    return tokens.narrow(dim, int(frame_idx) * int(S), int(S))


def torch_recurrent_sana_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    recall_gate,
    eps: float = 1e-6,
    return_components: bool = False,
):
    """Frame-wise Gated Delta Net recurrence.

    Args:
        q/k/v/q_rot/k_rot: Tensors with shape (B, heads, head_dim, T*S).
        beta: Update gate with shape (B, T, S, heads).
        decay: Decay gate with shape (B, heads, T).
        recall_gate: Kept for checkpoint/API compatibility.
        eps: Stabilizer for the denominator.
        return_components: Return numerator and denominator separately.
    """
    del recall_gate

    B, num_heads, head_dim, token_count = q.shape
    frame_count = beta.shape[1]
    spatial_tokens = token_count // frame_count

    def to_frame_seq(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(B, num_heads, head_dim, frame_count, spatial_tokens).permute(0, 1, 3, 2, 4)

    q = to_frame_seq(q)
    k = to_frame_seq(k)
    v = to_frame_seq(v)
    q_rot = to_frame_seq(q_rot)
    k_rot = to_frame_seq(k_rot)

    beta = rearrange(beta, "b t s h -> b h t 1 s")
    decay = decay.view(B, num_heads, frame_count, 1, 1)

    state_kv = torch.zeros(B, num_heads, head_dim, head_dim, device=q.device, dtype=q.dtype)
    state_z = torch.zeros(B, num_heads, head_dim, 1, device=q.device, dtype=q.dtype)

    num_list = []
    den_list = []

    for frame_idx in range(frame_count):
        q_t = q[:, :, frame_idx]
        k_t = k[:, :, frame_idx]
        v_t = v[:, :, frame_idx]
        q_rot_t = q_rot[:, :, frame_idx]
        k_rot_t = k_rot[:, :, frame_idx]
        beta_t = beta[:, :, frame_idx]
        decay_t = decay[:, :, frame_idx]

        state_kv = state_kv * decay_t
        state_z = state_z * decay_t

        v_pred = torch.matmul(state_kv, k_rot_t)
        delta_v = (v_t - v_pred) * beta_t
        state_kv = state_kv + torch.matmul(delta_v, k_rot_t.transpose(-1, -2))

        z_pred = torch.matmul(state_z.transpose(-1, -2), k_t)
        delta_z = (1.0 - z_pred) * beta_t
        state_z = state_z + torch.matmul(k_t, delta_z.transpose(-1, -2))

        num_list.append(torch.matmul(state_kv, q_rot_t))
        den_list.append(torch.matmul(state_z.transpose(-1, -2), q_t))

    def restore_shape(tensor: torch.Tensor, output_dim: int) -> torch.Tensor:
        return tensor.permute(0, 1, 3, 2, 4).reshape(B, num_heads, output_dim, token_count)

    final_num = restore_shape(torch.stack(num_list, dim=2), head_dim)
    final_den = restore_shape(torch.stack(den_list, dim=2), 1)

    if return_components:
        return final_num, final_den
    return final_num / (final_den + eps)


def torch_recurrent_sana_gdn_stateful(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    recall_gate,
    eps: float = 1e-6,
    return_components: bool = False,
    state_kv_init: torch.Tensor | None = None,
    state_z_init: torch.Tensor | None = None,
    return_final_state: bool = False,
    return_state_at_frame: int | None = None,
):
    del recall_gate
    B, H, D, N = q.shape
    T = beta.shape[1]
    S = N // T

    def to_frame_seq(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q = to_frame_seq(q)
    k = to_frame_seq(k)
    v = to_frame_seq(v)
    q_rot = to_frame_seq(q_rot)
    k_rot = to_frame_seq(k_rot)

    beta = rearrange(beta, "b t s h -> b h t 1 s")
    decay = decay.view(B, H, T, 1, 1)

    state_kv = (
        state_kv_init.clone() if state_kv_init is not None else torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    )
    state_z = (
        state_z_init.clone() if state_z_init is not None else torch.zeros(B, H, D, 1, device=q.device, dtype=q.dtype)
    )

    num_list = []
    den_list = []
    state_kv_selected = None
    state_z_selected = None

    for frame_idx in range(T):
        q_t = q[:, :, frame_idx]
        k_t = k[:, :, frame_idx]
        v_t = v[:, :, frame_idx]
        q_rot_t = q_rot[:, :, frame_idx]
        k_rot_t = k_rot[:, :, frame_idx]
        beta_t = beta[:, :, frame_idx]
        decay_t = decay[:, :, frame_idx]

        state_kv = state_kv * decay_t
        state_z = state_z * decay_t

        v_pred = torch.matmul(state_kv, k_rot_t)
        delta_v = (v_t - v_pred) * beta_t
        state_kv = state_kv + torch.matmul(delta_v, k_rot_t.transpose(-1, -2))

        z_pred = torch.matmul(state_z.transpose(-1, -2), k_t)
        delta_z = (1.0 - z_pred) * beta_t
        state_z = state_z + torch.matmul(k_t, delta_z.transpose(-1, -2))

        num_list.append(torch.matmul(state_kv, q_rot_t))
        den_list.append(torch.matmul(state_z.transpose(-1, -2), q_t))
        if return_state_at_frame is not None and int(frame_idx) == int(return_state_at_frame):
            state_kv_selected = state_kv.clone()
            state_z_selected = state_z.clone()

    def restore_shape(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        return tensor.permute(0, 1, 3, 2, 4).reshape(B, H, target_dim, N)

    final_num = restore_shape(torch.stack(num_list, dim=2), D)
    final_den = restore_shape(torch.stack(den_list, dim=2), 1)
    result = (final_num, final_den) if return_components else final_num / (final_den + eps)

    if return_final_state:
        if return_state_at_frame is not None:
            if state_kv_selected is None or state_z_selected is None:
                raise ValueError(f"return_state_at_frame={return_state_at_frame} out of range for T={T}")
            return result, (state_kv_selected, state_z_selected)
        return result, (state_kv, state_z)
    return result


@ATTENTION_BLOCKS.register_module()
class V2VBiGDNAttention(nn.Module):
    """Bidirectional GDN attention with 1D q/k projections over frames."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        eps: float = 1e-8,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
        use_output_gate: bool = True,
        update_rule_func: str = "torch_recurrent_sana_gdn",
        t_conv_kernel_size: int = 3,
        **kwargs: object,
    ) -> None:
        del t_conv_kernel_size, kwargs
        super().__init__()

        heads = heads or int(out_dim // dim * heads_ratio)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps
        self.scale = self.dim**-0.5

        self.kernel_func = nn.ReLU(inplace=False)
        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.beta_proj = nn.Linear(in_dim, heads, bias=True)
        self.gate_proj = nn.Linear(in_dim, heads, bias=True)

        A = torch.empty(self.heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(torch.rand(self.heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.use_output_gate = use_output_gate
        self.output_gate = nn.Linear(in_dim, out_dim, bias=True) if use_output_gate else None

        if update_rule_func != "torch_recurrent_sana_gdn":
            raise ValueError(f"Unsupported update rule function: {update_rule_func}")
        self.update_rule_func = torch_recurrent_sana_gdn

        self.gdn_variant = _resolve_gdn_variant()
        self.use_fused_gdn = self.gdn_variant != "pytorch"

        self.q = nn.Conv1d(in_dim, in_dim, kernel_size=1, padding=0, bias=use_bias)
        self.k = nn.Conv1d(in_dim, in_dim, kernel_size=1, padding=0, bias=use_bias)
        self.v = nn.Linear(in_dim, in_dim, bias=use_bias)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def _init_gdn_gates_for_linear_equiv(self) -> None:
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 5.0)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        with torch.no_grad():
            self.dt_bias.fill_(-8.0)
            self.A_log.fill_(math.log(1.0))
        if self.output_gate is not None:
            nn.init.zeros_(self.output_gate.weight)
            nn.init.ones_(self.output_gate.bias)

    def _apply_rotary_emb(self, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x_rotated = torch.view_as_complex(
            hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)).contiguous()
        )
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return x_out.type_as(hidden_states)

    def _compute_frame_gates(self, x: torch.Tensor, hw: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, C = x.shape
        T, H, W = hw
        S = H * W
        x_frame = x.reshape(B, T, S, C)

        beta = self.beta_proj(x_frame).sigmoid().to(x.dtype)

        a_out = self.gate_proj(x_frame.mean(dim=2)).float()
        dt_bias = self.dt_bias.float().view(1, 1, -1)
        A_val = self.A_log.float().exp().view(1, 1, -1)
        decay_log = -A_val * F.softplus(a_out + dt_bias)
        decay = decay_log.exp().transpose(1, 2).to(x.dtype)

        return beta, decay

    def _fused_statecached_forward(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        kv_cache,
        save_kv_cache: bool,
    ):
        prep = _prepare_fused_gdn_inputs(self, x, HW)
        B, N, C = prep.B, prep.N, prep.C
        T, S = prep.T, prep.S
        H, D = prep.H, prep.D
        qkv, beta, decay = prep.qkv, prep.beta_p, prep.decay
        k_scale = prep.k_scale
        q_nw, k_nw = prep.q_nw, prep.k_nw

        rotary_emb_local = rotary_emb[:, :, -N:] if rotary_emb is not None else None
        rope_cos, rope_sin = prepare_rope_tables(rotary_emb_local, N, D, x.device)

        q_inv_rms = _precompute_inv_rms(qkv, 0, C, 1e-5)
        k_inv_rms = _precompute_inv_rms(qkv, 1, C, 1e-5)

        state_kv_init = kv_cache[0] if kv_cache is not None else None
        state_z_init = kv_cache[1] if kv_cache is not None else None

        if self.gdn_variant != "chunkwise":
            raise RuntimeError("The public V2V fused path only supports USE_CHUNKWISE_GDN=1.")

        num_fwd, den_fwd, state_kv_final, state_z_final = fused_bidi_stateful_chunkwise_shared_phase_a(
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
            init_state_kv=state_kv_init,
            init_state_z=state_z_init,
        )
        if kv_cache is not None and save_kv_cache:
            kv_cache[0] = state_kv_final.detach().clone()
            kv_cache[1] = state_z_final.detach().clone()
        num_bwd, den_bwd = None, None

        if self.output_gate is not None:
            gate_bnhd = self.output_gate(x).reshape(B, N, H, D)
        else:
            gate_bnhd = None

        out = fused_bidi_merge(num_fwd, num_bwd, den_fwd, den_bwd, self.eps, gate=gate_bnhd)
        out = out.reshape(B, N, C).to(x.dtype)

        out = self.proj(out)
        if kv_cache is not None:
            return out, kv_cache
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del mask, block_mask, kwargs
        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for V2VBiGDNAttention.")

        B, N, C = x.shape
        T, H, W = HW
        S = H * W

        if self.use_fused_gdn:
            return self._fused_statecached_forward(x, HW, rotary_emb, None, False)

        x_qk = rearrange(x.reshape(B, T, S, C), "b t s c -> (b s) c t")
        q = rearrange(self.q(x_qk), "(b s) c t -> b (t s) c", b=B, s=S)
        k = rearrange(self.k(x_qk), "(b s) c t -> b (t s) c", b=B, s=S)
        v = self.v(x)

        dtype = q.dtype
        q = self.kernel_func(self.q_norm(q).transpose(-1, -2).reshape(B, self.heads, self.dim, N))
        k = self.kernel_func(self.k_norm(k).transpose(-1, -2).reshape(B, self.heads, self.dim, N))
        v = v.transpose(-1, -2).reshape(B, self.heads, self.dim, N)

        k_scale = (self.dim**-0.5) * (S**-0.5)
        k = k * k_scale

        if rotary_emb is not None:
            q_rot = self._apply_rotary_emb(q, rotary_emb)
            k_rot = self._apply_rotary_emb(k, rotary_emb)
        else:
            q_rot = q
            k_rot = k

        beta, decay = self._compute_frame_gates(x, HW)

        dtype_orig = x.dtype
        if getattr(self, "fp32_attention", False):
            q = q.float()
            k = k.float()
            v = v.float()
            q_rot = q_rot.float()
            k_rot = k_rot.float()
            beta = beta.float()
            decay = decay.float()

        num_fwd, den_fwd = self.update_rule_func(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            recall_gate=1,
            eps=self.eps,
            return_components=True,
        )

        def to_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(B, self.heads, self.dim, T, S).permute(0, 1, 3, 2, 4)

        def from_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 1, 3, 2, 4).reshape(B, self.heads, self.dim, N)

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
        beta_bwd = flip_and_shift(beta, dim=1, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        num_bwd_flipped, den_bwd_flipped = self.update_rule_func(
            from_time_structure(q_bwd),
            from_time_structure(k_bwd),
            from_time_structure(v_bwd),
            from_time_structure(q_rot_bwd),
            from_time_structure(k_rot_bwd),
            beta_bwd,
            decay_bwd,
            recall_gate=1,
            eps=self.eps,
            return_components=True,
        )

        def flip_back(tensor: torch.Tensor) -> torch.Tensor:
            actual_dim = tensor.shape[2]
            t_struct = tensor.view(B, self.heads, actual_dim, T, S)
            return torch.flip(t_struct, dims=[3]).reshape(B, self.heads, actual_dim, N)

        out = (num_fwd + flip_back(num_bwd_flipped)) / (den_fwd + flip_back(den_bwd_flipped) + self.eps)

        if getattr(self, "fp32_attention", False) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        out = out.permute(0, 3, 1, 2).reshape(B, N, C)
        if self.output_gate is not None:
            out = out * F.silu(self.output_gate(x).to(torch.float32)).to(dtype)
        return self.proj(out)


@ATTENTION_BLOCKS.register_module()
class V2VStateCachedBiGDNAttention(nn.Module):
    """Fixed-RoPE bidirectional GDN with cumulative recurrent-state cache."""

    fixed_rope_cache_type = "state"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        eps: float = 1e-8,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
        use_output_gate: bool = True,
        update_rule_func: str = "torch_recurrent_sana_gdn",
        t_conv_kernel_size: int = 3,
        **kwargs: object,
    ) -> None:
        del t_conv_kernel_size, kwargs
        super().__init__()

        heads = heads or int(out_dim // dim * heads_ratio)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps
        self.scale = self.dim**-0.5

        self.kernel_func = nn.ReLU(inplace=False)
        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.beta_proj = nn.Linear(in_dim, heads, bias=True)
        self.gate_proj = nn.Linear(in_dim, heads, bias=True)

        A = torch.empty(self.heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(torch.rand(self.heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.use_output_gate = use_output_gate
        self.output_gate = nn.Linear(in_dim, out_dim, bias=True) if use_output_gate else None

        if update_rule_func != "torch_recurrent_sana_gdn":
            raise ValueError(f"Unsupported update rule function: {update_rule_func}")
        self.update_rule_func = torch_recurrent_sana_gdn

        self.gdn_variant = _resolve_gdn_variant()
        self.use_fused_gdn = self.gdn_variant != "pytorch"

        self.q = nn.Conv1d(in_dim, in_dim, kernel_size=1, padding=0, bias=use_bias)
        self.k = nn.Conv1d(in_dim, in_dim, kernel_size=1, padding=0, bias=use_bias)
        self.v = nn.Linear(in_dim, in_dim, bias=use_bias)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def _init_gdn_gates_for_linear_equiv(self) -> None:
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 5.0)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        with torch.no_grad():
            self.dt_bias.fill_(-8.0)
            self.A_log.fill_(math.log(1.0))
        if self.output_gate is not None:
            nn.init.zeros_(self.output_gate.weight)
            nn.init.ones_(self.output_gate.bias)

    def _apply_rotary_emb(self, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x_rotated = torch.view_as_complex(
            hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)).contiguous()
        )
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return x_out.type_as(hidden_states)

    def _compute_frame_gates(self, x: torch.Tensor, hw: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, C = x.shape
        T, H, W = hw
        S = H * W
        x_frame = x.reshape(B, T, S, C)
        beta = self.beta_proj(x_frame).sigmoid().to(x.dtype)

        a_out = self.gate_proj(x_frame.mean(dim=2)).float()
        dt_bias = self.dt_bias.float().view(1, 1, -1)
        A_val = self.A_log.float().exp().view(1, 1, -1)
        decay_log = -A_val * F.softplus(a_out + dt_bias)
        decay = decay_log.exp().transpose(1, 2).to(x.dtype)
        return beta, decay

    def _fused_statecached_forward(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        kv_cache,
        save_kv_cache: bool,
    ):
        prep = _prepare_fused_gdn_inputs(self, x, HW)
        B, N, C = prep.B, prep.N, prep.C
        T, S = prep.T, prep.S
        H, D = prep.H, prep.D
        qkv, beta, decay = prep.qkv, prep.beta_p, prep.decay
        k_scale = prep.k_scale
        q_nw, k_nw = prep.q_nw, prep.k_nw

        rotary_emb_local = rotary_emb[:, :, -N:] if rotary_emb is not None else None
        rope_cos, rope_sin = prepare_rope_tables(rotary_emb_local, N, D, x.device)

        q_inv_rms = _precompute_inv_rms(qkv, 0, C, 1e-5)
        k_inv_rms = _precompute_inv_rms(qkv, 1, C, 1e-5)

        state_kv_init = kv_cache[0] if kv_cache is not None else None
        state_z_init = kv_cache[1] if kv_cache is not None else None

        if self.gdn_variant != "chunkwise":
            raise RuntimeError("The public V2V fused path only supports USE_CHUNKWISE_GDN=1.")

        num_fwd, den_fwd, state_kv_final, state_z_final = fused_bidi_stateful_chunkwise_shared_phase_a(
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
            init_state_kv=state_kv_init,
            init_state_z=state_z_init,
        )
        if kv_cache is not None and save_kv_cache:
            kv_cache[0] = state_kv_final.detach().clone()
            kv_cache[1] = state_z_final.detach().clone()
        num_bwd, den_bwd = None, None

        gate_bnhd = self.output_gate(x).reshape(B, N, H, D) if self.output_gate is not None else None
        out = fused_bidi_merge(num_fwd, num_bwd, den_fwd, den_bwd, self.eps, gate=gate_bnhd)
        out = out.reshape(B, N, C).to(x.dtype)

        out = self.proj(out)
        if kv_cache is not None:
            return out, kv_cache
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: list[int] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del mask, block_mask, chunk_size, chunk_split_strategy, chunk_index
        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for V2VStateCachedBiGDNAttention.")

        B, N, C = x.shape
        T, H, W = HW
        S = H * W

        kv_cache = kwargs.get("kv_cache", None)
        save_kv_cache = kwargs.get("save_kv_cache", False)
        cache_frame_index = _get_cache_frame_index(kwargs.get("cache_frame_indices", None), T=T)

        if self.use_fused_gdn and T > 1 and cache_frame_index is None:
            return self._fused_statecached_forward(x, HW, rotary_emb, kv_cache, save_kv_cache)

        x_qk = rearrange(x.reshape(B, T, S, C), "b t s c -> (b s) c t")
        q = rearrange(self.q(x_qk), "(b s) c t -> b (t s) c", b=B, s=S)
        k = rearrange(self.k(x_qk), "(b s) c t -> b (t s) c", b=B, s=S)
        v = self.v(x)

        dtype = q.dtype
        q = self.kernel_func(self.q_norm(q).transpose(-1, -2).reshape(B, self.heads, self.dim, N))
        k = self.kernel_func(self.k_norm(k).transpose(-1, -2).reshape(B, self.heads, self.dim, N))
        v = v.transpose(-1, -2).reshape(B, self.heads, self.dim, N)

        beta, decay = self._compute_frame_gates(x, HW)
        state_kv_init = kv_cache[0] if kv_cache is not None else None
        state_z_init = kv_cache[1] if kv_cache is not None else None

        k_scale = (self.dim**-0.5) * (S**-0.5)
        k = k * k_scale

        if rotary_emb is not None:
            rotary_emb_local = rotary_emb[:, :, -N:]
            q_rot = self._apply_rotary_emb(q, rotary_emb_local)
            k_rot = self._apply_rotary_emb(k, rotary_emb_local)
        else:
            q_rot = q
            k_rot = k

        dtype_orig = x.dtype
        use_fp32_attention = getattr(self, "fp32_attention", False)
        if use_fp32_attention:
            q = q.float()
            k = k.float()
            v = v.float()
            q_rot = q_rot.float()
            k_rot = k_rot.float()
            beta = beta.float()
            decay = decay.float()
            if state_kv_init is not None:
                state_kv_init = state_kv_init.float()
                state_z_init = state_z_init.float()

        (num_fwd, den_fwd), (state_kv_final, state_z_final) = torch_recurrent_sana_gdn_stateful(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            recall_gate=1,
            eps=self.eps,
            return_components=True,
            state_kv_init=state_kv_init,
            state_z_init=state_z_init,
            return_final_state=True,
            return_state_at_frame=cache_frame_index,
        )

        if kv_cache is not None and save_kv_cache:
            kv_cache[0] = state_kv_final.detach().clone()
            kv_cache[1] = state_z_final.detach().clone()

        def to_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(B, self.heads, self.dim, T, S).permute(0, 1, 3, 2, 4)

        def from_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 1, 3, 2, 4).reshape(B, self.heads, self.dim, N)

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
        beta_bwd = flip_and_shift(beta, dim=1, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        num_bwd_flipped, den_bwd_flipped = self.update_rule_func(
            from_time_structure(q_bwd),
            from_time_structure(k_bwd),
            from_time_structure(v_bwd),
            from_time_structure(q_rot_bwd),
            from_time_structure(k_rot_bwd),
            beta_bwd,
            decay_bwd,
            recall_gate=1,
            eps=self.eps,
            return_components=True,
        )

        def flip_back(tensor: torch.Tensor) -> torch.Tensor:
            actual_dim = tensor.shape[2]
            t_struct = tensor.view(B, self.heads, actual_dim, T, S)
            return torch.flip(t_struct, dims=[3]).reshape(B, self.heads, actual_dim, N)

        out = (num_fwd + flip_back(num_bwd_flipped)) / (den_fwd + flip_back(den_bwd_flipped) + self.eps)

        if use_fp32_attention and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        out = out.permute(0, 3, 1, 2).reshape(B, N, C)
        if self.output_gate is not None:
            out = out * F.silu(self.output_gate(x).to(torch.float32)).to(dtype)

        out = self.proj(out)
        if kv_cache is not None:
            return out, kv_cache
        return out


@ATTENTION_BLOCKS.register_module()
class V2VAfterRoPEGatedSoftmaxAttention(nn.Module):
    """Fixed-RoPE softmax attention with after-RoPE Q/K/V cache."""

    fixed_rope_cache_type = "softmax"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        eps: float = 1e-8,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
        use_output_gate: bool = True,
        update_rule_func: str = "torch_recurrent_sana_gdn",
        t_conv_kernel_size: int = 3,
        **kwargs: object,
    ) -> None:
        del eps, update_rule_func, kwargs
        super().__init__()

        heads = heads or int(out_dim // dim * heads_ratio)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.scale = self.dim**-0.5

        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.use_output_gate = use_output_gate
        self.output_gate = nn.Linear(in_dim, out_dim, bias=True) if use_output_gate else None

        self.q = nn.Conv1d(
            in_dim, in_dim, kernel_size=t_conv_kernel_size, padding=t_conv_kernel_size // 2, bias=use_bias
        )
        self.k = nn.Conv1d(
            in_dim, in_dim, kernel_size=t_conv_kernel_size, padding=t_conv_kernel_size // 2, bias=use_bias
        )
        self.v = nn.Linear(in_dim, in_dim, bias=use_bias)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def _init_gdn_gates_for_linear_equiv(self) -> None:
        if self.output_gate is not None:
            nn.init.zeros_(self.output_gate.weight)
            nn.init.ones_(self.output_gate.bias)

    @staticmethod
    def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x_rotated = torch.view_as_complex(
            hidden_states.transpose(1, 2).to(torch.float64).unflatten(3, (-1, 2)).contiguous()
        )
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).transpose(1, 2)
        return x_out.type_as(hidden_states)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del mask, block_mask
        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for V2VAfterRoPEGatedSoftmaxAttention.")

        B, N, C = x.shape
        T, H, W = HW
        S = H * W

        kv_cache = kwargs.get("kv_cache", None)
        save_kv_cache = kwargs.get("save_kv_cache", False)
        cache_frame_index = _get_cache_frame_index(kwargs.get("cache_frame_indices", None), T=T)

        x_qk = rearrange(x.reshape(B, T, S, C), "b t s c -> (b s) c t")

        padding_size = self.q.kernel_size[0] // 2
        center_q_weight = self.q.weight[:, :, padding_size : padding_size + 1]
        q = F.conv1d(
            x_qk,
            center_q_weight,
            bias=self.q.bias,
            stride=self.q.stride,
            padding=0,
            dilation=self.q.dilation,
            groups=self.q.groups,
        )
        q = rearrange(q, "(b s) c t -> b (t s) c", b=B, s=S)

        padding_size = self.k.kernel_size[0] // 2
        center_k_weight = self.k.weight[:, :, padding_size : padding_size + 1]
        k = F.conv1d(
            x_qk,
            center_k_weight,
            bias=self.k.bias,
            stride=self.k.stride,
            padding=0,
            dilation=self.k.dilation,
            groups=self.k.groups,
        )
        k = rearrange(k, "(b s) c t -> b (t s) c", b=B, s=S)
        v = self.v(x)

        dtype = q.dtype
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.heads, self.dim)
        k = k.reshape(B, N, self.heads, self.dim)
        v = v.reshape(B, N, self.heads, self.dim)

        if rotary_emb is not None:
            rotary_emb_current = rotary_emb[:, :, -N:]
            q = self._apply_rotary_emb(q, rotary_emb_current)
            k = self._apply_rotary_emb(k, rotary_emb_current)

        q = rearrange(q, "b n h c -> b h c n")
        k = rearrange(k, "b n h c -> b h c n")
        v = rearrange(v, "b n h c -> b h c n")

        if kv_cache is not None:
            previous_q, previous_k, previous_v = kv_cache[0], kv_cache[1], kv_cache[2]
            if save_kv_cache:
                kv_cache[0] = _select_token_frame(q, cache_frame_index, S=S, dim=-1).detach().clone()
                kv_cache[1] = _select_token_frame(k, cache_frame_index, S=S, dim=-1).detach().clone()
                kv_cache[2] = _select_token_frame(v, cache_frame_index, S=S, dim=-1).detach().clone()

            q = torch.cat([previous_q.to(q.device, q.dtype), q], dim=-1) if previous_q is not None else q
            k = torch.cat([previous_k.to(k.device, k.dtype), k], dim=-1) if previous_k is not None else k
            v = torch.cat([previous_v.to(v.device, v.dtype), v], dim=-1) if previous_v is not None else v

        q = rearrange(q, "b h c n -> b n h c")
        k = rearrange(k, "b h c n -> b n h c")
        v = rearrange(v, "b h c n -> b n h c")

        q = q[:, -N:].contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if _flash_attn_available:
            out = flash_attn_func(q, k, v, causal=False)
            if isinstance(out, tuple):
                out = out[0]
        elif _xformers_available:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if hasattr(F, "scaled_dot_product_attention"):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = F.softmax(attn, dim=-1)
                out = attn @ v
            out = out.transpose(1, 2)

        out = out.reshape(B, N, C).to(dtype)
        if self.output_gate is not None:
            out = out * F.silu(self.output_gate(x).to(torch.float32)).to(dtype)

        out = self.proj(out)
        if kv_cache is not None:
            return out, kv_cache
        return out


@ATTENTION_BLOCKS.register_module()
class V2VGatedSoftmaxAttention(nn.Module):
    """Softmax attention block used in the GDN/FA hybrid layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        eps: float = 1e-8,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
        use_output_gate: bool = True,
        update_rule_func: str = "torch_recurrent_sana_gdn",
        t_conv_kernel_size: int = 3,
        **kwargs: object,
    ) -> None:
        del eps, update_rule_func, t_conv_kernel_size, kwargs
        super().__init__()

        heads = heads or int(out_dim // dim * heads_ratio)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.scale = self.dim**-0.5

        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.use_output_gate = use_output_gate
        self.output_gate = nn.Linear(in_dim, out_dim, bias=True) if use_output_gate else None

        self.q = nn.Linear(in_dim, in_dim, bias=use_bias)
        self.k = nn.Linear(in_dim, in_dim, bias=use_bias)
        self.v = nn.Linear(in_dim, in_dim, bias=use_bias)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def _init_gdn_gates_for_linear_equiv(self) -> None:
        if self.output_gate is not None:
            nn.init.zeros_(self.output_gate.weight)
            nn.init.ones_(self.output_gate.bias)

    @staticmethod
    def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x_rotated = torch.view_as_complex(hidden_states.transpose(1, 2).to(torch.float64).unflatten(3, (-1, 2)))
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).transpose(1, 2)
        return x_out.type_as(hidden_states)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del mask, block_mask, kwargs
        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for V2VGatedSoftmaxAttention.")

        B, N, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        dtype = q.dtype
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.heads, self.dim)
        k = k.reshape(B, N, self.heads, self.dim)
        v = v.reshape(B, N, self.heads, self.dim)

        if rotary_emb is not None:
            q = self._apply_rotary_emb(q, rotary_emb)
            k = self._apply_rotary_emb(k, rotary_emb)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if _flash_attn_available:
            out = flash_attn_func(q, k, v, causal=False)
            if isinstance(out, tuple):
                out = out[0]
        elif _xformers_available:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        else:
            if getattr(self, "fp32_attention", False):
                q = q.float()
                k = k.float()
                v = v.float()
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if hasattr(F, "scaled_dot_product_attention"):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = F.softmax(attn, dim=-1)
                out = attn @ v

            out = out.transpose(1, 2)

        out = out.reshape(B, N, C).to(dtype)
        if self.output_gate is not None:
            out = out * F.silu(self.output_gate(x).to(torch.float32)).to(dtype)
        return self.proj(out)
