# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F

from sana_robot.kernels.inference.ops.dispatch import SM90, SM110, device_architecture
from sana_robot.kernels.inference.ops.qk_norm_rope.portable import SOFTMAX_LAYOUT, fused_qk_norm_rope
from sana_robot.kernels.inference.ops.sigmoid_gate import fused_sigmoid_gate, serves_channels
from sana_robot.kernels.inference.ops.softmax_read import (
    MAX_HEADS,
    NORM_CHANNELS,
    qk_norm_serves,
    rope_read_serves,
    softmax_qk_norm,
    softmax_rope_read,
)
from sana_robot.policy.attention.causal_softmax import CausalSoftmaxAttention
from sana_robot.policy.attention.condition import ConditionSoftmaxAttention
from sana_robot.policy.attention.softmax import rotate


class SoftmaxKernels:
    """The inference kernels of a softmax layer, each in use once its install function has set it up: the fused q / k RMSNorm and rotation (``fused_qk_rope``), the packed qkv and output gate projection with the fused sigmoid gate (the ``packed_qkv_gate_weight`` buffer) and the fused read of a streaming window (``fused_softmax_read``)."""

    fused_qk_rope = False
    fused_softmax_read = False

    def serves(self, x: torch.Tensor) -> bool:
        """Return whether the kernels serve a call: inference on one sample on a CUDA device with bf16 weights, whose projections are bf16."""
        return not self.training and x.is_cuda and x.shape[0] == 1 and self.qkv.weight.dtype == torch.bfloat16

    def packed_qkv_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Return the packed projection [B, N, 4C] of ``x``: the q, k and v projections, then the output gate logits."""
        return F.linear(x, self.packed_qkv_gate_weight, self.packed_qkv_gate_bias)

    def project_unnormalized(self, x: torch.Tensor):
        """Return the q, k and v projections [B, N, C] of ``x`` before normalization and the output gate logits [B, N, C], which are None without the packed projection."""
        (batch, tokens, channels) = x.shape
        if "packed_qkv_gate_weight" not in self._buffers:
            (q, k, v) = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
            return (q, k, v, None)
        packed = self.packed_qkv_gate(x)
        (q, k, v) = packed[..., : 3 * channels].reshape(batch, tokens, 3, channels).unbind(2)
        return (q, k, v, packed[..., 3 * channels :])

    def gate_and_project_packed(self, out, x, dtype, gate_logits) -> torch.Tensor:
        """Gate the attention output [B, N, heads, head_dim] and project it; the gate logits of the packed projection, when given, are applied in one kernel launch."""
        if gate_logits is None:
            return self.gate_and_project(out, x, dtype)
        out = out.reshape(x.shape).to(dtype).contiguous()
        return self.proj(fused_sigmoid_gate(out, gate_logits))


class FusedConditionSoftmaxAttention(SoftmaxKernels, ConditionSoftmaxAttention):
    """``ConditionSoftmaxAttention`` with the installed inference kernels. Training and batches of several samples run the original forward."""

    def forward(self, x, rotary_emb, layout):
        if not self.serves(x):
            return super().forward(x, rotary_emb, layout)
        (batch, tokens, _) = x.shape
        (q, k, v, gate_logits) = self.project_unnormalized(x)
        if self.fused_qk_rope:
            rotated = fused_qk_norm_rope(
                q, k, self.q_norm.weight, self.k_norm.weight, rotary_emb, SOFTMAX_LAYOUT, self.dim, self.q_norm.eps
            )
            (q, k) = (rotated.q_rotated, rotated.k_rotated)
        else:
            q = rotate(self.q_norm(q).reshape(batch, tokens, self.heads, self.dim), rotary_emb)
            k = rotate(self.k_norm(k).reshape(batch, tokens, self.heads, self.dim), rotary_emb)
        v = v.reshape(batch, tokens, self.heads, self.dim)
        out = self._attend_chunks(q, k, v, layout)
        return self.gate_and_project_packed(out, x, v.dtype, gate_logits)


class FusedCausalSoftmaxAttention(SoftmaxKernels, CausalSoftmaxAttention):
    """``CausalSoftmaxAttention`` with the installed inference kernels on a streaming window. The teacher-forcing sequence, a packed row of windows and training run the original forward."""

    def forward(
        self,
        x,
        rotary_emb,
        chunk_bounds=None,
        teacher_forcing_clean_segments=None,
        observation_spans=(),
        kv_cache=None,
        save_kv_cache: bool = False,
        observation_prefill: bool = False,
        window_chunks: int = 1,
        sink_phases=None,
        windows=None,
    ):
        if kv_cache is None or windows is not None or not self.serves(x):
            return super().forward(
                x,
                rotary_emb,
                chunk_bounds=chunk_bounds,
                teacher_forcing_clean_segments=teacher_forcing_clean_segments,
                observation_spans=observation_spans,
                kv_cache=kv_cache,
                save_kv_cache=save_kv_cache,
                observation_prefill=observation_prefill,
                window_chunks=window_chunks,
                sink_phases=sink_phases,
                windows=windows,
            )
        (batch, tokens, _) = x.shape
        (q, k, v, gate_logits) = self.project_unnormalized(x)
        if self.fused_softmax_read and qk_norm_serves(tokens, self.q_norm.eps, self.k_norm.eps):
            (q, k) = softmax_qk_norm(
                q, k, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps, self.k_norm.eps, self.heads
            )
        else:
            q = self.q_norm(q).reshape(batch, tokens, self.heads, self.dim)
            k = self.k_norm(k).reshape(batch, tokens, self.heads, self.dim)
        v = v.reshape(batch, tokens, self.heads, self.dim)
        manager = kv_cache["kv"]
        if observation_prefill:
            (keys, values, valid, phases) = manager.observe(k.detach(), v.detach(), q)
        elif save_kv_cache:
            (keys, values, valid, phases) = manager.commit(k.detach(), v.detach(), q)
        else:
            (keys, values, valid, phases) = manager.read(k, v, q)
        if phases is not None:
            if self.fused_softmax_read and rope_read_serves(q, keys, phases):
                (q, keys) = softmax_rope_read(q, keys, phases)
            else:
                query_phases = phases.narrow(2, keys.shape[1] - q.shape[1], q.shape[1])
                (q, keys) = (rotate(q, query_phases), rotate(keys, phases))
        if valid is None:
            out = self.attend(q, keys, values)
        else:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=valid
            ).transpose(1, 2)
        return self.gate_and_project_packed(out, x, v.dtype, gate_logits)


FUSED_CLASSES = {
    ConditionSoftmaxAttention: FusedConditionSoftmaxAttention,
    CausalSoftmaxAttention: FusedCausalSoftmaxAttention,
}


def softmax_layers(policy, layer_class) -> list:
    """Return the softmax layers of a policy whose class is ``layer_class`` or the class that hosts its kernels."""
    classes = (layer_class, FUSED_CLASSES[layer_class])
    return [block.attn for block in policy.blocks if type(block.attn) in classes]


def host_kernels(attention):
    """Swap the class of a softmax layer in place for the one that hosts its kernels and return the layer; parameters and checkpoint keys do not change."""
    attention.__class__ = FUSED_CLASSES.get(type(attention), type(attention))
    return attention


def install_fused_qk_rope(policy) -> None:
    """Run the query and key RMSNorm and the rotary rotation of every softmax layer of SANA-Robot-base as one kernel launch.

    The kernel equals the policy's RMSNorm and float64 rotation bit for bit. The softmax layers of SANA-Robot-AR are left as they are: a
    streaming window stores its keys before the rotation and rotates every key at the read, which
    ``install_fused_softmax_read`` serves.

    Args:
        policy: The policy on its CUDA device with bf16 weights, in eval mode.
    """
    if device_architecture(next(policy.parameters()).device) is None:
        return
    for attention in softmax_layers(policy, ConditionSoftmaxAttention):
        if attention.dim & (attention.dim - 1) == 0 and attention.q_norm.eps == attention.k_norm.eps:
            host_kernels(attention).fused_qk_rope = True


def install_packed_qkv_gate(policy) -> None:
    """Compute the qkv projection and the output gate logits of every softmax layer as one matrix product and apply the sigmoid gate in one kernel launch, on every architecture.

    Every layer gets the non-persistent buffers ``packed_qkv_gate_weight`` [4C, C] (the rows of ``qkv.weight``: q,
    k, v, then the rows of ``output_gate.weight``) and ``packed_qkv_gate_bias`` [4C] (zeros, then
    ``output_gate.bias``), which its ``packed_qkv_gate`` method multiplies.

    Args:
        policy: The policy on its CUDA device with bf16 weights, in eval mode.
    """
    if device_architecture(next(policy.parameters()).device) is None:
        return
    for layer_class in FUSED_CLASSES:
        for attention in softmax_layers(policy, layer_class):
            if "packed_qkv_gate_weight" in attention._buffers or not serves_channels(attention.proj.in_features):
                continue
            weight = torch.cat([attention.qkv.weight, attention.output_gate.weight], dim=0).detach().contiguous()
            bias = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
            bias[attention.qkv.weight.shape[0] :] = attention.output_gate.bias.detach()
            host_kernels(attention)
            attention.register_buffer("packed_qkv_gate_weight", weight, persistent=False)
            attention.register_buffer("packed_qkv_gate_bias", bias, persistent=False)


def install_fused_softmax_read(policy) -> None:
    """Run the query and key RMSNorm of a streaming window of SANA-Robot-AR and the rotation of its queries and of every key it reads as two kernel launches, on H100 and Jetson Thor.

    Both kernels equal the policy's arithmetic bit for bit. The softmax memory is used as it is: its keys stay
    stored before the rotation and every read rotates all of them.

    Args:
        policy: The policy on its CUDA device with bf16 weights, in eval mode.
    """
    if device_architecture(next(policy.parameters()).device) not in (SM90, SM110):
        return
    for attention in softmax_layers(policy, CausalSoftmaxAttention):
        head_dim = attention.dim
        if (
            attention.proj.in_features == NORM_CHANNELS
            and head_dim & (head_dim - 1) == 0
            and attention.heads <= MAX_HEADS
        ):
            host_kernels(attention).fused_softmax_read = True
