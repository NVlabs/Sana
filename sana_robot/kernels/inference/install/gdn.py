# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from sana_robot.kernels.inference.ops.dispatch import SM90, SM110, SM120, device_architecture
from sana_robot.policy.attention.causal_gdn import CausalGatedDeltaNet
from sana_robot.policy.attention.condition import PREFILL, ConditionGatedDeltaNet, ConditionPass

PACKED_ROW_MULTIPLE = 64


@lru_cache(maxsize=None)
def gdn_read_kernels(device: torch.device):
    """Return the module that holds the Gated DeltaNet read and commit kernels of the device's architecture (``supports``, ``seeded_gdn_read`` and ``seeded_gdn_read_commit``), or None for a device without them."""
    architecture = device_architecture(device)
    if architecture == SM90:
        from sana_robot.kernels.inference.ops.gdn_read import sm90 as kernels
    elif architecture == SM110:
        from sana_robot.kernels.inference.ops.gdn_read import sm110 as kernels
    elif architecture == SM120:
        from sana_robot.kernels.inference.ops.gdn_read import sm120 as kernels
    else:
        return None
    return kernels


@torch.library.custom_op("sana_robot::gdn_causal_seeded_read", mutates_args=())
def gdn_causal_seeded_read(
    qkv: torch.Tensor,
    beta_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    rotary_emb: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    o_norm_weight: torch.Tensor,
    carry_s: torch.Tensor,
    decay: torch.Tensor,
    key_scale: float,
    num_heads: int,
    norm_eps: float,
) -> torch.Tensor:
    """Return the output [N, C] of one chunk that reads the state ``carry_s`` [H, D, D] scaled by ``decay`` [H] together with its own tokens, before the output projection: query and key RMSNorm, rotary phases, the beta sigmoid, the read, the output norm and gate, on the kernels of the device's architecture."""
    return gdn_read_kernels(qkv.device).seeded_gdn_read(
        qkv,
        beta_logits,
        gate_logits,
        rotary_emb,
        q_norm_weight,
        k_norm_weight,
        o_norm_weight,
        carry_s,
        decay,
        key_scale,
        num_heads,
        norm_eps,
    )


@gdn_causal_seeded_read.register_fake
def seeded_read_layout(
    qkv,
    beta_logits,
    gate_logits,
    rotary_emb,
    q_norm_weight,
    k_norm_weight,
    o_norm_weight,
    carry_s,
    decay,
    key_scale,
    num_heads,
    norm_eps,
):
    """Return an empty tensor laid out as the output of ``gdn_causal_seeded_read``, for tracing."""
    return qkv.new_empty((qkv.shape[0], qkv.shape[1] // 3))


@torch.library.custom_op("sana_robot::gdn_causal_seeded_read_commit", mutates_args=())
def gdn_causal_seeded_read_commit(
    qkv: torch.Tensor,
    beta_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    rotary_emb: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    o_norm_weight: torch.Tensor,
    carry_s: torch.Tensor,
    decay: torch.Tensor,
    key_scale: float,
    num_heads: int,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the output of ``gdn_causal_seeded_read`` and the state [H, D, D] after the chunk has written its tokens into the decayed ``carry_s`` by the delta rule, a new float32 tensor."""
    return gdn_read_kernels(qkv.device).seeded_gdn_read_commit(
        qkv,
        beta_logits,
        gate_logits,
        rotary_emb,
        q_norm_weight,
        k_norm_weight,
        o_norm_weight,
        carry_s,
        decay,
        key_scale,
        num_heads,
        norm_eps,
    )


@gdn_causal_seeded_read_commit.register_fake
def seeded_read_commit_layout(
    qkv,
    beta_logits,
    gate_logits,
    rotary_emb,
    q_norm_weight,
    k_norm_weight,
    o_norm_weight,
    carry_s,
    decay,
    key_scale,
    num_heads,
    norm_eps,
):
    """Return empty tensors laid out as the output and the state of ``gdn_causal_seeded_read_commit``, for tracing."""
    return (qkv.new_empty((qkv.shape[0], qkv.shape[1] // 3)), carry_s.new_empty(carry_s.shape))


def pack_input_projection(attention) -> None:
    """Register on a Gated DeltaNet layer the weight ``packed_input_weight`` [rows, C] and the bias ``packed_input_bias`` [rows] of one projection in place of three: the rows of ``qkv`` (3C), ``output_gate`` (C) and ``beta_proj`` (heads) in this order, then zero rows up to a multiple of 64."""
    weight = torch.cat((attention.qkv.weight, attention.output_gate.weight, attention.beta_proj.weight)).detach()
    qkv_bias = weight.new_zeros(attention.qkv.out_features)
    bias = torch.cat((qkv_bias, attention.output_gate.bias, attention.beta_proj.bias)).detach()
    padding = -weight.shape[0] % PACKED_ROW_MULTIPLE
    attention.register_buffer("packed_input_weight", F.pad(weight, (0, 0, 0, padding)), persistent=False)
    attention.register_buffer("packed_input_bias", F.pad(bias, (0, padding)), persistent=False)


class GatedDeltaNetKernels:
    """What the fused Gated DeltaNet layers share: the packed input projection of one chunk and the kernels that read and write the state from it."""

    def kernels_apply(self, x: torch.Tensor) -> bool:
        """Return whether the kernels serve a forward on ``x``: one sequence, bf16 weights, no autograd."""
        return x.shape[0] == 1 and self.packed_input_weight.dtype == torch.bfloat16 and not torch.is_grad_enabled()

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return the packed projections [1, N, rows] of the tokens ``x`` [1, N, C]: q, k and v (3C), the output gate logits (C) and the beta logits (heads) of every token, then the padding."""
        return F.linear(x, self.packed_input_weight, self.packed_input_bias)

    def read(self, x, rotary_emb, state, decay):
        """Return the layer output [1, N, C] of the chunk ``x`` that reads ``state`` [1, heads, D, D] scaled by ``decay`` [heads] together with its own tokens."""
        out = torch.ops.sana_robot.gdn_causal_seeded_read(*self._kernel_arguments(x, rotary_emb, state, decay))
        return self.proj(out[None])

    def read_and_write(self, x, rotary_emb, state, decay):
        """Return the layer output of ``read`` and the state [1, heads, D, D] after the chunk has written its tokens by the delta rule; a ``state`` of None is the zero state."""
        if state is None:
            state = x.new_zeros(1, self.heads, self.dim, self.dim, dtype=torch.float32)
        arguments = self._kernel_arguments(x, rotary_emb, state, decay)
        (out, written) = torch.ops.sana_robot.gdn_causal_seeded_read_commit(*arguments)
        return (self.proj(out[None]), written[None])

    def _kernel_arguments(self, x, rotary_emb, state, decay) -> tuple:
        """Return the arguments of the read and commit kernels for the chunk ``x``; keys are scaled by 1 / sqrt(head_dim * tokens)."""
        (tokens, channels) = x.shape[1:]
        packed = self.project_input(x)[0]
        gate_end = 4 * channels
        return (
            packed[:, : 3 * channels],
            packed[:, gate_end : gate_end + self.heads],
            packed[:, 3 * channels : gate_end],
            rotary_emb,
            self.q_norm.weight,
            self.k_norm.weight,
            self.o_norm.weight,
            state[0].float(),
            decay,
            float(self.dim * tokens) ** (-0.5),
            self.heads,
            self.q_norm.eps,
        )


class FusedCausalGatedDeltaNet(GatedDeltaNetKernels, CausalGatedDeltaNet):
    """``CausalGatedDeltaNet`` whose streaming forwards (the sink, an observation, a denoising step, a commit) run as fused kernels; the teacher-forcing sequence and every other call run the parent's code."""

    def forward(
        self,
        x,
        rotary_emb,
        chunk_bounds=None,
        teacher_forcing_clean_segments=None,
        observation_spans=None,
        kv_cache=None,
        save_kv_cache: bool = False,
        observation_prefill: bool = False,
        windows=None,
    ):
        if kv_cache is None or windows is not None or not self.kernels_apply(x):
            return super().forward(
                x,
                rotary_emb,
                chunk_bounds,
                teacher_forcing_clean_segments,
                observation_spans,
                kv_cache,
                save_kv_cache,
                observation_prefill,
                windows,
            )
        decay = self.decay_from_gate(self.gate_proj(x.mean(dim=1)).float())[0]
        if observation_prefill:
            (out, kv_cache["observation"]) = self.read_and_write(x, rotary_emb, kv_cache.get("S"), decay)
        elif save_kv_cache:
            (out, kv_cache["S"]) = self.read_and_write(x, rotary_emb, kv_cache.get("S"), decay)
            kv_cache.pop("observation", None)
        else:
            out = self.read(x, rotary_emb, kv_cache["observation"], decay)
        return out


class FusedConditionGatedDeltaNet(GatedDeltaNetKernels, ConditionGatedDeltaNet):
    """``ConditionGatedDeltaNet`` whose prefix-caching passes run as fused kernels: ``PREFILL`` writes the condition chunk into a zero state, ``TARGET`` reads it. A forward over both chunks runs the parent's code."""

    def forward(self, x, rotary_emb, layout):
        if not isinstance(layout, ConditionPass) or not self.kernels_apply(x):
            return super().forward(x, rotary_emb, layout)
        if layout.phase == PREFILL:
            decay = x.new_ones(self.heads, dtype=torch.float32)
            (out, state) = self.read_and_write(x, rotary_emb, None, decay)
            layout.cache.store(self, state)
            return out
        decay = self._prediction_decay(x, layout.slices())[0]
        return self.read(x, rotary_emb, layout.cache.load(self), decay)


def install_fused_gdn(policy) -> None:
    """Run the Gated DeltaNet layers of a policy as fused kernels at inference.

    The streaming forwards of SANA-Robot-AR and the prefix-caching passes of SANA-Robot-base then take one packed
    input projection and the read or commit kernel of the device's architecture; training forwards run as before.

    Args:
        policy: ``SanaRobotAR`` or ``SanaRobotBase`` on its CUDA device with bf16 weights, in eval mode. Every Gated
            DeltaNet layer the device's kernels serve becomes a ``FusedCausalGatedDeltaNet`` or a
            ``FusedConditionGatedDeltaNet`` with two buffers that are not saved in checkpoints,
            ``packed_input_weight`` and ``packed_input_bias`` (``pack_input_projection``). A device or a layer size
            without kernels leaves the layers as they are.
    """
    device = next(policy.parameters()).device
    kernels = gdn_read_kernels(device)
    if kernels is None:
        return
    fused_types = {CausalGatedDeltaNet: FusedCausalGatedDeltaNet, ConditionGatedDeltaNet: FusedConditionGatedDeltaNet}
    for module in policy.modules():
        fused_type = fused_types.get(type(module))
        if fused_type is not None and kernels.supports(device, module.qkv.in_features, module.heads):
            pack_input_projection(module)
            module.__class__ = fused_type
