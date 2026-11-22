# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

from sana_robot.kernels.inference.ops.blockglue import portable, sm110, sm120
from sana_robot.kernels.inference.ops.dispatch import SM90, SM110, SM120, device_architecture
from sana_robot.policy.embedding.embedders import FinalLayer
from sana_robot.policy.sana_robot_ar import SanaRobotAR
from sana_robot.policy.sana_robot_base import SanaRobotBase

KERNELS = {SM90: portable, SM110: sm110, SM120: sm120}
MAX_SOURCES = 5
(ATTENTION_SHIFT, ATTENTION_SCALE, ATTENTION_GATE, MLP_SHIFT, MLP_SCALE, MLP_GATE) = range(6)


def serving_architecture(module, channels: int) -> str | None:
    """Return the architecture whose kernels serve token rows of ``channels`` channels on the module's device, or None."""
    architecture = device_architecture(next(module.parameters()).device)
    if architecture in KERNELS and KERNELS[architecture].serves_width(channels):
        return architecture
    return None


def bf16_rows(tokens: torch.Tensor) -> torch.Tensor:
    """Return tokens [1, N, C] as contiguous bf16 rows, the layout the kernels read."""
    return tokens.to(torch.bfloat16).contiguous()


def fused_loop_serves(policy, hidden: torch.Tensor) -> bool:
    """Return whether the fused layer loop serves a call: inference outside autograd on one bf16 sample."""
    return (
        not policy.training and not torch.is_grad_enabled() and hidden.shape[0] == 1 and hidden.dtype == torch.bfloat16
    )


def run_fused_layers(policy, hidden, modulation, attention, cross_attention) -> torch.Tensor:
    """Run the layer stack on one bf16 sample: ``attention(layer, tokens)``, ``cross_attention(layer, tokens)`` and each layer's MLP are module calls, and three kernel launches per layer do the AttnRes mix, the LayerNorm and modulation of both sublayers, the gates and the residual sums on the per-token ``modulation`` [1, 1, N, 6C]. Row 0 of the source buffer is the embedding, every finished group of ``attn_res_block_size`` layers the next row, and the row after the finished ones the running sum of the current group."""
    kernels = KERNELS[policy.blockglue_architecture]
    attn_res = policy.attn_res
    block_size = policy.attn_res_block_size
    (_, tokens, channels) = hidden.shape
    sources = hidden.new_empty(math.ceil(len(policy.blocks) / block_size) + 1, 1, tokens, channels)
    sources[0] = hidden
    time_rows = modulation.reshape(tokens, 6, channels).unbind(1)
    source_count = 1
    (partial_block, mlp_output, mlp_gate_table) = (None, None, None)
    for layer, block in enumerate(policy.blocks):
        table = block.scale_shift_table
        if layer == 0:
            attention_input = sources[0]
            modulated = kernels.layer_norm_modulate(
                attention_input,
                time_rows[ATTENTION_SHIFT],
                time_rows[ATTENTION_SCALE],
                table[ATTENTION_SHIFT],
                table[ATTENTION_SCALE],
            )
        else:
            (attention_input, modulated) = kernels.sublayer_boundary(
                sources,
                attn_res.attn_proj.weight.squeeze(),
                source_count,
                partial_block,
                mlp_output,
                gate_t0=time_rows[MLP_GATE],
                gate_table=mlp_gate_table,
                shift_t0=time_rows[ATTENTION_SHIFT],
                scale_t0=time_rows[ATTENTION_SCALE],
                shift_table=table[ATTENTION_SHIFT],
                scale_table=table[ATTENTION_SCALE],
            )
            if layer % block_size == 0:
                source_count += 1
                partial_block = None
            else:
                partial_block = sources[source_count]
        attention_output = bf16_rows(attention(layer, modulated))
        (attention_delta, cross_input) = kernels.gate_residual(
            attention_output, attention_input, time_rows[ATTENTION_GATE], table[ATTENTION_GATE]
        )
        cross_delta = bf16_rows(cross_attention(layer, cross_input))
        (_, modulated) = kernels.sublayer_boundary(
            sources,
            attn_res.mlp_proj.weight.squeeze(),
            source_count,
            partial_block,
            cross_delta,
            extra_delta=attention_delta,
            shift_t0=time_rows[MLP_SHIFT],
            scale_t0=time_rows[MLP_SCALE],
            shift_table=table[MLP_SHIFT],
            scale_table=table[MLP_SCALE],
            write_hidden=False,
        )
        partial_block = sources[source_count]
        mlp_output = bf16_rows(block.mlp(modulated))
        mlp_gate_table = table[MLP_GATE]
    (mixed, _) = kernels.sublayer_boundary(
        sources,
        attn_res.final_proj.weight.squeeze(),
        source_count,
        partial_block,
        mlp_output,
        gate_t0=time_rows[MLP_GATE],
        gate_table=mlp_gate_table,
    )
    return mixed


class FusedGlueSanaRobotAR(SanaRobotAR):
    """SANA-Robot-AR whose streaming windows run the glue between the sublayers as fused kernels; every other call runs the original layer loop."""

    def _run_layers(
        self,
        hidden,
        text,
        modulation,
        text_spans,
        rope_linear,
        rope_softmax,
        attention_arguments,
        softmax_arguments,
        kv_caches,
    ):
        if kv_caches is None or not fused_loop_serves(self, hidden):
            return super()._run_layers(
                hidden,
                text,
                modulation,
                text_spans,
                rope_linear,
                rope_softmax,
                attention_arguments,
                softmax_arguments,
                kv_caches,
            )

        def attention(layer, tokens):
            softmax = layer in self.softmax_layer_indices
            layer_arguments = dict(
                attention_arguments,
                rotary_emb=rope_softmax if softmax else rope_linear,
                kv_cache=kv_caches[layer],
                **(softmax_arguments if softmax else {}),
            )
            return self.blocks[layer].attn(tokens, **layer_arguments)

        def cross_attention(layer, tokens):
            return self.blocks[layer].cross_attn(tokens, text, text_spans=text_spans)

        return run_fused_layers(self, hidden, modulation, attention, cross_attention)


class FusedGlueSanaRobotBase(SanaRobotBase):
    """SANA-Robot-base whose inference loop runs the glue between the sublayers as fused kernels, with and without prefix caching; every other call runs the original layer loop."""

    @torch.no_grad()
    def _run_layers_inference(self, hidden, text, modulation, text_arguments, rope_linear, rope_softmax, layout):
        if not fused_loop_serves(self, hidden):
            return super()._run_layers_inference(
                hidden, text, modulation, text_arguments, rope_linear, rope_softmax, layout
            )

        def attention(layer, tokens):
            rotary_emb = rope_softmax if layer in self.softmax_layer_indices else rope_linear
            return self.blocks[layer].attn(tokens, rotary_emb=rotary_emb, layout=layout)

        def cross_attention(layer, tokens):
            return self.blocks[layer].cross_attn(tokens, text, **text_arguments)

        return run_fused_layers(self, hidden, modulation, attention, cross_attention)


class FusedFinalLayer(FinalLayer):
    """``FinalLayer`` whose LayerNorm and modulation run as one kernel launch for one bf16 sample outside autograd; every other call runs the original forward."""

    def forward(self, hidden: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        (batch, tokens, channels) = hidden.shape
        if (
            self.training
            or torch.is_grad_enabled()
            or batch != 1
            or hidden.dtype != torch.bfloat16
            or tuple(time_embedding.shape) != (1, 1, tokens, channels)
        ):
            return super().forward(hidden, time_embedding)
        time_rows = time_embedding.reshape(tokens, channels)
        (shift_table, scale_table) = self.scale_shift_table.unbind(0)
        modulated = KERNELS[self.blockglue_architecture].layer_norm_modulate(
            hidden.contiguous(), time_rows, time_rows, shift_table, scale_table
        )
        return self.linear(modulated)


def install_blockglue(policy) -> None:
    """Run the glue between the sublayers of the policy's inference layer loop as fused kernels.

    Three kernel launches per layer replace the AttnRes mix, the LayerNorm and modulation of both sublayers, the
    gates and the residual sums; attention, cross-attention and the MLP stay module calls. ``SanaRobotAR`` is served
    in its streaming windows and ``SanaRobotBase`` in its inference loop with and without prefix caching, both for
    one bf16 sample; every other call runs the original loop. The policy stays as it is on a device without a
    variant of the kernels, at a token width they do not serve, and with more than ``MAX_SOURCES`` AttnRes sources.

    Args:
        policy: ``SanaRobotAR`` or ``SanaRobotBase`` on its device with bf16 weights, in eval mode.
    """
    architecture = serving_architecture(policy, policy.hidden_size)
    sources = math.ceil(len(policy.blocks) / policy.attn_res_block_size) + 1
    if architecture is None or sources > MAX_SOURCES:
        return
    policy.blockglue_architecture = architecture
    if isinstance(policy, SanaRobotAR):
        policy.__class__ = FusedGlueSanaRobotAR
    elif isinstance(policy, SanaRobotBase):
        policy.__class__ = FusedGlueSanaRobotBase


def install_fused_final_layer(policy) -> None:
    """Run the LayerNorm and modulation of the policy's video head as one kernel launch before its projection.

    The head stays as it is on a device without a variant of the kernel and at a token width it does not serve.

    Args:
        policy: ``SanaRobotAR`` or ``SanaRobotBase`` on its device with bf16 weights, in eval mode.
    """
    final_layer = policy.final_layer
    architecture = serving_architecture(final_layer, final_layer.scale_shift_table.shape[1])
    if architecture is None:
        return
    final_layer.blockglue_architecture = architecture
    final_layer.__class__ = FusedFinalLayer
