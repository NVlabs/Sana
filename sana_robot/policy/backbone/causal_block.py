# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.policy.attention.causal_gdn import CausalGatedDeltaNet
from sana_robot.policy.backbone.block import PolicyBlock
from sana_robot.policy.embedding import modulation as row_modulation
from sana_robot.policy.embedding.modulation import RowModulation


class CausalPolicyBlock(PolicyBlock):
    """Layer of SANA-Robot-AR. Training passes a ``RowModulation``, one modulation row per distinct timestep, which the fused kernels apply without expanding it to every token."""

    def attention_sublayer(self, hidden, text, modulation, text_arguments: dict, **attention_arguments):
        """The parent's attention sublayer, with the fused modulation kernels for a ``RowModulation`` when they apply."""
        if not isinstance(modulation, RowModulation):
            return super().attention_sublayer(hidden, text, modulation, text_arguments, **attention_arguments)
        if not row_modulation.kernels_apply(hidden, modulation):
            return super().attention_sublayer(hidden, text, modulation.expand(), text_arguments, **attention_arguments)
        (shift, scale, gate) = modulation.sublayer_rows(self.scale_shift_table, 0).unbind(1)
        input_dtype = torch.float32 if isinstance(self.attn, CausalGatedDeltaNet) else torch.bfloat16
        attention_input = row_modulation.modulate(
            hidden, shift, scale, modulation, eps=self.norm1.eps, out_dtype=input_dtype
        )
        attention_delta = row_modulation.gate(self.attn(attention_input, **attention_arguments), gate, modulation)
        return attention_delta + self.cross_attn(hidden + attention_delta, text, **text_arguments)

    def mlp_sublayer(self, hidden, modulation):
        """The parent's MLP sublayer, with the fused modulation kernels for a ``RowModulation`` when they apply."""
        if not isinstance(modulation, RowModulation):
            return super().mlp_sublayer(hidden, modulation)
        if not row_modulation.kernels_apply(hidden, modulation):
            return super().mlp_sublayer(hidden, modulation.expand())
        (shift, scale, gate) = modulation.sublayer_rows(self.scale_shift_table, 3).unbind(1)
        mlp_input = row_modulation.modulate(
            hidden, shift, scale, modulation, eps=self.norm2.eps, out_dtype=torch.bfloat16
        )
        return row_modulation.gate(self.mlp(mlp_input), gate, modulation)
