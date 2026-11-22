# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sana_robot.policy.backbone.norms import RMSNorm


class PackedGroupedText(NamedTuple):
    """Grouped text packed for one variable-length attention call at batch size 1: ``spans`` are the (offset, length) token span of every group and ``kv_lens`` its text token count; the text arrives as the sum(kv_lens) real tokens."""

    spans: tuple[tuple[int, int], ...]
    kv_lens: tuple[int, ...]
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0


class MultiHeadCrossAttention(nn.Module):
    """Text cross-attention with RMS-normalized queries and keys. Every token span reads its own text group."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.kv_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.q_norm = RMSNorm(hidden_size, scale_factor=1.0, eps=1e-06)
        self.k_norm = RMSNorm(hidden_size, scale_factor=1.0, eps=1e-06)

    def forward(self, x, text, text_mask=None, prompt_group_spans=None):
        """Attend the tokens of every span in ``prompt_group_spans`` to its text group [B, groups, L, hidden] and return the outputs in token order."""
        if isinstance(prompt_group_spans, PackedGroupedText):
            return self._attend_packed(x, text, prompt_group_spans)
        outputs = []
        for group, (offset, length) in enumerate(prompt_group_spans):
            group_mask = None if text_mask is None else text_mask[:, group]
            span = x[:, int(offset) : int(offset) + int(length)].contiguous()
            outputs.append(self._attend(span, text[:, group], group_mask))
        return torch.cat(outputs, dim=1)

    def _attend(self, x, text, mask):
        """Attend tokens [B, N, hidden] to one text group [B, L, hidden] under its token mask [B, L]."""
        (batch, _, channels) = x.shape
        (k, v) = self.kv_linear(text).view(batch, -1, 2, channels).unbind(2)
        q = self.q_norm(self.q_linear(x)).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(k).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = (1 - mask.to(q.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(batch, -1, channels))

    def _attend_packed(self, x, text, packed: PackedGroupedText):
        """Attend every span to its packed text group in one variable-length attention call."""
        from flash_attn import flash_attn_varlen_func

        (_, tokens, channels) = x.shape
        q = self.q_norm(self.q_linear(x)).view(tokens, self.num_heads, self.head_dim)
        (k, v) = self.kv_linear(text).view(-1, 2, channels).unbind(1)
        k = self.k_norm(k).view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        out = flash_attn_varlen_func(
            q,
            k.to(q.dtype),
            v.to(q.dtype),
            cu_seqlens_q=packed.cu_seqlens_q,
            cu_seqlens_k=packed.cu_seqlens_k,
            max_seqlen_q=int(packed.max_seqlen_q),
            max_seqlen_k=int(packed.max_seqlen_k),
        )
        return self.proj(out.reshape(1, tokens, channels))
