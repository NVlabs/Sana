# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

from sana_robot.policy.attention.causal_softmax_varlen import GatherRows, csr_table
from sana_robot.policy.attention.cross_attention import MultiHeadCrossAttention


class ChunkTextSpans(NamedTuple):
    """Descriptor that routes each chunk's queries to its own caption span in cross-attention. ``q_lens``/``kv_lens`` pair query segments with caption spans one to one; cu_seqlens are int32 prefix sums."""

    q_lens: tuple
    kv_lens: tuple
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    kv_index: Optional[torch.Tensor] = None
    kv_ptr: Optional[torch.Tensor] = None
    kv_pos: Optional[torch.Tensor] = None


def chunk_text_spans(q_lens, kv_lens, device, kv_index=None, kv_rows=None) -> ChunkTextSpans:
    """Return the ChunkTextSpans descriptor for host-int segment lengths."""
    q_lens = tuple((int(v) for v in q_lens))
    kv_lens = tuple((int(v) for v in kv_lens))
    offsets_q = [0]
    for length in q_lens:
        offsets_q.append(offsets_q[-1] + length)
    offsets_k = [0]
    for length in kv_lens:
        offsets_k.append(offsets_k[-1] + length)
    dedup = {}
    if kv_index is not None:
        kv_index = kv_index.to("cpu", torch.int64).reshape(-1)
        (ptr, pos) = csr_table(kv_index, int(kv_index.max()) + 1 if kv_rows is None else int(kv_rows))
        dedup = dict(kv_index=kv_index.to(device), kv_ptr=ptr.to(device), kv_pos=pos.to(device))
    return ChunkTextSpans(
        q_lens=q_lens,
        kv_lens=kv_lens,
        cu_seqlens_q=torch.tensor(offsets_q, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(offsets_k, dtype=torch.int32, device=device),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        **dedup,
    )


def merge_text_spans(window_texts) -> tuple:
    """Return the packed text and the ``ChunkTextSpans`` of several windows laid end to end, from every window's own (packed text, spans)."""
    (q_lens, kv_lens, kv_index, text_rows) = ([], [], [], 0)
    for text, spans in window_texts:
        q_lens.extend(spans.q_lens)
        kv_lens.extend(spans.kv_lens)
        if spans.kv_index is not None:
            kv_index.append(spans.kv_index.to("cpu") + text_rows)
        text_rows += int(text.shape[1])
    packed_text = torch.cat([text for (text, _spans) in window_texts], dim=1)
    spans = chunk_text_spans(
        q_lens,
        kv_lens,
        packed_text.device,
        kv_index=torch.cat(kv_index) if kv_index else None,
        kv_rows=text_rows if kv_index else None,
    )
    return (packed_text, spans)


class ChunkTextCrossAttention(MultiHeadCrossAttention):
    """Text cross-attention of SANA-Robot-AR: one variable-length attention call in which every (chunk, text group) token span reads its own text span."""

    def forward(self, x, text, text_spans: ChunkTextSpans):
        from flash_attn import flash_attn_varlen_func

        (batch, tokens, channels) = x.shape
        (k, v) = self.kv_linear(text).view(1, -1, 2, channels).unbind(2)
        q = self._normalize(self.q_norm, self.q_linear(x)).view(batch, tokens, self.num_heads, self.head_dim)
        k = self._normalize(self.k_norm, k).view(1, -1, self.num_heads, self.head_dim)
        v = v.view(1, -1, self.num_heads, self.head_dim)
        if text_spans.kv_index is not None:
            (k, v) = (
                GatherRows.apply(rows[0], text_spans.kv_index, text_spans.kv_ptr, text_spans.kv_pos).unsqueeze(0)
                for rows in (k, v)
            )
        q = q.reshape(-1, self.num_heads, self.head_dim)
        out = flash_attn_varlen_func(
            q,
            k.reshape(-1, self.num_heads, self.head_dim).to(q.dtype),
            v.reshape(-1, self.num_heads, self.head_dim).to(q.dtype),
            cu_seqlens_q=text_spans.cu_seqlens_q,
            cu_seqlens_k=text_spans.cu_seqlens_k,
            max_seqlen_q=int(text_spans.max_seqlen_q),
            max_seqlen_k=int(text_spans.max_seqlen_k),
            dropout_p=0.0,
        )
        return self.proj(out.reshape(batch, tokens, channels))

    def _normalize(self, norm, x):
        """Apply ``norm``; in training on CUDA through one fused ``F.rms_norm`` that stays in the dtype of ``x``."""
        fused = (
            self.training
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and norm.weight.dtype == x.dtype
        )
        if not fused:
            return norm(x)
        with torch.autocast("cuda", enabled=False):
            return F.rms_norm(x, (x.shape[-1],), norm.weight, norm.eps)
