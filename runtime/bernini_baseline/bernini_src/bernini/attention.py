# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Variable-length attention with an auto-selected backend.

Backend priority, probed once at import time:
  1. FlashAttention-3  (``flash_attn_interface``)  -- fastest on Hopper / H100
  2. FlashAttention-2  (``flash_attn``)            -- general CUDA GPUs
  3. PyTorch SDPA                                  -- always available, no extra dep

All backends share the same varlen contract: ``q``/``k``/``v`` are packed as
``[total_tokens, num_heads, head_dim]`` and ``cu_seqlens_*`` give the per-sample
offsets into ``total_tokens``.
"""

import torch
import torch.nn.functional as F

_BACKEND = None
_flash_varlen = None


def _select_backend():
    global _BACKEND, _flash_varlen
    if _BACKEND is not None:
        return
    try:
        from flash_attn_interface import flash_attn_varlen_func  # FA3

        _flash_varlen, _BACKEND = flash_attn_varlen_func, "fa3"
        return
    except Exception:
        pass
    try:
        from flash_attn import flash_attn_varlen_func  # FA2

        _flash_varlen, _BACKEND = flash_attn_varlen_func, "fa2"
        return
    except Exception:
        pass
    _BACKEND = "sdpa"


def get_attention_backend() -> str:
    _select_backend()
    return _BACKEND


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal):
    """Varlen attention via SDPA: run each sample's segment, then concatenate."""
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    outs = []
    for i in range(len(cq) - 1):
        # [seq, H, D] -> [1, H, seq, D]
        qi = q[cq[i] : cq[i + 1]].transpose(0, 1).unsqueeze(0)
        ki = k[ck[i] : ck[i + 1]].transpose(0, 1).unsqueeze(0)
        vi = v[ck[i] : ck[i + 1]].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
        outs.append(oi.squeeze(0).transpose(0, 1))  # back to [seq, H, D]
    return torch.cat(outs, dim=0)


def varlen_attention(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal: bool = False,
):
    """Variable-length attention. Returns ``[total_q_tokens, num_heads, head_dim]``."""
    _select_backend()

    if _BACKEND == "fa3":
        out = _flash_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            causal=causal,
        )
        return out[0] if isinstance(out, tuple) else out

    if _BACKEND == "fa2":
        return _flash_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            causal=causal,
        )

    return _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
