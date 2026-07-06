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

"""Sequence-parallel tensor ops.

Every op is a no-op when Ulysses is disabled (single-GPU) and otherwise
delegates to Open-VeOmni, imported lazily so single-GPU inference carries no
dependency on it.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .state import get_parallel_state


def gather_seq_scatter_heads(x, seq_dim, head_dim, unpadded_dim_size=0):
    """All-to-all: gather the sequence dim, scatter the head dim."""
    if not get_parallel_state().ulysses_enabled:
        return x
    ps = get_parallel_state()
    x = _all_to_all_tensor(x, scatter_dim=head_dim, gather_dim=seq_dim, group=ps.ulysses_group)
    if unpadded_dim_size and unpadded_dim_size % ps.ulysses_size != 0:
        x = unpad_tensor(x, dim=seq_dim, padding_size=x.size(seq_dim) - unpadded_dim_size)
    return x


def gather_heads_scatter_seq(x, head_dim, seq_dim):
    """All-to-all: gather the head dim, scatter the sequence dim."""
    if not get_parallel_state().ulysses_enabled:
        return x
    ps = get_parallel_state()
    dim_size = x.size(seq_dim)
    if dim_size % ps.ulysses_size != 0:
        x = pad_tensor(x, seq_dim, ps.ulysses_size - (dim_size % ps.ulysses_size))
    return _all_to_all_tensor(x, scatter_dim=seq_dim, gather_dim=head_dim, group=ps.ulysses_group)


def slice_input_tensor(x, dim):
    """Keep only this rank's slice of `x` along `dim`."""
    if not get_parallel_state().ulysses_enabled:
        return x
    ps = get_parallel_state()
    return x.tensor_split(ps.ulysses_size, dim=dim)[ps.ulysses_rank].contiguous()


def slice_input_tensor_scale_grad(x, dim):
    """`slice_input_tensor` variant used inside autograd-tracked code paths."""
    return slice_input_tensor(x, dim=dim)


def gather_outputs(x, gather_dim, padding_dim=None, unpad_dim_size=None, group=None):
    """Gather a sequence-sharded tensor back to its full length."""
    if not get_parallel_state().ulysses_enabled:
        return x
    group = get_parallel_state().ulysses_group if group is None else group
    world = dist.get_world_size(group)
    outputs = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(outputs, x.contiguous(), group=group)
    out = torch.cat(outputs, dim=gather_dim).contiguous()
    if padding_dim is not None and unpad_dim_size is not None and out.size(padding_dim) > unpad_dim_size:
        slc = [slice(None)] * out.ndim
        slc[padding_dim] = slice(0, unpad_dim_size)
        out = out[slc].contiguous()
    return out


def padding_tensor_for_seqeunce_parallel(x, dim):
    """Pad `x` along `dim` so its size is divisible by the Ulysses world size."""
    if not get_parallel_state().ulysses_enabled:
        return x
    ps = get_parallel_state()
    remainder = x.size(dim) % ps.ulysses_size
    if remainder == 0:
        return x
    return pad_tensor(x, dim=dim, padding_size=ps.ulysses_size - remainder)


def _all_to_all_tensor(x, scatter_dim, gather_dim, group):
    world = dist.get_world_size(group)
    if scatter_dim <= 1 and gather_dim <= 1:
        if scatter_dim != 0:
            gather_dim_bef = x.shape[gather_dim]
            scatter_dim_bef = x.shape[scatter_dim]
            x = (
                x.reshape([gather_dim_bef, world, scatter_dim_bef // world] + list(x.shape[2:]))
                .transpose(0, 1)
                .reshape([gather_dim_bef * world, scatter_dim_bef // world] + list(x.shape[2:]))
                .contiguous()
            )
        output = torch.empty_like(x)
        dist.all_to_all_single(output, x.contiguous(), group=group)
        if scatter_dim == 0:
            output = torch.cat(output.split(x.size(0) // world), dim=gather_dim)
        return output.contiguous()

    input_list = [t.contiguous() for t in torch.tensor_split(x, world, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def pad_tensor(x, dim, padding_size, padding_value=0):
    """Append `padding_size` entries along `dim` (F.pad based, low peak memory)."""
    pad_config = [0, 0] * x.ndim
    pad_config[(x.ndim - 1 - dim) * 2 + 1] = padding_size
    return F.pad(x, pad_config, value=padding_value)


def unpad_tensor(x, dim, padding_size):
    """Inverse of `pad_tensor`: drop the last `padding_size` entries along `dim`."""
    slc = [slice(None)] * x.ndim
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def gen_cu_seqlens_for_cross_attn(q_len, batch_seqlens_q, batch_seqlens_k, device="cpu"):
    """cu_seqlens / max_seqlens for cross-attention under Ulysses sequence parallel.

    Each rank holds a contiguous ``q_len / sp_world`` slice of the query
    sequence; this maps the per-sample query/key lengths onto that local slice.
    """
    ps = get_parallel_state()
    sp_world = ps.ulysses_size
    rank = ps.ulysses_rank
    rank_q_len = (q_len + ((sp_world - (q_len % sp_world)) % sp_world)) // sp_world
    start = rank_q_len * rank
    end = min(q_len, start + rank_q_len)
    offset = 0
    cu_seqlens_q = [start]
    index = []
    max_seqlen_q = -1
    max_seqlen_k = -1
    for i, length in enumerate(batch_seqlens_q):
        offset = min(offset + length, end)
        if offset <= start:
            continue
        cu_seqlens_q.append(offset)
        index.append(i)
        max_seqlen_q = max(max_seqlen_q, cu_seqlens_q[-1] - cu_seqlens_q[-2])
        max_seqlen_k = max(max_seqlen_k, batch_seqlens_k[i])
        if offset >= end:
            break
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    max_seqlen_q = torch.tensor(max_seqlen_q, device=device)
    cu_seqlens_q -= start
    cu_seqlens_k = torch.zeros(len(batch_seqlens_k) + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.tensor(batch_seqlens_k, dtype=torch.int32, device=device).cumsum(dim=0)
    cu_seqlens_k = cu_seqlens_k[index[0] : index[-1] + 2]
    max_seqlen_k = torch.tensor(max_seqlen_k, device=device)
    return cu_seqlens_k, cu_seqlens_q, max_seqlen_k, max_seqlen_q, rank_q_len
