"""Ulysses-style Context Parallel (CP) primitives for LTX-2.0 bidirectional training.

Motivation
----------
A single training step on 121-latent (~106k token) bidirectional video requires
~76 GB activation memory per GPU, even with FSDP2. FSDP2 shards params/grads
but not activations — each rank still processes the full sequence. Context
Parallel splits the *sequence dimension* across `cp_size` GPUs within a node so
each rank only sees `T/cp_size` tokens of compute and activation, turning the
memory wall into a compute wall.

Ulysses algorithm (DeepSpeed, arXiv 2309.14509):
    Before attention:  all-to-all turns  [B, N_local, H,       D]
                                    into [B, N_full,  H_local, D]
    Run attention on the full sequence with local heads.
    After attention:   reverse all-to-all from [B, N_full, H_local, D]
                                           back [B, N_local, H,       D]

Why Ulysses (vs ring attention):
    * Attention kernel is *unchanged* — we can keep xformers
      `memory_efficient_attention` inside the all-to-all sandwich.
    * Comm cost per layer: 2 × (seq_size × hidden × bf16) / cp_size bytes.
      On H100 NVLink at ~450 GB/s aggregate, <10 ms per layer for our sizes.
    * Only self-attention needs CP. Cross-attention to text (attn2) already
      has full text K/V on every rank; each rank runs attn2 on its local
      query shard and the output is naturally in local-seq layout.

Global state
------------
The CP process group is set once at training startup via `init_context_parallel`.
Queries via `get_cp_group()` etc. return `None` / `1` when disabled so the rest
of the code can be written CP-unaware.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

_CP_GROUP: dist.ProcessGroup | None = None
_CP_SIZE: int = 1
_CP_RANK: int = 0


def init_context_parallel(cp_group: dist.ProcessGroup | None) -> None:
    """Register a process group as THE context-parallel group for this process.

    Pass ``None`` (or a size-1 group) to disable CP — callers will then behave
    as if CP was not requested.
    """
    global _CP_GROUP, _CP_SIZE, _CP_RANK
    if cp_group is None or cp_group.size() == 1:
        _CP_GROUP = None
        _CP_SIZE = 1
        _CP_RANK = 0
    else:
        _CP_GROUP = cp_group
        _CP_SIZE = cp_group.size()
        _CP_RANK = cp_group.rank()


def get_cp_group() -> dist.ProcessGroup | None:
    return _CP_GROUP


def get_cp_size() -> int:
    return _CP_SIZE


def get_cp_rank() -> int:
    return _CP_RANK


def cp_enabled() -> bool:
    return _CP_SIZE > 1


# ---------------------------------------------------------------------------
# Seq-dim sharding helpers
# ---------------------------------------------------------------------------


def shard_along_seq(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Split `x` along `dim` into cp_size equal shards and return this rank's.

    The input is required to be contiguous along `dim` and evenly divisible.
    Returns ``x`` unchanged when CP is disabled.
    """
    if not cp_enabled():
        return x
    size = x.shape[dim]
    if size % _CP_SIZE != 0:
        raise ValueError(f"shard_along_seq: dim {dim} size {size} not divisible by cp_size={_CP_SIZE}")
    chunk = size // _CP_SIZE
    start = _CP_RANK * chunk
    return x.narrow(dim, start, chunk).contiguous()


def shard_rope_pe(pe: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
    """Shard precomputed RoPE tensors along the time axis.

    LTX RoPE can be either INTERLEAVED (single tensor [B, T, D] or [T, D]) or
    SPLIT ((cos, sin) each [B, H, T, D]). Only the time axis needs slicing.
    The full-sequence positions stay correct because RoPE is applied BEFORE
    the all-to-all, so each rank rotates its local q, k by the correct
    absolute positions.
    """
    if not cp_enabled() or pe is None:
        return pe
    # SPLIT: tuple of (cos, sin) each [..., T, D] with T at dim=-2
    if isinstance(pe, tuple):
        cos, sin = pe
        time_dim = -2 if cos.ndim >= 2 else 0
        return (
            shard_along_seq(cos, dim=time_dim),
            shard_along_seq(sin, dim=time_dim),
        )
    # INTERLEAVED: single tensor. T is at dim 1 when shape is [B, T, D]
    # (3D) and dim 0 when shape is [T, D] (2D).
    time_dim = 1 if pe.ndim >= 3 else 0
    return shard_along_seq(pe, dim=time_dim)


# ---------------------------------------------------------------------------
# Ulysses all-to-all (autograd-aware)
# ---------------------------------------------------------------------------
#
# Tensor layout used here: [B, N, H, D] (batch, seq, heads, head_dim).
#
# seq_to_head:  [B, N_local, H_full, D]  →  [B, N_full, H_local, D]
# head_to_seq:  [B, N_full, H_local, D]  →  [B, N_local, H_full, D]
#
# Backward of one is the other, so we define two separate autograd.Functions.


def _seq_to_head_impl(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Forward impl of seq→head all-to-all (no autograd)."""
    cp_size = group.size()
    b, n_loc, h, d = x.shape
    if h % cp_size != 0:
        raise ValueError(f"seq_to_head: heads={h} not divisible by cp_size={cp_size}")
    h_loc = h // cp_size
    # Split heads into cp_size chunks, put chunk axis first for all-to-all.
    # [B, N_local, H_full, D] -> [B, N_local, cp_size, H_local, D]
    x = x.view(b, n_loc, cp_size, h_loc, d)
    # -> [cp_size, B, N_local, H_local, D]
    x = x.permute(2, 0, 1, 3, 4).contiguous()
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=group)
    # [cp_size, B, N_local, H_local, D] -> [B, cp_size*N_local, H_local, D]
    y = y.permute(1, 0, 2, 3, 4).contiguous().view(b, cp_size * n_loc, h_loc, d)
    return y


def _head_to_seq_impl(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Forward impl of head→seq all-to-all (no autograd)."""
    cp_size = group.size()
    b, n_full, h_loc, d = x.shape
    if n_full % cp_size != 0:
        raise ValueError(f"head_to_seq: seq={n_full} not divisible by cp_size={cp_size}")
    n_loc = n_full // cp_size
    # [B, N_full, H_local, D] -> [B, cp_size, N_local, H_local, D]
    x = x.view(b, cp_size, n_loc, h_loc, d)
    # -> [cp_size, B, N_local, H_local, D]
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=group)
    # [cp_size, B, N_local, H_local, D] -> [B, N_local, cp_size*H_local, D]
    y = y.permute(1, 2, 0, 3, 4).contiguous().view(b, n_loc, cp_size * h_loc, d)
    return y


class _SeqToHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return _seq_to_head_impl(x, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse of seq→head is head→seq.
        return _head_to_seq_impl(grad_output.contiguous(), ctx.group), None


class _HeadToSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return _head_to_seq_impl(x, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse of head→seq is seq→head.
        return _seq_to_head_impl(grad_output.contiguous(), ctx.group), None


def seq_to_head(x: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """All-to-all swap: [B, N_local, H_full, D] → [B, N_full, H_local, D]."""
    group = group or _CP_GROUP
    if group is None or group.size() == 1:
        return x
    return _SeqToHead.apply(x, group)


def head_to_seq(x: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """All-to-all swap: [B, N_full, H_local, D] → [B, N_local, H_full, D]."""
    group = group or _CP_GROUP
    if group is None or group.size() == 1:
        return x
    return _HeadToSeq.apply(x, group)
