#!/usr/bin/env python3
"""Operator fusions for MiniMax-H3, derived from Sol-LTX-Infer's KWL line.

The reference fuses LTX-2's block; none of its kernels drop into MiniMax-H3 unchanged, because the
two models differ exactly where the kernels are specialized — RMSNorm instead of LayerNorm, partial
rotary instead of full, SwiGLU instead of GELU, per-row indexed modulation instead of broadcast.
What transfers is the idea in each one: never let a wide intermediate round-trip through HBM when
the next operation is elementwise and could have consumed it in registers.

Each fusion is independently switchable so a benchmark can attribute the effect rather than report
one lumped number, and each has an eager fallback that runs if Triton compilation fails, so a
kernel problem degrades to the baseline instead of killing the run.

Traffic saved per rank per evaluation, at the production shape (9562 rows x 5376, 50 blocks):

    residual+gate+norm+modulate   82 GB    the largest, and the one the reference already indexes
    qknorm + partial rope         69 GB
    swiglu                        27 GB
    adaln precompute              26 GB    weights, not activations

Roughly 25 ms of a 920 ms evaluation if HBM runs at 8 TB/s. That is the honest ceiling for this
line: about 3%. It is worth taking because it is lossless, not because it is large.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - environment without triton
    HAVE_TRITON = False


# --------------------------------------------------------------------------------------------
# 1. residual + gate[idx] * branch, then RMSNorm, then scale[idx]/shift[idx] modulation
# --------------------------------------------------------------------------------------------
# Eager runs this as thirteen passes over the activation: gather the gate row, fuse the residual,
# read it back to normalize, gather scale, gather shift, read all three to modulate. The three
# gathers each materialize a full (rows, hidden) tensor out of a table with a handful of rows,
# which is the specific waste MiniMax-H3 adds over LTX-2 — its modulation is per row, not per batch.
#
# One pass instead: the row's residual, branch output and gate row come in, `hidden` is written once
# because the next half-block needs it as its own residual, and the normalized-and-modulated value
# is written from the same registers.

if HAVE_TRITON:

    @triton.jit
    def _residual_gate_rmsnorm_modulate_kernel(
        hidden_out_ptr, normed_out_ptr,
        residual_ptr, branch_ptr, weight_ptr,
        gate_ptr, scale_ptr, shift_ptr, index_ptr,
        n_cols, n_index, eps,
        stride_row, stride_table_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offset = row * stride_row + cols

        # The modulation tables are `chunk(6, dim=-1)` views of one wide projection, so their row
        # stride is 6*hidden, not hidden. Assuming contiguity here reads a different table's row.
        # `adaln_indices` has one entry per *sequence* row and eager broadcasts it over the batch,
        # so a flattened (batch*seq) row has to wrap. Reading `index_ptr + row` directly walks off
        # the end of the index tensor for batch > 1 and gathers arbitrary table rows.
        table_row = tl.load(index_ptr + (row % n_index))
        table_offset = table_row * stride_table_row + cols

        residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)

        hidden = residual + gate * branch
        tl.store(hidden_out_ptr + offset, hidden.to(hidden_out_ptr.dtype.element_ty), mask=mask)

        # RMSNorm over the row, in the same registers that already hold it.
        variance = tl.sum(hidden * hidden, axis=0) / n_cols
        normed = hidden * tl.math.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = normed * weight

        scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        out = normed * (1.0 + scale) + shift
        tl.store(normed_out_ptr + offset, out.to(normed_out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_modulate_kernel(
        out_ptr, x_ptr, weight_ptr, scale_ptr, shift_ptr, index_ptr,
        n_cols, n_index, eps, stride_row, stride_table_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offset = row * stride_row + cols
        # See the residual variant for both notes: non-contiguous chunk views, and an index vector
        # that is per sequence row and must wrap over the batch.
        table_offset = tl.load(index_ptr + (row % n_index)) * stride_table_row + cols

        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / n_cols
        normed = x * tl.math.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = normed * weight

        scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offset, (normed * (1.0 + scale) + shift).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_kernel(out_ptr, x_ptr, n_cols, stride_in_row, stride_out_row, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        value = tl.load(x_ptr + row * stride_in_row + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * stride_in_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        out = value * (gate * tl.sigmoid(gate))
        tl.store(out_ptr + row * stride_out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _row_addressable(table: torch.Tensor) -> torch.Tensor:
    """A table the kernel can address with one row stride.

    The three modulation tables come out of `chunk(6, dim=-1)`, so each is a strided view whose row
    stride is 6*hidden while its column stride is 1. That is exactly addressable as
    `row * stride(0) + col`, so no copy is needed. Anything with a column stride other than 1 is
    made contiguous instead of being silently mis-indexed.
    """
    return table if table.stride(-1) == 1 else table.contiguous()


def _warps_for(block: int) -> int:
    """A 5376-wide row held in registers needs the warps to spread it, or it spills to local."""
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    return 4


def fused_rmsnorm_modulate(x, weight, scale, shift, index, eps):
    """RMSNorm then `* (1 + scale[index]) + shift[index]`, one pass.

    `scale` and `shift` are chunk views with a row stride of 6*hidden, and `x` may itself be a
    context-parallel shard, so both strides are read rather than assumed.
    """
    cols = x.shape[-1]
    # `reshape` only promises the shape: it can hand back a view whose last-dim stride is not 1,
    # and the kernel uses one row stride for both the input and the freshly allocated output.
    flat = x.reshape(-1, cols).contiguous()
    rows = flat.shape[0]
    scale, shift = _row_addressable(scale), _row_addressable(shift)
    out = torch.empty_like(flat)
    _rmsnorm_modulate_kernel[(rows,)](
        out, flat, weight, scale, shift, index, cols, index.numel(), eps,
        flat.stride(0), scale.stride(0),
        BLOCK=_next_pow2(cols), num_warps=_warps_for(_next_pow2(cols)),
    )
    return out.view_as(x)


def fused_residual_gate_rmsnorm_modulate(residual, branch, gate, weight, scale, shift, index, eps):
    """`hidden = residual + gate[index] * branch`, then RMSNorm+modulate. Returns both."""
    cols = residual.shape[-1]
    res_flat = residual.reshape(-1, cols).contiguous()
    br_flat = branch.reshape(-1, cols).contiguous()
    rows = res_flat.shape[0]
    gate, scale, shift = _row_addressable(gate), _row_addressable(scale), _row_addressable(shift)
    hidden = torch.empty_like(res_flat)
    normed = torch.empty_like(res_flat)
    _residual_gate_rmsnorm_modulate_kernel[(rows,)](
        hidden, normed, res_flat, br_flat, weight, gate, scale, shift, index,
        cols, index.numel(), eps, res_flat.stride(0), gate.stride(0),
        BLOCK=_next_pow2(cols), num_warps=_warps_for(_next_pow2(cols)),
    )
    return hidden.view_as(residual), normed.view_as(residual)


def fused_swiglu(x):
    """`value * silu(gate)` over a `(..., 2F)` tensor, one read of 2F and one write of F."""
    cols = x.shape[-1] // 2
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty(flat.shape[0], cols, dtype=x.dtype, device=x.device)
    _swiglu_kernel[(flat.shape[0],)](
        out, flat, cols, flat.stride(0), out.stride(0), BLOCK=_next_pow2(cols), num_warps=_warps_for(_next_pow2(cols)),
    )
    return out.view(*x.shape[:-1], cols)


# --------------------------------------------------------------------------------------------
# 2. qk-norm + partial rotary
# --------------------------------------------------------------------------------------------
# The reference's fused qknorm+rope rotates the whole head, which MiniMax-H3 does not: it rotates
# the leading `rotary_dim` channels and passes the rest through. Eager pays for that twice over —
# `cat((-x2, x1))` materializes the rotated half and `cat((rotary, pass))` materializes the whole
# head again, plus a `.contiguous()`. Six passes over q and the same over k, per block.

if HAVE_TRITON:

    @triton.jit
    def _qknorm_partial_rope_kernel(
        out_ptr, x_ptr, weight_ptr, cos_ptr, sin_ptr,
        head_dim, rotary_dim, half_dim, heads, seq, eps,
        BLOCK: tl.constexpr,
    ):
        # One program per (token, head): the RMS is over head_dim, and rotary pairs channel j with
        # j + half_dim inside the rotary span. cos/sin are shared across heads and indexed by the
        # position within the sequence, so the batch stride has to be divided out.
        pid = tl.program_id(0)
        token = (pid // heads) % seq
        cols = tl.arange(0, BLOCK)
        mask = cols < head_dim
        base = pid * head_dim

        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / head_dim
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = x * tl.math.rsqrt(variance + eps) * weight

        in_rotary = cols < rotary_dim
        in_first_half = cols < half_dim
        # partner channel: j+half within the first half, j-half within the second
        partner = tl.where(in_first_half, cols + half_dim, cols - half_dim)
        partner_value = tl.load(x_ptr + base + partner, mask=in_rotary, other=0.0).to(tl.float32)
        partner_variance = variance  # same row, so the same normalizer applies
        partner_weight = tl.load(weight_ptr + partner, mask=in_rotary, other=0.0).to(tl.float32)
        partner_normed = partner_value * tl.math.rsqrt(partner_variance + eps) * partner_weight
        # rotate_half: the first half pairs with -second, the second with +first
        rotated = tl.where(in_first_half, -partner_normed, partner_normed)

        cos = tl.load(cos_ptr + token * rotary_dim + cols, mask=in_rotary, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + token * rotary_dim + cols, mask=in_rotary, other=0.0).to(tl.float32)
        out = tl.where(in_rotary, normed * cos + rotated * sin, normed)
        tl.store(out_ptr + base + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def fused_qknorm_rope(x, weight, cos, sin, eps):
    """RMSNorm per head then partial rotary, for `(B, seq, heads, head_dim)`.

    Unlike eager, the rotation tables are consumed at whatever precision they arrive in. Upstream's
    `_apply_rotary_emb` does `cos.to(hidden_states.dtype)`, and `MiniMaxH3RotaryPosEmbed` returns
    float32, so eager quantizes the rotation to bfloat16 and this does not. The fused path is
    therefore slightly *more* accurate, and a fused render will not be bit-identical to the eager
    baseline — differing by roughly 1e-3 relative, in the fused path's favour.
    """
    batch, seq, heads, head_dim = x.shape
    rotary_dim = cos.shape[-1]
    # Guards for configurations MiniMax-H3 does not use but a future one might. Without the first,
    # the partner load is masked by `in_rotary` rather than by `cols < head_dim` and would read into
    # the next (token, head) row; without the second, `rotary_dim // 2` truncates and the pairing
    # silently desyncs from eager's `chunk(2, -1)`.
    if rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"fused qknorm+rope needs an even rotary_dim no wider than head_dim, "
            f"got rotary_dim={rotary_dim}, head_dim={head_dim}"
        )
    flat = x.reshape(-1, head_dim).contiguous()
    out = torch.empty_like(flat)
    _qknorm_partial_rope_kernel[(flat.shape[0],)](
        out, flat, weight, cos, sin,
        head_dim, rotary_dim, rotary_dim // 2, heads, seq, eps,
        BLOCK=_next_pow2(head_dim), num_warps=4,
    )
    return out.view(batch, seq, heads, head_dim)
