# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

TILE = 64
SUB = 2


@dataclass
class RowModulation:
    """``t0`` of a teacher-forcing forward: one modulation row per distinct timestep. ``rows``: ``[R, 6C]`` ``t_block`` output rows."""

    rows: torch.Tensor
    index: torch.Tensor
    tiles: torch.Tensor
    order: torch.Tensor
    row_offsets: torch.Tensor
    batch: int
    tokens: int

    @property
    def shape(self):
        return torch.Size((self.batch, 1, self.tokens, self.rows.shape[-1]))

    def expand(self) -> torch.Tensor:
        """The reference ``[B, 1, N, 6C]`` per-token ``t0``."""
        return self.rows.index_select(0, self.index).reshape(self.batch, 1, self.tokens, -1)

    def select_tokens(self, order: torch.Tensor) -> "RowModulation":
        """The same rows for the tokens ``order`` (``[N']`` positions, shared by the batch)."""
        index = self.index.reshape(self.batch, self.tokens).index_select(1, order).reshape(-1)
        (tiles, tile_order, row_offsets) = token_tiles(index, self.rows.shape[0])
        return RowModulation(
            self.rows,
            index,
            tiles,
            tile_order,
            row_offsets,
            self.batch,
            int(order.numel()),
        )

    def sublayer_rows(self, table: torch.Tensor, first: int) -> torch.Tensor:
        """Return ``table[first:first + 3] + rows`` per row, [R, 3, C]: the (shift, scale, gate) rows of one sublayer."""
        channels = table.shape[-1]
        rows = self.rows.reshape(self.rows.shape[0], -1, channels)[:, first : first + 3]
        return table[None, first : first + 3, :] + rows


def build_row_modulation(token_timestep: torch.Tensor, condition: torch.Tensor, t_block) -> RowModulation:
    """Rows of ``t_block(condition)`` for the distinct values of ``token_timestep`` (``[B, 1, N]``). ``condition`` (``[B, 1, N, C]``) is the per-token time embedding; tokens with equal timesteps have equal rows, so ``t_block`` runs on the first token of every distinct timestep."""
    (batch, tokens) = (token_timestep.shape[0], token_timestep.shape[-1])
    flat = token_timestep.reshape(-1)
    (values, index) = torch.unique(flat, return_inverse=True)
    count = values.numel()
    positions = torch.arange(flat.numel(), device=flat.device)
    first = torch.full((count,), flat.numel(), device=flat.device, dtype=positions.dtype)
    first = first.scatter_reduce(0, index, positions, reduce="amin")
    rows = t_block(condition.reshape(-1, condition.shape[-1]).index_select(0, first))
    (tiles, order, row_offsets) = token_tiles(index, count)
    return RowModulation(rows, index, tiles, order, row_offsets, batch, tokens)


def token_tiles(index: torch.Tensor, count: int, tile: int = TILE):
    """Tiles of at most ``tile`` consecutive tokens with one row each, and the tiles grouped by row."""
    total = index.numel()
    device = index.device
    change = torch.ones(total, dtype=torch.bool, device=device)
    change[1:] = index[1:] != index[:-1]
    starts = change.nonzero().flatten()
    lengths = torch.diff(starts, append=starts.new_tensor([total]))
    per_run = (lengths + tile - 1) // tile
    run = torch.repeat_interleave(torch.arange(starts.numel(), device=device), per_run)
    run_first_tile = torch.cumsum(per_run, 0) - per_run
    within = torch.arange(run.numel(), device=device) - run_first_tile[run]
    first_token = starts[run] + within * tile
    counts = torch.minimum(lengths[run] - within * tile, torch.full_like(within, tile))
    row = index[first_token]
    tiles = torch.stack((first_token, counts, row), dim=1).to(torch.int32).contiguous()
    order = torch.argsort(row, stable=True).to(torch.int32)
    row_offsets = torch.zeros(count + 1, dtype=torch.int64, device=device)
    row_offsets[1:] = torch.cumsum(torch.bincount(row, minlength=count), 0)
    return (tiles, order, row_offsets.to(torch.int32))


@triton.jit
def _bf16_rne(x):
    """Round float32 to bfloat16 to nearest, ties to even, as PyTorch's cast does. Infinities stay infinite and a NaN stays a NaN."""
    bits = x.to(tl.uint32, bitcast=True)
    rounded = bits + 32767 + (bits >> 16 & 1) >> 16
    nan = bits & 2147483647 > 2139095040
    out = tl.where(nan, bits >> 16 | 64, rounded)
    return out.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _store_cast(x, DTYPE: tl.constexpr):
    """``x`` (float32) in the storage dtype DTYPE, bfloat16 rounded like PyTorch."""
    if DTYPE == tl.bfloat16:
        return _bf16_rne(x)
    return x.to(DTYPE)


@triton.jit
def _round(x, ROUND: tl.constexpr):
    """Round float32 ``x`` to the dtype ROUND (0: float32, 1: bfloat16, 2: float16) and back."""
    if ROUND == 1:
        x = _bf16_rne(x).to(tl.float32)
    elif ROUND == 2:
        x = x.to(tl.float16).to(tl.float32)
    return x


@triton.jit
def _mod_fwd_kernel(
    X,
    OUT,
    SHIFT,
    SCALE,
    TILES,
    MEAN,
    RSTD,
    C,
    stride_x,
    stride_o,
    stride_row,
    eps,
    NORM: tl.constexpr,
    ROUND_MOD: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SUB: tl.constexpr,
    TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    start = tl.load(TILES + tile * 3)
    count = tl.load(TILES + tile * 3 + 1)
    row = tl.load(TILES + tile * 3 + 2).to(tl.int64)
    cols = tl.arange(0, BLOCK_C)
    cmask = cols < C
    shift = tl.load(SHIFT + row * stride_row + cols, mask=cmask, other=0.0).to(tl.float32)
    scale = tl.load(SCALE + row * stride_row + cols, mask=cmask, other=0.0).to(tl.float32)
    one_plus = _round(1.0 + scale, ROUND_MOD)
    subs = tl.arange(0, SUB)
    for step in range(0, TILE, SUB):
        tok = step + subs
        tmask = tok < count
        token = (start + tok).to(tl.int64)
        mask = tmask[:, None] & cmask[None, :]
        x = tl.load(X + token[:, None] * stride_x + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        if NORM:
            mean = tl.sum(x, axis=1) / C
            centered = tl.where(mask, x - mean[:, None], 0.0)
            var = tl.sum(centered * centered, axis=1) / C
            rstd = 1.0 / tl.sqrt(var + eps)
            x = centered * rstd[:, None]
            tl.store(MEAN + token, mean, mask=tmask)
            tl.store(RSTD + token, rstd, mask=tmask)
        out = libdevice.add_rn(libdevice.mul_rn(x, one_plus[None, :]), shift[None, :])
        tl.store(OUT + token[:, None] * stride_o + cols[None, :], _store_cast(out, OUT.dtype.element_ty), mask=mask)


@triton.jit
def _mod_bwd_kernel(
    X,
    DOUT,
    DX,
    SCALE,
    TILES,
    MEAN,
    RSTD,
    PART_SHIFT,
    PART_SCALE,
    C,
    stride_x,
    stride_d,
    stride_dx,
    stride_row,
    stride_part,
    NORM: tl.constexpr,
    ROUND_MOD: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SUB: tl.constexpr,
    TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    start = tl.load(TILES + tile * 3)
    count = tl.load(TILES + tile * 3 + 1)
    row = tl.load(TILES + tile * 3 + 2).to(tl.int64)
    cols = tl.arange(0, BLOCK_C)
    cmask = cols < C
    scale = tl.load(SCALE + row * stride_row + cols, mask=cmask, other=0.0).to(tl.float32)
    one_plus = _round(1.0 + scale, ROUND_MOD)
    acc_shift = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_scale = tl.zeros((BLOCK_C,), dtype=tl.float32)
    subs = tl.arange(0, SUB)
    for step in range(0, TILE, SUB):
        tok = step + subs
        tmask = tok < count
        token = (start + tok).to(tl.int64)
        mask = tmask[:, None] & cmask[None, :]
        x = tl.load(X + token[:, None] * stride_x + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(DOUT + token[:, None] * stride_d + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        if NORM:
            mean = tl.load(MEAN + token, mask=tmask, other=0.0)
            rstd = tl.load(RSTD + token, mask=tmask, other=0.0)
            x = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        acc_shift += tl.sum(dout, axis=0)
        acc_scale += tl.sum(dout * x, axis=0)
        dn = dout * one_plus[None, :]
        if NORM:
            mean_dn = tl.sum(dn, axis=1) / C
            mean_dnx = tl.sum(dn * x, axis=1) / C
            dn = (dn - mean_dn[:, None] - x * mean_dnx[:, None]) * rstd[:, None]
        tl.store(DX + token[:, None] * stride_dx + cols[None, :], _store_cast(dn, DX.dtype.element_ty), mask=mask)
    tl.store(PART_SHIFT + tile.to(tl.int64) * stride_part + cols, acc_shift, mask=cmask)
    tl.store(PART_SCALE + tile.to(tl.int64) * stride_part + cols, acc_scale, mask=cmask)


@triton.jit
def _gate_fwd_kernel(
    A, OUT, GATE, TILES, C, stride_a, stride_o, stride_row, BLOCK_C: tl.constexpr, SUB: tl.constexpr, TILE: tl.constexpr
):
    tile = tl.program_id(0)
    start = tl.load(TILES + tile * 3)
    count = tl.load(TILES + tile * 3 + 1)
    row = tl.load(TILES + tile * 3 + 2).to(tl.int64)
    cols = tl.arange(0, BLOCK_C)
    cmask = cols < C
    gate = tl.load(GATE + row * stride_row + cols, mask=cmask, other=0.0).to(tl.float32)
    subs = tl.arange(0, SUB)
    for step in range(0, TILE, SUB):
        tok = step + subs
        tmask = tok < count
        token = (start + tok).to(tl.int64)
        mask = tmask[:, None] & cmask[None, :]
        a = tl.load(A + token[:, None] * stride_a + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        tl.store(
            OUT + token[:, None] * stride_o + cols[None, :],
            _store_cast(a * gate[None, :], OUT.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _gate_bwd_kernel(
    A,
    DOUT,
    DA,
    GATE,
    TILES,
    PART,
    C,
    stride_a,
    stride_d,
    stride_da,
    stride_row,
    stride_part,
    ROUND_PROD: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SUB: tl.constexpr,
    TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    start = tl.load(TILES + tile * 3)
    count = tl.load(TILES + tile * 3 + 1)
    row = tl.load(TILES + tile * 3 + 2).to(tl.int64)
    cols = tl.arange(0, BLOCK_C)
    cmask = cols < C
    gate = tl.load(GATE + row * stride_row + cols, mask=cmask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    subs = tl.arange(0, SUB)
    for step in range(0, TILE, SUB):
        tok = step + subs
        tmask = tok < count
        token = (start + tok).to(tl.int64)
        mask = tmask[:, None] & cmask[None, :]
        a = tl.load(A + token[:, None] * stride_a + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(DOUT + token[:, None] * stride_d + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(dout * a, axis=0)
        da = _round(dout * gate[None, :], ROUND_PROD)
        tl.store(DA + token[:, None] * stride_da + cols[None, :], _store_cast(da, DA.dtype.element_ty), mask=mask)
    tl.store(PART + tile.to(tl.int64) * stride_part + cols, acc, mask=cmask)


@triton.jit
def _row_reduce_kernel(
    PART, ORDER, OFFSETS, OUT, C, stride_part, stride_out, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < C
    lo = tl.load(OFFSETS + row)
    hi = tl.load(OFFSETS + row + 1)
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for first in range(lo, hi, BLOCK_T):
        idx = first + tl.arange(0, BLOCK_T)
        tmask = idx < hi
        tiles = tl.load(ORDER + idx, mask=tmask, other=0).to(tl.int64)
        acc += tl.load(
            PART + tiles[:, None] * stride_part + cols[None, :], mask=tmask[:, None] & cmask[None, :], other=0.0
        )
    tl.store(OUT + row.to(tl.int64) * stride_out + cols, tl.sum(acc, axis=0), mask=cmask)


_ROUND = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_NO_FMA = dict(enable_fp_fusion=False)


def _block_c(channels: int) -> int:
    return triton.next_power_of_2(channels)


def _row_sums(partial: torch.Tensor, mod: RowModulation, count: int) -> torch.Tensor:
    out = torch.empty(count, partial.shape[1], device=partial.device, dtype=torch.float32)
    channels = partial.shape[1]
    _row_reduce_kernel[count, triton.cdiv(channels, 256)](
        partial,
        mod.order,
        mod.row_offsets,
        out,
        channels,
        partial.stride(0),
        out.stride(0),
        BLOCK_T=32,
        BLOCK_C=256,
        num_warps=4,
    )
    return out


class _Modulate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, shift, scale, mod, eps, out_dtype):
        channels = x.shape[-1]
        flat = x.reshape(-1, channels)
        if flat.stride(-1) != 1:
            flat = flat.contiguous()
        out = torch.empty(flat.shape, device=x.device, dtype=out_dtype)
        norm = eps is not None
        mean = rstd = None
        if norm:
            mean = torch.empty(flat.shape[0], device=x.device, dtype=torch.float32)
            rstd = torch.empty_like(mean)
        tiles = mod.tiles
        _mod_fwd_kernel[tiles.shape[0],](
            flat,
            out,
            shift,
            scale,
            tiles,
            mean if norm else out,
            rstd if norm else out,
            channels,
            flat.stride(0),
            out.stride(0),
            shift.stride(0),
            0.0 if eps is None else float(eps),
            NORM=norm,
            ROUND_MOD=_ROUND[scale.dtype],
            BLOCK_C=_block_c(channels),
            SUB=SUB,
            TILE=TILE,
            num_warps=8,
            **_NO_FMA,
        )
        ctx.save_for_backward(flat, scale, mean, rstd)
        ctx.mod = mod
        ctx.norm = norm
        (ctx.x_shape, ctx.x_dtype) = (x.shape, x.dtype)
        ctx.shift_dtype = shift.dtype
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad):
        (flat, scale, mean, rstd) = ctx.saved_tensors
        mod = ctx.mod
        channels = flat.shape[-1]
        grad = grad.reshape(-1, channels)
        if grad.stride(-1) != 1:
            grad = grad.contiguous()
        dx = torch.empty(flat.shape, device=flat.device, dtype=ctx.x_dtype)
        tiles = mod.tiles
        part_shift = torch.empty(tiles.shape[0], channels, device=flat.device, dtype=torch.float32)
        part_scale = torch.empty_like(part_shift)
        _mod_bwd_kernel[tiles.shape[0],](
            flat,
            grad,
            dx,
            scale,
            tiles,
            mean if ctx.norm else flat,
            rstd if ctx.norm else flat,
            part_shift,
            part_scale,
            channels,
            flat.stride(0),
            grad.stride(0),
            dx.stride(0),
            scale.stride(0),
            part_shift.stride(0),
            NORM=ctx.norm,
            ROUND_MOD=_ROUND[scale.dtype],
            BLOCK_C=_block_c(channels),
            SUB=SUB,
            TILE=TILE,
            num_warps=8,
            **_NO_FMA,
        )
        count = scale.shape[0]
        d_shift = _row_sums(part_shift, mod, count).to(ctx.shift_dtype)
        d_scale = _row_sums(part_scale, mod, count).to(scale.dtype)
        return (dx.reshape(ctx.x_shape), d_shift, d_scale, None, None, None)


class _Gate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, gate_rows, mod):
        channels = a.shape[-1]
        flat = a.reshape(-1, channels)
        if flat.stride(-1) != 1:
            flat = flat.contiguous()
        out_dtype = torch.promote_types(a.dtype, gate_rows.dtype)
        out = torch.empty(flat.shape, device=a.device, dtype=out_dtype)
        tiles = mod.tiles
        _gate_fwd_kernel[tiles.shape[0],](
            flat,
            out,
            gate_rows,
            tiles,
            channels,
            flat.stride(0),
            out.stride(0),
            gate_rows.stride(0),
            BLOCK_C=_block_c(channels),
            SUB=SUB,
            TILE=TILE,
            num_warps=8,
            **_NO_FMA,
        )
        ctx.save_for_backward(flat, gate_rows)
        ctx.mod = mod
        (ctx.a_shape, ctx.a_dtype, ctx.out_dtype) = (a.shape, a.dtype, out_dtype)
        return out.reshape(a.shape)

    @staticmethod
    def backward(ctx, grad):
        (flat, gate_rows) = ctx.saved_tensors
        mod = ctx.mod
        channels = flat.shape[-1]
        grad = grad.reshape(-1, channels)
        if grad.stride(-1) != 1:
            grad = grad.contiguous()
        da = torch.empty(flat.shape, device=flat.device, dtype=ctx.a_dtype)
        tiles = mod.tiles
        part = torch.empty(tiles.shape[0], channels, device=flat.device, dtype=torch.float32)
        _gate_bwd_kernel[tiles.shape[0],](
            flat,
            grad,
            da,
            gate_rows,
            tiles,
            part,
            channels,
            flat.stride(0),
            grad.stride(0),
            da.stride(0),
            gate_rows.stride(0),
            part.stride(0),
            ROUND_PROD=_ROUND[ctx.out_dtype],
            BLOCK_C=_block_c(channels),
            SUB=SUB,
            TILE=TILE,
            num_warps=8,
            **_NO_FMA,
        )
        d_gate = _row_sums(part, mod, gate_rows.shape[0]).to(gate_rows.dtype)
        return (da.reshape(ctx.a_shape), d_gate, None)


def modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, mod: RowModulation, eps=None, out_dtype=torch.float32
):
    """``n * (1 + scale[row]) + shift[row]`` of ``[B, N, C]`` tokens, computed in float32 and stored as ``out_dtype``; ``shift``/``scale`` are ``[R, C]``. ``eps=None``: ``x`` is the normalized ``n``; otherwise ``n = LayerNorm(x, eps)`` (no affine) inside the kernel."""
    return _Modulate.apply(x, shift, scale, mod, eps, out_dtype)


def gate(a: torch.Tensor, gate_rows: torch.Tensor, mod: RowModulation) -> torch.Tensor:
    """``gate_rows[row] * a`` of ``[B, N, C]`` tokens in the product dtype."""
    return _Gate.apply(a, gate_rows, mod)


def kernels_apply(h: torch.Tensor, mod: RowModulation) -> bool:
    """Whether the fused path reproduces the reference arithmetic for this call: CUDA bf16 or float32 tokens (the AttnRes of SANA-Robot-AR sums its sources in float32) under bf16 autocast (LayerNorm in float32, products in bf16)."""
    return (
        h.is_cuda
        and h.dtype in (torch.bfloat16, torch.float32)
        and (mod.rows.dtype == torch.bfloat16)
        and torch.is_autocast_enabled("cuda")
        and (torch.get_autocast_dtype("cuda") == torch.bfloat16)
        and (h.shape[0] * h.shape[1] == mod.index.numel())
    )
