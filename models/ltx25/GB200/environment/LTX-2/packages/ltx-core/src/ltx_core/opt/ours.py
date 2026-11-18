"""Our own bf16 fusions (O1/O2/O5) -- the gaps the vendor leaves in bf16.

O1 fused AdaLN : rms_norm(x)*(1+scale)+shift in one Triton kernel.
                 Vendor's blockwise::adanorm hard-codes float8_e4m3fn output, so
                 there is no bf16 path at all. 48 call sites (one per block).
O2 fused add+AdaLN : residual add folded into the same kernel (y is not None).
O5 ada-value cache : get_ada_values() re-runs .to(device,dtype) on an INVARIANT
                 parameter on every call, 4-6x per block x 48 blocks per forward.
                 Pure launch overhead; memoized here.

Profile motivation (stage-1, aten level): glue ~49%, aten::add 25.7%, aten::mul 14%.
The vendor kernels (rms_fma/gated_attention) only reach 48 of the 288 attention-side
sites, which is why they measured just 1.035x combined.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from ltx_core.opt.nvfp4 import _e2m1x2_ptx


@triton.jit
def _adaln_kernel(
    X, Y, SCALE, SHIFT, OUT, FMA, Q, S,
    H, eps,
    s_stride, sh_stride, NCB,
    HAS_Y: tl.constexpr,
    EMIT_FP4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(X + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_Y:
        y = tl.load(Y + row * H + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + y
        tl.store(FMA + row * H + cols, x.to(FMA.dtype.element_ty), mask=mask)
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    s = tl.load(SCALE + row * s_stride + cols, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(SHIFT + row * sh_stride + cols, mask=mask, other=0.0).to(tl.float32)
    out = x * rstd * (1.0 + s) + sh
    tl.store(OUT + row * H + cols, out.to(OUT.dtype.element_ty), mask=mask)
    if EMIT_FP4:
        # NVFP4 epilogue: per-16 block absmax -> e4m3 scale (swizzled into the
        # 128x4 tile layout _scaled_mm needs), values -> packed E2M1 pairs.
        nb: tl.constexpr = BLOCK // 16
        amax = tl.max(tl.reshape(tl.abs(out), (nb, 16)), axis=1)
        s8 = tl.maximum(amax / 6.0, 1e-8).to(tl.float8e4nv)
        sblk = tl.arange(0, nb)
        rb = row // 128
        rin = row % 128
        so = (((rb * NCB + sblk // 4) * 32 + (rin % 32)) * 4 + (rin // 32)) * 4 + (sblk % 4)
        tl.store(S + so, s8, mask=sblk < H // 16)
        sf = tl.reshape(tl.broadcast_to(s8.to(tl.float32)[:, None], (nb, 16)), (BLOCK,))
        lo, hi = tl.split(tl.reshape(out / sf, (BLOCK // 2, 2)))
        pair = tl.arange(0, BLOCK // 2)
        tl.store(Q + row * (H // 2) + pair, _e2m1x2_ptx(hi, lo), mask=pair < H // 2)


CONTIG_STATS = {"adaln_x": [0, 0], "adaln_scale": [0, 0],
                "adaln_shift": [0, 0], "rope_x": [0, 0]}


def _note(key, t):
    """[calls, of which non-contiguous]"""
    st = CONTIG_STATS[key]
    st[0] += 1
    if not t.is_contiguous():
        st[1] += 1
    return t


def _bcast_stride(t: torch.Tensor, rows: int, h: int) -> int:
    """Row stride for a (b, st, h) modulation tensor flattened to `rows` rows."""
    if t.shape[-1] != h:
        raise ValueError(f"modulation last dim {t.shape[-1]} != hidden {h}")
    n = t.numel() // h
    if n == 1:
        return 0
    # actual row stride: unbind(dim=2) views from the ada-cache have stride
    # num*h here, and assuming h forced a .contiguous() copy of 66.8 MB
    return t.stride(-2)


def fused_adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float,
                y: torch.Tensor | None = None, emit_fp4: bool = False):
    """out = rms_norm(x + y) * (1 + scale) + shift, single kernel. Returns (x+y, out)
    or, with emit_fp4, (x+y, out, packed_fp4, swizzled_scales)."""
    h = x.shape[-1]
    rows = x.numel() // h
    xc = _note("adaln_x", x).contiguous()
    out = torch.empty_like(xc)
    fma = torch.empty_like(xc) if y is not None else xc
    # keep the strided view; only copy if the LAST dim is not unit-stride,
    # which the kernel's column indexing does require
    sc = _note("adaln_scale", scale)
    sh = _note("adaln_shift", shift)
    if sc.stride(-1) != 1:
        sc = sc.contiguous()
    if sh.stride(-1) != 1:
        sh = sh.contiguous()
    if emit_fp4:
        ks = h // 16
        ncb = -(-ks // 4)
        nrb = -(-rows // 128)
        q = torch.empty(rows, h // 2, dtype=torch.uint8, device=x.device)
        # pad rows exist only when rows is not a multiple of 128; e4m3 garbage
        # can be NaN and the tile load is 128 rows wide, so zero it then.
        sq = (torch.empty if rows % 128 == 0 else torch.zeros)(
            nrb * 128 * ncb * 4, dtype=torch.float8_e4m3fn, device=x.device)
    else:
        ncb = 0
        q = sq = out
    _adaln_kernel[(rows,)](
        xc, y.contiguous() if y is not None else xc, sc, sh, out, fma, q, sq,
        h, eps,
        _bcast_stride(sc, rows, h), _bcast_stride(sh, rows, h), ncb,
        HAS_Y=y is not None,
        EMIT_FP4=emit_fp4,
        BLOCK=triton.next_power_of_2(h),
    )
    if emit_fp4:
        return fma, out, q.view(torch.float4_e2m1fn_x2), sq
    return fma, out


# Emitting fp4 is only useful where the consumer linear was actually swapped:
# the video-side AdaLN outputs (H=4096, thousands of rows). Audio runs 126 rows
# at H=2048 and is not swapped, so the epilogue would be pure waste there.
_FP4_ADALN = os.environ.get("LTX_FP4_ADALN", "0") == "1"
_FP4_MIN_H = int(os.environ.get("LTX_FP4_ADALN_MIN_H", "4096"))
_FP4_MIN_ROWS = int(os.environ.get("LTX_FP4_ADALN_MIN_ROWS", "1024"))


class FusedAdaZero:
    """O1: drop-in for PytorchAdaZeroFunction."""

    def __call__(self, x, eps, scale, shift):
        h = x.shape[-1]
        if _FP4_ADALN and h >= _FP4_MIN_H and x.numel() // h >= _FP4_MIN_ROWS:
            from ltx_core.opt.nvfp4 import CACHE
            _, out, q, s = fused_adaln(x, scale, shift, eps, emit_fp4=True)
            CACHE.put(out, q, s)
            return out
        return fused_adaln(x, scale, shift, eps)[1]


_ADA_CACHE: dict = {}


def install_ada_cache(transformer_mod) -> int:
    """O5: memoize the invariant .to(device,dtype) inside get_ada_values."""
    cls = transformer_mod.BasicAVTransformerBlock
    orig = cls.get_ada_values
    if getattr(orig, "_ltx_cached", False):
        return 0

    def cached(self, scale_shift_table, batch_size, timestep, indices):
        num = scale_shift_table.shape[0]
        key = (scale_shift_table.data_ptr(), tuple(scale_shift_table.shape),
               indices.start, indices.stop, timestep.device, timestep.dtype)
        tbl = _ADA_CACHE.get(key)
        if tbl is None:
            tbl = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(
                device=timestep.device, dtype=timestep.dtype
            )
            _ADA_CACHE[key] = tbl
        return (tbl + timestep.reshape(batch_size, timestep.shape[1], num, -1)[:, :, indices, :]).unbind(dim=2)

    cached._ltx_cached = True
    cls.get_ada_values = cached
    return 1


def install(which: str | None = None) -> dict:
    """which: comma list of {adaln, adacache, all}."""
    import ltx_core.model.transformer.transformer as tmod

    parts = {p.strip() for p in (which or os.environ.get("LTX_OURS", "")).split(",") if p.strip()}
    applied = {}
    if parts & {"adacache", "all"}:
        applied["ada_cache"] = install_ada_cache(tmod)
    return applied


# --- O5b: resident modulation tables + rope index cache ------------------------
# The scale_shift_table Parameters live on CPU and get `.to(device=...)`d on EVERY
# forward at three sites (transformer.py:197 get_ada_values, transformer.py:441
# prompt table, model.py:429 _process_output), and rope.py:158 ships a CPU index
# tensor across as well. Besides being pure per-forward overhead, a CPU->GPU copy
# is exactly what makes CUDA-graph capture illegal:
#   "Cannot copy between CPU and CUDA tensors during CUDA graph capture"
# Making them device-resident turns all four `.to()` calls into no-ops and unblocks O7.

_ROPE_CACHE: dict = {}


def make_tables_resident(model, device=None, dtype=None) -> int:
    """Move every *scale_shift_table* parameter/buffer onto the compute device."""
    if device is None:
        device = next(p.device for p in model.parameters() if p.is_cuda)
    moved = 0
    for mod in model.modules():
        for name, param in list(mod.named_parameters(recurse=False)):
            if "scale_shift_table" in name and param.device != device:
                mod._parameters[name] = torch.nn.Parameter(
                    param.data.to(device=device, dtype=dtype or param.dtype),
                    requires_grad=param.requires_grad,
                )
                moved += 1
        for name, buf in list(mod.named_buffers(recurse=False)):
            if "scale_shift_table" in name and buf is not None and buf.device != device:
                mod._buffers[name] = buf.to(device=device)
                moved += 1
    return moved


def install_rope_index_cache(rope_mod) -> int:
    """Memoize the CPU->GPU move of the invariant rope index vector.

    rope.py:158 does ``indices.to(device=fractional_positions.device)`` on a CPU
    float32 vector every forward. Pre-move it here (cached) so that line becomes a
    no-op. `precompute_freqs_cis` calls `generate_freqs` through the module global,
    so patching the module attribute takes effect.
    """
    orig = rope_mod.generate_freqs
    if getattr(orig, "_ltx_cached", False):
        return 0

    def patched(indices, indices_grid, *a, **kw):
        if not indices.is_cuda and getattr(indices_grid, "is_cuda", False):
            key = (indices.data_ptr(), tuple(indices.shape), indices.dtype, indices_grid.device)
            hit = _ROPE_CACHE.get(key)
            if hit is None:
                hit = indices.to(device=indices_grid.device)
                _ROPE_CACHE[key] = hit
            indices = hit
        return orig(indices, indices_grid, *a, **kw)

    patched._ltx_cached = True
    rope_mod.generate_freqs = patched
    return 1


def install_graph_prereqs(model) -> dict:
    import ltx_core.model.transformer.rope as rmod
    import ltx_core.model.transformer.transformer as tmod

    return {
        "ada_cache": install_ada_cache(tmod),
        "tables_moved": make_tables_resident(model),
        "rope_cache": install_rope_index_cache(rmod),
    }
