"""Triton kernels for the Wan2.2 DiT block (GB200), BF16 or MXFP8 output.

ln_mod        [residual (+ gate*) branch ->] LayerNorm [affine] [* (1+scale) + shift] per row, the batch
              index of a row picking the modulation vector. Mirrors diffusers' rounding: the updated
              residual is rounded to bf16 first and the norm reads that rounded value, as the eager
              block does when it stores `hidden_states` between the two ops.
qk_rms_rope   RMSNorm over the whole hidden dim (Wan's `rms_norm_across_heads`) then the interleaved
              rotary pairs of every head, for q and k in one launch; with ROPE=False it is the plain
              RMSNorm used by the cross-attention query.
gelu_tanh     2-D tiles, no row reduction.
mx_quant      bf16 rows -> MXFP8 (attention outputs feeding the output projections).

MXFP8 output = E4M3 values + one E8M0 scale per 32 along K, in cuBLASLt's 32_4_4 swizzle (the
`mxfp8.MXActivation` the MXFP8 linears consume).
"""
import torch
import triton
import triton.language as tl

from mxfp8 import MXActivation, _allocate, _mx_e8m0_from_amax, _mx_scale_offsets


@triton.jit
def _emit(out_ptr, q_ptr, s_ptr, row, cols, m, val, N, num_groups, column_blocks,
          CHUNK: tl.constexpr, c, OUT_FP8: tl.constexpr):
    if OUT_FP8:
        v = val.to(tl.bfloat16).to(tl.float32)
        vals = tl.reshape(v, [CHUNK // 32, 32])
        amax = tl.max(tl.abs(vals), axis=1)
        sb, inv = _mx_e8m0_from_amax(amax)
        qv = tl.reshape(vals * inv[:, None], [CHUNK])
        tl.store(q_ptr + row.to(tl.int64) * N + cols, qv.to(tl.float8e4nv), mask=m)
        grp = c * (CHUNK // 32) + tl.arange(0, CHUNK // 32)
        tl.store(s_ptr + _mx_scale_offsets(row, grp, column_blocks), sb.to(tl.uint8), mask=grp < num_groups)
    else:
        tl.store(out_ptr + row.to(tl.int64) * N + cols, val.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _ln_mod_kernel(out_ptr, q_ptr, s_ptr, hout_ptr, res_ptr, br_ptr, gate_ptr, w_ptr, b_ptr,
                   scale_ptr, shift_ptr, N, rows_per_batch, mod_stride, gate_stride, num_groups, column_blocks, eps,
                   HAS_BRANCH: tl.constexpr, HAS_GATE: tl.constexpr, AFFINE: tl.constexpr,
                   MOD: tl.constexpr, OUT_FP8: tl.constexpr, CHUNK: tl.constexpr, NCHUNK: tl.constexpr):
    row = tl.program_id(0)
    bidx = (row // rows_per_batch).to(tl.int64)
    base = row.to(tl.int64) * N
    mbase = bidx * mod_stride
    gbase = bidx * gate_stride
    # pass 1: the (rounded) residual, stored, and its sum
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        m = cols < N
        x = tl.load(res_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        if HAS_BRANCH:
            b = tl.load(br_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
            if HAS_GATE:
                g = tl.load(gate_ptr + gbase + cols, mask=m, other=0.0).to(tl.float32)
                x = x + b * g
            else:
                x = x + b
            x = x.to(tl.bfloat16).to(tl.float32)
            tl.store(hout_ptr + base + cols, x.to(tl.bfloat16), mask=m)
        acc += x
    mean = tl.sum(acc, axis=0) / N
    # pass 2: variance about the mean (inputs re-read from L2)
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        m = cols < N
        x = tl.load(res_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        if HAS_BRANCH:
            b = tl.load(br_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
            if HAS_GATE:
                g = tl.load(gate_ptr + gbase + cols, mask=m, other=0.0).to(tl.float32)
                x = x + b * g
            else:
                x = x + b
            x = x.to(tl.bfloat16).to(tl.float32)
        d = tl.where(m, x - mean, 0.0)
        acc += d * d
    rstd = tl.math.rsqrt(tl.sum(acc, axis=0) / N + eps)
    # pass 3: normalize, affine, modulate, emit
    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        m = cols < N
        x = tl.load(res_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        if HAS_BRANCH:
            b = tl.load(br_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
            if HAS_GATE:
                g = tl.load(gate_ptr + gbase + cols, mask=m, other=0.0).to(tl.float32)
                x = x + b * g
            else:
                x = x + b
            x = x.to(tl.bfloat16).to(tl.float32)
        y = (x - mean) * rstd
        if AFFINE:
            y = y * tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32) \
                + tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        if MOD:
            sc = tl.load(scale_ptr + mbase + cols, mask=m, other=0.0).to(tl.float32)
            sh = tl.load(shift_ptr + mbase + cols, mask=m, other=0.0).to(tl.float32)
            y = y * (1.0 + sc) + sh
        _emit(out_ptr, q_ptr, s_ptr, row, cols, m, y, N, num_groups, column_blocks, CHUNK, c, OUT_FP8)


def ln_mod(res, rows_per_batch, eps, *, branch=None, gate=None, weight=None, bias=None, scale=None,
           shift=None, out_fp8=False, chunk=1024, num_warps=4):
    """res/branch: (rows, N) bf16 contiguous. gate/scale/shift: per-batch fp32 rows sharing one row
    stride (views into the (B, 6, D) modulation table). Returns (new_residual or None, normed)."""
    rows, N = res.shape
    hout = torch.empty_like(res) if branch is not None else res
    mods = [t for t in (scale, shift) if t is not None]
    mod_stride = mods[0].stride(0) if mods else 0
    for t in mods:
        assert t.stride(-1) == 1 and t.stride(0) == mod_stride and t.dtype == torch.float32
    gate_stride = gate.stride(0) if gate is not None else 0
    assert gate is None or (gate.stride(-1) == 1 and gate.dtype == torch.float32)
    dummy = res
    if out_fp8:
        q, s = _allocate(rows, N, res.device)
        out = dummy
    else:
        out = torch.empty_like(res)
        q, s = dummy, dummy
    ng = N // 32
    _ln_mod_kernel[(rows,)](
        out, q, s, hout, res, branch if branch is not None else dummy, gate if gate is not None else dummy,
        weight if weight is not None else dummy, bias if bias is not None else dummy,
        scale if scale is not None else dummy, shift if shift is not None else dummy,
        N, rows_per_batch, mod_stride, gate_stride, ng, -(-ng // 4), eps,
        HAS_BRANCH=branch is not None, HAS_GATE=gate is not None, AFFINE=weight is not None,
        MOD=scale is not None, OUT_FP8=out_fp8, CHUNK=chunk, NCHUNK=triton.cdiv(N, chunk), num_warps=num_warps)
    normed = MXActivation(q, s, (rows, N), torch.bfloat16) if out_fp8 else out
    return (hout if branch is not None else None), normed


@triton.jit
def _qk_rms_rope_kernel(qo_ptr, ko_ptr, q_ptr, k_ptr, wq_ptr, wk_ptr, cos_ptr, sin_ptr, rows_per_batch,
                        P, eps, HALF_HEAD: tl.constexpr, ROPE: tl.constexpr, CHUNK: tl.constexpr,
                        NCHUNK: tl.constexpr):
    """One row of q (program_id(1)==0) or k (==1); P = D/2 interleaved pairs."""
    row = tl.program_id(0)
    which = tl.program_id(1)
    if which == 0:
        x_ptr = q_ptr
        o_ptr = qo_ptr
        w_ptr = wq_ptr
    else:
        x_ptr = k_ptr
        o_ptr = ko_ptr
        w_ptr = wk_ptr
    base = row.to(tl.int64) * (2 * P)
    tok = (row % rows_per_batch).to(tl.int64)
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        i = c * CHUNK + tl.arange(0, CHUNK)
        m = i < P
        a = tl.load(x_ptr + base + 2 * i, mask=m, other=0.0).to(tl.float32)
        b = tl.load(x_ptr + base + 2 * i + 1, mask=m, other=0.0).to(tl.float32)
        acc += a * a + b * b
    rinv = tl.math.rsqrt(tl.sum(acc, axis=0) / (2 * P) + eps)
    for c in tl.static_range(NCHUNK):
        i = c * CHUNK + tl.arange(0, CHUNK)
        m = i < P
        a = tl.load(x_ptr + base + 2 * i, mask=m, other=0.0).to(tl.float32)
        b = tl.load(x_ptr + base + 2 * i + 1, mask=m, other=0.0).to(tl.float32)
        wa = tl.load(w_ptr + 2 * i, mask=m, other=0.0).to(tl.float32)
        wb = tl.load(w_ptr + 2 * i + 1, mask=m, other=0.0).to(tl.float32)
        a = (a * rinv * wa).to(tl.bfloat16).to(tl.float32)
        b = (b * rinv * wb).to(tl.bfloat16).to(tl.float32)
        if ROPE:
            j = i % HALF_HEAD
            cs = tl.load(cos_ptr + tok * HALF_HEAD + j, mask=m, other=1.0)
            sn = tl.load(sin_ptr + tok * HALF_HEAD + j, mask=m, other=0.0)
            oa = a * cs - b * sn
            ob = a * sn + b * cs
        else:
            oa = a
            ob = b
        tl.store(o_ptr + base + 2 * i, oa.to(tl.bfloat16), mask=m)
        tl.store(o_ptr + base + 2 * i + 1, ob.to(tl.bfloat16), mask=m)


def qk_rms_rope(q, k, wq, wk, cos, sin, rows_per_batch, eps, head_dim=128, rope=True, chunk=512, num_warps=4):
    """q, k: (rows, D) bf16 contiguous; cos/sin: (rows_per_batch, head_dim/2) fp32 (pair i of a head
    uses entry i). k may be None (cross-attention query alone). Returns (q_out, k_out)."""
    rows, D = q.shape
    qo = torch.empty_like(q)
    ko = torch.empty_like(k) if k is not None else qo
    P = D // 2
    grid = (rows, 2 if k is not None else 1)
    _qk_rms_rope_kernel[grid](qo, ko, q, k if k is not None else q, wq, wk if wk is not None else wq,
                              cos if cos is not None else q, sin if sin is not None else q, rows_per_batch,
                              P, eps, HALF_HEAD=head_dim // 2, ROPE=rope, CHUNK=chunk,
                              NCHUNK=triton.cdiv(P, chunk), num_warps=num_warps)
    return qo, (ko if k is not None else None)


@triton.jit
def _gelu_kernel(o_ptr, q_ptr, s_ptr, x_ptr, rows, F, num_groups, column_blocks, BLOCK_ROWS: tl.constexpr,
                 BLOCK_COLS: tl.constexpr, OUT_FP8: tl.constexpr):
    rp = tl.program_id(0)
    cp = tl.program_id(1)
    row = rp * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = cp * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    rmask = row < rows
    m = rmask[:, None] & (col < F)[None, :]
    off = row[:, None].to(tl.int64) * F + col[None, :]
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0          # tanh
    y = (0.5 * x * (1.0 + t)).to(tl.bfloat16)
    if OUT_FP8:
        v = y.to(tl.float32)
        vals = tl.reshape(v, [BLOCK_ROWS, BLOCK_COLS // 32, 32])
        amax = tl.max(tl.abs(vals), axis=2)
        sb, inv = _mx_e8m0_from_amax(amax)
        qv = tl.reshape(vals * inv[:, :, None], [BLOCK_ROWS, BLOCK_COLS])
        tl.store(q_ptr + off, qv.to(tl.float8e4nv), mask=m)
        grp = cp * (BLOCK_COLS // 32) + tl.arange(0, BLOCK_COLS // 32)
        tl.store(s_ptr + _mx_scale_offsets(row[:, None], grp[None, :], column_blocks), sb.to(tl.uint8),
                 mask=rmask[:, None] & (grp < num_groups)[None, :])
    else:
        tl.store(o_ptr + off, y, mask=m)


def gelu_tanh(x, out_fp8=False, block_rows=16, block_cols=512, num_warps=4):
    rows, F = x.shape
    if out_fp8:
        q, s = _allocate(rows, F, x.device)
        o = x
    else:
        o = torch.empty_like(x)
        q, s = x, x
    ng = F // 32
    _gelu_kernel[(triton.cdiv(rows, block_rows), triton.cdiv(F, block_cols))](
        o, q, s, x, rows, F, ng, -(-ng // 4), BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols, OUT_FP8=out_fp8,
        num_warps=num_warps)
    return MXActivation(q, s, (rows, F), torch.bfloat16) if out_fp8 else o


def mx_quant(x):
    import mxfp8
    q, s = mxfp8.mxfp8_quantize_swizzled(x)
    return MXActivation(q, s, x.shape, x.dtype)


# -- v2: coalesced pair loads for q/k; two-pass LayerNorm with per-chunk Chan/Welford statistics --

@triton.jit
def _qk_rms_rope_v2_kernel(qo_ptr, ko_ptr, q_ptr, k_ptr, wq_ptr, wk_ptr, cos_ptr, sin_ptr, rows_per_batch,
                           P, eps, HALF_HEAD: tl.constexpr, ROPE: tl.constexpr, CHUNK: tl.constexpr,
                           NCHUNK: tl.constexpr):
    row = tl.program_id(0)
    which = tl.program_id(1)
    if which == 0:
        x_ptr = q_ptr
        o_ptr = qo_ptr
        w_ptr = wq_ptr
    else:
        x_ptr = k_ptr
        o_ptr = ko_ptr
        w_ptr = wk_ptr
    base = row.to(tl.int64) * (2 * P)
    tok = (row % rows_per_batch).to(tl.int64)
    acc = tl.zeros([2 * CHUNK], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        e = c * 2 * CHUNK + tl.arange(0, 2 * CHUNK)
        x = tl.load(x_ptr + base + e, mask=e < 2 * P, other=0.0).to(tl.float32)
        acc += x * x
    rinv = tl.math.rsqrt(tl.sum(acc, axis=0) / (2 * P) + eps)
    for c in tl.static_range(NCHUNK):
        e = c * 2 * CHUNK + tl.arange(0, 2 * CHUNK)
        m = e < 2 * P
        x = tl.load(x_ptr + base + e, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + e, mask=m, other=0.0).to(tl.float32)
        y = (x * rinv * w).to(tl.bfloat16).to(tl.float32)
        if ROPE:
            a, b = tl.split(tl.reshape(y, [CHUNK, 2]))
            i = c * CHUNK + tl.arange(0, CHUNK)
            j = i % HALF_HEAD
            pm = i < P
            cs = tl.load(cos_ptr + tok * HALF_HEAD + j, mask=pm, other=1.0)
            sn = tl.load(sin_ptr + tok * HALF_HEAD + j, mask=pm, other=0.0)
            y = tl.reshape(tl.join(a * cs - b * sn, a * sn + b * cs), [2 * CHUNK])
        tl.store(o_ptr + base + e, y.to(tl.bfloat16), mask=m)


def qk_rms_rope_v2(q, k, wq, wk, cos, sin, rows_per_batch, eps, head_dim=128, rope=True, chunk=512, num_warps=4):
    rows, D = q.shape
    qo = torch.empty_like(q)
    ko = torch.empty_like(k) if k is not None else qo
    P = D // 2
    _qk_rms_rope_v2_kernel[(rows, 2 if k is not None else 1)](
        qo, ko, q, k if k is not None else q, wq, wk if wk is not None else wq,
        cos if cos is not None else q, sin if sin is not None else q, rows_per_batch, P, eps,
        HALF_HEAD=head_dim // 2, ROPE=rope, CHUNK=chunk, NCHUNK=triton.cdiv(P, chunk), num_warps=num_warps)
    return qo, (ko if k is not None else None)


@triton.jit
def _ln_mod_v2_kernel(out_ptr, q_ptr, s_ptr, hout_ptr, res_ptr, br_ptr, gate_ptr, w_ptr, b_ptr,
                      scale_ptr, shift_ptr, N, rows_per_batch, mod_stride, gate_stride, num_groups, column_blocks,
                      eps, HAS_BRANCH: tl.constexpr, HAS_GATE: tl.constexpr, AFFINE: tl.constexpr,
                      MOD: tl.constexpr, OUT_FP8: tl.constexpr, CHUNK: tl.constexpr, NCHUNK: tl.constexpr):
    row = tl.program_id(0)
    bidx = (row // rows_per_batch).to(tl.int64)
    base = row.to(tl.int64) * N
    mbase = bidx * mod_stride
    gbase = bidx * gate_stride
    # pass 1: residual (stored) and running mean / M2, merged chunk by chunk (Chan et al.)
    mean = 0.0
    m2 = 0.0
    cnt = 0.0
    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        m = cols < N
        x = tl.load(res_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        if HAS_BRANCH:
            b = tl.load(br_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
            if HAS_GATE:
                x = x + b * tl.load(gate_ptr + gbase + cols, mask=m, other=0.0).to(tl.float32)
            else:
                x = x + b
            x = x.to(tl.bfloat16).to(tl.float32)
            tl.store(hout_ptr + base + cols, x.to(tl.bfloat16), mask=m)
        nc = tl.sum(m.to(tl.float32), axis=0)
        mc = tl.sum(x, axis=0) / nc
        d = tl.where(m, x - mc, 0.0)
        m2c = tl.sum(d * d, axis=0)
        tot = cnt + nc
        delta = mc - mean
        mean = mean + delta * nc / tot
        m2 = m2 + m2c + delta * delta * cnt * nc / tot
        cnt = tot
    rstd = tl.math.rsqrt(m2 / N + eps)
    if HAS_BRANCH:
        tl.debug_barrier()          # pass 2 re-reads residual elements other threads stored
    # pass 2: normalize from the stored residual (or the input when there is no branch)
    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        m = cols < N
        if HAS_BRANCH:
            x = tl.load(hout_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        else:
            x = tl.load(res_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        if AFFINE:
            y = y * tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32) \
                + tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        if MOD:
            sc = tl.load(scale_ptr + mbase + cols, mask=m, other=0.0).to(tl.float32)
            sh = tl.load(shift_ptr + mbase + cols, mask=m, other=0.0).to(tl.float32)
            y = y * (1.0 + sc) + sh
        _emit(out_ptr, q_ptr, s_ptr, row, cols, m, y, N, num_groups, column_blocks, CHUNK, c, OUT_FP8)


def ln_mod_v2(res, rows_per_batch, eps, *, branch=None, gate=None, weight=None, bias=None, scale=None,
              shift=None, out_fp8=False, chunk=1024, num_warps=4):
    rows, N = res.shape
    hout = torch.empty_like(res) if branch is not None else res
    mods = [t for t in (scale, shift) if t is not None]
    mod_stride = mods[0].stride(0) if mods else 0
    gate_stride = gate.stride(0) if gate is not None else 0
    dummy = res
    if out_fp8:
        q, s = _allocate(rows, N, res.device)
        out = dummy
    else:
        out = torch.empty_like(res)
        q, s = dummy, dummy
    ng = N // 32
    _ln_mod_v2_kernel[(rows,)](
        out, q, s, hout, res, branch if branch is not None else dummy, gate if gate is not None else dummy,
        weight if weight is not None else dummy, bias if bias is not None else dummy,
        scale if scale is not None else dummy, shift if shift is not None else dummy,
        N, rows_per_batch, mod_stride, gate_stride, ng, -(-ng // 4), eps,
        HAS_BRANCH=branch is not None, HAS_GATE=gate is not None, AFFINE=weight is not None,
        MOD=scale is not None, OUT_FP8=out_fp8, CHUNK=chunk, NCHUNK=triton.cdiv(N, chunk), num_warps=num_warps)
    normed = MXActivation(q, s, (rows, N), torch.bfloat16) if out_fp8 else out
    return (hout if branch is not None else None), normed


# -- per-tensor FP8 quantization (attention inputs): one pass for amax, one pass to quantize ----------

@triton.jit
def _amax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    tl.atomic_max(out_ptr, tl.max(tl.abs(x), axis=0))


@triton.jit
def _quant_pt_kernel(x_ptr, q_ptr, amax_ptr, scale_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.maximum(tl.load(amax_ptr), 1e-12) / 448.0
    if pid == 0:
        tl.store(scale_ptr, s)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(q_ptr + offs, (x / s).to(tl.float8e4nv), mask=m)


def fp8_quant_pertensor(x, block=8192):
    """bf16 contiguous -> (E4M3 tensor of the same shape, fp32 scale (1,1,1,1)); x ~= q * scale."""
    n = x.numel()
    amax = torch.zeros(1, device=x.device, dtype=torch.float32)
    scale = torch.empty(1, 1, 1, 1, device=x.device, dtype=torch.float32)
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    grid = (triton.cdiv(n, block),)
    _amax_kernel[grid](x, amax, n, BLOCK=block, num_warps=8)
    _quant_pt_kernel[grid](x, q, amax, scale, n, BLOCK=block, num_warps=8)
    return q, scale
