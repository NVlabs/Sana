"""Triton kernels for the non-attention part of the MiniMax-H3 block, tuned for GB200.

qknorm_rope_v2   per-head RMSNorm + partial rotary over a [ROWS, head_dim] tile per program. The
                 shipped kernel runs one program per (token, head) row of 128 values, which leaves
                 the memory system mostly idle (162 ms per forward against a ~16 ms traffic floor).
residual_gate    `residual + gate[idx] * branch`, used once after the last block.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _qknorm_rope_v2_kernel(
    out_ptr, x_ptr, w_ptr, cos_ptr, sin_ptr,
    n_rows, heads, x_tok_stride, cos_stride, eps,
    HEAD_DIM: tl.constexpr, ROT: tl.constexpr, HALF: tl.constexpr, ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    tok = rows // heads
    head = rows % heads
    cols = tl.arange(0, HEAD_DIM)
    base = tok.to(tl.int64) * x_tok_stride + head * HEAD_DIM

    x = tl.load(x_ptr + base[:, None] + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / HEAD_DIM
    r = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + cols).to(tl.float32)
    normed = x * r[:, None] * w[None, :]

    in_rot = cols < ROT
    first = cols < HALF
    partner = tl.where(first, cols + HALF, cols - HALF)
    pmask = rmask[:, None] & in_rot[None, :]
    px = tl.load(x_ptr + base[:, None] + partner[None, :], mask=pmask, other=0.0).to(tl.float32)
    pw = tl.load(w_ptr + partner, mask=in_rot, other=0.0).to(tl.float32)
    pn = px * r[:, None] * pw[None, :]
    rotated = tl.where(first[None, :], -pn, pn)

    cs = tok.to(tl.int64)[:, None] * cos_stride + cols[None, :]
    c = tl.load(cos_ptr + cs, mask=pmask, other=1.0).to(tl.float32)
    s = tl.load(sin_ptr + cs, mask=pmask, other=0.0).to(tl.float32)
    out = tl.where(in_rot[None, :], normed * c + rotated * s, normed)
    tl.store(out_ptr + rows.to(tl.int64)[:, None] * HEAD_DIM + cols[None, :],
             out.to(out_ptr.dtype.element_ty), mask=rmask[:, None])


def qknorm_rope_v2(x, weight, cos, sin, eps, rows_per_program=64, num_warps=8):
    """`x` is `(B, seq, heads, head_dim)` (or `(seq, heads, head_dim)`) whose last two dims are
    packed; the token stride may be wider than heads*head_dim (a view into a fused QKV output)."""
    shape = x.shape
    heads, head_dim = shape[-2], shape[-1]
    if x.stride(-1) != 1 or x.stride(-2) != head_dim:
        x = x.contiguous()
    tokens = x.numel() // (heads * head_dim)
    tok_stride = x.stride(-3)
    rot = cos.shape[-1]
    if cos.stride(-1) != 1:
        cos, sin = cos.contiguous(), sin.contiguous()
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    n_rows = tokens * heads
    grid = (triton.cdiv(n_rows, rows_per_program),)
    _qknorm_rope_v2_kernel[grid](
        out, x, weight, cos, sin, n_rows, heads, tok_stride, cos.stride(0), eps,
        HEAD_DIM=head_dim, ROT=rot, HALF=rot // 2, ROWS=rows_per_program, num_warps=num_warps,
    )
    return out


@triton.jit
def _residual_gate_kernel(out_ptr, res_ptr, br_ptr, gate_ptr, idx_ptr, n_cols, table_stride,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    off = row.to(tl.int64) * n_cols + cols
    t = tl.load(idx_ptr + row)
    res = tl.load(res_ptr + off, mask=m, other=0.0).to(tl.float32)
    br = tl.load(br_ptr + off, mask=m, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + t * table_stride + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + off, (res + g * br).to(out_ptr.dtype.element_ty), mask=m)


def residual_gate(residual, branch, gate, index):
    cols = residual.shape[-1]
    res = residual.reshape(-1, cols).contiguous()
    br = branch.reshape(-1, cols).contiguous()
    gate = gate if gate.stride(-1) == 1 else gate.contiguous()
    out = torch.empty_like(res)
    block = triton.next_power_of_2(cols)
    _residual_gate_kernel[(res.shape[0],)](out, res, br, gate, index, cols, gate.stride(0),
                                           BLOCK=block, num_warps=8)
    return out.view_as(residual)


# -- q and k in one launch, a block of tokens per program, looping over heads --------------------
# cos/sin and the two norm weights are loaded once per program and reused across all 56 heads.

@triton.jit
def _qk_rope_v3_kernel(
    oq_ptr, ok_ptr, xq_ptr, xk_ptr, wq_ptr, wk_ptr, cos_ptr, sin_ptr,
    seq, xq_tok_stride, xk_tok_stride, cos_stride, eps,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, ROT: tl.constexpr, HALF: tl.constexpr,
    TOK: tl.constexpr,
):
    pid = tl.program_id(0)
    which = tl.program_id(1)
    toks = pid * TOK + tl.arange(0, TOK)
    tmask = toks < seq
    cols = tl.arange(0, HEAD_DIM)
    in_rot = cols < ROT
    first = cols < HALF
    partner = tl.where(first, cols + HALF, cols - HALF)
    pmask = tmask[:, None] & in_rot[None, :]
    cs = toks.to(tl.int64)[:, None] * cos_stride + cols[None, :]
    c = tl.load(cos_ptr + cs, mask=pmask, other=1.0).to(tl.float32)
    s = tl.load(sin_ptr + cs, mask=pmask, other=0.0).to(tl.float32)
    if which == 0:
        x_ptr = xq_ptr
        o_ptr = oq_ptr
        w_ptr = wq_ptr
        tok_stride = xq_tok_stride
    else:
        x_ptr = xk_ptr
        o_ptr = ok_ptr
        w_ptr = wk_ptr
        tok_stride = xk_tok_stride
    w = tl.load(w_ptr + cols).to(tl.float32)
    pw = tl.load(w_ptr + partner, mask=in_rot, other=0.0).to(tl.float32)
    xbase = toks.to(tl.int64)[:, None] * tok_stride
    obase = toks.to(tl.int64)[:, None] * (HEADS * HEAD_DIM)
    for hh in range(HEADS):
        off = xbase + hh * HEAD_DIM
        x = tl.load(x_ptr + off + cols[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / HEAD_DIM
        r = tl.math.rsqrt(var + eps)
        normed = x * r[:, None] * w[None, :]
        px = tl.load(x_ptr + off + partner[None, :], mask=pmask, other=0.0).to(tl.float32)
        pn = px * r[:, None] * pw[None, :]
        rotated = tl.where(first[None, :], -pn, pn)
        out = tl.where(in_rot[None, :], normed * c + rotated * s, normed)
        tl.store(o_ptr + obase + hh * HEAD_DIM + cols[None, :], out.to(o_ptr.dtype.element_ty),
                 mask=tmask[:, None])


def qknorm_rope_v3(q, k, wq, wk, cos, sin, eps, tok=8, num_warps=4):
    """q, k: `(B, seq, heads, head_dim)` with packed (heads, head_dim); returns (q_out, k_out)."""
    heads, head_dim = q.shape[-2], q.shape[-1]
    for t in (q, k):
        assert t.stride(-1) == 1 and t.stride(-2) == head_dim
    seq = q.numel() // (heads * head_dim)
    rot = cos.shape[-1]
    oq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ok = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    grid = (triton.cdiv(seq, tok), 2)
    _qk_rope_v3_kernel[grid](
        oq, ok, q, k, wq, wk, cos, sin, seq, q.stride(-3), k.stride(-3), cos.stride(0), eps,
        HEADS=heads, HEAD_DIM=head_dim, ROT=rot, HALF=rot // 2, TOK=tok, num_warps=num_warps,
    )
    return oq, ok


# -- MXFP8 producers ---------------------------------------------------------------------------
from mxfp8 import MXActivation, _allocate, _mx_e8m0_from_amax, _mx_scale_offsets  # noqa: E402


@triton.jit
def _swiglu_mx_v2_kernel(q_ptr, s_ptr, x_ptr, rows, F, num_groups, column_blocks, x_row_stride,
                         BLOCK_ROWS: tl.constexpr, GROUPS: tl.constexpr):
    """2-D tiles of [BLOCK_ROWS, GROUPS*32] outputs; SwiGLU has no row reduction, so no need for a
    whole row per program."""
    rp = tl.program_id(0)
    gp = tl.program_id(1)
    row = rp * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    group = gp * GROUPS + tl.arange(0, GROUPS)
    col = gp * (GROUPS * 32) + tl.arange(0, GROUPS * 32)
    rmask = row < rows
    m = rmask[:, None] & (col < F)[None, :]
    base = x_ptr + row[:, None].to(tl.int64) * x_row_stride + col[None, :]
    v = tl.load(base, mask=m, other=0.0).to(tl.float32)
    g = tl.load(base + F, mask=m, other=0.0).to(tl.float32)
    out = (v * (g * tl.sigmoid(g))).to(tl.bfloat16).to(tl.float32)
    vals = tl.reshape(out, [BLOCK_ROWS, GROUPS, 32])
    amax = tl.max(tl.abs(vals), axis=2)
    sb, inv = _mx_e8m0_from_amax(amax)
    qv = tl.reshape(vals * inv[:, :, None], [BLOCK_ROWS, GROUPS * 32])
    tl.store(q_ptr + row[:, None].to(tl.int64) * F + col[None, :], qv.to(tl.float8e4nv), mask=m)
    tl.store(s_ptr + _mx_scale_offsets(row[:, None], group[None, :], column_blocks), sb.to(tl.uint8),
             mask=rmask[:, None] & (group < num_groups)[None, :])


SWIGLU_CFGS = [dict(block_rows=r, groups=g, num_warps=w) for r, g, w in
               ((32, 8, 4), (32, 16, 8), (64, 8, 8), (16, 16, 4), (128, 4, 8), (64, 16, 8))]


def swiglu_mx_v2(x, block_rows=32, groups=8, num_warps=4):
    F = x.shape[-1] // 2
    flat = x.reshape(-1, x.shape[-1])
    rows = flat.shape[0]
    q, s = _allocate(rows, F, x.device)
    ng = F // 32
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(ng, groups))
    _swiglu_mx_v2_kernel[grid](q, s, flat, rows, F, ng, -(-ng // 4), flat.stride(0),
                               BLOCK_ROWS=block_rows, GROUPS=groups, num_warps=num_warps)
    return MXActivation(q, s, (*x.shape[:-1], F), x.dtype)


@triton.jit
def _resnorm_mx_v2_kernel(hid_ptr, q_ptr, s_ptr, res_ptr, br_ptr, w_ptr, gate_ptr, scale_ptr, shift_ptr,
                          idx_ptr, N, num_indices, num_groups, column_blocks, eps, row_stride, tstride,
                          CHUNK: tl.constexpr, NCHUNK: tl.constexpr):
    """Two passes over column chunks of one row. Pass 1 forms and stores the residual and the sum of
    squares; pass 2 re-reads the inputs (L2-resident) and emits the MXFP8 normalized output.
    Smaller live state than holding the padded 8192-wide row, so more rows are in flight per SM."""
    row = tl.program_id(0)
    t = tl.load(idx_ptr + (row % num_indices))
    base = row.to(tl.int64) * row_stride
    tb = t.to(tl.int64) * tstride
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        m = cols < N
        r = tl.load(res_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(br_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + tb + cols, mask=m, other=0.0).to(tl.float32)
        hval = r + g * b
        tl.store(hid_ptr + base + cols, hval.to(hid_ptr.dtype.element_ty), mask=m)
        acc += hval * hval
    rinv = tl.math.rsqrt(tl.sum(acc, axis=0) / N + eps)
    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        m = cols < N
        r = tl.load(res_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(br_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + tb + cols, mask=m, other=0.0).to(tl.float32)
        hval = r + g * b
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        sc = tl.load(scale_ptr + tb + cols, mask=m, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + tb + cols, mask=m, other=0.0).to(tl.float32)
        out = (hval * rinv * w * (1.0 + sc) + sh).to(tl.bfloat16).to(tl.float32)
        vals = tl.reshape(out, [CHUNK // 32, 32])
        amax = tl.max(tl.abs(vals), axis=1)
        sb, inv = _mx_e8m0_from_amax(amax)
        qv = tl.reshape(vals * inv[:, None], [CHUNK])
        tl.store(q_ptr + row.to(tl.int64) * N + cols, qv.to(tl.float8e4nv), mask=m)
        grp = c * (CHUNK // 32) + tl.arange(0, CHUNK // 32)
        tl.store(s_ptr + _mx_scale_offsets(row, grp, column_blocks), sb.to(tl.uint8), mask=grp < num_groups)


RESNORM_CFGS = [dict(chunk=c, num_warps=w) for c, w in
                ((256, 2), (512, 4), (1024, 4), (1024, 8), (2048, 8))]


def resnorm_mx_v2(residual, branch, gate, weight, scale, shift, index, eps, chunk=1024, num_warps=4):
    N = residual.shape[-1]
    res = residual.reshape(-1, N)
    br = branch.reshape(-1, N)
    if not res.is_contiguous():
        res = res.contiguous()
    if not br.is_contiguous():
        br = br.contiguous()
    gate = gate if gate.stride(-1) == 1 else gate.contiguous()
    scale = scale if scale.stride(-1) == 1 else scale.contiguous()
    shift = shift if shift.stride(-1) == 1 else shift.contiguous()
    assert gate.stride(0) == scale.stride(0) == shift.stride(0)
    rows = res.shape[0]
    hid = torch.empty_like(res)
    q, s = _allocate(rows, N, residual.device)
    ng = N // 32
    _resnorm_mx_v2_kernel[(rows,)](hid, q, s, res, br, weight, gate, scale, shift, index, N, index.numel(),
                                   ng, -(-ng // 4), eps, res.stride(0), gate.stride(0),
                                   CHUNK=chunk, NCHUNK=triton.cdiv(N, chunk), num_warps=num_warps)
    return hid.view_as(residual), MXActivation(q, s, residual.shape, residual.dtype)


# -- BF16 counterparts (lossless path) ---------------------------------------------------------

@triton.jit
def _swiglu_bf16_v2_kernel(o_ptr, x_ptr, rows, F, x_row_stride, BLOCK_ROWS: tl.constexpr,
                           BLOCK_COLS: tl.constexpr):
    rp = tl.program_id(0)
    cp = tl.program_id(1)
    row = rp * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = cp * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    m = (row < rows)[:, None] & (col < F)[None, :]
    base = x_ptr + row[:, None].to(tl.int64) * x_row_stride + col[None, :]
    v = tl.load(base, mask=m, other=0.0).to(tl.float32)
    g = tl.load(base + F, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + row[:, None].to(tl.int64) * F + col[None, :],
             (v * (g * tl.sigmoid(g))).to(o_ptr.dtype.element_ty), mask=m)


def swiglu_bf16_v2(x, block_rows=16, block_cols=512, num_warps=4):
    F = x.shape[-1] // 2
    flat = x.reshape(-1, x.shape[-1])
    rows = flat.shape[0]
    out = torch.empty(rows, F, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(F, block_cols))
    _swiglu_bf16_v2_kernel[grid](out, flat, rows, F, flat.stride(0), BLOCK_ROWS=block_rows,
                                 BLOCK_COLS=block_cols, num_warps=num_warps)
    return out.view(*x.shape[:-1], F)


@triton.jit
def _resnorm_bf16_v2_kernel(hid_ptr, o_ptr, res_ptr, br_ptr, w_ptr, gate_ptr, scale_ptr, shift_ptr,
                            idx_ptr, N, num_indices, eps, row_stride, tstride,
                            CHUNK: tl.constexpr, NCHUNK: tl.constexpr):
    row = tl.program_id(0)
    t = tl.load(idx_ptr + (row % num_indices))
    base = row.to(tl.int64) * row_stride
    tb = t.to(tl.int64) * tstride
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        m = cols < N
        r = tl.load(res_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(br_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + tb + cols, mask=m, other=0.0).to(tl.float32)
        hval = r + g * b
        tl.store(hid_ptr + base + cols, hval.to(hid_ptr.dtype.element_ty), mask=m)
        acc += hval * hval
    rinv = tl.math.rsqrt(tl.sum(acc, axis=0) / N + eps)
    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        m = cols < N
        r = tl.load(res_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(br_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + tb + cols, mask=m, other=0.0).to(tl.float32)
        hval = r + g * b
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        sc = tl.load(scale_ptr + tb + cols, mask=m, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + tb + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(o_ptr + base + cols, (hval * rinv * w * (1.0 + sc) + sh).to(o_ptr.dtype.element_ty), mask=m)


def resnorm_bf16_v2(residual, branch, gate, weight, scale, shift, index, eps, chunk=2048, num_warps=8):
    N = residual.shape[-1]
    res = residual.reshape(-1, N).contiguous()
    br = branch.reshape(-1, N).contiguous()
    gate = gate if gate.stride(-1) == 1 else gate.contiguous()
    scale = scale if scale.stride(-1) == 1 else scale.contiguous()
    shift = shift if shift.stride(-1) == 1 else shift.contiguous()
    assert gate.stride(0) == scale.stride(0) == shift.stride(0)
    hid = torch.empty_like(res)
    out = torch.empty_like(res)
    _resnorm_bf16_v2_kernel[(res.shape[0],)](hid, out, res, br, weight, gate, scale, shift, index, N,
                                            index.numel(), eps, res.stride(0), gate.stride(0),
                                            CHUNK=chunk, NCHUNK=triton.cdiv(N, chunk), num_warps=num_warps)
    return hid.view_as(residual), out.view_as(residual)
