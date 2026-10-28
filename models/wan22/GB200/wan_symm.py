"""Ulysses all-to-all as direct peer stores into symmetric memory (the H3 "V1" scheme).

Each rank writes its token rows of q/k/v, head group j, straight into rank j's receive buffers in their
final (S, H/P, D) layout, then one device-side barrier; after attention it writes its head group of the
output back into each owner's (S/P, H, D) buffer, then a second barrier.  No NCCL kernel, no layout copy
on either side.  Batch 1 per rank (CFG parallel), bf16.
"""
import torch
import triton
import triton.language as tl

BR = 32


@triton.jit
def _scatter_rows(src, peer_ptrs, dst_off, T, D: tl.constexpr,
                  s_stride, s_row_step, s_col_step, d_row0, d_stride, d_col0,
                  BR: tl.constexpr):
    """dst_j[(d_row0 + r)*d_stride + d_col0 + h*D + d] = src[(j*s_row_step + r)*s_stride + j*s_col_step + h*D + d]"""
    j = tl.program_id(0)
    rb = tl.program_id(1)
    h = tl.program_id(2)
    r = rb * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    m = (r < T)[:, None]
    s = src + (j * s_row_step + r)[:, None].to(tl.int64) * s_stride + j * s_col_step + h * D + d[None, :]
    x = tl.load(s, mask=m)
    base = tl.load(peer_ptrs + j).to(tl.pointer_type(tl.bfloat16))
    o = base + dst_off + (d_row0 + r)[:, None].to(tl.int64) * d_stride + d_col0 + h * D + d[None, :]
    tl.store(o, x, mask=m)


@triton.jit
def _scatter_qkv(q, k, v, peer_ptrs, oq, ok, ov, T, HG: tl.constexpr, D: tl.constexpr,
                 s_stride, s_col_step, row0, BR: tl.constexpr):
    """q/k/v [T, P*HG*D] token rows -> peer j's [S, HG, D] buffers at rows row0.., heads j*HG.. (one launch)."""
    j = tl.program_id(0)
    rb = tl.program_id(1)
    th = tl.program_id(2)
    t = th // HG
    h = th % HG
    if t == 0:
        src = q
        off = oq
    elif t == 1:
        src = k
        off = ok
    else:
        src = v
        off = ov
    r = rb * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    m = (r < T)[:, None]
    x = tl.load(src + r[:, None].to(tl.int64) * s_stride + j * s_col_step + h * D + d[None, :], mask=m)
    base = tl.load(peer_ptrs + j).to(tl.pointer_type(tl.bfloat16))
    tl.store(base + off + (row0 + r)[:, None].to(tl.int64) * (HG * D) + h * D + d[None, :], x, mask=m)


class SymmUlysses:
    """Per (group, shape) symmetric buffers: q/k/v receive [S, H/P, D] and output receive [S/P, H, D]."""

    def __init__(self, group, P, rank):
        self.group, self.P, self.rank = group, P, rank
        self.bufs = {}

    def _get(self, S, H, Dh):
        key = (S, H, Dh)
        if key not in self.bufs:
            import torch.distributed._symmetric_memory as symm_mem
            if hasattr(symm_mem, "enable_symm_mem_for_group"):
                symm_mem.enable_symm_mem_for_group(self.group.group_name)
            Hp, Sl = H // self.P, S // self.P
            nkv, no = S * Hp * Dh, Sl * H * Dh
            buf = symm_mem.empty(3 * nkv + no, dtype=torch.bfloat16, device="cuda")
            hdl = symm_mem.rendezvous(buf, self.group.group_name)
            ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device="cuda")
            views = (buf[0:nkv].view(1, S, Hp, Dh), buf[nkv:2 * nkv].view(1, S, Hp, Dh),
                     buf[2 * nkv:3 * nkv].view(1, S, Hp, Dh), buf[3 * nkv:].view(1, Sl, H, Dh))
            self.bufs[key] = (hdl, ptrs, (0, nkv, 2 * nkv, 3 * nkv), views)
        return self.bufs[key]

    def __call__(self, q, k, v, attn):
        """q/k/v [Sl, H*Dh] (my token rows, all heads, contiguous) -> attention output [Sl, H, Dh]."""
        P, me = self.P, self.rank
        Sl, HD = q.shape
        Dh = 128
        H = HD // Dh
        Hp, S = H // P, Sl * P
        hdl, ptrs, (oq, ok, ov, oo), (qb, kb, vb, ob) = self._get(S, H, Dh)
        _scatter_qkv[(P, triton.cdiv(Sl, BR), 3 * Hp)](q, k, v, ptrs, oq, ok, ov, Sl, Hp, Dh, HD, Hp * Dh,
                                                     me * Sl, BR=BR, num_warps=4)
        hdl.barrier(channel=0)
        o = attn(qb, kb, vb).reshape(S, Hp, Dh)
        if not o.is_contiguous():
            o = o.contiguous()
        # my head group of every peer's token rows -> peer j's output rows, heads me*Hp..
        _scatter_rows[(P, triton.cdiv(Sl, BR), Hp)](o, ptrs, oo, Sl, Dh, Hp * Dh, Sl, 0, 0, H * Dh, me * Hp * Dh,
                                                   BR=BR, num_warps=4)
        hdl.barrier(channel=1)
        return ob[0]
