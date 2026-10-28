"""Sol-style sparse attention at FA4 block granularity.

Sol-Attn routes 64x64 blocks and computes the exact part with an M=64 tcgen05 kernel, which runs the
tensor cores at half rate (0.57 PF/s on real Wan data vs cuDNN dense 1.56).  Here the same idea runs on
FA4's native tiles:

  route   q256 block means . kv128 block means, top-rho per q block plus the diagonal band
  exact   FA4 block-sparse forward over the routed (q256, kv128) blocks (~1.1x its density ideal)
  approx  every other kv64 block enters the softmax as one centroid (mean k, mean v, weight = its length),
          the Sol approximation, in a Triton kernel that also merges the two parts by LSE

On a real 5B layer (step 25), rho=0.35 gives rel_l2 0.122 vs dense (Sol release: 0.133) with an exact
part of 4.6 ms vs Sol's 9.1 ms.
"""
import math
import os

import torch
import triton
import triton.language as tl

QB, KB, CB = 256, 128, 64
LOG2E = 1.4426950408889634


@triton.autotune(
    configs=[triton.Config({"BM": bm, "BC": bc}, num_warps=w, num_stages=st)
             for bm in (64, 128) for bc in (64, 128) for w in (4, 8) for st in (2, 3)],
    key=["S", "NC", "H"])
@triton.jit
def _approx_merge_kernel(q, kc, vc, logc, sel, o_e, lse_e, out,
                         S, H, NC, NQ, NK, scale_log2,
                         D: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr, QBLK: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < S
    d = tl.arange(0, D)
    row_off = (b * S + rows).to(tl.int64)[:, None] * (H * D) + h * D + d[None, :]
    qt = tl.load(q + row_off, mask=rmask[:, None], other=0.0)
    qblk = (pid_m * BM) // QBLK
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for c0 in range(0, NC, BC):
        c = c0 + tl.arange(0, BC)
        cm = c < NC
        c_off = (b * NC + c).to(tl.int64)[:, None] * (H * D) + h * D + d[None, :]
        kt = tl.load(kc + c_off, mask=cm[:, None], other=0.0)
        vt = tl.load(vc + c_off, mask=cm[:, None], other=0.0)
        exact = tl.load(sel + ((b * H + h) * NQ + qblk) * NK + c // 2, mask=cm, other=1)
        valid = cm & (exact == 0)
        lc = tl.load(logc + c, mask=cm, other=0.0)
        s = tl.dot(qt, tl.trans(kt)).to(tl.float32) * scale_log2 + lc[None, :]
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt).to(tl.float32)
        m_i = m_new
    # merge with the exact part: both are unnormalised over disjoint key sets
    le = tl.load(lse_e + (b * H + h).to(tl.int64) * S + rows, mask=rmask, other=0.0) * 1.4426950408889634
    mx = tl.maximum(le, m_i)
    we = tl.exp2(le - mx)
    wa = tl.exp2(m_i - mx)                           # 0 when no centroid is approximate
    oe = tl.load(o_e + row_off, mask=rmask[:, None], other=0.0).to(tl.float32)
    y = (oe * we[:, None] + acc * wa[:, None]) / (we + l_i * wa)[:, None]
    tl.store(out + row_off, y.to(out.dtype.element_ty), mask=rmask[:, None])


def _lens64(S, dev):
    lens = [float(CB)] * (S // CB) + ([float(S % CB)] if S % CB else [])
    return torch.tensor(lens, device=dev)


def _means64(x):
    """(B, S, H, D) bf16 -> (B, ceil(S/64), H, D) fp32 block means (one bf16 read, fp32 accumulate)."""
    B, S, H, D = x.shape
    full = S // CB
    out = torch.empty((B, -(-S // CB), H, D), device=x.device, dtype=torch.float32)
    torch.mean(x[:, :full * CB].view(B, full, CB, H, D), 2, dtype=torch.float32, out=out[:, :full])
    if S % CB:
        torch.mean(x[:, full * CB:], 1, keepdim=True, dtype=torch.float32, out=out[:, full:])
    return out


def _coarsen(m, lens, f):
    """Length-weighted merge of f consecutive block means: (B, n, H, D) -> (B, ceil(n/f), H, D)."""
    B, n, H, D = m.shape
    pad = -n % f
    w = torch.nn.functional.pad(lens, (0, pad))
    s = torch.nn.functional.pad(m * lens.view(1, n, 1, 1), (0, 0, 0, 0, 0, pad))
    return s.view(B, (n + pad) // f, f, H, D).sum(2) / w.view((n + pad) // f, f).sum(1).view(1, -1, 1, 1)


class FA4Sol:
    def __init__(self, rho=None):
        self.rho = float(os.environ.get("WAN_FA4SOL_RHO", "0.35")) if rho is None else rho
        self._shape = {}
        from flash_attn.cute.interface import _flash_attn_fwd
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
        self._fwd, self._BST = _flash_attn_fwd, BlockSparseTensorsTorch
        self.last_density = None
        self._graphs = {}

    def _consts(self, S, dev, sink_end=0):
        if (S, sink_end) not in self._shape:
            nq, nk = math.ceil(S / QB), math.ceil(S / KB)
            i = torch.arange(nq, device=dev)[:, None]
            j = torch.arange(nk, device=dev)[None, :]
            band = (j >= 2 * i - 1) & (j <= 2 * i + 2)
            if sink_end:                                # exact-KV sink [0, sink_end): every query keeps it
                band = band | (j * KB < sink_end)
            lens = _lens64(S, dev)
            self._shape[(S, sink_end)] = (nq, nk, band, torch.arange(nk, device=dev, dtype=torch.int32), lens, torch.log2(lens))
        return self._shape[(S, sink_end)]

    def __call__(self, q, k, v, sink_end=0):
        """q/k/v (B, S, H, 128) bf16 contiguous -> (B, S, H, 128).

        With WAN_FA4SOL_GRAPH=1 and inputs at fixed addresses (the symmetric-memory Ulysses buffers), the
        whole call is captured into a CUDA graph on its second use and replayed after that; the returned
        tensor is then the graph's static output, valid until the next call."""
        if os.environ.get("WAN_FA4SOL_GRAPH", "0") != "1":
            return self._run(q, k, v, sink_end)
        key = (q.data_ptr(), k.data_ptr(), v.data_ptr(), tuple(q.shape), sink_end)
        g = self._graphs.get(key)
        if g is None:                                   # first use: eager (compiles FA4, autotunes Triton)
            self._graphs[key] = "warm"
            return self._run(q, k, v, sink_end)
        if g == "warm":
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._run(q, k, v, sink_end)
            self._graphs[key] = g = (graph, out)
        g[0].replay()
        return g[1]

    def _run(self, q, k, v, sink_end=0):
        B, S, H, D = q.shape
        nq, nk, band, j, lens, log2lens = self._consts(S, q.device, sink_end)
        q64, k64, v64 = _means64(q), _means64(k), _means64(v)
        qbar, kbar = _coarsen(q64, lens, QB // CB), _coarsen(k64, lens, KB // CB)
        kc, vc = k64.to(torch.bfloat16), v64.to(torch.bfloat16)
        rs = torch.einsum("bqhd,bkhd->bhqk", qbar, kbar)
        top = rs.topk(max(1, round(self.rho * nk)), dim=-1).indices
        m = torch.zeros_like(rs, dtype=torch.bool).scatter_(-1, top, True) | band
        cnt = m.sum(-1, dtype=torch.int32)
        # selected block ids first, ascending: slot = running count, unselected go to a spill column
        pos = torch.where(m, torch.cumsum(m, -1, dtype=torch.int32) - 1, nk)
        idx = torch.zeros(m.shape[:-1] + (nk + 1,), device=q.device, dtype=torch.int32)
        idx.scatter_(-1, pos.long(), j.expand_as(pos))
        idx = idx[..., :nk].contiguous()
        bst = self._BST(mask_block_cnt=torch.zeros_like(cnt), mask_block_idx=torch.zeros_like(idx),
                        full_block_cnt=cnt, full_block_idx=idx, block_size=(QB, KB))
        o_e, lse_e = self._fwd(q, k, v, return_lse=True, block_sparse_tensors=bst)[:2]
        out = torch.empty_like(q)
        nc = kc.shape[1]
        _approx_merge_kernel[lambda meta: (triton.cdiv(S, meta["BM"]), B * H)](
            q, kc, vc, log2lens, m.to(torch.int8), o_e, lse_e.contiguous(), out,
            S, H, nc, nq, nk, D ** -0.5 * LOG2E, D=D, QBLK=QB)
        if os.environ.get("WAN_FA4SOL_STATS") == "1":
            self.last_density = float(m.float().mean())
        return out
