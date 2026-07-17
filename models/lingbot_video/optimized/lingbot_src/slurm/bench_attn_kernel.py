"""Microbenchmark: refiner per-rank attention workload under CP4 Ulysses.

After the Ulysses all-to-all, each rank holds the FULL packed sequence (2 batch_cfg
segments) for local_heads = 16//4 = 4 heads, head_dim 128. We compare the current
FA2-2.8.3 varlen kernel against torch SDPA (cuDNN flash / default) doing the same
block-diagonal attention as two dense per-segment calls.

This decides whether a cuDNN-SDPA kernel path is worth a full GB200 job.
"""
import os, time, torch
import torch.nn.functional as F

dev = torch.device("cuda")
dt = torch.bfloat16
torch.manual_seed(0)

# refiner per-rank shape: 2 segments (batch_cfg), n_video ~252960 + small text each.
NVIDEO = 252960
TEXT = 128            # approx text tokens per segment
SEG = NVIDEO + TEXT
NSEG = 2
Sfull = SEG * NSEG
H = 4                 # local heads after CP4 head-split
D = 128
scale = 1.0 / (D ** 0.5)

print(f"[bench] Sfull={Sfull} per-seg={SEG} H={H} D={D} dtype={dt}")

# ---- FA2 varlen (the shim path) ----
try:
    from flash_attn_interface import flash_attn_varlen_func
    HAVE_FA = True
except Exception as e:
    print("no FA:", e); HAVE_FA = False

def make_qkv_flat():
    q = torch.randn(Sfull, H, D, device=dev, dtype=dt)
    k = torch.randn(Sfull, H, D, device=dev, dtype=dt)
    v = torch.randn(Sfull, H, D, device=dev, dtype=dt)
    return q, k, v

cu = torch.tensor([0, SEG, 2*SEG], device=dev, dtype=torch.int32)

def run_fa(q, k, v):
    r = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                               max_seqlen_q=SEG, max_seqlen_k=SEG, causal=False)
    return r[0] if isinstance(r, tuple) else r

def run_sdpa(q, k, v, backend):
    # q,k,v: (Sfull, H, D) -> per segment (1, H, seg, D)
    outs = []
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(backend):
        for i in range(NSEG):
            s, e = i*SEG, (i+1)*SEG
            qi = q[s:e].transpose(0, 1).unsqueeze(0)  # (1,H,seg,D)
            ki = k[s:e].transpose(0, 1).unsqueeze(0)
            vi = v[s:e].transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(qi, ki, vi)
            outs.append(o.squeeze(0).transpose(0, 1))  # (seg,H,D)
    return torch.cat(outs, dim=0)

def bench(fn, name, iters=5):
    q, k, v = make_qkv_flat()
    for _ in range(2):
        fn(q, k, v)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(q, k, v)
    torch.cuda.synchronize()
    dt_ms = (time.time()-t0)/iters*1000
    print(f"[{name}] {dt_ms:.1f} ms/attn  ({dt_ms*48/1000:.2f} s over 48 layers)")
    return dt_ms

from torch.nn.attention import SDPBackend, sdpa_kernel

def run_sdpa_prio(q, k, v):
    # priority list [cuDNN, flash, efficient] with set_priority — the real code path.
    backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    try:
        ctx = sdpa_kernel(backends, set_priority=True)
    except TypeError:
        ctx = sdpa_kernel(backends)
    outs = []
    with ctx:
        for i in range(NSEG):
            s, e = i*SEG, (i+1)*SEG
            qi = q[s:e].transpose(0, 1).unsqueeze(0).contiguous()
            ki = k[s:e].transpose(0, 1).unsqueeze(0).contiguous()
            vi = v[s:e].transpose(0, 1).unsqueeze(0).contiguous()
            o = F.scaled_dot_product_attention(qi, ki, vi)
            outs.append(o.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)

res = {}
try:
    res['prio_refiner'] = bench(run_sdpa_prio, "SDPA-prio(refiner-shape)")
except Exception as e:
    print("prio refiner failed:", repr(e))

# --- base 480p shape: n_video ~48360, 2 segments ---
try:
    _NV, _T = 48360, 128
    SEG_R = SEG; NV_R = NVIDEO  # keep refiner globals if needed
    def bench_base():
        seg = _NV + _T
        cu_b = torch.tensor([0, seg, 2*seg], device=dev, dtype=torch.int32)
        def mk():
            return (torch.randn(2*seg, H, D, device=dev, dtype=dt) for _ in range(3))
        q,k,v = mk()
        backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        try: ctx = sdpa_kernel(backends, set_priority=True)
        except TypeError: ctx = sdpa_kernel(backends)
        def fn():
            outs=[]
            with ctx:
                for i in range(2):
                    s,e=i*seg,(i+1)*seg
                    qi=q[s:e].transpose(0,1).unsqueeze(0).contiguous()
                    ki=k[s:e].transpose(0,1).unsqueeze(0).contiguous()
                    vi=v[s:e].transpose(0,1).unsqueeze(0).contiguous()
                    outs.append(F.scaled_dot_product_attention(qi,ki,vi))
            return outs
        for _ in range(2): fn()
        torch.cuda.synchronize(); import time as _t; t0=_t.time()
        for _ in range(5): fn()
        torch.cuda.synchronize()
        print(f"[SDPA-prio(base-shape seg={seg})] {(_t.time()-t0)/5*1000:.1f} ms/attn — NO CRASH")
    bench_base()
except Exception as e:
    print("prio base failed:", repr(e))

if HAVE_FA:
    res['fa2'] = bench(run_fa, "FA2-varlen")
try:
    res['cudnn'] = bench(lambda q,k,v: run_sdpa(q,k,v,SDPBackend.CUDNN_ATTENTION), "SDPA-cuDNN")
except Exception as e:
    print("cudnn sdpa failed:", repr(e))
try:
    res['flash'] = bench(lambda q,k,v: run_sdpa(q,k,v,SDPBackend.FLASH_ATTENTION), "SDPA-flash")
except Exception as e:
    print("flash sdpa failed:", repr(e))

# correctness: compare cuDNN vs FA2 outputs
if HAVE_FA:
    q,k,v = make_qkv_flat()
    of = run_fa(q,k,v).float()
    try:
        oc = run_sdpa(q,k,v,SDPBackend.CUDNN_ATTENTION).float()
        d = (of-oc).abs().max().item()
        rel = d / of.abs().max().item()
        print(f"[correctness] FA2 vs cuDNN max_abs_diff={d:.4e} rel={rel:.4e}")
    except Exception as e:
        print("correctness cudnn failed:", repr(e))

print("[bench] summary:", {k: f"{v:.1f}ms" for k,v in res.items()})
