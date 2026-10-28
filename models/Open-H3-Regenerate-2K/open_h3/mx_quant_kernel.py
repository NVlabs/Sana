"""One-pass MXFP8 activation quantizer: bf16 (M, K) -> e4m3 (M, K) + e8m0 scales already in the cuBLAS 128x4 blocked
layout that torch._scaled_mm expects. Replaces fast_dit._mx_quant + _to_blocked (several pointwise passes plus the
permute/reshape copies of the scale tensor) with a single read of x and a single write of each output.

Numerics are identical to fast_dit._mx_quant (ceil exponent, clamp to +-448, scale clamp to [1, 254]).

    python bench/mx_quant_kernel.py      # self-test vs the reference quantizer + timing
"""
import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _mx_quant_kernel(x_ptr, q_ptr, s_ptr, M, K, stride_xm, n_col_blocks,
                     BLOCK_M: tl.constexpr, BLOCK_KB: tl.constexpr):
    # program: BLOCK_M rows x BLOCK_KB groups of 32 columns
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    groups = pid_k * BLOCK_KB + tl.arange(0, BLOCK_KB)
    cols = groups[:, None] * 32 + tl.arange(0, 32)[None, :]                       # (KB, 32)
    rmask = rows < M
    offs = rows.to(tl.int64)[:, None, None] * stride_xm + cols[None, :, :]         # (M, KB, 32), int64: M*K ~ 2^31
    mask = rmask[:, None, None] & (cols[None, :, :] < K)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=2)                                              # (M, KB)
    amax = tl.maximum(amax, 2.0 ** -120)
    exponent = tl.ceil(tl.log2(amax / 448.0))
    exponent = tl.minimum(tl.maximum(exponent, -126.0), 127.0)
    y = x * tl.exp2(-exponent)[:, :, None]
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)
    tl.store(q_ptr + rows.to(tl.int64)[:, None, None] * K + cols[None, :, :], y.to(tl.float8e4nv), mask=mask)
    # e8m0 scale -> blocked layout: tile (r // 128, c // 4); inside a 128x4 tile the byte for (r, c) sits at
    # (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)  -- the 32x16 swizzle used by cuBLAS / torchao to_blocked
    biased = (exponent + 127.0).to(tl.uint8)
    r = rows[:, None]
    c = groups[None, :]
    tile = (r // 128).to(tl.int64) * n_col_blocks + (c // 4)
    inner = (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)
    tl.store(s_ptr + tile * 512 + inner, biased, mask=rmask[:, None] & (c < K // 32))


@triton.jit
def _swiglu_mx_quant_kernel(h_ptr, q_ptr, s_ptr, M, K, stride_hm, n_col_blocks,
                            BLOCK_M: tl.constexpr, BLOCK_KB: tl.constexpr):
    # h = [hidden | gate] (M, 2K) bf16; x = bf16(hidden * silu(gate)) (diffusers SwiGLU numerics), then MXFP8 like above
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    groups = pid_k * BLOCK_KB + tl.arange(0, BLOCK_KB)
    cols = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    rmask = rows < M
    mask = rmask[:, None, None] & (cols[None, :, :] < K)
    base = rows.to(tl.int64)[:, None, None] * stride_hm + cols[None, :, :]       # M*2K > 2^31 at the refine shape
    hidden = tl.load(h_ptr + base, mask=mask, other=0.0)
    gate = tl.load(h_ptr + base + K, mask=mask, other=0.0)
    g = gate.to(tl.float32)
    silu = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16)                               # torch silu on bf16: bf16 result
    x = (hidden * silu).to(tl.float32)                                             # bf16 * bf16 -> bf16, as torch
    amax = tl.max(tl.abs(x), axis=2)
    amax = tl.maximum(amax, 2.0 ** -120)
    exponent = tl.ceil(tl.log2(amax / 448.0))
    exponent = tl.minimum(tl.maximum(exponent, -126.0), 127.0)
    y = x * tl.exp2(-exponent)[:, :, None]
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)
    tl.store(q_ptr + rows.to(tl.int64)[:, None, None] * K + cols[None, :, :], y.to(tl.float8e4nv), mask=mask)
    biased = (exponent + 127.0).to(tl.uint8)
    r = rows[:, None]
    c = groups[None, :]
    tile = (r // 128).to(tl.int64) * n_col_blocks + (c // 4)
    inner = (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)
    tl.store(s_ptr + tile * 512 + inner, biased, mask=rmask[:, None] & (c < K // 32))


def swiglu_mx_quant_blocked(h):
    """h (..., 2K) bf16 = [hidden | gate] -> MXFP8 of hidden * silu(gate): (data (rows, K) e4m3, blocked e8m0 scales).
    Replaces SwiGLU (writes a (rows, K) bf16 tensor) + mx_quant_blocked (reads it back)."""
    k2 = h.shape[-1]
    k = k2 // 2
    h2 = h.reshape(-1, k2)
    m = h2.shape[0]
    assert k % 32 == 0 and h2.stride(-1) == 1
    n_row_blocks, n_col_blocks = triton.cdiv(m, 128), triton.cdiv(k // 32, 4)
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=h.device)
    s = torch.zeros((n_row_blocks * n_col_blocks * 512,), dtype=torch.uint8, device=h.device)
    BLOCK_M, BLOCK_KB = 32, 8
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(k // 32, BLOCK_KB))
    _swiglu_mx_quant_kernel[grid](h2, q, s, m, k, h2.stride(0), n_col_blocks, BLOCK_M=BLOCK_M, BLOCK_KB=BLOCK_KB,
                                  num_warps=4)
    return q, s.view(torch.float8_e8m0fnu)


def mx_quant_blocked(x):
    """x (..., K) bf16 contiguous on the last dim -> (data (rows, K) e4m3, scales flat e8m0 blocked)."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    assert k % 32 == 0 and x2.stride(-1) == 1
    n_row_blocks, n_col_blocks = triton.cdiv(m, 128), triton.cdiv(k // 32, 4)
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    # padded rows/cols of the blocked tensor are never read back (same zero padding as fast_dit._to_blocked)
    s = torch.zeros((n_row_blocks * n_col_blocks * 512,), dtype=torch.uint8, device=x.device)
    BLOCK_M, BLOCK_KB = 32, 8
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(k // 32, BLOCK_KB))
    _mx_quant_kernel[grid](x2, q, s, m, k, x2.stride(0), n_col_blocks, BLOCK_M=BLOCK_M, BLOCK_KB=BLOCK_KB,
                           num_warps=4)
    return q, s.view(torch.float8_e8m0fnu)


def _selftest_swiglu():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fast_dit
    torch.manual_seed(1)
    for m in (133435, 37531):
        h = torch.randn(m, 2 * 14336, device="cuda", dtype=torch.bfloat16) * 2
        ref = fast_dit._mx_swiglu(h)
        q_ref, s_ref = mx_quant_blocked(ref)
        q, s = swiglu_mx_quant_blocked(h)
        same_q = torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        same_s = torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
        mismatch = (q.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean().item()
        for fn, name in ((lambda: swiglu_mx_quant_blocked(h), "fused"),
                         (lambda: mx_quant_blocked(fast_dit._mx_swiglu(h)), "swiglu+quant")):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(10):
                fn()
            e1.record()
            torch.cuda.synchronize()
            print(f"SWQ M={m} {name:13s} {e0.elapsed_time(e1) / 10:.3f} ms", flush=True)
        print(f"SWQ M={m} data bit-exact={same_q} (mismatch {mismatch:.2e}) scales bit-exact={same_s}", flush=True)


if __name__ == "__main__":
    import os
    if os.environ.get("SWQ_ONLY"):
        _selftest_swiglu()
        raise SystemExit
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fast_dit
    torch.manual_seed(0)
    for m, k in ((133435, 5376), (133435, 14336), (37531, 5376), (300, 5376)):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        x[:, torch.randperm(k, device="cuda")[:16]] *= 20
        q_ref, s_ref = fast_dit._mx_quant(x)
        s_ref = fast_dit._to_blocked(s_ref)
        q, s = mx_quant_blocked(x)
        same_q = torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        same_s = s.numel() == s_ref.numel() and torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
        ref_c = torch.compile(lambda t: (lambda d, sc: (d, fast_dit._to_blocked(sc)))(*fast_dit._mx_quant(t)))
        for fn, name in ((lambda: mx_quant_blocked(x), "triton"), (lambda: ref_c(x), "inductor")):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(10):
                fn()
            e1.record()
            torch.cuda.synchronize()
            print(f"MXQ M={m} K={k} {name:8s} {e0.elapsed_time(e1) / 10:.3f} ms", flush=True)
        print(f"MXQ M={m} K={k} data bit-exact={same_q} scales bit-exact={same_s}", flush=True)
