"""Sol-Attn's omitted-block residual merge (Sol-H3 h3_runtime/sol_residual.py) launched with block_n 128 / 4 warps / 1 stage
instead of the reference block_n 32 / 8 warps / 2 stages. Same kernel, same math (zeroth-order centroid residual over the
omitted KV blocks, merged with BSA's O / LSE); only the launch changes. GB200, 38k rows: 3.13 -> 2.74 ms per call, rel L2
5.3e-5 vs the reference launch."""
from __future__ import annotations

import math

BLOCK_N, NUM_WARPS, NUM_STAGES = 128, 4, 1


def make(sr):
    """Drop-in for sparse_attention.merge_sol_residual (input checks as the reference)."""
    import torch
    import triton

    def merge_sol_residual(q, kc, vc, exact_mask, exact_o_bhsd, exact_lse):
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
            raise ValueError("SOL/BSA requires q=[1,T,H,128]")
        tokens, heads, head_dim = q.shape[1:]
        blocks = math.ceil(tokens / 64)
        if tuple(exact_mask.shape) != (1, heads, blocks, blocks):
            raise ValueError(f"expected exact_mask={(1, heads, blocks, blocks)}, got {tuple(exact_mask.shape)}")
        if tuple(exact_o_bhsd.shape) != (1, heads, tokens, head_dim) or tuple(exact_lse.shape) != (1, heads, tokens):
            raise ValueError("unexpected BSA output / LSE shape")
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        grid = (triton.cdiv(tokens, 64), heads, triton.cdiv(head_dim, 128))
        sr._merge_residual_kernel[grid](
            q, kc, vc, exact_mask, exact_o_bhsd, exact_lse, out,
            q.stride(1), q.stride(2), q.stride(3), kc.stride(1), kc.stride(2), kc.stride(3),
            vc.stride(1), vc.stride(2), vc.stride(3), exact_mask.stride(1), exact_mask.stride(2), exact_mask.stride(3),
            exact_o_bhsd.stride(1), exact_o_bhsd.stride(2), exact_o_bhsd.stride(3), exact_lse.stride(1), exact_lse.stride(2),
            tokens=tokens, blocks=blocks, heads=heads, head_dim=head_dim, scale_log2=head_dim ** -0.5 * math.log2(math.e),
            block_m=64, block_n=BLOCK_N, block_dv=128, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        return out

    return merge_sol_residual
