"""SOL Attention backend (PISA2 CuTe DSL for B200/SM100).

Integrates the vendored ``sol_attn`` release kernel
(``lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128``) as a first-class
sparse-attention backend, alongside PISA. The kernel is a BF16 piecewise-sparse
forward that consumes *prepared* inputs (block K/V centroids + a per-block FP32
global threshold), so this module owns the prep pipeline that turns dense
``q/k/v`` into those prepared tensors, then calls the CuTe DSL runner.

Hard constraints imposed by the release kernel (see ``_validate_prepared`` in the
runner): head_dim == 128, BF16 Q/K/V/KC/VC, contiguous ``[B, H, T, 128]``,
non-causal, softmax scale fixed at ``128 ** -0.5``, and device capability
SM100 ``(10, 0)``. Any call that does not satisfy these falls back to dense
scaled-dot-product attention so a model can enable the backend unconditionally
and only the eligible self-attention layers use the sparse kernel.

Design mirrors the PISA dispatch hook already used by the Wan runtime: a
``dispatch_attention_fn``-shaped entry that fires only for real self-attention of
a hooked layer and delegates everything else to the original op.

All heavy imports (CuTe DSL / Triton kernel compilation) are lazy so this module
imports cleanly on a login node without a GPU.
"""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path
from typing import Any, Callable

HEAD_DIM = 128
DEFAULT_BLOCK_SIZE = 64
DEFAULT_TARGET_DENSITY = 0.05
_FIXED_SCALE = HEAD_DIM ** -0.5

# Calibrated tau cache, keyed by (shape, block_size, density). PISA2 is a fixed
# global-threshold scheme: the runtime knob is tau, and realized density floats per
# input/layer. We derive tau ONCE (bisect to a target density) and then keep it
# fixed, or take tau directly. Density is never re-forced per call.
_TAU_CACHE: dict = {}


class _SolContext:
    """(step, layer) clock so dense guards can keep early steps / chosen layers exact.

    ``sol_attn_begin_forward`` (installed as a transformer forward pre-hook) advances
    the step and resets the per-forward layer counter; the dispatch hook increments
    ``layer`` once per eligible self-attention call (= one DiT block).
    """

    step = -1
    layer = 0
    dense_steps = 0
    dense_layers = frozenset()
    # Sparse-attention config, read INSIDE the opaque custom op (invisible to
    # torch.compile) so the guard/counter/kernel never graph-break the compiled
    # DiT block. Set by ``make_sol_attn_dispatch``.
    target_density = DEFAULT_TARGET_DENSITY
    block_size = DEFAULT_BLOCK_SIZE
    tau = None
    grid = None


_SOL_CTX = _SolContext()


def sol_attn_begin_forward():
    """Advance the denoising-step clock and reset the per-forward layer counter."""
    _SOL_CTX.step += 1
    _SOL_CTX.layer = 0


_MORTON_CACHE: dict = {}


def _morton3d_perm(grid, device):
    """Permutation ordering (F,H,W) video tokens along a 3D Morton (Z-order) curve.

    Block-sparse routing keeps whole 64-token blocks; on raster (frame,h,w) order a
    block is a horizontal strip (spatially incoherent). Morton order makes each block
    a compact 3D neighbourhood, which is what makes PISA2 sparse attention preserve
    quality on video (the SVG/Sol-Attn reference reorders the same way).
    Returns (perm, inv_perm) int64 tensors of length F*H*W (perm[i] = raster index of
    the i-th Morton token).
    """
    import torch

    key = tuple(int(x) for x in grid)
    hit = _MORTON_CACHE.get(key)
    if hit is not None:
        return hit[0].to(device), hit[1].to(device)
    F, H, W = key
    ff, hh, ww = torch.meshgrid(
        torch.arange(F), torch.arange(H), torch.arange(W), indexing="ij"
    )
    ff, hh, ww = ff.reshape(-1), hh.reshape(-1), ww.reshape(-1)  # raster-order coords
    bits = max(F, H, W).bit_length()

    def _spread(x):
        code = torch.zeros_like(x)
        for i in range(bits):
            code |= ((x >> i) & 1) << (3 * i)
        return code

    code = _spread(ff) | (_spread(hh) << 1) | (_spread(ww) << 2)
    perm = torch.argsort(code)
    inv = torch.argsort(perm)
    _MORTON_CACHE[key] = (perm, inv)
    return perm.to(device), inv.to(device)


def install_wan_morton_forward(transformer, grid):
    """Reorder tokens to Morton3D ONCE for the whole block stack, not per attention call.

    Only self-attention needs token-to-token interaction; every other op (FFN, norms,
    projections, residual adds) is per-token and order-invariant. So we permute the
    hidden states + RoPE frequencies once at the block-stack input and un-permute once
    at the output — a single permute pair per forward instead of a gather/scatter on
    every one of the 40 attention calls (which the profile showed was the bottleneck,
    ~21ms/call > the kernel itself). Attention then runs on already-local blocks with
    NO per-call reorder.
    """
    import torch

    dev = next(transformer.parameters()).device
    perm, inv = _morton3d_perm(grid, dev)

    rope = transformer.rope
    _orig_rope = rope.forward

    def _rope_fwd(hidden_states):
        fc, fs = _orig_rope(hidden_states)
        return fc.index_select(1, perm), fs.index_select(1, perm)

    rope.forward = _rope_fwd

    def _pre(_module, args):
        return (args[0].index_select(1, perm),) + tuple(args[1:])

    def _post(_module, _args, output):
        return output.index_select(1, inv)

    transformer.blocks[0].register_forward_pre_hook(_pre)
    transformer.blocks[-1].register_forward_hook(_post)
    return int(perm.numel())


def _parse_layer_ranges(spec) -> frozenset:
    out = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return frozenset(out)

# Vendored backend tree: techniques/sparse_backends/sol_attn/
_SOL_ATTN_ROOT = Path(__file__).resolve().parent / "sol_attn"


@functools.lru_cache(maxsize=1)
def _load_backend() -> dict[str, Any]:
    """Import the vendored sol_attn kernel + prep helpers (lazy, cached).

    Returns a dict of the callables we need. Raises on import failure so the
    caller can decide to fall back to dense.
    """
    import sys

    root = str(_SOL_ATTN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Public thin wrapper -> the evidence-bound CuTe DSL runner.
    from sol_attn import make_pisa2_sm100  # type: ignore
    # Triton "aligned" prep: BF16 centroids + canonical global threshold + route mask.
    from kernels import (  # type: ignore
        online_piecewise_sparse_attn_bf16_aligned as aligned,
    )

    return {
        "make_pisa2_sm100": make_pisa2_sm100,
        "aligned": aligned,
        "prepare_qkv": aligned.prepare_qkv,
        "materialize_route_mask": aligned.materialize_route_mask,
        "canonical": aligned.canonical_pisa2,
        "legacy": aligned.legacy,
    }


_COLMASK_ROOT = Path(__file__).resolve().parent / "sol_attn_colmask"


@functools.lru_cache(maxsize=1)
def _load_colmask() -> dict:
    """Import the reference Wan colmask adapter (fast g256 kernel + fixed-tau wrapper
    with a correctness gate and compile-op cache). This is the exact path the
    Sol-Attn experiment branch benchmarks (kernel_speedup ~3.5-4.7x vs dense)."""
    import sys

    root = str(_COLMASK_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from integrations.wan import run, calibrate_tau  # type: ignore

    return {"run": run, "calibrate_tau": calibrate_tau}


def sol_attn_supported(q) -> bool:
    """True iff the release kernel can run this tensor on this device."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return False
    if not (hasattr(q, "is_cuda") and q.is_cuda):
        return False
    if q.shape[-1] != HEAD_DIM or q.ndim != 4:
        return False
    try:
        return tuple(torch.cuda.get_device_capability(q.device)) == (10, 0)
    except Exception:
        return False


def _calibrate_threshold(bk, q, kc, unit_scale, target_density, block_size):
    """Bisection on tau so the materialized route density ~= target_density.

    Self-contained and correct-by-construction: it measures realized density
    from the backend's own ``materialize_route_mask`` and never lets the kernel
    recalibrate. Returns the frozen FP32 global threshold tensor.
    """
    import torch

    os.environ.setdefault("PISA2_ALLOW_LOW_TAU", "1")  # intentional tau sweep
    canonical = bk["canonical"]
    materialize = bk["materialize_route_mask"]

    def thresh_for_tau(tau: float):
        return canonical.compute_global_qck_threshold(
            q, unit_scale, kc, unit_scale, _FIXED_SCALE, block_size, float(tau)
        )

    def density_for(thresh) -> float:
        mask = materialize(
            q, kc, thresh, group_size=block_size, block_size=block_size, scale=_FIXED_SCALE
        )
        return float(mask.float().mean().item())

    key = (tuple(q.shape), block_size, round(float(target_density), 4))
    tau = _TAU_CACHE.get(key)
    if tau is None:
        # Calibrate tau ONCE per (shape, density); reuse across all subsequent
        # attention calls of this generation. Only this first call pays the
        # route-mask bisection cost.
        lo, hi = -8.0, 8.0
        mid = 0.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if density_for(thresh_for_tau(mid)) > target_density:
                lo = mid  # higher tau -> stricter -> lower density
            else:
                hi = mid
        tau = mid
        _TAU_CACHE[key] = tau
    return thresh_for_tau(tau).to(torch.float32).contiguous()


def sol_attn_attention(
    q,
    k,
    v,
    *,
    tau: float | None = None,
    target_density: float = DEFAULT_TARGET_DENSITY,
    block_size: int = DEFAULT_BLOCK_SIZE,
    grid=None,
    dense_fn: Callable | None = None,
):
    """Sparse attention over ``[B, H, T, 128]`` BF16 tensors via the SOL kernel.

    When ``grid=(F,H,W)`` with F*H*W==T, tokens are Morton3D-reordered so 64-token
    blocks are compact 3D neighbourhoods (needed for quality on video), and the
    output is reordered back. Falls back to ``dense_fn`` (or torch SDPA) — always on
    the ORIGINAL token order — when the kernel's constraints are not met.
    """
    import torch

    q0 = q.contiguous().to(torch.bfloat16)
    k0 = k.contiguous().to(torch.bfloat16)
    v0 = v.contiguous().to(torch.bfloat16)
    T = q0.shape[2]

    if not sol_attn_supported(q0):
        return _dense(q0, k0, v0, dense_fn)

    try:
        cm = _load_colmask()
        os.environ.setdefault("PISA2_ALLOW_LOW_TAU", "1")
        # Morton3D reorder: 64-token blocks become compact 3D neighbourhoods.
        q, k, v, inv = q0, k0, v0, None
        if grid is not None and int(grid[0]) * int(grid[1]) * int(grid[2]) == T:
            perm, inv = _morton3d_perm(grid, q0.device)
            q = q0[:, :, perm, :].contiguous()
            k = k0[:, :, perm, :].contiguous()
            v = v0[:, :, perm, :].contiguous()
        # Fixed tau: take it directly, or calibrate ONCE per shape to target density.
        _tau = tau
        if _tau is None:
            ck = (tuple(q.shape), round(float(target_density), 4))
            _tau = _TAU_CACHE.get(ck)
            if _tau is None:
                _tau = float(cm["calibrate_tau"](
                    q, k, v, target_density=target_density, block_size=block_size
                )["threshold"])
                _TAU_CACHE[ck] = _tau
        # Fast colmask adapter: prep + g256 CuteDSL kernel + compile-op cache.
        out = cm["run"](q, k, v, tau=_tau, block_size=block_size)
        if inv is not None:
            out = out[:, :, inv, :].contiguous()
    except Exception as exc:  # kernel/prep failure -> never break the model
        if os.environ.get("SOL_ATTN_STRICT", "0") == "1":
            raise
        print(f"[sol_attn] fell back to dense: {type(exc).__name__}: {exc}")
        return _dense(q0, k0, v0, dense_fn)
    return out.to(q0.dtype)


def _dense(q, k, v, dense_fn):
    if dense_fn is not None:
        return dense_fn(q, k, v)
    import torch

    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


# ---------------------------------------------------------------------------
# Dispatch hook (mirrors the PISA pattern the Wan runtime installs). A model
# runtime installs this over diffusers' ``dispatch_attention_fn`` and it fires
# only for eligible self-attention; everything else delegates to the original.
# ---------------------------------------------------------------------------

_SOL_OP_REGISTERED = False


def _ensure_sol_op():
    """Register ``sol::sparse_attn`` as an opaque custom op (idempotent, lazy).

    The op wraps the ENTIRE sparse path — layer-counter advance, dense guard,
    tau calibration, and the CuteDSL kernel — behind a single operator boundary.
    ``torch.compile`` treats a registered custom op as a black box: it never
    traces inside, so the Python side effects (``_SOL_CTX`` mutation), the
    data-dependent guard branch, and the untraceable CuteDSL kernel no longer
    graph-break the compiled DiT block. Inductor can then fuse the surrounding
    ``transpose`` views into the adjacent projection kernels instead of
    materializing them across a break. All config is read from ``_SOL_CTX``
    (globals) inside the op, so the traced dispatch stays a pure tensor graph.
    """
    global _SOL_OP_REGISTERED
    if _SOL_OP_REGISTERED:
        return
    import torch

    # Explicit schema (not type-hint inference): this module uses
    # ``from __future__ import annotations`` and imports torch lazily, so
    # infer_schema would fail to eval the stringized ``torch.Tensor`` hints.
    @torch.library.custom_op(
        "sol::sparse_attn", mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v) -> Tensor",
    )
    def _sol_sparse_attn(q, k, v):
        # Opaque to dynamo: counter, guard, calibration and kernel all run here.
        layer = _SOL_CTX.layer
        _SOL_CTX.layer += 1
        if _SOL_CTX.step < _SOL_CTX.dense_steps or layer in _SOL_CTX.dense_layers:
            return torch.nn.functional.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous()
            )
        return sol_attn_attention(
            q, k, v, tau=_SOL_CTX.tau, target_density=_SOL_CTX.target_density,
            block_size=_SOL_CTX.block_size, grid=_SOL_CTX.grid, dense_fn=None,
        )

    @_sol_sparse_attn.register_fake
    def _(q, k, v):
        # Real op always returns a contiguous [B, H, S, D] tensor.
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)

    _SOL_OP_REGISTERED = True


def make_sol_attn_dispatch(original_dispatch, *, tau=None,
                           target_density=DEFAULT_TARGET_DENSITY,
                           dense_steps=0, dense_layers="", grid=None,
                           block_size=DEFAULT_BLOCK_SIZE):
    """Build a drop-in replacement for diffusers ``dispatch_attention_fn``.

    Fires only for eligible real self-attention (head_dim 128, non-causal,
    same-seqlen KV, SM100); cross-attention and everything else delegate to
    ``original_dispatch`` (dense). The sparse path is a single opaque custom op
    (``sol::sparse_attn``) so it survives ``regional_compile`` without a
    graph break. ``dense_steps`` keeps the first N denoising steps exact and
    ``dense_layers`` (e.g. "0-2,27-29") keeps chosen DiT blocks exact; both are
    enforced INSIDE the op. ``tau`` sets the fixed global threshold directly; if
    None, tau is derived once from ``target_density``.
    """
    _SOL_CTX.dense_steps = int(dense_steps)
    _SOL_CTX.dense_layers = _parse_layer_ranges(dense_layers)
    _SOL_CTX.target_density = float(target_density)
    _SOL_CTX.block_size = int(block_size)
    _SOL_CTX.tau = None if tau is None else float(tau)
    _SOL_CTX.grid = grid
    _ensure_sol_op()
    # Capture the op ONCE at creation; a per-call ``import torch`` in the traced
    # dispatch could itself graph-break under regional_compile.
    import torch
    _sparse_op = torch.ops.sol.sparse_attn

    def sol_attn_dispatch_attention_fn(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        scale=None, enable_gqa=False, attention_kwargs=None, *,
        backend=None, parallel_config=None,
    ):
        def _dense():
            return original_dispatch(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                enable_gqa, attention_kwargs, backend=backend,
                parallel_config=parallel_config,
            )

        # Under static-shape regional_compile these predicates fold at trace
        # time, so the self-attn branch becomes a straight-line op call with no
        # runtime branch and no graph break.
        self_attn = (
            parallel_config is None
            and attn_mask is None
            and not is_causal
            and dropout_p == 0.0
            and query.shape[-1] == HEAD_DIM
            and key.shape[1] == query.shape[1]   # self-attention (same seqlen)
            and sol_attn_supported(query)
        )
        if not self_attn:
            return _dense()

        # diffusers passes [B, S, H, D]; the kernel wants [B, H, S, D]. The
        # transposes are traceable views; the opaque op owns everything else.
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        val = value.transpose(1, 2)
        out = _sparse_op(q, k, val)
        return out.transpose(1, 2)

    return sol_attn_dispatch_attention_fn


__all__ = [
    "sol_attn_attention",
    "sol_attn_supported",
    "make_sol_attn_dispatch",
    "HEAD_DIM",
    "DEFAULT_TARGET_DENSITY",
]


if __name__ == "__main__":
    # GPU self-test (run on a GB200/SM100 node): compare SOL sparse attention
    # against dense SDPA on a representative shape and print error stats.
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("self-test requires a CUDA (SM100) device")
    B, H, T = 1, 24, 16384
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, H, T, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16)
    import time
    os.environ["SOL_ATTN_STRICT"] = "1"
    # First call pays the one-time tau calibration + kernel compile.
    sparse = sol_attn_attention(q, k, v, target_density=0.05)
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    rel = (sparse.float() - dense.float()).norm() / max(dense.float().norm().item(), 1e-9)

    def _timeit(fn, n=10):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1000.0

    ts = _timeit(lambda: sol_attn_attention(q, k, v, target_density=0.05))  # cached tau
    td = _timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    print(f"[sol_attn self-test] T={T} rel_l2_vs_dense={rel.item():.4f} "
          f"sparse={ts:.2f}ms dense={td:.2f}ms kernel_speedup={td / ts:.2f}x")
