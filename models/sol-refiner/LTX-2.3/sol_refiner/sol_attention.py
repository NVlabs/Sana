"""Density calibration and Morton ordering for Sana’s SOL-Attn backend."""

from __future__ import annotations
from functools import lru_cache
import torch.nn.functional as F
import torch

BLOCK_SIZE = 64
HEAD_DIM = 128
LOG2E = 1.4426950408889634

_MORTON: dict = {}


def _prep_small(q, k, scale):
    """centroids + threshold mean/var only - never materialises route_means."""
    b, h, t, d = q.shape
    n = t // BLOCK_SIZE
    q_blocks = q.reshape(b, h, n, BLOCK_SIZE, d).float()
    k_blocks = k.reshape(b, h, n, BLOCK_SIZE, d).float()
    kc = k_blocks.mean(dim=3)
    qc = q_blocks.mean(dim=3)
    del q_blocks, k_blocks
    log2_scale = float(scale) * LOG2E
    kc_mean = kc.mean(dim=2)
    kc_var = torch.clamp((kc * kc).mean(dim=2) - kc_mean * kc_mean, min=0.0)
    tmean = (qc * kc_mean[:, :, None, :]).sum(dim=-1) * log2_scale
    tvar = (qc * qc * kc_var[:, :, None, :]).sum(dim=-1) * (log2_scale * log2_scale)
    return qc, kc, tmean, tvar, n, log2_scale


def calibrate_tau(q, k, scale, target_density, cache):
    """Bisect tau so the routed block density ~= target_density. Cached per shape."""
    ckey = (str(q.device), tuple(q.shape), round(float(target_density), 4))
    hit = cache.get(ckey)
    if hit is not None:
        return hit
    # never materialise the full route_means (b,h,nq,nk): at T=252,960 that is
    # 1.93 GiB of fp32 which OOMs. Recompute it per head-chunk inside density()
    # instead - the einsum is ~8 GFLOP per chunk (microseconds on H100), so we
    # trade negligible compute for an ~8x lower peak. Result is bit-identical.
    qc, kc, tmean, tvar, n, log2_scale = _prep_small(q, k, scale)
    idx = torch.arange(n, device=q.device)
    local = (idx[:, None] - idx[None, :]).abs() <= 1  # (n,n) diagonal band always dense

    def density(tau):
        thr = tmean + float(tau) * torch.sqrt(tvar + 1.0e-6)
        h = qc.shape[1]
        step = max(1, min(h, 4))
        total = 0
        count = 0
        for i in range(0, h, step):
            rm = (
                torch.einsum(
                    "bhnd,bhkd->bhnk", qc[:, i : i + step], kc[:, i : i + step]
                )
                * log2_scale
            )
            m = (rm > thr[:, i : i + step][..., None]) | local[None, None]
            total += int(m.sum().item())
            count += m.numel()
            del rm, m
        return total / max(count, 1)

    lo, hi, mid = -8.0, 8.0, 0.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if density(mid) > target_density:  # higher tau -> stricter -> lower density
            lo = mid
        else:
            hi = mid
    cache[ckey] = mid
    return mid


def _morton3d_perm(grid, device):
    key = tuple(int(x) for x in grid)
    hit = _MORTON.get(key)
    if hit is not None:
        return hit[0].to(device), hit[1].to(device)
    Fr, Hh, Ww = key
    ff, hh, ww = torch.meshgrid(
        torch.arange(Fr), torch.arange(Hh), torch.arange(Ww), indexing="ij"
    )
    ff, hh, ww = ff.reshape(-1), hh.reshape(-1), ww.reshape(-1)
    bits = max(Fr, Hh, Ww).bit_length()

    def _spread(x):
        code = torch.zeros_like(x)
        for i in range(bits):
            code |= ((x >> i) & 1) << (3 * i)
        return code

    code = _spread(ff) | (_spread(hh) << 1) | (_spread(ww) << 2)
    perm = torch.argsort(code)
    inv = torch.argsort(perm)
    _MORTON[key] = (perm.cpu(), inv.cpu())
    return perm.to(device), inv.to(device)


@lru_cache(maxsize=1)
def _kernel():
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[4])
    if root not in sys.path:
        sys.path.insert(0, root)
    from techniques.sparse_backends.sol_attn_backend import _load_sol_attn

    return _load_sol_attn()


@torch.no_grad()
def solattn_sm90_attention(q, k, v, *, tau=1.5, target_density=None, cache=None):
    """Use Sana's integrated kernel with the reference density calibration."""
    kernel = _kernel()
    from sol_attn import get_sol_attn_backend

    if get_sol_attn_backend(q.device) != "cute_sm90":
        raise RuntimeError("The refiner's SOL profile requires the CuTe SM90 backend")
    if q.shape[-1] != HEAD_DIM:
        raise ValueError("SOL profile requires head_dim=128")
    tokens = q.shape[2]
    padding = -tokens % BLOCK_SIZE
    if padding:
        q, k, v = (F.pad(x, (0, 0, 0, padding)) for x in (q, k, v))
    if target_density is not None:
        tau = calibrate_tau(
            q, k, HEAD_DIM**-0.5, target_density, {} if cache is None else cache
        )
    out = kernel(
        *(x.transpose(1, 2).contiguous() for x in (q, k, v)),
        tau=tau,
        thresh_type="diag",
        kv_splits=1,
    )
    return out[:, :tokens].transpose(1, 2).contiguous()
