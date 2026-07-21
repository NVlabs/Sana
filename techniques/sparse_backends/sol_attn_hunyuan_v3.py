"""SOL Attention v3 for HunyuanVideo — aligned with Sparse-VideoGen PISA.

Reference: hp-l33/Sparse-VideoGen @ pisa-bidirectional,
``pisa_kernels/kernels/piecewise_sparse_attn_hyvideo.py`` (vendored unmodified
except import-path renames under ``pisa_hyvideo/pisa_hyvideo_kernels/``).

Alignment semantics — how this differs from our v2 split-merge:

  1. ONE fused piecewise kernel over the joint [video, valid-text] sequence
     (video and text queries alike), instead of a python-level split + LSE merge.
  2. Per-QUERY-BLOCK top-k routing (route score = qc.kc + log(var(k)) per KV
     block), instead of a single global column threshold (colmask tau).
  3. Non-selected KV blocks are NOT dropped: they still contribute through
     their block centroid (mean-k, summed-v, length-weighted) inside the online
     softmax — the "piecewise" approximation conserves attention mass. This is
     the key quality difference vs the colmask kernel, which discards
     unrouted blocks entirely.
  4. The text suffix is a forced sink: its blocks get route score +inf, so
     every query attends the text EXACTLY (top_k >= text_sink_blocks).
  5. No Morton reorder: the kernel keeps raster order (routing plus the exact
     text sink is how the reference preserves quality).
  6. The reference kernel has NO key-padding mask, so text padding is handled
     OUTSIDE: q/k/v are cropped to [video + valid text], the kernel runs on
     the cropped sequence, and the output is scattered back (padded query rows
     get zeros — they are masked as keys everywhere downstream, so their
     values never reach valid tokens).

Because top_k == all blocks at density 1.0 makes every block exact, this path
degenerates to true dense attention — the "density->1.0 must equal dense"
diagnostic that could never pass through the colmask kernel DOES hold here.

Falls back to dense SDPA on any failure unless ``SOL_ATTN_STRICT=1``. All heavy
imports are lazy; imports cleanly on a login node.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

HEAD_DIM = 128
DEFAULT_BLOCK_SIZE = 64
DEFAULT_TARGET_DENSITY = 0.15

_PISA_HYVIDEO_ROOT = Path(__file__).resolve().parent / "pisa_hyvideo"
_MORTON_CACHE_V3: dict = {}


def _morton3d_perm_v3(grid, device):
    """(perm, inv) ordering (F,H,W) raster video tokens along a 3D Morton curve.

    Optional pre-permutation for the piecewise kernel: raster-order 64-token
    blocks on the Hunyuan grid (33,45,80) are sub-row horizontal strips; Morton
    order makes each block a compact 3D neighbourhood, which sharpens both the
    top-k routing signal and the centroid approximation. Purely external — the
    vendored kernel is order-agnostic.
    """
    import torch

    key = tuple(int(x) for x in grid)
    hit = _MORTON_CACHE_V3.get(key)
    if hit is None:
        F, H, W = key
        # Upstream axis-to-bit-lane assignment (Sparse-VideoGen attention.py
        # _morton3d_perm): x=w in the FASTEST lane, y=h, z=f slowest. The
        # earlier f-fastest variant produced pathological 64-token blocks
        # spanning the whole clip on the (33,45,80) grid; this order gives
        # mean block extents ~(4.0, 4.2, 4.4) with max (6, 13, 20).
        total = F * H * W
        linear = torch.arange(total, dtype=torch.long)
        frame_area = H * W
        z = linear // frame_area
        rem = linear - z * frame_area
        y = rem // W
        x = rem - y * W

        def _part1by2(n):
            n = n & 0x1FFFFF
            n = (n | (n << 32)) & 0x1F00000000FFFF
            n = (n | (n << 16)) & 0x1F0000FF0000FF
            n = (n | (n << 8)) & 0x100F00F00F00F00F
            n = (n | (n << 4)) & 0x10C30C30C30C30C3
            n = (n | (n << 2)) & 0x1249249249249249
            return n

        code = _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)
        perm = linear[torch.argsort(code)]
        hit = (perm, torch.argsort(perm))
        _MORTON_CACHE_V3[key] = hit
    return hit[0].to(device), hit[1].to(device)


@functools.lru_cache(maxsize=1)
def _load_pisa_hyvideo():
    """Import the vendored PISA hyvideo kernel (unique package name — no
    collision with the ``kernels`` packages of the sol_attn vendor trees)."""
    import sys

    root = str(_PISA_HYVIDEO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from pisa_hyvideo_kernels.piecewise_sparse_attn_hyvideo import (  # type: ignore
        hyvideo_piecewise_attention,
    )

    return hyvideo_piecewise_attention


def sol_v3_supported(q) -> bool:
    """Triton TMA tensor-descriptor path: needs a CUDA SM90+ device."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return False
    if not (hasattr(q, "is_cuda") and q.is_cuda):
        return False
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM:
        return False
    try:
        return torch.cuda.get_device_capability(q.device)[0] >= 9
    except Exception:
        return False


def _dense_reference(q, k, v, key_valid):
    import torch

    B, _H, S, _D = q.shape
    am = torch.zeros(B, 1, 1, S, device=q.device, dtype=q.dtype)
    am = am.masked_fill(~key_valid[:, None, None, :], float("-inf"))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am)


def sol_attn_hunyuan_v3(q, k, v, *, video_len, key_valid,
                        target_density=DEFAULT_TARGET_DENSITY,
                        block_size=DEFAULT_BLOCK_SIZE, grid=None,
                        morton=False):
    """PISA piecewise attention on the Hunyuan joint sequence.

    ``q,k,v``: ``[B,H,S,D]`` with the first ``video_len`` tokens video and the
    rest text padded to ``max_sequence_length``; ``key_valid``: bool ``[B,S]``.
    ``morton=True`` (needs ``grid=(F,H,W)``, F*H*W == video_len) permutes the
    video sub-range along a 3D Morton curve around the kernel call — an
    external layout change only; upstream default is raster (morton=False).
    """
    import torch

    B, H, S, D = q.shape
    tl_pad = S - int(video_len)

    q0 = q.contiguous().to(torch.bfloat16)
    k0 = k.contiguous().to(torch.bfloat16)
    v0 = v.contiguous().to(torch.bfloat16)
    kv_bool = key_valid.bool()

    if tl_pad <= 0 or not sol_v3_supported(q0):
        return _dense_reference(q0, k0, v0, kv_bool).to(q.dtype)

    try:
        # Valid text length; padding is a contiguous suffix by construction
        # (diffusers masks indices >= video_len + effective_text_len). The
        # kernel has no key mask, so a non-uniform batch would need per-item
        # crops — fall back to dense for that (never happens at B=1).
        tl_valid_per_b = kv_bool[:, video_len:].sum(dim=1)
        tl_valid = int(tl_valid_per_b[0].item())
        if not bool((tl_valid_per_b == tl_valid).all()):
            return _dense_reference(q0, k0, v0, kv_bool).to(q.dtype)
        s_eff = int(video_len) + tl_valid

        # Matching the upstream integration exactly: ONLY video queries go
        # through the piecewise kernel (over [video, valid-text] keys with the
        # text sink); text queries get an exact dense SDPA over the same valid
        # keys; padded rows stay zero.
        qv = q0[:, :, :video_len]
        kj = k0[:, :, :s_eff]
        vj = v0[:, :, :s_eff]
        inv = None
        if morton and grid is not None and \
                int(grid[0]) * int(grid[1]) * int(grid[2]) == int(video_len):
            perm, inv = _morton3d_perm_v3(grid, q0.device)
            tail = torch.arange(int(video_len), s_eff, device=q0.device)
            idx = torch.cat([perm, tail])
            qv = qv[:, :, perm]
            kj = kj[:, :, idx]
            vj = vj[:, :, idx]

        fn = _load_pisa_hyvideo()
        out_video = fn(
            qv.contiguous(), kj.contiguous(), vj.contiguous(),
            density=float(target_density),
            block_size=int(block_size),
            text_sink_tokens=int(tl_valid),
        )
        if inv is not None:
            out_video = out_video[:, :, inv]

        out = torch.zeros(B, H, S, D, device=q0.device, dtype=out_video.dtype)
        out[:, :, :video_len] = out_video
        if tl_valid > 0:
            qt = q0[:, :, video_len:s_eff]
            out[:, :, video_len:s_eff] = torch.nn.functional.scaled_dot_product_attention(
                qt, k0[:, :, :s_eff], v0[:, :, :s_eff])
    except Exception as exc:  # never break the model
        if os.environ.get("SOL_ATTN_STRICT", "0") == "1":
            raise
        print(f"[sol_attn_v3:hunyuan] fell back to dense: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return _dense_reference(q0, k0, v0, kv_bool).to(q.dtype)
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Dispatch hook — same seam as v1/v2, own context + opaque custom op.
# ---------------------------------------------------------------------------


class _V3Context:
    step = -1
    layer = 0
    dense_steps = 0
    dense_layers = frozenset()
    target_density = DEFAULT_TARGET_DENSITY
    block_size = DEFAULT_BLOCK_SIZE
    video_len = 0
    grid = None
    morton = False


_V3_CTX = _V3Context()
_V3_OP_REGISTERED = False


def sol_v3_begin_forward():
    _V3_CTX.step += 1
    _V3_CTX.layer = 0


def _parse_layer_ranges_v3(spec) -> frozenset:
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


def _ensure_v3_op():
    global _V3_OP_REGISTERED
    if _V3_OP_REGISTERED:
        return
    import torch

    @torch.library.custom_op(
        "sol3::attn_hunyuan_v3", mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, Tensor key_valid) -> Tensor",
    )
    def _v3_op(q, k, v, key_valid):
        layer = _V3_CTX.layer
        _V3_CTX.layer += 1
        kv = key_valid.bool()
        if _V3_CTX.step < _V3_CTX.dense_steps or layer in _V3_CTX.dense_layers:
            return _dense_reference(
                q.contiguous(), k.contiguous(), v.contiguous(), kv)
        return sol_attn_hunyuan_v3(
            q, k, v, video_len=_V3_CTX.video_len, key_valid=kv,
            target_density=_V3_CTX.target_density,
            block_size=_V3_CTX.block_size,
            grid=_V3_CTX.grid, morton=_V3_CTX.morton)

    @_v3_op.register_fake
    def _(q, k, v, key_valid):
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)

    _V3_OP_REGISTERED = True


def make_sol_v3_dispatch(original_dispatch, *, video_len,
                         target_density=DEFAULT_TARGET_DENSITY,
                         dense_steps=0, dense_layers="",
                         block_size=DEFAULT_BLOCK_SIZE, grid=None,
                         morton=False):
    """Drop-in replacement for diffusers ``dispatch_attention_fn`` (same
    eligibility rules as v2). ``morton=True`` needs ``grid``."""
    _V3_CTX.dense_steps = int(dense_steps)
    _V3_CTX.dense_layers = _parse_layer_ranges_v3(dense_layers)
    _V3_CTX.target_density = float(target_density)
    _V3_CTX.block_size = int(block_size)
    _V3_CTX.video_len = int(video_len)
    _V3_CTX.grid = None if grid is None else tuple(int(x) for x in grid)
    _V3_CTX.morton = bool(morton)
    _ensure_v3_op()
    import torch

    _op = torch.ops.sol3.attn_hunyuan_v3
    _bool_dtype = torch.bool
    _vl = int(video_len)

    def sol_v3_dispatch_attention_fn(
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

        eligible = (
            parallel_config is None
            and not is_causal
            and dropout_p == 0.0
            and query.shape[-1] == HEAD_DIM
            and key.shape[1] == query.shape[1]
            and query.shape[1] > _vl
            and attn_mask is not None
            and sol_v3_supported(query)
        )
        if not eligible:
            return _dense()

        key_valid = attn_mask
        if key_valid.dtype != _bool_dtype:
            key_valid = key_valid > -1.0
        key_valid = key_valid.reshape(key_valid.shape[0], -1)
        out = _op(query.transpose(1, 2), key.transpose(1, 2),
                  value.transpose(1, 2), key_valid)
        return out.transpose(1, 2)

    return sol_v3_dispatch_attention_fn


__all__ = [
    "sol_attn_hunyuan_v3",
    "make_sol_v3_dispatch",
    "sol_v3_begin_forward",
    "sol_v3_supported",
]
