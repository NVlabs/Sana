"""SOL Attention v2 for HunyuanVideo — independent dense-text / sparse-video path.

HunyuanVideo self-attention is one joint bidirectional sequence
``[video (F*H*W tokens), text (padded to max_seq)]`` with a key-padding mask on
the text tail (padding keys invisible; no causal mask). This module decomposes
that attention so that EVERYTHING involving text stays exact dense and ONLY the
text-independent block — video queries x video keys — runs sparse:

  1. video -> video : SPARSE. SOL Attention SM100 colmask kernel on Morton-reordered
     video tokens; the kernel returns its own per-query LSE (natural log of
     ``sum(exp(q.k * scale))`` over the routed keys).
  2. video -> text  : DENSE over the VALID text keys only, computed in fp32 in
     query chunks, with its own LSE.
     (1) and (2) are combined with an exact two-way online-softmax LSE merge —
     the two key sets are disjoint, so the merge is exact w.r.t. the sparse
     video x video approximation.
  3. text -> all    : DENSE SDPA over all valid keys (a few hundred rows).

Independent implementation: the vendored colmask kernel is consumed strictly
through its public adapter (``integrations.wan.run`` / ``calibrate_tau``) and is
not modified; nothing here imports from or alters ``sol_attn_backend.py``. The
dispatch hook registers its own opaque custom op (``sol2::attn_hunyuan_v2``) so
the path is torch.compile-safe, mirroring the pattern the v1 backend uses.

All heavy imports are lazy so this module imports cleanly on a login node.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

HEAD_DIM = 128
DEFAULT_BLOCK_SIZE = 64
DEFAULT_TARGET_DENSITY = 0.15
DEFAULT_Q_CHUNK = 32768
_FIXED_SCALE = HEAD_DIM ** -0.5

_COLMASK_ROOT = Path(__file__).resolve().parent / "sol_attn_colmask"

# tau calibrated once per (video q shape, block_size, density) and frozen.
_TAU_CACHE_V2: dict = {}
_MORTON_CACHE_V2: dict = {}


@functools.lru_cache(maxsize=1)
def _load_colmask_v2() -> dict:
    """Import the vendored colmask kernel through its public adapter (unmodified)."""
    import sys

    root = str(_COLMASK_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from integrations.wan import calibrate_tau, run  # type: ignore

    return {"run": run, "calibrate_tau": calibrate_tau}


def _morton3d_perm_v2(grid, device):
    """(perm, inv) ordering (F,H,W) raster tokens along a 3D Morton curve.

    64-token blocks on raster order are thin horizontal strips; on Morton order
    they are compact 3D neighbourhoods, which is what block-sparse routing needs
    to preserve video quality.
    """
    import torch

    key = tuple(int(x) for x in grid)
    hit = _MORTON_CACHE_V2.get(key)
    if hit is None:
        F, H, W = key
        ff, hh, ww = torch.meshgrid(
            torch.arange(F), torch.arange(H), torch.arange(W), indexing="ij"
        )
        ff, hh, ww = ff.reshape(-1), hh.reshape(-1), ww.reshape(-1)
        bits = max(F, H, W).bit_length()

        def _spread(x):
            code = torch.zeros_like(x)
            for i in range(bits):
                code |= ((x >> i) & 1) << (3 * i)
            return code

        code = _spread(ff) | (_spread(hh) << 1) | (_spread(ww) << 2)
        perm = torch.argsort(code)
        hit = (perm, torch.argsort(perm))
        _MORTON_CACHE_V2[key] = hit
    return hit[0].to(device), hit[1].to(device)


def sol_v2_supported(q) -> bool:
    """True iff the colmask kernel can run this tensor on this device (SM100)."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return False
    if not (hasattr(q, "is_cuda") and q.is_cuda):
        return False
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM:
        return False
    try:
        return tuple(torch.cuda.get_device_capability(q.device)) == (10, 0)
    except Exception:
        return False


def _dense_reference(q, k, v, key_valid):
    """Full dense SDPA over the joint sequence with the text-padding key mask."""
    import torch

    B, _H, S, _D = q.shape
    am = torch.zeros(B, 1, 1, S, device=q.device, dtype=q.dtype)
    am = am.masked_fill(~key_valid[:, None, None, :], float("-inf"))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am)


def _video_to_text_dense(qv, kt, vt, tvalid, q_chunk):
    """Dense video-query x valid-text-key pass, chunked over queries, fp32.

    Returns ``(out_vt fp32 [B,H,Sv,D], lse_vt fp32 [B,H,Sv])`` where ``lse_vt``
    is the natural-log LSE of ``q.k * scale`` over the valid text keys — the
    same convention the colmask kernel stores, so the two merge directly.
    """
    import torch

    B, H, Sv, D = qv.shape
    ktf = kt.float()
    vtf = vt.float()
    neg = ~tvalid[:, None, None, :]  # [B,1,1,tl] broadcast over heads/queries
    out = torch.empty(B, H, Sv, D, device=qv.device, dtype=torch.float32)
    lse = torch.empty(B, H, Sv, device=qv.device, dtype=torch.float32)
    for s0 in range(0, Sv, q_chunk):
        s1 = min(s0 + q_chunk, Sv)
        scores = torch.einsum(
            "bhqd,bhkd->bhqk", qv[:, :, s0:s1].float(), ktf
        ) * _FIXED_SCALE
        scores = scores.masked_fill(neg, float("-inf"))
        lse[:, :, s0:s1] = torch.logsumexp(scores, dim=-1)
        # A batch row with ZERO valid text keys yields softmax(all -inf) = NaN;
        # its lse is -inf so its merge weight is exactly 0 — zero the NaNs so
        # 0 * out stays 0 instead of poisoning the merge.
        probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out[:, :, s0:s1] = torch.einsum("bhqk,bhkd->bhqd", probs, vtf)
    return out, lse


def sol_attn_hunyuan_v2(q, k, v, *, video_len, key_valid, grid,
                        tau=None, target_density=DEFAULT_TARGET_DENSITY,
                        block_size=DEFAULT_BLOCK_SIZE, q_chunk=DEFAULT_Q_CHUNK):
    """Dense-text / sparse-video attention for the Hunyuan joint sequence.

    ``q,k,v``: ``[B,H,S,D]``; first ``video_len`` tokens are the video grid
    (``grid=(F,Hp,Wp)``, ``F*Hp*Wp == video_len``), the rest are text padded to
    ``max_sequence_length``. ``key_valid``: bool ``[B,S]``. Falls back to full
    dense SDPA on any kernel/constraint failure unless ``SOL_ATTN_STRICT=1``.
    """
    import torch

    F = torch.nn.functional
    B, H, S, D = q.shape
    tl = S - int(video_len)

    q0 = q.contiguous().to(torch.bfloat16)
    k0 = k.contiguous().to(torch.bfloat16)
    v0 = v.contiguous().to(torch.bfloat16)
    kv_bool = key_valid.bool()

    if tl <= 0 or not sol_v2_supported(q0):
        return _dense_reference(q0, k0, v0, kv_bool).to(q.dtype)

    try:
        cm = _load_colmask_v2()
        os.environ.setdefault("SOL_ATTN_ALLOW_LOW_TAU", "1")

        qv, kvid, vvid = (t[:, :, :video_len] for t in (q0, k0, v0))
        kt, vt = k0[:, :, video_len:], v0[:, :, video_len:]
        tvalid = kv_bool[:, video_len:]  # [B, tl]

        # --- 1. sparse video x video (Morton order; kernel LSE) ---------------
        perm, inv = _morton3d_perm_v2(grid, q0.device)
        qv_r = qv[:, :, perm, :].contiguous()
        kv_r = kvid[:, :, perm, :].contiguous()
        vv_r = vvid[:, :, perm, :].contiguous()
        _tau = tau
        if _tau is None:
            ck = (tuple(qv_r.shape), int(block_size), round(float(target_density), 4))
            _tau = _TAU_CACHE_V2.get(ck)
            if _tau is None:
                _tau = float(cm["calibrate_tau"](
                    qv_r, kv_r, vv_r, target_density=float(target_density),
                    block_size=int(block_size))["threshold"])
                _TAU_CACHE_V2[ck] = _tau
        out_vv_r, lse_vv_r = cm["run"](
            qv_r, kv_r, vv_r, tau=float(_tau), block_size=int(block_size),
            return_lse=True)
        out_vv = out_vv_r[:, :, inv, :].float()   # [B,H,Sv,D]
        lse_vv = lse_vv_r[:, :, inv].float()      # [B,H,Sv]

        # --- 2. dense video -> valid text + exact disjoint-key LSE merge ------
        out_vt, lse_vt = _video_to_text_dense(qv, kt, vt, tvalid, int(q_chunk))
        m = torch.maximum(lse_vv, lse_vt)
        # If a row somehow has no finite side (cannot happen for video keys),
        # keep the merge NaN-free by clamping the shared max.
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        w_vv = torch.exp(lse_vv - m).unsqueeze(-1)
        w_vt = torch.exp(lse_vt - m).unsqueeze(-1)
        out_v = (w_vv * out_vv + w_vt * out_vt) / (w_vv + w_vt)

        # --- 3. dense text -> all valid keys ----------------------------------
        qt = q0[:, :, video_len:]
        am_t = torch.zeros(B, 1, tl, S, device=q.device, dtype=q0.dtype)
        am_t = am_t.masked_fill(~kv_bool[:, None, None, :], float("-inf"))
        out_t = F.scaled_dot_product_attention(qt, k0, v0, attn_mask=am_t)

        out = torch.cat([out_v.to(q0.dtype), out_t], dim=2)
    except Exception as exc:  # never break the model
        if os.environ.get("SOL_ATTN_STRICT", "0") == "1":
            raise
        print(f"[sol_attn_v2:hunyuan] fell back to dense: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return _dense_reference(q0, k0, v0, kv_bool).to(q.dtype)
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Dispatch hook — own context + own opaque custom op; installs over diffusers'
# ``dispatch_attention_fn`` exactly like the v1 backend, without touching it.
# ---------------------------------------------------------------------------


class _V2Context:
    step = -1
    layer = 0
    dense_steps = 0
    dense_layers = frozenset()
    target_density = DEFAULT_TARGET_DENSITY
    block_size = DEFAULT_BLOCK_SIZE
    q_chunk = DEFAULT_Q_CHUNK
    tau = None
    grid = None
    video_len = 0


_V2_CTX = _V2Context()
_V2_OP_REGISTERED = False


def sol_v2_begin_forward():
    """Advance the denoising-step clock; install as a transformer pre-hook."""
    _V2_CTX.step += 1
    _V2_CTX.layer = 0


def _parse_layer_ranges_v2(spec) -> frozenset:
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


def _ensure_v2_op():
    global _V2_OP_REGISTERED
    if _V2_OP_REGISTERED:
        return
    import torch

    @torch.library.custom_op(
        "sol2::attn_hunyuan_v2", mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, Tensor key_valid) -> Tensor",
    )
    def _v2_op(q, k, v, key_valid):
        layer = _V2_CTX.layer
        _V2_CTX.layer += 1
        kv = key_valid.bool()
        if _V2_CTX.step < _V2_CTX.dense_steps or layer in _V2_CTX.dense_layers:
            return _dense_reference(
                q.contiguous(), k.contiguous(), v.contiguous(), kv)
        return sol_attn_hunyuan_v2(
            q, k, v, video_len=_V2_CTX.video_len, key_valid=kv,
            grid=_V2_CTX.grid, tau=_V2_CTX.tau,
            target_density=_V2_CTX.target_density,
            block_size=_V2_CTX.block_size, q_chunk=_V2_CTX.q_chunk)

    @_v2_op.register_fake
    def _(q, k, v, key_valid):
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)

    _V2_OP_REGISTERED = True


def make_sol_v2_dispatch(original_dispatch, *, video_len, grid, tau=None,
                         target_density=DEFAULT_TARGET_DENSITY,
                         dense_steps=0, dense_layers="",
                         block_size=DEFAULT_BLOCK_SIZE,
                         q_chunk=DEFAULT_Q_CHUNK):
    """Drop-in replacement for diffusers ``dispatch_attention_fn``.

    Fires only for the joint [video, text] self-attention (head_dim 128,
    non-causal, seq > video_len, mask present, SM100); everything else —
    cross-attention, ineligible shapes — delegates to ``original_dispatch``.
    """
    _V2_CTX.dense_steps = int(dense_steps)
    _V2_CTX.dense_layers = _parse_layer_ranges_v2(dense_layers)
    _V2_CTX.target_density = float(target_density)
    _V2_CTX.block_size = int(block_size)
    _V2_CTX.q_chunk = int(q_chunk)
    _V2_CTX.tau = None if tau is None else float(tau)
    _V2_CTX.grid = tuple(int(x) for x in grid)
    _V2_CTX.video_len = int(video_len)
    _ensure_v2_op()
    import torch

    _op = torch.ops.sol2.attn_hunyuan_v2
    _bool_dtype = torch.bool
    _vl = int(video_len)

    def sol_v2_dispatch_attention_fn(
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
            and sol_v2_supported(query)
        )
        if not eligible:
            return _dense()

        key_valid = attn_mask
        if key_valid.dtype != _bool_dtype:
            key_valid = key_valid > -1.0  # additive mask: 0 valid, -inf masked
        key_valid = key_valid.reshape(key_valid.shape[0], -1)  # [B, S]
        # diffusers passes [B, S, H, D]; the op wants [B, H, S, D].
        out = _op(query.transpose(1, 2), key.transpose(1, 2),
                  value.transpose(1, 2), key_valid)
        return out.transpose(1, 2)

    return sol_v2_dispatch_attention_fn


__all__ = [
    "sol_attn_hunyuan_v2",
    "make_sol_v2_dispatch",
    "sol_v2_begin_forward",
    "sol_v2_supported",
]
