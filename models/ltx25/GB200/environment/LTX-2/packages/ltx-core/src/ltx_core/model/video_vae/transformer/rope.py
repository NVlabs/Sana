"""Absolute 3D RoPE for Neighborhood Attention (W-slabs).
Same structure as SwiGLU / decode tiling:
  - ``apply_abs_rope_slab`` processes whatever W extent it is given
  - ``apply_abs_rope`` splits W into a **fixed** ``num_tiles`` (Dynamo-safe)
    and calls that once per slab
A Python ``range(0, W, chunk_size)`` specializes ``W`` under
``mark_dynamic`` and raises ``ConstraintViolationError``. Donor
``ABS_ROPE=chunked`` uses a fixed tile count for the same reason.
Knobs live on ``NeighborhoodAttention3D`` (via ``configure_abs_rope``), not
module globals.
No T/H/W origin/offset is threaded in here for tiled decode, and none is
needed: every attention call here is ``natten.na3d``, a local window with no
causal masking, and RoPE's rotation makes the attention score between a
query and key depend only on their *relative* (Δt, Δh, Δw) -- shifting every
position in a call by the same constant leaves every relative offset, and
therefore the entire attention output, unchanged. Since every tiled-decode
call processes exactly one tile in full isolation (no attention call ever
sees two tiles' tokens at once), using each tile's local 0-based positions
is provably identical to using its true absolute origin. The one thing that
*does* depend on absolute origin -- whether a tile contains the latent's
true temporal origin, for ``LinearPixelShuffleUpsample``'s causal frame-drop
-- is unrelated to RoPE and tracked separately (``drop_leading_frame`` in
``DiffusionVideoDecoder.forward_pre_diffusion``).
Per-module ``attn.rope_use_custom_op`` selects the out-of-place strategy:
  - ``True`` (default): opaque ``torch.library.custom_op`` -- Dynamo sees one
    node (cheap compile; used for pre-diffusion's differently-shaped stages).
  - ``False``: traced-through math with the innermost per-axis rotation marked
    ``torch.compiler.nested_compile_region`` (better runtime once compiled;
    used for diffusion-step blocks that share one shape).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

DEFAULT_ABS_ROPE_NUM_TILES = 4
DEFAULT_ROPE_USE_CUSTOM_OP = True


def _t_positions(t: int, device: torch.device) -> torch.Tensor:
    return torch.arange(t, dtype=torch.float32, device=device)


def _h_positions(h: int, device: torch.device) -> torch.Tensor:
    return torch.arange(h, dtype=torch.float32, device=device)


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """Default split of head_dim across (T, H, W) RoPE chunks."""
    assert head_dim % 8 == 0, f"head_dim={head_dim} must be a multiple of 8 for default split"
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    assert d_t > 0
    assert d_hw > 0
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Inverse RoPE frequencies: ``1 / base**(i/dim)`` for ``i`` in ``[0, dim, 2)``."""
    assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    inv_freqs = 1.0 / np.power(float(base), exponents)
    return torch.from_numpy(inv_freqs).to(torch.float32)


def _rot_abs_axis_impl(
    xc: torch.Tensor,
    pos: torch.Tensor,
    inv: torch.Tensor,
    axis: int,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Absolute RoPE on one axis chunk ``xc[..., D]`` (D even) → new tensor."""
    out_dtype = xc.dtype
    pairs = xc.reshape(*xc.shape[:-1], xc.shape[-1] // 2, 2)
    xe = pairs[..., 0].to(compute_dtype)
    xo = pairs[..., 1].to(compute_dtype)
    shape = [1, 1, 1, 1, 1, inv.shape[0]]
    shape[axis] = pos.shape[0]
    ang = (pos[:, None] * inv[None, :]).reshape(shape)
    c = ang.cos().to(compute_dtype)
    s = ang.sin().to(compute_dtype)
    re = xe * c - xo * s
    ro = xe * s + xo * c
    out = torch.stack([re, ro], dim=-1).reshape(xc.shape)
    return out.to(out_dtype) if out.dtype != out_dtype else out


# Transparent path: innermost per-axis rotation marked nested_compile_region.
_rot_abs_axis = torch.compiler.nested_compile_region(_rot_abs_axis_impl)


def apply_abs_rope_slab(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    w_pos: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rotate this W-extent (full volume or one slab). Returns a new tensor.
    ``w_pos`` are local W indices for this slab (length must match ``x``'s W).
    """
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    xt = _rot_abs_axis(x[..., :d_t], _t_positions(t, x.device), inv_t, axis=1, compute_dtype=compute_dtype)
    xh = _rot_abs_axis(
        x[..., d_t : d_t + d_h],
        _h_positions(h, x.device),
        inv_h,
        axis=2,
        compute_dtype=compute_dtype,
    )
    xw = _rot_abs_axis(x[..., d_t + d_h :], w_pos, inv_w, axis=3, compute_dtype=compute_dtype)
    return torch.cat([xt, xh, xw], dim=-1)


def _apply_abs_rope_chunked(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """W-slab loop used by both the traced-through ``use_custom_op=False`` path
    (Dynamo sees this loop + the per-axis rotation) and from inside the opaque
    ``ltx_core::abs_rope`` custom_op.
    """
    slabs = torch.chunk(x, num_tiles, dim=3)
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in slabs:
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        parts.append(
            apply_abs_rope_slab(
                slab,
                rope_split,
                inv_freqs,
                w_pos=w_pos,
                compute_dtype=compute_dtype,
            )
        )
        w_off = w_off + w_slab
    return torch.cat(parts, dim=3)


@torch.library.custom_op("ltx_core::abs_rope", mutates_args=())
def _abs_rope_op(
    x: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    num_tiles: int,
    compute_dtype_is_bf16: bool,
) -> torch.Tensor:
    """Opaque out-of-place abs-RoPE: Dynamo sees one node, not an unrolled
    ``num_tiles``-iteration loop plus per-slab rotation math (same rationale
    as ``ltx_core::swiglu_tiled``).
    """
    compute_dtype = torch.bfloat16 if compute_dtype_is_bf16 else torch.float32
    return _apply_abs_rope_chunked(
        x,
        (d_t, d_h, d_w),
        (inv_t, inv_h, inv_w),
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
    )


@_abs_rope_op.register_fake
def _abs_rope_fake(
    x: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    num_tiles: int,
    compute_dtype_is_bf16: bool,
) -> torch.Tensor:
    del inv_t, inv_h, inv_w, d_t, d_h, d_w, num_tiles, compute_dtype_is_bf16
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def apply_abs_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int = DEFAULT_ABS_ROPE_NUM_TILES,
    compute_dtype: torch.dtype = torch.float32,
    use_custom_op: bool = DEFAULT_ROPE_USE_CUSTOM_OP,
) -> torch.Tensor:
    """Apply abs-RoPE on ``x`` ``[B,T,H,W,NH,HD]`` via a fixed count of W-slabs.
    ``num_tiles`` must be a Python int (not derived from ``W``) so Dynamo can
    keep ``W`` dynamic under ``mark_dynamic``. Uneven remainders go to the last
    chunk (``torch.chunk``).
    ``use_custom_op`` selects the out-of-place strategy (see module docstring).
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"compute_dtype must be float32 or bfloat16, got {compute_dtype}")

    if not use_custom_op:
        return _apply_abs_rope_chunked(
            x,
            rope_split,
            inv_freqs,
            num_tiles=num_tiles,
            compute_dtype=compute_dtype,
        )

    d_t, d_h, d_w = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    return _abs_rope_op(
        x,
        inv_t,
        inv_h,
        inv_w,
        d_t,
        d_h,
        d_w,
        num_tiles,
        compute_dtype == torch.bfloat16,
    )


def apply_abs_rope_triton_placeholder(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    del x, rope_split, inv_freqs
    raise NotImplementedError("abs-RoPE Triton kernel is not implemented; use apply_abs_rope")


def configure_abs_rope(
    module_root: nn.Module,
    *,
    num_tiles: int = DEFAULT_ABS_ROPE_NUM_TILES,
    compute_dtype: torch.dtype = torch.float32,
    use_custom_op: bool = DEFAULT_ROPE_USE_CUSTOM_OP,
) -> None:
    """Set abs-RoPE knobs on every ``NeighborhoodAttention3D`` under ``module_root``."""
    # Local import avoids a circular import with attention.py.
    from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D  # noqa: PLC0415

    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"compute_dtype must be float32 or bfloat16, got {compute_dtype}")
    for module in module_root.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.rope_num_tiles = num_tiles
            module.rope_compute_dtype = compute_dtype
            module.rope_use_custom_op = use_custom_op


def qkv_rope(
    attn: object,
    x: torch.Tensor,
    *,
    num_tiles: int = DEFAULT_ABS_ROPE_NUM_TILES,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V proj, norm+scale, absolute RoPE (local 0-based positions --
    see module docstring for why no absolute origin is needed).
    """
    q, k, v = attn.project_qkv(x)  # type: ignore[attr-defined]
    q = attn.q_norm(q)  # type: ignore[attr-defined]
    k = attn.k_norm(k)  # type: ignore[attr-defined]
    q = q * attn.scale  # type: ignore[attr-defined]

    inv_freqs = tuple(
        rope_inv_freqs(d, attn.rope_base).to(device=x.device, dtype=torch.float32)  # type: ignore[attr-defined]
        for d in attn.rope_dim_split  # type: ignore[attr-defined]
    )
    use_custom_op = getattr(attn, "rope_use_custom_op", DEFAULT_ROPE_USE_CUSTOM_OP)
    q = apply_abs_rope(
        q,
        attn.rope_dim_split,  # type: ignore[attr-defined]
        inv_freqs,
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
        use_custom_op=use_custom_op,
    )
    k = apply_abs_rope(
        k,
        attn.rope_dim_split,  # type: ignore[attr-defined]
        inv_freqs,
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
        use_custom_op=use_custom_op,
    )
    return q, k, v
