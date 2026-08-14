"""Opaque full-volume abs-RoPE for deterministic (pre-diffusion) NA.
Owns QKV + opaque ``custom_op`` packaging for det ``NA.forward``. Det stages
have differently shaped T/H/W; the opaque op keeps Dynamo from specializing
on each stage shape. Diffusion paths use ``diff_attn/`` RoPE — not this file.
No T/H/W origin/offset is threaded for tiled decode, and none is needed:
every attention call here is ``natten.na3d``, a local window with no
cross-tile tokens. Absolute-vs-local RoPE differs by a global phase that
cancels inside the attention softmax over that window, so the attention
output is unchanged. Since every tiled-decode call processes exactly one
tile in isolation, using each tile's local 0-based positions is identical
to using its true absolute origin. Absolute origin still matters for
whether a tile contains the latent's first frame (``drop_leading_frame``),
which is handled outside RoPE in the decoder stage / per-tile decode.
"""

from __future__ import annotations

import torch

from ltx_core.model.video_vae.transformer.rope_math import (
    DEFAULT_ABS_ROPE_NUM_TILES,
    h_positions,
    rot_abs_axis_impl,
    t_positions,
)


def _apply_opaque_rope_slab(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    w_pos: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Rotate one W-extent with raw abs-RoPE (runs inside the opaque op)."""
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    xt = rot_abs_axis_impl(x[..., :d_t], t_positions(t, x.device), inv_t, axis=1, compute_dtype=compute_dtype)
    xh = rot_abs_axis_impl(
        x[..., d_t : d_t + d_h],
        h_positions(h, x.device),
        inv_h,
        axis=2,
        compute_dtype=compute_dtype,
    )
    xw = rot_abs_axis_impl(x[..., d_t + d_h :], w_pos, inv_w, axis=3, compute_dtype=compute_dtype)
    return torch.cat([xt, xh, xw], dim=-1)


def _apply_opaque_tiled_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Fixed-``num_tiles`` W split + per-slab rotation (body of the opaque op)."""
    slabs = torch.chunk(x, num_tiles, dim=3)
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in slabs:
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        parts.append(
            _apply_opaque_rope_slab(
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
    """Opaque out-of-place abs-RoPE: Dynamo sees one node."""
    compute_dtype = torch.bfloat16 if compute_dtype_is_bf16 else torch.float32
    return _apply_opaque_tiled_rope(
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


def _apply_opaque_abs_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"compute_dtype must be float32 or bfloat16, got {compute_dtype}")
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


def det_qkv_rope(attn: object, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V proj + norm/scale + opaque full-volume abs-RoPE."""
    q, k, v = attn.project_qkv(x)  # type: ignore[attr-defined]
    q = attn.q_norm(q)  # type: ignore[attr-defined]
    k = attn.k_norm(k)  # type: ignore[attr-defined]
    q = q * attn.scale  # type: ignore[attr-defined]

    inv_freqs = (
        attn.rope_inv_t.to(device=x.device),  # type: ignore[attr-defined]
        attn.rope_inv_h.to(device=x.device),  # type: ignore[attr-defined]
        attn.rope_inv_w.to(device=x.device),  # type: ignore[attr-defined]
    )
    num_tiles = getattr(attn, "rope_num_tiles", DEFAULT_ABS_ROPE_NUM_TILES)
    compute_dtype = getattr(attn, "rope_compute_dtype", torch.float32)
    q = _apply_opaque_abs_rope(
        q,
        attn.rope_dim_split,  # type: ignore[attr-defined]
        inv_freqs,
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
    )
    k = _apply_opaque_abs_rope(
        k,
        attn.rope_dim_split,  # type: ignore[attr-defined]
        inv_freqs,
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
    )
    return q, k, v
