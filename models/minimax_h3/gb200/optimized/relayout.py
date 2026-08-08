"""The two layout changes an Ulysses all-to-all needs, each as a single kernel.

The benchmark that motivated this file is unambiguous: carrying q, k and v in one collective
instead of three measured **0.953x** — a 4.7% regression — while the same idea in SGLang's runtime
is a win. The difference is not the collective. It is what happens on either side of it.

`all_to_all_single` scatters along dimension 0, so the rank that owns each head group has to lead.
Expressed in PyTorch that is

    torch.stack((q, k, v), dim=1)                       full copy #1
      .reshape(rows, 3, world, heads_local, head_dim)
      .permute(2, 0, 1, 3, 4).contiguous()              full copy #2

two full passes over 3 x rows x heads x head_dim bfloat16 before a single byte moves between GPUs.
The measured pre-collective copies were 42 ms/step against a 341 ms attention, and packing made that
worse rather than better because the destination-major permute is a nastier stride pattern than the
three separate ones it replaced.

SGLang does not pay either copy. `pack_qkv_destination_major` reads q, k and v *through their own
strides* and writes the destination-major buffer directly, so there is no stack, no intermediate,
and no `.contiguous()` — one pass instead of two, and the input is allowed to stay a strided view of
the fused QKV projection. `usp_merge_heads` does the same for the return trip. Both are ported here.

Bit-exactness: these kernels move elements, they do not compute. Every value is loaded and stored
unchanged, so the result is bit-identical to the PyTorch permutes they replace — unlike the
arithmetic fusions, where a reordered reduction legitimately changes the last bits. `test_relayout`
checks that as an equality, not a tolerance.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_qkv_kernel(
    out_ptr, q_ptr, k_ptr, v_ptr,
    total_elements, rows, heads_local, head_dim,
    stride_q_row, stride_q_head,
    stride_k_row, stride_k_head,
    stride_v_row, stride_v_head,
    BLOCK: tl.constexpr,
):
    """One thread per (destination, row, local head, dim) element of the output.

    The output index is decomposed rather than the input index: that way the *stores* are perfectly
    coalesced along `dim`, and the loads gather across `global_head`, which is the direction the
    input is contiguous in anyway.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    dim = offsets % head_dim
    head_slot = offsets // head_dim
    local_head = head_slot % heads_local
    row_slot = head_slot // heads_local
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * heads_local + local_head

    q = tl.load(q_ptr + row * stride_q_row + global_head * stride_q_head + dim, mask=mask)
    k = tl.load(k_ptr + row * stride_k_row + global_head * stride_k_head + dim, mask=mask)
    v = tl.load(v_ptr + row * stride_v_row + global_head * stride_v_head + dim, mask=mask)

    base = head_slot * (3 * head_dim) + dim
    tl.store(out_ptr + base, q, mask=mask)
    tl.store(out_ptr + base + head_dim, k, mask=mask)
    tl.store(out_ptr + base + 2 * head_dim, v, mask=mask)


def can_pack_qkv(q, k, v) -> bool:
    return (
        q.is_cuda and q.ndim == 3
        and q.shape == k.shape == v.shape
        and q.dtype == k.dtype == v.dtype
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and not torch.compiler.is_compiling()
    )


def pack_qkv_destination_major(q, k, v, world: int) -> torch.Tensor:
    """`(rows, heads, head_dim)` x3 -> `(world, rows, heads_local, 3 * head_dim)` contiguous.

    q, k and v may be arbitrary strided views as long as the head dimension is contiguous, which is
    what lets the fused QKV projection's output be consumed without materialising anything.
    """
    rows, heads, head_dim = q.shape
    if heads % world:
        raise ValueError(f"heads ({heads}) must divide the Ulysses degree ({world})")
    heads_local = heads // world

    out = torch.empty((world, rows, heads_local, 3 * head_dim), dtype=q.dtype, device=q.device)
    total = rows * heads * head_dim
    if total == 0:
        return out

    block = 1024
    _pack_qkv_kernel[(triton.cdiv(total, block),)](
        out, q, k, v,
        total, rows, heads_local, head_dim,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        BLOCK=block, num_warps=8,
    )
    return out


def pack_qkv_reference(q, k, v, world: int) -> torch.Tensor:
    """What the kernel replaces, kept so the test compares against the real thing."""
    rows, heads, head_dim = q.shape
    heads_local = heads // world
    out = torch.empty((world, rows, heads_local, 3 * head_dim), dtype=q.dtype, device=q.device)
    for index, tensor in enumerate((q, k, v)):
        shards = tensor.reshape(rows, world, heads_local, head_dim).permute(1, 0, 2, 3)
        out[..., index * head_dim : (index + 1) * head_dim].copy_(shards)
    return out


@triton.jit
def _merge_heads_kernel(
    out_ptr, x_ptr, total_elements, world, rows, inner, BLOCK: tl.constexpr,
):
    """`(world, rows, heads_local, head_dim)` -> `(rows, world, heads_local, head_dim)`.

    Both sides are contiguous, so this is a leading-dimension transpose of blocks of
    `inner = heads_local * head_dim` elements. The output index is the one decomposed, which keeps
    the stores linear and puts the gather on the load side.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    tail = offsets % inner
    slot = offsets // inner          # slot = row * world + source_rank
    source = slot % world
    row = slot // world
    src = (source * rows + row) * inner + tail
    tl.store(out_ptr + offsets, tl.load(x_ptr + src, mask=mask), mask=mask)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """`(world, rows, heads_local, head_dim)` -> `(rows, world * heads_local, head_dim)`.

    Bit-exact replacement for `x.permute(1, 0, 2, 3).contiguous().reshape(rows, -1, head_dim)` on
    the Ulysses output path.
    """
    world, rows, heads_local, head_dim = x.shape
    if not x.is_contiguous() or torch.compiler.is_compiling():
        return x.permute(1, 0, 2, 3).contiguous().reshape(rows, world * heads_local, head_dim)

    out = torch.empty((rows, world, heads_local, head_dim), dtype=x.dtype, device=x.device)
    total = out.numel()
    if total == 0:
        return out.reshape(rows, world * heads_local, head_dim)

    block = 1024
    _merge_heads_kernel[(triton.cdiv(total, block),)](
        out, x, total, world, rows, heads_local * head_dim, BLOCK=block, num_warps=8,
    )
    return out.reshape(rows, world * heads_local, head_dim)


def test_relayout(device="cuda") -> dict:
    """Both kernels must be bit-identical to the permutes they replace — they only move data."""
    torch.manual_seed(0)
    report = {}

    for world in (2, 4):
        rows, heads, head_dim = 9563, 56, 128
        fused = torch.randn(rows, 3, heads, head_dim, device=device, dtype=torch.bfloat16)
        q, k, v = fused[:, 0], fused[:, 1], fused[:, 2]      # strided views, on purpose

        got = pack_qkv_destination_major(q, k, v, world)
        want = pack_qkv_reference(q, k, v, world)
        report[f"pack_world{world}"] = {
            "bit_identical": bool(torch.equal(got, want)),
            "max_abs": float((got.float() - want.float()).abs().max()),
        }

        packed = torch.randn(world, rows, heads // world, head_dim, device=device,
                             dtype=torch.bfloat16)
        got = merge_heads(packed)
        want = packed.permute(1, 0, 2, 3).contiguous().reshape(rows, heads, head_dim)
        report[f"merge_world{world}"] = {
            "bit_identical": bool(torch.equal(got, want)),
            "max_abs": float((got.float() - want.float()).abs().max()),
        }

    report["all_bit_identical"] = all(r["bit_identical"] for r in report.values()
                                      if isinstance(r, dict))
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(test_relayout(), indent=2))
