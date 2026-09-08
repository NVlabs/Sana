"""A custom Ulysses attention path for MiniMax-H3, replacing `dispatch_attention_fn`.

diffusers is the starting point, not the constraint. Its dispatch costs 94 ms/step of layout churn
and collectives around a 341 ms SDPA, and the measurements say exactly where that goes: 42 ms in
`.contiguous()` copies and 52 ms in four collectives, three of which carry q, k and v separately.
SGLang's MiniMax-H3 runtime does the same job differently, and two of its choices are worth taking.

**One collective instead of three.** The checkpoint stores a single fused QKV matrix. diffusers'
conversion splits it into `to_q`/`to_k`/`to_v`, and the dispatch then pays three permuted copies and
three all-to-alls. Keeping it packed means one copy and one collective over the same bytes. Whether
that is faster is not obvious — an earlier probe found three separate collectives already overlap
(0.72 ms against 0.97 ms if they were serial), and packing them with a `torch.cat` was *slower*
because of the concatenation. The difference here is that no concatenation is needed: the tensor is
already one buffer coming out of the projection.

**A flat layout.** SGLang carries `(total, heads, head_dim)` with `cu_seqlens` rather than
`(batch, seq, heads, head_dim)`. At batch 1 the batch dimension is pure ceremony, and it is what
makes the dispatch's permutations five-dimensional. Dropping it makes the pre-collective permute a
single transpose.

What is deliberately *not* taken from SGLang here: its variable-length flash attention. The backend
sweep already measured flash-attn 2.8.3 at 71% slower than SDPA on this hardware (sm_100), so
switching the attention kernel on the strength of someone else's benchmark would be going against
our own measurement. The layout is worth taking; the kernel is not.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


_ROW_COUNTS: dict = {}


def _qkv_wire_dtype(attention_fn) -> str:
    """Select the wire format for the next call."""
    if attention_fn is None:
        return "bf16"
    native_int8 = bool(getattr(attention_fn, "native_int8_qkv", False))
    configured = os.environ.get("H3_ULYSSES_COMM_DTYPE")
    wire_dtype = (configured if configured is not None
                  else ("int8_qkv" if native_int8 else "bf16")).lower()
    if wire_dtype not in {"bf16", "int8_qkv"}:
        raise ValueError(
            "H3_ULYSSES_COMM_DTYPE must be 'bf16' or 'int8_qkv', "
            f"got {wire_dtype!r}"
        )
    if wire_dtype == "int8_qkv" and not native_int8:
        raise ValueError("int8 QKV transport is supported by the SOL/BSA backend only")
    if wire_dtype == "int8_qkv":
        scope = os.environ.get("H3_ULYSSES_INT8_SCOPE", "all").lower()
        if scope not in {"all", "sparse"}:
            raise ValueError(
                "H3_ULYSSES_INT8_SCOPE must be 'all' or 'sparse', "
                f"got {scope!r}"
            )
        # The quality profile keeps policy-dense calls in BF16.  The backend owns the
        # request/layer clock and can make this decision before the call increments it.
        if scope == "sparse" and not attention_fn.use_int8_qkv_for_next_call():
            return "bf16"
    return wire_dtype


def _output_wire_dtype(attention_fn, qkv_wire_dtype: str) -> str:
    """Select the return-trip format independently from the QKV experiment.

    ``match_qkv`` is useful with the quality-preserving sparse QKV scope: the
    return trip is compressed only for calls that already crossed the lossy
    QKV boundary.  Dense reference attention is always kept in BF16.
    """
    configured = os.environ.get("H3_ULYSSES_OUTPUT_DTYPE")
    if configured is None:
        native_int8 = bool(
            attention_fn is not None
            and getattr(attention_fn, "native_int8_qkv", False)
        )
        configured = "fp8" if native_int8 else "bf16"
    configured = configured.lower()
    if configured not in {"bf16", "int8", "fp8", "match_qkv"}:
        raise ValueError(
            "H3_ULYSSES_OUTPUT_DTYPE must be 'bf16', 'int8', 'fp8', or "
            "'match_qkv', "
            f"got {configured!r}"
        )
    if attention_fn is None or configured == "bf16":
        return "bf16"
    if configured == "match_qkv":
        return "int8" if qkv_wire_dtype == "int8_qkv" else "bf16"
    return configured


def _row_counts(rows_local: int, world: int, group) -> list:
    """Every rank's row count, gathered once and cached.

    This is the detail that makes an even-shaped test worthless. `PartitionAnythingSharder` shards
    with `tensor_split`, and the packed sequence is 38247 rows over 4 ranks — so the shards are
    [9562, 9562, 9562, 9561], not four equal blocks. `ulysses_anything` exists precisely because
    the length is not divisible; assuming it is divisible inside the fast path puts the assumption
    back where the flag was meant to remove it.

    diffusers handles this with `gather_size_by_comm` on every call. The shape is fixed for the
    whole request, so gathering once and caching is the same answer without 9800 collectives.
    """
    # The key must be something every rank agrees on. Keying it on `rows_local` — which is the
    # whole point of this function, because it *differs* per rank — makes the cache hit on some
    # ranks and miss on others the moment the length changes, so some ranks issue the gather and
    # some do not, and the job deadlocks on mismatched collective counts. The group is the only
    # thing here that is the same everywhere.
    key = id(group)
    cached = _ROW_COUNTS.get(key)
    if cached is not None:
        counts, seen_rows_local = cached
        if seen_rows_local != rows_local:
            # Purely local check, so it cannot itself desynchronise. A request has one packed
            # length, so this never fires in production; it fires when a caller changes shape
            # without calling `reset_row_counts()`, and failing loudly beats hanging for 10 min.
            raise RuntimeError(
                f"row count cached for rows_local={seen_rows_local} but called with "
                f"{rows_local}; call reset_row_counts() when the sequence length changes"
            )
        return counts

    buffer = torch.zeros(world, dtype=torch.long, device="cuda")
    buffer[dist.get_rank(group)] = rows_local
    dist.all_reduce(buffer, group=group)
    counts = buffer.tolist()
    _ROW_COUNTS[key] = (counts, rows_local)
    return counts


def reset_row_counts() -> None:
    """Forget the cached shard sizes. Every rank must call this, or the next gather desynchronises."""
    _ROW_COUNTS.clear()


def _all_to_all_varlen(x: torch.Tensor, out_numel: int, in_splits: list, out_splits: list,
                       group) -> torch.Tensor:
    """`all_to_all_single` with explicit per-rank sizes, on a flattened buffer."""
    flat = x.reshape(-1)
    out = torch.empty(out_numel, dtype=x.dtype, device=x.device)
    dist.all_to_all_single(out, flat, output_split_sizes=out_splits,
                           input_split_sizes=in_splits, group=group)
    return out


def _all_to_all(x: torch.Tensor, group) -> torch.Tensor:
    """A plain `all_to_all_single` on a contiguous buffer, without the functional wrapper.

    diffusers routes every collective through `torch.distributed._functional_collectives`, which
    returns an `AsyncCollectiveTensor` that has to be flattened, dispatched and then waited on. This
    path issues four collectives per attention, so at 50 blocks and 49 evaluations that wrapper is
    entered 9800 times per request and is waited on immediately every time — it can never overlap
    anything. SGLang's runtime makes the same observation in a comment on its own version:

        "USP calls this collective many times per denoising step and waits immediately, so avoid
         the extra wrapper overhead of functional collectives."
    """
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    return out


def _packed_qkv_all_to_all(q, k, v, world: int, group, packed_send=None,
                           wire_dtype: str = "bf16") -> torch.Tensor:
    """Trade sequence rows for heads, carrying q, k and v in one collective.

    q, k and v are `(rows_local, heads, head_dim)` — this rank's slice of the sequence with every
    head — and may be strided views of the fused projection's output. The result is
    `(rows_full, heads_local, 3 * head_dim)`: the whole sequence with this rank's heads.

    `all_to_all_single` scatters along dimension 0, so the rank owning each head group has to lead.
    Getting there in PyTorch costs a `torch.stack` and then a five-dimensional
    `permute(...).contiguous()` — two full passes over the QKV buffer before a byte moves, and the
    reason the first packed measurement came out at 0.953x. `pack_qkv_destination_major` reads the
    three views through their own strides and writes the destination-major buffer in one pass.
    """
    from .relayout import can_pack_qkv, pack_qkv_destination_major, pack_qkv_reference

    rows_local, heads, head_dim = q.shape
    heads_local = heads // world
    counts = _row_counts(rows_local, world, group)
    rows_full = sum(counts)

    if packed_send is not None:
        expected = ((world, rows_local, heads_local, 400)
                    if wire_dtype == "int8_qkv"
                    else (world, rows_local, heads_local, 3 * head_dim))
        if packed_send.shape != expected or not packed_send.is_contiguous():
            raise ValueError(
                f"prepacked QKV must be contiguous with shape {expected}, "
                f"got {tuple(packed_send.shape)}"
            )
        x = packed_send
    elif can_pack_qkv(q, k, v):
        x = pack_qkv_destination_major(q, k, v, world)
    else:
        x = pack_qkv_reference(q, k, v, world)

    # I send `rows_local` rows to every peer and receive `counts[j]` rows from peer j. Those are
    # different numbers whenever the sequence does not divide evenly, which is the only case this
    # code ever runs in.
    if wire_dtype == "int8_qkv":
        from .comm_quant import QKV_PACKET

        records = heads_local
        out = torch.empty(rows_full * records * QKV_PACKET,
                          dtype=torch.uint8, device=x.device)
        dist.all_to_all_single(
            out, x.reshape(-1),
            input_split_sizes=[rows_local * records * QKV_PACKET] * world,
            output_split_sizes=[count * records * QKV_PACKET for count in counts],
            group=group,
        )
        return out.reshape(rows_full, heads_local, QKV_PACKET)

    block = heads_local * 3 * head_dim
    x = _all_to_all_varlen(
        x, rows_full * block,
        in_splits=[rows_local * block] * world,
        out_splits=[c * block for c in counts],
        group=group,
    )
    return x.reshape(rows_full, heads_local, 3 * head_dim)


def _packed_out_all_to_all(
    out: torch.Tensor,
    rows_local: int,
    world: int,
    group,
    wire_dtype: str = "bf16",
    mxfp8_shape: tuple[int, ...] | None = None,
):
    """The inverse: full sequence with local heads back to local rows with every head.

    `out` is `(rows_full, heads_local, head_dim)` contiguous, so the pre-collective step is free —
    dimension 0 is already the sequence, which is what gets scattered. The split sizes run the other
    way from the forward collective: I send peer j its own `counts[j]` rows and receive `rows_local`
    rows back from each of them.
    """
    _, heads_local, head_dim = out.shape
    counts = _row_counts(rows_local, world, group)
    if wire_dtype == "fp8":
        from .comm_quant import (
            dequantize_merge_output_fp8,
            merge_output_fp8_as_mxfp8,
            quantize_output_fp8,
        )

        if head_dim != 128:
            raise ValueError("FP8 output transport requires head_dim=128")
        packet = quantize_output_fp8(out)
        block = heads_local * head_dim
        received = torch.empty(
            rows_local * world * block,
            dtype=torch.uint8,
            device=out.device,
        )
        dist.all_to_all_single(
            received,
            packet.reshape(-1),
            input_split_sizes=[count * block for count in counts],
            output_split_sizes=[rows_local * block] * world,
            group=group,
        )
        received = received.reshape(world, rows_local, heads_local, head_dim)
        if mxfp8_shape is not None:
            return merge_output_fp8_as_mxfp8(received, world, mxfp8_shape)
        return dequantize_merge_output_fp8(received, world)
    if wire_dtype == "int8":
        from .comm_quant import OUTPUT_PACKET, dequantize_merge_output, quantize_output

        if head_dim != 128:
            raise ValueError("int8 output transport requires head_dim=128")
        packet = quantize_output(out)
        records = heads_local
        received = torch.empty(
            rows_local * world * records * OUTPUT_PACKET,
            dtype=torch.uint8,
            device=out.device,
        )
        dist.all_to_all_single(
            received,
            packet.reshape(-1),
            input_split_sizes=[count * records * OUTPUT_PACKET for count in counts],
            output_split_sizes=[rows_local * records * OUTPUT_PACKET] * world,
            group=group,
        )
        return dequantize_merge_output(
            received.reshape(world, rows_local, heads_local, OUTPUT_PACKET), world
        )
    if wire_dtype != "bf16":
        raise ValueError(f"unsupported output wire dtype {wire_dtype!r}")

    from .relayout import merge_heads

    block = heads_local * head_dim

    x = _all_to_all_varlen(
        out, rows_local * world * block,
        in_splits=[c * block for c in counts],
        out_splits=[rows_local * block] * world,
        group=group,
    )                                                   # dim 0 now indexes the head group
    return merge_heads(x.reshape(world, rows_local, heads_local, head_dim))


def install(transformer, group=None, attention_fn=None):
    """Replace each attention forward with the packed Ulysses path.

    ``attention_fn`` receives full-sequence ``(tokens, local_heads, head_dim)``
    Q/K/V tensors after the Ulysses exchange. When omitted, dense SDPA is used.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return lambda: None

    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    if world == 1:
        return lambda: None

    reuse_fp8_mxfp8 = os.environ.get("H3_REUSE_OUTPUT_FP8_AS_MXFP8", "1")
    if reuse_fp8_mxfp8 not in {"0", "1"}:
        raise ValueError("H3_REUSE_OUTPUT_FP8_AS_MXFP8 must be '0' or '1'")
    reuse_fp8_mxfp8 = reuse_fp8_mxfp8 == "1"

    from .parallel_hooks import with_cp_reapplied
    restores = []

    def make(attn):
        original = attn.forward

        def forward(hidden_states, rotary_emb=None, attention_mask=None):
            if attention_mask is not None:
                # The packed path assumes one attention document. diffusers builds the sequence
                # without padding rows so this holds, and the CP plan asserts it, but a caller that
                # supplied a mask would otherwise get it silently ignored.
                raise RuntimeError("the packed Ulysses path does not carry an attention mask")

            batch, rows_local, _ = hidden_states.shape
            heads, head_dim = attn.heads, attn.head_dim

            rows = batch * rows_local
            if getattr(attn, "fused_projections", False):
                # The MXFP8 path shares one activation quantization across q/k/v.
                # The pack kernel accepts the resulting strided views directly.
                q, k, v = (
                    attn.to_qkv(hidden_states)
                    .reshape(rows, 3, heads, head_dim)
                    .unbind(1)
                )
            else:
                # Separate BF16 projections avoid an otherwise unnecessary QKV concat.
                q = attn.to_q(hidden_states).reshape(rows, heads, head_dim)
                k = attn.to_k(hidden_states).reshape(rows, heads, head_dim)
                v = attn.to_v(hidden_states).reshape(rows, heads, head_dim)
            wire_dtype = _qkv_wire_dtype(attention_fn)

            if rotary_emb is not None:
                from .relayout import (
                    can_qknorm_rope_pack,
                    qknorm_rope_pack_qkv_destination_major,
                )

                cos, sin = rotary_emb
                if not can_qknorm_rope_pack(q, k, v, cos, sin, world):
                    raise RuntimeError(
                        "H3_QKNORM_PACK_FUSION=1 but the request shape is unsupported; "
                        "refusing a silent fallback in a performance arm"
                    )
                packed_send = qknorm_rope_pack_qkv_destination_major(
                    q, k, v,
                    attn.norm_q.weight, attn.norm_k.weight,
                    cos, sin, attn.norm_q.eps, attn.norm_k.eps, world,
                    wire_dtype=wire_dtype,
                )
                packed = _packed_qkv_all_to_all(
                    q, k, v, world=world, group=group, packed_send=packed_send,
                    wire_dtype=wire_dtype,
                )
            else:
                if wire_dtype == "int8_qkv":
                    raise RuntimeError("int8 QKV transport requires the fused RoPE path")
                q, k = attn.norm_q(q), attn.norm_k(k)
                packed = _packed_qkv_all_to_all(q, k, v, world=world, group=group)

            if wire_dtype == "int8_qkv":
                out = attention_fn.int8_qkv(packed).contiguous()
            else:
                # Strided views; SDPA and SOL only require stride(-1)==1.
                q, k, v = packed.split(head_dim, dim=-1)
            if wire_dtype != "int8_qkv" and attention_fn is None:
                out = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k.transpose(0, 1).unsqueeze(0),
                    v.transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                )
                out = out.squeeze(0).transpose(0, 1).contiguous()
            elif wire_dtype != "int8_qkv":
                out = attention_fn(q, k, v).contiguous()

            output_wire_dtype = _output_wire_dtype(attention_fn, wire_dtype)
            output_shape = (batch, rows_local, heads * head_dim)
            reuse_output = (
                reuse_fp8_mxfp8
                and output_wire_dtype == "fp8"
                and getattr(attn.to_out[0], "layout", None) == "MXFP8Swizzled"
            )
            out = _packed_out_all_to_all(
                out,
                rows_local,
                world,
                group,
                wire_dtype=output_wire_dtype,
                mxfp8_shape=output_shape if reuse_output else None,
            )
            if not reuse_output:
                out = out.reshape(output_shape).to(hidden_states.dtype)
            return attn.to_out[1](attn.to_out[0](out))

        return original, forward

    def do_install():
        for block in transformer.transformer_blocks:
            original, forward = make(block.attn)
            restores.append((block.attn, "forward", original))
            block.attn.forward = forward

    with_cp_reapplied(transformer, do_install)

    def uninstall():
        def undo():
            for module, name, original in restores:
                setattr(module, name, original)
            restores.clear()

        with_cp_reapplied(transformer, undo)

    return uninstall
