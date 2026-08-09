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

import torch
import torch.distributed as dist


_ROW_COUNTS: dict = {}


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


def _packed_qkv_all_to_all(q, k, v, world: int, group) -> torch.Tensor:
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
    from relayout import can_pack_qkv, pack_qkv_destination_major, pack_qkv_reference

    rows_local, heads, head_dim = q.shape
    heads_local = heads // world
    counts = _row_counts(rows_local, world, group)
    rows_full = sum(counts)

    if can_pack_qkv(q, k, v):
        x = pack_qkv_destination_major(q, k, v, world)
    else:
        x = pack_qkv_reference(q, k, v, world)

    # I send `rows_local` rows to every peer and receive `counts[j]` rows from peer j. Those are
    # different numbers whenever the sequence does not divide evenly, which is the only case this
    # code ever runs in.
    block = heads_local * 3 * head_dim
    x = _all_to_all_varlen(
        x, rows_full * block,
        in_splits=[rows_local * block] * world,
        out_splits=[c * block for c in counts],
        group=group,
    )
    return x.reshape(rows_full, heads_local, 3 * head_dim)


def _packed_out_all_to_all(out: torch.Tensor, rows_local: int, world: int, group) -> torch.Tensor:
    """The inverse: full sequence with local heads back to local rows with every head.

    `out` is `(rows_full, heads_local, head_dim)` contiguous, so the pre-collective step is free —
    dimension 0 is already the sequence, which is what gets scattered. The split sizes run the other
    way from the forward collective: I send peer j its own `counts[j]` rows and receive `rows_local`
    rows back from each of them.
    """
    from relayout import merge_heads

    _, heads_local, head_dim = out.shape
    counts = _row_counts(rows_local, world, group)
    block = heads_local * head_dim

    x = _all_to_all_varlen(
        out, rows_local * world * block,
        in_splits=[c * block for c in counts],
        out_splits=[rows_local * block] * world,
        group=group,
    )                                                   # dim 0 now indexes the head group
    return merge_heads(x.reshape(world, rows_local, heads_local, head_dim))


def _eager_rope(x: torch.Tensor, rotary_emb) -> torch.Tensor:
    """diffusers' `_apply_rotary_emb`, verbatim, on the flat `(rows, heads, head_dim)` layout.

    Only used by the equivalence test, so it is a transcription rather than an optimization: same
    `chunk`, same `cat`, same order of operations, so any difference it shows against the reference
    is the layout and not the arithmetic.
    """
    cos, sin = rotary_emb
    rotary_dim = cos.shape[-1]
    rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = cos.to(x.dtype)[:, None, :]
    sin = sin.to(x.dtype)[:, None, :]
    x1, x2 = rot.chunk(2, dim=-1)
    rot = rot * cos + torch.cat((-x2, x1), dim=-1) * sin
    return torch.cat((rot, passthrough), dim=-1).contiguous()


def install(transformer, packed: bool = True, group=None, use_fusion: bool = True,
            attention_fn=None):
    """Replace each attention's forward with the packed-collective path. Returns uninstall.

    `use_fusion=False` keeps the layout change but runs eager qk-norm and rotary embedding, which is
    what lets `test_packed_equiv.py` ask whether the pack, the collective and the relayout are
    correct without the answer being clouded by the fusion's own reassociation.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return lambda: None

    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    if world == 1 or not packed:
        return lambda: None

    from cache_line import with_cp_reapplied
    from fusions import fused_qknorm_rope

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

            # Three separate projections, never concatenated. The earlier sweep measured a fused
            # QKV GEMM as a *loss* here, so three GEMMs is what we run — and once the pack kernel
            # reads through strides, there is no reason to glue their outputs together either. The
            # `torch.cat` this replaces wrote and re-read rows x 21504 bfloat16 per block, ~7 ms a
            # step, purely to satisfy a layout the collective no longer needs.
            rows = batch * rows_local
            q = attn.to_q(hidden_states).reshape(rows, heads, head_dim)
            k = attn.to_k(hidden_states).reshape(rows, heads, head_dim)
            v = attn.to_v(hidden_states).reshape(rows, heads, head_dim)

            if rotary_emb is not None and use_fusion:
                cos, sin = rotary_emb
                q = fused_qknorm_rope(
                    q.unsqueeze(0), attn.norm_q.weight, cos, sin, attn.norm_q.eps).squeeze(0)
                k = fused_qknorm_rope(
                    k.unsqueeze(0), attn.norm_k.weight, cos, sin, attn.norm_k.eps).squeeze(0)
            elif rotary_emb is not None:
                q, k = attn.norm_q(q), attn.norm_k(k)
                q, k = _eager_rope(q, rotary_emb), _eager_rope(k, rotary_emb)
            else:
                q, k = attn.norm_q(q), attn.norm_k(k)

            packed = _packed_qkv_all_to_all(q, k, v, world=world, group=group)
            q, k, v = packed.split(head_dim, dim=-1)    # strided views; SDPA needs only stride(-1)==1

            # This is the one point in the whole model where a rank holds the entire packed
            # sequence, so it is the only place a sequence-level operator — sparse attention —
            # can be correct under context parallelism. `attention_fn` takes and returns
            # `(rows_full, heads_local, head_dim)`.
            if attention_fn is not None:
                out = attention_fn(q, k, v).contiguous()
            else:
                # SDPA wants (batch, heads, seq, dim); the flat rows are one document, batch is 1.
                out = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k.transpose(0, 1).unsqueeze(0),
                    v.transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                )
                out = out.squeeze(0).transpose(0, 1).contiguous()   # (rows_full, heads_local, dim)

            out = _packed_out_all_to_all(out, rows_local, world, group)
            out = out.reshape(batch, rows_local, heads * head_dim).to(hidden_states.dtype)
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
