"""Wire the MiniMax-H3 fusions into a loaded model, one switch per fusion.

The kernels themselves live in `fusions.py` and are verified against eager there. This module only
decides where they replace upstream code, and it keeps the fusions independently switchable so a
benchmark can attribute an effect to one of them rather than reporting a lump.

Ordering with context parallelism matters exactly as it does for the caches: `HookRegistry` wraps
whatever `forward` is installed, so the last thing registered ends up outermost. The block-level
patch therefore goes on *inside* the CP split, or it would see the caller's full sequence going in
while the inner forward returns the rank's shard. The attention processor and the feed-forward sit
below the block and need no such care.

Shapes under CP: `norm_q`, `norm_k` and the rotary run *before* the Ulysses all-to-all, so they see
the rank's sequence shard with every head, and the rotary tables the CP plan split match it row for
row. The all-to-all that switches to full-sequence, local-head layout happens later, inside
`dispatch_attention_fn`.
"""

from __future__ import annotations

import torch

from fusions import (
    HAVE_TRITON, fused_qknorm_rope, fused_residual_gate_rmsnorm_modulate,
    fused_rmsnorm_modulate, fused_swiglu,
)


def _patch_blocks(transformer, use_modulate: bool, use_swiglu: bool) -> list:
    """Replace the block forward with one that fuses its elementwise chain."""
    restores = []
    for block in transformer.transformer_blocks:
        original = block.forward
        restores.append((block, "forward", original))

        def make(block=block, original=original):
            def forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
                eps = block.norm1.eps

                if use_modulate:
                    normed = fused_rmsnorm_modulate(
                        hidden_states, block.norm1.weight, scale_msa, shift_msa, adaln_indices, eps)
                else:
                    normed = block.norm1(hidden_states)
                    normed = normed * (1.0 + scale_msa.index_select(0, adaln_indices)) \
                        + shift_msa.index_select(0, adaln_indices)

                attn_output = block.attn(normed, rotary_emb, attention_mask)

                if use_modulate:
                    # One kernel produces both the new residual and the next half's normalized,
                    # modulated input, which is where most of the saving in this fusion sits.
                    hidden_states, normed = fused_residual_gate_rmsnorm_modulate(
                        hidden_states, attn_output, gate_msa, block.norm2.weight,
                        scale_mlp, shift_mlp, adaln_indices, block.norm2.eps)
                else:
                    hidden_states = hidden_states + gate_msa.index_select(0, adaln_indices) * attn_output
                    normed = block.norm2(hidden_states)
                    normed = normed * (1.0 + scale_mlp.index_select(0, adaln_indices)) \
                        + shift_mlp.index_select(0, adaln_indices)

                if use_swiglu:
                    swiglu, _, out_proj = block.ff.net
                    ff_output = out_proj(fused_swiglu(swiglu.proj(normed)))
                else:
                    ff_output = block.ff(normed)

                return hidden_states + gate_mlp.index_select(0, adaln_indices) * ff_output

            return forward

        block.forward = make()
    return restores


def _patch_attention(transformer) -> list:
    """Fuse qk-norm with the partial rotary inside every attention processor."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    restores = []
    for block in transformer.transformer_blocks:
        attn = block.attn
        original = attn.forward
        restores.append((attn, "forward", original))

        def make(attn=attn):
            def forward(hidden_states, rotary_emb=None, attention_mask=None):
                processor = attn.processor
                if attn.fused_projections:
                    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
                else:
                    query = attn.to_q(hidden_states)
                    key = attn.to_k(hidden_states)
                    value = attn.to_v(hidden_states)

                query = query.unflatten(-1, (attn.heads, -1))
                key = key.unflatten(-1, (attn.heads, -1))
                value = value.unflatten(-1, (attn.heads, -1))

                if rotary_emb is not None:
                    cos, sin = rotary_emb
                    query = fused_qknorm_rope(query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
                    key = fused_qknorm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
                else:
                    query = attn.norm_q(query)
                    key = attn.norm_k(key)

                hidden_states = dispatch_attention_fn(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                    backend=processor._attention_backend, parallel_config=processor._parallel_config,
                )
                hidden_states = hidden_states.flatten(2, 3).type_as(query)
                return attn.to_out[1](attn.to_out[0](hidden_states))

            return forward

        attn.forward = make()
    return restores


def install(transformer, modulate: bool = True, swiglu: bool = True, qknorm_rope: bool = True):
    """Install the requested fusions. Returns an uninstall callable."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is unavailable, so no fusion can be installed")

    from cache_line import with_cp_reapplied

    restores = []

    def do_install():
        if modulate or swiglu:
            restores.extend(_patch_blocks(transformer, modulate, swiglu))
        if qknorm_rope:
            restores.extend(_patch_attention(transformer))

    # The block patch must land inside the context-parallel split; installing it after the CP hooks
    # would put it outside them and feed it the unsharded sequence.
    with_cp_reapplied(transformer, do_install)

    def uninstall():
        def do_uninstall():
            for module, name, original in restores:
                setattr(module, name, original)
            restores.clear()

        with_cp_reapplied(transformer, do_uninstall)

    return uninstall
