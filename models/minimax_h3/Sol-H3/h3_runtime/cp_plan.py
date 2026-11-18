"""Context-parallel plan for `MiniMaxH3Transformer3DModel`.

Upstream ships no `_cp_plan` for MiniMax-H3 (flux, wan, ltx2, cosmos3 and helios all have one), so
diffusers cannot shard its packed sequence. Everything else is already in place:
`MiniMaxH3AttnProcessor` carries `_parallel_config` and forwards it to `dispatch_attention_fn`, and
the default `native` attention backend is registered as context-parallel capable. This module
supplies the missing plan and passes it to `enable_parallelism(cp_plan=...)`, so the vendored
diffusers source stays untouched and the baseline keeps running against pristine upstream.

Why the plan looks different from wan's
---------------------------------------
wan splits `hidden_states` at the model boundary. MiniMax-H3 cannot: its forward takes three
*per-modality* streams and scatters them into the packed buffer with `index_copy` using global row
indices, then reverses that with `index_select` at the heads:

    proj_in / audio_proj_in / context_embedder+token_refiner   three streams projected
    hidden_states = zeros(seq_len); index_copy(text|video|audio)   <-- needs global indices
    adaln_indices = timestep_indices * MODALITY_NUM + token_tags.clamp(min=0)
    for block in transformer_blocks: ...                        <-- the shardable region
    norm_out(hidden_states, temb, timestep_indices)
    proj_out(...).index_select(1, video_indices)                <-- needs global indices

Splitting at the boundary would shrink `sequence_length` while the scatter indices stayed global.
So the region that gets sharded is the block stack alone, entered after the scatter and left before
the gather back to modalities.

Per-tensor reasoning
--------------------
* `rope` — the loop hands the same `rotary_emb` to every block, so splitting the rope module's two
  outputs once covers all of them. `(seq_len, 2*3*rope_freq_dim)` each, hence `split_dim=0`.
* `adaln_indices` — also rebuilt full and passed fresh to every block by the loop, so unlike
  `hidden_states` it has to be split at *every* block, hence the `*` wildcard. Each block uses it
  only as `index_select(0, adaln_indices)` on its modulation table, so a block holding local rows
  needs exactly the matching local slice.
* `hidden_states` — enters the stack once; blocks 1..N-1 receive block 0's already-sharded output,
  so it is split at block 0 only. Block 0 therefore carries two split hooks, one from the wildcard
  and one of its own. That is safe: the hook replaces positional arguments in place rather than
  moving them into kwargs, and the two hooks touch different parameters.
* `timestep_indices` — `norm_out` runs on local rows and indexes its shift/scale table per row, so
  its index vector is split to match. Splitting here rather than gathering before `norm_out` avoids
  a second gather point.
* `proj_out` / `audio_proj_out` — the gather. It has to happen before `index_select(1, ...)` maps
  rows back to video and audio, and it cannot be attached to the last block: a module id maps to
  either a split plan or a gather plan, and the last block already needs the `adaln_indices` split.
* `temb` — indexed by timestep, not by row. Shared by all rows, so it is never split.
* `attention_mask` — not split, and it must stay `None`. diffusers builds the packed sequence
  without padding rows precisely so no mask is needed; a mask would be `(seq_len, seq_len)` and
  would have to be sliced on rows while staying full on keys. `assert_no_attention_mask` below
  guards the assumption instead of letting it silently produce wrong numbers.
"""

from __future__ import annotations

from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput


MINIMAX_H3_CP_PLAN = {
    # (cos, sin), each (seq_len, 2 * 3 * rope_freq_dim) -> split the sequence axis of both outputs.
    "rope": {
        0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
    },
    # Passed full to every block by the sampling loop, so every block splits it.
    "transformer_blocks.*": {
        "adaln_indices": ContextParallelInput(split_dim=0, expected_dims=1, split_output=False),
    },
    # Threaded block to block, so only the entry point splits it.
    "transformer_blocks.0": {
        "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
    },
    # Runs on local rows; its per-row modulation index must be local too.
    "norm_out": {
        "timestep_indices": ContextParallelInput(split_dim=0, expected_dims=1, split_output=False),
    },
    # Back to the full sequence before the heads select video and audio rows by global index.
    "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    "audio_proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
}


def assert_no_attention_mask(transformer) -> None:
    """Fail loudly if a run ever produces a padded packed sequence.

    The plan does not shard `attention_mask`. diffusers builds the sequence without padding rows, so
    the mask is `None` and the assumption holds — but a future packing change would make it
    `(seq_len, seq_len)`, and an unsharded mask against sharded rows would broadcast into wrong
    numbers rather than raise.
    """
    original = transformer.forward

    def guarded(*args, **kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None and len(args) >= 11:
            mask = args[10]
        if mask is not None:
            raise RuntimeError(
                "The packed sequence carries padding rows, so `attention_mask` is not None. The "
                "context-parallel plan does not shard it, and an unsharded mask against sharded "
                "rows is silently wrong. Shard the mask on its row axis before enabling CP here."
            )
        return original(*args, **kwargs)

    transformer.forward = guarded
