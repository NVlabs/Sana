<!-- ported from Sol-LTX-Infer python/sglang/multimodal_gen/runtime/efficiency/spec.py @ 29d0d9e; adapted to the Cosmos3 target spec in this worktree -->

# Cosmos3 Wiring Notes

This is the concrete wiring plan for promoting the token-prune loop onto
Cosmos3.

## ModelSpec

`efficiency/models/cosmos3_spec.py` already declares:

```python
Capability.BLOCKS
Capability.PRUNABLE_TOKENS
```

Keep `PRUNABLE_TOKENS` only if the model spec exposes a correct prunable token
span. The current default returns the full sequence. The implementation should
refine it to the generated video-token span so prompt, understanding, text, and
other non-video tokens are not dropped.

## Gather And Scatter

The generic path handles a plain `[B, S, C]` hidden tensor. Add
`prune_gather(payload, keep_idx, ctx)` and
`prune_scatter(output, keep_idx, full_len, ctx, compensation)` if the Cosmos3
forward needs hidden-state pruning to stay aligned with:

- token coordinates or patch positions;
- timestep or guidance side tensors;
- attention masks;
- sequence-parallel shard metadata;
- any per-token conditioning carried through the DiT block loop.

## Runtime Integration

The future patch should compose:

```python
TokenPrune(
    keep_ratio=0.5,
    method="feat_norm",
    compensation="prev",
)
```

around the Cosmos3 DiT block loop in
`runtime/models/dits/cosmos3video.py`, using env/config to disable it by
default. OFF must skip the gather/scatter path and recover the baseline block
loop.

## Sequence Parallelism

The official Cosmos3 config uses multiple GPUs. If token pruning is active under
sequence parallelism, set the spec behavior so selection is per-rank local when
needed to keep shards balanced. Validate ON/OFF with the same prompt, seed, and
official config before claiming speedup.
