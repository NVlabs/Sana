# Dimension: token_prune - token reduction, merging, masking, and routing

A search dimension for reducing token-level compute through pruning, merging,
masking, routing, mediator tokens, region-aware policies, dynamics-aware
selection, or token-wise caching. Native subagents should read
`search_space/02_token_pruning.md`, then inspect and modify the target-model
inference path directly in their isolated worktree.

## What it searches

`efficiency/techniques/token_prune.py` (`TokenPrune`) scores a model-declared
prunable token segment, gathers the kept tokens before the transformer-block
loop, runs the blocks on that shorter sequence, then scatters back to the
original sequence length after the loop. Subagents are free to replace this
framework helper with a direct inference-code experiment when that exposes the
model-specific token layout more clearly.

The helper's default knobs are only a starting point:

- `keep_ratio`: how many prunable tokens survive the gather.
- `method`: the token scoring strategy.
- `compensation`: how dropped-token hidden states are filled during scatter.

The open-ended search space is broader:

- ToMe-style token merging and importance-preserving merging.
- Shape-stable token/compute masking when gather/scatter is unsafe.
- Region-aware, dynamics-aware, cluster-aware, or attention-guided token
  selection.
- Dynamic token-density schedules by timestep, layer, region, modality, or sample
  difficulty.
- Context/reference token pruning for in-context or edit-style generation.
- Token-wise feature caching or conservative/aggressive alternating token
  policies when selection rather than caching is the core mechanism.

`keep_ratio >= 1.0` or an inactive schedule is the OFF path and leaves the
baseline hidden states unchanged. Any active candidate must restore the full
token layout cleanly before downstream computation observes it.

## Exploration Mode

Do not wait for a predeclared prunable-token seam. Inspect the real token layout,
position tensors, masks, cross-attention inputs, guidance branches, sequence
parallel metadata, and generated-video spans, then implement a candidate directly
where it is safest. Main-agent integration can later normalize the
implementation.

## Independent test

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/token_prune/test_token_prune.py
```

CPU-only; validates the token-prune technique through `efficiency` against a
registered model spec. The search-level check that this dimension stays
model-agnostic lives in `search/test_search.py`.

## Run it in the search

```bash
python search/search.py --model <model-id>
```

The main agent decides whether to launch this dimension. The CPU search output
is diagnostic only and must not block direct inference-code exploration.
