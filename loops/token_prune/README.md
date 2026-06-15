# Dimension: token_prune - feature-norm token pruning

A search dimension for token pruning, token masking, or token-routing
experiments. Native subagents should read `search_space/02_token_pruning.md`,
then inspect and modify the Cosmos3 inference path directly in their isolated
worktree.

## What it searches

`efficiency/techniques/token_prune.py` (`TokenPrune`) scores a model-declared
prunable token segment, gathers the kept tokens before the transformer-block
loop, runs the blocks on that shorter sequence, then scatters back to the
original sequence length after the loop. Subagents are free to replace this
framework helper with a direct inference-code experiment when that exposes the
model-specific token layout more clearly.

The dimension searches:

- `keep_ratio`: how many prunable tokens survive the gather.
- `method`: the token scoring strategy.
- `compensation`: how dropped-token hidden states are filled during scatter.

`keep_ratio >= 1.0` or an inactive schedule is the OFF path and leaves the
baseline hidden states unchanged. Any active candidate must restore the full
token layout cleanly before downstream computation observes it.

## Exploration Mode

Do not wait for a predeclared prunable-token seam. Inspect the real token layout,
position tensors, masks, cross-attention inputs, and generated-video spans, then
implement a candidate directly where it is safest. Main-agent integration can
later normalize the implementation.

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
