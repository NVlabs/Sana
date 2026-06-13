# Dimension: token_prune - feature-norm token pruning

A **model-agnostic search dimension**. It searches mid-loop token pruning
configs and composes them against whatever model the search targets. It names
no model; model specifics live in `models/<id>.toml` +
`efficiency/models/<id>_spec.py`.

## What it searches

`efficiency/techniques/token_prune.py` (`TokenPrune`) scores a model-declared
prunable token segment, gathers the kept tokens before the transformer-block
loop, runs the blocks on that shorter sequence, then scatters back to the
original sequence length after the loop. The primary migrated recipe scores
tokens by feature L2 norm (`method = "feat_norm"`), keeps the top fraction in
ascending sequence order, and fills dropped tokens from the previous full hidden
state (`compensation = "prev"`).

The dimension searches:

- `keep_ratio`: how many prunable tokens survive the gather.
- `method`: the token scoring strategy.
- `compensation`: how dropped-token hidden states are filled during scatter.

`keep_ratio >= 1.0` or an inactive schedule is the OFF path and leaves the
baseline hidden states unchanged. The active dimension writes the exclusive
`TOKEN_SET` seam, so only one token-set-changing technique can be active in a
composed plan.

## Why it's model-agnostic

`TokenPrune` only requires a model to expose the `prunable_tokens` capability.
The search calls `compose([build_technique("token_prune", **cfg)], spec)` for
the target model; if that model has not declared the required seam, the
dimension is skipped automatically. The dimension never chooses the token span
itself.

The per-model seam is the adapter's `prunable_segment` implementation and its
profile status in `models/<id>.toml [seam_status].prunable_tokens`. A model
should refine that segment to the generated video-token span, leaving prompt,
text, or other non-video tokens outside the pruned range.

## Migrated LTX-2.3 experience (the search prior)

`reference/recipe.md` records the LTX-2.3 stage-2 midpoint prune:
`keep_ratio = 0.5`, `method = "feat_norm"`, `compensation = "prev"`, active on
stage-2 steps `1-2` via `efficiency/presets.ltx_full_opt`. `reference/report.md`
records the warmed runtime result from `45.1s` to `41.1s`, approximately
`1.10x` speedup, with OFF recovering the baseline path. Those values seed
`dimension.toml`; the search remains free to evaluate neighboring ratios.

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

The target model profile decides whether this dimension is eligible,
composable, or skipped.
