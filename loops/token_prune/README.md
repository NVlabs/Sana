# Loop: token_prune

## Purpose

Port the proven LTX-2.3 feature-norm token-pruning recipe into a bounded
Cosmos3 acceleration loop.

## LTX-2.3 Reference

The source recipe is the LTX-2.3 stage-2 midpoint prune from
`Sol-LTX-Infer @ 29d0d9e`. It uses the generic
`efficiency.techniques.token_prune.TokenPrune` technique with:

- `keep_ratio = 0.5`
- `method = "feat_norm"`
- `compensation = "prev"`
- active only in stage2 steps `1,2`
- `keep_ratio >= 1.0` or disabled schedule as the OFF path

The proven LTX-2.3 result is warmed denoise/runtime improvement from `45.1s` to
`41.1s` with OFF matching the baseline path. That is the success story this
loop preserves for Cosmos3 rather than re-implementing token scoring.

## Mapping To `efficiency/`

The loop references the shared implementation instead of copying it:

- `efficiency/techniques/token_prune.py` owns scoring, gather, scatter, and
  `prev` compensation.
- `efficiency/presets.py` shows the LTX schedule:
  `by_stage({"stage2": const(0.5)}, default=1.0)` plus
  `by_stage({"stage2": at_steps("1-2", True, False)}, default=False)`.
- `efficiency/selftest.py` sections `[4]` and `[5]` are mirrored by this loop's
  independent Cosmos3 test.

## Cosmos3 Wiring Objective

Cosmos3 already declares `Capability.PRUNABLE_TOKENS` in
`efficiency/models/cosmos3_spec.py`. The implementation branch that promotes
this loop should:

- keep `PRUNABLE_TOKENS` declared only while the runtime exposes a valid
  prunable token segment;
- refine `cosmos3_spec.prunable_segment` to return the generated video-token
  span, not understanding/text/prompt tokens;
- call the composed plan around the Cosmos3 DiT block loop in
  `runtime/models/dits/cosmos3video.py`;
- add `prune_gather` and `prune_scatter` accessors if the pruned forward must
  carry coordinates, timestep embeddings, masks, or other per-token side data;
- guard runtime activation with env/config so OFF recovers the baseline path.

## Candidate

Use `candidate.toml` in this loop or the launcher copy at
`candidates/token_prune.toml`.

## Eval

`eval.toml` points at `evals/profiles/official_video_t2v.toml`.

## Independent Test

Run the CPU-only gate with the torch-enabled env:

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/token_prune/test_token_prune.py
```

The test composes `TokenPrune` against `get_model_spec("Cosmos3")`, asserts
ratio `1.0` is an identity, then checks ratio `0.5` gathers `S=16` to `K=8` and
scatters back to `S=16` across steps.

## Status

`ready-for-codex`
