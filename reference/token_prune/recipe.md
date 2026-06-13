<!-- ported from Sol-LTX-Infer python/sglang/multimodal_gen/runtime/efficiency/presets.py @ 29d0d9e; corroborated by scripts/slurm_ltx23_efftest_warm.sh @ 29d0d9e -->

# LTX-2.3 Token-Prune Recipe

This is the LTX-2.3 stage-2 midpoint token-prune configuration migrated for the
Cosmos3 token-prune loop. It is a reference recipe only; do not execute this
file.

## Technique

Use the shared `TokenPrune` runtime technique:

```python
TokenPrune(
    keep_ratio=by_stage({"stage2": const(0.5)}, default=1.0),
    method="feat_norm",
    compensation="prev",
    enabled=by_stage({"stage2": at_steps("1-2", True, False)}, default=False),
)
```

## Runtime Meaning

- Step `0` or the first active call with no previous buffer runs the full token
  set and seeds `prev`.
- Active stage2 steps score each prunable token by feature L2 norm averaged over
  the batch.
- The block loop runs only the top `round(S * 0.5)` prunable tokens in ascending
  sequence order.
- Scatter restores the original sequence length and fills dropped tokens from
  the previous full hidden-state buffer.
- Stage1 and all non-enabled steps use `keep_ratio=1.0` or disabled scheduling,
  which is the OFF identity path.

## LTX Validation Snippet

The warmed smoke script used the framework-scored path with:

```bash
export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_RATIO=0.5
export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_METHOD=feat_norm
export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_STEPS=1,2
bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl_stage1_cache_core
```

The paired OFF run leaves those prune env vars unset and runs the same HQ path.

## Proven Result

The LTX-2.3 direction report recorded warmed runtime improvement from `45.1s` to
`41.1s` with the stage-2 midpoint prune enabled, while OFF recovered the
baseline path.
