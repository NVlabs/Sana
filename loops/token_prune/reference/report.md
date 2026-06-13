<!-- ported from Sol-LTX-Infer scripts/slurm_ltx23_efftest_warm.sh @ 29d0d9e; report datum attributed to the LTX-2.3 token_prune direction note @ 29d0d9e -->

# LTX-2.3 Token-Prune Report Excerpt

The LTX-2.3 efficiency framework smoke compared the same HQ path with and
without framework-scored stage-2 midpoint token pruning.

## Validated Configuration

- Base path: LTX-2.3 HQ 1080p 10s runner.
- Optimized stack context: KWL fusions, NVFP4 FFN, stage1 cache core, and PISA
  stage2 sparse attention were active in the warmed smoke.
- Token prune: `ratio=0.5`, `method=feat_norm`, `steps=1,2`.
- OFF comparator: identical command with prune env vars unset.
- Validation signal: the ON run logs the `stage2 midpoint prune` line and both
  ON/OFF runs produce `out.mp4`.

## Result

The proven warmed LTX-2.3 direction result was:

| Mode | Warmed Time |
| --- | ---: |
| OFF / baseline path | `45.1s` |
| ON / token prune | `41.1s` |

That is approximately `1.10x` speedup for the pruned path. The Cosmos3 loop
keeps the same technique parameters and moves only the model-specific seam
wiring.

## Porting Constraint

This report is a migrated attribution for the loop. The actual scoring and
gather/scatter implementation remains in the shared `efficiency/` package.
