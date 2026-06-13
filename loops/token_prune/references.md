# References

All Sol-LTX-Infer paths below were read from the read-only checkout at
`/home/haozhel/lustre/auto-video/Sol-LTX-Infer` on upstream commit
`29d0d9e`.

## Migrated Reference Files

- `reference/recipe.md`: distilled from
  `python/sglang/multimodal_gen/runtime/efficiency/presets.py`,
  `python/sglang/multimodal_gen/runtime/efficiency/techniques/token_prune.py`,
  and `scripts/slurm_ltx23_efftest_warm.sh`.
- `reference/report.md`: attributed summary of the LTX-2.3 warmed token-prune
  result and validation pattern from `scripts/slurm_ltx23_efftest_warm.sh`,
  `scripts/efficiency_selftest.py`, and the direction report datum
  `45.1s -> 41.1s`.
- `reference/wiring_cosmos3.md`: concrete Cosmos3 wiring notes derived from the
  token-prune seam contract in
  `python/sglang/multimodal_gen/runtime/efficiency/spec.py` and the current
  target spec in this repo.

## Upstream Names

- Source branch/report: `Sol-LTX-Infer @ 29d0d9e`, LTX-2.3 HQ efficiency
  framework smoke and warmed stage-2 midpoint prune.
- In-repo target loop: `loops/token_prune/`.
