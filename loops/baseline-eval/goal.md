# Goal: baseline-eval

Run the official Cosmos3-Super baseline through the autovideo launch/collect
flow and produce the baseline report artifacts.

## Done When

- `candidates/baseline.toml` launches with `--mode sbatch --confirm-submit`
- `scripts/collect_run.py` marks the run `completed`
- `outputs/patch_summary.md` records total, denoise, and decode timing
- sampled frames exist for visual comparison
