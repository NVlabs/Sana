# Bernini (full) — text-to-video baseline

Onboards **full Bernini** (`ByteDance/Bernini-Diffusers`: Qwen2.5-VL-7B planner +
Wan2.2-T2V-A14B renderer) into Sol-LTX-Infer, parallel to `sana_video`.

## Baseline

- Hot latency ≈ **129s `text_to_vae_decode`** (median over the 5-prompt
  validation set). The DiT diffusion loop dominates; VAE decode + planner + text
  encode are small.
- The runnable code is **vendored and self-contained** under
  `runtime/bernini_baseline/bernini_src/` (a clean copy of the upstream pristine
  tree, no git history). Model weights + third-party libs are referenced by
  absolute path (they cannot live in git).

## Validation set (first 5 official t2v prompts)

`models/bernini/prompts/t2v_val5.json` (+ `.txt`):
`prompt_00_polar_bear_guitar`, `prompt_01_fox_autumn_forest`,
`prompt_02_dog_beach_frisbee`, `prompt_03_astronaut_satellite`,
`prompt_04_chef_pancake`. Used as the baseline timing set and the quality
comparison set.

## Files

- `models/bernini.toml` — flat profile (`text_to_vae_decode` 口径).
  `[baseline]` filled after the run.
- `models/bernini/model.toml` — copy contract; `bernini_src` is copied into
  experiment worktrees (editable); weights/deps are `reference_only`.
- `candidates/bernini_baseline.toml`, `evals/profiles/official_video_t2v_bernini.toml`.
- `runtime/bernini_baseline/` — the runtime (see its README).

## Produce + persist the baseline

```bash
python scripts/launch_candidate.py candidates/bernini_baseline.toml --mode sbatch --confirm-submit
python scripts/collect_run.py runs/<produced_run_dir>
```

Copy `runs/<run_dir>/outputs/benchmark.json` (median `text_to_vae_decode` over
the 5 prompts) into `models/bernini.toml [baseline]`. Reference videos persist
at `runs/<run_dir>/outputs/videos/<prompt>.mp4` (+ `out.mp4` = polar bear).
