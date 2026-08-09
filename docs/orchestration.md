# Orchestration Design

## Principle

`autovideo` should not duplicate implementation logic from `Sol-LTX-Infer`.
Instead, it should describe and launch transfeat that apply one of three
implementation modes:

- `baseline`: run the known reference command with official settings.
- `env_only`: set environment variables or runtime flags around existing code.
- `patch`: apply or point to a code change in `Sol-LTX-Infer`.
- `methodology`: document a profile-driven implementation goal that is not yet
  directly runnable, such as KWL-style fusion or model-specific NVFP4.

## M0

M0 creates the repo structure and documentation:

- folder layout
- transfeat manifest schema
- run artifact contract
- launch-agent protocol

## M1

M1 creates the runnable baseline path:

- `transfeat/wan22_ti2v_5b/baseline.toml`
- `scripts/launch_transfeat.py`
- dry-run mode for local validation
- Slurm mode for cluster execution
- run bundle generation under `runs/`

The baseline uses the official Cosmos3-Super config from `Sol-LTX-Infer`:

- model: `nvidia/Cosmos3-Super`
- resolution: `1280x720`
- frames: `189`
- fps: `24`
- steps: `35`
- guidance scale: `6.0`
- flow shift: `10.0`
- max sequence length: `4096`
- seed: `42`
- GPUs: `4`

## Launch Flow

1. Read a transfeat manifest.
2. Resolve repo paths and the `Sol-LTX-Infer` commit.
3. Create `runs/<timestamp>-<transfeat>/`.
4. Write `metadata.json`, `manifest.resolved.toml`, `launch.sh`, and
   `job.sbatch`.
5. In `dry-run`, stop after writing the bundle.
6. In `local`, execute `launch.sh`.
7. In `sbatch`, render `job.sbatch`; only submit when `--confirm-submit` is set.
8. When submitted, record the Slurm job ID in `metadata.json`.
9. Collect the bundle with `scripts/collect_run.py`.

## Promotion Criteria

A transfeat is not considered successful just because it runs faster. It must
also pass the quality contract:

- OFF equals baseline when the transfeat is disabled.
- ON is benchmarked with the official config.
- Timing is warmed or clearly labeled as compile-dominated.
- Output is compared against the baseline using quantitative metrics and visual
  artifact inspection.
- New artifacts such as snow, blur, mosaic/blocking, banding, ghosting, or
  flicker block promotion.

## M1.5

M1.5 closes the baseline loop:

```text
transfeat manifest
  -> launch_transfeat.py
  -> runs/<id>/
  -> Slurm/local execution
  -> collect_run.py
  -> metadata.json + outputs/collection.json + outputs/patch_summary.md
```

The collector is intentionally conservative. It marks a dry-run bundle as
`prepared`, a bundle with log and video as `completed`, and a bundle with error
patterns or missing video after execution as `failed`.
