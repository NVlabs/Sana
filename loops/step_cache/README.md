# Loop: step_cache

## Purpose

Bring the proven LTX-2.3 diffusion-cache recipe into an autovideo loop for
Cosmos3: skip selected denoise-step/block recompute while preserving a clean
OFF path.

## LTX-2.3 Source Story

The LTX cache work came from `Sol-LTX-Infer` at
`29d0d9e464000a2472345dcad51054b15aacca8d`. The non-HQ cache runner carried
fixed cache variants (`cache_pab_late12_w3`, TeaCache threshold/start variants,
and `cache_dbcache_aggressive`) plus KWL combinations. The matrix runner paired
HQ and non-HQ TeaCache variants across prompts, emitted compare videos, and
called the cache report helper for baseline-vs-candidate timing and skip stats.

This loop migrates only the cache-specific pieces:

- `reference/recipe.md`: cache env/run knobs from the two LTX scripts.
- `reference/make_cache_report.py`: a canonical-artifact report helper for
  `benchmark.json`, `quality.json`, `run.log`, and `patch_summary.md`.
- `reference/report.md`: the required cache-report shape.

## Mapping To `efficiency/`

The generic runtime primitives already live in the repo:

- `efficiency/techniques/step_cache.py`: `StepCache`, a whole-step
  denoiser-output cache. It is active exactly on its skip schedule and writes
  the exclusive `STEP_OUTPUT` seam.
- `efficiency/techniques/teacache.py`: `TeaCache`, a thresholded residual-replay
  cache driven by a model-supplied timestep/modulated-input signal.

The LTX full-opt preset represents the tuned stage-1 skip cluster as
`16-28` for `stage1`. The independent test mirrors that shape: active in the
skip cluster, inactive before/after it and in other stages, with OFF returning
the byte-identical result from `run_step()`. Because inactive techniques are
true plan-level no-ops, `StepCache` seeds its buffer on the first active step
when no cached output exists; later active steps can reuse that cached output.

## Cosmos3 Wiring Needed

`efficiency/models/cosmos3_spec.py` already declares the conservative Cosmos3
model spec. The future Cosmos3 runtime patch should:

- Build a `Plan` with `StepCache(skip=by_stage({"stage1": at_steps("16-28",
  True, False)}, default=False))`, or a Cosmos3-tuned schedule after profiling.
- Wrap the denoise-step call with `plan.on_step(TechniqueContext(...), run_step)`.
- Keep a persistent `scratch` dictionary for the generation and a stable
  `cache_key` per sample/stream.
- Pass a stage label that separates any stage-specific schedules.
- Account for the first active step seeding the cache if no earlier active
  cache state exists.
- Leave the feature disabled by default and expose an explicit env knob, so
  disabled/OFF behavior is the baseline path.
- For TeaCache, stash the model's timestep-conditioned modulated-input signal
  under `("teacache_signal", cache_key)` before `plan.on_step`; without that
  signal, `TeaCache` computes the baseline step.
- Log cache stats in `run.log`: computes, hits, calls, and skipped step indices.

## Candidate

`candidate.toml` is a methodology manifest for the launcher. It keeps the
official Cosmos3 config and records the expected future touch points, but does
not claim an implemented runtime speedup until the Cosmos3 denoise loop is
wired in `Sol-LTX-Infer`.

## Eval

`eval.toml` points at `evals/profiles/official_video_t2v.toml`.

## Independent Test

Run:

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/step_cache/test_step_cache.py
```

The test is CPU-only and validates `StepCache` through the `efficiency`
composition engine against the registered Cosmos3 spec.

## Status

ready-for-codex
