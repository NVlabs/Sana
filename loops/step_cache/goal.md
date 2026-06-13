# Goal: step_cache

## Objective

Wire a Cosmos3 diffusion step cache that can skip selected denoise-step
recompute through the generic `efficiency` engine while preserving a disabled
baseline path.

## Context

- Execution repo: `Sol-LTX-Infer`
- Orchestration repo: `autovideo`
- Eval profile: `evals/profiles/official_video_t2v.toml`
- Proven LTX source: `Sol-LTX-Infer` at
  `29d0d9e464000a2472345dcad51054b15aacca8d`

## Bounded Cosmos3 Work

1. Instantiate the `efficiency` `StepCache` plan in the Cosmos3 text-to-video
   runtime.
2. Wrap the per-denoise-step compute with `Plan.on_step`.
3. Add explicit env controls for enabled/disabled, skip steps, and optional
   delta scale.
4. Emit cache stats to `run.log` in a parsable form.
5. Keep TeaCache as the second flavor only after the model can provide a stable
   modulated-input signal.

## Constraints

- Preserve the official target-model config for comparable numbers.
- Default disabled behavior must match the baseline path.
- Do not run GPU or Slurm work while preparing this loop.
- Keep cache state scoped per sample/stream through `TechniqueContext.cache_key`.
- Do not edit shared `efficiency/` code for this loop.

## Done When

- `candidates/step_cache.toml` can be launched in dry-run mode.
- `python loops/step_cache/test_step_cache.py` passes in the sana env.
- A future runtime patch can be evaluated with canonical artifacts:
  `benchmark.json`, `quality.json`, `risk_notes.md`, `patch_summary.md`, and
  `collection.json`.
- The cache report states baseline vs candidate timing, speedup, quality gates,
  and skipped-step/hit/compute counts.
