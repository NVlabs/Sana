# Goal: kwl_fusion

## Objective

Implement and evaluate Cosmos3 operator-fusion build wiring modeled on the
LTX-2.3 KWL path. The bounded target is a flag-gated, lossless operator-only
fusion candidate for `nvidia/Cosmos3-Super` that can be launched through the
autovideo candidate runner.

## Context

- Execution repo: `Sol-LTX-Infer`
- Orchestration repo: `autovideo`
- Eval profile: `evals/profiles/official_video_t2v.toml`
- Generic transform: `efficiency/transforms/kwl_fusions.py`
- Reference loop: `loops/kwl_fusion/reference/`

## Implementation Scope

- Wire Cosmos3 model build code to consume `SGLANG_HQ_KWL_*` flags.
- Add only Cosmos3-specific fusion code needed for profiled hot op chains.
- Keep every fusion independently disableable.
- Preserve the official target-model config for comparable numbers.
- Do not change scheduler, step count, prompt, CFG, resolution, frames, or
  quality gates.

## Done When

- `candidates/kwl_fusion.toml` launches in dry-run mode.
- OFF env settings recover the baseline Cosmos3 path.
- ON env settings install the intended Cosmos3 fused paths.
- `scripts/collect_run.py` produces canonical artifacts:
  `benchmark.json`, `quality.json`, `risk_notes.md`, `patch_summary.md`, and
  `collection.json`.
- Official-profile speed and quality gates are reported.
