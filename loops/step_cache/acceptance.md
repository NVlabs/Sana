# Acceptance

## Required Gates

- artifact: pass
- official_config: pass
- off_identity: pass
- performance: pass for promotion, exploratory result recorded for first wiring
- quantitative_quality: pass
- visual_artifact: pass

## Cache-Specific Gates

- Disabled env path produces baseline behavior with the same prompt, seed, and
  official config.
- Enabled path logs cache stats with calls, computes, hits, and skipped steps.
- Candidate report compares baseline and candidate total, denoise, and
  stage-level seconds.
- Cache schedule is documented, including eligible stages and steps.
- TeaCache is not promoted until the Cosmos3 timestep/modulated-input signal is
  wired and logged.

## Promotion Threshold

Use `evals/profiles/official_video_t2v.toml`:

- experimental: denoise speedup >= 1.03x
- promotion: denoise speedup >= 1.10x with warmup recorded

## Rejection Conditions

- OFF path changes output or bypasses the baseline compute path.
- Output video is missing or empty.
- Official config differs from baseline without a separate baseline run.
- Medium/high visual artifact regression.
- Speedup falls below the profile threshold after tuning.
- Cache state leaks across prompts, samples, or stages.
