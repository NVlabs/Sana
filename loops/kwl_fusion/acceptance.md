# Acceptance

## Required Gates

- artifact: all required official-profile artifacts exist.
- official_config: candidate config matches `evals/profiles/official_video_t2v.toml`.
- off_identity: all KWL env flags disabled recover the baseline Cosmos3 path.
- performance: denoise speedup is recorded against the baseline candidate.
- quantitative_quality: pass the official quality profile or record a concrete
  deferred reason for dry-run only.
- visual_artifact: pass the official visual gate before promotion.

## Promotion Threshold

Use `evals/profiles/official_video_t2v.toml`:

- experimental: `>= 1.03x` denoise speedup.
- promotion: `>= 1.10x` denoise speedup with warmup/cache state recorded.

## KWL-Specific Checks

- No sparse attention, token pruning, step cache, quantization, scheduler
  changes, prompt changes, CFG changes, LoRA changes, resolution changes, or
  frame-count changes are included in this candidate.
- Each fused path has an env flag and can be leave-one-out ablated.
- OFF is the baseline code path, not a separate approximation.
- Any non-bitwise difference is explained as kernel-level floating-point order
  only and must pass side-by-side visual review.

## Rejection Conditions

- Output video missing or empty.
- Official config changed without a new baseline.
- OFF path differs from baseline.
- Medium or high new visual artifact.
- Speedup below threshold after tuning.
- A fused path cannot be disabled cleanly.
