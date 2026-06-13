# Acceptance

## Required Gates

- artifact: pass with canonical `outputs/benchmark.json`, `outputs/quality.json`,
  `outputs/risk_notes.md`, `outputs/patch_summary.md`, and
  `outputs/collection.json`.
- official_config: pass against `evals/profiles/official_video_t2v.toml`.
- performance: at least 1.03x denoise speedup for experimental status and 1.10x
  for promotion unless an exploratory result is explicitly recorded.
- off_identity: disabled `SGLANG_HQ_ENABLE_TE_NVFP4_FFN` must recover the
  baseline path; enabled NVFP4 is lossy and is not byte-exact by design.
- quantitative_quality: record frame metrics and PSNR, but use PSNR only as a
  diagnostic signal.
- visual_artifact: pass with `outputs/side_by_side.mp4` and the configured
  visual judge result for any non-dry-run NVFP4 candidate.

## Promotion Threshold

Use `evals/profiles/official_video_t2v.toml`:

- `performance.primary_metric = "denoise_s"`
- `performance.min_speedup_for_experimental = 1.03`
- `performance.min_speedup_for_promotion = 1.10`
- `visual_artifact.side_by_side_required = true`

## Rejection Conditions

- NVFP4 cannot be disabled without changing the baseline path.
- The official config changes without a separate matching baseline.
- Output video is missing, empty, wrong duration, or wrong frame count.
- Side-by-side visual judge finds medium or high new artifacts.
- Speedup is below the promotion threshold after tuning.
- Runtime requires unavailable CUDA, TransformerEngine, or Blackwell support and
  no blocker is recorded.
