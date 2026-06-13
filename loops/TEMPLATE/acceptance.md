# Acceptance

## Required Gates

- artifact: pass
- official_config: pass
- performance: pass or exploratory result recorded
- off_identity: pass or not_applicable
- quantitative_quality: pass or explicitly_deferred
- visual_artifact: pass or explicitly_deferred for dry-run only

## Promotion Threshold

Use `evals/profiles/official_video_t2v.toml` unless this loop documents a stricter
candidate-specific threshold.

## Rejection Conditions

- output video missing or empty
- official config changed without a separate baseline
- medium/high new visual artifact
- speedup below threshold after tuning
- implementation cannot be disabled cleanly
