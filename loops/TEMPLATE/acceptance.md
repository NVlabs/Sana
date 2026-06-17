# Acceptance

## Required Gates

- artifact: pass
- official_config: pass
- performance: pass or rejected with a failure signature
- off_identity: pass or not_applicable; a failed guarded OFF path is a hard reject
- quantitative_quality: aligned LPIPS passes the target tier or the candidate is rejected
- visual_artifact: aligned pairwise Gemini passes the target tier or the candidate is rejected

Use the authoritative gate in `docs/fanout-loop-contract.md`. Collector
`quality.json` is telemetry; it is not promotion authority when it contradicts
the aligned gate.

## Promotion Threshold

Use `evals/profiles/official_video_t2v.toml` unless this loop documents a stricter
candidate-specific threshold.

## Rejection Conditions

- output video missing or empty
- official config changed without a separate baseline
- medium/high new visual artifact
- speedup below threshold after tuning
- implementation cannot be disabled cleanly

Each rejected candidate must be logged in `SEARCH_JOURNAL.md` with root cause and
the next-hypothesis requirement. Rejection does not complete the loop unless it
contributes to max_iters, early_stop, a real blocker, or structured-negative
evidence.

## Loop Completion

The loop is complete only when one of these is recorded in `AGENT-STATUS.json`
and `SUMMARY.md`:

- max_iters reached
- early_stop_patience reached
- real blocker
- structured negative
- explicit orchestrator release after reviewing best_per_tier
