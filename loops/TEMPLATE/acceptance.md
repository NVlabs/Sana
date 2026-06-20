# Acceptance

## Required Gates

- artifact: pass
- official_config: pass
- performance: measured and compared against baseline/frontier
- off_identity: pass or not_applicable; a failed guarded OFF path is a hard reject
- quantitative_quality: aligned LPIPS recorded for frontier retention and final speed-target selection
- visual_artifact: aligned pairwise Gemini recorded for frontier retention and final speed-target selection

Use the authoritative gate in `docs/fanout-loop-contract.md`. Collector
`quality.json` is telemetry; it is not the quality source of truth when it
contradicts the aligned gate.

## Frontier Retention

Retain a candidate when quality improves or speed/memory improves. Discard it
only when neither quality nor speed/memory improves.
Final low/medium/high selection happens after the loop budget closes as 1.5x,
2.0x, and 3.0x speed targets. Within a target, choose the best joint quality
profile using aligned pairwise Gemini severity/status and aligned LPIPS together;
LPIPS alone is not the selector.

## Rejection Conditions

- output video missing or empty
- official config changed without a separate baseline
- no quality improvement and no speed/memory improvement
- implementation cannot be disabled cleanly

Each rejected candidate must be logged in `SEARCH_JOURNAL.md` with root cause and
the next-hypothesis requirement. Rejection does not complete the loop unless it
contributes to max_iters or a real blocker. A structured-negative decision is
logged as a proposal/failure signature and does not stop the default
fixed-budget loop by itself.

Rejected/discarded candidates increment `no_improve_count` as telemetry; retained
quality or speed improvements reset it. Default fixed-budget frontier mode does
not stop on this counter.

## Loop Completion

The loop is complete only when one of these is recorded in `AGENT-STATUS.json`
and `SUMMARY.md`:

- max_iters reached
- real blocker
- terminal_pending_review with an agent_recommendation after budget triggers
- explicit orchestrator release after reviewing retained frontier candidates
