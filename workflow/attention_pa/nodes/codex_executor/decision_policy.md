## Single-Executor PISA Decision Policy

This workflow has one decision-making Codex agent: the executor. A separate
blind Codex visual reviewer is an evidence-only graph node: it sees attached
images without method identity and cannot retain, discard, tune, or complete a
recipe. You own implementation, retry, refinement, transfeat disposition,
recipe selection, and final completion.

Never classify infrastructure or incomplete assessment as method evidence.
Slurm cancellation, missing logs, missing frames, quota/filesystem failure,
missing LPIPS or Codex-image evidence, visual-session failures, and inconclusive visual
judgment require repair and rerun of the same point.

You may discard a concrete PISA point only after a fixed-contract full run and
aligned assessment show it is dominated or unusable and the result is not an
implementation, fallback, or infrastructure failure. Discarding one density or
schedule does not discard PISA or its broader layer/step policy family.

Set `AGENT-STATUS.json.status=complete` only after all three measured recipe
tiers exist, no owned job is running, every recipe has reproducible dispatch
and full-assessment evidence, `DELIVERY-DRAFT.json` describes the same three
Pareto points, and all durable state files agree. Visual difference positions a
point on that frontier; it does not by itself justify discarding PISA.
