## KR Retention Policy

This workflow is portfolio-owned and reviewer-discarded. As executor, you are
responsible for preserving every candidate through implementation, retry, and
reviewer judgment, but preservation does not force immediate same-family
refinement.

### Authority Boundary

- You may not make a final discard decision.
- Do not write a candidate into `discarded_candidates` or `rejected_candidates`
  as a terminal method decision.
- If evidence is negative or incomplete, record it as one of:
  `needs_reviewer_judgment`, `needs_retry`, `needs_rewrite`,
  `needs_operator_refinement`, or `infra_blocked`.
- Only the reviewer may write `REVIEWER-STATUS.json` with
  `"status": "discarded"`.
- You may mark a method `retained_parked`, `needs_retry_parked`, or
  `needs_reviewer_judgment` and move to a higher-impact family without
  discarding it.

### Non-Discard Cases

Do not discard for any of these conditions:

- Slurm allocation cancellation, no-output hang, missing stdout/stderr,
  filesystem delay, quota/intermittent infra, missing API key, or incomplete
  collection. These are retryable or diagnosable infrastructure failures.
- Microbench failure that indicates a possible implementation bug. Rewrite or
  repair the executor implementation; do not discard the method.
- Single-DiT/module-level evidence with no speedup when there is still plausible
  operator-level, layout, memory, launch, or kernel refinement space for the
  same method family. Preserve or park the method for reviewer judgment; do not
  let this condition block exploration of another ranked hotspot.

### Only Discardable Condition

A method can be discarded only after reviewer judgment verifies all of these:

- the negative result is valid method evidence: there is no implementation bug,
  mathematical/semantic bug, or infrastructure/execution bug such as missing
  evaluation, accidental fallback, launch failure, incomplete collection, or
  missing artifacts;
- the candidate has no measured warmup-after inference acceleration under the
  required gate;
- reviewer finds no credible remaining optimization space for that method at the
  operator/module level.

Until all three are true, preserve the method. Continue with retry or repair
when implementation, mathematical, or infrastructure/execution evidence is
incomplete. No acceleration is not enough by itself; discard also requires no
credible remaining optimization space.

### Required Status Language

When updating `AGENT-STATUS.json`, preserve method ownership. Prefer records
like:

```json
{
  "candidate_id": "<id>",
  "decision": "retained_parked",
  "reason": "method is preserved; current profile ranks another family higher",
  "next_decision": "switch_family_without_discard",
  "evidence": ["runs/<candidate>_microbench/gate_assess.json"]
}
```

For infra failures:

```json
{
  "candidate_id": "<id>",
  "decision": "needs_retry",
  "reason": "full run was cancelled/no-output before runtime heartbeat; not method evidence",
  "next_decision": "retry_full_diffusion_with_heartbeat",
  "evidence": ["runs/<run>/reject_note.json"]
}
```
