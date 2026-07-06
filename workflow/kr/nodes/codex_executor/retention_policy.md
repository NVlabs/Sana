## KR Retention Policy

This workflow is method-owned and reviewer-discarded. As executor, you are
responsible for carrying a candidate method through implementation, retry, and
operator-level refinement until reviewer judgment is possible.

### Authority Boundary

- You may not make a final discard decision.
- Do not write a candidate into `discarded_candidates` or `rejected_candidates`
  as a terminal method decision.
- If evidence is negative or incomplete, record it as one of:
  `needs_reviewer_judgment`, `needs_retry`, `needs_rewrite`,
  `needs_operator_refinement`, or `infra_blocked`.
- Only the reviewer may write `REVIEWER-STATUS.json` with
  `"status": "discarded"`.

### Non-Discard Cases

Do not discard for any of these conditions:

- Slurm allocation cancellation, no-output hang, missing stdout/stderr,
  filesystem delay, quota/intermittent infra, missing API key, or incomplete
  collection. These are retryable or diagnosable infrastructure failures.
- Microbench numerical drift by itself. If the math/algorithm is correct and
  there is no semantic error, retain the method and move to reviewer judgment;
  do not launch full diffusion unless terminal validation is requested.
- Microbench failure that indicates a possible implementation bug. Rewrite or
  repair the executor implementation; do not discard the method.
- Single-DiT/module-level evidence with no speedup when there is still plausible
  operator-level, layout, memory, launch, or kernel refinement space for the
  same method family.

### Only Discardable Condition

A method can be discarded only after reviewer judgment verifies all of these:

- there is a smooth single-DiT/module-level evaluation with durable artifacts;
- the candidate has no meaningful speed, memory, or correctness/quality proxy
  improvement at that level;
- the negative result is not caused by infra, collection, missing eval, or
  accidental launch failure;
- reviewer finds no credible remaining optimization space for that method at
  the operator/module level.

Until all four are true, continue with retry, repair, or a narrower
operator-level refinement of the same method.

### Required Status Language

When updating `AGENT-STATUS.json`, preserve method ownership. Prefer records
like:

```json
{
  "candidate_id": "<id>",
  "decision": "needs_reviewer_judgment",
  "reason": "smooth single-DiT eval exists but speedup is neutral; reviewer must decide whether operator-level refinement remains",
  "next_decision": "reviewer_judgment_required",
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
