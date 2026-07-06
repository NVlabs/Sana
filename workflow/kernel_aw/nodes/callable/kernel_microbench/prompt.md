Use this callable node contract as the ordinary loop evaluation when you have a
concrete target-model transformer-kernel candidate whose OFF and ON single-DiT or module
paths can be compared in one process.

Required artifact contract:

- write a durable `microbench.json`;
- write a durable `gate_assess.json`;
- include median/p25/p75/min/max latency, iteration count, OFF/ON ordering,
  tensor diff, shape/dtype, launch/profile evidence, and expected transformer
  contribution.
- include `candidate_id`, `measurement_scope`, `prompt_count`,
  `steps_per_prompt`, `calls_per_dit`, and whether startup/compile is one-time;
- use the registry-resolved active implementation path. A synthetic module trace
  must be labeled `screening_only` and cannot stand in for cumulative full-DiT
  evidence;
- write the current gate path to `AGENT-STATUS.json.active_gate` and the matching
  candidate to `active_candidate_id` so stale gates are not reused;
- multiply recurring savings by all prompts, steps, model calls, and call sites
  represented by the denominator. Charge compile/startup once per process.

Do not use the five-prompt bundled wall measurement as isolated denoise time or
divide it by the denoising-step count to estimate one DiT. Use a stage-isolated
full DiT measurement for contribution percentages. If unavailable, mark the
projection unknown until `dit_profile` produces it.

Do not launch full diffusion for a new kernel candidate as part of this ordinary
loop contract. Full diffusion is reserved for terminal validation after reviewer
exit intent.

Microbench gate failure is not a discard decision in workflow `kernel_aw`.

- If tensor drift is caused only by floating-point contraction, reduction order,
  or rounding, record the drift and keep the method for reviewer semantic
  judgment.
- If the algorithm is mathematically correct and has no semantic error, keep
  the method even when strict numerical drift is large.
- If the microbench exposes an implementation or algorithm semantic error,
  rewrite the implementation and rerun the microbench.
- If latency regresses but the method still has plausible operator-level or
  layout refinement space, retain or park it for reviewer judgment. Do not let a
  preserved low-priority method prevent switching to a larger profiled hotspot.
