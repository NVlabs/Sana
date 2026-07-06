Use this callable node contract as the ordinary loop evaluation when you have a
concrete KWL candidate whose OFF and ON single-DiT or module paths can be
compared in one process.

Required artifact contract:

- write a durable `microbench.json`;
- write a durable `gate_assess.json`;
- include median/p25/p75/min/max latency, iteration count, OFF/ON ordering,
  tensor diff, shape/dtype, launch/profile evidence, and expected transformer
  contribution.

Do not launch full diffusion for a new KWL candidate as part of this ordinary
loop contract. Full diffusion is reserved for terminal validation after reviewer
exit intent.

Microbench gate failure is not a discard decision in workflow `kr`.

- If tensor drift is caused only by floating-point contraction, reduction order,
  or rounding, record the drift and keep the method for reviewer semantic
  judgment.
- If the algorithm is mathematically correct and has no semantic error, keep
  the method even when strict numerical drift is large.
- If the microbench exposes an implementation or algorithm semantic error,
  rewrite the implementation and rerun the microbench.
- If latency regresses but the method still has plausible operator-level or
  layout refinement space, continue refining or ask reviewer for judgment.
