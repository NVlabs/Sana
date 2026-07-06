Use this callable node contract when you have a concrete KWL candidate whose OFF
and ON module paths can be compared in one process.

Required artifact contract:

- write a durable `microbench.json`;
- write a durable `gate_assess.json`;
- include median/p25/p75/min/max latency, iteration count, OFF/ON ordering,
  tensor diff, shape/dtype, launch/profile evidence, and expected full
  contribution.

Do not launch full diffusion for a new KWL candidate before this contract is
satisfied.
