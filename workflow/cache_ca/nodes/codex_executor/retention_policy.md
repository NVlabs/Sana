## Single-Executor Cache Decision Policy

This workflow has one decision-making Codex agent: the executor. A separate
blind Codex visual reviewer is an evidence-only graph node: it sees attached
images without method identity and cannot retain, discard, tune, or complete a
recipe. You own implementation, full evaluation, parameter adaptation,
retain/discard decisions, final recipe selection, and workflow completion.

### Evidence Boundary

Every cache-method judgment requires full-run evidence:

- completed target-model full run with `outputs/benchmark.json`;
- frames or `outputs/out.mp4`;
- fixed prompt/config: the first five prompts of the target model's validation
  set at the model's official eval profile (resolution, duration, frame count,
  fps, steps, guidance, flow shift, motion score);
- durable merged `assess_verdict.json` from the workflow-owned visual reviewer;
- numeric speed fields and numeric `lpips_max`;
- independent blind Codex pairwise visual judgment.

Microbench, single-DiT, and module measurements are diagnostics only. They may
select the next implementation or parameter point but cannot retain, discard,
or complete a cache family by themselves.

The final state must also include `CACHE-SEARCH-STATE.json` with explicit
TeaCache, EasyCache, and TaylorSeer family state. These are the only config
families. Every viable family requires a faithful full-run seed config and
at least one child refinement selected from measured speed, cache statistics,
LPIPS, and Codex visual evidence.

A cross-family quality decision additionally requires matched full-run E2E
time: normalize each config as `config_total_s / baseline_total_s`, use
the same baseline and execution conditions, and compare only points within 2%
relative `time_ratio`. Tune toward a common target when points are unmatched.
Module timing, single-DiT timing, projected speedup, or different time budgets
cannot establish that one family has better quality than another.

### Retry And Refinement Cases

Do not discard for any of these conditions:

- Slurm cancellation, no-output hang, missing logs, filesystem or quota delay,
  missing LPIPS dependency, missing Codex visual result, incomplete
  frame collection, or malformed assessment artifacts;
- one quality-failing operating point when a less aggressive threshold, reuse
  span, layer scope, payload, refresh rule, warmup/tail guard, correction, or
  fallback can plausibly repair quality;
- one speed-negative point when signal calibration, cache hit rate, payload
  boundary, bookkeeping overhead, or a distinct implementation can plausibly
  improve runtime;
- one failed TeaCache, EasyCache, or TaylorSeer implementation.

Record these as `needs_retry`, `needs_rewrite`, `needs_cache_refinement`, or
`infra_blocked`, preserve the evidence, and continue the same family or switch
only with an explicit reason.

### Discard Standard

As the sole executor, you may discard a concrete config or exhausted method
family only when all applicable conditions hold:

- a completed fixed-contract full target-model run and aligned LPIPS/Codex assessment
  exist;
- the negative result is not caused by infrastructure, collection, prompt or
  config mismatch, accidental fallback, or implementation failure;
- there is no meaningful speed or memory improvement, or the observed quality
  loss cannot be repaired without removing the method's benefit;
- you have tested or ruled out the credible cache-level refinement axes for
  that concrete method;
- for a family-level discard, a faithful seed and evidence-driven child were
  evaluated, or concrete source evidence proves structural inapplicability.

Write the full discard checks and evidence paths into `AGENT-STATUS.json` and
`CACHE-SEARCH-STATE.json`. A discarded operating point does not automatically
discard its broader family.

### Completion Standard

Set `AGENT-STATUS.json.status` to `complete` only when:

- all-three-family coverage and adaptive child trials are complete;
- every retained or discarded conclusion is backed by full evaluation;
- at least one shared E2E time target has measured, matched recipes from every
  applicable family;
- the quality winner and exact recipe at each shared target are recorded from
  LPIPS and blind Codex visual evidence, together with the overall Pareto frontier;
- `DELIVERY-DRAFT.json` contains distinct conservative, balanced, and
  aggressive measured points, with visual offsets disclosed rather than
  automatically rejected;
- `CACHE-SEARCH-STATE.json`, `SEARCH_JOURNAL.md`, and `SUMMARY.md` agree;
- no Slurm job or assessment owned by the workflow is still running.

The programmatic `eval_gate` records complete evidence. The separate delivery
gate ends the workflow only after the unified three-point draft validates;
otherwise it resumes this same executor with exact contract errors.
