# Search loop — kwl_fusion

This dimension is a **bounded search loop**, not a one-shot checklist. Follow
`docs/fanout-loop-contract.md`: a candidate failure rejects/logs that candidate
and returns to the loop unless max_iters, a real blocker, or explicit
orchestrator release applies. A structured-negative decision is recorded as a
proposal/failure signature; it does not stop the default fixed-budget loop.

## Loop (see `dimension.toml [loop]`)
- **Granularity: per_strategy** — the axis searched.
- **Objective:** beat the model baseline (`models/<id>.toml [baseline]`) on
  **latency OR peak memory** — either improvement counts (Pareto over the two).
- **Budget:** fixed `max_iters = 40`. Default patience early stop is disabled;
  run the full budget unless a real blocker or explicit orchestrator release
  applies.
- **Per iteration:** start from `search_space/`, derive a
  model-specific candidate from traces/code, record it in the candidate manifest,
  then compose against the model spec → (GPU) run → measure latency + peak_mem +
  quality. `dimension.toml` records search axes and loop metadata, not fixed hyperparameter candidates.
- **Next hypothesis:** each iteration must state why it is expected to improve
  over the previous loop or avoid a recorded failure signature. Do not repeat a
  rejected mechanism with cosmetic parameter changes.

## Frontier retention and final tier selection
KWL uses the fixed-budget frontier loop, but its frontier rule is stricter than
lossy dimensions. It is an exact implementation dimension, so speed is never
allowed to buy intentional quality loss.

During the 40-iteration search, keep a candidate in the retained frontier when
one of these is true:

- **Speed/memory retention:** latency or peak memory improves, OFF identity
  passes, and ON quality/numeric evidence does not regress against baseline.
- **Quality/numeric retention:** tensor stability, aligned LPIPS/Gemini, or
  visual output improves, and latency/peak memory does not meaningfully regress.
- **Both improved:** quality/numeric evidence and latency/peak memory improve.

Discard a candidate if OFF identity passes but it has no quality/numeric
improvement and no speed/memory improvement, or if the measured speed/memory path
regresses without a compensating numeric improvement. Reject hard-invalid
candidates, such as missing artifacts, broken OFF identity, runtime failure,
silent fallback, or semantic changes, with a failure signature.

After the loop exits, the main agent selects delivery winners from the retained
frontier using `evals/tiers.toml`:
- **low** — best-quality exact profile at or above 1.5x.
- **medium** — best-quality exact profile at or above 2.0x.
- **high** — best-quality exact profile at or above 3.0x.

Quality evidence comes from OFF identity, module-level tensor diff when
available, aligned LPIPS on the canonical baseline frames, and aligned pairwise
Gemini. Collector `quality.json` is telemetry and cannot override the aligned
gate during final selection. KWL may enforce reliable exact/numeric gates as
hard requirements; LPIPS and Gemini are still recorded and considered for final
cross-profile quality ranking.

## Required KWL preflight
- Record the hot-path evidence for the operator chain being fused: profile,
  trace, code inspection, or kernel timeline.
- Record launch count, memory traffic, dtype, tensor shapes, backend, and
  fallback behavior before and after the candidate.
- Record compile/autotune/CUDA-graph state as cold, warm, cache-reused, or graph
  replay. Do not mix these timing modes in one speedup number.
- Prove OFF identity before reporting any speedup.
- For ON, record expected numerical tolerance: bit-exact, dtype-rounding-only,
  reduction-order drift, or FMA/epilogue drift.

## Keep / output
Keep retained frontier candidates with their quality evidence, speed/memory
evidence, manifest, run artifacts, and improvement axis. These frontier
candidates feed final tier selection, then the **integration stage**, which
stacks dimension winners into final low/medium/high delivery profiles (composed
targets 1.5x / 2.0x / 3.0x in `evals/tiers.toml [targets]`).

`no_improve_count` is telemetry in the default fixed-budget mode; it does not
stop the loop. When the loop exits for budget, write
`status=terminal_pending_review` and include an `agent_recommendation` for the
main orchestrator: select_tiers_for_integration, restart_with_new_direction,
validate, drop, or mark blocked.

## Reject
- OFF path not byte-identical on guarded paths / baseline path altered.
- Scheduler, step count, token set, prompt/guidance, LoRA state, resolution,
  frame count, attention semantics, cache semantics, pruning semantics, or
  precision policy changes.
- Speed/memory improves but aligned quality or numeric evidence regresses.
- Quality/numeric evidence does not improve and speed/mem does not improve or
  regresses.
- Candidate reports speedup from silent backend fallback, skipped work, cold/warm
  timing mismatch, or changed output shape.
- State (cache/prune/etc.) leaks across samples or stages.

Every reject must be recorded in `SEARCH_JOURNAL.md` with candidate id, run dir,
gate artifacts, root cause, and the next-hypothesis requirement. Rejection does
not finish the dimension by itself.

## Structured negative
A structured-negative proposal may be recorded only after evidence covers at
least six exact/lossless method families from `search_space/05_kernel_fusion.md`,
the top remaining hot spots, backend availability, fallback behavior, and the
expected speed ceiling. It does not terminate the default fixed-budget loop by
itself; a single failed fused-kernel candidate is not enough.
