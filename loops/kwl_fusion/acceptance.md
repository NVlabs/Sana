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
KWL uses the same fixed-budget frontier rule as the other open-ended
dimensions. It is no longer restricted to bit-exact or lossless-only
implementations. Bit-exact candidates are preferred when they are competitive,
but quality-gated non-bit-exact kernel/operator paths are valid candidates when
they preserve the KWL semantic boundary and record aligned quality evidence.

During the 40-iteration search, keep a candidate in the retained frontier when
one of these is true:

- **Speed/memory retention:** module-level or DiT-block-level warm paired
  microbench latency or peak memory improves, OFF identity passes, and tensor
  drift is within the declared tolerance. OFF and ON must be timed in the same
  process/allocation/GPU with the same tensors and warmed cache state. ON may be
  non-bit-exact; record the declared tolerance class and final aligned quality
  evidence only after microbench promotion.
- **Quality retention:** aligned LPIPS/Gemini, visual output, or reliable
  numeric stability improves, even when latency/peak memory does not improve.
- **Both improved:** quality/numeric evidence and latency/peak memory improve.

Discard a candidate if OFF identity passes but it has no quality/numeric
improvement and no speed/memory improvement. Reject hard-invalid candidates,
such as missing artifacts, broken OFF identity, runtime failure, silent
fallback, or semantic changes, with a failure signature.

After the loop exits, the main agent selects delivery winners from the retained
frontier using `evals/tiers.toml`:
- **low** — best-quality retained profile at or above 1.5x.
- **medium** — best-quality retained profile at or above 2.0x.
- **high** — best-quality retained profile at or above 3.0x.

Quality evidence comes first from OFF identity, module-level tensor diff,
microbench numeric tolerance, and warm paired DiT/module microbench latency.
Aligned LPIPS on the canonical baseline frames and aligned pairwise Gemini are
final full-denoise validation only after the microbench passes. Collector
`quality.json` is telemetry and cannot override the aligned gate during final
selection.

## Required KWL preflight
- Record the hot-path evidence for the operator chain being fused: profile,
  trace, code inspection, or kernel timeline.
- Record launch count, memory traffic, dtype, tensor shapes, kernel path, and
  fallback behavior before and after the candidate.
- Write a module-level or DiT-block-level warm paired microbenchmark before any
  full denoising/video run. Record tensor constructors, shape/dtype, warmup
  count, timed iterations, OFF/ON ordering, median/p25/p75/min/max latency,
  baseline latency, candidate latency, max/mean diff, peak memory when relevant,
  launch/profile evidence, expected full contribution, and a durable JSON
  result.
- For lossless candidates, record the equivalence argument: same inputs,
  parameters, masks, shape contract, dtype contract, dependency order, output
  placement, and aliasing/in-place behavior.
- Record compile/autotune/CUDA-graph state only after local module/kernel
  candidates are exhausted. Label cold, warm, cache-reused, or graph replay
  timing separately. Do not mix these timing modes in one speedup number.
- Prove OFF identity before reporting any speedup.
- For ON, record expected numerical tolerance: bit-exact, dtype-rounding-only,
  reduction-order drift, FMA/epilogue drift, fast-math drift, or
  approximate-kernel drift.

## Prohibited Startup Paths
Do not implement, resume, or rerun backend-selection, SDPA-backend,
FlashAttention/FlashInfer dispatch, framework dispatch, or env-flag-only
probes. They do not count as KWL candidates for this dimension. A candidate must
change a fused operator, module-local kernel, layout/copy/allocation path,
custom epilogue, or DiT fusion boundary and must be proven by microbench before
full denoising.
If prior local status or journal files contain backend-selection work, mark it
stale/cancelled and start a new module/DiT microbench candidate.

## Full Denoise Policy
A full denoising/video run is allowed only after a candidate passes warm paired
module/DiT latency/numeric gates. Full denoising is a visual sanity check and
gross regression guard, not the primary speed authority for sub-percent KWL
candidates. Do not claim or reject small speedups from a single candidate run
against a historical canonical baseline; use the warm paired DiT/module median
and expected full-contribution estimate. If the full run shows visual artifacts,
first debug the kernel/module implementation for bugs: aliasing, in-place
mutation, layout, split/concat restoration, masks, dtype promotion, stream
ordering, or stale workspace. Do not treat severe visual drift as harmless
accumulated numeric noise without module-level evidence.

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
- Full denoising run launched before a passing microbench.
- Backend-selection, SDPA-backend, framework-dispatch, or env-flag-only probe.
- Scheduler, step count, token set, prompt/guidance, LoRA state, resolution,
  frame count, attention semantics, cache semantics, pruning semantics, or
  unrelated precision/quantization policy changes that belong in another
  dimension.
- Quality/numeric evidence does not improve and speed/mem does not improve.
- Candidate reports speedup from silent backend fallback, skipped work,
  cold/warm timing mismatch, unpaired historical-baseline full-run timing, or
  changed output shape.
- State (cache/prune/etc.) leaks across samples or stages.

Every reject must be recorded in `SEARCH_JOURNAL.md` with candidate id, run dir,
gate artifacts, root cause, and the next-hypothesis requirement. Rejection does
not finish the dimension by itself.

## Structured negative
A structured-negative proposal may be recorded only after evidence covers at
least seven KWL method families from `search_space/05_kernel_fusion.md`,
including exact-preferred and quality-gated approximate variants where relevant,
the top remaining hot spots, kernel implementation availability, fallback
behavior, and the expected speed ceiling. It does not terminate the default
fixed-budget loop by itself; a single failed fused-kernel candidate is not
enough.
