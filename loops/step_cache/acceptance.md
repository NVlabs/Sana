# Search loop — step_cache

This dimension is a **bounded search loop**, not a one-shot checklist. Follow
`docs/fanout-loop-contract.md`: a candidate failure rejects/logs that candidate
and returns to the loop unless max_iters, a real blocker, or explicit
orchestrator release applies. A structured-negative decision is recorded as a
proposal/failure signature; it does not stop the default fixed-budget loop.

## Loop (see `dimension.toml [loop]`)
- **Granularity: per_step** — the axis searched.
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
- **Cache-family coverage:** before declaring structured negative,
  compare at least five distinct cache families from `search_space/01_cache.md`,
  such as TeaCache-style timestep-aware reuse, EasyCache-style runtime-adaptive
  transform reuse, PAB-style attention broadcast, block/layer feature caching,
  and token-wise, CFG-aware, content-adaptive, or predictive/delta caching when
  applicable.

## Frontier retention and final tier selection
During the 40-iteration search, keep a candidate in the retained frontier if it
improves **quality** or improves **latency/peak memory**. It does not need to pass
a low/medium/high tier at retention time. Discard it if quality does not improve
and speed/memory does not improve or regresses. Hard-invalid candidates, such as
missing artifacts, broken OFF identity, or runtime failure, are rejected with a
failure signature.

After the loop exits, the main agent selects delivery winners from the retained
frontier using `evals/tiers.toml`:
- **low** — best-quality profile at or above 1.5x.
- **medium** — best-quality profile at or above 2.0x.
- **high** — best-quality profile at or above 3.0x.

Quality evidence comes from OFF identity (when applicable), aligned LPIPS on the
canonical baseline frames, and aligned pairwise Gemini. Collector `quality.json`
is telemetry and cannot override the aligned gate during final selection. For
lossy generative caching, LPIPS and Gemini are joint ranking signals, not hard
absolute thresholds; Gemini artifact severity/status is considered with LPIPS,
then speed breaks ties.

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
- Quality does not improve and speed/mem does not improve or regresses.
- State (cache/prune/etc.) leaks across samples or stages.

Every reject must be recorded in `SEARCH_JOURNAL.md` with candidate id, run dir,
gate artifacts, root cause, and the next-hypothesis requirement. Rejection does
not finish the dimension by itself.
