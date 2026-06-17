# Search loop — step_cache

This dimension is a **bounded search loop**, not a one-shot checklist. Follow
`docs/fanout-loop-contract.md`: a candidate failure rejects/logs that candidate
and returns to the loop unless max_iters, early_stop, a real blocker, or
structured-negative evidence applies.

## Loop (see `dimension.toml [loop]`)
- **Granularity: per_step** — the axis searched.
- **Objective:** beat the model baseline (`models/<id>.toml [baseline]`) on
  **latency OR peak memory** — either improvement counts (Pareto over the two).
- **Budget:** `max_iters = 20` (hyperparameter) with **early stop** after 5
  iterations with no Pareto improvement.
- **Per iteration:** start from `search_space/`, derive a
  model-specific candidate from traces/code, record it in the candidate manifest,
  then compose against the model spec → (GPU) run → measure latency + peak_mem +
  quality. `dimension.toml` records search axes and loop metadata, not fixed hyperparameter candidates.
- **Next hypothesis:** each iteration must state why it is expected to improve
  over the previous loop or avoid a recorded failure signature. Do not repeat a
  rejected mechanism with cosmetic parameter changes.

## Acceptance = quality is a hard, PER-TIER constraint
A candidate counts only if it (a) beats baseline on latency or peak_mem **and**
(b) meets a risk tier's quality budget (`evals/tiers.toml`). It is binned into the
**tightest (cleanest, low-first) tier it satisfies**:
- **low** — near-lossless: off==baseline identity for guarded paths; LPIPS Δ ≤ 0.01; no new artifacts.
- **medium** — controlled loss: LPIPS Δ ≤ 0.04; no medium/high artifacts.
- **high** — preview: LPIPS Δ ≤ 0.09; visible-but-described loss OK.

Promotion authority is OFF identity (when applicable), aligned LPIPS on the
canonical baseline frames, and aligned pairwise Gemini. Collector `quality.json`
is telemetry and cannot override the aligned gate.

## Keep / output
Keep the best (latency, peak_mem) config **per tier**. These per-tier winners feed
the **integration stage**, which stacks dimensions into the final low/medium/high
delivery profiles (composed targets ~1.35x / 2.2x / 3.0x+ in `evals/tiers.toml [targets]`).
A successful candidate updates best_per_tier and the loop continues for a better
point until a stop condition or orchestrator release.

## Reject
- OFF path not byte-identical on guarded paths / baseline path altered.
- Improves speed/mem but fails every tier's quality budget.
- State (cache/prune/etc.) leaks across samples or stages.

Every reject must be recorded in `SEARCH_JOURNAL.md` with candidate id, run dir,
gate artifacts, root cause, and the next-hypothesis requirement. Rejection does
not finish the dimension by itself.
