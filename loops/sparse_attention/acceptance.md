# Search loop — sparse_attention

This dimension is a **bounded search loop**, not a one-shot checklist.

## Loop (see `dimension.toml [loop]`)
- **Granularity: per_module** — the axis searched.
- **Objective:** beat the model baseline (`models/<id>.toml [baseline]`) on
  **latency OR peak memory** — either improvement counts (Pareto over the two).
- **Budget:** `max_iters = 20` (hyperparameter) with **early stop** after 5
  iterations with no Pareto improvement.
- **Per iteration:** pick a config from `dimension.toml [search_space]` (seeded by
  the LTX-2.3 priors) → compose against the model spec → (GPU) run → measure
  latency + peak_mem + quality.

## Acceptance = quality is a hard, PER-TIER constraint
A candidate counts only if it (a) beats baseline on latency or peak_mem **and**
(b) meets a risk tier's quality budget (`evals/tiers.toml`). It is binned into the
**tightest (cleanest, low-first) tier it satisfies**:
- **low** — near-lossless: off==baseline identity for guarded paths; LPIPS Δ ≤ 0.01; no new artifacts.
- **medium** — controlled loss: LPIPS Δ ≤ 0.04; no medium/high artifacts.
- **high** — preview: LPIPS Δ ≤ 0.09; visible-but-described loss OK.

## Keep / output
Keep the best (latency, peak_mem) config **per tier**. These per-tier winners feed
the **integration stage**, which stacks dimensions into the final low/medium/high
delivery profiles (composed targets ~1.35x / 2.2x / 3.0x+ in `evals/tiers.toml [targets]`).

## Reject
- OFF path not byte-identical on guarded paths / baseline path altered.
- Improves speed/mem but fails every tier's quality budget.
- State (cache/prune/etc.) leaks across samples or stages.
