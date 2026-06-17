# Acceleration search agent — shared rules

You drive **one** acceleration-search dimension on the Cosmos3-Super model in
your own isolated worktree. Claude (the orchestrator) reviews + gates + merges;
you do not merge into `main`. Read the dimension spec (below this section, after
the `--- dimension.toml ---` separator), `acceptance.md`, and
`docs/fanout-loop-contract.md`, then run the bounded search loop described
there. This is not a one-candidate goal.

## Mental model
- The **acceleration engine** (`efficiency/`) + the **search harness** (`search/`)
  + the **per-dimension specs** (`loops/<dim>/dimension.toml`) are model-agnostic.
- A model plugs in via **`models/<id>.toml`** + **`efficiency/models/<id>_spec.py`**.
  For Cosmos3: profile is `models/cosmos3.toml`, spec is `efficiency.get_model_spec("Cosmos3")`.
- A *dimension* declares: a Technique/Transform (efficiency registry name), a
  parameter search space, required capabilities, and the LTX-2.3 prior seeds.
- A candidate is **rendered** from the model profile + the technique cfg via
  `search.plan_eval.render_candidate(profile, technique, cfg, kind=...)` -> a
  launcher-valid TOML manifest. The render call:
    1. `compose([technique], spec)` -- type-checks the technique against the spec
       (refused if the model doesn't declare a required capability).
    2. For build-transforms, runs `plan.apply_transforms` to extract the
       SGLANG_HQ_* env the kernel/runtime consumes.
    3. For runtime techniques (StepCache, TeaCache, TokenPrune), publishes the
       cfg as SGLANG_HQ_* env via the small table in `search/plan_eval.py`.

## Per-iteration loop
Each iteration must be an evidence-backed attempt to improve on the previous
state, not a blind grid point:

```
0. read prior SEARCH_JOURNAL.md, best_per_tier, rejected failure signatures.
1. write the next hypothesis: what changed, why it should improve, what prior
   failure it avoids, and what evidence would reject it.
2. cfg <- enumerate or derive dimension.toml [search_space] (seed with [[seeds]]
   first, then use traces/code and prior failures to choose the next point).
3. manifest <- render_candidate(load_profile("cosmos3"), technique, cfg, kind=...).
4. write exactly one manifest under candidates/cosmos3__<technique>__<short_cfg>.toml.
5. preflight: static checks, dry-run render, and OFF identity when the dimension
   has an inactive path.
6. python scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit
   (HSG `batch`, 4 GPU/node; the profile [env] supplies cache + python).
7. wait for terminal (sacct State in COMPLETED/FAILED/CANCELLED/TIMEOUT).
8. authoritative assess:
      /home/haozhel/lustre/miniconda3/envs/sana/bin/python search/plan_eval.py \
          --assess <run_dir> \
          --baseline-frames /home/haozhel/lustre/auto-video/runs/20260613-175619-baseline/outputs/frames
   -> speedup, LPIPS, aligned pairwise Gemini, tier.
9. decision:
   - promote: update best_per_tier, record the winner, then continue to step 0.
   - reject: record a failure signature and continue to step 0 with a
     meaningfully different hypothesis.
   - block: record the real blocker and stop.
   - structured_negative: only stop after the evidence covers the meaningful
     mechanism space.
10. early stop after 5 iters with no Pareto improvement or no new diagnostic
    information. max_iters = 20.
```

The older linear recipe is kept here only as the concrete command skeleton:
```
1. cfg <- enumerate dimension.toml [search_space] (seed with [[seeds]] first).
2. manifest <- render_candidate(load_profile("cosmos3"), technique, cfg, kind=...).
3. write manifest under candidates/cosmos3__<technique>__<short_cfg>.toml.
4. python scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit
   (HSG `batch`, 4 GPU/node; the profile [env] supplies cache + python).
5. wait for terminal (sacct State in COMPLETED/FAILED/CANCELLED/TIMEOUT).
6. python scripts/collect_run.py <run_dir>     -> benchmark.json + frames.
7. /home/haozhel/lustre/miniconda3/envs/sana/bin/python search/plan_eval.py --assess <run_dir> --baseline-frames \
        /home/haozhel/lustre/auto-video/runs/20260613-175619-baseline/outputs/frames
   -> {speedup, gemini_overall, max_artifact_severity, tier}.
8. keep best_per_tier (low/medium/high) per evals/tiers.toml.
9. early stop after 5 iters with no Pareto improvement (latency, peak_mem).
   max_iters = 20.
```

## Hard guardrails (the orchestrator will reject merges that violate)
- **OFF == byte-identical baseline** on guarded paths. The framework promises an
  inactive technique is a no-op. If your dimension changes the OFF path, you
  broke an invariant -- fix or back out.
- **Quality is per-tier and aligned-gated**. A candidate that fails OFF identity,
  aligned LPIPS, or aligned pairwise Gemini for its tier (`evals/tiers.toml`) is
  rejected, regardless of speedup. Collector `quality.json` Gemini is telemetry;
  it is not promotion authority when it contradicts the aligned gate.
- **Failure loops back.** A failed candidate gate is a rejection, not goal
  completion. Record the failure signature in `SEARCH_JOURNAL.md` and continue
  unless max_iters, early_stop, structured-negative evidence, or a real blocker
  applies.
- **Success also loops.** A promoted candidate updates `best_per_tier`; it does
  not end the dimension by itself. Keep searching for a better point until a
  stop condition applies or the orchestrator releases you.
- **No cosmetic repeats.** The next candidate after a reject must address the
  recorded root cause. Do not resubmit the same mechanism/window/density after
  an orchestrator cancellation unless the orchestrator explicitly says the
  cancellation was accidental.
- **WARMUP before quoting timings.** SGLang's `--warmup` is off by default in
  the run script; if you need warmed timings, run an extra prompt first or use
  the existing `benchmark.json` denoise_s (which already excludes loader time).
- **No model identity in the dimension or the engine.** All Cosmos3-specific
  knowledge stays in `models/cosmos3.toml` + `efficiency/models/cosmos3_spec.py`
  + (when wiring a model-side seam) the submodule `Sol-LTX-Infer/`. Never edit
  `loops/<dim>/` to reference a model.
- **Bounded loops.** max_iters=20, early_stop_patience=5; log any truncation and
  distinguish `early_stopped_structured_negative`, `blocked`, and `complete`.
- **No editing other dimensions or the engine in your worktree.** If your
  dimension needs a new framework primitive, surface it in `SUMMARY.md` and let
  the orchestrator handle the engine edit on `main`.

## Finish
Write `SUMMARY.md` with: per-tier winners (cfg + run_dir + tier + speedup), the
search trajectory (iters used + early-stop reason if any), rejected candidates
with failure signatures, and any framework/wiring issues you hit so the
orchestrator can resolve them before integration. Only commit/push when the
orchestrator explicitly asks; otherwise leave the worktree inspectable.

---
