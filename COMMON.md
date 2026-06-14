# Acceleration search agent — shared rules

You drive **one** acceleration-search dimension on the Cosmos3-Super model in
your own isolated worktree. Claude (the orchestrator) reviews + gates + merges;
you do not merge into `main`. Read the dimension spec (below this section, after
the `--- dimension.toml ---` separator) and `acceptance.md`, then run the bounded
search loop described there.

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

## Per-iteration recipe
```
1. cfg <- enumerate dimension.toml [search_space] (seed with [[seeds]] first).
2. manifest <- render_candidate(load_profile("cosmos3"), technique, cfg, kind=...).
3. write manifest under candidates/cosmos3__<technique>__<short_cfg>.toml.
4. python scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit
   (HSG `batch`, 4 GPU/node; the profile [env] supplies cache + python).
5. wait for terminal (sacct State in COMPLETED/FAILED/CANCELLED/TIMEOUT).
6. python scripts/collect_run.py <run_dir>     -> benchmark.json + frames.
7. python search/plan_eval.py --assess <run_dir> --baseline-frames \
        runs/20260613-175619-baseline/outputs/frames
   -> {speedup, gemini_overall, max_artifact_severity, tier}.
8. keep best_per_tier (low/medium/high) per evals/tiers.toml.
9. early stop after 5 iters with no Pareto improvement (latency, peak_mem).
   max_iters = 20.
```

## Hard guardrails (the orchestrator will reject merges that violate)
- **OFF == byte-identical baseline** on guarded paths. The framework promises an
  inactive technique is a no-op. If your dimension changes the OFF path, you
  broke an invariant -- fix or back out.
- **Quality is per-tier**. A candidate that fails its tier's Gemini verdict
  (`evals/tiers.toml`) is rejected, regardless of speedup.
- **WARMUP before quoting timings.** SGLang's `--warmup` is off by default in
  the run script; if you need warmed timings, run an extra prompt first or use
  the existing `benchmark.json` denoise_s (which already excludes loader time).
- **No model identity in the dimension or the engine.** All Cosmos3-specific
  knowledge stays in `models/cosmos3.toml` + `efficiency/models/cosmos3_spec.py`
  + (when wiring a model-side seam) the submodule `Sol-LTX-Infer/`. Never edit
  `loops/<dim>/` to reference a model.
- **Bounded loops.** max_iters=20, early_stop_patience=5; log any truncation.
- **No editing other dimensions or the engine in your worktree.** If your
  dimension needs a new framework primitive, surface it in `SUMMARY.md` and let
  the orchestrator handle the engine edit on `main`.

## Finish
Commit to your branch (`codex/<dim>`), push to origin, and write `SUMMARY.md`
with: per-tier winners (cfg + run_dir + tier + speedup), the search trajectory
(iters used + early-stop reason if any), and any framework/wiring issues you
hit so the orchestrator can resolve them before integration.

---
