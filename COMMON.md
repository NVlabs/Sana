# Acceleration search agent — shared rules

You drive **one** acceleration-search dimension on the target model selected by
`models/<id>.toml` in your own isolated worktree. The orchestrator reviews,
gates, and merges; you do not merge into `main`. Read the dimension spec (below
this section, after the `--- dimension.toml ---` separator), `acceptance.md`, and
`docs/fanout-loop-contract.md`, then run the fixed-budget frontier loop described
there. This is not a one-candidate goal.

## Mental model
- The **acceleration engine** (`efficiency/`) + the **search harness** (`search/`)
  + the **per-dimension specs** (`loops/<dim>/dimension.toml`) are model-agnostic.
- A model plugs in via **`models/<id>.toml`** plus the live inference/runtime
  code under `Sol-LTX-Infer/`; candidate manifests declare the capabilities
  they require.
- A *dimension* declares method families, search axes, required capabilities,
  and loop metadata. It is not a fixed candidate grid.
- A candidate is **rendered** from the model profile + the technique cfg via
  `search.plan_eval.render_candidate(profile, technique, cfg, kind=...)` -> a
  launcher-valid TOML manifest. The render call:
    1. `compose([technique], spec)` -- type-checks the technique against the spec
       (refused if the model doesn't declare a required capability).
    2. For build-transforms, runs `plan.apply_transforms` to extract the
       SGLANG_HQ_* env the kernel/runtime consumes.
3. For runtime techniques such as StepCache, TeaCache, and TokenPrune, publishes the
       cfg as SGLANG_HQ_* env via the small table in `search/plan_eval.py`.

## Per-iteration loop
Each iteration must be an evidence-backed attempt to improve on the previous
state, not a blind grid point:

```
0. read this goal's current-experiment SEARCH_JOURNAL.md,
   frontier_candidates, discarded/rejected failure signatures.
1. write the next hypothesis: what changed, why it should improve, what prior
   failure it avoids, and what evidence would reject it.
2. cfg <- derive from search_space/ plus target-model traces/code and
   current-experiment failures; do not rely on a fixed grid or old experiment
   reports.
3. manifest <- render or write exactly one runnable candidate manifest.
4. write exactly one manifest under candidates/<model_id>__<technique>__<short_cfg>.toml.
5. preflight: static checks, dry-run render, and OFF identity when the dimension
   has an inactive path.
6. python scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit
   (HSG `batch`, 4 GPU/node; the profile [env] supplies cache + python).
7. wait for terminal (sacct State in COMPLETED/FAILED/CANCELLED/TIMEOUT).
8. authoritative assess:
      /lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/hunyuanvideo15/bin/python search/plan_eval.py \
          --assess <run_dir> \
          --baseline-frames /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/agent_deploy/Sol-LTX-Infer/runs/20260613-175619-baseline/outputs/frames
   -> speedup, LPIPS, aligned pairwise Gemini, speed-target bucket, and
      quality-ranking evidence.
9. decision:
   - retain: if quality improves OR speed/memory improves, append the candidate
     to frontier_candidates, record the improvement axis, then continue to step 0.
   - discard: if quality does not improve and speed/memory does not improve or
     regresses, record the reason and continue to step 0.
   - reject: for hard-invalid candidates, record a failure signature and continue to step 0 with a
     meaningfully different hypothesis.
   - block: record the real blocker and stop.
   - structured_negative: record it as a proposal/failure signature and continue
     the default fixed-budget loop unless the orchestrator explicitly releases
     the dimension.
10. max_iters = 40. Default early_stop_patience = 0, so patience early stop is
    disabled in fixed-budget frontier mode.
```

Concrete command skeleton:
```
1. cfg <- derive from search_space/ plus traces/code and current-experiment
   failures.
2. manifest <- render_candidate(load_profile("<model_id>"), technique, cfg, kind=...)
   when the renderer is applicable, or write a direct candidate manifest.
3. write manifest under candidates/<model_id>__<technique>__<short_cfg>.toml.
4. python scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit
   (HSG `batch`, 4 GPU/node; the profile [env] supplies cache + python).
5. wait for terminal (sacct State in COMPLETED/FAILED/CANCELLED/TIMEOUT).
6. python scripts/collect_run.py <run_dir>     -> benchmark.json + frames.
7. /lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/hunyuanvideo15/bin/python search/plan_eval.py --assess <run_dir> --baseline-frames \
        /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/agent_deploy/Sol-LTX-Infer/runs/20260613-175619-baseline/outputs/frames
   -> {speedup, gemini_overall, max_artifact_severity, speed-target bucket}.
8. retain/discard/reject through tools/symposium/loop_control.py.
9. after max_iters, the main agent selects low/medium/high winners from retained
   frontier candidates per evals/tiers.toml: 1.5x, 2.0x, and 3.0x speed targets,
   ranked by aligned pairwise Gemini severity/status plus aligned LPIPS.
```

## Hard guardrails (the orchestrator will reject merges that violate)
- **OFF == byte-identical baseline** on guarded paths. The framework promises an
  inactive technique is a no-op. If your dimension changes the OFF path, you
  broke an invariant -- fix or back out.
- **Quality evidence is aligned-gated and jointly ranked**. During fan-out,
  quality is not a hard per-tier retention gate: retain a candidate if quality
  improves or speed/memory improves. Final low/medium/high selection uses
  `evals/tiers.toml` speed targets: low=1.5x, medium=2.0x, high=3.0x. Within a
  target, rank quality using aligned pairwise Gemini severity/status and aligned
  LPIPS together; LPIPS alone is not the selector. Collector `quality.json`
  Gemini is telemetry; it is not the quality source of truth when it contradicts
  the aligned gate.
- **Failure loops back.** A failed candidate gate is a rejection, not goal
  completion. Record the failure signature in `SEARCH_JOURNAL.md` and continue
  unless max_iters, a real blocker, or explicit orchestrator release applies. A
  structured-negative decision is logged as a proposal/failure signature and
  does not stop the default fixed-budget loop by itself.
- **Success also loops.** A retained candidate updates `frontier_candidates`; it
  does not end the dimension by itself. Keep searching for a better point until a
  stop condition applies or the orchestrator releases you.
- **No cosmetic repeats.** The next candidate after a reject must address the
  recorded root cause. Do not resubmit the same mechanism/window/density after
  an orchestrator cancellation unless the orchestrator explicitly says the
  cancellation was accidental.
- **WARMUP before quoting timings.** SGLang's `--warmup` is off by default in
  the run script; if you need warmed timings, run an extra prompt first or use
  the existing `benchmark.json` denoise_s (which already excludes loader time).
- **No model identity in the dimension or the engine.** Model-specific
  knowledge stays in `models/<id>.toml`, candidate manifests, and the runtime
  code under `Sol-LTX-Infer/`. Never edit `loops/<dim>/` to reference a model.
- **Bounded loops.** max_iters=40, early_stop_patience=0; log any truncation and
  distinguish `terminal_pending_review`, `structured_negative`, `blocked`, and
  `complete`. Use `python3 tools/symposium/loop_control.py status-summary`
  for watcher logic; do not grep only for `status=complete`.
- Record candidate `purpose` explicitly when it is not a normal fan-out
  frontier point. Integration delivery profiles use `--purpose delivery`;
  upper-bound/high-blocker probes use `--purpose blocker_probe` or
  `--purpose unsafe_probe`; controls use `--purpose control`.
- Before final workflow completion, run `python3 tools/fanout_audit.py --run
  <fanout_run_id_or_path>` and fix any errors it reports.
- **No editing other dimensions or the engine in your worktree.** If your
  dimension needs a new framework primitive, surface it in `SUMMARY.md` and let
  the orchestrator handle the engine edit on `main`.

## Finish
Write `SUMMARY.md` with: retained frontier candidates (cfg + run_dir + quality
evidence + speedup + improvement axis), discarded/rejected candidates with
reasons/signatures, the search trajectory (iters used + terminal reason if any),
and any framework/wiring issues you hit so the orchestrator can select tiers or
resolve them before integration. Only commit/push when the orchestrator
explicitly asks; otherwise leave the worktree inspectable.

---
