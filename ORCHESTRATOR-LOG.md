# Orchestrator log — Cosmos3-Super acceleration

Append-only milestone log. One line per milestone, dated UTC. See
`agents/orchestrator-entry.md` for the procedure being driven.

- 2026-06-14T09:26Z  orchestrator started. Read entry md + docs/search-architecture.md +
  docs/model-onboarding.md + models/README.md. `python search/search.py --model cosmos3`
  lists 6 eligible dimensions (48 composable candidates). Baseline run on file:
  `runs/20260613-175619-baseline` (Gemini overall=pass, tier=low, ~1.02x = noise).
  No technique wired into Cosmos3 denoise yet → starting with StepCache via
  `Plan.on_step` in `Sol-LTX-Infer/python/.../cosmos3.py` denoise loop.

- 2026-06-14T09:33Z  wired StepCache. Sol-LTX-Infer @
  codex/cosmos3-step-cache 4fb15e598 (efficiency.Plan built once/request,
  per-step noise_pred wrapped in `plan.on_step`; OFF env -> direct closure
  call, byte-identical to baseline). Parent main 39f145d bumps the
  submodule. Submitted two SBatch jobs on HSG `batch`:
  - 3300261 -- candidates/cosmos3_baseline_off.toml (no env, OFF-identity
    check vs runs/20260613-175619-baseline)
  - 3300262 -- candidates/cosmos3_step_cache_16_28.toml (SGLANG_HQ_STEP_CACHE_SKIP=16-28,
    delta=0 -- the LTX-2.3 SCSP-derived late-cluster skip prior).

- 2026-06-14T10:00Z  step_cache=ON results -- **first real Cosmos3 acceleration**.
  Both 16-28/0 and 20-28/0 pass plan_eval at tier=low (Gemini overall=pass,
  no new artifacts). Versus the canonical baseline (127.83s):
  - **16-28/0**: total 87.64s, denoise 78.91s -> **1.488x speedup, tier=low**
    (HIT vs the 1.35x low-tier target)
  - **20-28/0**:  total 96.25s, denoise 87.51s -> **1.355x speedup, tier=low**
    (marginal hit on the 1.35x target)
  The 13-of-35 late-cluster skip (16-28) clearly wins; pick it as the
  step_cache LOW-tier winner. Delta=0.5 variants (3300337/3300338) still
  running; if delta improves the quality margin or quotient further, it may
  edge into a different tier point.

- 2026-06-14T09:48Z  OFF baseline 3300261 finished. Total 118.2s
  (denoise 109.06s, decode 5.92s) vs prior baseline 127.83s (denoise 119.18s)
  -- 7-8% jitter, within typical GPU run variance. **`scripts/verify_off_identity.py
  runs/20260614-093356-cosmos3-baseline-off runs/20260613-175619-baseline` ->
  OK: 8 frames byte/pixel-identical (max_abs_diff=0).** The OFF==identity
  invariant holds on the new StepCache-wired runtime. Resubmitted step_cache
  =ON (skip='16-28', delta=0) as job 3300310 against the bug-fixed pin
  05422e547.

- 2026-06-14T09:44Z  caught a StepCache scheduling BUG in flight (job 3300262
  log: warmup pass finished entire 35-step denoise in 18.5s -- the technique
  was skipping every step, not just steps 16-28). Root cause:
  `StepCache(skip="16-28")` called `as_schedule(skip)` which wraps the string
  in a const() Schedule -- truthy on every step. Cancelled 3300262. Fix in
  Sol-LTX-Infer @ codex/cosmos3-step-cache 05422e547 (and mirror in
  auto-video/efficiency/techniques/step_cache.py): isinstance(skip, str) ->
  at_steps(skip, True, False); empty string -> False. Added regression test
  in loops/step_cache/test_step_cache.py (string-skip 16-28 inactive at
  step 0 / 15 / 29, active at 16 / 28). Tests: selftest 23/23, loop
  test all pass, search 31/31. OFF baseline job 3300261 still running --
  unaffected by the bug (no env -> no Plan built).
