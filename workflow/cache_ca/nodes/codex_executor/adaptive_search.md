## Required Adaptive Cache Search

The complete search space is TeaCache, EasyCache, and TaylorSeer. Do not explore
or retain any other cache family, and do not combine the three families into a
hybrid candidate. The search must use faithful adaptive mechanisms rather than
ending with fixed call-index reuse schedules.

Before requesting workflow completion, implement and complete at least one
faithful full-run candidate from each applicable family:

- **TeaCache**: use a cheap model-derived signal and an accumulated, optionally
  rescaled change indicator to decide refresh versus reuse. Search the signal,
  payload boundary, threshold, rescaling/calibration, warmup/tail guards,
  periodic refresh, and maximum consecutive hits.
- **EasyCache**: reuse a transformation vector, residual update, or equivalent
  model-local payload, and adapt refresh online from a runtime error estimate.
  Search the payload location, online error signal, refresh threshold,
  correction rule, layer/block scope, maximum reuse span, and dense fallback.
- **TaylorSeer**: forecast a denoiser, block, residual, or feature trajectory
  from cached history with a Taylor-style update rather than returning a stale
  tensor unchanged. Search forecast location, Taylor order, history length,
  derivative/update rule, error threshold, refresh cadence, prediction span,
  and sensitive-step/layer fallback.

If source inspection proves that one of these families cannot be represented on the
active target-model path, record the exact structural reason and evidence in
`CACHE-SEARCH-STATE.json`. As the sole executor, you may mark that family
inapplicable only from concrete source-path evidence. A failed first
implementation is not proof of inapplicability.

Do not label fixed alternating stale-output reuse as TeaCache, EasyCache, or
TaylorSeer. A candidate must implement the defining decision signal and payload
behavior of the family it claims.

### Evidence-Driven Parameter Adaptation

Do not run a static Cartesian grid and do not choose unrelated parameter points
without reference to prior evidence. Treat every completed full assessment as
feedback for the next point:

- quality passes but speed is below target: cautiously increase cache scope,
  threshold, reuse span, or forecast aggressiveness;
- quality moves beyond the intended tier: tighten the threshold, shorten the
  reuse/forecast span, increase refreshes, reduce Taylor order/correction
  magnitude, or narrow the payload/layer scope; retain the measured point as
  frontier evidence rather than treating visual difference as automatic
  rejection;
- the cache records no hits: repair or recalibrate the signal before judging
  the method;
- hits occur but wall time does not improve: move the payload boundary or
  reduce bookkeeping instead of only changing the threshold;
- infra or assessment failure: rerun the same point after repair rather than
  treating it as parameter evidence.

For every viable family, evaluate a conservative seed point and at least one
evidence-driven child refinement before declaring its parameter space
exhausted. Continue bracketing the quality/speed boundary while credible
refinement remains. Once the feasible ranges are known, adapt parameters toward
shared `time_ratio` targets and compare family quality only at matched E2E
times. The order of families and parameter updates should respond to the
accumulated evidence rather than follow a fixed roadmap.

### Durable Search State

Maintain `CACHE-SEARCH-STATE.json` in the experiment worktree. It must be valid
JSON and contain, at minimum:

```json
{
  "schema_version": 1,
  "allowed_families": ["teacache", "easycache", "taylorseer"],
  "active_family": "teacache | easycache | taylorseer",
  "families": {
    "teacache": {"status": "pending", "trials": []},
    "easycache": {"status": "pending", "trials": []},
    "taylorseer": {"status": "pending", "trials": []}
  },
  "matched_time_targets": [],
  "next_trial": {
    "family": "<family>",
    "parent_candidate_id": "<prior candidate or empty>",
    "parameters": {},
    "reason": "how prior speed, LPIPS, blind Codex visual, hit-rate, and trace evidence selected this point"
  }
}
```

Each trial record must include candidate id, claimed family, method-defining
signal, reuse/forecast payload, complete parameter values, parent candidate,
adaptation reason, run directory, cache statistics, baseline and candidate E2E
times, `time_ratio`, speedup, LPIPS, Codex visual verdict, outcome, target time
budget, and proposed next adjustment. Each `matched_time_targets` entry must
record the target ratio, tolerance, one measured recipe per applicable family,
evidence paths, quality comparison, and winner. Keep `SEARCH_JOURNAL.md` and
`AGENT-STATUS.json` consistent with this state.

Workflow completion is not ready merely because the iteration budget is
consumed. The final state must show all-three-family coverage, adaptive
parent-to-child trials, at least one shared matched-time comparison, and three
distinct conservative/balanced/aggressive points on the resulting Pareto
frontier. At each compute budget select the highest-quality measured recipe,
including documented visual offsets. Missing coverage requires continued
executor work unless the executor records documented structural
inapplicability. Never name an overall winner from unmatched E2E times.
