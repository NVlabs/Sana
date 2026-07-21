# Search Architecture — model-agnostic acceleration search

The deliverable is a **fully-automated serving-acceleration search engine**.
Cosmos3-Super is the first target it is applied to, not something it is built
around. Model-specific knowledge is confined to one small adapter per model;
everything else is generic.

## Three planes (strict separation)

```
GENERIC (write once, every model):
  efficiency/                 the acceleration engine — Techniques/Transforms,
                              diagnostics, compose checks, schedules
  loops/<dim>/dimension.toml  a SEARCH DIMENSION: method family + search axes +
                              budget/quality metadata. Never names a model.
  search/search.py            observe dimensions and run optional compose diagnostics
                              -> [GPU] eval vs baseline -> 1.5/2/3x delivery targets

MODEL/RUNTIME:
  models/<id>.toml                 profile: official config, baseline, run entry,
                                   base env
  Sol-LTX-Infer/                   live inference code that subagents inspect/edit
```

Why this shape: goal agents should not wait for a pre-exposed interface before
trying an acceleration idea. The dimensions describe method families and quality
contracts; each subagent reads the live inference code and implements a
model-specific candidate directly. The `efficiency` engine and `ModelSpec`
remain useful for smoke tests and merge diagnostics, but they do not gate launch.

## Launchability Is Main-Agent Policy
`search/search.py` reports method families, loop budgets, and compose diagnostics.
The main agent decides which dimensions to wake. A failed compose diagnostic is
not a reason to skip exploration; it usually means the useful implementation
should happen directly in `Sol-LTX-Infer/` before being normalized.

## Search pipeline
1. Load `models/<id>.toml` and observe available method families.
2. For each dimension the main agent chooses to wake: spawn a native Codex goal
   from `search_space/` plus `loops/<dim>/exploration.md` and
   `docs/fanout-loop-contract.md`.
3. The subagent runs the bounded dimension loop: observe current-experiment
   results, propose a new hypothesis, implement exactly one candidate,
   preflight, launch, gate, then retain/discard/reject and loop. Compose checks
   are optional diagnostics.
4. **(GPU stage)** Render a run bundle from the profile + cfg → existing
   `scripts/launch_candidate.py` → `scripts/collect_run.py` → `benchmark.json`/
   `quality.json` plus aligned gate artifacts → compare vs the profile
   `[baseline]` → bin into 1.5x/2.0x/3.0x speed targets per `evals/tiers.toml`.
5. Emit the delivery `final_matrix` (low/medium/high speed targets).

## Map to the videogen-accel plan
- "four/five directions" → the generic `loops/*` dimensions (cache, token-prune,
  sparse-attn, nvfp4, kwl).
- "integration search → low/mid/high" → step 4–5 here (eval + speed-target
  quality ranking), model as a parameter.
- The model-specific wiring the plan implies (which token span, where the
  attention path is) → discovered by subagents in inference code first, then
  optionally normalized by the main agent during integration.

## Status
CPU observation and compose diagnostics are implemented + tested
(`search/test_search.py`). GPU half (eval + tiering) is `plan_eval()` — wires the
existing launcher/collector/eval-profile.

## Quality eval pipeline (per candidate) — what ranks a config

Each search-loop iteration runs a candidate and assesses quality through THREE
authoritative stages; the verdict (plus latency + peak_mem) decides whether it
is retained during fan-out and how it is ranked after the budget closes
(`evals/tiers.toml`):

```
1. off_identity      guarded paths must be byte/numeric-identical to baseline when
                     the technique is OFF (the framework invariant).
2. quantitative      PSNR/MSE/mean abs diff over extracted frames, temporal
                     flicker and patch-boundary metrics, plus LPIPS over
                     stratified frame pairs. Missing baseline frames or missing
                     LPIPS blocks final selection until evidence is backfilled.
3. perceptual (REQUIRED)  aligned NVIDIA-Gemini pairwise visual-artifact judge:
                     extract frames -> side_by_side vs baseline ->
                     tools/vision/nvidia_gemini_judge.py (rubric:
                     evals/rubrics/gemini_visual_artifact_gate.md) -> verdict
                     {overall, new_artifacts[severity], recommendation}.
```

Stage 3 is mandatory: LPIPS/PSNR cannot see blur, mosaic, snow, ghosting,
temporal flicker, or degraded text/faces/hands — only the multimodal judge can.
Final target selection considers both Gemini and LPIPS: aligned pairwise Gemini
artifact severity/status first, aligned LPIPS second, then higher speed as a
tie-breaker. LPIPS alone is not the selector.

The combined verdict (all three stages) plus speed/memory evidence determines
whether a candidate is retained during fan-out and how it can be selected later.
During the per-dimension loop, quality is not a hard per-tier retention gate:
retain candidates that improve quality or speed/memory, and discard candidates
where quality does not improve and speed/memory does not improve or regresses.
The source of truth is structured JSON from the authoritative gate, never prose
logs or release notes. After the fixed budget closes, the main agent selects the
best-quality retained frontier candidate for each speed target:
low >= 1.5x, medium >= 2.0x, high >= 3.0x.
Integration then stacks selected winners. Integration is a mandatory fan-in loop:
composed profiles must be generated and gated themselves, and a tier with no
eligible composition must be recorded as an explicit blocker.

## Loop control and failed candidates

A native goal dimension is not complete when one candidate fails. Failed
candidates must be rejected or discarded, logged with a failure signature or
reason, and used to choose the next hypothesis. Retained candidates update the
frontier, then the dimension keeps searching for a better point. A dimension
stops only at:

- `max_iters` (default `40` per fan-out dimension);
- a real external blocker;
- explicit main-orchestrator release after review.

A dimension agent may record a structured-negative proposal as a failure
signature, but it cannot terminate the default fixed-budget loop by itself.

`early_stop_patience` defaults to `0`, so patience early stop is disabled in the
fixed-budget frontier mode. `no_improve_count` is telemetry. When budget fires,
the dimension should report `terminal_pending_review` so the main agent can
select low/medium/high winners from the frontier, restart with a new direction,
request validation, drop the dimension, mark a blocker, or integrate selected
winners.

See `docs/fanout-loop-contract.md` for the exact state machine and required
`SEARCH_JOURNAL.md` / `AGENT-STATUS.json` fields.

**Live-verified** on HSG: the Gemini judge runs against `inference-api.nvidia.com`
(`gcp/google/gemini-3.5-flash`, `NVIDIA_API_KEY`); a self-vs-self baseline check
returned `overall=pass`, `new_artifacts=[]`, `recommendation=promote`. The judge
helper is stdlib-only (urllib + ffmpeg frame sampling), so it runs anywhere. The
eval stage itself (real candidate runs) is GPU-side — `search.plan_eval` (stub).
