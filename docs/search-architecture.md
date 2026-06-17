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
                              -> [GPU] eval vs baseline -> low/mid/high tiers

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
3. The subagent runs the bounded dimension loop: observe prior results, propose
   a new hypothesis, implement exactly one candidate, preflight, launch, gate,
   then keep/reject and loop. Compose checks are optional diagnostics.
4. **(GPU stage)** Render a run bundle from the profile + cfg → existing
   `scripts/launch_candidate.py` → `scripts/collect_run.py` → `benchmark.json`/
   `quality.json` plus aligned gate artifacts → compare vs the profile
   `[baseline]` → bin into low/mid/high per `evals/tiers.toml`.
5. Emit the tiered `final_matrix` (the plan's deliverable).

## Map to the videogen-accel plan
- "four/five directions" → the generic `loops/*` dimensions (cache, token-prune,
  sparse-attn, nvfp4, kwl).
- "integration search → low/mid/high" → step 4–5 here (eval + tiering), model as a
  parameter.
- The model-specific wiring the plan implies (which token span, where the
  attention path is) → discovered by subagents in inference code first, then
  optionally normalized by the main agent during integration.

## Status
CPU observation and compose diagnostics are implemented + tested
(`search/test_search.py`). GPU half (eval + tiering) is `plan_eval()` — wires the
existing launcher/collector/eval-profile.

## Quality eval pipeline (per candidate) — what bins a config into a tier

Each search-loop iteration runs a candidate and assesses quality through THREE
authoritative stages; the verdict (plus latency + peak_mem) decides its tier
(`evals/tiers.toml`):

```
1. off_identity      guarded paths must be byte/numeric-identical to baseline when
                     the technique is OFF (the framework invariant).
2. quantitative      PSNR/MSE/mean abs diff over extracted frames, temporal
                     flicker and patch-boundary metrics, plus LPIPS over
                     stratified frame pairs. Missing baseline frames or missing
                     LPIPS blocks promotion.
3. perceptual (REQUIRED)  aligned NVIDIA-Gemini pairwise visual-artifact judge:
                     extract frames -> side_by_side vs baseline ->
                     tools/vision/nvidia_gemini_judge.py (rubric:
                     evals/rubrics/gemini_visual_artifact_gate.md) -> verdict
                     {overall, new_artifacts[severity], recommendation}.
```

Stage 3 is mandatory: LPIPS/PSNR cannot see blur, mosaic, snow, ghosting,
temporal flicker, or degraded text/faces/hands — only the multimodal judge can,
and it is what separates a clean low-tier config from a degraded one. The verdict
maps to a tier by `gemini_overall` + max new-artifact `severity` (see
`evals/tiers.toml`): low = pass & no artifacts; medium = pass & ≤low severity;
high = ≤medium severity (high severity is always rejected).

The combined verdict (all three stages) + the (latency, peak_mem) improvement bin
the candidate into the tightest (cleanest, low-first) tier it satisfies.
Promotion authority is OFF identity, aligned LPIPS on canonical baseline frames,
and aligned pairwise Gemini. Collector `quality.json` is telemetry and may be
used for bookkeeping, but its video-sampled Gemini output is not promotion
authority when it contradicts aligned LPIPS or aligned pairwise gate artifacts.
The source of truth is structured JSON from the authoritative gate, never prose
logs or release notes. Each dimension keeps the best config per tier; integration
stacks per-tier winners. Integration is a mandatory fan-in loop: composed
profiles must be generated and gated themselves, and a tier with no eligible
composition must be recorded as an explicit blocker.

## Loop control and failed candidates

A native goal dimension is not complete when one candidate fails. Failed
candidates must be rejected, logged with a failure signature, and used to choose
the next hypothesis. Successful candidates update `best_per_tier`, then the
dimension keeps searching for a better point. A dimension stops only at:

- `max_iters`;
- `early_stop_patience` with no Pareto improvement or new diagnostic signal;
- a real external blocker;
- structured-negative evidence that covers the meaningful mechanism space;
- explicit main-orchestrator release after review.

See `docs/fanout-loop-contract.md` for the exact state machine and required
`SEARCH_JOURNAL.md` / `AGENT-STATUS.json` fields.

**Live-verified** on HSG: the Gemini judge runs against `inference-api.nvidia.com`
(`gcp/google/gemini-3.5-flash`, `NVIDIA_API_KEY`); a self-vs-self baseline check
returned `overall=pass`, `new_artifacts=[]`, `recommendation=promote`. The judge
helper is stdlib-only (urllib + ffmpeg frame sampling), so it runs anywhere. The
eval stage itself (real candidate runs) is GPU-side — `search.plan_eval` (stub).
