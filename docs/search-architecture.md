# Search Architecture — model-agnostic acceleration search

The deliverable is a **fully-automated serving-acceleration search engine**.
Cosmos3-Super is the first target it is applied to, not something it is built
around. Model-specific knowledge is confined to one small adapter per model;
everything else is generic.

## Three planes (strict separation)

```
GENERIC (write once, every model):
  efficiency/                 the acceleration engine — Techniques/Transforms,
                              ModelSpec contract, compose() type-check, schedules
  loops/<dim>/dimension.toml  a SEARCH DIMENSION: technique + param search space +
                              required capability. Never names a model.
  search/search.py            enumerate (model x dimensions x configs) -> compose
                              -> [GPU] eval vs baseline -> low/mid/high tiers

MODEL-SPECIFIC (one small adapter per served model):
  efficiency/models/<id>_spec.py   ModelSpec: the seams this model exposes (~50 LOC)
  models/<id>.toml                 profile: official config, baseline, run entry,
                                   base env, per-seam wiring status
```

Why this shape: the `efficiency` engine already separates a generic Technique
from a tiny per-model `ModelSpec` (the engine type-checks the former against the
latter). The search just lifts that one level up — dimensions are generic, the
model is a parameter. Adding a served model = write a spec + a profile; the
search auto-applies every dimension whose required seam the model declares.

## Eligibility is automatic
`compose([technique(cfg)], spec)` raises `CompositionError` when the model does
not declare the seam a technique needs. The search catches that and skips the
dimension for that model — surfaced in the search plan. So:
- a model that declares `SWAPPABLE_ATTENTION` gets the sparse-attention dimension;
- one that doesn't, automatically doesn't — no per-model edit to the dimension.

## Search pipeline
1. Load `models/<id>.toml` + its `ModelSpec`.
2. For each `loops/<dim>/dimension.toml` whose `requires_capabilities` ⊆ the
   spec's capabilities: enumerate the `search_space` (the migrated LTX-2.3
   proven configs are the `seeds`/priors).
3. Compose each candidate against the spec (auto-reject incompatible).
4. **(GPU stage)** Render a run bundle from the profile + cfg → existing
   `scripts/launch_candidate.py` → `scripts/collect_run.py` → `benchmark.json`/
   `quality.json` → compare vs the profile `[baseline]` → bin into low/mid/high
   per `evals/profiles/official_video_t2v.toml`.
5. Emit the tiered `final_matrix` (the plan's deliverable).

## Map to the videogen-accel plan
- "four/five directions" → the generic `loops/*` dimensions (cache, token-prune,
  sparse-attn, nvfp4, kwl).
- "integration search → low/mid/high" → step 4–5 here (eval + tiering), model as a
  parameter.
- The model-specific wiring the plan implies (which token span, where the
  attention seam is) → the per-model `ModelSpec`, the only place it lives.

## Status
CPU half (enumerate + compose + eligibility) is implemented + tested
(`search/test_search.py`). GPU half (eval + tiering) is `plan_eval()` — wires the
existing launcher/collector/eval-profile. Per-model seam wiring tracked in each
`models/<id>.toml [seam_status]`.

## Quality eval pipeline (per candidate) — what bins a config into a tier

Each search-loop iteration runs a candidate and assesses quality through THREE
stages; the verdict (plus latency + peak_mem) decides its tier (`evals/tiers.toml`):

```
1. off_identity      guarded paths must be byte/numeric-identical to baseline when
                     the technique is OFF (the framework invariant).
2. quantitative      LPIPS (frame + clip) vs baseline; frame metrics; PSNR diagnostic
                     only (bf16 chaos floor -> never a hard gate).
3. perceptual (REQUIRED)  NVIDIA-Gemini pairwise visual-artifact judge:
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
the candidate into the loosest tier it satisfies. Each dimension keeps the best
config per tier; integration stacks per-tier winners.

**Live-verified** on HSG: the Gemini judge runs against `inference-api.nvidia.com`
(`gcp/google/gemini-3.5-flash`, `NVIDIA_API_KEY`); a self-vs-self baseline check
returned `overall=pass`, `new_artifacts=[]`, `recommendation=promote`. The judge
helper is stdlib-only (urllib + ffmpeg frame sampling), so it runs anywhere. The
eval stage itself (real candidate runs) is GPU-side — `search.plan_eval` (stub).
