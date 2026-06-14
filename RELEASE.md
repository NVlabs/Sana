# Cosmos3-Super acceleration -- tier matrix (in progress)

End-to-end serving acceleration of `nvidia/Cosmos3-Super` (Cosmos3, 64B, video
T2V) driven by the model-agnostic search engine in this repo. Each tier is a
composed config (feature flags + env) with a verified speedup and a Gemini
visual-artifact pass against the 1280x720 / 189 frames / 35 step baseline.

## Baseline

| Metric          | Value     | Source                                                      |
|-----------------|-----------|-------------------------------------------------------------|
| Total           | 130.41 s  | `models/cosmos3.toml [baseline]` (canonical)                |
| Denoise         | 121.42 s  | per-stage timing from `runs/20260612-175151-baseline-…/outputs/run.log` |
| Decode          |   5.80 s  | same                                                         |
| OFF==identity   | 0 px diff | `verify_off_identity.py` -- new-runtime OFF run vs the canonical baseline |
| Quality (Gemini)| pass / none | self-vs-self verdict @ runs/20260613-175619-baseline       |

## Tier matrix

| Tier          | Target  | Achieved          | Config (feature flags / env)                            | Verdict                                  | Rollback |
|---------------|---------|-------------------|---------------------------------------------------------|------------------------------------------|----------|
| **LOW**       | 1.35x   | **1.823x**  ✓ HIT | `SGLANG_HQ_STEP_CACHE_SKIP=10-28 SGLANG_HQ_STEP_CACHE_DELTA=0.5` | Gemini `pass`, max-artifact `none`, tier `low` | Unset env -> byte-identical baseline |
| MEDIUM        | 2.20x   | (in progress)     | TBD -- exploring more-aggressive skip + teacache + token_prune | -- | -- |
| HIGH          | 3.00x   | (in progress)     | TBD                                                       | -- | -- |

LOW is delivered: a single dimension (step_cache) already beats the 1.35x low-tier
target with quality clean. The composed-tier targets (1.35 / 2.20 / 3.00 in
`evals/tiers.toml [targets]`) loosen quality budgets toward higher tiers; the
search will only promote a config into MEDIUM/HIGH if it actually hits the
speedup target.

### Search trajectory (step_cache dimension)

| skip      | delta | total_s | speedup | Gemini | max-artifact | tier   |
|-----------|-------|---------|---------|--------|--------------|--------|
| 16-28     | 0.0   | 87.64   | 1.488x  | pass   | none         | low    |
| 16-28     | 0.5   | 85.67   | 1.522x  | pass   | none         | low    |
| 20-28     | 0.0   | 96.25   | 1.355x  | pass   | none         | low    |
| 20-28     | 0.5   | 98.00   | 1.331x  | pass   | none         | low    |
| 12-28     | 0.5   | 75.26   | 1.733x  | pass   | none         | low    |
| 8-28      | 0.5   | 61.61   | 2.117x  | fail   | high         | REJECT |
| **10-28** | **0.5** | **71.53** | **1.823x** | **pass** | **none** | **low (WIN)** |

### Search trajectory (teacache dimension)

| threshold | start_step | max_hits | total_s | speedup | Gemini | tier |
|-----------|------------|----------|---------|---------|--------|------|
| 0.04      | 6          | 1        | 100.41  | 1.299x  | pass   | low  |

TeaCache becomes functional after wiring `teacache_signal` in
`cosmos3video.forward`. The LTX-2.3 prior (c04/s6) at 1.30x is below the
12-28/0.5 step_cache speedup; since both write the exclusive STEP_OUTPUT seam,
step_cache wins as the dimension's representative.

### MEDIUM / HIGH plan

The aggressive 8-28/0.5 result shows the upper bound on single-dimension
step_cache on Cosmos3-Super (the Gemini judge starts flagging high-severity
artifacts somewhere between 17 and 21 skipped of 35 steps). To reach the
MEDIUM (2.20x) and HIGH (3.00x) composed targets, the search needs to stack
step_cache with another dimension that writes a *different* exclusive seam:

- **token_prune** -- writes `TOKEN_SET`. The most plausible next dimension on
  Cosmos3 (LTX-2.3 prior delivers ~1.2-1.4x). Still needs: (a) refine
  `prunable_segment` in `cosmos3_spec.py` to the video-token span, (b) wrap
  `gen_layers` with `plan.before_blocks` / `plan.after_blocks` in
  `cosmos3video.forward`, (c) define `prune_gather`/`prune_scatter` so
  `cos_gen`/`sin_gen` stay aligned with the pruned hidden.
- **sparse_attention** -- writes `ATTENTION_BACKEND`. The seam is declared; the
  LTX-2.3 piecewise kernel is keyed to visual *self*-attention, but Cosmos3's
  GEN pathway is cross-attention to cached UND K/V. The kernel's
  `piecewise_only_video_self_attention=true` flag would make it a no-op on
  Cosmos3 unless the kernel is extended.
- **nvfp4_ffn / kwl_fusion** -- writes `FFN_PRECISION` / `KERNEL_FUSION`. Both
  drive via `SGLANG_LTX2_*` env that the Cosmos3 FFN loader does not currently
  read; would need a Cosmos3-side NVFP4/KWL build hook.

## What is wired vs not (per the model-onboarding playbook)

`models/cosmos3.toml [seam_status]` is the source of truth. Snapshot:

| Seam                | Status          | Unlocks         | Notes                                                  |
|---------------------|-----------------|------------------|--------------------------------------------------------|
| `blocks`            | wired           | (foundation)     | `get_blocks` -> `Cosmos3OmniTransformer.gen_layers`     |
| `prunable_tokens`   | declared        | `token_prune`    | default whole-seq segment; refine to video patch span for real prune |
| `swappable_attention`| declared       | `sparse_attention` | seam exists (USPAttention); component-name + kernel validation needed |
| `step_cache` (whole-step wrap) | wired | `step_cache`     | `Plan.on_step` wraps the per-step noise_pred compute; OFF==identity verified |
| `teacache_signal`   | wired           | `teacache`       | `cosmos3video.forward` stashes `time_embed` under `("teacache_signal", "step_pred")` |
| `ffn_precision`     | transform-env   | `nvfp4_ffn`      | LTX-2-keyed `SGLANG_LTX2_TE_NVFP4_*` env -- needs Cosmos3-side TE NVFP4 loader wiring |
| `kernel_fusion`     | transform-env   | `kwl_fusion`     | LTX-2-keyed kernel fusions; same as above |
| `residual_tuple`    | todo            | residual-cache   | block forward does not yet return a residual-compatible tuple |

## Files / commands (the operator handbook)

- **Search**: `python search/search.py --model cosmos3` -- lists eligible dimensions + per-seam wiring status.
- **Render**: `search.plan_eval.render_candidate(profile, technique, cfg, kind=...)` -> a launcher-valid TOML manifest with the SGLANG_HQ_* env for the technique.
- **Run**: `python scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit`.
- **Assess**: `python search/plan_eval.py --assess <run_dir> --baseline-frames runs/20260613-175619-baseline/outputs/frames`.
- **Tier report**: `python scripts/tier_report.py --model cosmos3 --verdict evals/verdicts/*.json`.

## Verdicts (per-candidate JSONs)

`evals/verdicts/cosmos3__step_cache__*.json` -- one per assessed candidate.

## Audit

`ORCHESTRATOR-LOG.md` -- append-only milestone log for this run.
