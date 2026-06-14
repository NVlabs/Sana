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
| **LOW**       | 1.35x   | **1.733x**  ✓ HIT | `SGLANG_HQ_STEP_CACHE_SKIP=12-28 SGLANG_HQ_STEP_CACHE_DELTA=0.5` | Gemini `pass`, max-artifact `none`, tier `low` | Unset env -> byte-identical baseline |
| MEDIUM        | 2.20x   | (in progress)     | TBD -- exploring more-aggressive skip + teacache + token_prune | -- | -- |
| HIGH          | 3.00x   | (in progress)     | TBD                                                       | -- | -- |

LOW is delivered: a single dimension (step_cache) already beats the 1.35x low-tier
target with quality clean. The composed-tier targets (1.35 / 2.20 / 3.00 in
`evals/tiers.toml [targets]`) loosen quality budgets toward higher tiers; the
search will only promote a config into MEDIUM/HIGH if it actually hits the
speedup target.

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
