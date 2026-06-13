# Dimension: step_cache — diffusion step caching

A **model-agnostic search dimension**. It searches whole-denoise-step caching
configs and composes them against whatever model the search targets. It names no
model; model specifics live in `models/<id>.toml` + `efficiency/models/<id>_spec.py`.

## What it searches
Two cache flavors from the `efficiency` engine (generic, written once):
- **`efficiency/techniques/step_cache.py` (`StepCache`)** — skip a scheduled set
  of late steps and reuse (or delta-extrapolate) the previous denoiser output.
  Decision is schedule-based (deterministic). Search space: `skip` schedule,
  `delta_scale`.
- **`efficiency/techniques/teacache.py` (`TeaCache`)** — skip while the rescaled
  cumulative rel-L1 distance of the timestep-modulated input stays under a
  threshold. Decision is content-adaptive. Search space: `threshold`,
  `start_step`, `max_continuous_hits`.

Both write the exclusive `STEP_OUTPUT` seam (only one whole-step cache at a time)
and are OFF==byte-identical baseline. The search space + LTX-2.3 seeds are in
`dimension.toml`; provenance/recipe/report in `reference/`.

## Why it's model-agnostic
`StepCache` wraps the whole step, so it needs no structural capability — it
composes against any registered `ModelSpec` (`requires_capabilities = []`). The
search calls `compose([build_technique("step_cache", **cfg)], spec)` for the
target model; nothing here knows whether that model is Cosmos3, LTX-2.3, or the
next one.

One genuine per-model hook, kept OUT of this dimension: **TeaCache needs the
model to stash its timestep-modulated-input signal** under
`("teacache_signal", cache_key)` each step. That is a runtime seam declared/wired
in the model adapter (tracked in each `models/<id>.toml [seam_status].teacache_signal`),
not here. Until a model wires it, TeaCache composes but falls back to full
compute at runtime; StepCache works immediately.

## Migrated LTX-2.3 experience (the search priors)
`reference/recipe.md` — LTX cache env/knobs (from `run_ltx23_sglang_nonhq_cache_10s.sh`,
`run_ltx23_teacache_hq_nonhq_matrix_10s.sh`); `reference/make_cache_report.py` —
the cache-report helper (canonical artifacts); `reference/report.md` — required
report shape. These feed `dimension.toml`'s `[[seeds]]` (e.g. TeaCache c04_s6).

## Independent test
```bash
~/lustre/miniconda3/envs/sana/bin/python loops/step_cache/test_step_cache.py
```
CPU-only; validates the cache techniques through `efficiency` against a model
spec (the registered target and/or a local fixture spec). The search-level check
that this dimension stays model-agnostic lives in `search/test_search.py`.

## Run it in the search
```bash
python search/search.py --model cosmos3   # lists this dimension's composable candidates
```
See `acceptance.md` for promotion gates and `references.md` for provenance.
