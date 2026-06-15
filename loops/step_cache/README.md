# Dimension: step_cache — diffusion step caching

A search dimension for whole-denoise-step and internal feature caching. Native
subagents should read `search_space/01_cache.md`, then inspect and modify the
Cosmos3 inference path directly in their isolated worktree.

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

Current exploration starts from `search_space/` plus this loop's
`exploration.md`. The values in `dimension.toml` are metadata and search axes,
not a fixed grid.

## Exploration Mode

Do not wait for a predeclared seam or adapter hook. If a useful signal or cache
site exists inside the inference code, implement the candidate there, prove the
OFF path is baseline-identical, and document the discovered per-step/per-layer
policy. Main-agent integration can later normalize or wrap the implementation.

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
See `acceptance.md` for promotion gates.
