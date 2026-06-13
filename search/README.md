# `search/` — the model-agnostic acceleration search

The deliverable: a **fully-automated serving-acceleration search**. Given a served
model, it searches the acceleration-config space across all dimensions and returns
risk-tiered (low/mid/high) configs — without any dimension knowing which model it is.

## How it stays model-agnostic
- **Dimensions** (`loops/<dim>/dimension.toml`) declare a technique + a param
  search space + the capability a model must expose to be eligible. They never
  name a model.
- **Model** (`models/<id>.toml` + `efficiency/models/<id>_spec.py`) is the only
  model-specific surface.
- `search.py` composes every candidate against the model's `ModelSpec`;
  `compose()` type-checks it, so dimensions the model hasn't wired are
  auto-skipped. Swap `--model`, the eligible set changes automatically.

## Run
```bash
# CPU (needs torch): enumerate eligible dimensions + composable candidate space
~/lustre/miniconda3/envs/sana/bin/python search/search.py --model cosmos3
~/lustre/miniconda3/envs/sana/bin/python search/test_search.py
```

## Pipeline (this skeleton = the CPU half)
```
load model profile + ModelSpec
  -> for each dimension eligible for this model's seams:
       enumerate search_space (seeds from migrated LTX-2.3 recipes as priors)
       compose([technique(cfg)], spec)        # framework rejects incompatible
  -> [GPU stage, stubbed — plan_eval()]:
       render run bundle from profile + cfg -> scripts/launch_candidate.py
       collect benchmark.json/quality.json -> compare vs profile baseline
       bin into low/mid/high tiers per evals/profiles/official_video_t2v.toml
```
The enumerate+compose half runs here and is tested. The eval+tiering half is the
GPU stage (`plan_eval()` stub) — it reuses the existing launcher/collector + eval
profile, so the search produces the plan's `final_matrix` of tiered configs.

See `docs/search-architecture.md` for the full design.
