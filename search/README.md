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
- `search.py` gives the main agent a CPU-only diagnostic view. It can still run
  compose checks, but those checks do not gate subagent launch; subagents inspect
  and edit inference code directly.

## Run
```bash
# CPU (needs torch): enumerate eligible dimensions + composable candidate space
~/lustre/miniconda3/envs/sana/bin/python search/search.py --model cosmos3
~/lustre/miniconda3/envs/sana/bin/python search/test_search.py
```

## Pipeline (this skeleton = the CPU half)
```
load model profile + ModelSpec
  -> for each method family the main agent decides to wake:
       start from search_space + loops/<dim>/exploration.md
       run the bounded fan-out loop:
         observe prior results -> hypothesize -> implement one candidate
         -> preflight -> launch -> authoritative gate -> keep/reject and loop
       compose([technique(cfg)], spec)        # optional diagnostic, not the driver
  -> [GPU stage, stubbed — plan_eval()]:
       render run bundle from profile + cfg -> scripts/launch_candidate.py
       collect benchmark.json/quality.json -> authoritative aligned assess
       -> compare vs profile baseline
       bin into low/mid/high tiers per evals/profiles/official_video_t2v.toml
```
The enumerate+compose half runs here and is tested. The eval+tiering half is the
GPU stage (`plan_eval()` stub) — it reuses the existing launcher/collector + eval
profile, so the search produces the plan's `final_matrix` of tiered configs.

Candidate failure rejects/logs that candidate and returns to the loop; candidate
success updates best_per_tier and also returns to the loop. Stop only at
max_iters, early_stop, real blocker, structured-negative evidence, or explicit
orchestrator release. See `docs/fanout-loop-contract.md`.

See `docs/search-architecture.md` for the full design.
