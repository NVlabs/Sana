# `search/` — the model-agnostic acceleration search

The deliverable: a **fully-automated serving-acceleration search**. Given a served
model, it searches the acceleration-config space across all dimensions and returns
speed-targeted (low/mid/high) configs — without any dimension knowing which model it is.

## How it stays model-agnostic
- **Dimensions** (`loops/<dim>/dimension.toml`) declare techniques, search axes,
  and optional `[[method_baseline]]` entries that classify method-family starting
  points as `wired`, `config_wired`, `runtime_patch`, or
  `upper_bound_probe`. They never name a model.
- **Model** (`models/<id>.toml` + runtime code under `Sol-LTX-Infer/`) is the
  model-specific surface. Config manifests declare required capabilities.
- `search.py` gives the main agent a CPU-only diagnostic view. It can still run
  compose checks and print the method-baseline catalog, but those checks do not
  gate subagent launch; subagents inspect and edit inference code directly.

## Run
```bash
# CPU (needs torch): enumerate eligible dimensions + composable config space
~/lustre/miniconda3/envs/sana/bin/python search/search.py --model cosmos3
~/lustre/miniconda3/envs/sana/bin/python search/test_search.py
```

## Pipeline (this skeleton = the CPU half)
```
load model profile + config capability contract
  -> for each method family the main agent decides to wake:
       start from search_space + loops/<dim>/exploration.md
       run the bounded fan-out loop:
         observe current-experiment results -> hypothesize -> implement one config
         -> preflight -> launch -> authoritative gate -> retain/discard/reject and loop
       compose([technique(cfg)], manifest-derived spec)  # diagnostic, not driver
  -> [GPU stage, stubbed — plan_eval()]:
       render run bundle from profile + cfg -> scripts/launch_config.py
       collect benchmark.json/quality.json -> authoritative aligned assess
       -> compare vs profile baseline
       bin into 1.5x/2.0x/3.0x speed targets per evals/tiers.toml
```
The enumerate+compose half runs here and is tested. The eval+tiering half is the
GPU stage (`plan_eval()` stub) — it reuses the existing launcher/collector + eval
profile, so the search produces the plan's `final_matrix` of tiered configs.

Config failure rejects/logs that config and returns to the loop; config
success retains the config in the frontier only when quality or speed/memory
improves, then also returns to the loop. Stop only at max_iters, real blocker,
or explicit orchestrator release; structured-negative is logged as a proposal,
not a dimension-agent stop. Default fan-out
budget is fixed max_iters=40 with early_stop_patience=0; budget exits are
terminal_pending_review handoffs so the main agent can select low/medium/high
winners from the retained frontier. Those winners are 1.5x/2.0x/3.0x speed
targets selected by joint Gemini+LPIPS quality ranking. See
`docs/fanout-loop-contract.md`.

See `docs/search-architecture.md` for the full design.
