# Sub-Loops

Each acceleration line should live as an independent sub-loop with its own goal,
candidate manifest, eval profile, and artifacts.

The purpose is to let multiple agents work independently without sharing one
mutable `Sol-LTX-Infer` checkout.

## Sub-Loop Shape

```text
loops/<loop-id>/
  README.md
  goal.md
  acceptance.md
  candidate.toml
  eval.toml
  runs/
  scratch/
```

`runs/` and `scratch/` are ignored by git.

## Method Baseline Catalog

Each `dimension.toml` may declare `[[method_baseline]]` entries. These are not
fixed hyperparameter grids and are not delivery winners by themselves. They are
method-family starting points that tell goal agents whether a baseline is:

- `wired`: existing helper/runtime path can launch after normal manifest work;
- `candidate_wired`: helper/env exists, but target-runtime consumption still
  needs proof or a small adapter;
- `runtime_patch`: the method family must patch the live inference path;
- `upper_bound_probe`: diagnostic speed-ceiling probe that must not become a
  delivery winner without full quality evidence and safe fallback behavior.

The catalog exists to keep open-ended agents from only tuning the first wired
helper when the search-space markdown describes broader method families.

## Loop IDs

Suggested first loops:

- `baseline-eval`
- `sparse-attention-pisa`
- `step-cache`
- `token-prune`
- `kwl-fusion`
- `nvfp4`
- `full-stack`

## Done Criteria

A loop is complete when:

- `goal.md` is precise enough for Codex interactive goal mode
- `acceptance.md` defines promotion/rejection gates
- `candidate.toml` can launch or clearly states why it is methodology-only
- `eval.toml` points at an eval profile
- collector output exists for at least dry-run; GPU runs when applicable
