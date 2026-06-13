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
  references.md
  runs/
  scratch/
```

`runs/` and `scratch/` are ignored by git.

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
- `references.md` links to the relevant successful branches/scripts
- collector output exists for at least dry-run; GPU runs when applicable
