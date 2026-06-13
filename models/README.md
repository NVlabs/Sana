# `models/` — the model adapter layer

This is the **only** place model-specific knowledge lives. Everything else
(`efficiency/` engine, `loops/*` search dimensions, `search/` harness) is
model-agnostic. The acceleration search plugs a model in here and auto-applies
every eligible search dimension.

## A model = two small things
1. **A `ModelSpec`** — `efficiency/models/<id>_spec.py`: declares the structural
   seams the model exposes (capabilities + a few accessors, ~50 lines). This is
   the irreducible per-model adapter; `compose()` type-checks generic techniques
   against it. (`cosmos3_spec.py` is the Cosmos3 one; `ltx2_spec.py` the reference.)
2. **A profile** — `models/<id>.toml`: the official benchmark config, baseline
   numbers, run entrypoint, base env, and `[seam_status]` (which dimensions are
   wired vs need a one-time seam wiring). The search reads this.

## Adding a served model
```
1. write efficiency/models/<id>_spec.py  (register_model_spec; declare seams you've wired)
2. write models/<id>.toml                 (official config + baseline + run_script + seam_status)
3. python search/search.py --model <id>   (auto-lists eligible dimensions + candidate space)
```
No loop/dimension changes are needed — the same `loops/*` dimensions apply to any
model whose spec declares the required capability. A dimension a model hasn't
wired yet is auto-skipped (compose() refuses it), surfaced in the search plan.

## Current models
- `cosmos3` — nvidia/Cosmos3-Super (the first target). Baseline verified 130.4s.
- (reference) `LTX2` spec exists in `efficiency/models/ltx2_spec.py` (the proven
  template); add `models/ltx2.toml` to run the search against it too.
