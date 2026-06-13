# Dimension: sparse_attention - PISA sparse attention

A **model-agnostic search dimension**. It searches PISA/piecewise sparse
attention backend configs and composes them against whichever model the search
targets. It names no model in its schema; model specifics live in
`models/<id>.toml` and `efficiency/models/<id>_spec.py`.

## What it searches

The generic implementation is
`efficiency/transforms/sparse_attention.py` (`SparseAttention`). It is a
build-time transform that installs an attention backend by setting the existing
SGLang HQ backend env/config:

- `SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS`
- `SGLANG_HQ_ATTENTION_BACKEND_CONFIG`

`dimension.toml` searches a small grid of real transform params:
`sparsity`, `component`, and `stage2_dense_layers`. The transform defaults keep
the dense fallback available and use the existing `piecewise_attn` backend
rather than reimplementing sparse attention in this repo.

## Why it is model-agnostic

`SparseAttention` writes the exclusive `ATTENTION_BACKEND` seam and requires
`Capability.SWAPPABLE_ATTENTION`. The search calls
`compose([build_transform("sparse_attention", **cfg)], spec)` for the target
model. If that model has not declared the swappable-attention capability, the
dimension is automatically skipped.

To enable this dimension for a model, wire the model's attention-backend seam in
its adapter, add `Capability.SWAPPABLE_ATTENTION` to
`efficiency/models/<id>_spec.py`, and record the wiring state in
`models/<id>.toml [seam_status]`. The dimension itself does not change.

The current `cosmos3` profile intentionally does not declare
`swappable_attention`, so `python search/search.py --model cosmos3` should show
this dimension as `[skip]`. That is the correct model-agnostic behavior until
the target model wires the seam.

## Migrated LTX-2.3 priors

`reference/recipe.md` captures the proven LTX-2.3 PISA recipe:
`piecewise_sparsity=0.9`, `piecewise_block_size=64`, and
`piecewise_stage1_dense_steps=3`, with stage 2 routed to `piecewise_attn`.
Those values seed `dimension.toml` as priors; `reference/report.md` preserves
the reported sparse-attention results and `references.md` records provenance.

## Independent test

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/sparse_attention/test_sparse_attention.py
```

CPU-only; validates the transform through `efficiency` against a local fixture
that declares `SWAPPABLE_ATTENTION`, and verifies that an unwired target spec is
rejected by composition.

## Run it in the search

```bash
python search/search.py --model cosmos3   # skip until that model wires swappable_attention
```

See `acceptance.md` for promotion gates and `references.md` for provenance.
