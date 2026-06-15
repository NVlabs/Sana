# Dimension: sparse_attention - PISA sparse attention

A search dimension for sparse, routed, or approximate attention experiments.
Native subagents should read `search_space/04_sparse_attention.md`, then inspect
and modify Cosmos3 self-attention, cross-attention, and joint/GEN attention
paths directly in their isolated worktree.

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

## Exploration Mode

Do not wait for a predeclared swappable-attention seam. Inspect the live
attention modules and dispatch paths, then implement the candidate directly
where the backend or routing decision actually occurs. Main-agent integration
can later normalize the implementation.

## Independent test

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/sparse_attention/test_sparse_attention.py
```

CPU-only; validates the transform through `efficiency` against Cosmos3 and a
local fixture that declares `SWAPPABLE_ATTENTION`.

## Run it in the search

```bash
python search/search.py --model cosmos3
```

See `acceptance.md` for promotion gates.
