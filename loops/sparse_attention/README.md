# Dimension: sparse_attention - training-free sparse attention

A search dimension for training-free sparse, routed, approximate, or
mask-reuse attention experiments. Native subagents should read
`search_space/04_sparse_attention.md`, then inspect and modify target-model
self-attention, cross-attention, and joint/GEN attention paths directly in their
isolated worktree.

## What it searches

The generic implementation is
`efficiency/transforms/sparse_attention.py` (`SparseAttention`). It is a
build-time transform that installs an attention backend by setting the existing
SGLang HQ backend env/config:

- `SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS`
- `SGLANG_HQ_ATTENTION_BACKEND_CONFIG`

The currently wired transform covers a PISA/piecewise-style backend with params
such as `sparsity`, `block_size`, `component`, `route_mode`, `dense_fallback`,
and `stage2_dense_layers`. That is one runnable family, not the whole
dimension. Search should also consider training-free families such as
spatial/temporal head routing, semantic-aware permutation, online precise mask
search, proxy-mask prediction, rotating anchors, QK co-clustering, head-wise
adaptive budgets, and MInference-style dynamic patterns.

When a candidate uses an axis that the current backend does not consume, it must
patch the target runtime directly and prove the env/config is not metadata-only.

## Exploration Mode

Do not wait for a predeclared swappable-attention seam. Inspect the live
attention modules and dispatch paths, then implement the candidate directly
where the backend or routing decision actually occurs. Main-agent integration
can later normalize the implementation.

Before spending GPU iterations, record an attention preflight:

- attention call timing and which path dominates;
- token/frame/tile layout and sequence length;
- self-attention vs cross-attention vs joint/GEN attention sensitivity;
- available sparse kernels or backend hooks;
- dense fallback behavior and OFF identity.

Use the same frontier rule as step cache: retain a candidate when quality
improves or speed/memory improves; discard it when neither improves or
speed/memory regresses. A failed target-selection check is not a dimension-level
stop.

## Independent test

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/sparse_attention/test_sparse_attention.py
```

CPU-only; validates the transform through `efficiency` against the registered
target model and a local fixture that declares `SWAPPABLE_ATTENTION`.

## Run it in the search

```bash
python search/search.py --model <model-id>
```

See `acceptance.md` for frontier retention and final tier selection rules.
