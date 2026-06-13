# Dimension: kwl_fusion - KWL operator fusion

A **model-agnostic search dimension**. It searches build-time KWL operator-fusion
bundles and composes them against whatever model the search targets. It names no
model; model specifics live in `models/<id>.toml` +
`efficiency/models/<id>_spec.py`.

## What it searches

`efficiency/transforms/kwl_fusions.py` registers **`kwl_fusions`**, a build-time
`ModelTransform` that writes the `KERNEL_FUSION` seam and emits
`SGLANG_HQ_KWL_*` build flags. Its real search parameter is `flags`: a tuple of
fusion switches. `dimension.toml` searches:

- the OFF/identity bundle (`flags = []`);
- the full LTX-2.3 proven bundle;
- leave-one-out variants that disable one KWL flag at a time.

The searched flags cover block/guidance sharing, fused QK/RoPE and QKNorm/RoPE,
RMS/AdaLN paths, dual and cross-attention dual modulation, Ada value handling,
residual gate fusion, FFN `proj_in + GELU`, gate-to-output compile, audio QKVG,
and tiled VAE compile.

## Why it's model-agnostic

`KWLFusions` declares `required_capabilities = []`, so the search can compose it
with any registered `ModelSpec`. The transform itself only declares a generic
build-time operator-fusion intent; each model adapter decides whether its build
path consumes those flags.

The per-model seam is kept OUT of this dimension: a model must wire its module
construction to consume the KWL flag bundle and record that wiring in
`models/<id>.toml [seam_status]`. If a model has not wired fused operator paths,
the dimension can still compose, but a real promotion run must show that ON
installs the intended fused kernels and OFF recovers the baseline path.

## Migrated LTX-2.3 experience (the search prior)

The full bundle in `dimension.toml` is seeded from the migrated LTX-2.3 KWL
recipe and report:

- `reference/kwl_fusion/recipe.sh` preserves the KWL wrapper and flag mapping.
- `reference/kwl_fusion/kwl_ops.py` preserves the official operator installer reference.
- `reference/kwl_fusion/report.md` records the 1.26x official HQ KWL result and the
  lossless/operator-only acceptance interpretation.

KWL is treated as operator-only: it must not change scheduler, step count,
prompting, guidance, LoRA state, resolution, frame count, token set, or attention
semantics. OFF is expected to be identity; ON may differ at low-order numeric
levels because fused kernels change floating-point operation ordering, so
side-by-side visual review remains part of promotion.

## Independent test

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/kwl_fusion/test_kwl_fusion.py
```

CPU-only; validates the transform through `efficiency.compose`, checks the full
KWL env bundle, checks a subset/ablation bundle, and smoke-imports the migrated
reference helper.

## Run it in the search

```bash
python search/search.py --model <id>
```

The search enumerates this dimension's composable build-transform candidates for
the selected model profile. See `acceptance.md` for promotion gates and
`references.md` for provenance.
