# Dimension: kwl_fusion - KWL operator fusion

A search dimension for kernel and operator fusion experiments. Native subagents
should read `search_space/05_kernel_fusion.md`, then inspect and modify Cosmos3
hot inference paths directly in their isolated worktree.

## What it searches

`efficiency/transforms/kwl_fusions.py` registers **`kwl_fusions`**, a build-time
diagnostic helper that can emit `SGLANG_HQ_KWL_*` build flags. This helper can
be reused, replaced, or ignored by subagents. `dimension.toml` records search
axes only; it does not define a fixed flag bundle to copy.

## Exploration Mode

Do not wait for a predeclared kernel-fusion seam. Inspect actual hot operator
chains, module construction, and compile behavior, then implement the candidate
where it is easiest to prove OFF identity and warm-speed benefit.

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
KWL env bundle and checks a subset/ablation bundle.

## Run it in the search

```bash
python search/search.py --model <id>
```

The search reports this dimension and its quality gates. See `acceptance.md` for
promotion gates.
