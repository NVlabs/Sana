# Dimension: nvfp4_ffn - NVFP4 FFN quantization

A search dimension for low-precision FFN or linear-layer experiments. Native
subagents should read `search_space/03_quantization.md`, then inspect and
modify Cosmos3 module loading and inference code directly in their isolated
worktree.

## What it searches

`efficiency/transforms/nvfp4_ffn.py` registers the `nvfp4_ffn` transform.
`NVFP4FFN` is a load-time diagnostic helper, not a runtime `Technique`: it
delegates to the model loader via the existing TE NVFP4 FFN environment contract.

The search grid in `dimension.toml` covers the transform's real class params:

- `disable_rht`
- `disable_stochastic_rounding`
- `disable_2d_quantization`

Current exploration starts from `search_space/` plus model-specific module
profiling. Subagents choose module scope, dense guards, precision format, and
fallback policy from evidence gathered in code/traces.

## Exploration Mode

Do not wait for a predeclared precision seam. Inspect the loader and FFN/linear
modules directly, then implement the candidate where it is easiest to prove a
clean OFF path and controlled ON behavior.

## Deploy requirement

The real kernel path needs a CUDA/TransformerEngine build with NVFP4 support on
B200/GB200-class hardware. That is a deployment prerequisite, not a model
coupling. The CPU loop test does not import TransformerEngine or run kernels.

## Quality policy

NVFP4 FFN quantization is lossy. Disabled NVFP4 must recover the baseline path,
but enabled NVFP4 is not expected to be byte-identical. Promotion requires
`outputs/side_by_side.mp4` and the configured visual judge result. PSNR is
recorded as a diagnostic only and is not a hard promotion gate.

## Independent test

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/nvfp4_ffn/test_nvfp4_ffn.py
```

CPU-only; validates the transform contract through `efficiency`. The
search-level check that this dimension stays model-agnostic lives in
`search/test_search.py`.

## Run it in the search

```bash
python search/search.py --model <id>
```

See `acceptance.md` for promotion gates.
