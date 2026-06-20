# Dimension: nvfp4_ffn - NVFP4 hot-linear quantization

A search dimension for low-precision hot-linear experiments across FFN/MLP,
attention projection, output projection, or profiled linear subsets. Native
subagents should read `search_space/03_quantization.md`, then inspect and modify
target-model module loading and inference code directly in their isolated
worktree.

## What it searches

`efficiency/transforms/nvfp4_ffn.py` registers the `nvfp4_ffn` transform.
`NVFP4FFN` is a load-time diagnostic helper, not a runtime `Technique`: it
delegates to the model loader via the existing TE NVFP4 environment contract.

The transform exposes a search surface rather than a fixed grid. Some axes are
already consumed by the current target runtime; others are metadata until a
candidate explicitly wires and validates them in the loader.

Already wired for the current target path:

- TE recipe flags: `disable_rht`, `disable_stochastic_rounding`,
  `disable_2d_quantization`
- row padding policy: `pad_m_to`
- FP4 GEMM backend override: `fp4_gemm_backend`

Candidate-wired axes that must be proven before they count:

- `module_scope`: FFN only, attention projections, output projections, or a
  profiled subset
- `dense_layers`: BF16 fallback layers or block windows
- `dense_steps`: BF16 fallback denoising steps
- `row_scaled_activation`: TE recipe variant when supported by the installed
  TransformerEngine version
- fused TE epilogues: disabled in the active Cosmos3 manifest unless a future
  adapter preserves Cosmos3 bias-free SwiGLU semantics
- `fallback_policy`: BF16 fallback and unsupported-hardware behavior

Current exploration starts from `search_space/` plus model-specific module
profiling. Subagents choose module scope, dense guards, precision format, and
fallback policy from evidence gathered in code/traces.

## Required Preflight

Before spending GPU iterations, record a preflight note in `SEARCH_JOURNAL.md`:

- GPU capability and whether it is Blackwell/SM100 or later.
- TransformerEngine import/version and whether `NVFP4BlockScaling` is available.
- A minimal `te.Linear` or target-loader smoke result.
- Whether each planned env var is actually consumed by the target loader.
- OFF path identity with NVFP4 disabled.

If Blackwell-class NVFP4 or TransformerEngine support is missing, stop with a
real blocker instead of running synthetic candidates.

## Exploration Mode

Do not wait for a predeclared precision seam. Inspect the loader and hot linear
modules directly, including FFN/MLP, attention projections, and output
projections, then implement the candidate where it is easiest to prove a clean
OFF path and controlled ON behavior.

Each candidate should name the exact module family it touches, for example:

- FFN `proj_in` / `proj_out` only
- fused `proj_in + GELU` only for a model whose FFN semantics actually match
- fused `proj_out + bias/gate` only for a model whose residual/gate path matches
- attention projections only after profiling shows they matter
- explicit exclusions for small or quality-sensitive layers

## Deploy requirement

The real kernel path needs a CUDA/TransformerEngine build with NVFP4 support on
B200/GB200-class hardware. That is a deployment prerequisite, not a model
coupling. The CPU loop test does not import TransformerEngine or run kernels.

## Quality policy

NVFP4 hot-linear quantization is lossy. Disabled NVFP4 must recover the baseline
path, but enabled NVFP4 is not expected to be byte-identical. Frontier retention
and final tier selection require `outputs/side_by_side.mp4` and the configured
visual judge result. PSNR is recorded as a diagnostic only and is not a final
selector. Reliable numeric/precision checks and silent-fallback detection are
recorded; they become hard gates only when the candidate contract explicitly
declares them. LPIPS and Gemini are both used for quality ranking inside speed
targets.

Retained frontier candidates must record:

- quantized module list and any BF16 dense guard;
- TE recipe flags and FP4 GEMM backend;
- hardware, CUDA, and TransformerEngine version;
- warm/cold compile state;
- latency, peak memory, LPIPS, pairwise Gemini, and run artifacts;
- fallback or blocker reason if the candidate could not exercise real NVFP4.

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

See `acceptance.md` for frontier retention and final tier selection rules.
