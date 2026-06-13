# Dimension: nvfp4_ffn - NVFP4 FFN quantization

A **model-agnostic search dimension**. It searches load-time NVFP4 FFN
quantization configs and composes them against whatever model the search
targets. It names no model; model specifics live in `models/<id>.toml` and
`efficiency/models/<id>_spec.py`.

## What it searches

`efficiency/transforms/nvfp4_ffn.py` registers the `nvfp4_ffn` transform.
`NVFP4FFN` is a load-time `ModelTransform`, not a runtime `Technique`: it writes
the exclusive `FFN_PRECISION` seam and delegates to the model loader via the
existing TE NVFP4 FFN environment contract.

The search grid in `dimension.toml` covers the transform's real class params:

- `disable_rht`
- `disable_stochastic_rounding`
- `disable_2d_quantization`

The LTX-2.3 best selective video-FFN recipe is carried as the seed prior. Its
reference scripts, report, and helper live under `reference/`.

## Why it's model-agnostic

The dimension only names the registry transform, its params, and the seam it
writes. `requires_capabilities = []` matches the transform class: composition
does not require a structural runtime hook. The search builds candidates with
`build_transform("nvfp4_ffn", **cfg)` and lets `compose()` validate exclusive
seam conflicts against the selected `ModelSpec`.

The genuine per-model work is the loader seam. A target model must declare and
track that wiring in `models/<id>.toml [seam_status].ffn_precision`, then consume
the transform's loader env while preserving a clean BF16/off path. That adapter
work stays outside this loop.

## Migrated LTX-2.3 experience

The migrated reference material comes from `Sol-LTX-Infer` commit
`29d0d9e464000a2472345dcad51054b15aacca8d`:

- `scripts/slurm_ltx23_best_nvfp4_video_attn_ffn_sglquant_1080p10s.sh`
- `scripts/bench_te_nvfp4_gelu_epilogue.py`
- `docs/diffusion/quantization.md`

The LTX recipe used a 1080p/10s two-stage run with selective NVFP4 transformer
overrides for video attention and FFN linears. This dimension carries only the
FFN quantization methodology forward. The attention quantization path is
reference context, not part of this search dimension.

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

See `acceptance.md` for promotion gates and `references.md` for provenance.
