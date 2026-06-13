# Loop: nvfp4_ffn

## Purpose

Bring the LTX-2.3 TE NVFP4 FFN quantization recipe into an autovideo loop and
verify the existing `efficiency.transforms.nvfp4_ffn.NVFP4FFN` transform surface
for Cosmos3.

## LTX-2.3 Provenance

The migrated reference material comes from `Sol-LTX-Infer` commit
`29d0d9e464000a2472345dcad51054b15aacca8d`
(`origin/codex/cosmos3-run-env`):

- `scripts/slurm_ltx23_best_nvfp4_video_attn_ffn_sglquant_1080p10s.sh`
- `scripts/bench_te_nvfp4_gelu_epilogue.py`
- `docs/diffusion/quantization.md`

The LTX recipe used a 1080p/10s two-stage LTX-2.3 run with selective NVFP4
transformer overrides for video attention and FFN linears. Its important knobs
were the FlashInfer FP4 GEMM backend, FP4 quantization backend, fused FFN
epilogues, and the TE NVFP4 FFN loader flags captured under `reference/`.

This is a Blackwell/TransformerEngine path. The real GEMMs need a CUDA/TE build
with NVFP4 support on B200/GB200-class hardware; the loop test does not import
TransformerEngine or run kernels.

## Efficiency Mapping

The in-repo transform is `efficiency/transforms/nvfp4_ffn.py`.

`NVFP4FFN` is a load-time `ModelTransform`, not a runtime `Technique`. It writes
the exclusive `FFN_PRECISION` seam and delegates to the existing loader by
setting these env keys:

- `SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1`
- `SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=1`
- `SGLANG_LTX2_TE_NVFP4_DISABLE_RHT=1`
- `SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1`
- `SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION=1`

The independent test composes `NVFP4FFN` against `get_model_spec("Cosmos3")`
and asserts `plan.apply_transforms(..., env={})` emits those keys. It also
checks the no-FP4 variant leaves the primary env key unset, mirroring
`efficiency/selftest.py` section `[7]`.

## Cosmos3 Wiring Still Needed

This loop does not edit shared `efficiency/` code. A future Cosmos3 implementation
should wire the methodology into the execution repo by:

- teaching the Cosmos3 loader to consume `SGLANG_HQ_ENABLE_TE_NVFP4_FFN`;
- identifying the FFN projection linears in `Cosmos3OmniTransformer.gen_layers`;
- applying TE NVFP4 only to the intended FFN projections while preserving a
  clean BF16/off path;
- keeping the official `evals/profiles/official_video_t2v.toml` config unchanged;
- producing `outputs/side_by_side.mp4` and a visual judge result for every lossy
  quant candidate.

## Quality Policy

NVFP4 FFN quantization is lossy. Disabled NVFP4 must recover the baseline path,
but enabled NVFP4 is not expected to be byte-identical. Promotion requires
side-by-side visual review and the configured visual judge. PSNR is recorded as
a diagnostic only and is not a hard promotion gate.

## Candidate

The launcher-runnable manifest is `candidates/nvfp4_ffn.toml`; this loop keeps
an identical copy at `loops/nvfp4_ffn/candidate.toml`.

## Test

Run the required CPU-only gate with the torch environment:

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/nvfp4_ffn/test_nvfp4_ffn.py
```

## Status

`ready-for-codex`
