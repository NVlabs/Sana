# Quantization Report Excerpt

ported from Sol-LTX-Infer docs/diffusion/quantization.md @ 29d0d9e

## ModelOpt NVFP4 Family

The upstream quantization report treats `modelopt-nvfp4` as a checkpoint and
loader family, not only a numeric precision. Mixed transformer overrides with a
`config.json` use `--transformer-path`; raw NVFP4 exports use
`--transformer-weights-path`.

Validated diffusion NVFP4 examples in the upstream report include:

- `black-forest-labs/FLUX.1-dev` with a mixed BF16+NVFP4 transformer override.
- `black-forest-labs/FLUX.2-dev` with the official raw
  `black-forest-labs/FLUX.2-dev-NVFP4` export.
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers` with the primary transformer quantized and
  `transformer_2` kept BF16.

## LTX-2 Bring-Up Pattern

For local LTX-2 ModelOpt NVFP4 exports, the report uses
`build_modelopt_nvfp4_transformer` with the `ltx2-nvfp4` fallback preset to
create a mixed transformer override. For two-stage LTX-2 pipelines, the report
also describes mixing stage transformers and keeping a stage BF16 when full
stage-1 NVFP4 is slower for a target resolution or GPU.

## Blackwell Backend Note

The report calls out Blackwell/B200 NVFP4 backend selection. The validated
Wan2.2 path currently prefers FlashInfer FP4 GEMM through:

```bash
SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn
```

The same setting appears in the migrated LTX recipe. It is a backend workaround
for FP4 GEMM shapes where another JIT/CUTLASS path may reject the shape; a
validated fallback would be the longer-term implementation fix.

## Quality Policy For This Loop

NVFP4 is lossy weight/activation quantization. For autovideo candidates, PSNR is
diagnostic only because diffusion outputs have a BF16 chaos floor. A promoted
quant candidate must ship:

- `outputs/side_by_side.mp4` comparing baseline and NVFP4 output;
- visual judge output in `outputs/quality.json`;
- canonical benchmark and risk artifacts named by `docs/artifact-contract.md`.

The disabled NVFP4 path must recover the baseline loader path, but enabled
NVFP4 is not expected to be byte-exact.
