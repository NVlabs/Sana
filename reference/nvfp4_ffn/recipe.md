# NVFP4 FFN Recipe Notes

ported from Sol-LTX-Infer scripts/slurm_ltx23_best_nvfp4_video_attn_ffn_sglquant_1080p10s.sh @ 29d0d9e

## Scope

This is the LTX-2.3 reference recipe for selective NVFP4 video attention and FFN
transformer overrides. The autovideo direction only carries the FFN quantization
methodology forward; attention quantization is reference context, not part of
the Cosmos3 `nvfp4_ffn` candidate.

## LTX Run Shape

- Model: `Lightricks/LTX-2.3`
- Pipeline: `LTX2TwoStagePipeline`
- Device mode: resident two-stage run
- Resolution/duration: `1920x1088`, `241` frames, `24` fps
- Steps: `30` inference steps
- Seed: `42`
- Quantized components:
  - `outputs/ltx23-selective-nvfp4-video-attn-ffn-transformer-mat`
  - `outputs/ltx23-selective-nvfp4-video-attn-ffn-stage2-lora-transformer-mat`

## Important Env Knobs

The recipe used these FP4/NVFP4 settings:

```bash
SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn
SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND=flashinfer
SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU=1
SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE=1
SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE=0
```

The in-repo `NVFP4FFN` transform maps the autovideo-facing HQ flag onto the
existing LTX TE NVFP4 loader env:

```bash
SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1
SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=1
SGLANG_LTX2_TE_NVFP4_DISABLE_RHT=1
SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1
SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION=1
```

## Runtime Requirement

This path is reference-only without a Blackwell CUDA environment and a
TransformerEngine build that supports `NVFP4BlockScaling`. The migrated test
does not import TransformerEngine and only checks that the `efficiency/`
transform emits the expected environment contract.
