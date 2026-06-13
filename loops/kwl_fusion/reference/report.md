# Ported Report Excerpt: LTX-2.3 KWL

Ported from Sol-LTX-Infer `docs/ltx23_official_hq_kwl_report.md` and
`docs/diffusion/ltx2_dit_fusion_report.md` @ `29d0d9e`.

## Scope

The official HQ KWL branch kept the faithful Lightricks two-stage pipeline and
applied only kernel-wise lossless execution changes inside the official
`ltx_core` transformer build path. The KWL path did not include sparse
attention, FP4, step-count changes, scheduler changes, prompt changes, CFG
changes, or LoRA-strength changes.

## Techniques Applied

The report lists these KWL paths as preserving the official algorithmic graph,
with differences limited to kernel-level floating-point rounding and launch
grouping:

- fused bf16 RMSNorm plus AdaLN scale/shift;
- fused Q/K RMSNorm pair;
- fused Q/K RMSNorm plus SPLIT RoPE pair;
- FFN `proj_in + bias + GELU(tanh)` via fused ATen addmm activation, followed
  by the original official `proj_out`.

Installed modules in the 22B official transformer:

- 48 transformer blocks;
- 288 attention modules;
- 96 FFN modules.

## Benchmark Excerpt

| Variant | E2E wall time | Speedup vs faithful baseline | Stage 1 denoise | Stage 2 denoise |
| --- | ---: | ---: | ---: | ---: |
| Official faithful HQ baseline | 321.71s | 1.00x | 74s | 27s |
| Official HQ + KWL kernels | 256.27s | 1.26x | 68s | 26s |
| Official HQ + KWL kernels, second process | 287.33s | 1.12x | 68s | 24s |

The Slurm logs confirmed the KWL transformer patch installed 48 transformer
blocks, 288 attention modules, and 96 FFN modules.

## Quality Interpretation

The KWL path has no algorithm-level quality loss by construction, but diffusion
is numerically sensitive. Kernel-level bf16 rounding differences can amplify
into visible pixel differences over a full trajectory, so side-by-side visual
inspection remains part of the acceptance gate.

## Lossless Interpretation From The Fusion Report

The final LTX fusion report defines lossless as preserving the sampling
algorithm, scheduler, step count, CFG/STG semantics, LoRA weights, attention
semantics, resolution, and frame count. It explicitly does not require bitwise
identity because bf16 casts, fused multiply-adds, GEMM tiling, Inductor, CuTeDSL,
and Triton execution order can change low-order bits.

## Retained Switches And Leave-One-Out Discipline

The retained LTX switches were:

```bash
SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN=1
SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=1
SGLANG_LTX2_FUSED_ADALN=1
SGLANG_LTX2_FUSED_QKNORM_ROPE=1
SGLANG_LTX2_FUSED_DUAL_MODULATE=1
SGLANG_LTX2_FUSED_ADA_VALUES_ALL=1
SGLANG_LTX2_FUSED_RESIDUAL_GATE=1
SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=1
SGLANG_LTX2_COMPILE_GATE_TO_OUT=1
SGLANG_LTX2_FUSED_AUDIO_QKVG=1
SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=1
SGLANG_LTX2_VAE_COMPILE_MODE=max-autotune-no-cudagraphs
```

The report also records candidates that were tested and not retained, such as
A2V gate-to-output compile and stage-2 audio gate-to-output compile, because
full-run measurements were negative. Cosmos3 should follow the same
leave-one-out rule: keep only fusions with measured benefit and clean OFF
behavior.
