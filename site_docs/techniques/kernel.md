# Kernel fusion

Kernel fusion reduces memory-bound overhead inside DiT blocks. The target is the repeated glue around GEMMs: layout movement, normalization, activation, gate application, QK normalization, RoPE, and precision conversion.

The paper frames kernel fusion as a late-stage local-tuning problem: after other methods shift the bottlenecks, the fusion agent profiles fragmented operators and chooses custom kernels, compiler fusion, or GEMM epilogues for the current tensor shapes and precision formats.

## In Sol-Engine

| Pipeline | Fusion path | Role |
|---|---|---|
| LTX-2.3 | kernel fusion | Operator-level optimization before other methods |
| SANA-Video | Linear attention BF16, QKV merge, compile | Part of the 2.77x path |
| Wan-5B | regional compile + QKV merge + cross-KV cache + BF16 block glue | lossless kernel path, **1.52×** |
| Wan-14B | fused QKV + compiled AdaLN / QK-norm+RoPE / FFN glue + invariant caches | **1.13×** |
| LingBot | cuDNN attention backend | **1.79×** |

## Methods

| Method | Role in the design space |
|---|---|
| [AdaLN and residual gate fusion](kernel/adaln_residual_gate.md) | AdaLN normalization, scale, shift, and residual gate fusion |
| [GEMM epilogues](kernel/gemm_epilogues.md) | GEMM + activation / bias / FFN epilogue fusion described by CUTLASS and ByteTransformer-style paths |
| [QK-norm + RoPE fusion](kernel/qknorm_rope.md) | fused Q/K normalization and rotary-position work on attention Q/K paths |
| [Attention output gate fusion](kernel/attention_output_gate.md) | attention output projection, gate application, and output-path glue fusion |
| [Residual and modulation glue fusion](kernel/residual_modulation.md) | residual update and modulation-path fusion around DiT blocks |
| [QKV merge](kernel/qkv_merge.md) | merged projection path used in the SANA-Video optimization stack |
| [torch.compile](kernel/torch_compile.md) | compiler-driven graph fusion for stable operator sequences |

## Scope

Fusion should not change scheduler settings, prompts, seeds, or the number of denoising steps. It is an implementation-level optimization that reduces launch and memory traffic overhead.
