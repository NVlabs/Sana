## Transformer Kernel Scope

Optimize the Hunyuan transformer/DiT denoising path. New candidates should target
kernel-level or runtime-level work inside repeated transformer blocks, attention
paths, FFN paths, and transformer glue code.

Do not spend new candidate work on these parts:

- text encoders, tokenization, prompt handling, or text-encoder cache behavior;
- VAE encode/decode, VAE tiling, VAE slicing, VAE postprocess, or video export;
- scheduler settings, denoising step count, prompt/guidance, LoRA state,
  resolution, frame count, output shape, cache/prune policy, sparse-attention
  semantics, or quantization policy.

Existing VAE or text-encoder evidence may be summarized as historical context,
but it should not be extended into new candidates under this transformer-kernel
scope. Ordinary loop evaluation must stay at single-DiT or module level. Full
diffusion evaluation is reserved for terminal validation after reviewer exit
intent because that is the authoritative end-to-end visual quality gate.

## Kernel Technique References

Use the Sana Sol-Engine kernel-fusion examples as reference starting points, not
as a fixed roadmap. First consider these directions, then adapt or extend them
according to the actual Hunyuan transformer structure, tensor shapes, dtype,
attention backend, and profiler evidence:

- AdaLN and residual gate fusion: normalize, scale, shift, gate, and residual
  glue around DiT blocks.
- GEMM epilogues: fuse memory-bound work after GEMMs, such as bias, activation,
  FFN output glue, residual updates, or normalization-adjacent epilogues.
- QK-norm plus RoPE fusion on attention Q/K paths.
- Attention output gate fusion after attention value aggregation and output
  projection.
- Residual and modulation glue fusion around transformer block boundaries.
- QKV merge when model layout allows equivalent merged projection execution.
- `torch.compile` or compiler fusion for stable, repeated transformer regions;
  record cold compile, warm timing, cache behavior, and failure modes separately.

Reference source:
https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/kernel/
