# Wan2.2-A14B

Wan2.2-A14B is the 14B (2-expert MoE) text-to-video path in Sol-Engine. The optimized configuration runs single-GPU with kernel fusion, EasyCache, and PISA sparse attention.

## Speed

Full optimization uses kernel fusion + EasyCache + PISA and reaches **2.172x** speedup (449.67s → 207.01s). Measured on a single GB200 with 720x1280, 81 frames, 40 denoising steps (dual guidance 4.0/3.0), warmup excluded, 5-prompt median. Both MoE experts fit in one 192 GB HBM, so this is single-GPU inference.

The line decomposes multiplicatively: kernel fusion **1.13x** × EasyCache (~14/40 steps reused) **1.42x** × PISA (density 0.10) **1.35x**.

## Launch

```bash
# baseline (vanilla WanPipeline, single GPU)
bash scripts/wan14b/run_baseline.sh

# optimized (~207s, 2.172x)
bash scripts/wan14b/run_optimized.sh
```

The optimized script enables kernel fusion, EasyCache, and PISA together, with attention routed through `DIFFUSERS_ATTN_BACKEND=_native_cudnn`.

## Techniques

- [Cache](../techniques/cache.md): EasyCache reuses denoising work (threshold 0.30).
- [Kernel fusion](../techniques/kernel.md): fused QKV, compiled AdaLN / QK-norm+RoPE / FFN glue, invariant RoPE & conditioning caches.
- [Sparse attention](../techniques/sparse.md): [PISA](../techniques/sparse/pisa.md) block-sparse video self-attention (density 0.10, dense guards on first/last layers & steps).
