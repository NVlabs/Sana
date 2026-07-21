# Wan2.2 TI2V-5B

Wan2.2 TI2V-5B is the 5B dense text-to-video path in Sol-Engine. The optimized configuration runs single-GPU with kernel fusion and EasyCache.

## Speed

Full optimization uses lossless kernel fusion + EasyCache and reaches **2.885x** speedup (70.25s → 24.35s). Measured on a single GB200 with 704x1280, 121 frames, 50 denoising steps (guidance 5.0), warmup excluded, 5-prompt median.

The line decomposes multiplicatively: lossless kernel fusion **1.52x** × EasyCache (~47% of steps reused) **1.90x**.

## Launch

```bash
# baseline (vanilla WanPipeline, single GPU)
bash scripts/wan5b/run_baseline.sh

# optimized (~24.35s, 2.885x)
bash scripts/wan5b/run_optimized.sh
```

## Techniques

- [Cache](../techniques/cache.md): EasyCache reuses denoising work (threshold 0.036).
- [Kernel fusion](../techniques/kernel.md): regional `torch.compile`, [QKV merge](../techniques/kernel/qkv_merge.md), cross-attention K/V cache, BF16 block glue — algorithmically exact (no approximation).
