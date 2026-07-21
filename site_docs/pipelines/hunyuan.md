# HunyuanVideo-13B

HunyuanVideo-13B is a 13B text-to-video path in Sol-Engine. The optimized
configuration runs single-GPU with torch.compile, TeaCache, and **SOL Attention**
(`techniques/sparse_backends/`).

## Speed

Full optimization uses compile + TeaCache + SOL Attention and reaches **5.03x**
speedup (856.1s → 170.4s). Measured hot-vs-hot on a single GB200 with 1280x720,
129 frames, 50 denoising steps, one warmup pass with all technique clocks
cold-started for the timed pass.

## Launch

```bash
# baseline (vanilla diffusers HunyuanVideoPipeline, single GPU)
python3 scripts/launch_candidate.py candidates/hunyuan_video_baseline.toml --mode sbatch --confirm-submit

# optimized (~170s, 5.03x)
python3 scripts/launch_candidate.py candidates/hunyuan_video_full_v3.toml --mode sbatch --confirm-submit
```

The optimized candidate enables torch.compile, TeaCache, and SOL Attention
together; every technique is env-gated (all flags off = byte-identical baseline).

## Techniques

- [Cache](../techniques/cache.md): TeaCache reuses denoising steps
  (threshold 0.15, start step 6, max 2 consecutive hits).
- [Kernel fusion](../techniques/kernel.md): torch.compile over the transformer.
- [Sparse attention](../techniques/sparse.md): **SOL Attention** block-sparse
  video self-attention (density 0.15); text conditioning and the first
  denoising steps stay exact dense.
