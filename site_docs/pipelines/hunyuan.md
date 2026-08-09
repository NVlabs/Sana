# HunyuanVideo-13B

HunyuanVideo-13B is a 13B text-to-video path in Sol-Engine. The optimized
configuration runs single-GPU with torch.compile, TeaCache, and **SOL Attention**
(`techniques/sparse_backends/`).

## Performance status

Full optimization uses compile + TeaCache + the released Sol-Attn kernel.
Historical timing used the retired sparse backend and is not quoted for this
configuration; a fresh same-config benchmark is pending.

## Launch

```bash
# baseline (vanilla diffusers HunyuanVideoPipeline, single GPU)
python3 scripts/launch_config.py config/hunyuan_video/baseline.toml --mode sbatch --confirm-submit

# optimized release stack
python3 scripts/launch_config.py config/hunyuan_video/full.toml --mode sbatch --confirm-submit
```

The optimized config enables torch.compile, TeaCache, and SOL Attention
together; every technique is env-gated (all flags off = byte-identical baseline).

## Techniques

- [Cache](../techniques/cache.md): TeaCache reuses denoising steps
  (threshold 0.15, start step 6, max 2 consecutive hits).
- [Kernel fusion](../techniques/kernel.md): torch.compile over the transformer.
- [Sparse attention](../techniques/sparse.md): **Sol-Attn** at `tau=1.0`;
  valid text K/V is an exact sink and valid text-query rows stay dense.
