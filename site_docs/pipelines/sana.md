# SANA-Video

SANA-Video is the 2B text-to-video path in Sol-Engine. The optimized configuration uses EasyCache, attention/kernel fusion, and `torch.compile`.

## Speed

Full optimization uses EasyCache + fusion + compile and reaches 2.77x speedup.
Measured on GB200 with 832x480, 81 frames, and 50 denoising steps, warmup excluded.

## Launch

```bash
PY=.conda/ltx23/bin/python

$PY scripts/sana/sana_video_sglang_run.py \
  --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --prompt "a corgi running on the beach" \
  --output out/sana_baseline

$PY scripts/sana/sana_video_sglang_run.py \
  --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --prompt "a corgi running on the beach" \
  --output out/sana_fullopt \
  --easycache 0.1 --linattn-bf16 --qkv-merge --compile
```

## Techniques

- [Cache](../techniques/cache.md): EasyCache reuses denoising work.
- [Kernel fusion](../techniques/kernel.md): linear attention BF16 path and QKV merge reduce memory-bound overhead.
