# LingBot-Video

LingBot-Video is the MoE 30B-A3B (~3B active) two-stage text-to-video path in Sol-Engine: a 480p base DiT followed by a 1080p refiner. The optimized configuration runs 4-GPU (CP4 + FSDP) with a cuDNN attention backend, refiner-only PISA, and per-stage EasyCache.

## Speed

Full optimization reaches **2.60x** speedup (375.53s → 144.36s). Measured on 4×GB200 (CP4 Ulysses + FSDP), base 480x832 → refiner 1088x1920, 121 frames (base 40 steps, refiner 8), warmup excluded, 3-prompt median. The baseline is the model author's recommended 4-GPU config (CP4 + FSDP + batched CFG + FA2), so the optimized run keeps the **same topology** — 2.60x is pure optimization, not parallelization.

The line decomposes multiplicatively: attention-backend swap (FA2 → cuDNN) **1.79x** × refiner-only PISA (density 0.10) **1.12x** × per-stage EasyCache **1.30x** (base 1.18x × refiner 1.10x).

## Launch

```bash
# baseline (official recommended 4-GPU config, FA2)
bash scripts/lingbot/run_baseline.sh

# optimized (~144s, 2.60x) — 4 GPUs
bash scripts/lingbot/run_optimized.sh
```

## Techniques

- [Kernel fusion](../techniques/kernel.md): cuDNN attention backend.
- [Sparse attention](../techniques/sparse.md): [PISA](../techniques/sparse/pisa.md) applied to the 1080p refiner (density 0.10).
- [Cache](../techniques/cache.md): EasyCache applied per stage (base threshold 0.08, refiner threshold 0.25).
