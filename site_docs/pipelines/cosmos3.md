# Cosmos3-Super

Cosmos3-Super is the 64B multi-GPU pipeline. The optimized configuration combines TeaCache with NVFP4.

## Speed

Conservative full optimization uses TeaCache + NVFP4 and reaches ~2.26x speedup.
Measured on 4x GB200 with 1280x720, 189 frames, and 35 denoising steps, warmup excluded.

## Launch

```bash
MODEL_REPO=nvidia/Cosmos3-Super \
ROOT=out/cosmos3 \
PROMPT_FILE=prompt.txt \
PROMPT_TAG=demo \
bash scripts/cosmos/slurm_cosmos3_super.sh baseline

MODEL_REPO=nvidia/Cosmos3-Super \
ROOT=out/cosmos3 \
PROMPT_FILE=prompt.txt \
PROMPT_TAG=demo \
bash scripts/cosmos/slurm_cosmos3_super.sh fullopt
```

The launcher contains Slurm headers for cluster use. On a single multi-GPU host, run it as a normal shell script after adjusting site-specific paths near the top.

## Techniques

- [Cache](../techniques/cache.md): TeaCache uses residual similarity across denoising steps.
- [Quantization](../techniques/quant.md): NVFP4 is applied selectively, with first and last steps kept dense.
