# MiniMax-H3

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is a 33B audio-video generation model. Sol-Engine provides hardware-specific full-stack runtimes that combine kernel optimization, Sol-Attn, caching, and memory-efficient decoding. Code and full docs: [`models/minimax_h3`](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3).

## Speed

| Hardware | GPUs | Workload | Baseline (s) | Optimized (s) | Speedup |
|---|---:|---:|---:|---:|---:|
| GB200 | 8 | 768p@5s | 27.21 | 6.88 | **3.95x** |
| GB10 (DGX Spark) | 1 | 480p@5s | 710.6 | 181.3 | **3.92x** |
| RTX 4090 | 1 | 768p@5s | 2239.22 | 504.33 | **4.44x** |
| RTX 5090 | 1 | 768p@5s | 1045.4 | 231.2 | **4.52x** |
| H100 | 4 | 768p@5s | 81.47 | 22.89 | **3.56x** |
| A100 | 4 | 768p@5s | 217.32 | 61.28 | **3.55x** |

Speedups are measured against the matching baseline runtime on the same hardware.

## Entry points

| Entry | Hardware | Headline |
|---|---|---|
| [Full-opt runtimes](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3) | GB200, GB10, RTX 4090, RTX 5090, H100, A100 | Table above |
| [Sol-H3](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3) | 8x B300 | T2V 1.653 / 3.732 / 6.612 s for 124 / 243 / 362 frames (warm 8-GPU medians) |
| [HyperFlow](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/HyperFlow) | 8x B300 | Eight-step H3, 2.86–4.15x over the original HyperFlow runtime |
| [Sol-H3-RTX5090](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-RTX5090) | 1x RTX 5090 | One T2VA request in 39.80 s (1344x768, 121 frames, with audio) |
| [Sol-H3-Spark](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-Spark) | 1x DGX Spark | 56 s hot E2E reported on the project page (1344x768, 121 frames, with audio) |
| [Sol-H3 in ComfyUI](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-ComfyUI) | RTX 5090, DGX Spark | Two stage: 35.5 s (RTX 5090), 64.3 s (DGX Spark); 989.0 s for original 50-step H3 on RTX 5090 |
| [Super acceleration](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/super_acceleration) | 2x GB200 (H3 + LTX-2.5 Refiner) | 6.760544632 s median across 20 hot complete requests |

Each entry has its own timing boundary; see its README before comparing numbers.

## Launch

```bash
python scripts/launch_config.py config/minimax_h3/minimax_h3_h100_fullopt.toml --mode local
```

Select the full-opt config for your GPU from [`config/minimax_h3`](https://github.com/NVlabs/Sana/tree/sol-engine/config/minimax_h3).

## Project pages

- [MiniMax-H3 with Sol Engine (GB200)](https://nvlabs.github.io/Sana/Sol-Engine/H3/)
- [H100 and A100](https://nvlabs.github.io/Sana/Sol-Engine/H3-DataCenter/) · [DGX Spark and RTX 5090](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/)
- [Sol-H3 on 8x B300](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) · [Sol-H3 on DGX Spark](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/)
- [Super acceleration](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/)
