# MiniMax-H3 on GB10

## Overview

This single-GPU runtime targets DGX Spark. Because the released BF16 weights exceed the available
unified memory, it uses the pruned FP8 DiT and FP8 Qwen3-VL conditioner while preserving the
MiniMax-H3 FL2VA pipeline.

## Performance

| GPUs | Workload | E2E speedup |
|---:|---|---:|
| 1 | 832x480 @ 5 s | **3.92x** |

The speedup is measured against the matching unoptimized GB10 runtime. The released configuration
is pinned by [`minimax_h3_gb10_fullopt.toml`](../../../candidates/minimax_h3_gb10_fullopt.toml).

## Full-Opt

- **Parallelism:** single-GPU execution without context parallelism.
- **Attention:** Triton Sol-Attn with `tau=1.0`, a 951-token exact prefix KV sink, and the first 10
  steps and first two blocks dense.
- **Cache:** FirstBlockCache with threshold `0.08`.
- **Runtime:** fused QKV, quantization, AdaLN, RoPE, and SwiGLU kernels plus batched VAE tiles.

## Usage

The command below reproduces the
[`demo_prompt`](../demo_prompt.json) benchmark with seed `0`, 50 denoising steps, and
the 5-second workload shown above. Run it from the repository root:

```bash
python scripts/launch_candidate.py \
  candidates/minimax_h3_gb10_fullopt.toml \
  --mode local \
  --env PYTHON_BIN=/path/to/venv/bin/python
```

## Environment

- **Runtime:** PyTorch 2.11.0 with CUDA 13.0 and Triton 3.6.
- **Weights:** pruned FP8 DiT, FP8 Qwen3-VL conditioner, and released MiniMax-H3 VAEs.
- **Placement:** one GB10 with unified CPU/GPU memory; keep only one model process resident.

Set `HF_HOME` to the checkpoint root and `H3_DIFFUSERS_SRC` when the pinned Diffusers source is
stored outside this repository.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
