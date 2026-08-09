# MiniMax-H3 on RTX 5090

## Overview

This runtime runs the released BF16 FL2VA checkpoint on one RTX 5090. The 33B DiT, Qwen3-VL
conditioner, and VAEs use SGLang layerwise component offload so only the active component resides on
the 32 GiB GPU.

## Performance

| GPUs | Workload | E2E speedup |
|---:|---|---:|
| 1 | 1344x768 @ 5 s | **4.52x** |

The speedup is measured against the matching dense runtime. The released configuration is pinned by
[`minimax_h3_rtx5090_fullopt.toml`](../../../candidates/minimax_h3_rtx5090_fullopt.toml).

## Full-Opt

- **Parallelism:** single-GPU execution with layerwise component offload.
- **Attention:** SM120 Sol-Attn with `tau=1.0`, diagonal thresholding, an exact prefix KV sink, dense
  prefix queries, and the first 10 steps and first two blocks dense.
- **Cache:** TeaCache with threshold `0.10`, five retained steps, and one cooldown step.
- **Runtime:** regional `torch.compile`; the complete BF16 video VAE becomes resident only for
  decoding and returns to offload afterward.

## Usage

Run from the repository root:

```bash
python scripts/launch_candidate.py \
  candidates/minimax_h3_rtx5090_fullopt.toml \
  --mode local \
  --env H3_MODEL_PATH=/path/to/MiniMax-H3/FL2VA \
  --env PYTHON_BIN=/path/to/venv/bin/python
```

## Environment

- **Runtime:** PyTorch 2.11.0 with CUDA 13.0, Triton 3.6, and the pinned SGLang checkout.
- **Weights:** released BF16 FL2VA checkpoint supplied through `H3_MODEL_PATH`.
- **Placement:** one RTX 5090 with 32 GiB GPU memory and sufficient host memory for layerwise offload.

The selected environment must provide `ffmpeg` and `ffprobe`. Runtime registration is process-local
and does not modify the installed SGLang checkout.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
