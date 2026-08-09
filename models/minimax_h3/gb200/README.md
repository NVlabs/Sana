# MiniMax-H3 on GB200

## Overview

This runtime runs the released BF16 FL2VA checkpoint on eight GB200 GPUs. MiniMax-H3 packs text,
audio, conditioning video, and target video into one sequence, so the runtime distributes the
sequence with Ulysses-8 and keeps the complete model resident without offload.

## Performance

| GPUs | Workload | E2E speedup |
|---:|---|---:|
| 8 | 1344x768 @ 5 s | **3.97x** |

The speedup is measured against the matching dense runtime. The released configuration is pinned by
[`minimax_h3_fullopt.toml`](../../../candidates/minimax_h3_fullopt.toml).

## Full-Opt

- **Parallelism:** Ulysses-8 across the packed multimodal sequence.
- **Attention:** Sol-Attn with `tau=1.0`, diagonal thresholding, an exact prefix KV sink, dense
  prefix queries, and the first 10 steps and first two blocks dense.
- **Cache:** FirstBlockCache with threshold `0.08` and a synchronized global skip decision.
- **Runtime:** packed QKV exchange, Triton relayout, AdaLN precomputation, and sharded compiled VAE
  decoding.

## Usage

The command below reproduces the
[`t2va_example_1`](../prompts/t2va_example_1.json) benchmark with seed `0`, 50 denoising steps, and
the 5-second workload shown above. Run it from the repository root:

```bash
python scripts/launch_candidate.py \
  candidates/minimax_h3_fullopt.toml \
  --mode sbatch --confirm-submit \
  --env H3_MODEL_PATH=/path/to/MiniMax-H3-diffusers \
  --env PYTHON_BIN=/path/to/python
```

## Environment

- **Runtime:** vendored Diffusers source, Sol-Attn, and NVIDIA CUTLASS DSL 4.5 or newer.
- **Weights:** converted BF16 FL2VA checkpoint supplied through `H3_MODEL_PATH`.
- **Placement:** eight GB200 GPUs in one NVLink domain; no model offload.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
