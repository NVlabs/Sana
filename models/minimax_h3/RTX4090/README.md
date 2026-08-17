# MiniMax-H3 on RTX 4090

## Overview

This runtime runs the released BF16 FL2VA checkpoint on one RTX 4090. The 33B DiT, Qwen3-VL
conditioner, and VAEs use SGLang layerwise component offload so only the active component resides on
the 24 GiB GPU.

## Performance

| GPUs | Workload | Baseline (s) | Optimized (s) | Speedup |
|---:|---|---:|---:|---:|
| 1 | 768p@5s | 2239.22 | 504.33 | **4.44x** |

The speedup is one warm end-to-end timing sample against the matching dense runtime on the same
GPU. Both arms use the released BF16 FL2VA weights, prompt, seed, 50 measured denoising steps, and
layerwise component offload. The released full-opt configuration is pinned by
[`minimax_h3/rtx4090_fullopt.toml`](../../../config/minimax_h3/rtx4090_fullopt.toml).

The controlled attribution experiment held the optimized runtime, 50-step warmup, prompt, seed,
and measured workload fixed:

| Controlled arm | TeaCache | Sol-Attn | E2E time (s) | Incremental speedup | Cumulative speedup |
|---|---:|---:|---:|---:|---:|
| lossless opt | off | off | 1951.27 | 1.00x | 1.00x |
| + Cache | on | off | 613.71 | **3.18x** | **3.18x** |
| + Cache + Sol-Attn | on | on | 504.33 | **1.22x** | **3.87x** |

Here, **lossless opt** means that neither TeaCache reuse nor Sol-Attn sparsification is enabled. It
does not claim bitwise or video-quality equivalence. No perceptual or embedding similarity metric
was measured, and the timings have not been repeated to characterize variance.

## Full-Opt

- **Parallelism:** single-GPU execution with layerwise component offload.
- **Attention:** SM89 Sol-Attn with `tau=1.0`, diagonal thresholding, an exact prefix KV sink, dense
  prefix queries, and the first 10 steps and first two blocks dense.
- **Cache:** TeaCache with threshold `0.10`, five retained steps, and one cooldown step.
- **Runtime:** regional `torch.compile`; the complete BF16 video VAE becomes resident only for
  decoding and returns to offload afterward.

## Usage

The released config reproduces the [`demo_prompt`](../demo_prompt.json) full-opt benchmark with
seed `0`, 50 denoising steps, and the workload above. Run it from the repository root:

```bash
python3 scripts/run.py config/minimax_h3/rtx4090_fullopt.toml
```

`scripts/run.py` takes either config dialect -- a flat single-file config or a
config manifest -- and renders the same run bundle under `runs/`:
`launch.sh`, `job.sbatch`, `manifest.resolved.toml`, `metadata.json` and
`outputs/`. Add `--print` to resolve without running, or `--set KEY=VALUE` to
override one value for a single run without editing the config:

```bash
python3 scripts/run.py config/minimax_h3/rtx4090_fullopt.toml \
  --set PYTHON_BIN=/path/to/venv/bin/python
```

It contains no scheduler. To run under Slurm either put that same command in
your own job script, or call the renderer directly, which is the one thing
`run.py` does not do:

```bash
python3 scripts/launch_config.py config/minimax_h3/rtx4090_fullopt.toml --mode sbatch --confirm-submit
```

## Environment

- **Runtime:** PyTorch 2.11 with CUDA 12.8, Triton 3.6, and the pinned SGLang checkout.
- **Weights:** released BF16 FL2VA checkpoint supplied through `H3_MODEL_PATH`.
- **Placement:** one RTX 4090 with 24 GiB GPU memory and sufficient host memory for layerwise offload.

The selected environment must provide `ffmpeg` and `ffprobe`. Runtime registration is process-local
and does not modify the installed SGLang checkout.

## Validation

- Sana commit: `6fb7eb11c3435555ec6d6adf0d5572d339d2c6eb`.
- SGLang commit: `6fa3f9df11c8bdbc0e3b4ddc87a3d873343aca72`.
- First sparse forward: `[1, 38247, 56, 128]` at 19.65% effective block density.
- Real-QKV gate: maximum absolute error `0.0625`, mean absolute error `0.000289`, and relative L2
  `0.000609`; all passed their declared limits.
- TeaCache measured 14 block-stack computations and 35 reuses over 49 decisions.

The timing and correctness evidence was collected with the same runtime stack before it was split
into this RTX4090-named package. The hardware-specific package fixes the profile name, SM89 cache
paths, capability check, and benchmark device metadata without changing the attention or TeaCache
implementations.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
