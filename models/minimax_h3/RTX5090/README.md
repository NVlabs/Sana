# MiniMax-H3 on RTX 5090

## Overview

This runtime runs the released BF16 FL2VA checkpoint on one RTX 5090. The 33B DiT, Qwen3-VL
conditioner, and VAEs use SGLang layerwise component offload so only the active component resides on
the 32 GiB GPU.

## Performance

| GPUs | Workload | Baseline (s) | Optimized (s) | Speedup |
|---:|---|---:|---:|---:|
| 1 | 768p@5s | 1045.4 | 231.2 | **4.52x** |

The speedup is measured against the matching baseline runtime. The released configuration is pinned
by [`minimax_h3_rtx5090_fullopt.toml`](../../../config/minimax_h3/rtx5090_fullopt.toml).

## Full-Opt

- **Parallelism:** single-GPU execution with layerwise component offload.
- **Attention:** SM120 Sol-Attn with `tau=1.0`, diagonal thresholding, an exact prefix KV sink, dense
  prefix queries, and the first 10 steps and first two blocks dense.
- **Cache:** TeaCache with threshold `0.10`, five retained steps, and one cooldown step.
- **Runtime:** regional `torch.compile`; the complete BF16 video VAE becomes resident only for
  decoding and returns to offload afterward.

## Usage

One command runs any arm. It reproduces the [`demo_prompt`](../demo_prompt.json)
benchmark with seed `0`, 50 denoising steps and the workload above. Run it from
the repository root:

```bash
python3 scripts/run.py config/minimax_h3/rtx5090_fullopt.toml                 # the optimized arm
python3 scripts/run.py config/minimax_h3/rtx5090_dense.toml                 # the control it is measured against
```

`scripts/run.py` takes either config dialect -- a flat single-file config or a
config manifest -- and renders the same run bundle under `runs/`:
`launch.sh`, `job.sbatch`, `manifest.resolved.toml`, `metadata.json` and
`outputs/`. Add `--print` to resolve without running, or `--set KEY=VALUE` to
override one value for a single run without editing the config:

```bash
python3 scripts/run.py config/minimax_h3/rtx5090_fullopt.toml \
  --set PYTHON_BIN=/path/to/venv/bin/python
```

It contains no scheduler. To run under Slurm either put that same command in
your own job script, or call the renderer directly, which is the one thing
`run.py` does not do:

```bash
python3 scripts/launch_config.py config/minimax_h3/rtx5090_fullopt.toml --mode sbatch --confirm-submit
```

## Environment

- **Runtime:** PyTorch 2.11.0 with CUDA 13.0, Triton 3.6, and the pinned SGLang checkout.
- **Weights:** released BF16 FL2VA checkpoint supplied through `H3_MODEL_PATH`.
- **Placement:** one RTX 5090 with 32 GiB GPU memory and sufficient host memory for layerwise offload.

The selected environment must provide `ffmpeg` and `ffprobe`. Runtime registration is process-local
and does not modify the installed SGLang checkout.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
