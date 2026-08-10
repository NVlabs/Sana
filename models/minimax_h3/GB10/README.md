# MiniMax-H3 on GB10

## Overview

This single-GPU runtime targets DGX Spark. Because the released BF16 weights exceed the available
unified memory, it uses the pruned FP8 DiT and FP8 Qwen3-VL conditioner while preserving the
MiniMax-H3 FL2VA pipeline.

## Performance

| GPUs | Workload | E2E latency | Speedup |
|---:|---|---:|---:|
| 1 | 832x480 @ 5 s | 710.6 s -> 181.3 s | **3.92x** |

E2E latency is shown as dense -> full-opt. The speedup is measured against the matching unoptimized
GB10 runtime. The released configuration is pinned by
[`minimax_h3_gb10_fullopt.toml`](../../../config/minimax_h3/gb10_fullopt.toml).

## Full-Opt

- **Parallelism:** single-GPU execution without context parallelism.
- **Attention:** Triton Sol-Attn with `tau=1.0`, a 951-token exact prefix KV sink, and the first 10
  steps and first two blocks dense.
- **Cache:** FirstBlockCache with threshold `0.08`.
- **Runtime:** fused QKV, quantization, AdaLN, RoPE, and SwiGLU kernels plus batched VAE tiles.

## Usage

One command runs any arm. It reproduces the [`demo_prompt`](../demo_prompt.json)
benchmark with seed `0`, 50 denoising steps and the workload above. Run it from
the repository root:

```bash
python3 scripts/run.py config/minimax_h3/gb10_fullopt.toml                 # the optimized arm
python3 scripts/run.py config/minimax_h3/gb10_baseline.toml                 # the control it is measured against
```

`scripts/run.py` takes either config dialect -- a flat single-file config or a
config manifest -- and renders the same run bundle under `runs/`:
`launch.sh`, `job.sbatch`, `manifest.resolved.toml`, `metadata.json` and
`outputs/`. Add `--print` to resolve without running, or `--set KEY=VALUE` to
override one value for a single run without editing the config:

```bash
python3 scripts/run.py config/minimax_h3/gb10_fullopt.toml \
  --set PYTHON_BIN=/path/to/venv/bin/python
```

It contains no scheduler. To run under Slurm either put that same command in
your own job script, or call the renderer directly, which is the one thing
`run.py` does not do:

```bash
python3 scripts/launch_config.py config/minimax_h3/gb10_fullopt.toml --mode sbatch --confirm-submit
```

## Environment

- **Runtime:** PyTorch 2.11.0 with CUDA 13.0 and Triton 3.6.
- **Weights:** pruned FP8 DiT, FP8 Qwen3-VL conditioner, and released MiniMax-H3 VAEs.
- **Placement:** one GB10 with unified CPU/GPU memory; keep only one model process resident.

Set `HF_HOME` to the checkpoint root and `H3_DIFFUSERS_SRC` when the pinned Diffusers source is
stored outside this repository.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
