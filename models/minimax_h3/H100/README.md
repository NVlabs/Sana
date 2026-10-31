# MiniMax-H3 on H100

## Overview

This self-contained runtime runs the released BF16 FL2VA checkpoint on four H100 GPUs. It uses
SGLang FSDP inference with Ulysses-4, keeps the model resident without offload, and does not modify
the installed SGLang checkout.

## Performance

| GPUs | Workload | Baseline (s) | Optimized (s) | Speedup |
|---:|---|---:|---:|---:|
| 4 | 768p@5s | 81.47 | 22.89 | **3.56x** |

The speedup is measured against the matching baseline runtime. The released configuration is pinned
by [`minimax_h3_h100_fullopt.toml`](../../../config/minimax_h3/minimax_h3_h100_fullopt.toml).

## Full-Opt

- **Parallelism:** FSDP inference with Ulysses-4 and no model offload.
- **Attention:** CuTe SM90 Sol-Attn with `tau=1.0`, exact thresholding, a full-prefix KV sink, dense
  prefix queries, and the first 10 steps and first two blocks dense.
- **Cache:** FirstBlockCache with threshold `0.08` and synchronized decisions across ranks.
- **Runtime:** pinned SGLang BF16 execution without `torch.compile` or token reordering.

## Usage

One command runs any arm. It reproduces the [`demo_prompt`](../demo_prompt.json)
benchmark with seed `0`, 50 denoising steps and the workload above. Run it from
the repository root:

```bash
python3 scripts/run.py config/minimax_h3/minimax_h3_h100_fullopt.toml                 # the optimized arm
python3 scripts/run.py config/minimax_h3/h100_dense.toml                 # the control it is measured against
```

`scripts/run.py` takes either config dialect -- a flat single-file config or a
config manifest -- and renders the same run bundle under `runs/`:
`launch.sh`, `job.sbatch`, `manifest.resolved.toml`, `metadata.json` and
`outputs/`. Add `--print` to resolve without running, or `--set KEY=VALUE` to
override one value for a single run without editing the config:

```bash
python3 scripts/run.py config/minimax_h3/minimax_h3_h100_fullopt.toml \
  --set H3_STORAGE_ROOT=/shared/path/Sana
```

It contains no scheduler. To run under Slurm either put that same command in
your own job script, or call the renderer directly, which is the one thing
`run.py` does not do:

```bash
python3 scripts/launch_config.py config/minimax_h3/minimax_h3_h100_fullopt.toml --mode sbatch --confirm-submit
```

## Environment

- **Runtime:** `lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86`, PyTorch 2.11.0, and Triton 3.6.
- **Weights:** released BF16 FL2VA checkpoint; set `H3_MODEL_PATH` for an offline local copy.
- **Placement:** four H100 GPUs with shared access to `H3_STORAGE_ROOT`.

The launcher supports Pyxis, Apptainer/Singularity, and native execution. Site-specific Slurm account
and partition settings remain external to the config.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.

## Sol-Engine runtime without SGLang (`h3_h100.py`)

The GB200 stack (fused bf16 kernels, AdaLN precompute, Sol-Attn, sharded VAE decode) on N x 80 GB GPUs, no step
cache. The DiT blocks and the Qwen3-VL decoder layers are sharded 1/N per rank and rebuilt per layer by an
all-gather that hides behind compute (`shard.py`); the block stack runs Ulysses over the packed sequence
(`forward_mp.py`). On one GPU the same layers stay in pinned host memory and are copied in per layer behind
compute instead (offload mode, automatic at `--nproc_per_node 1`). Dense attention on Hopper uses
FlashAttention-4's sm90 kernel. Run it from the sol-engine image
with this directory next to `../GB200`, on the converted diffusers checkpoint:

```bash
torchrun --standalone --nproc_per_node 4 h3_h100.py --model-path /models/MiniMax-H3-diffusers \
  --prompt-file ../demo_prompt.json --attention sol      # or dense
```

768p, 5 s, 50 steps, seed 0, warm, one H100 node (seconds; quality = PSNR vs this runtime's dense output):

| runtime | GPUs | time | vs SGLang |
|---|---:|---:|---:|
| SGLang 0.5.22, TP2 x Ulysses2 (its H100 recipe) | 4 | 77.53 | 1.00x |
| `h3_h100.py --attention dense` | 4 | 78.15 | 0.99x |
| `h3_h100.py --attention sol` (19.67 dB) | 4 | 61.11 | 1.27x |
| SGLang 0.5.22, `--dit-layerwise-offload --text-encoder-cpu-offload` | 1 | 278.23 | 1.00x |
| `h3_h100.py --attention dense`, offload | 1 | 270.95 | 1.03x |
| `h3_h100.py --attention sol`, offload (19.67 dB) | 1 | 204.65 | 1.36x |

The 4-GPU rows share one node, the 1-GPU rows another. Between the two exact runtimes (this dense vs SGLang)
the PSNR is 19.31 dB (4 GPU) / 17.66 dB (1 GPU); this dense at 1 vs 4 GPUs: 42.5 dB. Peak 67.3 GiB per GPU at 4 GPUs
(first request, before the AdaLN weights are freed), 47.9 GiB after; 43.8 / 27.3 GiB at 1 GPU. The pinned nightly image above is no longer on
Docker Hub (404 as of 2026-10-30).
