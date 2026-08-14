# LTX-2.5 on 4xGB200

## What this runtime contains

This is the native Lightricks LTX-2.5 two-stage MGPU inference pipeline, pinned
and vendored under `ltx_src/`. Both launch arms use the same driver and the same
four GPUs:

| Arm | Stage 1 | Stage 2 | Cache | Compile | SOL Attention |
|---|---|---|---|---|---|
| `dense.toml` | vendor SP4 | 2x2 TDP | off | off | off |
| `fullopt.toml` | CFGP4 | 2x2 TDP | Stage-1 FBCache, threshold 0.08 | per block | off |

The VAE decoder is distributed over the same 2x2 grid in both arms. The public
BF16 x2 latent upsampler sits between the two denoising stages. Stage 1 runs 30
steps; Stage 2 uses sigmas `[0.625, 0.4, 0.0]`, which means two denoising
updates. Seed is 42 and output is 24 FPS.

## Profiles

Select a profile with `--set LTX25_PROFILE=...`; all other model and sampling
settings remain fixed.

| Profile | Stage-1 resolution | Final resolution | Frames | Duration |
|---|---:|---:|---:|---:|
| `default5s` | 768x512 | 1536x1024 | 121 | 5.0 s |
| `4k5s` | 1920x1088 | 3840x2176 | 121 | 5.0 s |
| `1080p20s` | 960x544 | 1920x1088 | 481 | 20.0 s |

The dimensions are the pipeline-valid multiples used by the measured runs;
therefore the 4K profile is 3840x2176 rather than 3840x2160.

## Cache-aware compile

FBCache owns the eager block loop. On every Stage-1 step it executes compiled
block 0, computes the first-block residual signal, and synchronizes the skip
decision across all four CFG ranks with an AND reduction. A cache hit reuses the
previous full-stack video and audio residuals; a miss calls compiled blocks
1-47 and updates them. Stage 2 is explicitly marked and bypasses the cache even
when its per-rank token count happens to match Stage 1.

The compile settings are fixed to:

```text
mode=max-autotune-no-cudagraphs fullgraph=false capture=false
```

This keeps the dynamic cache decision outside compiled blocks and avoids CUDA
Graph output-buffer reuse for residuals retained across diffusion steps. The
Inductor, Triton, and CUDA caches live under `LTX25_COMPILE_CACHE_ROOT`. By
default this resolves below the repository at `.cache/ltx25/`; the path is
independent of hostname and Slurm job.

## Environment

The default interpreter is repository-local:

```text
<repo>/.venv/bin/python
```

The validated environment is present there in the B200 deployment. Its frozen
uv workspace and all four LTX package sources are vendored under
`environment/LTX-2/`; use `setup_env.sh` to rebuild it without another source
checkout. A compatible Diffusers or SGLang environment can be selected with
`--set PYTHON_BIN=/absolute/path/to/python`.

No runtime path points to another repository beside this one. Model weights
remain external read-only assets selected by `LTX25_WEIGHTS_ROOT`.

## Run

Configuration resolution is safe on a login node and does not allocate a GPU:

```bash
cd /home/yitongl/code/agent_deploy/sol-engine
python3 scripts/run.py models/ltx25/GB200/fullopt.toml --print
```

Run inference only inside a four-GPU allocation. For example:

```bash
srun -A nvr_elm_llm --partition interactive --gpus 4 -t 01:00:00 \
  python3 scripts/run.py models/ltx25/GB200/fullopt.toml

srun -A nvr_elm_llm --partition interactive --gpus 4 -t 01:00:00 \
  python3 scripts/run.py models/ltx25/GB200/fullopt.toml \
  --set LTX25_PROFILE=4k5s

srun -A nvr_elm_llm --partition interactive --gpus 4 -t 01:00:00 \
  python3 scripts/run.py models/ltx25/GB200/fullopt.toml \
  --set LTX25_PROFILE=1080p20s
```

Use `dense.toml` in the same commands for the dense control. Each default run
executes the three bundled prompts once as warmup and once as measurement. The
worker fleet stays resident across all six requests.

## Timing and artifacts

The authoritative number is `benchmark.json:request_s_mean`. It is the measured
request wall time after the model is resident and includes prompt encoding,
Stage 1, Stage 2, upsampling, video/audio VAE decoding, and output encoding.
Model load, first-use `torch.compile`, autotuning, and all warmup requests are
excluded. Rank-0 component means, per-request cache hits, and the resolved run
configuration are stored beside it.

Standard artifacts are:

```text
out.mp4
benchmark.json
run_config.json
run.log
timing.<worker-pid>.requests.jsonl
```

## Measured steady-state reference

These are the three-prompt means from the validated 4xGB200 runs, with SOL
Attention disabled:

| Profile | Dense E2E | Fullopt E2E | Speedup |
|---|---:|---:|---:|
| `default5s` | 39.227 s | 8.382 s | **4.680x** |
| `4k5s` | 75.574 s | 38.485 s | **1.964x** |
| `1080p20s` | 92.448 s | 36.692 s | **2.520x** |

The 0.08 cache threshold is validated for this 30-step guidance regime; it is
not claimed to be a universal LTX-2.5 threshold.

## Source boundary

`SOURCE_SNAPSHOT.json` records the public LTX release, optimized source commit,
environment lock, and the two post-commit Stage-1 scoping corrections. Runtime
source, environment package sources, and the default `.venv` are colocated with
this repository. Model weights remain external; generated videos, logs, and
compiler caches stay under ignored repository-local directories. The abandoned
LTX SOL sparse-attention experiment is not included.
