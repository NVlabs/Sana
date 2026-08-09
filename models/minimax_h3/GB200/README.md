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
[`minimax_h3_fullopt.toml`](../../../transfeat/minimax_h3/fullopt.toml).

## Full-Opt

- **Parallelism:** Ulysses-8 across the packed multimodal sequence.
- **Attention:** Sol-Attn with `tau=1.0`, diagonal thresholding, an exact prefix KV sink, dense
  prefix queries, and the first 10 steps and first two blocks dense.
- **Cache:** FirstBlockCache with threshold `0.08` and a synchronized global skip decision.
- **Runtime:** packed QKV exchange, Triton relayout, AdaLN precomputation, and sharded compiled VAE
  decoding.

## Usage

One command runs any arm. It reproduces the [`demo_prompt`](../demo_prompt.json)
benchmark with seed `0`, 50 denoising steps and the workload above. Run it from
the repository root:

```bash
python3 scripts/run.py models/minimax_h3/GB200/fullopt.toml                 # the optimized arm
python3 scripts/run.py models/minimax_h3/GB200/dense.toml                 # the control it is measured against
```

`scripts/run.py` takes either config dialect -- a flat single-file config or a
transfeat manifest -- and renders the same run bundle under `runs/`:
`launch.sh`, `job.sbatch`, `manifest.resolved.toml`, `metadata.json` and
`outputs/`. Add `--print` to resolve without running, or `--set KEY=VALUE` to
override one value for a single run without editing the config:

```bash
python3 scripts/run.py models/minimax_h3/GB200/fullopt.toml \
  --set PYTHON_BIN=/path/to/venv/bin/python
```

It contains no scheduler. To run under Slurm either put that same command in
your own job script, or call the renderer directly, which is the one thing
`run.py` does not do:

```bash
python3 scripts/launch_transfeat.py models/minimax_h3/GB200/fullopt.toml --mode sbatch --confirm-submit
```

## Environment

- **Runtime:** vendored Diffusers source, Sol-Attn, and NVIDIA CUTLASS DSL 4.5 or newer.
- **Weights:** converted BF16 FL2VA checkpoint supplied through `H3_MODEL_PATH`.
- **Placement:** eight GB200 GPUs in one NVLink domain; no model offload.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
