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
[`demo_prompt`](../demo_prompt.json) benchmark with seed `0`, 50 denoising steps, and
the 5-second workload shown above. Run it from the repository root:

```bash
python scripts/launch_candidate.py \
  candidates/minimax_h3_fullopt.toml \
  --mode sbatch --confirm-submit \
  --env H3_MODEL_PATH=/path/to/MiniMax-H3-diffusers \
  --env PYTHON_BIN=/path/to/python
```

### Single-file launch

The two `.toml` files beside this README are self-contained launch configs for
the same two arms — no second profile, no `--env` arguments, no scheduler:

```bash
python3 scripts/run.py models/minimax_h3/gb200/baseline.toml     # dense reference
python3 scripts/run.py models/minimax_h3/gb200/optimized.toml    # full stack
```

Point `PYTHON_BIN` and `H3_MODEL_PATH` at your install by editing the config, and
add `--print` to resolve without running. Under a scheduler, put that same line
in your own job script; see [simple-launch](../../../docs/simple-launch.md).

`optimized.toml` runs Ulysses-4 on one node rather than the Ulysses-8 in
`minimax_h3_fullopt.toml`, because an NVL72 node exposes 4 GPUs and degree 8
needs two of them.

## Environment

- **Runtime:** vendored Diffusers source, Sol-Attn, and NVIDIA CUTLASS DSL 4.5 or newer.
- **Weights:** converted BF16 FL2VA checkpoint supplied through `H3_MODEL_PATH`.
- **Placement:** eight GB200 GPUs in one NVLink domain; no model offload.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
