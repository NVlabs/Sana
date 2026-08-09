# MiniMax-H3 on A100

## Overview

This self-contained runtime runs the released BF16 FL2VA checkpoint on four A100 GPUs. It uses
SGLang FSDP inference with Ulysses-4, keeps the model resident without offload, and does not modify
the installed SGLang checkout.

## Performance

| GPUs | Workload | E2E speedup |
|---:|---|---:|
| 4 | 1344x768 @ 5 s | **3.55x** |

The speedup is measured against the matching dense runtime. The released configuration is pinned by
[`minimax_h3_a100_fullopt_exact.toml`](../../../candidates/minimax_h3_a100_fullopt_exact.toml).

## Full-Opt

- **Parallelism:** FSDP inference with Ulysses-4 and no model offload.
- **Attention:** Triton Sol-Attn with `tau=1.0`, exact thresholding, a full-prefix KV sink, dense
  prefix queries, and the first 10 steps and first two blocks dense.
- **Cache:** FirstBlockCache with threshold `0.08` and synchronized decisions across ranks.
- **Runtime:** pinned SGLang BF16 execution without `torch.compile` or token reordering.

## Usage

One command runs any arm. It reproduces the [`demo_prompt`](../demo_prompt.json)
benchmark with seed `0`, 50 denoising steps and the workload above. Run it from
the repository root:

```bash
python3 scripts/run.py candidates/minimax_h3_a100_fullopt_exact.toml                 # the optimized arm
python3 scripts/run.py candidates/minimax_h3_a100_dense.toml                 # the control it is measured against
```

`scripts/run.py` takes either config dialect -- a flat single-file config or a
candidate manifest -- and renders the same run bundle under `runs/`:
`launch.sh`, `job.sbatch`, `manifest.resolved.toml`, `metadata.json` and
`outputs/`. Add `--print` to resolve without running, or `--set KEY=VALUE` to
override one value for a single run without editing the config:

```bash
python3 scripts/run.py candidates/minimax_h3_a100_fullopt_exact.toml \
  --set H3_STORAGE_ROOT=/shared/path/Sana
```

It contains no scheduler. To run under Slurm either put that same command in
your own job script, or call the renderer directly, which is the one thing
`run.py` does not do:

```bash
python3 scripts/launch_candidate.py candidates/minimax_h3_a100_fullopt_exact.toml --mode sbatch --confirm-submit
```

## Environment

- **Runtime:** `lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86`, PyTorch 2.11.0, and Triton 3.6.
- **Weights:** released BF16 FL2VA checkpoint; set `H3_MODEL_PATH` for an offline local copy.
- **Placement:** four A100 GPUs with shared access to `H3_STORAGE_ROOT`.

The launcher supports Pyxis, Apptainer/Singularity, and native execution. Site-specific Slurm account
and partition settings remain external to the candidate.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
