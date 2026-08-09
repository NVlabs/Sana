# MiniMax-H3

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is a 33B audio-video generation model.
Sol-Engine provides hardware-specific full-stack runtimes that combine kernel optimization,
Sol-Attn, caching, and memory-efficient decoding.

## Performance

| Hardware | GPUs | Workload | End-to-end speedup | Full-opt candidate |
|---|---:|---:|---:|---|
| GB200 | 8 | 1344x768 @ 5 s | **3.97x** | [`minimax_h3_fullopt.toml`](../../candidates/minimax_h3_fullopt.toml) |
| GB10 (DGX Spark) | 1 | 832x480 @ 5 s | **3.92x** | [`minimax_h3_gb10_fullopt.toml`](../../candidates/minimax_h3_gb10_fullopt.toml) |
| RTX 5090 | 1 | 1344x768 @ 5 s | **4.52x** | [`minimax_h3_rtx5090_fullopt.toml`](../../candidates/minimax_h3_rtx5090_fullopt.toml) |
| H100 | 4 | 1344x768 @ 5 s | **3.56x** | [`minimax_h3_h100_fullopt_exact.toml`](../../candidates/minimax_h3_h100_fullopt_exact.toml) |
| A100 | 4 | 1344x768 @ 5 s | **3.55x** | [`minimax_h3_a100_fullopt_exact.toml`](../../candidates/minimax_h3_a100_fullopt_exact.toml) |

Speedups are measured against the matching dense runtime on the same hardware. Each platform uses
its validated release workload, so the table compares relative acceleration rather than absolute
latency across GPUs.

## Usage

Run the launcher from the repository root and select the full-opt candidate for your GPU from the
table above:

```bash
# Workstation
python scripts/launch_candidate.py <candidate> --mode local

# Slurm cluster
python scripts/launch_candidate.py <candidate> \
  --mode sbatch --confirm-submit
```

`local` executes the generated launch script directly. `sbatch` renders and submits a Slurm job;
without `--confirm-submit`, it only prepares the job bundle. The candidate selects the matching
hardware runtime and enables its complete optimization stack.

For example, the following H100 command reproduces the released
[`t2va_example_1`](prompts/t2va_example_1.json) benchmark at 1344x768 for 5 seconds, using seed `0`
and 50 denoising steps:

```bash
python scripts/launch_candidate.py \
  candidates/minimax_h3_h100_fullopt_exact.toml \
  --mode sbatch --confirm-submit \
  --env H3_STORAGE_ROOT="$PWD"
```

Each run is stored under `runs/` with the generated video, `benchmark.json`, and launch logs.

## Runtime Notes

Platform-specific environment and implementation notes are available for
[GB200](gb200/), [GB10](gb10/), [RTX 5090](rtx5090/), [H100](h100/), and [A100](a100/).
Full-opt uses approximate caching and sparse attention; keep the released candidate unchanged when
reproducing the reported performance and quality.
