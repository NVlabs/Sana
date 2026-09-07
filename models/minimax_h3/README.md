# MiniMax-H3

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is a 33B audio-video generation model.
Sol-Engine provides hardware-specific full-stack runtimes that combine kernel optimization,
Sol-Attn, caching, and memory-efficient decoding.

## Performance

| Hardware | GPUs | Workload | Baseline (s) | Optimized (s) | Speedup | Config |
|---|---:|---:|---:|---:|---:|---|
| GB200 | 8 | 768p@5s | 27.21 | 6.88 | **3.95x** | [`minimax_h3_fullopt.toml`](../../config/minimax_h3/fullopt.toml) |
| GB10 (DGX Spark) | 1 | 480p@5s | 710.6 | 181.3 | **3.92x** | [`minimax_h3_gb10_fullopt.toml`](../../config/minimax_h3/gb10_fullopt.toml) |
| RTX 4090 | 1 | 768p@5s | 2239.22 | 504.33 | **4.44x** | [`minimax_h3/rtx4090_fullopt.toml`](../../config/minimax_h3/rtx4090_fullopt.toml) |
| RTX 5090 | 1 | 768p@5s | 1045.4 | 231.2 | **4.52x** | [`minimax_h3/rtx5090_fullopt.toml`](../../config/minimax_h3/rtx5090_fullopt.toml) |
| H100 | 4 | 768p@5s | 81.47 | 22.89 | **3.56x** | [`minimax_h3_h100_fullopt.toml`](../../config/minimax_h3/minimax_h3_h100_fullopt.toml) |
| A100 | 4 | 768p@5s | 217.32 | 61.28 | **3.55x** | [`minimax_h3_a100_fullopt.toml`](../../config/minimax_h3/minimax_h3_a100_fullopt.toml) |

Speedups are measured against the matching baseline runtime on the same hardware. Each platform
uses its validated release workload, so the table compares relative acceleration rather than
absolute latency across GPUs.

## Four-step T2V, I2V, and Ref2VA

The standalone [`Sol-H3`](Sol-H3/) runtime covers distilled
four-step T2V, first-frame I2V, and the MiniMax-H3 Ref2VA partition. It supports synchronized video
and audio, 5/10/15-second outputs at 1344x768, Ulysses sequence parallelism over as many as eight
GPUs, and SOL/BSA sparse attention. Ref2VA uses the fast reference-image sizing profile by default
and retains an explicit Diffusers-compatible preprocessing mode.

Validation on 8x NVIDIA B300 covers all three task paths:

| Task | Validation |
|---|---|
| T2V | Warm 8-GPU pipeline medians of 1.653 / 3.732 / 6.612 s for 124 / 243 / 362 frames; see the full 1/4/8-GPU matrix in [`Sol-H3`](Sol-H3/#t2v-benchmark-matrix) |
| I2V | Native first-frame path accepted at 124 frames and in a resident 362-frame T2V-to-I2V regression; no separate I2V headline latency is claimed |
| Ref2VA | Warm pipeline medians of 2.192 / 4.348 / 5.947 s for 124 / 243 / 362 frames, using an 832x1104 image reference and a 125-character / 26-token prompt |

The latency values exclude checkpoint loading, warmup, and MP4 encoding.

## Super acceleration: H3 -> LTX-2.5

The separate [Super Acceleration profile](super_acceleration/) runs MiniMax-H3
Stage 1 on one GB200 and a resident LTX-2.5 Stage 2 Refiner on a second GB200.
A four-GPU node runs two independent pairs; it does not use four-way model,
tensor, or context parallelism. The profile uses a direct BF16 video plus PCM
handoff and full-temporal input-VAE tiling, and deliberately lives outside the
Lightweight/YAML interface.

Formal job `6304303` measured a 6.760544632-second median across 20 hot complete
requests. This is an absolute latency result for a different composite profile,
not another row in the same-profile speedup table above. No matched end-to-end
baseline or perceptual quality gate exists, so no speedup or quality-pass claim
is made.

## Usage

Run the launcher from the repository root and select the full-opt config for your GPU from the
table above:

```bash
# Workstation
python scripts/launch_config.py <config> --mode local

# Slurm cluster
python scripts/launch_config.py <config> \
  --mode sbatch --confirm-submit
```

`local` executes the generated launch script directly. `sbatch` renders and submits a Slurm job;
without `--confirm-submit`, it only prepares the job bundle. The config selects the matching
hardware runtime and enables its complete optimization stack.

For example, the following H100 command reproduces the released
[`demo_prompt`](demo_prompt.json) benchmark at 1344x768 for 5 seconds, using seed `0`
and 50 denoising steps:

```bash
python scripts/launch_config.py \
  config/minimax_h3/minimax_h3_h100_fullopt.toml \
  --mode sbatch --confirm-submit \
  --env H3_STORAGE_ROOT="$PWD"
```

Each run is stored under `runs/` with the generated video, `benchmark.json`, and launch logs.

## Runtime Notes

Platform-specific environment and implementation notes are available for
[GB200](GB200/), [GB10](GB10/), [RTX 4090](RTX4090/), [RTX 5090](RTX5090/),
[H100](H100/), and [A100](A100/).
Full-opt uses approximate caching and sparse attention; keep the released config unchanged when
reproducing the reported performance and quality.
