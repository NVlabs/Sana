# Wan 2.1 720p on B200 with the SM100 colmask kernel

This document records the minimum framework integration and the completed
Wan 2.1 T2V experiment for the CuTeDSL colmask kernel in
`release/sm100-sol_attn-colmask`.

## Outcome

Slurm job `5425361` completed successfully on 2026-07-16. The Wan attention
shape was `[1, 40, 75600, 128]`; the sparse phase used the requested SM100
CuTeDSL kernel and produced a valid MP4.

After packaging the integration into this branch, Slurm job `5440265`
validated the relocated code directly on B200. All ten CPU contract tests
passed, the 8192-token real kernel gate passed, the following forward hit the
compiled-op cache, and the output was finite.

| Item | Result |
|---|---:|
| GPU | NVIDIA GB200/B200, SM100, compute capability 10.0 |
| Wan model | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| Output | 1280x720, 81 frames, 50 denoise steps |
| Seed | 37 |
| Target route density | 0.15 |
| Measured route density | 0.15004073083400726 |
| Calibrated tau | 1.501953125 |
| Real-QKV correctness gate | pass |
| Gate max absolute error | 0.015625 (limit 0.08) |
| Gate mean absolute error | 1.7737108919391176e-6 (limit 0.01) |
| Gate relative L2 | 0.00011463081318652257 (limit 0.01) |
| Full-shape CuTeDSL compile | 17.0530 s |
| Sparse steady-state denoise step | approximately 8.46 s |
| End-to-end command time | 577.09 s |
| Maximum process RSS | 106004928 KiB |

The branch-validation smoke (`5440265`) reported max absolute error
0.00048828125, mean absolute error 1.579303716425784e-5, relative L2
0.0025860415771603584, compile time 19.2490 seconds, and total smoke time
31.8258 seconds. Raw smoke evidence is under
`evidence/wan-b200/job-5440265/`.

The generated MP4 SHA256 was
`cb6ad69e520d25096a49ac369a0dd12f281fc3a1b8cb2f6fd1e7f7493534366b`.
The video itself is not stored in Git.

## K/R/W contract

### Kernel contract (K)

- contiguous BF16 Q/K/V in BHTD layout;
- equal Q/K/V shape, head dimension 128, block size 64;
- non-causal attention on an SM100 compute capability 10.0 GPU;
- BF16 KC/VC summaries and FP32 global threshold prepared with the release
  semantics;
- logical routing group size 256 over physical route tiles of 128;
- no substitution with the older INT8 Sparse-VideoGen SOL Attention path.

### Correctness reference (R)

The first real Wan sparse QKV invokes a one-batch, one-head gate at shape
`[1, 1, 75600, 128]`. It compares the CuTeDSL output against the release's
prepared Triton reference and rejects the run if any of these limits is
exceeded: max absolute error 0.08, mean absolute error 0.01, relative L2 0.01.
The gate also records the release kernel SHA256.

### Workload (W)

The exact workload is checked in as
`workloads/wan2.1-t2v-14b-720p81-sm100-colmask.json`. The first transformer
layer and first ten denoise steps use dense attention. Sparse steps use
Morton3D video-token reorder and target 15% route density.

## Repository contents

- `integrations/wan/adapter.py`: BF16 preparation, density calibration,
  real-QKV gate, CuTeDSL launch, and compiled-op cache.
- `integrations/wan/patches/sparse-videogen-pisa-bidirectional.patch`: minimal
  Sparse-VideoGen selection/CLI patch.
- `integrations/wan/smoke.py`: standalone B200 compile and correctness smoke.
- `integrations/wan/scripts/run_smoke_b200.sbatch`: smoke job template.
- `integrations/wan/scripts/run_wan_720p_b200.sbatch`: canonical HSG runner.
- `evidence/wan-b200/job-5425361/`: machine-readable result and raw adapter
  events from the completed run.

## Reproduce

Start from the exact release and Sparse-VideoGen base revisions:

```bash
git clone git@github.com:hp-l33/Sol-Attn.git
cd Sol-Attn
git checkout experiment/wan-b200-sm100-colmask-20260717

git clone <Sparse-VideoGen-remote> ../Sparse-VideoGen-wan-sm100
git -C ../Sparse-VideoGen-wan-sm100 checkout 91efef6
git -C ../Sparse-VideoGen-wan-sm100 apply \
  "$PWD/integrations/wan/patches/sparse-videogen-pisa-bidirectional.patch"
```

The validated software environment was Python 3.12.13, PyTorch
2.11.0+cu128, CUDA 12.8, Triton 3.7.0, cuda-python, and the NVIDIA CUTLASS
CuTe DSL package. Validate the CPU-side package contracts first:

```bash
PYTHONPATH=. python -m pytest tests/
shasum -a 256 --check SHA256SUMS
```

Run the standalone B200 smoke:

```bash
sbatch --export=ALL,\
SOL_REPO="$PWD",\
OUT_ROOT="$PWD/outputs/wan-smoke",\
ENV_ACTIVATE=/path/to/activate-wan-env.sh \
integrations/wan/scripts/run_smoke_b200.sbatch
```

Run the canonical HSG case:

```bash
sbatch --export=ALL,\
SOL_REPO="$PWD",\
WAN_REPO="$(realpath ../Sparse-VideoGen-wan-sm100)",\
OUT_ROOT=/path/to/outputs/wan-720p-sm100,\
ENV_ACTIVATE=/path/to/activate-wan-env.sh \
integrations/wan/scripts/run_wan_720p_b200.sbatch
```

Set `WAN_FAST_KERNELS` to a directory containing compatible Wan RoPE/Norm
extensions if available. Model downloads can be redirected with the normal
Hugging Face cache environment variables.

## Compile cache

Constructing the public CuTeDSL runner compiled for roughly 18-20 seconds in
the smoke environment. Wan calls attention many times, so the adapter caches
the compiled operation by `(device, T, B, H)` and passes fresh tensor pointers
on every launch. In the validation smoke, the first call was a cache miss and
the immediately following forward was a cache hit. Removing this cache makes
the end-to-end integration impractical.

## Provenance

The completed run used:

- Sol-Attn commit `fc0d18eb58b531a9b4072a53479afe702491ebff`;
- release kernel SHA256
  `e4e47b7e5fc2015b41e4462507372651e1f6eaf05ee7ddd54af3cac1301f283b`;
- Sparse-VideoGen integration commit
  `82dd1d9029143d7264080e59dff0fe028c8dda34`;
- full Wan Slurm job `5425361`;
- packaged-branch B200 smoke job `5440265`.

The adapter in this experiment branch is the same implementation relocated
into Sol-Attn and changed only to use package-relative repository discovery.
The Sparse-VideoGen patch imports it from `integrations.wan`. Job `5440265`
validates this relocated form rather than only the earlier framework-local
copy.

## Limitations and interpretation

- The HSG QOS allocated a four-GPU GB200 node, but the process set
  `CUDA_VISIBLE_DEVICES=0` and used one GPU only.
- The Wan RoPE/Norm extension did not import in job `5425361`; those operators
  fell back to PyTorch. The 577.09-second end-to-end time must therefore not be
  compared directly with runs that enabled the fast RoPE/Norm extensions.
- The recorded end-to-end time includes model load, CuTeDSL compilation,
  denoising, decode, and video export. It is not a kernel-only benchmark.
- Density calibration samples batch 0 and two heads. It is deterministic for
  a fixed QKV tensor, but it is a routing-density calibration rather than a
  quality guarantee.
