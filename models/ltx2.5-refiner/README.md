# LTX-2.5 Stage-2 Refiner + TAEHV wide on 4xGB200

This model directory is the GB200 implementation of the delivered standalone
LTX-2.5 Stage-2 refiner. One invocation processes one 1080p, 10-second input
video with four GB200 GPUs. It uses the Sol-Engine SM100 attention backend
already present in this repository and the repository's existing root
`.venv`; it does not depend on an adjacent checkout or create another Python
environment.

The H100 handoff's sampling and attention policy is preserved exactly. The two
intentional hardware changes are the requested four-way head/context-parallel
execution and dispatch from the former SM90 kernel to this repository's
GB200/SM100 kernel.

## Fixed inference contract

| Area | Fixed value |
|---|---|
| Workload | one input video; standalone Stage 2 only |
| Hardware | exactly 4xGB200 on one node |
| Output geometry | 1920x1088, 241 frames, 24 FPS |
| Nominal duration | 10.04 s by `frames / FPS`; 10.00 s between first and last frame |
| Batch | 1 |
| Parallelism | 4-way head/context parallel; full parameter replica on every rank |
| Precision | BF16 |
| Stage-2 sigmas | `[0.909375, 0.725, 0.421875, 0.0]` (3 denoising updates) |
| Refiner LoRA | distilled LoRA, strength `0.8` |
| Stage-2 tau | step 0: `1.0`; step 1: `1.25`; step 2: `1.5` |
| Self-attention layer 0 | dense at every step |
| Self-attention layers 1-47 | in-repository Sol-Attn SM100 kernel |
| Cross-attention | dense |
| Sol selection | `thresh_type=diag`, `kv_splits=auto`, strict mode enabled |
| TAEHV temporal execution | sequential memblock path for this workload |
| Eager control | `refiner.toml`: `torch.compile=0` |
| Compile arm | `refiner_compile.toml`: per-block Inductor compile; stateful Sol callable remains eager |
| Disabled in both arms | algorithm cache, quantization, offload, sink, reorder |
| Residency | all inference modules remain GPU-resident after preload |

The transformer parameters are replicated on all four ranks. Within each
self-attention operation, heads are partitioned four ways and communication
makes the complete token sequence visible to every local head before it invokes
the dense or Sol kernel. The partial-head outputs are then communicated back
into the distributed activation layout. Cross-attention remains dense.

Each rank must report 3 completed steps, 3 dense layer-0 executions, 141 Sol
calls, and 141 SM100 kernel calls. The layer-0 count is inferred when execution
reaches layer 1; the 141 sparse calls are separately proven by the backend's
actual kernel counter. Across all four ranks the expected totals are 12 dense
layer-0 executions and 564 Sol/kernel calls. A run that falls back from the
SM100 kernel is a failure, even if it still produces a video.

For 1920x1088 and 241 frames, the half-resolution TAEHV input contains
377,579,520 elements. That exceeds the upstream 100,000,000-element heuristic,
so encode and decode use the sequential temporal memblock path. This changes
activation scheduling only; it does not offload model weights.

## Source and weight provenance

The TAEHV implementation is vendored at
`GB200/vendor/taehv/taehv.py` from the official repository:

```text
repository: https://github.com/madebyollin/taehv
branch:     2026_03_11_taeltx23_wide
commit:     32ac0146b11007cda5a57b60a3b35653361fb8a4
source SHA: 607c2a578bc2684e6cd21e96f8c1d024b1c32912e6f58c724f14131a3d4a2773
weight SHA: 007788e6b9cb7f77e8589ae30ba7456b119d38b0d017e1d349c1c1d11e3d6339
license:    MIT
```

`GB200/vendor/taehv/SOURCE.json` is the machine-readable record. The official
weight is an external, shared, read-only model asset and is never committed to
Git. Its permanent path is:

```text
/lustre/fsw/portfolios/nvr/users/yitongl/pretrained_models/LTX-2.5-public/taehv/taeltx2_3_wide.pth
```

The original handoff pinned Sol-Engine commit
`5dd502af9938d924be206c332ad1e911b4a925a1`. This integration uses the
compatible current Sol-Engine code in this repository, including its native
SM100 implementation; it does not use an external `SOL_ENGINE_ROOT`.

TAEHV is an approximate autoencoder. It substantially reduces autoencoder
cost, but its detail and contrast are not expected to be perceptually identical
to the full LTX video VAE.

## Environment

Use the existing repository environment only:

```text
/home/yitongl/code/agent_deploy/sol-engine/.venv/bin/python
```

The checked GB200 deployment has Python 3.13.11, PyTorch 2.11.0+cu130,
Triton 3.6.0, and `nvidia-cutlass-dsl` 4.7.0. The launcher also uses the
in-repository LTX source under `models/ltx25/GB200/ltx_src`. Neither the setup
script nor the inference launcher runs `pip`, `uv`, or creates a venv.

The in-repository Sol helper detects the installed MLIR `nvvm.fmax` callable
signature directly. This is required because CUTLASS DSL 4.7 reports a CUDA
12.9 toolchain while exposing the newer two-positional-argument binding; CUDA
version alone is not a reliable API discriminator.

The split LTX-2.5 assets are read from:

```text
/lustre/fsw/portfolios/nvr/users/yitongl/pretrained_models/LTX-2.5-public
```

That root must contain the BF16 transformer, Gemma4 projection text encoder,
video-VAE latent statistics, x2 spatial upsampler, and the distilled refiner
LoRA. The TAEHV weight is installed below the same root as shown above.

## Prepare the official TAEHV weight

Do not download or hash the 60 MB weight on a login node. Submit the included
script to a small CPU compute allocation; it refuses to run without
`SLURM_JOB_ID`, verifies the pinned SHA-256, and never overwrites a mismatched
existing file:

```bash
cd /home/yitongl/code/agent_deploy/sol-engine

srun -A nvr_elm_llm \
  --partition cpu \
  --nodes 1 \
  --ntasks 1 \
  --cpus-per-task 1 \
  --mem 2G \
  --time 00:10:00 \
  bash models/ltx2.5-refiner/GB200/prepare_taehv_weight.sh
```

The script is idempotent when the already-installed file has the expected
hash. It does not modify the shared Python environment.

## Input manifest

The manifest is a JSON array containing exactly one row. `file` is relative to
`INPUT_ROOT`; prompt and seed are the only per-sample model inputs that may
vary:

```json
[
  {
    "index": 0,
    "prompt_id": "example",
    "file": "clips/example.mp4",
    "prompt": "A detailed caption that faithfully describes the input video.",
    "seed": 42
  }
]
```

The source clip must already be 1920x1088, 241 frames, and 24 FPS. The runner
rejects a different frame count, frame rate, batch size, GPU count, parallel
degree, parameter layout, or sampling policy instead of silently changing the
fixed configuration.

## Run one hot 4xGB200 inference

Imports, model loading, and inference must run inside the four-GPU allocation.
The login node is only for lightweight path checks and Slurm submission.

```bash
cd /home/yitongl/code/agent_deploy/sol-engine

export INPUT_ROOT=/absolute/path/to/input
export MANIFEST="$INPUT_ROOT/manifest.json"
export RUN_ROOT=/absolute/path/to/run/ltx25-refiner-1080p10s
mkdir -p "$RUN_ROOT/outputs" "$RUN_ROOT/metadata"

srun -A nvr_elm_llm \
  --partition batch \
  --qos interactive \
  --nodes 1 \
  --ntasks 1 \
  --gpus-per-node 4 \
  --cpus-per-task 32 \
  --mem 0 \
  --time 00:30:00 \
  python3 scripts/run.py models/ltx2.5-refiner/GB200/refiner.toml \
    --set INPUT_ROOT="$INPUT_ROOT" \
    --set MANIFEST="$MANIFEST" \
    --set OUTPUT_DIR="$RUN_ROOT/outputs" \
    --set METADATA_DIR="$RUN_ROOT/metadata"
```

The model fleet is loaded once, then the same manifest row is run once as an
excluded warm-up and once as the measured request. The four worker processes
are launched with `torchrun --standalone --nproc_per_node=4`. Every allocated
GPU holds a full parameter replica and participates in the head/context-parallel
self-attention for the same video; this is not data parallelism over four
videos.

## Run the matched torch.compile arm

The compile arm changes no model, prompt, seed, shape, sampling, parallelism,
or Sol-Attn setting. It compiles each transformer block with Inductor using
`mode=max-autotune-no-cudagraphs`, `fullgraph=False`, and capture disabled.
The fixed token shape is compiled statically (`seq_dim_dynamic=False`).
The stateful tau scheduler and CuTe Sol call form an explicit eager graph
boundary; norms, projections, dense cross-attention, MLPs, and dense layer 0
remain eligible for compilation. The first complete request performs tracing,
code generation, autotuning, and all three tau steps and is excluded. Only the
following identical request is measured.

Use the same command and path overrides as above, changing only the config and
using a separate run directory:

```bash
export RUN_ROOT=/absolute/path/to/run/ltx25-refiner-1080p10s-compile
mkdir -p "$RUN_ROOT/outputs" "$RUN_ROOT/metadata"

srun -A nvr_elm_llm \
  --partition batch \
  --qos interactive \
  --nodes 1 \
  --ntasks 1 \
  --gpus-per-node 4 \
  --cpus-per-task 32 \
  --mem 0 \
  --time 00:30:00 \
  python3 scripts/run.py models/ltx2.5-refiner/GB200/refiner_compile.toml \
    --set INPUT_ROOT="$INPUT_ROOT" \
    --set MANIFEST="$MANIFEST" \
    --set OUTPUT_DIR="$RUN_ROOT/outputs" \
    --set METADATA_DIR="$RUN_ROOT/metadata"
```

All generated-code caches are versioned under this persistent shared root,
which survives Slurm job-node teardown:

```text
/home/yitongl/code/.cache/sol-engine/ltx25-refiner/gb200-sm100_py31311_torch2110-cu130_triton360/stage2-headcp4-sol-eager-boundary-v1/1080p10s
```

The launcher creates and exports `inductor/`, `triton/`, `cuda/`, and
`cute_dsl/` below that root before Python starts. The last directory is
required because CuTe DSL otherwise defaults to a node-local temporary cache.
Change the versioned root whenever the source, compiler stack, GPU
architecture, input shape, parallel degree, or compile policy changes; do not
mix H100 and GB200 artifacts.

## Timing contract and validation

The authoritative latency is the post-warm-up resident `sample_wall_s`. It is
a hot end-to-end single-sample wall time: input decode/validation, prompt
encoding, TAEHV encode, x2 latent upsampling, three-step Stage-2 denoising,
TAEHV decode, video encode/mux, and output verification are included. Process
startup, model loading, Sol kernel first-use compilation/autotuning, and the
entire warm-up request are excluded. For the compile arm, Dynamo tracing,
Inductor/Triton/CuTe code generation or cache restoration, and compiler
autotuning are likewise warm-up costs and excluded. The measured request still
guards against any model-weight reload.

At minimum, a successful metadata record must prove all of the following:

- world size 4, parallel degree 4, and a full parameter replica on every rank;
- self-attention heads split four ways, with the full token sequence visible
  to every local head;
- compute capability SM100 on all four ranks;
- 3 completed attention steps and taus `[1.0, 1.25, 1.5]`;
- per-rank calls of 3 dense, 141 Sol, and 141 actual kernel executions;
- all timed inference modules resident and zero weight-load attempts;
- TAEHV temporal execution set to `sequential`;
- output video geometry of 1920x1088, 241 frames, and 24 FPS.

Do not carry the prior one-H100 latency numbers forward as GB200 results. A
GB200 latency is reportable only after this exact four-GPU hot run passes the
metadata checks above.

## Validated 4xGB200 eager control

The fixed contract passed end to end on 2026-08-14 in Slurm job `6145732`.
This is one validated smoke sample, not a multi-sample performance
distribution:

| Scope | Seconds |
|---|---:|
| Resident preload, excluded | 78.557829 |
| Complete warm-up, excluded | 21.834112 |
| Input decode and resize | 0.928891 |
| Gemma embedding | 0.119203 |
| TAEHV encode | 0.247842 |
| Latent x2 upsample | 0.048934 |
| Replica synchronization | 0.000307 |
| Denoise preparation | 0.000934 |
| Stage-2 transformer, 3 updates | 2.372426 |
| Denoise finish | 0.000301 |
| TAEHV decode | 0.302109 |
| H.264 encode and mux | 1.657777 |
| **Hot resident end to end** | **5.677548** |

Phase values are maxima across ranks and can come from different ranks, while
the end-to-end value is the maximum rank wall time; phase rows therefore are
diagnostic rather than an accounting identity. Validation recorded 32 total
heads, 8 local heads per rank, 63,240 full-sequence tokens per local head, and
the `cute_sm100` backend. Every rank completed 3 dense layer-0 executions and
141 Sol kernel executions, for aggregate counts of 12 dense and 564 Sol calls.

The validated video and metadata are outside Git at:

```text
/home/yitongl/code/agent_deploy/sol-engine/runs/ltx25-refiner-smoke/outputs/ltx25_refiner_1920x1088_241f.mp4
/home/yitongl/code/agent_deploy/sol-engine/runs/ltx25-refiner-smoke/metadata/
```

## Validated 4xGB200 compile result

Two new-process compile runs passed the same fixed contract and all attention
checks. Each rank again reported 3 dense layer-0 calls, 141 Sol calls, and 141
actual `cute_sm100` kernel calls, with no fallback or measured weight load.

| Arm | Hot Stage-2 transformer | Hot resident E2E |
|---|---:|---:|
| Eager control | 2.372426 s | 5.677548 s |
| Compile run A | 1.973195 s | 5.232343 s |
| Compile run B, persistent cache reused | 1.969754 s | 5.367247 s |
| Compile mean | **1.971475 s** | **5.299795 s** |

Against the eager control, the compile mean is a **1.203x transformer
speedup** (16.90% lower transformer latency) and a **1.071x hot E2E speedup**
(6.65% lower E2E latency). The two compiled transformer measurements differ
by only 0.17%; their wider E2E difference is from input decode and replica
synchronization, neither of which is compiled. These are one eager and two
compiled smoke measurements, not a performance distribution.

Cold cache compilation took 118.466213 s in the excluded first request. A new
process reusing the persistent disk cache took 31.838885 s for its excluded
request; Dynamo tracing and guards still occur per process. Neither value is
part of the hot latency above.

The compile outputs and complete metadata are outside Git at:

```text
/home/yitongl/code/agent_deploy/sol-engine/runs/ltx25-refiner-compile-1080p10s-20260814-a/
/home/yitongl/code/agent_deploy/sol-engine/runs/ltx25-refiner-compile-1080p10s-20260814-b/
```
