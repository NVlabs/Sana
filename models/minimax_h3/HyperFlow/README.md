# HyperFlow for Sol-Engine

Eight-step MiniMax-H3 video and audio generation using
[Video Rebirth's HyperFlow 1.0](https://github.com/Video-Rebirth/hyperflow) and the
existing [Sol-H3 runtime](../Sol-H3/). This integration targets the resident
**8 x NVIDIA B200** configuration and provides T2V, first-frame I2V, and
image-reference Ref2VA at 1344 x 768 and 24 FPS. T2V and single-image Ref2VA
have also been benchmarked on **8 x NVIDIA B300**, as reported below.

HyperFlow's two-time conditioning and checkpoint schedule are preserved. T2V
and I2V share the upstream `transformer`; Ref2VA keeps its distinct
`transformer_ref`. The text/image conditioner, tokenizers, schedulers, video
VAE, and audio VAE are shared. Both DiTs remain resident after initialization.

## Scope

| Task | Input | Output presets |
|---|---|---|
| `t2v` | Text | 5 / 10 / 15 seconds, with audio |
| `i2v` | Text and one first-frame image | Same |
| `ref2va` | Text and 1–9 ordered image references | Same |

The frame counts are 124 / 243 / 362, following the model's temporal alignment;
the encoded duration is therefore slightly longer than the nominal preset.
Every request uses eight DiT forwards. The CLI does not expose a step override.
DiT linear compute uses BF16 with unmerged LoRA branches; MXFP8 is not enabled.

The accelerated runner is limited to the above profile. Other GPU counts,
resolutions, last-frame conditioning, and reference audio/video are not exposed
or claimed as validated by this runner. The unmodified `hyperflow_h3/` source
retains the upstream workflow building blocks. Multiple image references are
accepted by the interface; the recorded GPU smoke checks used one reference.

## Performance

Sol-Engine achieves **2.86–4.15x speedup** over the original HyperFlow runtime
in the following **8 x NVIDIA B300 SXM6 AC** measurements. Both configurations
use the same base model and HyperFlow 1.0 adapter, **eight denoising steps**,
BF16 DiT linears, and **unmerged LoRA**. Sol-Engine enables fused LoRA kernels
and the full optimization stack; MXFP8 and reference KV caching are disabled.

| Task | Video preset | Original HyperFlow (s) | Sol-Engine (s) | Speedup |
|---|---:|---:|---:|---:|
| T2V | 5 s | 10.834 | **3.571** | **3.03x** |
| T2V | 10 s | 22.702 | **7.685** | **2.95x** |
| T2V | 15 s | 37.974 | **13.273** | **2.86x** |
| Ref2VA (one image) | 5 s | 15.649 | **3.773** | **4.15x** |
| Ref2VA (one image) | 10 s | 28.810 | **7.608** | **3.79x** |
| Ref2VA (one image) | 15 s | 44.790 | **12.578** | **3.56x** |

Measured on 2026-09-19 at **1344 x 768, 24 FPS**, with synchronized audio,
seed 42, and 124 / 243 / 362 output frames. Each entry is the median of
**five timed calls after two warmups**, with models resident on the same host.
Time includes conditioning, denoising, video/audio VAE decoding, and completion
synchronization across all GPUs. It excludes model loading, initialization,
warmup, prompt rewriting, RGB8 conversion, MP4 encoding, and file writes.

The baseline uses the original HyperFlow Python runtime with dense SDPA and
native PEFT LoRA branches. The speedups include approximate SOL/BSA attention,
compressed communication, and reference-image resizing; see
[numerical behavior](#optimizations-and-numerical-behavior).
These measurements cover one prompt per task and one seed. The
[benchmark record](VALIDATION.md#b300-performance-benchmark--2026-09-19)
documents the tested commit, configuration, and timing samples.

## Setup

Use Linux, Python 3.12, a working CUDA 13 toolchain/driver, and eight B200 or B300 GPUs
connected by NVLink/NVSwitch. Run from this directory in a checkout of Sana's
`sol-engine` branch, with the sibling `Sol-H3/` directory present. The package
base commit and source hashes are in [PROVENANCE.json](PROVENANCE.json).

```bash
cd models/minimax_h3/HyperFlow
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130
pip install -r requirements.txt
```

The dependency file pins the Diffusers revision used with HyperFlow. The
shared Sol-H3 runtime has one compatibility fallback for this revision; its
fusion and attention kernels are reused without a second copy. Use this
environment for HyperFlow instead of installing both model directories'
different Diffusers pins together. Install FFmpeg for MP4/audio export.

### Download the checkpoints

Download the base model from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
and the official adapter from [videorebirth/hyperflow](https://huggingface.co/videorebirth/hyperflow).
The following commands pin the base revision recorded in the adapter header
and the public HyperFlow release revision:

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 83db0c0efe6ef9824e0e194be110346c0a9542ed \
  --local-dir checkpoints/MiniMax-H3

hf download videorebirth/hyperflow \
  minimax_h3_hyperflow_8step_v1.0.safetensors hyperflow.json LICENSE \
  --revision a83ec7ad08afdbb0438615cfcfcb89320667a093 \
  --local-dir checkpoints/hyperflow
```

Keep the normal shared components and **both** `transformer/` and
`transformer_ref/` in the base snapshot. Do not replace the latter with a link
to the former. The official adapter's filename and SHA256 are:

```text
minimax_h3_hyperflow_8step_v1.0.safetensors
SHA256: 9297f4505bfdef59c3014d11274411809c19b0abfe26161cab2b425a696df447
```

This hash is published in the upstream
[hyperflow.json manifest](https://huggingface.co/videorebirth/hyperflow/blob/a83ec7ad08afdbb0438615cfcfcb89320667a093/hyperflow.json).
Weights are downloaded separately and governed by their upstream model license;
the integration code is Apache-2.0. The loader reads the HyperFlow adapter's
eight-step schedule and conditioning metadata.

## Command-line inference

Use the local files downloaded above:

```bash
export H3_MODEL="$PWD/checkpoints/MiniMax-H3"
export H3_ADAPTER="$PWD/checkpoints/hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors"

torchrun --standalone --nproc_per_node=8 infer.py \
  --model "$H3_MODEL" --adapter "$H3_ADAPTER" \
  --task t2v --duration 5 --seed 42 --warmup \
  --prompt "A fox walks through a snowy forest, with soft footsteps and wind." \
  --output outputs/t2v.mp4

torchrun --standalone --nproc_per_node=8 infer.py \
  --model "$H3_MODEL" --adapter "$H3_ADAPTER" \
  --task i2v --image /path/to/first-frame.png --duration 10 --seed 42 --warmup \
  --prompt "The subject slowly turns toward the camera in natural light." \
  --output outputs/i2v.mp4

torchrun --standalone --nproc_per_node=8 infer.py \
  --model "$H3_MODEL" --adapter "$H3_ADAPTER" \
  --task ref2va --reference-image /path/to/subject.png --duration 15 --seed 42 --warmup \
  --prompt "The subject in <Picture 1> walks through a sunlit room." \
  --output outputs/ref2va.mp4
```

Repeat `--reference-image` in the order referred to by `<Picture 1>`,
`<Picture 2>`, etc. `--prompt-file` accepts a UTF-8 file instead of `--prompt`.
Each launch initializes both DiTs; use the Python interface to amortize that
cost across requests. `--warmup` runs an additional request with the same task,
duration, prompt, and references before the reported request.

The printed `generation_seconds` includes conditioning, denoising, video/audio
VAE decoding, and completion synchronization. It excludes checkpoint loading,
the optional warmup request, and MP4 encoding. Without warmup, compilation can
be included in the reported time. See [Performance](#performance) for warmed
B300 latencies and [VALIDATION.md](VALIDATION.md) for their methodology.

## Resident Python interface

Launch your script with `torchrun --standalone --nproc_per_node=8`. Every rank
must call initialization and generation in the same order; only rank zero
receives the returned media. Requests are serial because they share schedulers
and request state.

```python
from sol_hyperflow import HyperFlowInference

with HyperFlowInference(model_path, adapter_path) as engine:
    for prompt in prompts:
        media = engine.generate(prompt, task="t2v", duration=5, seed=42)
        if media is not None:
            media.save(output_path_for(prompt))
```

Pass a PIL image as `image=` for I2V or a list of PIL images as `references=`
for Ref2VA. The engine releases a process group only when it created that group.

## Optimizations and numerical behavior

| Change | Principle and scope |
|---|---|
| Shared resident components | Load common encoders and VAEs once per rank; retain distinct base weights for the two DiTs. |
| Paired AdaLN tables | Precompute modulation for every `(t, endpoint)` pair and all four conditioning variants: none, image, audio, image+audio. Release the original modulation projections afterward. This is deterministic precomputation, not reuse of approximate denoising features. Unsupported schedules fail explicitly. |
| Conditioner pruning | Keep 51 of 64 language layers so `hidden_states[50]` remains the selected **pre-normalization** feature; remove the unused vocabulary head. Text and image embedding equality is checked during initialization. The recorded B200 check freed 14,233,381,376 bytes per rank. |
| BF16 LoRA consumer fusion | Reuse [Sana PR #503](https://github.com/NVlabs/Sana/pull/503): keep the adapter branches separate and consume their sums in fused QKV, attention-output, SwiGLU, and FFN-output kernels. Historical native/fused checks were bitwise equal for the tested cases. `--lora-mode separate` disables these consumer fusions for comparison. |
| Sequence parallelism and kernel fusion | Reuse Sol-H3's Ulysses implementation and fused normalization/rotary/activation kernels. Shared VAE optimizations parallelize decoding and image encoding. |
| SOL/BSA attention and transport | The default profile uses approximate sparse attention, INT8 QKV transport, and FP8 output transport. These are lossy optimizations; BF16 DiT linears do not make the whole pipeline numerically lossless. |
| Reference sizing | Cap reference-image area at 1344 x 768 without upscaling. This reduces reference tokens and changes preprocessing compared with the upstream short-edge rule. |

For the sparse policy, T2V/I2V keep the first denoising step and first two layers
dense (336 sparse and 64 dense attention calls per rank). Ref2VA uses 400 sparse
calls. `--attention-backend dense` selects all 400 dense calls, but does not by
itself disable compressed Ulysses transport. To also select BF16 transport:

```bash
export H3_ULYSSES_COMM_DTYPE=bf16
export H3_ULYSSES_OUTPUT_DTYPE=bf16
# Add --attention-backend dense to the inference command.
```

This still retains the eight-step adapter, fused kernels, and reference sizing;
it is not an unmodified upstream baseline. Reference-video KV caching and
prompt rewriting are not part of this integration.

## Source and tests

- `hyperflow_h3/`: six unchanged modules from the official
  [HyperFlow release](https://github.com/Video-Rebirth/hyperflow/tree/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3),
  with original attribution and SHA256 hashes recorded.
- `sol_hyperflow/`: resident loading, conditioning cache, memory pruning, and
  request lifecycle integration with Sol-H3.
- `infer.py`: public CLI. `validate_runtime.py`: optional eight-GPU checks.
- `tests/`: CPU contract and synthetic numerical tests; no checkpoint required.

See [VALIDATION.md](VALIDATION.md) for commands, results, and limitations, and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution.
