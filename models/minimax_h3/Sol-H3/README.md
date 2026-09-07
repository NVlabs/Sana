# Sol-H3

[Project page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/)

Production inference package for MiniMax-H3 with synchronized video and audio output.

- Text-to-video, first-frame image-to-video, and Ref2VA
- 1344x768 at 24 FPS
- 5, 10, or 15 second output
- 1, 2, 4, or 8 GPUs; validated on 8x NVIDIA B300
- Fast SOL/BSA profile enabled by default

The default profile prioritizes speed and uses lossy acceleration. Select `dense` when a dense-attention reference is required or when running on one GPU.

## Validation

All three task paths have been exercised on 8x NVIDIA B300 at 1344x768:

| Task | Result |
|---|---|
| T2V | 124 / 243 / 362 frames: 1.653 / 3.732 / 6.612 s warm 8-GPU pipeline medians |
| I2V | Native first-frame conditioning accepted at 124 frames and in a resident 362-frame regression |
| Ref2VA | 124 / 243 / 362 frames: 2.192 / 4.348 / 5.947 s warm pipeline medians |

The FastH3 preview adapter is published for T2V. I2V combines that adapter with MiniMax-H3's native
first-frame conditioning path, so its validation is a deployment compatibility result rather than
an upstream I2V training claim. Latency excludes checkpoint loading, warmup, and MP4 encoding.

Kernel validation for this PR is limited to SM103 B300 through the SM100-family backend. The
reachable SM90 and SM120 paths remain unvalidated pending hardware-specific smoke and numerical
checks.

### T2V benchmarks

The following results were measured on NVIDIA B300 SXM6 AC GPUs. Each cell is the median of three
requests after one warmup. All runs use the same prompt and seed at 1344x768 and 24 FPS, generate
synchronized stereo audio, and include text encoding, DiT denoising, and video/audio VAE decoding.
Checkpoint loading, warmup/compilation, and MP4 encoding are excluded.

| Engine | GPUs | 5 s / 124f | 10 s / 243f | 15 s / 362f |
|---:|---:|---:|---:|---:|
| Sol-H3 | 1 | **13.745 s** | **37.813 s** | **52.260 s** |
| SGLang | 1 | 129.898 s | 376.942 s | 746.885 s |
| Diffusers | 1 | 159.547 s | 442.533 s | 847.486 s |
| Sol-H3 | 4 | **2.918 s** | **6.993 s** | **12.542 s** |
| Diffusers | 4 | 48.552 s | 127.461 s | 237.171 s |
| SGLang | 4 | 35.328 s | 100.440 s | 194.930 s |
| Sol-H3 | 8 | **1.653 s** | **3.732 s** | **6.612 s** |
| SGLang | 8 | 18.250 s | 50.660 s | 99.513 s |
| Diffusers | 8 | 30.673 s | 71.316 s | 131.647 s |

#### Detailed Settings
  Diffusers: Base BF16 dense model with 50 scheduler points (49 DiT forwards); Ulysses on 4/8 GPUs.
  
  SGLang: Base BF16 dense model with 50 scheduler points (49 DiT forwards); Ulysses on 4/8 GPUs.
  
  Sol-H3: FastH3 adapter with five scheduler points (four DiT forwards); dense attention on 1 GPU and
SOL/BSA on 4/8 GPUs. The multi-GPU profile uses INT8 QKV transport, FP8 output transport, and
parallel video/audio VAE decoding.

  The official and SGLang rows are directly comparable. Sol-H3 uses a distilled four-forward adapter
and its multi-GPU profile uses approximate SOL/BSA attention, so the difference from either
49-forward baseline is not a runtime-only speedup.


## Setup

Recommended environment: Linux, Python 3.12, CUDA 13.0, and PyTorch 2.10.

```bash
conda create -n h3-infer python=3.12 -y
conda activate h3-infer
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130
pip install -r requirements.txt
```

Accept the MiniMax-H3 model terms on Hugging Face, authenticate, and download the checkpoints:

```bash
hf auth login
python download_checkpoints.py --output-dir ./checkpoints
```

For Ref2VA, download its model partition and four-step adapter instead:

```bash
python download_checkpoints.py --output-dir ./checkpoints --task ref2va
```

## Text-to-video

```bash
torchrun --standalone --nproc_per_node=8 infer.py \
  --model ./checkpoints/MiniMax-H3 \
  --adapter ./checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors \
  --task t2v --duration 5 \
  --prompt-file prompt.txt \
  --output output.mp4 --warmup
```

## Image-to-video

```bash
torchrun --standalone --nproc_per_node=8 infer.py \
  --model ./checkpoints/MiniMax-H3 \
  --adapter ./checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors \
  --task i2v --duration 10 \
  --image first_frame.png \
  --prompt-file prompt.txt \
  --output output.mp4 --warmup
```

## Ref2VA

References are read in command-line order. Repeat `--reference` to combine an image or video with optional audio.

```bash
torchrun --standalone --nproc_per_node=8 infer.py \
  --model ./checkpoints/MiniMax-H3 \
  --adapter ./checkpoints/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --task ref2va --duration 10 \
  --reference image:subject.png \
  --reference audio:voice.wav \
  --prompt-file prompt.txt \
  --output output.mp4 --warmup
```

Ref2VA accepts image, video, and audio references. Audio cannot be the only reference. Use `sol_bsa` (default) or `dense` for this task. Reference images use the validated fast `match` profile by default; pass `--reference-image-resize-mode diffusers` only when official 2048-short-edge preprocessing parity is required.

Main options:

| Option | Values |
|---|---|
| `--task` | `t2v`, `i2v`, or `ref2va` |
| `--duration` | `5`, `10`, or `15` |
| `--attention-backend` | `sol_bsa` (default), `sol`, or `dense` |
| `--prompt` / `--prompt-file` | Prompt text or UTF-8 prompt file |
| `--image` | First frame for `i2v` |
| `--reference` | Ordered `image:PATH`, `video:PATH`, or `audio:PATH` for `ref2va`; repeat as needed |
| `--reference-image-resize-mode` | `match` (fast default) or `diffusers` (official 2048 mode) |
| `--seed` | Random seed |
| `--output` | Output MP4 path |
| `--warmup` | Warm up the selected duration and prompt before generation |

For a service, keep all worker processes resident and warm up every duration used in production.

## Python API

```python
from h3_runtime import MiniMaxH3Inference

with MiniMaxH3Inference(MODEL_PATH, ADAPTER_PATH) as engine:
    result = engine.generate(prompt, duration=5, seed=1)
    if result is not None:
        result.save("output.mp4")
```

Ref2VA uses the same API with an explicit task and the Diffusers reference type:

```python
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
from h3_runtime import MiniMaxH3Inference

references = [MiniMaxH3Reference(image="subject.png")]
with MiniMaxH3Inference(MODEL_PATH, REF2VA_ADAPTER_PATH, task="ref2va") as engine:
    result = engine.generate(prompt, duration=10, references=references)
    if result is not None:
        result.save("output.mp4")
```
