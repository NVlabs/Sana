# LTX-Core

The foundational library for the LTX-2 Audio-Video generation model. This package contains the raw model definitions, component implementations, and loading logic used by `ltx-pipelines` and `ltx-trainer`.

## 📦 What's Inside?

- **`components/`**: Modular diffusion components (Schedulers, Guiders, Noisers, Patchifiers) following standard protocols
- **`conditioning/`**: Tools for preparing latent states and applying conditioning (image, video, keyframes)
- **`guidance/`**: Perturbation system for fine-grained control over attention mechanisms
- **`loader/`**: Utilities for loading weights from `.safetensors`, fusing LoRAs, and managing memory
- **`block_streaming/`**: Memory-efficient inference that streams transformer blocks through the GPU one at a time (from pinned CPU buffers or directly from disk)
- **`model/`**: PyTorch implementations of the LTX-2 Transformer, Video VAE, Audio VAE, Vocoder and Upscaler
- **`text_encoders/gemma`**: Gemma text encoder implementation with tokenizers, feature extractors, and separate encoders for audio-video and video-only generation
- **`quantization/`**: FP8 quantization backends (FP8 scaled MM, FP8 cast) for reduced memory footprint.

## 🚀 Quick Start

`ltx-core` provides the building blocks (models, components, and utilities) needed to construct inference flows. For ready-made inference pipelines use [`ltx-pipelines`](../ltx-pipelines/) or [`ltx-trainer`](../ltx-trainer/) for training.

## 🔧 Installation

```bash
# From the repository root
uv sync --frozen

# Or install as a package
pip install -e packages/ltx-core
```

## Building Blocks Overview

`ltx-core` provides modular components that can be combined to build custom inference flows:

### Core Models

- **Transformer** ([`model/transformer/`](src/ltx_core/model/transformer/)): The asymmetric dual-stream LTX-2 transformer (14B-parameter video stream, 5B-parameter audio stream) with bidirectional cross-modal attention for joint audio-video processing. Expects inputs in [`Modality`](src/ltx_core/model/transformer/modality.py) format
- **Video VAE** ([`model/video_vae/`](src/ltx_core/model/video_vae/)): Encodes/decodes video pixels to/from latent space with temporal and spatial compression
- **Audio VAE** ([`model/audio_vae/`](src/ltx_core/model/audio_vae/)): Encodes/decodes audio spectrograms to/from latent space
- **Vocoder** ([`model/audio_vae/`](src/ltx_core/model/audio_vae/)): Neural vocoder that converts mel spectrograms to audio waveforms
- **Text Encoder** ([`text_encoders/`](src/ltx_core/text_encoders/)): Gemma 3-based multilingual encoder with multi-layer feature extraction and thinking tokens that produces separate embeddings for video and audio conditioning
- **Spatial Upscaler** ([`model/upsampler/`](src/ltx_core/model/upsampler/)): Upsamples latent representations for higher-resolution generation

### Diffusion Components

- **Schedulers** ([`components/schedulers.py`](src/ltx_core/components/schedulers.py)): Noise schedules (LTX2Scheduler, LinearQuadratic, Beta) that control the denoising process
- **Guiders** ([`components/guiders.py`](src/ltx_core/components/guiders.py)): Guidance strategies (CFG, STG, APG) for controlling generation quality and adherence to prompts
- **Noisers** ([`components/noisers.py`](src/ltx_core/components/noisers.py)): Add noise to latents according to the diffusion schedule
- **Patchifiers** ([`components/patchifiers.py`](src/ltx_core/components/patchifiers.py)): Convert between spatial latents `[B, C, F, H, W]` and sequence format `[B, seq_len, dim]` for transformer processing

### Conditioning & Control

- **Conditioning** ([`conditioning/`](src/ltx_core/conditioning/)): Tools for preparing and applying various conditioning types (image, video, keyframes)
- **Guidance** ([`guidance/`](src/ltx_core/guidance/)): Perturbation system for fine-grained control over attention mechanisms (e.g., skipping specific attention layers)

### Utilities

- **Loader** ([`loader/`](src/ltx_core/loader/)): Model loading from `.safetensors`, LoRA fusion, weight remapping, and memory management
- **Quantization** ([`quantization/`](src/ltx_core/quantization/)): FP8 quantization backends for reduced memory footprint and faster inference
- **Block Streaming** ([`block_streaming/`](src/ltx_core/block_streaming/)): Streams transformer blocks through the GPU one block at a time, so the full model runs on machines without enough memory to hold all its weights at once

### Loader

The `loader/` module provides `SingleGPUModelBuilder`, a frozen dataclass that loads a PyTorch model from `.safetensors` checkpoints and optionally fuses one or more LoRA adapters.

#### Basic usage

```python
from ltx_core.loader import SingleGPUModelBuilder

builder = SingleGPUModelBuilder(
    model_class_configurator=MyModelConfigurator,
    model_path="/path/to/model.safetensors",
)
model = builder.build(device=torch.device("cuda"))
```

#### Loading LoRA adapters

Use the `.lora()` method to attach one or more LoRA adapters before calling `.build()`:

```python
from ltx_core.loader import SDOps

lora_sd_ops = SDOps(name="identity").with_matching()  # or a model-specific key-renaming SDOps

builder = (
    SingleGPUModelBuilder(
        model_class_configurator=MyModelConfigurator,
        model_path="/path/to/model.safetensors",
    )
    .lora("/path/to/lora_a.safetensors", 0.8, lora_sd_ops)
    .lora("/path/to/lora_b.safetensors", 0.5, lora_sd_ops)
)
model = builder.build(device=torch.device("cuda"))
```

#### Memory-efficient LoRA loading (`lora_load_device`)

By default, LoRA weights are loaded onto the **CPU** (`lora_load_device=torch.device("cpu")`).  This means each LoRA adapter is kept in CPU memory and transferred to the GPU sequentially during weight fusion, which keeps peak GPU memory low even when fusing large adapters.

If all adapters fit comfortably in GPU memory you can skip the CPU staging by setting `lora_load_device` to the target CUDA device:

```python
import torch
from ltx_core.loader import SingleGPUModelBuilder

# Load LoRA weights directly onto the GPU (faster, but uses more GPU memory)
builder = SingleGPUModelBuilder(
    model_class_configurator=MyModelConfigurator,
    model_path="/path/to/model.safetensors",
    lora_load_device=torch.device("cuda"),
).lora("/path/to/lora.safetensors", 1.0, lora_sd_ops)

model = builder.build(device=torch.device("cuda"))
```

### Quantization

The `quantization/` module provides FP8 quantization support for the LTX-2 transformer, significantly reducing memory usage while maintaining quality. Two backends are available:

#### FP8 Scaled MM

Uses PyTorch's `torch._scaled_mm` for efficient FP8 matrix multiplication. Weights are stored in FP8 format with per-tensor scaling, and inputs are quantized dynamically.

**Usage with QuantizationPolicy:**

```python
from ltx_core.quantization.fp8_scaled_mm import build_policy as build_fp8_scaled_mm_policy

# Discovers the layer set from the checkpoint's .weight_scale tensors
policy = build_fp8_scaled_mm_policy("/path/to/checkpoint.safetensors")
```

The policy carries `sd_ops`, `module_ops`, and `fuse_rule` that are passed to the model builder:

```python
import torch
from ltx_core.loader import SingleGPUModelBuilder

builder = SingleGPUModelBuilder(
    model_class_configurator=MyModelConfigurator,
    model_path="/path/to/checkpoint.safetensors",
    model_sd_ops=policy.sd_ops,
    module_ops=policy.module_ops,
    fuse_rule=policy.fuse_rule,
)
model = builder.build(device=torch.device("cuda"))
```

#### FP8 Cast

A simpler approach that casts weights to FP8 for storage and upcasts during inference:

```python
from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy

policy = build_fp8_cast_policy("/path/to/checkpoint.safetensors")
```

### Block Streaming

The `block_streaming/` module ([`src/ltx_core/block_streaming/`](src/ltx_core/block_streaming/)) lets the full model run on machines that lack the memory to hold all of its weights at once. It streams the transformer's blocks through a small rolling set of GPU buffers, loading each block's weights just before it runs and recycling them afterwards, so only a few blocks are resident on the GPU at any moment. Construct it with `StreamingModelBuilder`, which returns a `BlockStreamingWrapper` -- an `nn.Module` drop-in for the wrapped model.

#### Strategies

The strategy is chosen automatically from `cpu_slots_count` relative to the number of blocks:

- **RAM streaming** (default, `cpu_slots_count` omitted or `>= num_blocks`): all blocks are pre-loaded into pinned CPU buffers (with LoRA fusion) at build time, then copied to the GPU on demand. Fast; higher CPU memory.
- **Disk streaming** (`cpu_slots_count < num_blocks`): blocks are read from the `.safetensors` file on demand on a background worker thread. Slower; lowest CPU memory.

#### Basic usage

```python
import torch
from ltx_core.block_streaming import StreamingModelBuilder

builder = StreamingModelBuilder(
    model_class_configurator=MyModelConfigurator,
    model_path="/path/to/model.safetensors",
    blocks_attr="transformer_blocks",      # dotted path to the nn.ModuleList
    blocks_prefix="transformer_blocks",    # state-dict key prefix for block weights
)

# Omit cpu_slots_count for RAM streaming; pass a value < num_blocks for disk streaming.
model = builder.build(
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
    cpu_slots_count=4,
    gpu_slots_count=2,
)
```

For complete, production-ready pipeline implementations that combine these building blocks, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

# Architecture Overview

This section provides a deep dive into the internal architecture of the LTX-2 Audio-Video generation model.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [The Transformer](#the-transformer)
3. [Video VAE](#video-vae)
4. [Audio VAE](#audio-vae)
5. [Text Encoding (Gemma)](#text-encoding-gemma)
6. [Spatial Upscaler](#spatial-upsampler)
7. [Data Flow](#data-flow)

---

## High-Level Architecture

LTX-2 is an **asymmetric dual-stream diffusion transformer** that jointly models the text-conditioned distribution of video and audio signals, capturing true joint dependencies (unlike sequential T2V→V2A pipelines).

### Key Design Principles

- **Decoupled Latent Representations**: Separate modality-specific VAEs enable 3D RoPE (video) vs 1D RoPE (audio), independent compression optimization, and native V2A/A2V editing workflows
- **Asymmetric Dual-Stream**: 14B-parameter video stream (spatiotemporal dynamics) + 5B-parameter audio stream (1D temporal), sharing 48 transformer blocks but differing in width
- **Bidirectional Cross-Modal Attention**: 1D temporal RoPE enables sub-frame alignment, mapping visual cues to auditory events (lip-sync, foley, environmental acoustics)
- **Cross-Modality AdaLN**: Scaling/shift parameters conditioned on the other modality's hidden states for synchronization across differing diffusion timesteps/temporal resolutions

```text
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PREPARATION                        │
│                                                             │
│  Video Pixels → Video VAE Encoder → Video Latents           │
│  Audio Waveform → Audio VAE Encoder → Audio Latents         │
│  Text Prompt → Gemma 3 Encoder → Text Embeddings            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│     LTX-2 ASYMMETRIC DUAL-STREAM TRANSFORMER (48 Blocks)    │
│                                                             │
│  ┌──────────────────────┐      ┌──────────────────────┐     │
│  │  Video Stream (14B)  │      │  Audio Stream (5B)   │     │
│  │                      │      │                      │     │
│  │  3D RoPE (x,y,t)     │      │  1D RoPE (temporal)  │     │
│  │                      │      │                      │     │
│  │  Self-Attn           │      │  Self-Attn           │     │
│  │  Text Cross-Attn     │      │  Text Cross-Attn     │     │
│  │                      │◄────►│                      │     │
│  │  A↔V Cross-Attn      │      │  A↔V Cross-Attn      │     │
│  │  (1D temporal RoPE)  │      │  (1D temporal RoPE)  │     │
│  │  Cross-modality      │      │  Cross-modality      │     │
│  │  AdaLN               │      │  AdaLN               │     │
│  │  Feed-Forward        │      │  Feed-Forward        │     │
│  └──────────────────────┘      └──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT DECODING                          │
│                                                             │
│  Video Latents → Video VAE Decoder → Video Pixels           │
│  Audio Latents → Audio VAE Decoder → Mel Spectrogram        │
│  Mel Spectrogram → Vocoder → Audio Waveform (24 kHz)        │
└─────────────────────────────────────────────────────────────┘
```

---

## The Transformer

The core of LTX-2 is an **asymmetric dual-stream diffusion transformer** with 48 layers that processes both video and audio tokens simultaneously. The architecture allocates 14B parameters to the video stream and 5B parameters to the audio stream, reflecting the different information densities of the two modalities.

### Model Structure

**Source**: [`src/ltx_core/model/transformer/model.py`](src/ltx_core/model/transformer/model.py)

The `LTXModel` class implements the transformer. It supports both video-only and audio-video generation modes. For actual usage, see the [`ltx-pipelines`](../ltx-pipelines/) package which handles model loading and initialization.

### Transformer Block Architecture

**Source**: [`src/ltx_core/model/transformer/transformer.py`](src/ltx_core/model/transformer/transformer.py)

Each dual-stream block performs four operations sequentially:

1. **Self-Attention**: Within-modality attention for each stream
2. **Text Cross-Attention**: Textual prompt conditioning for both streams
3. **Audio-Visual Cross-Attention**: Bidirectional inter-modal exchange
4. **Feed-Forward Network (FFN)**: Feature refinement

```text
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCK                        │
│                                                             │
│  VIDEO (14B): Input → RMSNorm → AdaLN → Self-Attn →         │
│              RMSNorm → Text Cross-Attn →                    │
│              RMSNorm → AdaLN → A↔V Cross-Attn (1D RoPE) →   │
│              RMSNorm → AdaLN → FFN → Output                 │
│                                                             │
│  AUDIO (5B):  Input → RMSNorm → AdaLN → Self-Attn →         │
│              RMSNorm → Text Cross-Attn →                    │
│              RMSNorm → AdaLN → A↔V Cross-Attn (1D RoPE) →   │
│              RMSNorm → AdaLN → FFN → Output                 │
│                                                             │
│  RoPE: Video=3D (x,y,t), Audio=1D (t), Cross-Attn=1D (t)    │
│  AdaLN: Timestep-conditioned, cross-modality for A↔V CA     │
└─────────────────────────────────────────────────────────────┘
```

### Audio-Visual Cross-Attention Details

Bidirectional cross-attention enables tight temporal alignment: video and audio streams exchange information bidirectionally using 1D temporal RoPE (synchronization only, no spatial alignment). AdaLN gates condition on each modality's timestep for cross-modal synchronization.

### Perturbations

The transformer supports [**perturbations**](src/ltx_core/guidance/perturbations.py) that selectively skip attention operations.

Perturbations allow you to disable specific attention mechanisms during inference, which is useful for guidance techniques like STG (Spatio-Temporal Guidance).

**Supported Perturbation Types**:

- `SKIP_VIDEO_SELF_ATTN`: Skip video self-attention
- `SKIP_AUDIO_SELF_ATTN`: Skip audio self-attention
- `SKIP_A2V_CROSS_ATTN`: Skip audio-to-video cross-attention
- `SKIP_V2A_CROSS_ATTN`: Skip video-to-audio cross-attention

Perturbations are used internally by guidance mechanisms like STG (Spatio-Temporal Guidance). For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Video VAE

The Video VAE ([`src/ltx_core/model/video_vae/`](src/ltx_core/model/video_vae/)) encodes video pixels into latent representations and decodes them back.

### Architecture

- **Encoder**: Compresses `[B, 3, F, H, W]` pixels → `[B, 128, F', H/32, W/32]` latents
  - Where `F' = 1 + (F-1)/8` (frame count must satisfy `(F-1) % 8 == 0`)
  - Example: `[B, 3, 33, 512, 512]` → `[B, 128, 5, 16, 16]`
- **Decoder**: Expands `[B, 128, F, H, W]` latents → `[B, 3, F', H*32, W*32]` pixels
  - Where `F' = 1 + (F-1)*8`
  - Example: `[B, 128, 5, 16, 16]` → `[B, 3, 33, 512, 512]`

`ConvVideoDecoder` (above) is the default, single-forward-pass decoder. `model/video_vae/` also has `DiffusionVideoDecoder` -- a NATTEN-based decoder (requires the `natten` extra: `uv sync --package ltx-core --extra natten`) that iteratively denoises pixels via Euler steps (`default_num_inference_steps=2` for distilled) instead of a single deterministic pass. `VideoDecoderConfigurator.from_config` picks between the two based on the checkpoint's own `vae._class_name`. `DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER` is its checkpoint key-remapping `SDOps`. Compile is pipeline-side via `vae_compile` / `--vae-compile` (`build_compile_diffusion_decoder_op` ModuleOps; first decode pays cold compile).

The Video VAE is used internally by pipelines for encoding video pixels to latents and decoding latents back to pixels. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Audio VAE

The Audio VAE ([`src/ltx_core/model/audio_vae/`](src/ltx_core/model/audio_vae/)) processes audio spectrograms.

### Audio VAE Architecture

Compact neural audio representation optimized for diffusion-based training. Natively supports stereo: processes two-channel mel-spectrograms (16 kHz input) with channel concatenation before encoding.

- **Encoder**: `[B, mel_bins, T]` → `[B, 8, T/4, 16]` latents (4× temporal downsampling, 8 channels, 16 mel bins in latent space, ~1/25s per token, 128-dim feature vector)
- **Decoder**: `[B, 8, T, 16]` → `[B, mel_bins, T*4]` mel spectrogram
- **Vocoder**: HiFi-GAN-based, modified for stereo synthesis and upsampling (16 kHz mel → 24 kHz waveform, doubled generator capacity for stereo)

**Downsampling**:

- Temporal: 4× (time steps)
- Frequency: Variable (input mel_bins → fixed 16 in latent space)

The Audio VAE is used internally by pipelines for encoding mel spectrograms to latents and decoding latents back to mel spectrograms. The vocoder converts mel spectrograms to audio waveforms. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Text Encoding (Gemma)

LTX-2 uses **Gemma 3** (Gemma 3-12B) as the multilingual text encoder backbone, located in [`src/ltx_core/text_encoders/gemma/`](src/ltx_core/text_encoders/gemma/). Advanced text understanding is critical not only for global language support but for the phonetic and semantic accuracy of generated speech.

### Text Encoder Architecture

The text conditioning pipeline consists of three stages:

1. **Gemma 3 Backbone**: Decoder-only LLM processes text tokens → embeddings across all layers `[B, T, D, L]`
2. **Multi-Layer Feature Extractor**: Aggregates features from all decoder layers (not just final layer), applies mean-centered scaling, flattens to `[B, T, D×L]`, and projects via learnable matrix W (jointly optimized with LTX-2, LLM weights frozen)
3. **Text Connector**: Bidirectional transformer blocks with learnable registers (replacing padded positions, also referred to as "thinking tokens" in the paper) for contextual mixing. Separate connectors for video and audio streams (`Embeddings1DConnector`)

**Encoders**:

- `AVGemmaTextEncoderModel`: Audio-video generation (two connectors → `AVGemmaEncoderOutput` with separate video/audio contexts)
- `VideoGemmaTextEncoderModel`: Video-only generation (single connector → `VideoGemmaEncoderOutput`)

### System Prompts

System prompts are also used to enhance user's prompts.

- **Text-to-Video**: [`gemma_t2v_system_prompt.txt`](src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt)
- **Image-to-Video**: [`gemma_i2v_system_prompt.txt`](src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_i2v_system_prompt.txt)

**Important**: Video and audio receive **different** context embeddings, even from the same prompt. This allows better modality-specific conditioning and enables the model to synthesize speech that is synchronized with visual lip movement while being natural in cadence, accent, and emotional tone.

**Output Format**:

- Video context: `[B, seq_len, 4096]` - Video-specific text embeddings
- Audio context: `[B, seq_len, 2048]` - Audio-specific text embeddings

The text encoder is used internally by pipelines. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Upscaler

The Upscaler ([`src/ltx_core/model/upsampler/`](src/ltx_core/model/upsampler/)) upsamples latent representations for higher-resolution output.

The spatial upsampler is used internally by two-stage pipelines (e.g., [`TI2VidTwoStagesPipeline`](../ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py), [`ICLoraPipeline`](../ltx-pipelines/src/ltx_pipelines/ic_lora.py)) to upsample low-resolution latents before final VAE decoding. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Data Flow

### Complete Generation Pipeline

Here's how all the components work together conceptually ([`src/ltx_core/components/`](src/ltx_core/components/)):

**Pipeline Steps**:

1. **Text Encoding**: Text prompt → Gemma encoder → separate video/audio embeddings
2. **Latent Initialization**: Initialize noise latents in spatial format `[B, C, F, H, W]`
3. **Patchification**: Convert spatial latents to sequence format `[B, seq_len, dim]` for transformer
4. **Sigma Schedule**: Generate noise schedule (adapts to token count)
5. **Denoising Loop**: Iteratively denoise using transformer predictions
   - Create Modality inputs with per-token timesteps and RoPE positions
   - Forward pass through transformer (conditional and unconditional for CFG)
   - Apply guidance (CFG, STG, etc.)
   - Update latents using diffusion step (Euler, etc.)
6. **Unpatchification**: Convert sequence back to spatial format
7. **VAE Decoding**: Decode latents to pixel space (with optional upsampling for two-stage)

- [`TI2VidTwoStagesPipeline`](../ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py) - Two-stage text-to-video (recommended)
- [`ICLoraPipeline`](../ltx-pipelines/src/ltx_pipelines/ic_lora.py) - Video-to-video with IC-LoRA control
- [`DistilledPipeline`](../ltx-pipelines/src/ltx_pipelines/distilled.py) - Fast inference with distilled model
- [`KeyframeInterpolationPipeline`](../ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py) - Keyframe-based interpolation

See the [ltx-pipelines README](../ltx-pipelines/README.md) for usage examples.

## 🔗 Related Projects

- **[ltx-pipelines](../ltx-pipelines/)** - High-level pipeline implementations for text-to-video, image-to-video, and video-to-video
- **[ltx-trainer](../ltx-trainer/)** - Training and fine-tuning tools
