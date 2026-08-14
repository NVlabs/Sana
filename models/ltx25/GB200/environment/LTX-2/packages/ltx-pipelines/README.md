# LTX-2 Pipelines

High-level pipeline implementations for generating audio-video content with Lightricks' **LTX-2** model. This package provides ready-to-use pipelines for text-to-video, image-to-video, video-to-video, audio-to-video, keyframe interpolation, and retake tasks.

Pipelines are built using building blocks from [`ltx-core`](../ltx-core/) (schedulers, guiders, noisers, patchifiers) and handle the complete inference flow including model loading, encoding, decoding, and file I/O.

**Key Features:**

- 🎬 **Multiple Pipeline Types**: Text-to-video, image-to-video, video-to-video, audio-to-video, keyframe interpolation, and retake
- ⚡ **Optimized Performance**: Support for FP8 transformers, gradient estimation, and memory optimization
- 🎯 **Production Ready**: Two-stage pipelines for best quality output
- 🔧 **LoRA Support**: Easy integration with trained LoRA adapters
- 📦 **Self-Contained**: Handles model loading, encoding, decoding, and file I/O
- 🚀 **CLI Support**: All pipelines can be run as command-line scripts

## Quick Start

```bash
# From the repository root
uv sync --frozen

# Run a pipeline (example: two-stage text-to-video)
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path path/to/checkpoint.safetensors \
    --distilled-lora path/to/distilled_lora.safetensors 0.8 \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --gemma-root path/to/gemma \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4
```

See [Installation & Usage](docs/installation.md) for full setup, CLI modules, and shared flags.

## 📚 Documentation

| Topic | Description |
| ----- | ----------- |
| [Installation & Usage](docs/installation.md) | Install, requirements, running pipelines from the CLI, common flags |
| [Pipeline Selection Guide](docs/pipeline-selection.md) | Decision tree + feature comparison to pick the right pipeline |
| [Available Pipelines](docs/pipelines.md) | Full reference for all 11 pipelines |
| [Conditioning Types](docs/conditioning.md) | Image and video conditioning methods |
| [Multimodal Guidance](docs/multimodal-guidance.md) | CFG / STG / modality guidance parameters and tuning |
| [Optimization Tips](docs/optimization.md) | FP8 quantization, `torch.compile`, gradient estimation |
| [Multi-GPU Inference](docs/multigpu/README.md) | Run a single generation across GPUs for latency (SP, TDP, distributed VAE, distributed Gemma) |

## 🔗 Related Projects

- **[LTX-Core](../ltx-core/)** - Core model implementation and inference components (schedulers, guiders, noisers, patchifiers)
- **[LTX-Trainer](../ltx-trainer/)** - Training and fine-tuning tools
