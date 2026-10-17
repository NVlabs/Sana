# Sol-H3 in ComfyUI

English · [简体中文](README.zh-CN.md)

**MiniMax-H3 video with sound on one local RTX 5090, in a creative studio inside ComfyUI.**

ComfyUI is the harness. **Sol-H3 Studio** opens over the canvas: write a prompt, choose a mode
and a pipeline, click **Generate**, and watch the progress until the 5-second clip with native
audio plays. Behind it, the Sol engine runs the model in its own process; the node graph is
still there when you want it.

https://github.com/user-attachments/assets/20de4fc3-d9c7-4c64-a84e-da870deedfe5

| Studio | Gallery |
|---|---|
| ![Sol-H3 Studio](docs/design/studio-1920.png) | ![Gallery](docs/design/studio-gallery.jpg) |

## How long does a 5-second clip take?

| One 1344×768 clip, 121 frames with audio · model loaded | Device | Request → MP4 |
|---|---|---:|
| Original MiniMax-H3, 50 steps (BF16) | RTX 5090 | 989.0 s |
| 8-step LoRA (HyperFlow, BF16) | RTX 5090 | 193.6 s |
| **Sol-H3 · one stage** (8-step LoRA, FP8, Sol-Attn, compiled, fast VAE) | RTX 5090 | **67.2 s** |
| **Sol-H3 · two stage** (H3 draft + LTX refine, [`Sol-H3-RTX5090`](../Sol-H3-RTX5090/)) | RTX 5090 | **35.5 s** (27.9×) |
| **Sol-H3 · two stage** ([`Sol-H3-Spark`](../Sol-H3-Spark/)) | DGX Spark | **64.3 s** |

One stage also renders 480p (864×480) in 22.0 s. Timings run from request to the finished MP4
with audio, Qwen text encoding included, through ComfyUI; model loading is reported separately.
The one- and two-stage RTX 5090 rows were measured on an RTX 5090 host with a 90 GB RAM limit;
the DGX Spark row is the released Sol-H3-Spark package (mean of three). Details in
[validation](docs/validation.md).

## What you get

- **Sol-H3 Studio**: a full-screen workspace over ComfyUI in English or Chinese, with live
  progress, a gallery wall and a library of past videos. Models run in a separate worker process,
  so ComfyUI's own packages stay untouched; switch to **Nodes** to edit the same graph.
- **One stage**: the 8-step [HyperFlow](../HyperFlow/) LoRA merged into an FP8 DiT, with an exact
  AdaLN table, Sol-Attn and regional compile.
- **Two stage**: H3 draft + LTX refine from [`Sol-H3-RTX5090`](../Sol-H3-RTX5090/), or
  [`Sol-H3-Spark`](../Sol-H3-Spark/) on DGX Spark.
- **Text, first frame or reference images; 768p or 480p, 16:9, 5 s** with audio. Two stage takes
  text at 768p.
- **Fits a 32 GB card**: the DiT streams block by block, and Qwen waits in host RAM between requests.

## Getting started

**Requirements**

- Linux, one RTX 5090 (32 GB), at least 90 GB of host RAM
- ComfyUI 0.30 or newer
- A runtime environment: Python 3.12, PyTorch 2.12 (CUDA 13), Triton 3.7, FastVideo, and
  FlashAttention 4 and SageAttention 2.2 built for SM120
- For one stage, an overlay with Diffusers 0.40, PEFT and [`hyperflow_h3`](../HyperFlow/hyperflow_h3/)

**Weights**

- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [NVFP4 Qwen encoder](https://huggingface.co/Comfy-Org/MiniMax-H3) (`text_encoders/`)
- [HyperFlow 8-step LoRA](https://huggingface.co/videorebirth/hyperflow)
- [FastH3 dense-datafree adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)
- [Sol-H3-VAE](https://huggingface.co/Efficient-Large-Model/Sol-H3-VAE), the distilled noncausal decoder: optional. Without it, one stage decodes
  with the original H3 VAE, about 10 s slower per clip.

**Install**

```bash
git clone -b sol-engine https://github.com/NVlabs/Sana.git
ln -s "$PWD/Sana/models/minimax_h3/Sol-H3-ComfyUI" /path/to/ComfyUI/custom_nodes/
```

1. Build the one-stage checkpoints once: the FP8 DiT, its AdaLN table and the reference-image DiT
   ([steps](docs/install.md#one-stage)). For two stage, prepare
   [`Sol-H3-RTX5090`](../Sol-H3-RTX5090/docs/setup.md).
2. Run `scripts/configure.py` with your paths ([full command](docs/install.md#all-together)). It
   writes `config.json` and prints the `SOL_ENGINE_CONFIG=...` line to export.
3. Start ComfyUI. Sol-H3 Studio opens with it; the first request loads and compiles the models.

### More details

- [Installation guide](docs/install.md): every `configure.py` option, checkpoint builds, two
  stage, gallery, remote GPU
- [Validation](docs/validation.md): how the timings were measured
- Frontend only: run `python3 -m http.server` here and open `scripts/studio_preview.html`
  (stubbed backend, no GPU)

## Acknowledgements

The Studio follows the pattern of [FreeVideo](https://github.com/FlashML-org/FreeVideo).
HyperFlow is by [Video Rebirth](https://github.com/Video-Rebirth/hyperflow). Model execution
builds on [Diffusers](https://github.com/huggingface/diffusers),
[FastVideo](https://github.com/hao-ai-lab/FastVideo),
[FlashAttention](https://github.com/Dao-AILab/flash-attention) and
[SageAttention](https://github.com/thu-ml/SageAttention). MiniMax-H3 is by MiniMax and is subject
to its own model license. Showcase first frames were made with Qwen-Image-2.1.

## License

Apache-2.0 for the code in this directory ([LICENSE](LICENSE), [NOTICE](NOTICE)); external
components in [UPSTREAM_NOTICES.md](UPSTREAM_NOTICES.md). Model weights and external runtimes
are not included and keep their own terms.
