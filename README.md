<p align="center" style="border-radius: 10px">
  <img src="assets/sol-engine-logo.png" width="45%" alt="Sol-Engine logo"/>
</p>

<h3 align="center">
  Accelerated video-diffusion inference —
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/docs/pipelines/sana/">SANA-Video</a> ·
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/docs/pipelines/cosmos3/">Cosmos3-Super</a> ·
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/docs/pipelines/ltx/">LTX-2.3</a>
</h3>

<h3 align="center">
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/docs/">📖 Docs</a> &nbsp;|&nbsp;
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/docs/pipelines/sana/">Pipelines</a> &nbsp;|&nbsp;
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/cache/">Techniques</a> &nbsp;|&nbsp;
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/docs/installation/">Install</a>
</h3>

<p align="center">
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/"><img src="https://img.shields.io/badge/🏠_Homepage-Sol--Engine-76b900?style=flat-square" alt="Homepage"/></a>
  <a href="https://arxiv.org/abs/2606.23743"><img src="https://img.shields.io/badge/📄_arXiv-2606.23743-b31b1b?style=flat-square" alt="arXiv"/></a>
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/docs/"><img src="https://img.shields.io/badge/📖_Docs-github.io-blue?style=flat-square" alt="Docs"/></a>
  <a href="#-license"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square" alt="License"/></a>
</p>

<h4 align="center">
  Agent-native workflow · Full-stack acceleration techniques · A wide range of video generation models
</h4>

---

**Sol-Engine** is an efficiency-oriented inference codebase for high-resolution video
diffusion, built on [SGLang](https://github.com/sgl-project/sglang)'s `multimodal_gen`
runtime. It features an **agent-native inference workflow** and reduces three production
models into **one unambiguous acceleration line**. This is powered by a full-stack
solution composed of **five reusable acceleration techniques**, delivering a **2× to 3×
end-to-end speedup** across the three models. We are actively continuing development to
support a wider range of models.

## 📰 News

- **[2026/07/21]** 🔥 **SOL Attention merged** — [**SOL Attention**](techniques/sparse_backends/) block-sparse video attention lands as a acceleration technique and powers two new models: [**HunyuanVideo-13B**](models/hunyuan_video/) **~5.03×** and [**Wan2.1-T2V-14B**](models/wan21_t2v_14b/) **~3.48×** end-to-end. (paper of SOL Attention coming soon🚀)
- **[2026/07/15]** 🔥 **Three new models** — [Wan2.2 TI2V-5B](scripts/wan5b/run_optimized.sh) **~2.89×**, [Wan2.2-A14B](scripts/wan14b/run_optimized.sh) **~2.17×**, and [LingBot-Video](scripts/lingbot/run_optimized.sh) **~2.60×** end-to-end.
- **[2026/07/13]** ⚙️ **Agent workflow update** — refreshed the agent-native optimization workflow (a master orchestrator driving per-technique executor sub-agents with automatic quality gates). See the [agent-workflow](site_docs/agent-workflow.md) page.
- **[2026/06]** 📖 **Docs release** — full documentation site live: [3 pipeline designs + 5 acceleration techniques](https://nvlabs.github.io/Sana/Sol-Engine/docs/).
- **[2026/06]** 🔥 **SANA-Video** — EasyCache + kernel fusion + torch.compile → **~2.77×** end-to-end.
- **[2026/06]** 🔥 **LTX-2.3** — KWL fusion + cache + PISA + NVFP4 + token-prune → **~2.38×** end-to-end.
- **[2026/06]** 🔥 **Cosmos3-Super** — TeaCache + step-selective NVFP4 → **~2.27×** end-to-end (4×GB200).


## ⚡ Models & speedups

<div align="center">

| Model | Params | Acceleration line | Speedup |
|---|---|---|---|
| **[Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super)** | 64B | TeaCache + step-selective NVFP4 | **~2.27×** |
| **[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)** | 22B | kernel fusion + cache + PISA + NVFP4 + token-prune | **~2.38×** |
| **[SANA-Video](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers)** | 2B | EasyCache + kernel fusion + compile | **~2.77×** |
| **[Wan2.2 TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)** | 5B | EasyCache + kernel fusion + compile | **~2.89×** |
| **[Wan2.2-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)** | 14B (MoE) | kernel fusion + EasyCache + PISA | **~2.17×** |
| **[LingBot-Video](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b)** | 30B-A3B (MoE) | kernel fusion + refiner PISA + EasyCache | **~2.60×** |
| **[HunyuanVideo-13B](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)** | 13B | TeaCache + compile + [**SOL Attention**](techniques/sparse_backends/) | **~5.03×** |
| **[Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)** | 14B | kernel compile + EasyCache + [**SOL Attention**](techniques/sparse_backends/) | **~3.48×** |

</div>

<sub>GB200, warmup-excluded. SANA 480p (832×480, 81f, 50 steps); Cosmos3 1280×720, 189f, 35 steps; LTX 1088×1920, 241f. Wan-5B 704×1280, 121f, 50 steps (1 GPU); Wan-14B 720×1280, 81f, 40 steps (1 GPU); LingBot base 480×832→refiner 1088×1920, 121f (4 GPU CP4, same-topology baseline); HunyuanVideo 1280×720, 129f, 50 steps (1 GPU, hot-vs-hot); Wan2.1-14B 720×1280, 81f, 50 steps (1 GPU).</sub>

## 🧩 The five acceleration methods

Video diffusion inference exposes redundancy at three complementary levels. At
the **algorithm level**, adjacent denoising steps run structurally similar
computations over slowly changing latents, so cache can reuse or skip step
outputs. At the **model level**, long spatiotemporal sequences contain redundant
tokens and attention interactions, motivating sparse attention and token
pruning. At the **kernel level**, DiT blocks repeatedly launch memory-bound work
around GEMMs, layout movement, normalization, activation, and precision
conversion, which quantization and fusion reduce. Sol-Engine composes the five
methods across these levels.

<div align="center">

| # | Method | What it does |
|---|---|---|
| 1 | **[Cache](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/cache/)** | reuse a denoise step's output (TeaCache / EasyCache / fix-step) |
| 2 | **[Quantization](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/quant/)** | TransformerEngine NVFP4 4-bit, step-selective |
| 3 | **[Kernel fusion](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/kernel/)** | fuse the memory-bound DiT glue (AdaLN, QK-norm+RoPE, gates, FFN) |
| 4 | **[Sparse attention](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/sparse/)** | piecewise block-sparse video self-attention |
| 5 | **[Token pruning](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/token_prune/)** | drop low-salience video tokens at mid refine steps |

</div>

## 🌀 SOL Attention

[**SOL Attention**](techniques/sparse_backends/) is our block-sparse attention
technique for video DiTs: video-token self-attention runs through a sparse
kernel that computes only the most relevant key blocks, while text conditioning
and the earliest denoising steps stay exact dense to preserve quality. It plugs
into a model runtime through env-gated hooks (all flags off = byte-identical
baseline) and powers the full-optimization stacks of
[**HunyuanVideo-13B**](models/hunyuan_video/) (**~5.03×**) and
[**Wan2.1-T2V-14B**](models/wan21_t2v_14b/) (**~3.48×**).

## 🚀 Quick start (agent-native)

Sol-Engine is installed and launched the **agent-native** way. Rather than hand-running
the setup steps, you hand a coding agent — OpenAI **Codex** or **Claude Code** — a single
goal and let it create the environment, fetch the weights, and run all three models in
both `baseline` and `fullopt` settings, **troubleshooting and adapting the scripts to
your machine** as it goes.

From the repo root, give the agent this goal:

```text
/goal Execute the inference code for the three models using both baseline and full-opt
settings with the following requirements. Refer to AGENTS.md for the environment creation,
model download, and inference guides. For the environment, you need to create a new
environment. For model weights, you are allowed to reuse existing weights if they are
locally available; otherwise, you need to download them. Regarding adaptability, be aware
that the provided guides for environment creation, download scripts, and inference may
contain system incompatibilities, so you are expected to troubleshoot and adapt them to
your specific machine.
```

## 📖 Getting started

- 📚 **[Full documentation](https://nvlabs.github.io/Sana/Sol-Engine/docs/)** — a comprehensive guidebook to the whole project: pipeline designs, acceleration techniques, setup, and model references in one place
- 🛠️ **[Installation](https://nvlabs.github.io/Sana/Sol-Engine/docs/installation/)** — conda env, editable install, CUDA-JIT fixups, and the HF model repos + download helpers
- 🎬 **Optimized pipelines** — [SANA-Video](https://nvlabs.github.io/Sana/Sol-Engine/docs/pipelines/sana/) · [Cosmos3-Super](https://nvlabs.github.io/Sana/Sol-Engine/docs/pipelines/cosmos3/) · [LTX-2.3](https://nvlabs.github.io/Sana/Sol-Engine/docs/pipelines/ltx/) · [**HunyuanVideo-13B**](site_docs/pipelines/hunyuan.md) · [**Wan2.1-T2V-14B**](site_docs/pipelines/wan21_14b.md)
- ⚙️ **Acceleration techniques** — [Cache](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/cache/) · [Quantization](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/quant/) · [Kernel fusion](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/kernel/) · [Sparse attention](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/sparse/) · [Token pruning](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/token_prune/)

## ✅ To-do

- [x] **SANA-Video** acceleration line — EasyCache + fusion + compile
- [x] **Cosmos3-Super** acceleration line — TeaCache + step-selective NVFP4
- [x] **LTX-2.3** acceleration line — KWL fusion + cache + PISA + NVFP4 + token-prune
- [ ] More backends for each acceleration method
- [ ] Agent-native workflow without human-in-the-loop

## 🙏 Acknowledgements

Built on [SGLang](https://github.com/sgl-project/sglang) and
[🤗 Diffusers](https://github.com/huggingface/diffusers). Pipelines wrap
[SANA-Video](https://github.com/NVlabs/Sana), NVIDIA
[Cosmos](https://github.com/NVIDIA/Cosmos), and
[Lightricks LTX-Video](https://github.com/Lightricks/LTX-Video). Acceleration methods
draw on TeaCache, EasyCache, SVDQuant/Nunchaku, FlashAttention,
[TransformerEngine](https://github.com/NVIDIA/TransformerEngine), and the sparse-attention
/ token-reduction literature surveyed in the [docs](https://nvlabs.github.io/Sana/Sol-Engine/docs/).

## 📌 Citation

```bibtex
@misc{li2026solvideoinferenceengine,
  title         = {Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation},
  author        = {Yitong Li and Junsong Chen and Haopeng Li and Haozhe Liu and Jincheng Yu and Ligeng Zhu and Ping Luo and Song Han and Enze Xie},
  year          = {2026},
  eprint        = {2606.23743},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2606.23743},
  url           = {https://arxiv.org/abs/2606.23743},
}
```

## 📄 License

Code in this repository is released under the Apache-2.0 license. The paper is
available on arXiv under the arXiv.org perpetual, non-exclusive distribution
license. Model weights follow their respective upstream licenses (SANA-Video,
NVIDIA Cosmos, Lightricks LTX) — see each model card.
