<div align="center">

<img src="assets/bernini-icon.png" width="560" alt="Bernini"/>

<h4 align="center">Latent Semantic Planning for Video Diffusion</h4>

**Chenchen Liu<sup>\*</sup>, Junyi Chen<sup>\*</sup>, Lei Li<sup>\*</sup>, Lu Chi<sup>\*,§</sup>, Mingzhen Sun<sup>\*</sup>, Zhuoying Li<sup>\*</sup>, Yi Fu, Ruoyu Guo, Yiheng Wu, Ge Bai, Zehuan Yuan<sup>✉</sup>**

<sup>\*</sup> Equal contribution&nbsp;&nbsp;<sup>✉</sup> Corresponding author&nbsp;&nbsp;<sup>§</sup> Project lead

[![arXiv](https://img.shields.io/badge/arXiv-2605.22344-b31b1b.svg)](https://arxiv.org/abs/2605.22344)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://bernini-ai.github.io/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow)](https://huggingface.co/collections/ByteDance/bernini)

</div>

## 🎉 News

- **[2026-06-11]** We open-sourced the inference code and model weights of the full Bernini (**Bernini**) on [ByteDance/Bernini-Diffusers](https://huggingface.co/ByteDance/Bernini-Diffusers).
- **[2026-06-09]** We open-sourced the **1.3B** weights of the Bernini Renderer (**Bernini-R**) on [ByteDance/Bernini-R-1.3B-Diffusers](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers). Fine-tuned from Wan2.1-1.3B, the model performs close to the 14B variant on simple tasks such as style transfer, subtitle or watermark removal, and local editing, while lagging behind on more complex tasks such as human generation.
- **[2026-06-01]** We open-sourced the inference code and model weights of the Bernini Renderer (**Bernini-R**) on [ByteDance/Bernini-R-Diffusers](https://huggingface.co/ByteDance/Bernini-R-Diffusers).
- **[2026-05-22]** We released our paper [Bernini: Latent Semantic Planning for Video Diffusion](https://arxiv.org/abs/2605.22344).

## ✨ Highlights

Bernini is a unified framework for video generation and editing that combines
an MLLM-based semantic planner with a DiT-based renderer.

On video editing, Bernini reaches the first tier among leading closed-source
commercial models. The leaderboard below comes from our self-built arena
platform, where human annotators blindly vote on paired edits and the votes are
aggregated into a Bradley-Terry score and a pairwise win-rate matrix.

<img src="assets/arena.png" width="900" alt="Video editing arena: Bradley-Terry leaderboard and pairwise win-rate matrix"/>

Benchmark results across the released models:

| Model | EditVerse | OpenVE | OpenS2V | VBench | Bernini-v2v (OS) | Bernini-rv2v (OS) |
|---|---|---|---|---|---|---|
| [Bernini-R 1.3B](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers) | 7.74 | 3.65 | 62.18 | 84.69 | 3.15 | 3.21 |
| [Bernini-R 14B](https://huggingface.co/ByteDance/Bernini-R-Diffusers) | 7.99 | 3.78 | 62.94 | 84.64 | 3.25 | 3.34 |
| [Bernini 7B+14B](https://huggingface.co/ByteDance/Bernini-Diffusers) | 8.02 | 4.03 | 62.30 | 84.37 | 3.49 | 3.48 |

## 🧾 Models

The repository provides two model families. Pick one and follow its guide for
weight download, inference commands, and ready-to-run scripts:

|  | **Bernini** | **Bernini-R** |
|--|-------------|---------------|
| What it is | Full pipeline: MLLM-based semantic planner + DiT-based renderer | Renderer-only model fine-tuned from the Wan diffusion renderer |
| Strengths | Decomposes complex instructions and plans semantic changes before rendering; stronger instruction following | Strong rendering and consistency with fewer moving parts; simpler setup |
| Checkpoints | [`ByteDance/Bernini-Diffusers`](https://huggingface.co/ByteDance/Bernini-Diffusers) | [`ByteDance/Bernini-R-Diffusers`](https://huggingface.co/ByteDance/Bernini-R-Diffusers) (14B) · [`ByteDance/Bernini-R-1.3B-Diffusers`](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers) · [`ByteDance/Bernini-R`](https://huggingface.co/ByteDance/Bernini-R) (separate ckpts) |
| Guide | **[docs/bernini.md](docs/bernini.md)** | **[docs/bernini_r.md](docs/bernini_r.md)** |

Both families share the same task interface: `t2i`, `i2i`, `t2v`, `v2v`,
`rv2v`, and `r2v`.

## 📦 Installation

### Requirements

- **Python** 3.11.2.
- **CUDA GPU** — a Hopper GPU (H100/H800/H200) is recommended so FlashAttention-3
  can be used; other CUDA GPUs fall back to FlashAttention-2 or PyTorch SDPA.
- **CUDA toolkit** 12.4 (matches the pinned `torch==2.5.1+cu124`; 12.3+ is the
  minimum if you build FlashAttention-3).
- Pinned in `requirements.txt`: `torch==2.5.1+cu124`, `diffusers==0.35.2`,
  `accelerate==0.34.2`, `transformers==4.57.3`.

Reference environment (developed and tested on this setup):

| Component | Version      |
|-----------|--------------|
| GPU       | NVIDIA H100  |
| CUDA      | 12.4         |
| Python    | 3.11.2       |
| PyTorch   | 2.5.1+cu124  |

### Install

```bash
git clone https://github.com/bytedance/Bernini.git bernini && cd bernini
pip install -r requirements.txt
# Open-VeOmni is required. Install it with --no-deps so it does not pull in a
# different torch build and override the pinned torch==2.5.1+cu124:
pip install --no-deps git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.10
```

[Open-VeOmni](https://github.com/ByteDance-Seed/VeOmni) (Apache-2.0,
Python 3.11) is a **required** dependency — all inference paths import it,
including single-GPU.

Optional extras:

- **Faster attention** (FlashAttention-2 by default):
  - FlashAttention-2 — general CUDA GPUs (incl. A100/A800): `pip install flash-attn==2.8.3`.
  - FlashAttention-3 — Hopper only (H100/H800/H200, CUDA ≥ 12.3, PyTorch ≥ 2.4).
    `flash_attn_interface` is not on PyPI; build it from the
    [flash-attention](https://github.com/Dao-AILab/flash-attention) repo's
    `hopper/` directory at tag `v2.8.3`:
    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention && git checkout v2.8.3
    cd hopper && MAX_JOBS=$(nproc) python3 setup.py install --user
    ```

## 🚀 Usage

Weight download and per-task inference commands are model-specific — follow
**[docs/bernini.md](docs/bernini.md)** or
**[docs/bernini_r.md](docs/bernini_r.md)**. The pieces below are shared by both
pipelines.

### Case files

A run is described by a **case file** — a small JSON under
[`assets/testcases/`](assets/testcases/) that bundles one task's routing and
inputs (`task_type`, `guidance_mode`, `prompt`, source media, `output`). This
keeps long prompts out of the command line. Each task has a directory under
`assets/testcases/` with one or more bundled examples; see the
[case-file format](assets/testcases/README.md).

### Prompt enhancer (highly recommended)

`--use_pe` enhances the prompt through an OpenAI-compatible endpoint and is
recommended for best generation quality. The `openai` SDK is installed by
`requirements.txt`; configure the endpoint with environment variables:

```bash
export BERNINI_PE_API_KEY=...      # or OPENAI_API_KEY
export BERNINI_PE_BASE_URL=...     # or OPENAI_BASE_URL
export BERNINI_PE_MODEL=...        # vision-capable chat model
```

### Gradio demo

`gradio_demo.py` exposes the same pipeline through a Gradio UI for both
**Bernini** and **Bernini-R**: the task-type dropdown auto-fills
`guidance_mode` (still user-editable), uploaded media is routed to the matching
slot, and the result is rendered inline. Launch commands are in each model's
guide ([Bernini](docs/bernini.md#gradio-demo) ·
[Bernini-R](docs/bernini_r.md#gradio-demo)).

Add `--use_pe` (with the prompt-enhancer environment variables above) to enable
prompt enhancement; the in-UI checkbox is a per-request switch on top of this
flag.

## 📑 Citation

If you use Bernini in your research, please cite:

```bibtex
@article{bernini,
  title   = {Bernini: Latent Semantic Planning for Video Diffusion},
  author  = {Chenchen Liu and Junyi Chen and Lei Li and Lu Chi and Mingzhen Sun and Zhuoying Li and Yi Fu and Ruoyu Guo and Yiheng Wu and Ge Bai and Zehuan Yuan},
  journal = {arXiv preprint arXiv:2605.22344},
  year    = {2026}
}
```

## 🙏 Acknowledgements

Bernini builds on several outstanding open-source projects:

- [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni)

We thank the authors and communities of these projects for their contributions.

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE).
