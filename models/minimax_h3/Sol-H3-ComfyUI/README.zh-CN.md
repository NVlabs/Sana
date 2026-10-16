# Sol-H3 in ComfyUI

[English](README.md) · 简体中文

**在一张本地 RTX 5090 上，用 ComfyUI 里的创作工作台生成带声音的 MiniMax-H3 视频。**

ComfyUI 作为底座，**Sol-H3 Studio** 是盖在画布上的创作界面：写提示词、选模式和流程、点 **Generate**，
看着进度走完，5 秒带原生音频的视频就会直接播放。后面是 Sol 引擎，在独立进程里运行模型；需要时随时切回节点图。

[![Sol-H3 in ComfyUI 宣传片](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Sol-Engine/Sol-H3-ComfyUI/film/sol-h3-in-comfyui-poster.jpg)](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Sol-Engine/Sol-H3-ComfyUI/film/sol-h3-in-comfyui.mp4)

| 创作界面 | 作品墙 |
|---|---|
| ![Sol-H3 Studio](docs/design/studio-1920.png) | ![作品墙](docs/design/studio-gallery.jpg) |

## 生成一段 5 秒视频要多久？

| 一段 1344×768、121 帧、含音频的视频 · 模型已加载 | 设备 | 请求到 MP4 |
|---|---|---:|
| 原版 MiniMax-H3，50 步（BF16） | RTX 5090 | 989.0 秒 |
| 8 步 LoRA（HyperFlow，BF16） | RTX 5090 | 193.6 秒 |
| **Sol-H3 · 单阶段**（8 步 LoRA、FP8、Sol-Attn、编译、快速 VAE） | RTX 5090 | **67.2 秒** |
| **Sol-H3 · 两阶段**（H3 草稿 + LTX 精修，[`Sol-H3-RTX5090`](../Sol-H3-RTX5090/)） | RTX 5090 | **35.5 秒**（27.9 倍） |
| **Sol-H3 · 两阶段**（[`Sol-H3-Spark`](../Sol-H3-Spark/)） | DGX Spark | **64.3 秒** |

单阶段还支持 480p（864×480），22.0 秒。计时从请求到带音频的 MP4 完成，包含 Qwen 文本编码，全程经过 ComfyUI；
模型加载单独记录。RTX 5090 的单阶段和两阶段在一台内存上限 90 GB 的 RTX 5090 主机上实测；DGX Spark 一行是已发布的
Sol-H3-Spark 包三次平均。详见 [测试记录](docs/validation.md)。

## 功能

- **Sol-H3 Studio**：覆盖在 ComfyUI 上的全屏工作台，中英文可切换（右上角）：提示词与示例、模式、流程、分辨率、
  seed、实时进度（编码 → 第 k/8 步 → 解码）、成片播放与下载、最近作品和作品库。随时切到 **Nodes**，Studio 编辑的是同一张节点图。
- **两种流程**：*单阶段* 用 [HyperFlow](../HyperFlow/) 8 步 LoRA 加 Sol-H3 加速——LoRA 合并进 FP8 DiT、用精确查表
  替换 26 GB 的 AdaLN 投影、Sol-Attn 稀疏注意力（dense 步用 SageAttention）、逐块 `torch.compile`、蒸馏的非因果 VAE；
  *两阶段* 原样调用 [`Sol-H3-RTX5090`](../Sol-H3-RTX5090/) 的 H3 草稿 + LTX 精修（DGX Spark 上用 [`Sol-H3-Spark`](../Sol-H3-Spark/)）。
- **文生视频、首帧、参考图**（最多 4 张），在单阶段上运行。
- **768p 或 480p**，16:9，5 秒。
- **作品墙**：所有示例同时播放，悬停试听，点击即可复用它的提示词和首帧。
- **32 GB 显卡可用**：DiT 按块流式加载，解码器只在运行时占用显卡；Qwen 常驻，请求之间停放在主机内存。
- **进程隔离**：模型在独立 Python 环境的 worker 中运行，不改动 ComfyUI 的依赖；取消会停止 worker。

## 快速开始

从权重到 Studio 的完整步骤；所有选项见 [docs/install.md](docs/install.md)。

1. **运行环境**：Linux、一张 RTX 5090（32 GB）、≥ 90 GB 内存、CUDA 13 驱动。一个 Python 3.12 环境
   （PyTorch 2.12.0+cu130、Triton 3.7.0）运行 Qwen 和 H3 DiT，另装 FastVideo + FlashAttention 4（SM120）和为 SM120
   编译的 SageAttention 2.2。单阶段还需要一个包含 Diffusers 0.40、PEFT 和 [`hyperflow_h3`](../HyperFlow/hyperflow_h3/) 的小依赖层。
2. **权重**：[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[NVFP4 Qwen 编码器](https://huggingface.co/Comfy-Org/MiniMax-H3)
   （`text_encoders/`）、[HyperFlow 8 步 LoRA](https://huggingface.co/videorebirth/hyperflow)、
   [非因果 VAE](https://huggingface.co/hp-l33/h3-fast-vae-noncausal)、
   [FastH3 dense-datafree adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)（4 步与基线配置用）。
3. **单阶段权重**（在 GPU 主机上做一次，见 [安装说明](docs/install.md#one-stage)）：
   ```bash
   LORA=minimax_h3_hyperflow_8step_v1.0.safetensors
   python scripts/hyperflow/convert_fp8.py MiniMax-H3/transformer $LORA transformer_hf8
   python scripts/hyperflow/build_adaln_table.py OPTIONS.json t2va.pt T2VA_PAYLOAD.pt
   python scripts/hyperflow/build_adaln_table.py OPTIONS.json fl2va.pt FL2VA_PAYLOAD.pt
   python scripts/hyperflow/merge_adaln_tables.py transformer_hf8/adaln_table.pt t2va.pt fl2va.pt
   python scripts/hyperflow/convert_ref.py \
     https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/transformer_ref \
     $LORA transformer_hf8/adaln_table.pt transformer_ref_hf8    # 参考图
   ```
4. **两阶段**（可选）：按 [`../Sol-H3-RTX5090`](../Sol-H3-RTX5090/docs/setup.md) 准备，它的 `prepare.py` 会写出 `paths-5090.json`。
5. **装进 ComfyUI**（0.30 或更新）并写主机配置：
   ```bash
   ln -s "$PWD" /path/to/ComfyUI/custom_nodes/Sol-H3-ComfyUI    # 在本目录执行
   python3 scripts/fetch_assets.py gallery                          # 可选：作品墙示例
   python3 scripts/configure.py \
     --reuse-paths /path/to/h3-runtime/paths.json \
     --dense-lora /path/to/dense-datafree/adapter_model.safetensors \
     --hyperflow-weights /path/to/minimax_h3_hyperflow_8step_v1.0.safetensors \
     --hyperflow-deps /path/to/hyperflow-overlay \
     --hyperflow-fp8-dir /path/to/transformer_hf8 \
     --hyperflow-adaln-table /path/to/transformer_hf8/adaln_table.pt \
     --hyperflow-fast-vae-weights /path/to/fast_h3_vae_noncausal_24l.safetensors \
     --fast-vae-code /path/to/fast-vae-code \
     --hyperflow-ref-fp8-dir /path/to/transformer_ref_hf8 \
     --two-stage-package "$PWD/../Sol-H3-RTX5090" \
     --two-stage-paths /path/to/paths-5090.json \
     --gallery "$PWD/assets/gallery" \
     --output-dir ~/sol-h3-settings --gpu 0
   export SOL_ENGINE_CONFIG=~/sol-h3-settings/config.json
   cd /path/to/ComfyUI && python main.py
   ```
6. **打开 Studio**：随 ComfyUI 自动打开（或 **Workflow → Browse Templates → Sol-H3-ComfyUI → Sol-H3-Studio**）。
   第一次请求会加载并编译模型，之后复用。`scripts/hyperflow/e2e_comfy.py` 可以通过 ComfyUI API 对同一张图测速。

只看前端：在本目录运行 `python3 -m http.server`，打开 `scripts/studio_preview.html`（模拟后端，不需要 GPU）。
宣传片的制作脚本在 [`scripts/promo/`](scripts/promo/)。

## 目录

| 路径 | 内容 |
|---|---|
| `__init__.py`、`sol_h3_engine/comfy_*.py`、`bridge.py`、`config.py` | ComfyUI 节点和 `/solh3/*` 接口（仅标准库）、worker 桥接、配置 |
| `sol_h3_engine/hyperflow_*.py` | 单阶段：FP8 线性层、AdaLN 查表、分块编译、引擎 |
| `sol_h3_engine/two_stage_worker.py` | 两阶段：调用 `Sol-H3-RTX5090` / `Sol-H3-Spark` |
| `sol_h3_engine/runtimes/` | NVFP4 Qwen worker，以及 H3 4 步 / 50 步基线运行时 |
| `web/` | Sol-H3 Studio |
| `scripts/` | `configure.py`、权重构建（`hyperflow/`）、`fetch_assets.py`、作品墙、宣传片 |
| `docs/`、`tests/` | 安装、测试记录、架构、Studio 设计；CPU 测试 |

## 限制

- 仅 16:9、5 秒（121 帧、24 fps）；单阶段支持 768p 或 480p，两阶段为 768p。
- 两阶段仅支持文生视频；首帧和参考图在单阶段上运行。
- 参考图请求需要在主机内存里切换 DiT（模式切换时 46–72 秒）。
- 每张 GPU 一次处理一个请求；尚未提供从零安装器。
- 单阶段（FP8、编译、蒸馏 VAE）与 BF16 不是逐位一致，只做了目视检查。

## 致谢

Studio 参考了 [FreeVideo](https://github.com/FlashML-org/FreeVideo) 的模式。HyperFlow 来自
[Video Rebirth](https://github.com/Video-Rebirth/hyperflow)。模型执行基于 [Diffusers](https://github.com/huggingface/diffusers)、
[FastVideo](https://github.com/hao-ai-lab/FastVideo)、[FlashAttention](https://github.com/Dao-AILab/flash-attention) 和
[SageAttention](https://github.com/thu-ml/SageAttention)。MiniMax-H3 由 MiniMax 发布，受其自身模型许可约束。示例首帧由 Qwen-Image-2.1 生成。

## 许可

本目录代码采用 Apache-2.0（[LICENSE](LICENSE)、[NOTICE](NOTICE)）；外部组件见 [UPSTREAM_NOTICES.md](UPSTREAM_NOTICES.md)。
模型权重和外部运行时不包含在内，保留各自的条款。
