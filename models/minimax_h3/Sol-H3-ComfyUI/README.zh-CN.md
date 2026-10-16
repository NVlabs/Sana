# Sol-H3 in ComfyUI

[English](README.md) · 简体中文

**在一张本地 RTX 5090 上，用 ComfyUI 里的创作工作台生成带声音的 MiniMax-H3 视频。**

ComfyUI 作为底座，**Sol-H3 Studio** 是盖在画布上的创作界面：写提示词、选模式和流程、点 **Generate**，
看着进度走完，5 秒带原生音频的视频就会直接播放。后面是 Sol 引擎，在独立进程里运行模型；需要时随时切回节点图。

https://github.com/user-attachments/assets/20de4fc3-d9c7-4c64-a84e-da870deedfe5

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

- **Sol-H3 Studio**：覆盖在 ComfyUI 上的全屏工作台，中英文可切换，带实时进度、作品墙和作品库。模型在独立的
  worker 进程里运行，不改动 ComfyUI 的依赖；随时切到 **Nodes** 编辑同一张节点图。
- **单阶段**：[HyperFlow](../HyperFlow/) 8 步 LoRA 合并进 FP8 DiT，配合精确 AdaLN 查表、Sol-Attn 和分块编译。
- **两阶段**：[`Sol-H3-RTX5090`](../Sol-H3-RTX5090/) 的 H3 草稿 + LTX 精修，DGX Spark 上用 [`Sol-H3-Spark`](../Sol-H3-Spark/)。
- **文生视频、首帧、参考图；768p 或 480p，16:9，5 秒**，带音频。两阶段支持 768p 文生视频。
- **32 GB 显卡可用**：DiT 按块流式加载，Qwen 在请求之间停放在主机内存。

## 快速开始

**环境**

- Linux，一张 RTX 5090（32 GB），至少 90 GB 内存
- ComfyUI 0.30 或更新
- 运行环境：Python 3.12、PyTorch 2.12（CUDA 13）、Triton 3.7、FastVideo，以及为 SM120 编译的
  FlashAttention 4 和 SageAttention 2.2
- 单阶段另需一个依赖层：Diffusers 0.40、PEFT 和 [`hyperflow_h3`](../HyperFlow/hyperflow_h3/)

**权重**

- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [NVFP4 Qwen 编码器](https://huggingface.co/Comfy-Org/MiniMax-H3)（`text_encoders/`）
- [HyperFlow 8 步 LoRA](https://huggingface.co/videorebirth/hyperflow)
- [FastH3 dense-datafree adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)
- 蒸馏非因果 VAE：可选，尚未公开。不提供时单阶段用原版 H3 VAE 解码，每条约慢 10 秒。

**安装**

```bash
git clone -b sol-engine https://github.com/NVlabs/Sana.git
ln -s "$PWD/Sana/models/minimax_h3/Sol-H3-ComfyUI" /path/to/ComfyUI/custom_nodes/
```

1. 构建一次单阶段权重：FP8 DiT、AdaLN 查表和参考图 DiT（[步骤](docs/install.md#one-stage)）。两阶段按
   [`Sol-H3-RTX5090`](../Sol-H3-RTX5090/docs/setup.md) 准备。
2. 用你的路径运行 `scripts/configure.py`（[完整命令](docs/install.md#all-together)），它会写出 `config.json`
   并打印要 export 的 `SOL_ENGINE_CONFIG=...`。
3. 启动 ComfyUI，Sol-H3 Studio 会随之打开；第一次请求会加载并编译模型。

### 更多

- [安装说明](docs/install.md)：`configure.py` 全部选项、权重构建、两阶段、作品墙、远程 GPU
- [测试记录](docs/validation.md)：计时方法
- 只看前端：在本目录运行 `python3 -m http.server`，打开 `scripts/studio_preview.html`（模拟后端，不需要 GPU）

## 致谢

Studio 参考了 [FreeVideo](https://github.com/FlashML-org/FreeVideo) 的模式。HyperFlow 来自
[Video Rebirth](https://github.com/Video-Rebirth/hyperflow)。模型执行基于 [Diffusers](https://github.com/huggingface/diffusers)、
[FastVideo](https://github.com/hao-ai-lab/FastVideo)、[FlashAttention](https://github.com/Dao-AILab/flash-attention) 和
[SageAttention](https://github.com/thu-ml/SageAttention)。MiniMax-H3 由 MiniMax 发布，受其自身模型许可约束。示例首帧由 Qwen-Image-2.1 生成。

## 许可

本目录代码采用 Apache-2.0（[LICENSE](LICENSE)、[NOTICE](NOTICE)）；外部组件见 [UPSTREAM_NOTICES.md](UPSTREAM_NOTICES.md)。
模型权重和外部运行时不包含在内，保留各自的条款。
