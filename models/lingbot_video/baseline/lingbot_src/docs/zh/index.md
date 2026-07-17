# LingBot-Video 文档

公开文档按推理工作流组织。第一次使用建议从“快速开始”读起，然后根据需要进入对应的详细页面。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [快速开始](quickstart.md) | 安装依赖、设置模型目录，并跑通一次完整生成。 |
| [Prompt 准备](prompt_preparation.md) | 把普通 prompt 改写成结构化 JSON caption，并按需生成样本级 negative prompt。 |
| [Diffusers 推理](dit_inference.md) | 使用 diffusers 路径运行 T2I、T2V、TI2V、Refinement。 |
| [SGLang Diffusion 推理](sglang_diffusion.md) | 使用 SGLang Diffusion 并行加速推理。 |

## 推荐流程

1. 用 prompt rewriter 把普通 prompt 转成结构化 prompt。
2. 按需运行 auto negative，为该样本生成 negative prompt，或者使用默认设置的negative prompt。
3. 用 `scripts/inference.py` 运行 DiT 推理。
4. 选择 `--backend diffusers` 或者 `--backend sglang` 进行推理。