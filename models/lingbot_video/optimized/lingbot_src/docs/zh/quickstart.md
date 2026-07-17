# 快速开始

这页给出从安装到生成视频的最短公开流程。更详细的配置见
[Prompt 准备](prompt_preparation.md)、[DiT 推理](dit_inference.md) 和
[SGLang Diffusion](sglang_diffusion.md)。

## 1. 安装

```bash
git clone <repo_url>
cd <repo_name>

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

pip install -r requirements.txt
pip install -e .
```

基础 `requirements.txt` 已覆盖两条默认路径，无需额外安装：

- diffusers 推理（`scripts/inference.py --backend diffusers`）；
- prompt rewriter 的 transformers backend（`rewriter/inference.py --backend transformers`）。

### 可选加速依赖

> **💡 Rewriter 生产部署**：内置 rewriter 只提供单进程 `transformers` backend。
> 若要高吞吐，建议自行部署 VLM 并通过 OpenAI 兼容接口调用。需要保持 step 1 使用
> 不挂 LoRA 的 base VLM，step 2 使用同一个 base VLM 并启用 rewriter LoRA。adapter
> 路由方式参考官方文档 [vLLM](https://docs.vllm.ai) / [SGLang](https://docs.sglang.ai)。

DiT 加速可选 SGLang Diffusion：

```bash
python -m pip install --no-deps -r requirements-sglang.txt
```

推荐关键版本：

| 包 | 版本 |
| --- | --- |
| `torch` | `2.12.0.dev20260220+cu130` |
| `torchvision` | `0.26.0.dev20260220+cu130` |
| `transformers` | `5.8.1` |
| `diffusers` | `0.39.0` |
| `peft` | `0.19.1` |

## 2. 模型目录配置

从 [模型下载](../../README.md#-model-download) 获取 DiT 权重包和 rewriter 权重，
下载后把下面三个环境变量指向对应目录。rewriter 脚本会自动从这些环境变量读取权重：

```bash
export MODEL_DIR="<path_to_lingbot-video-model>"          # DiT 权重包根目录
export REWRITER_BASE_MODEL="<path_to_rewriter_base_vlm>"  # rewriter base VLM
export REWRITER_ADAPTER="<path_to_rewriter_lora>"         # rewriter LoRA
```

## 3. 端到端 T2V 示例

先把普通 prompt 改写成 DiT 需要的结构化 JSON prompt（详见 [Prompt 准备](prompt_preparation.md)）：

```bash
python rewriter/inference.py \
  --backend transformers \
  --mode t2v \
  --prompt "<plain_user_prompt>" \
  --duration 5 \
  --output prompt.json
```

然后生成样本级 negative prompt：

```bash
python rewriter/auto_negative.py \
  --backend transformers \
  --mode t2v \
  --caption prompt.json \
  --output negative.json
```

用 diffusers 跑 base 视频生成：

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --mode t2v \
  --prompt_json prompt.json \
  --negative_prompt_json negative.json \
  --output "outputs/base.mp4" \
  --height 480 \
  --width 832 \
  --fps 24 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3 \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32
```

runner 会自动创建 `--output` 指定的目录，无需手动 `mkdir`。

如果跳过 auto negative，去掉 `--negative_prompt_json negative.json`，runner 会使用对应模式的内置默认 negative prompt。详细的配置说明和 refiner 使用见 [DiT 推理](dit_inference.md)。
