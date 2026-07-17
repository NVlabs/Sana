# Prompt 准备

LingBot-Video 的 DiT 推理推荐使用结构化 JSON Prompt，而不是直接把用户随手写的
自然语言 prompt 送进 DiT。推理 prompt 生成流程如下：

1. 把普通 prompt 改写成 `prompt.json`。
2. 按需运行 auto negative，生成 `negative.json`。

## 结构化 JSON Prompt

rewriter 保存的文件格式是：

```json
{
  "caption": {
    "...": "structured JSON caption"
  },
  "duration": 5
}
```

完整的真实示例见 `assets/cases/<mode>/example_*/prompt.json`（`mode` 取 `t2i`、
`t2v`、`ti2v`）。

视频任务中，如果 `prompt.json` 包含 `duration`，runner 会根据 `duration` 和
`fps` 自动计算 `num_frames`。也可以显式传 `--num_frames` 覆盖。

**⚠️ 注意：结构化 caption 的 `prominent_elements[].actions[]` 里带有形如
`[0.0s - 5.0s]` 的动作时间戳，`--duration` 必须与这些时间戳覆盖的总时长一致
（例如动作最晚到 `5.0s`，就设 `--duration 5`）。**

## Prompt Rewriter

Prompt Rewriter 分两步把普通 prompt 转成结构化 JSON Prompt：

```text
user prompt (+ first frame, TI2V only)
   step1 EXPAND  (base VLM)        -> intermediate detailed caption
   step2 MAP     (base VLM + LoRA) -> structured JSON caption
```

两步共享同一个 base VLM。第二步会挂载 rewriter LoRA。TI2V 会把首帧图像同时送入两个阶段。

**💡 推荐：第一步 EXPAND 扩写不依赖 rewriter LoRA，可以换用更强的 VLM，以获得更丰富准确的中间 detailed caption。**

设置 Rewriter 权重：

```bash
export REWRITER_BASE_MODEL="<path_to_rewriter_base_vlm>"
export REWRITER_ADAPTER="<path_to_rewriter_lora>"
```

仓库自带的 rewriter 只提供 `transformers` backend（单进程、参考实现），已包含在
`requirements.txt` 中，无需额外安装。

> **💡 生产部署建议**：`transformers` backend 逐条推理、吞吐较低。若要提升吞吐，
> 建议自行把 VLM 部署成推理 server，并通过 OpenAI 兼容接口调用，而不是走进程内
> backend。需要保持两阶段语义：step 1 必须使用不挂 rewriter LoRA 的 base VLM；
> step 2 必须使用同一个 base VLM 并启用 rewriter LoRA。可以用两个 endpoint，
> 也可以用一个能按请求选择 adapter 的 server。具体部署方式参考官方文档：
> [vLLM](https://docs.vllm.ai) 、[SGLang](https://docs.sglang.ai)。

### T2V

```bash
python rewriter/inference.py \
  --backend transformers \
  --mode t2v \
  --prompt "<plain_user_prompt>" \
  --duration 5 \
  --output prompt.json
```

### TI2V

这里的首帧需要和后续 DiT 推理使用同一张图：

```bash
python rewriter/inference.py \
  --backend transformers \
  --mode ti2v \
  --prompt "<plain_user_prompt>" \
  --first-frame "<first_frame.png>" \
  --duration 5 \
  --output prompt.json
```

### T2I

```bash
python rewriter/inference.py \
  --backend transformers \
  --mode t2i \
  --prompt "<plain_user_prompt>" \
  --output prompt.json
```

## Auto Negative Prompt

auto negative 会读取结构化 caption，然后从默认 negative prompt 词表中删除与推理 prompt 冲突的词。它是 delete-only 的 per-sample pruning：只删默认项，不新增专门针对某个 case 的词。

**💡 推荐：大多数情况下直接使用内置的默认 negative prompt 即可；若对生成结果不满意，再用 auto negative 针对该样本做微调。**

```bash
python rewriter/auto_negative.py \
  --backend transformers \
  --mode t2v \
  --caption prompt.json \
  --output negative.json
```

推理时把输出文件传给 DiT：

```bash
--negative_prompt_json negative.json
```

如果不启用 auto negative，就不要传 `--negative_prompt_json`。runner 会自动使用
内置默认 negative prompt：

- `t2v` 和 `ti2v` 使用视频默认 negative prompt。
- `t2i` 使用图片默认 negative prompt。
