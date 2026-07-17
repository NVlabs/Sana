# SGLang Diffusion

本页说明 SGLang Diffusion 推理:

```bash
python scripts/inference.py --backend sglang ...
```

SGLang Diffusion 的主要优势是高吞吐服务化和大批量并行生成。对于单条 prompt，
direct diffusers 可能持平甚至更快，因为分布式启动、通信和 fused kernel 初始化
开销可能占主导。建议在需要同时处理多个请求/多个样本、运行 CP/FSDP 布局，或使用
可选 MoE fused kernel 时使用 SGLang。

如果没有安装可选的 SGLang 包,`--backend sglang` 会自动回退到 direct diffusers
并打印警告。安装可选依赖以启用 SGLang runtime:

```bash
python -m pip install --no-deps -r requirements-sglang.txt
```

这里使用 `--no-deps`，目的是保持 `requirements.txt` 中已经验证过的
PyTorch/CUDA 组合不被 SGLang 的依赖解析改掉。

## GPU 布局

并行推理有两个独立的维度:

- `--cfg_parallel_degree`:拆分 classifier-free guidance 的两个分支(条件 /
  无条件),取值只能是 `1` 或 `2`。
- `--context_parallel_degree`:把序列在多卡间做 context parallel 切分。

context parallel 下可加 `--context_parallel_ulysses_anything`,用 Ulysses
all-to-all 通信,支持任意(非整除)序列长度切分;下面示例默认开启,一般无需改动。

进程数必须等于两个维度的乘积:

```
nproc_per_node = cfg_parallel_degree × context_parallel_degree
```

根据你手上的 GPU 数量选布局:

| GPU 数 | 布局 | 参数 |
| ---: | --- | --- |
| 1 | 单卡 | 不用 `torchrun`;`--cfg_parallel_degree 1 --context_parallel_degree 1` |
| 2 | CFG×CP | `--cfg_parallel_degree 2 --context_parallel_degree 1` |
| 4 | CFG×CP | `--cfg_parallel_degree 2 --context_parallel_degree 2` |
| N(偶数) | CFG×CP | `--cfg_parallel_degree 2 --context_parallel_degree N/2` |
| N | 仅 CP | `--cfg_parallel_degree 1 --context_parallel_degree N` |

下面的示例用 shell 变量,任意 GPU 数都适用。按你的机器设一次即可:

```bash
export CFG=2                 # 1 或 2
export CP=4                  # context-parallel 度
export NPROC=$((CFG * CP))   # 要启动的进程数
```

单卡:设 `CFG=1 CP=1`,直接 `python scripts/inference.py ...` 运行,不用
`torchrun`。

## FSDP 推理切分

当你希望 base DiT 和 refiner DiT 同时加载到 GPU、同时又把它们的 transformer
权重切到多个进程上时，添加 `--enable_fsdp_inference`。这个开关与 backend 无关：
使用 `--backend sglang` 时，SGLang 负责 native 执行路径，PyTorch composable FSDP
负责切分 DiT module。

FSDP 可以和上面的 SGLang 布局一起用：

- 仅 CP：`--cfg_parallel_degree 1 --context_parallel_degree <GPU数>`。
  如果希望使用 batched CFG，再添加 `--batch_cfg` 和 `--refiner_batch_cfg`。
- CFG×CP：不要开启 `--batch_cfg` / `--refiner_batch_cfg`，设置
  `--cfg_parallel_degree 2 --context_parallel_degree <GPU数/2>`。
- 只做 FSDP 显存切分：用 `torchrun --nproc_per_node <GPU数>` 启动，并保持两个
  parallel degree 都是 `1`。

当 `--run_refiner` 和 `--enable_fsdp_inference` 同时存在时，runner 会先加载
base DiT 和 refiner DiT，然后在 base 采样开始前同时切分两个 DiT。运行日志会分别
为 base 和 refiner 打印 `fsdp_inference=FSDPInferenceInfo(...)`。

这里做的是 GPU 显存切分，不是 meta-init loader。初始化阶段每个 rank 仍会先在
host memory 上构建一份 transformer，然后再做 FSDP 切分；大 MoE checkpoint 需要
确保机器有足够系统内存支撑当前进程数。

## Dense 运行

Dense 模型没有专家,所以 `LINGBOT_MOE_*` 这些变量对它无效。只需配置 attention
backend 和 GPU 布局:

```bash
export DIFFUSERS_ATTN_BACKEND=_native_flash

torchrun --standalone --nproc_per_node $NPROC scripts/inference.py \
  --backend sglang \
  --model_dir "$MODEL_DIR" \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --height 480 \
  --width 832 \
  --fps 24 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3 \
  --cfg_parallel_degree $CFG \
  --context_parallel_degree $CP \
  --context_parallel_ulysses_anything \
  --enable_fsdp_inference \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32
```

`--prompt_json` 指向的文件可以是一个 dict 或非空 list。每个样本要么包含
`caption` 字段(其值为结构化 prompt),要么本身就是结构化 prompt 对象。若包含
`duration`,帧数由 `duration` 和 `fps` 推导。

## MoE 运行 — Grouped Experts

MoE 模型多了专家内核。Grouped MoE 是默认的质量优先专家 backend:

```bash
export DIFFUSERS_ATTN_BACKEND=_native_flash
export LINGBOT_MOE_PAD_BACKEND=vectorized
export LINGBOT_MOE_EXPERT_BACKEND=grouped_mm

torchrun --standalone --nproc_per_node $NPROC scripts/inference.py \
  --backend sglang \
  --model_dir "$MODEL_DIR" \
  --run_refiner \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --refiner_output "outputs/t2v_refined.mp4" \
  --height 480 \
  --width 832 \
  --fps 24 \
  --steps 40 \
  --refiner_steps 8 \
  --guidance_scale 3 \
  --refiner_guidance_scale 3 \
  --shift 3 \
  --refiner_shift 3 \
  --refiner_t_thresh 0.85 \
  --refiner_sigma_tail_steps 2 \
  --cfg_parallel_degree $CFG \
  --context_parallel_degree $CP \
  --context_parallel_ulysses_anything \
  --enable_fsdp_inference \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32 \
  --refiner_vae_dtype fp32 \
  --reuse_condition_features
```

## MoE 运行 — 速度优先 FP8

当最看重视觉吞吐时用这条路径。它把 MoE 专家内核换成 FP8 SGLang Triton
路径。导出 FP8 专家 backend,然后跑与上面 grouped MoE 完全相同的
`torchrun ... scripts/inference.py` 命令:

```bash
export DIFFUSERS_ATTN_BACKEND=_native_flash
export LINGBOT_MOE_PAD_BACKEND=vectorized
export LINGBOT_MOE_EXPERT_BACKEND=sglang_triton_fp8
```

GPU 布局与上面 grouped MoE 完全一致(见「GPU 布局」)。

FP8 MoE 可能改变相对于 grouped MoE 的生成细节。速度优先时使用 FP8；更关注
数值可复现时使用 grouped MoE。

## 加速选择(MoE)

| 需求 | 设置 |
| --- | --- |
| 默认 grouped experts | `LINGBOT_MOE_EXPERT_BACKEND=grouped_mm` |
| 快速看效果 | `LINGBOT_MOE_EXPERT_BACKEND=sglang_triton_fp8` |
| DiT 显存切分 | 添加 `--enable_fsdp_inference` |

GPU 布局(CFG×CP / 仅 CP)见上文「GPU 布局」章节。

8-GPU CP + FSDP smoke test 可以直接使用下面的 public 脚本。脚本默认走 direct
diffusers；如果要测试 SGLang Diffusion，在外部设置 `BACKEND=sglang`。

```bash
MODEL_DIR="<path_to_lingbot-video-dense>" ./scripts/multi-gpus/run_dense_t2i_fsdp_cp8.sh
MODEL_DIR="<path_to_lingbot-video-dense>" ./scripts/multi-gpus/run_dense_t2v_fsdp_cp8.sh
MODEL_DIR="<path_to_lingbot-video-dense>" ./scripts/multi-gpus/run_dense_ti2v_fsdp_cp8.sh

MODEL_DIR="<path_to_lingbot-video-moe>" ./scripts/multi-gpus/run_moe_t2i_fsdp_cp8.sh
MODEL_DIR="<path_to_lingbot-video-moe>" ./scripts/multi-gpus/run_moe_t2v_refiner_fsdp_cp8.sh
MODEL_DIR="<path_to_lingbot-video-moe>" ./scripts/multi-gpus/run_moe_ti2v_refiner_fsdp_cp8.sh
```

运行前先激活环境；如果没有激活环境，可以设置 `PYTHON_BIN="<path_to_python>"`。
