# DiT Inference

Use this guide for the direct diffusers reference path:

```bash
python scripts/inference.py --backend diffusers ...
```

The runner accepts `--mode t2i`, `--mode t2v`, and `--mode ti2v`.

**⚠️ Note: DiT inference does not accept raw natural-language prompts. You must first generate a structured JSON prompt with the [Rewriter](prompt_preparation.md), then pass it via `--prompt_json`.**

## Configure Model Path

```bash
export MODEL_DIR="<path_to_lingbot-video-model>"
```

If `prompt.json` contains `duration`, the runner derives frame count from
`duration` and `fps`. For video, `num_frames` must be `1` or `4n+1`; `5s` at
`24 fps` maps to `121` frames.

Append `--negative_prompt_json negative.json` when using auto negative. If it is
omitted, the runner uses the built-in default negative prompt for the selected
mode.

In all examples below, the directory given to `--output` (and `--refiner_output`)
is created automatically by the runner.

By default the runner prints concise model-loading status logs and shows the
denoising tqdm progress bar on the main process. Internal Hugging Face loader
progress bars are suppressed. Add `--quiet_progress` if you need quiet batch
logs.

## Example Assets

The repository ships ready-made structured JSON prompts (`prompt.json`) for each
mode under `assets/cases/`, and TI2V also includes the matching first frame
`first_frame.png`. You can use them directly.

See `assets/cases/manifest.json` for the full list.

## T2V Base Only

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --height 480 \
  --width 832 \
  --num_frames 121 \
  --fps 24 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3 \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32
```

## Base Plus Refiner

Pass `--run_refiner` to load and run the `refiner/` DiT from the same model
root. If `--run_refiner` is not passed, the runner does not load or run the
refiner.

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --run_refiner \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --refiner_output "outputs/t2v_refined.mp4" \
  --height 480 \
  --width 832 \
  --refiner_height 1088 \
  --refiner_width 1920 \
  --num_frames 121 \
  --fps 24 \
  --steps 40 \
  --refiner_steps 8 \
  --guidance_scale 3 \
  --refiner_guidance_scale 3 \
  --shift 3 \
  --refiner_shift 3 \
  --refiner_t_thresh 0.85 \
  --refiner_sigma_tail_steps 2 \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32 \
  --refiner_vae_dtype fp32 \
  --reuse_condition_features
```

The refiner is only supported for video modes. It reuses base condition
features when `--reuse_condition_features` is enabled.

## Multi-GPU FSDP Inference

Use `--enable_fsdp_inference` when the base DiT and refiner DiT need to stay
loaded at the same time but a replicated DiT does not fit comfortably on each
GPU. The flag shards every loaded DiT transformer with PyTorch composable FSDP:

- base-only inference shards the base `transformer/` DiT.
- base plus refiner inference shards both the base `transformer/` DiT and the
  `refiner/` DiT before base generation starts.
- VLM/text encoder, VAE, scheduler, and prompt features are not sharded by this
  flag.

FSDP inference can be combined with context parallel and CFG parallel. For
context parallel on all GPUs, use `--context_parallel_degree` equal to the GPU
count; add `--batch_cfg` when you want batched CFG. For FSDP-only memory
sharding, launch with `torchrun` and keep
`--cfg_parallel_degree 1 --context_parallel_degree 1`.

FSDP inference reduces GPU memory after the DiT is wrapped. During
initialization, each rank still constructs the transformer on host memory before
sharding, so large MoE checkpoints need enough system RAM for the launched
process count.

Example: multi-GPU base plus refiner with both DiTs sharded:

```bash
torchrun --standalone --nproc_per_node 8 scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --run_refiner \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --refiner_output "outputs/t2v_refined.mp4" \
  --height 480 \
  --width 832 \
  --refiner_height 1088 \
  --refiner_width 1920 \
  --fps 24 \
  --steps 40 \
  --refiner_steps 8 \
  --guidance_scale 3 \
  --refiner_guidance_scale 3 \
  --shift 3 \
  --refiner_shift 3 \
  --cfg_parallel_degree 1 \
  --context_parallel_degree 8 \
  --batch_cfg \
  --refiner_batch_cfg \
  --enable_fsdp_inference \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32 \
  --refiner_vae_dtype fp32 \
  --reuse_condition_features
```

The runtime log prints one `fsdp_inference=...` field for the base stage and
one for the refiner stage. When both are enabled successfully, both fields show
`FSDPInferenceInfo(enabled=True, ...)`.

## TI2V

Use the same first frame in the rewriter and DiT inference:
`--first-frame "<first_frame.png>"` for the rewriter and
`--image "<first_frame.png>"` for DiT.

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --mode ti2v \
  --image "assets/cases/ti2v/example_4/first_frame.png" \
  --prompt_json "assets/cases/ti2v/example_4/prompt.json" \
  --output "outputs/ti2v.mp4" \
  --height 480 \
  --width 832 \
  --num_frames 121 \
  --fps 24 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3
```

CFG parallel is not enabled for TI2V in this runner.

> **Multi-GPU configuration**: FSDP memory sharding and context parallel work
> exactly as for T2V — follow the "Multi-GPU FSDP Inference" section above,
> switching `--mode` to `ti2v` and adding `--image` (CFG parallel excepted, since
> TI2V does not support it).

## T2I

T2I uses `num_frames=1` internally and writes an image output:

**⚠️ Note: the refiner is not trained on images, so T2I does not support the refiner (`--run_refiner` only applies to video modes).**

> **Multi-GPU configuration**: FSDP memory sharding and context parallel work
> exactly as for T2V — follow the "Multi-GPU FSDP Inference" section above,
> switching `--mode` to `t2i`. Since T2I does not support the refiner, drop
> `--run_refiner` and every `refiner_*` flag.

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --mode t2i \
  --prompt_json "assets/cases/t2i/example_6/prompt.json" \
  --output "outputs/image.png" \
  --height 480 \
  --width 832 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3
```
