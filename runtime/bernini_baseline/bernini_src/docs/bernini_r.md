# Bernini-R — renderer only

[← Back to main README](../README.md)

Bernini-R is the renderer-only Bernini model, fine-tuned from the Wan diffusion
renderer. It skips the semantic-planning stage of the
[full Bernini pipeline](bernini.md), which makes it the recommended lightweight
choice for simpler inference, renderer benchmarking, and scenarios where output
consistency matters more than complex instruction following.

| Checkpoint | Base | Notes |
|------------|------|-------|
| [`ByteDance/Bernini-R-Diffusers`](https://huggingface.co/ByteDance/Bernini-R-Diffusers) | Wan2.2-T2V-A14B | **Recommended.** Self-contained diffusers-format directory. |
| [`ByteDance/Bernini-R`](https://huggingface.co/ByteDance/Bernini-R) | Wan2.2-T2V-A14B | Separate high-/low-noise checkpoints; needs the Wan2.2 base download. |
| [`ByteDance/Bernini-R-1.3B-Diffusers`](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers) | Wan2.1-1.3B | Lightweight variant; close to 14B on simple tasks (style transfer, subtitle/watermark removal, local editing), weaker on complex tasks such as human generation. |

Benchmarks for the 14B and 1.3B variants are in the
[main README](../README.md#-highlights).

## Download weights

### Diffusers format (recommended)

The diffusers-format directory is self-contained: it includes the Wan base
components plus the Bernini-R `transformer` / `transformer_2` weights. Pass the
directory directly as `--config` and do **not** pass `--high_noise_ckpt` /
`--low_noise_ckpt`.

```bash
pip install -U "huggingface_hub"
hf download ByteDance/Bernini-R-Diffusers \
    --local-dir pretrained_models/Bernini-R-Diffusers
```

```bash
python infer_single_gpu.py --config pretrained_models/Bernini-R-Diffusers \
    --case assets/testcases/t2i/t2i.json --num_frames 1 --guidance_mode t2v_apg
```

The 1.3B release works the same way with
`ByteDance/Bernini-R-1.3B-Diffusers`.

### Separate checkpoints

This layout keeps the Wan2.2 base and the trained Bernini-R renderer weights
separate. Use it only if you specifically need explicit high-noise / low-noise
checkpoint paths.

```bash
pip install -U "huggingface_hub"
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --local-dir pretrained_models/Wan2.2-T2V-A14B-Diffusers
hf download ByteDance/Bernini-R \
    --local-dir pretrained_models/Bernini-R
```

Load the renderer config from
[`configs/bernini_renderer_wan22`](../configs/bernini_renderer_wan22/config.json)
and pass both checkpoint paths (replace the file names with the actual
safetensors in your download):

```bash
torchrun --nproc-per-node 8 infer_multi_gpu.py \
    --config configs/bernini_renderer_wan22 \
    --high_noise_ckpt pretrained_models/Bernini-R/<high-noise>.safetensors \
    --low_noise_ckpt pretrained_models/Bernini-R/<low-noise>.safetensors \
    --case assets/testcases/t2v/t2v.json
```

A matching config for the Wan2.1-1.3B base is provided at
[`configs/bernini_renderer_wan21_1p3b`](../configs/bernini_renderer_wan21_1p3b/config.json).

## Run

> Make sure the environment is set up first — see
> [Installation](../README.md#-installation), which includes the required
> VeOmni install.

For single-GPU image tasks, use `infer_single_gpu.py`; for video tasks, use
`infer_multi_gpu.py` with `torchrun` and `--ulysses` sequence parallelism:

```bash
# Single-GPU text-to-image
python infer_single_gpu.py --config pretrained_models/Bernini-R-Diffusers \
    --case assets/testcases/t2i/t2i.json --num_frames 1 --guidance_mode t2v_apg

# Multi-GPU video editing
torchrun --nproc-per-node 8 infer_multi_gpu.py \
    --config pretrained_models/Bernini-R-Diffusers --ulysses 8 \
    --case assets/testcases/v2v/v2v_case1.json --guidance_mode v2v_apg
```

Inputs are described by case files under
[`assets/testcases/`](../assets/testcases/); see the
[case-file format](../assets/testcases/README.md).

### Run scripts

[`scripts/bernini_r/`](../scripts/bernini_r/) provides one script per task:

```bash
bash scripts/bernini_r/run_t2i.sh    # text-to-image
bash scripts/bernini_r/run_i2i.sh    # image editing
bash scripts/bernini_r/run_t2v.sh    # text-to-video
bash scripts/bernini_r/run_v2v.sh    # video editing
bash scripts/bernini_r/run_rv2v.sh   # reference + video editing
bash scripts/bernini_r/run_r2v.sh    # reference-to-video
```

The scripts use the diffusers layout and read these environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `BERNINI_R_CONFIG` | `./pretrained_models/Bernini-R-Diffusers` | model directory |
| `CASE_PATH` | a bundled example case | case JSON to run (single-case scripts) |
| `NPROC_PER_NODE` | 8 | number of processes (multi-GPU scripts) |
| `ULYSSES` | 8 | Ulysses sequence-parallel degree (multi-GPU scripts) |

For the separate-checkpoint layout, replace `--config` with
`--config configs/bernini_renderer_wan22 --high_noise_ckpt <hi> --low_noise_ckpt <lo>`.

See `python infer_single_gpu.py --help` for the full argument list.

## Gradio demo

```bash
# Single GPU
python gradio_demo.py --config pretrained_models/Bernini-R-Diffusers --port 7860

# 8 GPUs, 8-way Ulysses sequence parallel
torchrun --nproc-per-node 8 gradio_demo.py --ulysses 8 \
    --config pretrained_models/Bernini-R-Diffusers --port 7860 --share

# Or the script launcher (diffusers layout, honors BERNINI_R_CONFIG)
bash scripts/bernini_r/run_gradio.sh
```

See the [Gradio demo notes](../README.md#gradio-demo) in the main README for
the UI behavior and prompt-enhancer setup.
