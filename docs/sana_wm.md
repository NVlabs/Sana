<p align="center" style="border-radius: 10px">
  <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/sana-wm-logo.png" width="70%" alt="SANA-WM Logo"/>
</p>

# 🌍 SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer

<div align="center">
  <a href="https://nvlabs.github.io/Sana/WM"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2605.15178"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana-WM&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional"><img src="https://img.shields.io/static/v1?label=HF%20Weights&message=SANA-WM&color=yellow&logo=huggingface"></a> &ensp;
</div>

<div align="center">
  <video src="https://nvlabs.github.io/Sana/WM/media/videos/hero_reel_v9.mp4" controls muted loop width="90%"></video>
</div>

## 📽️ About SANA-WM

**SANA-WM** is an efficient 2.6 B-parameter open-source world model trained natively for one-minute video generation. It synthesises 720p, minute-scale videos with precise 6-DoF camera control, paired with an LTX-2 sink-bidirectional Euler refiner for high-fidelity decoding.

Core contributions:

- **Hybrid Linear Attention** — frame-wise Gated DeltaNet combined with softmax attention every $N$-th block for memory-efficient long-context modelling.
- **Dual-Branch Camera Control** — independent main and camera branches enable precise per-frame trajectory adherence (6 DoF).
- **Two-Stage Generation Pipeline** — a long-video refiner stitched on top of Stage-1 latents improves quality and temporal consistency.
- **Robust Annotation Pipeline** — metric-scale 6-DoF camera poses extracted from public corpora yield spatiotemporally consistent action supervision.

SANA-WM completes pre-training in 15 days on 64 H100s and generates a 60s 720p clip on a single GPU; the distilled variant runs on an RTX 5090 with NVFP4 quantisation.

> **Note** This is the initial release and currently ships **bidirectional inference only**. More variants are on the way — stay tuned.

## ⚙️ Environment Setup

```bash
bash ./environment_setup_sana_wm.sh sana-wm
conda activate sana-wm
```

## 🏃 Inference

All Stage-1 / Stage-2 weights, the VAE, and the LTX-2 Gemma text encoder are
fetched on first use from
[`Efficient-Large-Model/SANA-WM_bidirectional`](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional)
— no manual download required.

### Example 1 — image + prompt + action string

```bash
python inference_video_scripts/inference_sana_wm.py \
  --image      asset/sana_wm/demo_0.png \
  --prompt     asset/sana_wm/demo_0.txt \
  --action     "w-80,jw-40,w-40,lw-60,w-100" \
  --translation_speed 0.055 \
  --rotation_speed_deg 1.2 \
  --num_frames 321 \
  --output_dir results/sana_wm_demo
```

Action DSL: each segment is `<keys>-<frames>` joined by commas. Movement keys
`w` (forward), `a` (strafe left), `s` (back), `d` (strafe right) translate
on the world horizontal plane; rotation keys `i` (pitch up), `k` (pitch
down), `j` (yaw left), `l` (yaw right) act in the camera's local frame.
`none-N` holds the pose for `N` frames.

### Example 2 — image + prompt + camera trajectory (`.npy`)

```bash
python inference_video_scripts/inference_sana_wm.py \
  --image      asset/sana_wm/demo_0.png \
  --prompt     asset/sana_wm/demo_0.txt \
  --camera     asset/sana_wm/demo_0_pose.npy \
  --intrinsics asset/sana_wm/demo_0_intrinsics.npy \
  --num_frames 321 \
  --output_dir results/sana_wm_demo
```

`--camera` is a NumPy `.npy` of shape `(F, 4, 4)` (camera-to-world
matrices); `--intrinsics` is `.npy` of shape `(3, 3)`, `(F, 3, 3)`, or
`(4,) = (fx, fy, cx, cy)` in input-image pixels. If `--intrinsics` is
omitted we estimate it from `--image` with Pi3X and abort if the
resulting FOV is outside `[25°, 120°]`.

### Lower memory

For tight VRAM budgets, opt in to lazy-load + CPU offload:

```bash
... --offload_vae --offload_refiner
```

## 🎛️ Argument Reference

| Argument | Format / Default |
|------------------------|----------------------------------------------------------------------------------------|
| `--image` | First-frame RGB image. Aspect-preserving resized + center-cropped to 704×1280. |
| `--prompt` | UTF-8 text file with the conditioning prompt. |
| `--camera` | `(F, 4, 4)` `.npy` camera-to-world matrices. Mutually exclusive with `--action`. |
| `--action` | WASD/IJKL DSL. Rolled out via `action_string_to_c2w` to a `(F+1, 4, 4)` trajectory. |
| `--translation_speed` | Per-frame translation magnitude (default `0.05`). |
| `--rotation_speed_deg` | Per-frame rotation magnitude in degrees (default `1.2`). |
| `--intrinsics` | Optional `.npy` of shape `(3, 3)`, `(F, 3, 3)`, or `(4,)`. Pi3X-estimated if omitted. |
| `--num_frames` | Total frames to generate (default `161`; the demos above use `321`). |
| `--fps` | Output mp4 frame rate (default `16`). |
| `--step` | Stage-1 DiT sampling steps (default `60`). |
| `--cfg_scale` | Classifier-free-guidance scale (default `5.0`). |
| `--flow_shift` | Override the scheduler's `inference_flow_shift`. |
| `--no_refiner` | Skip the LTX-2 refiner and decode Stage-1 latents with the Sana VAE (faster, lower quality). |
| `--no_action_overlay` | Skip the WASD + joystick overlay on the output video. |
| `--offload_vae` | Move the VAE to CPU between encode / decode steps. |
| `--offload_refiner` | Lazy-load the LTX-2 refiner only when needed; release afterwards. |

## 📁 HF Repository Layout

`Efficient-Large-Model/SANA-WM_bidirectional`:

| Component | Path | Size |
|------------------------------------|--------------------------------------------|-------:|
| Sana DiT (Stage 1) | `dit/sana_wm_1600m_720p.safetensors` | 10 GB |
| LTX-2 VAE (diffusers) | `vae/` | 2 GB |
| LTX-2 refiner (Stage 2) | `refiner/refiner.safetensors` | 41 GB |
| Gemma text encoder for the refiner | `refiner/text_encoder/` | 46 GB |
| Inference config | `config.yaml` | — |

The Sana text encoder (`gemma-2-2b-it`) is fetched separately from
`Efficient-Large-Model/gemma-2-2b-it`.

## 📝 BibTeX

```bibtex
@misc{zhu2026sanawm,
      title={SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer},
      author={Haoyi Zhu and Haozhe Liu and Yuyang Zhao and Tian Ye and Junsong Chen and Jincheng Yu and Tong He and Song Han and Enze Xie},
      year={2026},
      eprint={2605.15178},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.15178},
}
```
