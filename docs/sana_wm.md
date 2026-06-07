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
  <video src="https://nvlabs.github.io/Sana/WM/media/videos/hero_reel_v9.mp4#t=1" autoplay playsinline controls muted loop width="90%" onloadedmetadata="this.currentTime=1;this.playbackRate=2"></video>
</div>

## 📽️ About SANA-WM

**SANA-WM** is an efficient 2.6 B-parameter open-source world model trained natively for one-minute video generation. It synthesises 720p, minute-scale videos with precise 6-DoF camera control, paired with an LTX-2 sink-bidirectional Euler refiner for high-fidelity decoding.

Core contributions:

- **Hybrid Linear Attention** — frame-wise Gated DeltaNet combined with softmax attention every $N$-th block for memory-efficient long-context modelling.
- **Dual-Branch Camera Control** — independent main and camera branches enable precise per-frame trajectory adherence (6 DoF).
- **Two-Stage Generation Pipeline** — a long-video refiner stitched on top of Stage-1 latents improves quality and temporal consistency.
- **Robust Annotation Pipeline** — metric-scale 6-DoF camera poses extracted from public corpora yield spatiotemporally consistent action supervision.

SANA-WM completes pre-training in 15 days on 64 H100s and generates a 60s 720p clip on a single GPU.

> **Note** Building on the original **bidirectional** pipeline (full-sequence
> Stage 1 + sink-bidirectional refiner), this release adds a new **streaming**
> pipeline: a chunk-causal distilled Stage 1 + chunk-causal refiner + causal-VAE
> decoder, overlapped on three CUDA streams and written progressively to MP4 so
> you can watch the clip as it generates. Streaming weights are released under
> [`SANA-WM_streaming`](https://huggingface.co/Efficient-Large-Model/SANA-WM_streaming).

## ⚙️ Environment Setup

```bash
bash ./environment_setup.sh sana
conda activate sana
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
  --action     "w-100,dw-60,w-100,aw-60" \
  --num_frames 321 \
  --output_dir results/sana_wm_demo
```

Action DSL: each segment is `<keys>-<frames>` joined by commas. The control
scheme is: `w` / `s` forward / back
(translation along the heading), `a` / `d` yaw left / right (turn),
`i` / `k` pitch up / down, `j` / `l` strafe left / right. `none-N` holds the
pose for `N` frames. Held keys ease in/out with light inertia (instant on a
fresh press, gentle coast on release); default speeds are gentle
(`--translation_speed 0.025`, `--rotation_speed_deg 0.6`).

> **⚠️ Mapping update (breaking change vs the first release).** The `--action`
> keys were remapped so the demo and CLI share one control scheme: **`a` / `d`
> now yaw** (previously strafe) and **`j` / `l` now strafe** (previously yaw);
> `w` / `s` (forward/back) and `i` / `k` (pitch) are unchanged, and the old
> implicit a/d→steer coupling is gone. Motion is also smoothed now and the
> default speeds are gentler. **If you have action strings from the earlier
> release, swap `a`/`d` ↔ `j`/`l`** to reproduce the same motion (the CLI also
> prints this notice once when `--action` is used).

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

### Example gallery

The release ships five first-frame + prompt + camera examples under
`asset/sana_wm/` — `demo_{0..4}.{png,txt}`, each with a rolled-out `_pose.npy`
trajectory and an `_intrinsics.npy`. Swap `demo_0` for any of them in the
commands above (works for both the bidirectional and streaming scripts). The
actions are gentle by design — slow forward drift with light left/right
look-around.

| Example | Scene | `--action` |
|---------|-------|------------|
| `demo_0` | salt-desert / black supercar | `w-100,dw-60,w-100,aw-60` |
| `demo_1` | bioluminescent cave         | `w-35,aw-60,dw-100,aw-55,w-25,none-50` |
| `demo_2` | mushroom forest / robot     | `w-25,aw-60,dw-100,aw-55,none-85` &nbsp;(+ `--translation_speed 0.015`) |
| `demo_3` | salt flat / supercar        | `w-70,none-40,dw-35,w-70,aw-35,none-72` |
| `demo_4` | ice plain / portal          | `w-95,aw-35,w-70,dw-35,none-87` |

The `_pose.npy` files already bake in these actions (and `demo_2`'s slower
speed), so `--camera asset/sana_wm/demo_N_pose.npy` reproduces the same motion
as the matching `--action` string.

### Lower memory

For tight VRAM budgets, opt in to lazy-load + CPU offload:

```bash
... --offload_vae --offload_refiner
```

### Streaming inference

The streaming pipeline replaces all three full-sequence stages with chunk-causal
variants and emits one decoded chunk per AR block straight into a progressive
MP4. Stage 1 runs the 4-step distilled student (CFG-baked-in, runs at
`cfg_scale=1`), the refiner runs chunk-causal AR with a sliding KV window, and
the causal LTX-2 VAE decodes chunk-by-chunk.

All streaming weights (DiT, causal VAE, refiner, and the Gemma text encoder)
are fetched on first use from
[`SANA-WM_streaming`](https://huggingface.co/Efficient-Large-Model/SANA-WM_streaming)
— no manual download required, exactly like the bidirectional path. The
inference YAML ships in-repo under `configs/sana_wm/`. Just run:

```bash
python inference_video_scripts/inference_sana_wm_streaming.py \
  --image       asset/sana_wm/demo_0.png \
  --prompt      asset/sana_wm/demo_0.txt \
  --action      "w-80,dw-40,w-80,aw-40" \
  --num_frames  241 \
  --output_dir  results/sana_wm_streaming
```

`--num_frames` defaults to **241 (~15s @ 16fps)**. It is snapped to
`8·refiner_block_size·k + 1` so the VAE and refiner chunking divide evenly
(241 = 24·10+1 needs no snap). Use a larger value (e.g. 961 for ~60s) for longer
clips.

Output lands at `results/sana_wm_streaming/<name>_streaming.mp4` and grows in
place — you can watch it while inference continues. Reaches **~0.93× realtime
on a single H100** after a one-time `torch.compile` warmup (~3 min cold, ~30 s
warm cache; the warmup amortises across runs that reuse the same shapes).

All speed-critical knobs are baked into the script as defaults — `torch.compile`
on the **refiner transformer** (`max-autotune-no-cudagraphs` mode), flash-only
SDPA, Inductor `coordinate_descent_tuning` + `epilogue_fusion`, cuDNN benchmark,
and the expandable CUDA allocator. The causal VAE decoder is intentionally **not**
compiled: `torch.compile` corrupts its cross-chunk causal cache (chunk 0 decodes
fine but later chunks come out blank/gray), so it runs eager. There is no
slow/fast toggle; the script is the fast config.

Overrides for advanced use:

- `--streaming_root <path>` — optional LOCAL bundle dir holding `sana_dit/`,
  `ltx2_causal_vae/`, `refiner_diffusers/`, `gemma3_12b/`. Unset by default, in
  which case each artefact is pulled from `hf://Efficient-Large-Model/SANA-WM_streaming`.
- `--config / --model_path / --causal_vae_path / --refiner_root /
  --refiner_gemma_root` — point at non-default weight paths (local path or
  `hf://` URI). `--config` defaults to the in-repo
  `configs/sana_wm/sana_wm_streaming_1600m_720p.yaml`.
- `--num_frame_per_block` (default 3, must match the checkpoint's
  `chunk_size`), `--denoising_step_list` (default
  `"1000,960,889,727,0"`), `--refiner_block_size` (3), `--refiner_kv_max_frames`
  (11) — change the canonical recipe at your own quality risk.

## 🎛️ Argument Reference

| Argument | Format / Default |
|------------------------|----------------------------------------------------------------------------------------|
| `--image` | First-frame RGB image. Aspect-preserving resized + center-cropped to 704×1280. |
| `--prompt` | UTF-8 text file with the conditioning prompt. |
| `--camera` | `(F, 4, 4)` `.npy` camera-to-world matrices. Mutually exclusive with `--action`. |
| `--action` | Control DSL (`w/s` move, `a/d` yaw, `i/k` pitch, `j/l` strafe). Rolled out via `action_string_to_c2w` (smoothed) to a `(F+1, 4, 4)` trajectory. |
| `--translation_speed` | Per-frame translation magnitude (default `0.025`). |
| `--rotation_speed_deg` | Per-frame rotation magnitude in degrees (default `0.6`). |
| `--intrinsics` | Optional `.npy` of shape `(3, 3)`, `(F, 3, 3)`, or `(4,)`. Pi3X-estimated if omitted. |
| `--num_frames` | Total frames to generate (default `161`; the demos above use `321`). |
| `--fps` | Output mp4 frame rate (default `16`). |
| `--step` | Stage-1 DiT sampling steps (default `60`). |
| `--cfg_scale` | Classifier-free-guidance scale (default `5.0`). |
| `--flow_shift` | Override the scheduler's `inference_flow_shift`. |
| `--no_refiner` | Skip the LTX-2 refiner and decode Stage-1 latents with the Sana VAE (faster, lower quality). |
| `--refiner_root` | LTX-2 refiner root containing `transformer/` and `connectors/`. |
| `--no_action_overlay` | Skip the WASD + joystick overlay on the output video. |
| `--offload_vae` | Move the VAE to CPU between encode / decode steps. |
| `--offload_refiner` | Lazy-load the LTX-2 refiner only when needed; release afterwards. |
| `--sampling_algo` | `flow_euler_ltx` (default, bidirectional). For streaming use the dedicated `inference_sana_wm_streaming.py`. |

## 📁 HF Repository Layout

`Efficient-Large-Model/SANA-WM_bidirectional`:

| Component | Path | Size |
|------------------------------------|---------------------------------------------|-------:|
| Sana DiT (Stage 1) | `dit/sana_wm_1600m_720p.safetensors` | 10 GB |
| LTX-2 VAE (diffusers) | `vae/` | 2 GB |
| LTX-2 refiner (Stage 2) | `refiner/{transformer,connectors}/` | 38 GB |
| Gemma text encoder for the refiner | `refiner/text_encoder/` | 46 GB |
| Inference config | `config.yaml` | — |

`Efficient-Large-Model/SANA-WM_streaming` (streaming variant):

| Component | Path |
|------------------------------------|----------------------------------------------|
| Chunk-causal Sana DiT (distilled) | `sana_dit/model.pt` |
| Causal LTX-2 VAE | `ltx2_causal_vae/` |
| Chunk-causal LTX-2 refiner | `refiner_diffusers/{transformer,connectors}/` |
| Gemma-3-12B text encoder (refiner) | `gemma3_12b/` |

The inference config ships in-repo at
`configs/sana_wm/sana_wm_streaming_1600m_720p.yaml` (not in the weights repo).

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
