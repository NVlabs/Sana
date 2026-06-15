<p align="center" style="border-radius: 10px">
  <img src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/asset/sana-wm-logo.png" width="70%" alt="SANA-WM Logo"/>
</p>

# 🌍 SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer

<div align="center">
  <a href="https://nvlabs.github.io/Sana/WM"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2605.15178"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana-WM&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional"><img src="https://img.shields.io/static/v1?label=HF%20Weights&message=SANA-WM&color=yellow&logo=huggingface"></a> &ensp;
</div>

<div align="center">
  <video src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/WM/media/videos/hero_reel_v9.mp4#t=1" autoplay playsinline controls muted loop width="90%" onloadedmetadata="this.currentTime=1;this.playbackRate=2"></video>
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
python inference_video_scripts/wm/inference_sana_wm.py \
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
python inference_video_scripts/wm/inference_sana_wm.py \
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
| `demo_1` | bioluminescent cave | `w-35,aw-60,dw-100,aw-55,w-25,none-50` |
| `demo_2` | mushroom forest / robot | `w-25,aw-60,dw-100,aw-55,none-85`  (+ `--translation_speed 0.015`) |
| `demo_3` | salt flat / supercar | `w-70,none-40,dw-35,w-70,aw-35,none-72` |
| `demo_4` | ice plain / portal | `w-95,aw-35,w-70,dw-35,none-87` |

The `_pose.npy` files already bake in these actions (and `demo_2`'s slower
speed), so `--camera asset/sana_wm/demo_N_pose.npy` reproduces the same motion
as the matching `--action` string.

### 80-scene benchmark

For the fixed 80-scene, 60s SANA-WM benchmark release and reproducible
bidirectional inference/evaluation workflow, see
[SANA-WM benchmark guide](sana-wm-bench.md).

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
python inference_video_scripts/wm/inference_sana_wm_streaming.py \
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
- `--config / --model_path / --causal_vae_path / --refiner_root / --refiner_gemma_root` — point at non-default weight paths (local path or
  `hf://` URI). `--config` defaults to the in-repo
  `configs/sana_wm/sana_wm_streaming_1600m_720p.yaml`.
- `--num_frame_per_block` (default 3, must match the checkpoint's
  `chunk_size`), `--denoising_step_list` (default
  `"1000,960,889,727,0"`), `--refiner_block_size` (3), `--refiner_kv_max_frames`
  (11) — change the canonical recipe at your own quality risk.

### ⚡ Quantized inference (fp8 / fp4)

Streaming supports **per-component** low-precision compute to cut peak VRAM, set
independently for the stage-1 DiT and the LTX-2 refiner:

```bash
python inference_video_scripts/wm/inference_sana_wm_streaming.py \
  --image       asset/sana_wm/demo_0.png \
  --prompt      asset/sana_wm/demo_0.txt \
  --action      "w-80,dw-40,w-80,aw-40" \
  --num_frames  241 \
  --stage1_precision  fp4 \
  --refiner_precision fp4 \
  --output_dir  results/sana_wm_streaming
```

`--stage1_precision` / `--refiner_precision` each take `bf16` (default, any GPU),
`fp8` (FP8 W8A8, Hopper **and** Blackwell), or `fp4` (NVFP4 W4A4, **Blackwell
only** — sm_100/sm_120). Quantization touches only the linear GEMMs (self-attn,
cross-attn, FFN), scoped **per transformer block** so the camera/action
conditioning math stays in native precision and action-following is preserved.

**Requirements.** fp8/fp4 need [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
≥ 2.0, which `environment_setup.sh` installs by default. If you skipped it
(`SANA_SKIP_TE=1`) the script exits with an install hint; bf16 needs nothing
extra. fp4 additionally requires a Blackwell GPU (GB200, B200, RTX 50-series).

Quantization is primarily a **memory** optimization — the GEMMs are
latency-bound at these chunk sizes, so bf16 is already near the compute roofline
and lower precision mainly buys VRAM headroom (and a modest GB200 speedup).

Steady-state throughput and peak VRAM, **SANA-WM stages only** (excludes the
one-time Pi3X intrinsics estimate and first-chunk `torch.compile` warmup),
default `--refiner_kv_max_frames 11`:

| stage-1 / refiner | peak VRAM | H100 (×realtime) | GB200 (×realtime) |
|---|---|---|---|
| bf16 / bf16 | 47.3 GB | 1.09× | 1.27× |
| fp8 / fp8 | 35.4 GB | ~1.00× *(est.)* | 1.16× |
| fp4 / fp4 | 29.4 GB | — (Blackwell only) | 1.16× |

> 🎯 **Runs on a 32 GB RTX 5090:** `--stage1_precision fp4 --refiner_precision fp4` fits in **29.4 GB** (the bf16 default needs 47 GB). fp4 requires Blackwell,
> which the 5090 is; the ×realtime figures above are measured on GB200/B200.

A tighter KV window (`--refiner_kv_max_frames 2`) drops VRAM further and is
faster, at a **quality cost** (more temporal flicker / drift — not recommended
for final renders):

| stage-1 / refiner | peak VRAM | H100 (×realtime) | GB200 (×realtime) |
|---|---|---|---|
| bf16 / bf16 | 37.4 GB | 1.26× | 1.57× |
| fp8 / fp8 | 25.4 GB | — | 1.32× |
| fp4 / fp4 | 25.0 GB | — | 1.25× |

**Picking a precision:** bf16 for best quality on any GPU; **fp8** for Hopper
(H100) users who want lower VRAM with no Blackwell requirement; **fp4** for
Blackwell, the only setting that fits the 47 GB bf16 model onto a 32 GB card. You
can mix (e.g. `--stage1_precision bf16 --refiner_precision fp8`).

> **Note:** quantization does not change the intrinsic long-rollout drift of the
> AR stage-1 backbone — very long clips slowly lose scene consistency regardless
> of precision. This is a property of the autoregressive teacher, not the quant.

## Chunk-Causal Stage-1 Teacher

The chunk-causal Stage-1 teacher is an intermediate research checkpoint: only
the Stage-1 Sana DiT is chunk-causal, while inference keeps the bidirectional
LTX-2 VAE and refiner path enabled by default. It is released mainly so users
can reproduce the CP/FSDP2 Stage-1 training path and experiment with future
chunk-causal models. It may show severe artifacts, temporal flicker, or weaker
camera adherence than the full bidirectional / streaming releases; keep the
default refiner on for qualitative samples and use `--no_refiner` only for fast
Stage-1 debugging.

The public config sets `scheduler.vis_sampler: chunk_flow_euler`, so the CLI's
default `--sampling_algo auto` selects the correct sampler:

```bash
python inference_video_scripts/wm/inference_sana_wm.py \
  --config     hf://Efficient-Large-Model/SANA-WM_chunk_causal/config.yaml \
  --model_path hf://Efficient-Large-Model/SANA-WM_chunk_causal/dit/sana_wm_chunk_causal_1600m_720p.safetensors \
  --image      asset/sana_wm/demo_0.png \
  --prompt     asset/sana_wm/demo_0.txt \
  --action     "w-240,dw-120,w-120,aw-180,w-300" \
  --num_frames 961 \
  --offload_refiner \
  --output_dir results/sana_wm_chunk_causal
```

`--offload_refiner` only changes model residency; the bidirectional refiner still
runs unless `--no_refiner` is passed.

`chunk_flow_euler` uses `interval_k = 1 / num_chunks` by default. Override it
with `--chunk_interval_k` only for ablations.

## Chunk-Causal Stage-1 Training on Sekai-Game

This repo includes the minimal chunk-causal Stage-1 training path:

- training script: `train_video_scripts/train_sana_wm_stage1.py`
- training config: `configs/sana_wm/stage1/sana_wm_stage1_sekai_chunk_causal_cp2_fsdp2.yaml`
- latent dataset loader: `diffusion/data/datasets/video/sana_wm_zip_latent_data.py`
- CP/FSDP2 tests: `tests/test_context_parallel*.py`

The example can run directly from the public HF dataset
[`Efficient-Large-Model/SANA-WM-example-training-dataset`](https://huggingface.co/datasets/Efficient-Large-Model/SANA-WM-example-training-dataset).
The config downloads the dataset on the main rank, waits for all ranks, and
trains from the chunk-causal Stage-1 teacher checkpoint with FSDP2 and CP2:

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  train_video_scripts/train_sana_wm_stage1.py \
  --config_path configs/sana_wm/stage1/sana_wm_stage1_sekai_chunk_causal_cp2_fsdp2.yaml
```

The dataset repo is about 235 GB and is laid out so the config paths resolve to:

```yaml
data:
  hf_dataset_repo: Efficient-Large-Model/SANA-WM-example-training-dataset
  hf_dataset_revision: 4d965e94b9ea11b9c5ba085251ffa7a0345e006f
  hf_dataset_local_dir: .
  data_dir:
    sekai_game: data/sekai_game_train_961frames_16fps_ovl640
  vae_cache_dir: data/vae_cache/LTX2VAE_diffusers_704x1280/sekai_game_train_961frames_16fps_ovl640
model:
  load_from: hf://Efficient-Large-Model/SANA-WM_chunk_causal/dit/sana_wm_chunk_causal_1600m_720p.safetensors
  attn_type: ChunkCausalGDNTriton
  camctrl_type: ChunkCausalGDNUCPESinglePathLiteLABothTriton
train:
  use_fsdp: true
  fsdp_version: 2
  cp_size: 2
```

Set `data.hf_dataset_local_dir` to a shared filesystem path if you do not want
the dataset under the repo checkout. Relative `data_dir` and `vae_cache_dir`
entries are resolved under that directory after download.

The Sekai-derived dataset is redistributed for non-commercial research use
only. See the dataset card, `LICENSE`, and `NOTICE.md` in the HF dataset repo
before training or redistributing derivatives.

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
| `--num_frames` | Total frames to generate (default `161`; the streaming demo uses `241`; the chunk-causal Stage-1 teacher example uses `961`). |
| `--fps` | Output mp4 frame rate (default `16`). |
| `--step` | Stage-1 DiT sampling steps (default `60`). |
| `--cfg_scale` | Classifier-free-guidance scale (default `5.0`). |
| `--flow_shift` | Override the scheduler's `inference_flow_shift`. |
| `--no_refiner` | The LTX-2 refiner is enabled by default. This flag skips it and decodes Stage-1 latents with the Sana VAE for fast, lower-quality debugging. |
| `--refiner_root` | LTX-2 refiner root containing `transformer/` and `connectors/`. |
| `--no_action_overlay` | Skip the WASD + joystick overlay on the output video. |
| `--offload_vae` | Move the VAE to CPU between encode / decode steps. |
| `--offload_refiner` | Lazy-load the LTX-2 refiner only when needed; release afterwards. |
| `--stage1_precision` | Streaming only. Stage-1 DiT compute precision: `bf16` (default), `fp8` (Hopper+/Blackwell), `fp4` (Blackwell only). fp8/fp4 need Transformer Engine. |
| `--refiner_precision` | Streaming only. LTX-2 refiner compute precision: `bf16` (default), `fp8`, `fp4`. See [Quantized inference](#-quantized-inference-fp8--fp4). |
| `--sampling_algo` | `auto` (default). Uses `chunk_flow_euler` for chunk-causal teacher configs and `flow_euler_ltx` otherwise. For streaming use the dedicated `wm/inference_sana_wm_streaming.py`. |
| `--chunk_interval_k` | Optional `chunk_flow_euler` interval override. Defaults to `1 / num_chunks`. |

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

`Efficient-Large-Model/SANA-WM_chunk_causal` (Stage-1 teacher):

| Component | Path |
|------------------------------------|------------------------------------------------------------|
| Chunk-causal Sana DiT (Stage 1) | `dit/sana_wm_chunk_causal_1600m_720p.safetensors` |
| Inference config | `config.yaml` |

The chunk-causal teacher repo intentionally contains only the Stage-1 config and
DiT weights. The CLI resolves the bidirectional VAE, LTX-2 refiner, and refiner
text encoder from `Efficient-Large-Model/SANA-WM_bidirectional` by default to
avoid duplicating large immutable artifacts.

The Sana text encoder (`gemma-2-2b-it`) is fetched separately from
`Efficient-Large-Model/gemma-2-2b-it`.

## 📝 BibTeX

```bibtex
@article{zhu2026sana,
  title={Sana-wm: Efficient minute-scale world modeling with hybrid linear diffusion transformer},
  author={Zhu, Haoyi and Liu, Haozhe and Zhao, Yuyang and Ye, Tian and Chen, Junsong and Yu, Jincheng and He, Tong and Han, Song and Xie, Enze},
  journal={arXiv preprint arXiv:2605.15178},
  year={2026}
}
```
