<p align="center" style="border-radius: 10px">
  <video src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Streaming/assets/teaser/sana-streaming-teaser.mp4?v=20260525ao" poster="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Streaming/assets/logos/teaser-sana-streaming-five-institutions.png" autoplay playsinline controls muted loop width="90%"></video>
</p>

# SANA-Streaming: Real-time Streaming Video Editing with Hybrid Diffusion Transformer

<div align="center">
  <a href="https://nvlabs.github.io/Sana/Streaming/"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2605.30409"><img src="https://img.shields.io/static/v1?label=Arxiv&message=SANA-Streaming&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/Efficient-Large-Model/SANA-Streaming"><img src="https://img.shields.io/static/v1?label=HF%20Weights&message=Streaming&color=yellow&logo=huggingface"></a> &ensp;
  <a href="https://huggingface.co/Efficient-Large-Model/SANA-Streaming_bidirectional"><img src="https://img.shields.io/static/v1?label=HF%20Weights&message=Bidirectional&color=yellow&logo=huggingface"></a>
</div>

## 📽️ About SANA-Streaming

**SANA-Streaming** is a real-time video-to-video editing system for minute-level,
high-resolution editing. Given a source video and a text instruction, it edits
the requested content while preserving source motion and non-edited regions.

Core contributions:

- **Hybrid Diffusion Transformer** — interleaves Gated DeltaNet (GDN) blocks with
  softmax-attention blocks, combining compact long-range memory with local
  source alignment.
- **Streaming Video Editing** — processes long videos with state caching and
  chunk-wise generation instead of full-sequence attention.
- **Cycle-Reverse Regularization** — improves temporal consistency by training
  the model to reconstruct source frames from edited content through flow
  matching.
- **Efficient System Co-design** — the paper reports fused GDN kernels and
  Mixed-Precision Quantization (MPQ) for RTX 5090 deployment, reaching
  1280×704 real-time editing at 24 end-to-end FPS and 58 DiT FPS.

This repository release exposes two practical inference paths:

- `bidirectional_short`: 5-second short-video editing with the bidirectional
  2B SANA-Streaming DiT.
- `long_streaming`: 1-min long-video editing with the streaming 2B
  SANA-Streaming DiT.

The current public script runs the released BF16 checkpoints. The MPQ deployment
recipe described in the paper is not required for the commands below.

## ⚙️ Environment Setup

```bash
bash ./environment_setup.sh sana
conda activate sana
```

The released V2V checkpoints were validated with `torch==2.10.0`,
`torchvision==0.25.0`, `triton==3.6.0`, `transformers==4.57.1`,
`accelerate==1.0.1`, and Hugging Face `diffusers` commit
`fbe8a75ad59fe5c0eec7f3691d2eb0ed890a0c90`. The fused GDN kernels and the
LTX-2 VAE path are sensitive to runtime package versions; use the pinned
package versions in `pyproject.toml` for reproducible bidirectional inference.

## 🏃 Inference

All DiT checkpoints and demo source videos are fetched on first use from the
Hugging Face repos below. Local paths and `hf://` URIs are both supported.

### Streaming long-video editing

The streaming model edits 969 frames by default with 4 denoising steps,
`cfg_scale=1.0`, `num_cached_blocks=2`, and sink-token caching enabled.

```bash
python inference_video_scripts/v2v/inference_sana_streaming.py \
  --mode long_streaming \
  --config configs/sana_streaming/sana_streaming_2b_720p.yaml \
  --model_path hf://Efficient-Large-Model/SANA-Streaming/dit/sana_streaming_ar.pth \
  --prompt "Transform the entire scene into a breathtaking Sci-Fi Art digital painting." \
  --video_path hf://Efficient-Large-Model/SANA-Streaming/source/09_style_transfer_source.mp4 \
  --num_frames 969 \
  --step 4 \
  --cfg_scale 1.0 \
  --num_cached_blocks 2 \
  --sink_token true \
  --output_dir results/sana_streaming_long \
  --output_name output.mp4
```

### Bidirectional short-video editing

The bidirectional model edits 81 frames by default with flow-DPM solver sampling,
50 denoising steps, and `cfg_scale=6.0`. A default negative prompt is applied
unless `--negative_prompt` is provided.

```bash
python inference_video_scripts/v2v/inference_sana_streaming.py \
  --mode bidirectional_short \
  --config configs/sana_streaming/sana_streaming_bidirectional_2b_720p.yaml \
  --model_path hf://Efficient-Large-Model/SANA-Streaming_bidirectional/dit/sana_bidirectional_short.pth \
  --prompt "Remove the thick, textured gold hoop earrings from the woman's ears. Carefully reconstruct the exposed earlobes to match her natural skin tone and texture. Ensure the lighting and soft shadows on the newly bare ears blend seamlessly with the rest of her face, leaving no trace or reflection of the metallic jewelry behind." \
  --video_path hf://Efficient-Large-Model/SANA-Streaming/source/00_local_editing_source.mp4 \
  --num_frames 81 \
  --step 50 \
  --cfg_scale 6.0 \
  --output_dir results/sana_streaming_bidirectional \
  --output_name output.mp4
```

### Example gallery

The release includes three source videos under
[`Efficient-Large-Model/SANA-Streaming`](https://huggingface.co/Efficient-Large-Model/SANA-Streaming/tree/main/source).
The same examples can be run with both `long_streaming` and
`bidirectional_short` by changing `--mode`.

| Example | Source video | Prompt |
|---------|--------------|--------|
| Local editing | `source/00_local_editing_source.mp4` | Remove the thick, textured gold hoop earrings from the woman's ears. Carefully reconstruct the exposed earlobes to match her natural skin tone and texture. Ensure the lighting and soft shadows on the newly bare ears blend seamlessly with the rest of her face, leaving no trace or reflection of the metallic jewelry behind. |
| Background editing | `source/05_background_editing_source.mp4` | Replace the background with a cinematic, rain-streaked windowpane at dusk. Feature softly out-of-focus city lights in moody cool teal and muted amber glowing through the wet glass. Add delicate condensation and trickling raindrops to the window surface, maintaining a shallow depth of field to enhance the deeply emotional, melancholic atmosphere without altering the subject's lighting or appearance. |
| Style transfer | `source/09_style_transfer_source.mp4` | Transform the entire scene into a breathtaking Sci-Fi Art digital painting. Re-render the background as an out-of-focus futuristic cityscape with glowing holographic bokeh and sleek technological structures. Re-imagine the subject in a highly detailed, futuristic illustration style, giving her skin a flawless, subtly luminescent quality. Keep her exact features, pose, and emotional expression intact, while rendering her hair, clothing, and phone with advanced, sleek synthetic textures. Bathe the composition in atmospheric neon blues, cool cyans, and deep purples to reflect a highly advanced civilization. |

## 🎛️ Argument Reference

| Argument | Format / Default |
|----------|------------------|
| `--mode` | `long_streaming` or `bidirectional_short` (default `long_streaming`). |
| `--prompt` | Text editing instruction. |
| `--video_path` | Source MP4 path. Supports local files and `hf://<repo>/<path>` URIs. |
| `--output_dir` | Output directory. |
| `--output_name` | Output MP4 filename (default `output.mp4`). |
| `--config` | YAML config path. Defaults are mode-specific under `configs/sana_streaming/`. |
| `--model_path` | DiT checkpoint path. Defaults to the released Hugging Face checkpoints. |
| `--num_frames` | Frames decoded from the source video (`969` for streaming, `81` for bidirectional). |
| `--height / --width` | Center-cropped output resolution (`704 × 1280`). |
| `--fps` | Output MP4 frame rate (`16`). |
| `--step` | Denoising steps (`4` for streaming, `50` for bidirectional). |
| `--cfg_scale` | CFG scale (`1.0` for streaming, `6.0` for bidirectional). |
| `--flow_shift` | Optional scheduler flow-shift override. |
| `--seed` | Random seed (`0`). |
| `--negative_prompt` | Optional negative prompt. Bidirectional mode uses a built-in default if omitted. |
| `--num_cached_blocks` | Streaming cache window size (`2`). |
| `--sink_token` | Keep the first chunk in the streaming cache window (`true`). |

## 📁 HF Repository Layout

`Efficient-Large-Model/SANA-Streaming_bidirectional`:

| Component | Path |
|-----------|------|
| Bidirectional SANA-Streaming DiT | `dit/sana_bidirectional_short.pth` |

`Efficient-Large-Model/SANA-Streaming`:

| Component | Path |
|-----------|------|
| Streaming SANA-Streaming DiT | `dit/sana_streaming_ar.pth` |
| Causal LTX-2 VAE release artifact | `ltx2_causal_vae_0516/` |
| Demo source videos | `source/{00_local_editing_source.mp4,05_background_editing_source.mp4,09_style_transfer_source.mp4}` |

The inference configs ship in-repo:

| Mode | Config |
|------|--------|
| `bidirectional_short` | `configs/sana_streaming/sana_streaming_bidirectional_2b_720p.yaml` |
| `long_streaming` | `configs/sana_streaming/sana_streaming_2b_720p.yaml` |

The text encoder is fetched separately from
`Efficient-Large-Model/gemma-2-2b-it`. The default VAE path in both configs is
`Lightricks/LTX-2`; `long_streaming` loads it through the local causal/chunk-tile
wrapper for streaming encode/decode.

## 📝 BibTeX

```bibtex
@article{zhao2026sana,
  title={SANA-Streaming: Real-time Streaming Video Editing with Hybrid Diffusion Transformer},
  author={Zhao, Yuyang and Pan, Yicheng and He, Qiyuan and Yu, Jincheng and Chen, Junsong and Ye, Tian and Liu, Haozhe and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2605.30409},
  year={2026}
}
```
