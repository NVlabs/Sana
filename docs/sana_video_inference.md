## Inference CLI

### Inference SANA-Video

```bash
python app/sana_video_pipeline.py \
        --config configs/sana_video_config/480ms/Sana_1600M_480px_adamW_fsdp.yaml \
        --model_path "hf://Efficient-Large-Model/SanaVideo_willquant/checkpoints/model.pth" \
        --save_path sana_video.mp4 \
        --prompt "In a whimsical forest setting, a small deer with antlers stands amidst oversized mushrooms and scattered carrots. The scene is vibrant with lush green moss and rocks, creating a magical atmosphere. The deer appears curious, moving slowly across the ground, surrounded by the towering fungi and colorful vegetables. The sky above is clear and bright, adding to the enchanting ambiance. A low-angle shot captures the deer's gentle exploration of this fantastical landscape."
```

### Inference SANA-Video Chunked Version

```bash
python app/sana_video_pipeline.py \
        --config configs/sana_video_config/480ms/Sana_1600M_480px_adamW_fsdp_chunk.yaml \
        --model_path "hf://Efficient-Large-Model/SanaVideo_chunk/checkpoints/model.pth" \
        --save_path sana_video_chunk_i2v.mp4 \
        --interval_k 0.2 \
        --image_path output/tmp_videos/wan_goodcase_i2v_eval/00000000_video_001.jpg \
        --prompt "In a whimsical forest setting, a small deer with antlers stands amidst oversized mushrooms and scattered carrots. The scene is vibrant with lush green moss and rocks, creating a magical atmosphere. The deer appears curious, moving slowly across the ground, surrounded by the towering fungi and colorful vegetables. The sky above is clear and bright, adding to the enchanting ambiance. A low-angle shot captures the deer's gentle exploration of this fantastical landscape."
```

## Sana Video + LTX2 Refiner Pipeline

Use [Sana-Video](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_720p_diffusers) to generate video latents, then refine with [LTX-2](https://huggingface.co/Lightricks/LTX-2) Stage-2 for enhanced quality. For more details, check our [Blog: Bet Small, Win Big](https://nvlabs.github.io/Sana/Video/bet-small-win-big/blog.html).

```python
"""Sana Video + LTX2 Refiner: Stage 1 generate latent → Stage 2 refine (3 steps)."""

import gc
import torch
from diffusers import SanaVideoPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.export_utils import encode_video

device = "cuda"
dtype = torch.bfloat16
prompt = "A cat walking on the grass, facing the camera."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_score = 30
height, width, frames, frame_rate = 704, 1280, 81, 16.0
seed = 42

# ── Load all models ──
sana_pipe = SanaVideoPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_720p_diffusers", torch_dtype=dtype,
)
sana_pipe.text_encoder.to(dtype)
sana_pipe.enable_model_cpu_offload()

ltx_pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=dtype)
ltx_pipe.load_lora_weights(
    "Lightricks/LTX-2", adapter_name="stage_2_distilled",
    weight_name="ltx-2-19b-distilled-lora-384.safetensors",
)
ltx_pipe.set_adapters("stage_2_distilled", 1.0)
ltx_pipe.vae.to(dtype)
ltx_pipe.enable_model_cpu_offload()

upsampler_model = LTX2LatentUpsamplerModel.from_pretrained(
    "Lightricks/ltx-2-patch-upsampler", torch_dtype=dtype,
)
upsampler = LTX2LatentUpsamplePipeline(upsampler_model).to(device)

# ── Stage 1: Sana-Video generates latent ──
motion_prompt = f" motion score: {motion_score}."
sana_out = sana_pipe(
    prompt=prompt + motion_prompt, negative_prompt=negative_prompt,
    height=height, width=width, frames=frames,
    guidance_scale=6, num_inference_steps=50,
    generator=torch.Generator(device=device).manual_seed(seed),
    output_type="latent",
)
sana_latent = sana_out.frames

del sana_pipe; gc.collect(); torch.cuda.empty_cache()

# ── Stage 1.5: Upsample latent ──
video_latent = sana_latent.squeeze(0).permute(1, 0, 2, 3)
packed = upsampler(video_latent, output_type="latent").frames
lF, _, lH, lW = packed.shape
pH = lH * ltx_pipe.vae_spatial_compression_ratio
pW = lW * ltx_pipe.vae_spatial_compression_ratio
pT = (lF - 1) * ltx_pipe.vae_temporal_compression_ratio + 1

dur = pT / frame_rate
audio_frames = round(dur * ltx_pipe.audio_sampling_rate / ltx_pipe.audio_hop_length / ltx_pipe.audio_vae_temporal_compression_ratio)
nch = ltx_pipe.audio_vae.config.latent_channels
mel = ltx_pipe.audio_vae.config.mel_bins // ltx_pipe.audio_vae_mel_compression_ratio
audio_latent = (
    ltx_pipe.audio_vae.latents_mean.unsqueeze(0).unsqueeze(0)
    .expand(1, audio_frames, nch * mel).to(dtype=dtype, device=device).contiguous()
    .unflatten(2, (nch, mel)).permute(0, 2, 1, 3).contiguous()
)

del video_latent; gc.collect(); torch.cuda.empty_cache()

# ── Stage 2: LTX2 refine ──
video, _ = ltx_pipe(
    latents=packed, audio_latents=audio_latent,
    prompt=prompt, negative_prompt=negative_prompt,
    height=pH, width=pW, num_frames=pT,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0, frame_rate=frame_rate,
    generator=torch.Generator(device=device).manual_seed(seed),
    output_type="np", return_dict=False,
)

video = torch.from_numpy((video * 255).round().astype("uint8"))
encode_video(video[0], fps=frame_rate, audio=None, audio_sample_rate=None, output_path="sana_ltx2_refined.mp4")
```
