import gc
from functools import wraps

import fire
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, SanaVideoPipeline
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline, LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import export_to_video


def cli(func):
    wraps(func)

    def wrapper(*args, **kwargs):
        return fire.Fire(func)

    return wrapper


@cli
def sana_video_ltx2_refine(
    prompt: str = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.",
    negative_prompt: str = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience.",
    sana_model_id: str = "/home/junsongc/junsongc/code/diffusion/Sana/output/Sana_1600M_720px_ltx2_t2v_ltx_resample/checkpoints/sana_ltxvae",
    ltx2_model_id: str = "Lightricks/LTX-2",
    sana_height: int = 704,
    sana_width: int = 1280,
    sana_frames: int = 81,
    motion_score: int = 30,
    sana_guidance_scale: float = 6.0,
    sana_num_steps: int = 50,
    frame_rate: float = 16.0,
    seed: int = 42,
    output_path: str = "sana_ltx2_refined.mp4",
    save_sana_output: str = "sana_original.mp4",
    skip_audio: bool = True,
    skip_upsampler: bool = True,
):
    """
    Use SanaVideoPipeline to generate video latent, then use LTX2 Pipeline for Stage-2 refinement.

    Key technical points:
    1. Manually pack video latent to skip diffusers' normalize step (consistent with original LTX-2 code)
    2. Create audio latent such that it becomes zero after normalize (consistent with zero audio latent in original code)
    """

    device = "cuda"
    dtype = torch.bfloat16

    # Step 1: Generate latent using Sana Video Pipeline
    sana_pipe = SanaVideoPipeline.from_pretrained(sana_model_id, torch_dtype=dtype)
    sana_pipe.text_encoder.to(dtype)
    sana_pipe.enable_model_cpu_offload(gpu_id=0)

    full_prompt = prompt + f" motion score: {motion_score}."
    generator = torch.Generator(device=device).manual_seed(seed)

    sana_output = sana_pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        height=sana_height,
        width=sana_width,
        frames=sana_frames,
        guidance_scale=sana_guidance_scale,
        num_inference_steps=sana_num_steps,
        generator=generator,
        output_type="latent",
        return_dict=True,
    )

    # Extract video latent
    video_latent = None
    for key in ("latents", "video_latents", "latent", "dit_latents", "dit_latent", "frames", "videos"):
        if hasattr(sana_output, key):
            video_latent = getattr(sana_output, key)
            if video_latent is not None:
                break

    if video_latent is None:
        raise ValueError("Failed to extract latents from Sana output.")
    if isinstance(video_latent, (list, tuple)):
        video_latent = video_latent[0]
    if video_latent.dim() == 4:
        video_latent = video_latent.unsqueeze(0)

    del sana_pipe, sana_output
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Load LTX2 Pipeline
    ltx2_pipe = LTX2Pipeline.from_pretrained(ltx2_model_id, torch_dtype=dtype)

    # Save Sana original output
    if save_sana_output:
        ltx2_pipe.vae.to(device)
        ltx2_pipe.vae.enable_tiling(
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=448,
            tile_sample_stride_width=448,
        )

        sana_latent_for_decode = video_latent.to(device=device, dtype=ltx2_pipe.vae.dtype)
        latents_mean = ltx2_pipe.vae.latents_mean.view(1, -1, 1, 1, 1).to(
            sana_latent_for_decode.device, sana_latent_for_decode.dtype
        )
        latents_std = ltx2_pipe.vae.latents_std.view(1, -1, 1, 1, 1).to(
            sana_latent_for_decode.device, sana_latent_for_decode.dtype
        )
        sana_latent_denorm = sana_latent_for_decode * latents_std / ltx2_pipe.vae.config.scaling_factor + latents_mean

        timestep = (
            None
            if not ltx2_pipe.vae.config.timestep_conditioning
            else torch.tensor([0.0], device=device, dtype=sana_latent_denorm.dtype)
        )
        with torch.no_grad():
            sana_decoded = ltx2_pipe.vae.decode(sana_latent_denorm, timestep, return_dict=False)[0]

        del sana_latent_for_decode, sana_latent_denorm
        sana_video = ltx2_pipe.video_processor.postprocess_video(sana_decoded, output_type="pil")
        export_to_video(sana_video[0], save_sana_output, fps=int(frame_rate))

        del sana_decoded, sana_video
        ltx2_pipe.vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    ltx2_pipe.enable_model_cpu_offload(gpu_id=0)

    # Latent Upsampler (optional)
    if skip_upsampler:
        print("Skipping latent upsampler")
        upscaled_video_latent = video_latent.to(device=device, dtype=dtype)
    else:
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            ltx2_model_id, subfolder="latent_upsampler", torch_dtype=dtype
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=ltx2_pipe.vae, latent_upsampler=latent_upsampler)
        upsample_pipe.enable_model_cpu_offload(device=device)

        upscaled_video_latent = upsample_pipe(
            latents=video_latent.to(device=device, dtype=dtype),
            latents_normalized=True,
            height=sana_height,
            width=sana_width,
            num_frames=sana_frames,
            output_type="latent",
            return_dict=False,
        )[0]

        latents_mean = ltx2_pipe.vae.latents_mean.view(1, -1, 1, 1, 1).to(
            upscaled_video_latent.device, upscaled_video_latent.dtype
        )
        latents_std = ltx2_pipe.vae.latents_std.view(1, -1, 1, 1, 1).to(
            upscaled_video_latent.device, upscaled_video_latent.dtype
        )
        scaling_factor = ltx2_pipe.vae.config.scaling_factor
        upscaled_video_latent = (upscaled_video_latent - latents_mean) * scaling_factor / latents_std

        del latent_upsampler, upsample_pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Step 3: Manually pack video latent (skip diffusers' normalize, consistent with original code)
    packed_video_latent = LTX2Pipeline._pack_latents(
        upscaled_video_latent,
        patch_size=ltx2_pipe.transformer_spatial_patch_size,
        patch_size_t=ltx2_pipe.transformer_temporal_patch_size,
    )

    _, _, latent_num_frames, latent_height, latent_width = upscaled_video_latent.shape
    pixel_height = latent_height * ltx2_pipe.vae_spatial_compression_ratio
    pixel_width = latent_width * ltx2_pipe.vae_spatial_compression_ratio
    pixel_num_frames = (latent_num_frames - 1) * ltx2_pipe.vae_temporal_compression_ratio + 1

    # Step 4: Create audio latent (becomes zero after normalize, consistent with original code)
    num_frames_pixel = (latent_num_frames - 1) * 8 + 1
    duration_s = num_frames_pixel / frame_rate
    audio_num_frames = round(duration_s * 16000 / 160 / 4)

    num_channels_audio, latent_mel_bins = 8, 16
    audio_latents_mean = ltx2_pipe.audio_vae.latents_mean
    packed_audio_latent = (
        audio_latents_mean.unsqueeze(0)
        .unsqueeze(0)
        .expand(1, audio_num_frames, num_channels_audio * latent_mel_bins)
        .to(dtype=dtype, device=device)
        .contiguous()
    )
    audio_latent = (
        packed_audio_latent.unflatten(2, (num_channels_audio, latent_mel_bins)).permute(0, 2, 1, 3).contiguous()
    )

    del video_latent, upscaled_video_latent
    gc.collect()
    torch.cuda.empty_cache()

    # Step 5: LTX2 Stage-2 Refinement
    ltx2_pipe.load_lora_weights(
        ltx2_model_id, adapter_name="stage_2_distilled", weight_name="ltx-2-19b-distilled-lora-384.safetensors"
    )
    ltx2_pipe.set_adapters("stage_2_distilled", 1.0)
    ltx2_pipe.vae.enable_tiling()
    ltx2_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        ltx2_pipe.scheduler.config,
        use_dynamic_shifting=False,
        shift_terminal=None,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    video, audio = ltx2_pipe(
        latents=packed_video_latent,
        audio_latents=audio_latent,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=pixel_height,
        width=pixel_width,
        num_frames=pixel_num_frames,
        num_inference_steps=3,
        noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        frame_rate=frame_rate,
        generator=generator,
        output_type="np",
        return_dict=False,
    )

    # Step 6: Save output
    video = (video * 255).round().astype("uint8")
    video = torch.from_numpy(video)
    encode_video(
        video[0],
        fps=frame_rate,
        audio=None if skip_audio else audio[0].float().cpu(),
        audio_sample_rate=None if skip_audio else ltx2_pipe.vocoder.config.output_sampling_rate,
        output_path=output_path,
    )
    print(f"Done! Output saved to: {output_path}")
