"""Single-GPU Open-H3-Regenerate-2K pipeline.

    low-resolution H3 latent -> author latent upscaler to the target grid (2560x1440 by default; 768P, 1080P or 4K)
    -> 3-step (or 2-step) refine (merged LoRA) -> VAE decode (released H3 VAE, or the fast H3 VAE decoder)

The low-resolution latent is given (a normalized native H3 latent), encoded from a draft video (`encode_video`), or
generated at 768P from the prompt with the optional HyperFlow 8-step adapter. One resident transformer serves both stages; StageMerger swaps the LoRA set per stage.
Text conditioning is computed up front (`encode_prompt`) so the 32B text encoder is never resident with the DiT.
"""
from pathlib import Path
import subprocess
import time

import torch

from . import adapters, standard_fwd, sol_engine_fwd, upscaler, vae_fast
from .sigmas import SCHEDULES, shifted_grid

NATIVE_SHAPE = (24, 37, 48, 84)        # 1344x768, 121 frames
# draft sizes a video is encoded at (W, H) and their latents; 576p is how drafts from other video models were prepared
INPUT_SIZES = {"768p": (1344, 768), "576p": (1024, 576)}
NATIVE_SHAPES = {name: (24, 37, h // 16, w // 16) for name, (w, h) in INPUT_SIZES.items()}
UPSCALED_SHAPE = (24, 37, 90, 160)     # 2560x1440
# output resolution -> (computed height, width, kept rows): 1080P and 4K are computed at 1088 / 2176 rows (whole latent
# patches) and the centre 1080 / 2160 rows are kept
RESOLUTIONS = {"768p": (768, 1344, 768), "1080p": (1088, 1920, 1080), "2k": (1440, 2560, 1440), "4k": (2176, 3840, 2160)}
UPSCALER_FILE = "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
# --fwd -> installer: sol_engine (default), sol_engine_fast (fastest), standard (unaccelerated reference)
FORWARDS = {
    "sol_engine": lambda model: sol_engine_fwd.install(model, "sol_engine"),
    "sol_engine_fast": lambda model: sol_engine_fwd.install(model, "sol_engine_fast"),
    "standard": standard_fwd.install,
}


def encode_prompt(base_model, prompt):
    """Prompt embeddings + token tags with the official MiniMax-H3 text encoder step (loaded, used, freed)."""
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
    encoder = MiniMaxH3TextEncoderStep().init_pipeline(str(base_model))
    encoder.load_components(dtype=torch.bfloat16)
    encoder.to("cuda")
    with torch.inference_mode():
        encoded = encoder(prompt=prompt, output=["prompt_embeds", "text_token_tags"])
    text = {"prompt_embeds": encoded["prompt_embeds"][0].detach().to(torch.bfloat16).cpu(),
            "text_token_tags": encoded["text_token_tags"].flatten().long().cpu()}
    del encoder, encoded
    torch.cuda.empty_cache()
    return text


def encode_video(base_model, path, size="768p", seed=42):
    """Draft MP4 -> normalized native latent (24, 37, H/16, W/16), the VAE loaded, used and freed.

    The draft is resampled to 24 fps, cover-scaled (bilinear) and center-cropped to INPUT_SIZES[size], cut or padded
    (last frame held) to 121 frames, then 3 more copies of the last frame give the 124 frames the H3 VAE turns into
    37 latent frames. Pixels are ImageNet-normalized and the FP32 VAE posterior is sampled under `seed`.
    """
    import imageio_ffmpeg
    import numpy as np
    from diffusers import AutoencoderKLMiniMaxH3
    width, height = INPUT_SIZES[size]
    raw = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vf",
        f"fps=24,scale={width}:{height}:force_original_aspect_ratio=increase:flags=bilinear,crop={width}:{height},"
        "tpad=stop_mode=clone:stop_duration=6", "-frames:v", "121", "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], capture_output=True, check=True).stdout
    if len(raw) != 121 * height * width * 3:
        raise ValueError(f"could not decode {path} into 121 frames at {width}x{height}")
    pixels = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(121, height, width, 3).copy())
    pixels = pixels.permute(3, 0, 1, 2)[None].cuda().float() / 255.0
    pixels = torch.cat([pixels, pixels[:, :, -1:].expand(-1, -1, 3, -1, -1)], dim=2)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        base_model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).cuda().eval()
    vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                      tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
    with torch.inference_mode():
        pixel_mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1, 1)
        posterior = vae.encode((pixels - pixel_mean) / pixel_std, return_dict=False)[0]
        latents = posterior.sample(generator=torch.Generator().manual_seed(seed))
        mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, 24, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device="cuda").view(1, 24, 1, 1, 1)
        latents = ((latents - mean) / std)[0].float().cpu()
    del vae, posterior, pixels
    torch.cuda.empty_cache()
    if tuple(latents.shape) != NATIVE_SHAPES[size]:
        raise ValueError(f"expected {NATIVE_SHAPES[size]}, got {tuple(latents.shape)}")
    return latents


def encode_silence(base_model):
    from diffusers import AutoencoderKLMiniMaxH3Audio
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        base_model, subfolder="audio_vae", torch_dtype=torch.float32, local_files_only=True).cuda().eval()
    with torch.inference_mode():
        silence = torch.zeros(2, 1, round(124 / 24 * 32000), device="cuda")
        audio = audio_vae.encode(silence, return_dict=False)[0].mode().float().cpu()
        mean = torch.tensor(audio_vae.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(audio_vae.config.latents_std).view(1, -1, 1)
        audio = ((audio - mean) / std).to(torch.bfloat16)
    del audio_vae
    torch.cuda.empty_cache()
    return audio


def load_video_vae(base_model, batch):
    from diffusers import AutoencoderKLMiniMaxH3
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        base_model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).cuda().eval()
    vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                      tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
    return vae_fast.install(vae.to(torch.bfloat16), batch=batch)


def _detach_peft(model):
    """Turn the PEFT layers HyperFlow injected back into plain linears; return their (A, B * scaling) factors."""
    pairs = {}
    for name, module in list(model.named_modules()):
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "get_base_layer"):
            key = next(iter(module.lora_A))
            pairs[name] = (module.lora_A[key].weight.detach().float(),
                           module.lora_B[key].weight.detach().float() * module.scaling[key])
            parent, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attribute, module.get_base_layer())
    return pairs


class Pipeline:
    def __init__(self, weights_dir, base_model, fwd="sol_engine", hyperflow=None, vae_batch=26, steps=3, start_sigma=None,
                 resolution="2k", vae="standard"):
        from diffusers import MiniMaxH3Transformer3DModel
        # fixed per pipeline: the sol_engine CUDA graphs are keyed by (stage, step), not by sigma or shape
        self.sigmas = SCHEDULES[steps] if start_sigma is None else shifted_grid(start_sigma, steps)
        height, width, self.keep_rows = RESOLUTIONS[resolution]
        self.target_shape = (24, 37, height // 16, width // 16)
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.set_float32_matmul_precision("highest")
        weights_dir, base_model = Path(weights_dir), Path(base_model)
        self.audio = encode_silence(base_model).to(self.device)
        model = MiniMaxH3Transformer3DModel.from_pretrained(
            base_model, subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            local_files_only=True)
        stages = {}
        self.metadata = None
        if hyperflow is not None:
            from .hyperflow import load_hyperflow_lora
            self.metadata = load_hyperflow_lora(model, hyperflow)
            stages["hyperflow"] = [(_detach_peft(model), 1.0)]
        lora, self.curve = adapters.attach(model, weights_dir)
        stages["refine"] = [(lora, 1.0)]
        model.requires_grad_(False)
        self.model = model.to(self.device).eval()
        self.fwd = FORWARDS[fwd](self.model)
        self.merger = adapters.StageMerger(self.model, stages, self.device, resident=self.fwd.resident)
        if self.fwd.mx_layers:
            self.merger.attach_quantized(self.fwd.mx_layers)
        del stages, lora
        self.node = upscaler.load(weights_dir / UPSCALER_FILE)
        self.fast_vae = self.vae = None
        if vae == "fast":
            from .fast_vae import WEIGHTS_FILE, FastVAE
            self.fast_vae = FastVAE(weights_dir / WEIGHTS_FILE, base_model)
        else:
            self.vae = load_video_vae(base_model, vae_batch)
        torch.cuda.synchronize()

    # ------------------------------------------------------------ stage switching
    def set_stage(self, stage):
        hyperflow = stage == "hyperflow"
        embedder = self.model.time_embedder
        if hasattr(embedder, "passthrough"):
            embedder.passthrough = not hyperflow
            if not hyperflow:
                embedder.set_endpoint(None)
        self.curve.strength = 0.0 if hyperflow else 1.0
        self.merger.merge(stage)

    def on_step(self, stage, index):
        if self.fwd.plan is not None:
            self.fwd.plan.stage, self.fwd.plan.step = stage, index
        if self.fwd.graphs is not None:
            self.fwd.graphs.key = (stage, index)

    # ------------------------------------------------------------ stages
    def generate_768p(self, text, seed):
        from .hyperflow.schedule import shift_sigmas, build_row_time_pairs, DEFAULT_SIGMAS_8STEP
        from .geometry import MiniMaxH3Geometry, build_t2va_layout, unpatchify_video_rows
        if self.metadata is None:
            raise ValueError("Generating the 768P latent needs the HyperFlow adapter (--hyperflow)")
        raw = self.metadata.sigmas or DEFAULT_SIGMAS_8STEP
        video_grid, audio_grid = shift_sigmas(raw, 12.), shift_sigmas(raw, 3.)
        geometry = MiniMaxH3Geometry(num_latent_frames=NATIVE_SHAPE[1], latent_height=NATIVE_SHAPE[2],
                                     latent_width=NATIVE_SHAPE[3], num_audio_latents=207)
        layout = build_t2va_layout(text["text_token_tags"], geometry)
        device = self.device
        for block in self.model.transformer_blocks:
            block.attn.processor.valid_rows = layout.token_tags.numel()
        static = dict(
            encoder_hidden_states=text["prompt_embeds"][None].to(device), token_tags=layout.token_tags.to(device),
            position_ids=layout.position_ids.to(device), video_indices=layout.video_indices.to(device),
            audio_indices=layout.audio_indices.to(device), text_indices=layout.text_indices.to(device),
            attention_kwargs={}, return_dict=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        video = torch.randn((1, geometry.num_video_rows, geometry.video_row_dim), generator=generator,
                            dtype=torch.float32).to(device)
        audio = torch.randn((1, layout.audio_indices.numel(), 32), generator=generator, dtype=torch.float32).to(device)
        with torch.inference_mode():
            for index in range(len(video_grid) - 1):
                current, endpoint, inverse = build_row_time_pairs(
                    video_indices=layout.video_indices, audio_indices=layout.audio_indices,
                    num_condition_video_rows=0, num_condition_audio_rows=0,
                    num_text_tokens=layout.text_indices.numel(),
                    video_timestep=float(1 - video_grid[index]), video_endpoint=float(1 - video_grid[index + 1]),
                    audio_timestep=float(1 - audio_grid[index]), audio_endpoint=float(1 - audio_grid[index + 1]),
                    condition_video_timestep=1.)
                self.model.time_embedder.set_endpoint(endpoint.to(device))
                self.on_step("hyperflow", index)
                prediction, audio_prediction = self.model(
                    hidden_states=video.bfloat16(), audio_hidden_states=audio.bfloat16(),
                    timestep=current.to(device), timestep_indices=inverse.to(device), **static)
                video = video + float(video_grid[index] - video_grid[index + 1]) * prediction.float()
                audio = audio + float(audio_grid[index] - audio_grid[index + 1]) * audio_prediction.float()
        return unpatchify_video_rows(video, geometry)[0]

    def upscale(self, native):
        """Native latent -> target grid. Above 2K the upscaler runs twice (to 2K, then to the target) so each pass stays
        within its trained range; a latent already at the target size is refined as is."""
        target = self.target_shape
        if tuple(native.shape) == target:
            return native.float().to(self.device)
        if target[2] > UPSCALED_SHAPE[2]:
            native = upscaler.upscale(self.node, native, UPSCALED_SHAPE)
        return upscaler.upscale(self.node, native, target).to(self.device)

    def refine(self, low, text, seed):
        from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
        from .geometry import MiniMaxH3Geometry, build_t2va_layout, pack_audio_latents, unpatchify_video_rows
        from .parallel import forward_padded
        device = self.device
        geometry = MiniMaxH3Geometry(num_latent_frames=low.shape[1], latent_height=low.shape[2],
                                     latent_width=low.shape[3], num_audio_latents=self.audio.shape[-1])
        audio = pack_audio_latents(self.audio[None]).to(device)
        layout = build_t2va_layout(text["text_token_tags"], geometry).to(device)
        prompt = text["prompt_embeds"][None].to(device)
        low = patchify_video_latents(low[None], geometry.patch_size).reshape(
            1, geometry.num_video_rows, geometry.video_row_dim).to(device, torch.float32)

        class Handle:                      # forward_padded expects a wrapper exposing .module
            def __init__(self, module):
                self.module = module

            def __call__(self, **kwargs):
                return self.module(**kwargs)

        with torch.inference_mode():
            noise = torch.randn(low.shape, device=device, dtype=torch.float32,
                                generator=torch.Generator(device=device).manual_seed(seed))
            sigmas = self.sigmas
            state = (1 - sigmas[0]) * low + sigmas[0] * noise
            for index, (sigma, following) in enumerate(zip(sigmas[:-1], sigmas[1:])):
                self.on_step("refine", index)
                velocity, _ = forward_padded(Handle(self.model), state.to(torch.bfloat16), audio, prompt, layout,
                                             sigma, 1)
                state = state + (sigma - following) * velocity.float()
        return unpatchify_video_rows(state, geometry)[0]

    def decode(self, latents):
        """Normalized (24, T, H, W) latent -> (121, rows, 16W, 3) uint8 frames on the GPU."""
        frames = self.fast_vae.decode_frames(latents)[:121] if self.fast_vae is not None else self._decode_released(latents)
        top = (frames.shape[1] - self.keep_rows) // 2
        return frames[:, top:top + self.keep_rows]

    def _decode_released(self, latents):
        from diffusers.models.attention_dispatch import attention_backend
        vae = self.vae
        dtype = next(vae.parameters()).dtype
        latents = latents[None].cuda().float()
        mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, 24, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device="cuda").view(1, 24, 1, 1, 1)
        pixel_mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1, 1)
        with torch.inference_mode(), attention_backend("native"):
            pixels = vae.decode((latents * std + mean).to(dtype), return_dict=False)[0].float()
            return ((pixels * pixel_std + pixel_mean).clamp(0, 1)[0, :, :121].permute(1, 2, 3, 0) * 255).round().byte()

    # ------------------------------------------------------------ one request
    def __call__(self, text, native=None, hf_seed=42, refine_seed=1766580932):
        """Returns (frames uint8 [121, rows, width, 3], latents dict, per-stage seconds)."""
        times = {}
        if self.fwd.plan is not None:
            sol_engine_fwd.set_prompt_length(self.fwd, text["text_token_tags"].numel())

        def stage(name, fn):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            times[name] = time.perf_counter() - start
            return out

        if native is None:
            stage("merge_hyperflow", lambda: self.set_stage("hyperflow"))
            native = stage("generate_768p", lambda: self.generate_768p(text, hf_seed))
        low = stage("upscale", lambda: self.upscale(native.to(self.device)))
        stage("merge_refine", lambda: self.set_stage("refine"))
        name = f"refine{len(self.sigmas) - 1}"
        refined = stage(name, lambda: self.refine(low, text, refine_seed))
        if self.fwd.graphs is not None:
            self.fwd.graphs.key = None
            torch.cuda.empty_cache()   # graph pools sit on top of the DiT peak; release the cache before the VAE
        if self.fwd.plan is not None:
            self.fwd.plan.stage = self.fwd.plan.step = None
        frames = stage("vae_decode", lambda: self.decode(refined))
        times[f"upscale+{name}"] = times["upscale"] + times[name]
        return frames, {"native": native.cpu(), "upscaled": low.cpu(), "refined": refined.cpu()}, times


def encode_mp4(frames, path, fps=24, crf=10):
    import imageio_ffmpeg
    frames = frames.cpu().numpy()
    height, width = frames.shape[1:3]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], input=frames.tobytes(), check=True)
