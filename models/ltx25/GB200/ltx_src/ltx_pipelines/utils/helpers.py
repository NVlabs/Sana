import logging
from collections.abc import Sequence

import torch

from ltx_core.components.noisers import Noiser
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    VideoGeneratedKeyframeSlots,
)
from ltx_core.devices import cleanup_accelerator_memory, cuda_activation_budget_bytes, get_preferred_device
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.audio_vae import encode_audio
from ltx_core.model.transformer import Modality
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TileSizeConfig, TilingConfig, VideoEncoder
from ltx_core.model.video_vae.diffusion_tiling import recommended_decode_tiling_config
from ltx_core.model.video_vae.model_configurator import (
    diffvae_tiling_geometry_from_vae_config,
    estimate_diffusion_decoder_weight_bytes,
    is_diffusion_video_vae,
)
from ltx_core.model.video_vae.transformer.config import DiffVAEMode
from ltx_core.text_encoders.gemma import LTXGemmaTextEncoder
from ltx_core.tiling import DimensionSizeConfig
from ltx_core.tools import LatentTools
from ltx_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import (
    ResizeMode,
    decode_audio_from_file,
    decode_image,
    decode_video_from_file,
    get_videostream_fps,
    is_exr_dir,
    load_exr_folder_conditioning_hdr,
    load_image_and_preprocess,
    resize_aspect_ratio_preserving,
    video_preprocess,
)
from ltx_pipelines.utils.media_io.color_config import HDRColorSpace


def get_device() -> torch.device:
    return get_preferred_device()


def cleanup_memory() -> None:
    cleanup_accelerator_memory()


# Historical Conv VAE ``TilingConfig.default()`` spatial long-side + temporal chunk
# (main ``latent_tile_splitters`` aspect-coupled a single 768px tile to H/W).
_CONV_AUTO_LONG_SIDE = DimensionSizeConfig(tile_size=768, overlap=64)
_CONV_AUTO_FRAMES = DimensionSizeConfig(tile_size=80, overlap=24)


def tiling_scale_factors_for_vae(vae_checkpoint_path: str) -> SpatioTemporalScaleFactors:
    """Scale factors decode will pass to ``to_splitters`` for this checkpoint."""
    if not is_diffusion_video_vae(vae_checkpoint_path):
        return VIDEO_SCALE_FACTORS
    metadata = SafetensorsModelStateDictLoader().metadata(vae_checkpoint_path)
    vae_config = metadata.get("config", {}).get("vae", {})
    return diffvae_tiling_geometry_from_vae_config(vae_config)["pixel_scale"]


def tiling_config_for_vae(
    vae_checkpoint_path: str,
    *,
    height: int,
    width: int,
    num_frames: int,
    diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    device: torch.device | None = None,
    free_bytes: int | None = None,
) -> TilingConfig:
    """Decode tiling for pipelines.
    Conv VAE: aspect-coupled long-side 768/64 + temporal 80/24.
    DiffVAE: halo/memory-aware :func:`recommended_decode_tiling_config`.
    """
    if not is_diffusion_video_vae(vae_checkpoint_path):
        _ = num_frames  # API parity with DiffVAE; Conv auto layout is aspect-only.
        return TileSizeConfig.from_long_side(
            long_side=_CONV_AUTO_LONG_SIDE,
            height=height,
            width=width,
            scale_factors=VIDEO_SCALE_FACTORS,
            frames=_CONV_AUTO_FRAMES,
        )

    metadata = SafetensorsModelStateDictLoader().metadata(vae_checkpoint_path)
    vae_config = metadata.get("config", {}).get("vae", {})
    geometry = diffvae_tiling_geometry_from_vae_config(vae_config)
    model_bytes = estimate_diffusion_decoder_weight_bytes(vae_checkpoint_path)

    if free_bytes is None:
        dev = device if device is not None else get_device()
        free_bytes = cuda_activation_budget_bytes(dev) if dev.type == "cuda" and torch.cuda.is_available() else 0

    return recommended_decode_tiling_config(
        **geometry,
        height=height,
        width=width,
        num_frames=num_frames,
        mode=diffvae_optimization,
        free_bytes=free_bytes,
        model_bytes=model_bytes,
    )


def ensure_tiling_config(
    tiling_config: TilingConfig | AutoTiling | None,
    *,
    scale_factors: SpatioTemporalScaleFactors,
    video_shape: VideoPixelShape,
    vae_checkpoint_path: str,
    diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    device: torch.device | None = None,
    free_bytes: int | None = None,
) -> TilingConfig | None:
    """Resolve pipeline tiling: ``AUTO_TILING`` → recommend; ``None`` → untiled; else validate.
    ``scale_factors`` must be the same factors decode will pass to ``to_splitters``
    (from :func:`tiling_scale_factors_for_vae`). Call only after ``video_shape.frames`` is known.
    """
    if tiling_config is None:
        return None
    if tiling_config is AUTO_TILING:
        return tiling_config_for_vae(
            vae_checkpoint_path,
            height=video_shape.height,
            width=video_shape.width,
            num_frames=video_shape.frames,
            diffvae_optimization=diffvae_optimization,
            device=device,
            free_bytes=free_bytes,
        )
    tiling_config.validate(scale_factors, video_shape)
    return tiling_config


def _conform_latent_length(latent: torch.Tensor, expected_frames_count: int) -> torch.Tensor:
    actual_frames = latent.shape[2]
    if actual_frames > expected_frames_count:
        latent = latent[:, :, :expected_frames_count]
    elif actual_frames < expected_frames_count:
        shape_as_list = list(latent.shape)
        shape_as_list[2] = expected_frames_count - actual_frames
        pad = torch.zeros(
            shape_as_list,
            device=latent.device,
            dtype=latent.dtype,
        )
        latent = torch.cat([latent, pad], dim=2)
    return latent


def video_latent_from_file(
    video_encoder: VideoEncoder,
    file_path: str,
    output_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    start_time: float = 0.0,
    max_duration: float | None = None,
    tiling_config: TilingConfig | None = None,
    color_space: HDRColorSpace | None = None,
) -> torch.Tensor | None:
    """Load video from a file or EXR-frame folder, and encode to latents.
    Args:
        video_encoder: Model used to encode pixel frames to latent space.
        file_path: Path to a video file, or a directory of ``*.exr`` frames
            (HDR; requires ``color_space`` from ``--hdr``).
        output_shape: Target pixel shape (height, width, frames, fps) for the conditioning.
        device: Device to run the encoder and hold tensors on.
        dtype: Dtype for the output latents.
        start_time: Start time in seconds to begin reading the video (default 0.0).
        max_duration: Maximum duration in seconds. If None, uses output_shape.frames at
            output_shape.fps (default None).
        tiling_config: Tiling configuration for the encoder. Defaults to
            ``TileSizeConfig.default()`` (independent H=W=768).
        color_space: HDR colour space when *file_path* is an EXR folder.
            Ignored for regular video files.
    Returns:
        Encoded video latents of shape (1, C, T, H, W) with T = required_latent_frames, or
        None (currently this function always returns a tensor).
    """
    fps = output_shape.fps
    max_duration = max_duration or output_shape.frames / fps
    if is_exr_dir(file_path):
        if color_space is None:
            raise ValueError(
                "EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT} to declare the source colour space."
            )
        # Center-crop matches the SDR video_preprocess path used for mp4 inputs.
        frame_start = round(start_time * fps)
        frame_cap = max(1, round(max_duration * fps))
        frames = torch.cat(
            list(
                load_exr_folder_conditioning_hdr(
                    exr_dir=file_path,
                    height=output_shape.height,
                    width=output_shape.width,
                    frame_cap=frame_cap,
                    dtype=dtype,
                    device=device,
                    color_space=color_space,
                    resize_mode=ResizeMode.CENTER_CROP,
                    frame_start=frame_start,
                )
            ),
            dim=2,
        )
    else:
        src_fps = get_videostream_fps(file_path)
        if src_fps != output_shape.fps:
            raise ValueError(f"Input video FPS {src_fps} does not match output FPS {output_shape.fps}, not supported")
        frame_gen = decode_video_from_file(
            path=file_path, device=device, start_time=start_time, max_duration=max_duration
        )
        frames = video_preprocess(frame_gen, output_shape.height, output_shape.width, dtype, device)
    latents = video_encoder.tiled_encode(frames, tiling_config or TileSizeConfig.default())
    required_latent_frames = VideoLatentShape.from_pixel_shape(
        output_shape, scale_factors=video_encoder.video_scale_factors
    ).frames
    return _conform_latent_length(latents, required_latent_frames)


def audio_latent_from_file(
    audio_encoder: torch.nn.Module,
    file_path: str,
    output_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> torch.Tensor | None:
    """Load audio from a file, and construct the audio latent conforming to video output shape.
    Args:
        audio_encoder: Model used to encode audio to latent space.
        file_path: Path to the audio or video file containing an audio stream.
            EXR-frame folders have no audio and return ``None``.
        output_shape: Target video pixel shape; used to derive required latent frames
            and, when max_duration is None, the audio duration (output_shape.frames / fps).
        device: Device to run the encoder and hold tensors on.
        dtype: Dtype for the output latents.
        start_time: Start time in seconds to begin reading the audio (default 0.0).
        max_duration: Maximum duration in seconds. If None, uses the full span implied
            by output_shape (default None).
    Returns:
        Encoded audio latents of shape (1, C, T, ...) with T = required_latent_frames, or
        None if the file has no audio stream.
    """
    if is_exr_dir(file_path):
        return None
    max_duration = max_duration or output_shape.frames / output_shape.fps
    audio_in = decode_audio_from_file(file_path, device, start_time, max_duration)
    if audio_in is None:
        return None
    latents = encode_audio(audio_in, audio_encoder, None).to(device, dtype)
    required_latent_frames = AudioLatentShape.from_video_pixel_shape(output_shape).frames
    return _conform_latent_length(latents, required_latent_frames)


def combined_image_conditionings(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    color_space: HDRColorSpace | None = None,
) -> list[ConditioningItem]:
    """Create a list of conditionings by replacing the latent at the first frame with the encoded image if present
    and using other encoded images as the keyframe conditionings."""
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
            color_space=color_space,
        )
        encoded_image = video_encoder(image)
        if img.frame_idx == 0:
            conditioning = VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=0,
            )
        else:
            conditioning = VideoConditionByKeyframeIndex(
                keyframes=encoded_image,
                strength=img.strength,
                frame_idx=img.frame_idx,
            )
        conditionings.append(conditioning)
    return conditionings


def image_conditionings_by_replacing_latent(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    color_space: HDRColorSpace | None = None,
) -> list[ConditioningItem]:
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
            color_space=color_space,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=img.frame_idx,
            )
        )

    return conditionings


def image_conditionings_by_adding_guiding_latent(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    color_space: HDRColorSpace | None = None,
) -> list[ConditioningItem]:
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
            color_space=color_space,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByKeyframeIndex(keyframes=encoded_image, frame_idx=img.frame_idx, strength=img.strength)
        )
    return conditionings


def evenly_spaced_keyframe_positions(num_keyframes: int, num_frames: int) -> list[int]:
    """Interior pixel-frame positions for *num_keyframes* generated keyframes. Endpoints excluded."""
    if num_keyframes < 0:
        raise ValueError(f"num_keyframes must be non-negative, got {num_keyframes}")
    if num_keyframes == 0:
        return []
    if num_frames < num_keyframes + 2:
        raise ValueError(
            f"Generated keyframes need at least num_keyframes + 2 target frames, got "
            f"num_keyframes={num_keyframes}, num_frames={num_frames}"
        )
    return torch.linspace(0, num_frames - 1, num_keyframes + 2).round().to(torch.int64).tolist()[1:-1]


def has_generated_keyframes(generated_keyframes: int | Sequence[int]) -> bool:
    """Whether a ``generated_keyframes`` request asks for any slots.
    Callers must not test the argument's truthiness directly: a ``Sequence`` may be a tensor or an
    array, whose truth value is ambiguous and raises.
    """
    if isinstance(generated_keyframes, int):
        return generated_keyframes > 0
    return len(generated_keyframes) > 0


def resolve_generated_keyframes(
    generated_keyframes: int | Sequence[int],
    num_frames: int,
) -> list[int]:
    """Normalize the pipeline-level ``generated_keyframes`` argument to pixel-frame indices.
    An ``int`` requests that many evenly spaced interior keyframes; a sequence gives the target
    pixel-frame indices explicitly. ``0`` / empty means the feature is off.
    """
    if isinstance(generated_keyframes, int):
        return evenly_spaced_keyframe_positions(generated_keyframes, num_frames)

    positions = sorted({int(position) for position in generated_keyframes})
    if positions and (positions[0] < 0 or positions[-1] >= num_frames):
        raise ValueError(
            f"Generated keyframe positions must lie in [0, {num_frames}), got "
            f"{sorted(int(p) for p in generated_keyframes)}"
        )
    return positions


def generated_keyframe_conditionings(
    generated_keyframes: int | Sequence[int],
    num_frames: int,
) -> list[ConditioningItem]:
    """Build the generated-keyframe conditioning, or an empty list when the feature is off.
    All slots go into one :class:`VideoGeneratedKeyframeSlots` so their tokens form a single
    contiguous, exactly-locatable range.
    """
    positions = resolve_generated_keyframes(generated_keyframes, num_frames)
    if not positions:
        return []
    return [VideoGeneratedKeyframeSlots(pixel_frame_indices=positions)]


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    """Create a noised latent state from empty state, conditionings, and noiser.
    Creates an empty latent state, applies conditionings, and then adds noise
    using the provided noiser. Returns the final noised state ready for diffusion.
    """
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = noiser(state, noise_scale)

    return state


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    """Apply a list of conditionings to a latent state.
    Iterates through the conditioning items and applies each one to the latent
    state in sequence. Returns the modified state with all conditionings applied.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState,
    context: torch.Tensor,
    sigma: torch.Tensor,
    enabled: bool = True,
) -> Modality:
    """Create a Modality from a latent state.
    Constructs a Modality object with the latent state's data, timesteps derived
    from the denoise mask and sigma, positions, and the provided context.
    When ``state.frozen`` is True, ``Modality.sigma`` is forced to 0 so prompt AdaLN
    and cross-modality gates match frozen conditioning (per-token timesteps are
    already zeroed via ``denoise_mask``).
    """
    if state.frozen:
        sigma = torch.zeros_like(sigma)
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
        keyframes_mask=state.keyframes_mask,
    )


def timesteps_from_mask(denoise_mask: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Compute timesteps from a denoise mask and sigma value.
    Multiplies the denoise mask by sigma to produce timesteps for each position
    in the latent state. Areas where the mask is 0 will have zero timesteps.
    When sigma is ``(B,)`` it is reshaped to ``(B, 1, ...)`` so the batch
    dimension aligns correctly with ``denoise_mask``.
    """
    if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
        sigma = sigma.view(-1, *([1] * (denoise_mask.dim() - 1)))
    return denoise_mask * sigma


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    """Clean a response from curly quotes and leading non-letter characters which Gemma tends to insert."""
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: LTXGemmaTextEncoder,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
    static_cache: bool = False,
) -> str:
    """Generate an enhanced prompt from a text encoder and a prompt."""
    if image_path:
        image = decode_image(image_path=image_path)
        image = torch.tensor(image)
        image = resize_aspect_ratio_preserving(image, image_long_side).to(torch.uint8)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed, static_cache=static_cache)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed, static_cache=static_cache)
    logging.info("Enhanced prompt: %s", prompt)
    return clean_response(prompt)


def assert_resolution(height: int, width: int, is_two_stage: bool) -> None:
    """Assert that the resolution is divisible by the required divisor.
    For two-stage pipelines, the resolution must be divisible by 64.
    For one-stage pipelines, the resolution must be divisible by 32.
    """
    divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {'two-stage' if is_two_stage else 'one-stage'} pipelines, "
            f"height and width must be multiples of {divisor}."
        )


def snap_frames_to_grid(frames: int, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> int:
    """Round ``frames`` down to the nearest ``k * scale_factors.time + 1``.
    The model's frame count must satisfy ``(frames - 1) % scale_factors.time == 0``
    (causal VAE temporal grid).
    """
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    time_scale = scale_factors.time
    return ((frames - 1) // time_scale) * time_scale + 1


def seconds_to_clamped_num_frames(
    seconds: float,
    *,
    frame_rate: float,
    min_frames: int = 1,
    max_frames: int = 1024,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> int:
    """Convert a duration in seconds to a frame count snapped to the VAE's temporal grid.
    Outlier durations are clamped to ``[min_frames, max_frames]`` (before snapping) so a
    misbehaving prediction can't request an OOM-sized generation. Snapping floors to the
    grid, which can undershoot ``min_frames``; when that happens the result is snapped up
    to the next grid point instead, so the ``[min_frames, max_frames]`` contract always holds.
    """
    raw_frames = round(seconds * frame_rate)
    raw_frames = max(min_frames, min(raw_frames, max_frames))
    frames = snap_frames_to_grid(raw_frames, scale_factors)
    if frames < min_frames:
        time_scale = scale_factors.time
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames
