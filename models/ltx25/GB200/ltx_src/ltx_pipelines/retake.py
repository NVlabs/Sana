from __future__ import annotations

import logging
from collections.abc import Iterator

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import (
    SpatioTemporalScaleFactors,
    VideoPixelShape,
)
from ltx_pipelines.utils.args import video_editing_arg_parser
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, detect_params
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    audio_latent_from_file,
    ensure_tiling_config,
    get_device,
    tiling_scale_factors_for_vae,
    video_latent_from_file,
)
from ltx_pipelines.utils.media_io import (
    HDRColorSpace,
    encode_video,
    get_videostream_metadata,
    is_exr_dir,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class RetakePipeline:
    """Regenerate a time region (retake) of an existing video.
    Given a source video file and a time window ``[start_time, end_time]``
    (in seconds), this pipeline keeps the video/audio outside that window
    unchanged and *regenerates* the content inside the window from a text
    prompt using the LTX-2 diffusion model.
    Parameters
    ----------
    checkpoint_path : str
        Path to the LTX-2 model checkpoint.
    gemma_root : str
        Root directory containing Gemma text-encoder weights.
    loras : list[LoraPathStrengthAndSDOps]
        Optional LoRA configs applied to the transformer.
    device : torch.device
        Target device (default: CUDA if available).
    quantization : QuantizationPolicy | None
        Optional quantization policy for the transformer.
    distilled : bool
        Set to ``True`` if using distilled model or passing distillation
        lora with full model. If set to ``True``, distilled sigma schedule
        (``DISTILLED_SIGMA_VALUES``) and a simple (non-guided) denoising
        function will be used during ``__call__``.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        distilled: bool = True,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.distilled = distilled
        if not distilled:
            self._scheduler = LTX2Scheduler()
        self.prompt_encoder = PromptEncoder(
            model_paths,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
        )
        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(),
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.audio_conditioner = AudioConditioner(
            model_paths.audio_vae(),
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            model_paths.video_vae(),
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            diffvae_optimization=diffvae_optimization,
        )
        self.audio_decoder = AudioDecoder(
            model_paths.audio_vae(),
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )

    # --------------------------------------------------------------------- #
    #  Public entry point                                                     #
    # --------------------------------------------------------------------- #

    def __call__(  # noqa: PLR0913
        self,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        *,
        fps: float | None = None,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        max_batch_size: int = 1,
        sigmas: torch.Tensor | None = None,
        color_space: HDRColorSpace | None = None,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor, TilingConfig | None]:
        """Regenerate ``[start_time, end_time]`` of the source video (retake).
        Parameters
        ----------
        video_path : str
            Path to the source video file, or a directory of scene-linear ``*.exr``
            frames (native HDR). Audio is optional for video files and absent for
            EXR folders.
        prompt : str
            Text prompt describing the *regenerated* section.
        start_time, end_time : float
            Time window (in seconds) of the section to regenerate.
        seed : int
            Random seed for reproducibility.
        fps : float | None
            Frame rate when *video_path* is an EXR-frame folder (required then).
            Ignored for regular video files.
        negative_prompt : str
            Negative prompt for CFG guidance (ignored in distilled mode).
        num_inference_steps : int
            Number of Euler denoising steps (ignored in distilled mode which
            uses a fixed 8-step schedule).
        video_guider_params, audio_guider_params : MultiModalGuiderParams | None
            Guidance parameters for video and audio modalities.  Ignored in
            distilled mode.
        regenerate_video : bool
            If ``True`` (default), regenerate video inside ``[start_time, end_time]``.
            If ``False``, video is preserved as-is (no regeneration).
        regenerate_audio : bool
            If True, regenerate audio in the [start_time, end_time] window; if False,
            audio is preserved as-is (no regeneration).
        enhance_prompt : bool
            Whether to enhance the prompt via the text encoder.
        Returns
        -------
        tuple[Iterator[torch.Tensor], torch.Tensor, TilingConfig]
            ``(video_frames_iterator, audio_waveform, tiling_config)``
        """
        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = self.dtype
        if vae_dtype is None:
            vae_dtype = dtype

        output_shape = get_videostream_metadata(video_path, fps=fps)

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1,
                frames=output_shape.frames,
                height=output_shape.height,
                width=output_shape.width,
                fps=output_shape.fps,
            ),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )

        initial_video_latent = self.image_conditioner(
            lambda enc: video_latent_from_file(
                video_encoder=enc,
                file_path=video_path,
                output_shape=output_shape,
                dtype=dtype,
                device=self.device,
                color_space=color_space,
            )
        )

        initial_audio_latent = self.audio_conditioner(
            lambda enc: audio_latent_from_file(
                audio_encoder=enc,
                file_path=video_path,
                output_shape=output_shape,
                dtype=dtype,
                device=self.device,
            )
        )

        prompts_to_encode = [prompt] if self.distilled else [prompt, negative_prompt]
        contexts = self.prompt_encoder(
            prompts_to_encode,
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_seed=seed,
        )

        v_context_p, a_context_p = contexts[0].video_encoding, contexts[0].audio_encoding
        video_modality_spec = ModalitySpec(
            context=v_context_p,
            conditionings=[TemporalRegionMask(start_time=start_time, end_time=end_time, fps=output_shape.fps)]
            if regenerate_video
            else [],
            initial_latent=initial_video_latent,
            frozen=not regenerate_video,
        )
        audio_modality_spec = ModalitySpec(
            context=a_context_p,
            conditionings=[TemporalRegionMask(start_time=start_time, end_time=end_time, fps=output_shape.fps)]
            if (initial_audio_latent is not None and regenerate_audio)
            else [],
            initial_latent=initial_audio_latent,
            frozen=initial_audio_latent is not None and not regenerate_audio,
        )

        # Build denoiser and resolve sigma schedule.
        if sigmas is None:
            sigmas = DISTILLED_SIGMAS if self.distilled else self._scheduler.execute(steps=num_inference_steps)
        sigmas = sigmas.to(dtype=torch.float32, device=self.device)

        if self.distilled:
            denoiser = SimpleDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
            )
        else:
            v_context_n, a_context_n = contexts[1].video_encoding, contexts[1].audio_encoding
            video_guider = MultiModalGuider(
                params=video_guider_params,
                negative_context=v_context_n,
            )
            audio_guider = MultiModalGuider(
                params=audio_guider_params,
                negative_context=a_context_n,
            )
            denoiser = GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=video_guider,
                audio_guider=audio_guider,
            )

        # Run diffusion stage
        video_state, audio_state = self.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=output_shape.width,
            height=output_shape.height,
            frames=output_shape.frames,
            fps=output_shape.fps,
            video=video_modality_spec,
            audio=audio_modality_spec,
            max_batch_size=max_batch_size,
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
        decoded_audio = self.audio_decoder(audio_state.latent)

        return decoded_video, decoded_audio, tiling_config


@torch.inference_mode()
def main() -> None:
    """CLI entry point for retake (regenerate a time region)."""
    logging.basicConfig(level=logging.INFO)
    parser = video_editing_arg_parser(distilled=True)
    parser.description = "Retake: regenerate a time region of a video with LTX-2."
    args = parser.parse_args()

    if args.start_time >= args.end_time:
        raise ValueError("start_time must be less than end_time")

    # Validate frame count (8k+1) and resolution (multiples of 32) at CLI stage
    video_scale = SpatioTemporalScaleFactors.default()
    exr_input = is_exr_dir(args.video_path)
    src = get_videostream_metadata(args.video_path, fps=args.frame_rate if exr_input else None)
    if (src.frames - 1) % video_scale.time != 0:
        snapped = ((src.frames - 1) // video_scale.time) * video_scale.time + 1
        raise ValueError(
            f"Video frame count must satisfy 8k+1 (e.g. 97, 193). Got {src.frames}; use a video with {snapped} frames."
        )
    if src.width % 32 != 0 or src.height % 32 != 0:
        raise ValueError(f"Video width and height must be multiples of 32. Got {src.width}x{src.height}.")

    pipeline = RetakePipeline(
        model_paths=args.model_paths,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        distilled=True,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )
    params = detect_params(args.model_paths.transformer())
    hdr = resolve_hdr_color_space(video_paths=[args.video_path], hdr=args.hdr)
    vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
    video_iter, audio, tiling_config = pipeline(
        video_path=args.video_path,
        prompt=args.prompt,
        start_time=args.start_time,
        end_time=args.end_time,
        seed=args.seed,
        fps=args.frame_rate if exr_input else None,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        video_guider_params=params.video_guider_params,
        audio_guider_params=params.audio_guider_params,
        vae_dtype=vae_dtype,
        color_space=hdr,
        tiling_config=AUTO_TILING,
        max_batch_size=args.max_batch_size,
    )

    video_chunks_number = get_video_chunks_number(src.frames, tiling_config)
    encode_video(
        video=video_iter,
        fps=int(src.fps),
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
        color_space=hdr,
    )


if __name__ == "__main__":
    main()
