import logging
from collections.abc import Iterator

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.iclora_utils import (
    append_ic_lora_reference_video_conditionings,
    read_lora_reference_downscale_factor,
    read_lora_reference_temporal_scale_factor,
)
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    VideoConditioningAction,
    VideoMaskConditioningAction,
    default_2_stage_distilled_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
)
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    ensure_tiling_config,
    get_device,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import (
    HDRColorSpace,
    decode_video_by_frame,
    encode_video,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
    video_preprocess,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class ICLoraPipeline:
    """
    Two-stage video generation pipeline with In-Context (IC) LoRA support.
    Allows conditioning the generated video on control signals such as depth maps,
    human pose, or image edges via the video_conditioning parameter.
    The specific IC-LoRA model should be provided via the loras parameter.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    Both stages use distilled models for efficiency.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ) -> None:
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(
            model_paths,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
        )
        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage_1 = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage_2 = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=(),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.upsampler = VideoUpsampler(
            model_paths.video_vae(),
            spatial_upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            diffvae_optimization=diffvae_optimization,
        )
        self.audio_decoder = AudioDecoder(
            model_paths.audio_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )

        # Read reference scale factors from LoRA metadata.
        # IC-LoRAs trained with scaled reference videos store these factors
        # so inference can resize/subsample reference videos to match training conditions.
        self.reference_downscale_factor = 1
        self.reference_temporal_scale_factor = 1
        for lora in loras:
            scale = read_lora_reference_downscale_factor(lora.path)
            if scale != 1:
                if self.reference_downscale_factor not in (1, scale):
                    raise ValueError(
                        f"Conflicting reference_downscale_factor values in LoRAs: "
                        f"already have {self.reference_downscale_factor}, but {lora.path} "
                        f"specifies {scale}. Cannot combine LoRAs with different reference scales."
                    )
                self.reference_downscale_factor = scale
            temporal = read_lora_reference_temporal_scale_factor(lora.path)
            if temporal != 1:
                if self.reference_temporal_scale_factor not in (1, temporal):
                    raise ValueError(
                        f"Conflicting reference_temporal_scale_factor values in LoRAs: "
                        f"already have {self.reference_temporal_scale_factor}, but {lora.path} "
                        f"specifies {temporal}. Cannot combine LoRAs with different temporal scales."
                    )
                self.reference_temporal_scale_factor = temporal

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: HDRColorSpace | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio, TilingConfig | None]:
        """
        Generate video with IC-LoRA conditioning.
        Args:
            prompt: Text prompt for video generation.
            seed: Random seed for reproducibility.
            height: Output video height in pixels (must be divisible by 64).
            width: Output video width in pixels (must be divisible by 64).
            num_frames: Number of frames to generate.
            frame_rate: Output video frame rate.
            images: List of (path, frame_idx, strength) tuples for image conditioning.
            video_conditioning: List of (path, strength) tuples for IC-LoRA video conditioning.
            enhance_prompt: Whether to enhance the prompt using the text encoder.
            tiling_config: Optional tiling configuration for VAE decoding.
            conditioning_attention_strength: Scale factor for IC-LoRA conditioning attention.
                Controls how strongly the conditioning video influences the output.
                0.0 = ignore conditioning, 1.0 = full conditioning influence. Default 1.0.
                When conditioning_attention_mask is provided, the mask is multiplied by
                this strength before being passed to the conditioning items.
            skip_stage_2: If True, skip Stage 2 upsampling and refinement. Output will be
                at half resolution (height//2, width//2). Default is False.
            conditioning_attention_mask: Optional pixel-space attention mask with the same
                spatial-temporal dimensions as the input reference video. Shape should be
                (B, 1, F, H, W) or (1, 1, F, H, W) where F, H, W match the reference
                video's pixel dimensions. Values in [0, 1].
                The mask is downsampled to latent space using VAE scale factors (with
                causal temporal handling for the first frame), then multiplied by
                conditioning_attention_strength.
                When None (default): scalar conditioning_attention_strength is used
                directly.
        Returns:
            Tuple of (video_iterator, audio_tensor).
        """
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        if vae_dtype is None:
            vae_dtype = self.dtype

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )

        # Stage 1: Initial low resolution video generation.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )

        # Encode conditionings using the video encoder block
        stage_1_conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                num_frames=num_frames,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
                color_space=color_space,
            )
        )

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)

        video_state, audio_state = self.stage_1(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(
                context=audio_context,
            ),
        )

        if skip_stage_2:
            # Skip Stage 2: Decode directly from Stage 1 output at half resolution
            logging.info("[IC-LoRA] Skipping Stage 2 (--skip-stage-2 enabled)")
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio, tiling_config

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
                color_space=color_space,
            )
        )

        video_state, audio_state = self.stage_2(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio, tiling_config

    def _create_conditionings(
        self,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
        color_space: HDRColorSpace | None = None,
    ) -> list[ConditioningItem]:
        """
        Create conditioning items for video generation.
        Args:
            conditioning_attention_strength: Scalar attention weight in [0, 1].
                If conditioning_attention_mask is also provided, the downsampled mask
                is multiplied by this strength. Otherwise this scalar is passed
                directly as the attention mask.
            conditioning_attention_mask: Optional pixel-space attention mask with shape
                (B, 1, F_pixel, H_pixel, W_pixel) matching the reference video's
                pixel dimensions. Downsampled to latent space with causal temporal
                handling, then multiplied by conditioning_attention_strength.
        Returns:
            List of conditioning items. IC-LoRA conditionings are appended last.
        """
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
            color_space=color_space,
        )

        append_ic_lora_reference_video_conditionings(
            conditionings,
            video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
            reference_downscale_factor=self.reference_downscale_factor,
            reference_temporal_scale_factor=self.reference_temporal_scale_factor,
            conditioning_attention_strength=conditioning_attention_strength,
            conditioning_attention_mask=conditioning_attention_mask,
            tiling_config=None,
            color_space=color_space,
        )

        if video_conditioning:
            logging.info("[IC-LoRA] Added %d video conditioning(s)", len(video_conditioning))

        return conditionings


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params(distilled=True)
    parser = default_2_stage_distilled_arg_parser(params=params)
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
        help=(
            "IC-LoRA reference: video file (SDR) or directory of scene-linear *.exr frames (HDR), "
            "plus strength. Example: --video-conditioning ref.mp4 1.0  or  --video-conditioning exr_dir/ 1.0"
        ),
    )
    parser.add_argument(
        "--conditioning-attention-mask",
        action=VideoMaskConditioningAction,
        nargs=2,
        metavar=("MASK_PATH", "STRENGTH"),
        default=None,
        help=(
            "Optional spatial attention mask: path to a grayscale mask video and "
            "attention strength. The mask video pixel values in [0,1] control "
            "per-region conditioning attention strength. The strength scalar is "
            "multiplied with the spatial mask. "
            "0.0 = ignore IC-LoRA conditioning, 1.0 = full conditioning influence. "
            "When not provided, full conditioning strength (1.0) is used. "
            "Example: --conditioning-attention-mask path/to/mask.mp4 0.5"
        ),
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help=(
            "Skip Stage 2 upsampling and refinement. Output will be at half resolution "
            "(height//2, width//2). Useful for faster iteration or when GPU memory is limited."
        ),
    )
    args = parser.parse_args()

    # Load mask video if provided via --conditioning-attention-mask
    conditioning_attention_mask = None
    conditioning_attention_strength = 1.0
    if args.conditioning_attention_mask is not None:
        mask_path, mask_strength = args.conditioning_attention_mask
        conditioning_attention_strength = mask_strength
        conditioning_attention_mask = _load_mask_video(
            mask_path=mask_path,
            height=args.height // 2,  # Stage 1 operates at half resolution
            width=args.width // 2,
            num_frames=args.num_frames,
        )

    pipeline = ICLoraPipeline(
        model_paths=args.model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )
    hdr = resolve_hdr_color_space(
        images=args.images,
        video_paths=[path for path, _ in args.video_conditioning],
        hdr=args.hdr,
    )
    vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
    video, audio, tiling_config = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        video_conditioning=args.video_conditioning,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        vae_dtype=vae_dtype,
        color_space=hdr,
        tiling_config=AUTO_TILING,
        conditioning_attention_strength=conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
        conditioning_attention_mask=conditioning_attention_mask,
    )
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
        color_space=hdr,
    )


def _load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> torch.Tensor:
    """Load a mask video and return a pixel-space tensor of shape (1, 1, F, H, W).
    The mask video is loaded, resized to (height, width), converted to
    grayscale, and normalised to [0, 1].
    Args:
        mask_path: Path to the mask video file.
        height: Target height in pixels.
        width: Target width in pixels.
        num_frames: Maximum number of frames to load.
    Returns:
        Tensor of shape ``(1, 1, F, H, W)`` with values in ``[0, 1]``.
    """
    device = get_device()
    frame_gen = decode_video_by_frame(path=mask_path, frame_cap=num_frames, device=device)
    mask_video = video_preprocess(frame_gen, height, width, torch.bfloat16, device)
    # mask_video shape: (1, C, F, H, W) — take mean over channels for grayscale
    mask = mask_video.mean(dim=1, keepdim=True)  # (1, 1, F, H, W)
    # Normalise to [0, 1] — video_preprocess applies normalize_latent,
    # so undo that: values are in [-1, 1], remap to [0, 1]
    mask = (mask + 1.0) / 2.0
    return mask.clamp(0.0, 1.0)


if __name__ == "__main__":
    main()
