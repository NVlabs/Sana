import logging
from collections.abc import Iterator

import torch

from diffusion.refiner.vendor.ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from diffusion.refiner.vendor.ltx_core.components.noisers import GaussianNoiser
from diffusion.refiner.vendor.ltx_core.components.schedulers import LTX2Scheduler
from diffusion.refiner.vendor.ltx_core.loader import LoraPathStrengthAndSDOps
from diffusion.refiner.vendor.ltx_core.loader.registry import Registry
from diffusion.refiner.vendor.ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from diffusion.refiner.vendor.ltx_core.quantization import QuantizationPolicy
from diffusion.refiner.vendor.ltx_core.types import Audio, VideoPixelShape
from diffusion.refiner.vendor.ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_arg_parser,
    detect_checkpoint_path,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.constants import (
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser
from diffusion.refiner.vendor.ltx_pipelines.utils.helpers import (
    assert_resolution,
    get_device,
    image_conditionings_by_adding_guiding_latent,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.media_io import encode_video
from diffusion.refiner.vendor.ltx_pipelines.utils.types import ModalitySpec


class KeyframeInterpolationPipeline:
    """
    Keyframe-based Two-stage video interpolation pipeline.
    Interpolates between keyframes to generate a video with smoother transitions.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    Stage 1 uses full model while Stage 2 uses distilled LORA for efficiency,
    as the upsampled video already has good quality and just needs refinement.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(checkpoint_path, gemma_root, self.dtype, self.device, registry=registry)
        self.image_conditioner = ImageConditioner(checkpoint_path, self.dtype, self.device, registry=registry)
        self.stage_1 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        stage_2_loras = (*tuple(loras), *tuple(distilled_lora))
        self.stage_2 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=stage_2_loras,
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(checkpoint_path, self.dtype, self.device, registry=registry)

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
        max_batch_size: int = 1,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
            streaming_prefetch_count=streaming_prefetch_count,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Stage 1: Initial low resolution video generation.
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = self.image_conditioner(
            lambda enc: image_conditionings_by_adding_guiding_latent(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_context_n,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_context_n,
        )

        video_state, audio_state = self.stage_1(
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=video_guider_factory,
                audio_guider_factory=audio_guider_factory,
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(
                context=a_context_p,
            ),
            streaming_prefetch_count=streaming_prefetch_count,
            max_batch_size=max_batch_size,
        )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: image_conditionings_by_adding_guiding_latent(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        video_state, audio_state = self.stage_2(
            denoiser=SimpleDenoiser(v_context_p, a_context_p),
            sigmas=distilled_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
            streaming_prefetch_count=streaming_prefetch_count,
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_2_stage_arg_parser(params=params)
    args = parser.parse_args()
    pipeline = KeyframeInterpolationPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        torch_compile=args.compile,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
        streaming_prefetch_count=args.streaming_prefetch_count,
        max_batch_size=args.max_batch_size,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
