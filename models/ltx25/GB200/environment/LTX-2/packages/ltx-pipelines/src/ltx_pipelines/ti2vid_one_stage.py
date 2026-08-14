import logging
from collections.abc import Iterator

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_1_stage_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    DurationPredictor,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration, ModalitySpec, OffloadMode


class TI2VidOneStagePipeline:
    """
    Single-stage text/image-to-video generation pipeline.
    Generates video at the target resolution in a single diffusion pass with
    classifier-free guidance (CFG). Supports optional image conditioning via
    the images parameter.
    Assumes full non distilled model is provided in the checkpoint_path.
    """

    def __init__(  # noqa: PLR0913
        self,
        checkpoint_path: str,
        vae_checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        vae_compile: bool = False,
    ):
        self.dtype = torch.bfloat16
        self.device = device or get_device()
        self._scheduler = LTX2Scheduler()
        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage = DiffusionStage.from_checkpoint(
            checkpoint_path=checkpoint_path,
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
            checkpoint_path=vae_checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            vae_compile=vae_compile,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        # None on checkpoints that predate DurationHead (2.4+ only) -- __call__ requires
        # an explicit num_frames in that case instead of crashing deep in a forward pass.
        self.duration_predictor = DurationPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        tiling_config: TilingConfig | None = None,
        max_batch_size: int = 1,
        sigmas: torch.Tensor | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        require_num_frames_source(num_frames, self.duration_predictor)
        assert_resolution(height=height, width=width, is_two_stage=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        num_frames = resolve_num_frames(
            num_frames,
            self.duration_predictor,
            video_encoding=v_context_p,
            audio_encoding=a_context_p,
            frame_rate=frame_rate,
        )

        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        sigmas = (sigmas if sigmas is not None else self._scheduler.execute(steps=num_inference_steps)).to(
            dtype=torch.float32, device=self.device
        )

        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_context_n,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_context_n,
        )

        video_state, audio_state = self.stage(
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=video_guider_factory,
                audio_guider_factory=audio_guider_factory,
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(
                context=a_context_p,
            ),
            max_batch_size=max_batch_size,
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator=generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params()
    parser = default_1_stage_arg_parser(params=params, supports_auto_duration=True)
    args = parser.parse_args()
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        vae_checkpoint_path=args.vae_checkpoint_path,
        vae_compile=args.vae_compile,
    )
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
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        max_batch_size=args.max_batch_size,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=1,
    )


if __name__ == "__main__":
    main()
