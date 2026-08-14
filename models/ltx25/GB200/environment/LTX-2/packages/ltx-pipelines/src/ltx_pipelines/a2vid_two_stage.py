import logging
from collections.abc import Iterator

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import default_2_stage_arg_parser
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    STAGE_2_DISTILLED_SIGMAS,
)
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class A2VidPipelineTwoStage:
    """
    Two-stage audio to video generation pipeline.
    Stage 1 generates video at half the target resolution with audio conditioning
    (video-only denoising, audio frozen), then Stage 2 upsamples by 2x and refines
    both video and audio using a distilled LoRA for higher quality output.
    """

    def __init__(  # noqa: PLR0913
        self,
        checkpoint_path: str,
        vae_checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
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
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._scheduler = LTX2Scheduler()

        self.prompt_encoder = PromptEncoder(
            checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, self.device, registry=registry, alloc_trim_strategy=alloc_trim_strategy
        )
        self.audio_conditioner = AudioConditioner(
            checkpoint_path, self.dtype, self.device, registry=registry, alloc_trim_strategy=alloc_trim_strategy
        )
        self.stage_1 = DiffusionStage.from_checkpoint(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        stage_2_loras = (*tuple(loras), *tuple(distilled_lora))
        self.stage_2 = DiffusionStage.from_checkpoint(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=stage_2_loras,
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.upsampler = VideoUpsampler(
            checkpoint_path,
            spatial_upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            vae_checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            vae_compile=vae_compile,
        )

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
        video_guider_params: MultiModalGuiderParams,
        images: list[tuple[str, int, float]],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, _ = ctx_n.video_encoding, ctx_n.audio_encoding

        # Encode audio.
        decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, audio_max_duration)
        if decoded_audio is None:
            raise ValueError(f"Failed to decode audio from {audio_path}. Please check the file and try again.")

        encoded_audio_latent = self.audio_conditioner(lambda enc: vae_encode_audio(decoded_audio, enc, None))
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
        encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]

        # Stage 1: encode image conditionings with the VAE encoder, then denoise
        # video-only (audio frozen).
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        sigmas = (
            stage_1_sigmas if stage_1_sigmas is not None else self._scheduler.execute(steps=num_inference_steps)
        ).to(dtype=torch.float32, device=self.device)

        video_state, _ = self.stage_1(
            denoiser=GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider=MultiModalGuider(
                    params=MultiModalGuiderParams(),
                ),
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
                frozen=True,
                noise_scale=0.0,
                initial_latent=encoded_audio_latent,
            ),
            max_batch_size=max_batch_size,
        )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LoRA.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        video_state, _ = self.stage_2(
            denoiser=SimpleDenoiser(v_context_p, a_context_p),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                frozen=True,
                noise_scale=0.0,
                initial_latent=encoded_audio_latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)

        # Return the original input audio instead of VAE-decoded audio to preserve fidelity.
        # decode_audio_from_file already returns normalised [-1, 1] float values.
        original_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)

        return decoded_video, original_audio


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = default_2_stage_arg_parser()
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the audio file to condition the video generation.",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds to read audio from (default: 0.0).",
    )
    parser.add_argument(
        "--audio-max-duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds. Defaults to video duration (num_frames / frame_rate).",
    )
    args = parser.parse_args()
    pipeline = A2VidPipelineTwoStage(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        vae_checkpoint_path=args.vae_checkpoint_path,
        vae_compile=args.vae_compile,
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
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        audio_path=args.audio_path,
        audio_start_time=args.audio_start_time,
        audio_max_duration=args.audio_max_duration
        if args.audio_max_duration is not None
        else args.num_frames / args.frame_rate,
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
