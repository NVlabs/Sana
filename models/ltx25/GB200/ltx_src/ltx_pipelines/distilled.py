import logging
from collections.abc import Iterator, Sequence
from functools import partial
from typing import Any

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    add_generated_keyframes_arg,
    default_2_stage_distilled_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    DurationPredictor,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    detect_model_version,
)
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    ensure_tiling_config,
    generated_keyframe_conditionings,
    get_device,
    has_generated_keyframes,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import (
    HDRColorSpace,
    encode_video,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration, ModalitySpec, OffloadMode

# Generation from which stage 1 is sampled with the ancestral (SDE) Euler sampler instead of the
# deterministic one.
ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)

# Fully ancestral noise injection: eta=0 is a plain Euler step, eta=1 injects the full
# variance-preserving amount at every step.
ANCESTRAL_ETA = 1.0
ANCESTRAL_S_NOISE = 1.0

# The loop's noise generator is seeded from the pipeline seed plus this offset. Without it the
# loop's first draw would be bit-identical to the initial latent noise: GaussianNoiser and the
# loop's ``_get_plain_noise`` both draw ``torch.randn`` at the same shape, dtype, and device from a
# freshly seeded generator. Mirrors the substep-seed offset in ``res2s_audio_video_denoising_loop``.
ANCESTRAL_NOISE_SEED_OFFSET = 10000


def should_use_ancestral_sampler(transformer_path: str) -> bool:
    """Whether a checkpoint's generation calls for the ancestral stage-1 sampler.
    Takes the transformer checkpoint (``ModelPaths.transformer()``) since that is the component
    carrying ``model_version`` in both the monolith and split layouts.
    This is what :class:`DistilledPipeline` resolves at construction into
    ``self.use_ancestral_sampler``; a named function rather than an inline comparison so the rule
    can be tested, and reused, without loading a checkpoint's weights.
    """
    return detect_model_version(transformer_path) >= ANCESTRAL_SAMPLER_SINCE_VERSION


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
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
    ):
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
        self.stage = DiffusionStage.from_checkpoint(
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
        # None on checkpoints that predate DurationHead (LTX 2.5 / gemma4 only) -- __call__ requires
        # an explicit num_frames in that case instead of crashing deep in a forward pass.
        self.duration_predictor = DurationPredictor.from_checkpoint(
            model_paths.duration_head_path,
            self.dtype,
            self.device,
        )
        self.use_ancestral_sampler = should_use_ancestral_sampler(model_paths.transformer())

    def _stage_1_sampler_kwargs(self, seed: int) -> dict[str, Any]:
        """Resolve stage 1's ``stepper`` / ``loop`` overrides for :class:`DiffusionStage`.
        Returns an empty dict for the deterministic sampler, letting ``DiffusionStage`` apply its
        own ``EulerDiffusionStep`` + ``euler_denoising_loop`` defaults rather than restating them.
        """
        if not self.use_ancestral_sampler:
            return {}

        return {
            "stepper": EulerAncestralDiffusionStep(eta=ANCESTRAL_ETA, s_noise=ANCESTRAL_S_NOISE),
            "loop": partial(
                euler_ancestral_denoising_loop,
                noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
                model_dtype=self.dtype,
            ),
        }

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: HDRColorSpace | None = None,
        generated_keyframes: int | Sequence[int] = 0,
    ) -> tuple[Iterator[torch.Tensor], Audio, int, TilingConfig | None]:
        """Generate a video.
        Stage 1 samples with the ancestral (SDE) Euler sampler or the deterministic one according
        to ``self.use_ancestral_sampler``, detected from the checkpoint generation. Stage 2 is
        always deterministic -- its 3-step refinement schedule is too short to remove freshly
        injected noise.
        """
        require_num_frames_source(num_frames, self.duration_predictor)
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)
        if has_generated_keyframes(generated_keyframes):
            self.stage.assert_generated_keyframes_supported()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        if vae_dtype is None:
            vae_dtype = dtype

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        num_frames = resolve_num_frames(
            num_frames,
            self.duration_predictor,
            video_encoding=video_context,
            audio_encoding=audio_context,
            frame_rate=frame_rate,
        )

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
        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_w, stage_1_h = width // 2, height // 2
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                color_space=color_space,
            )
        )
        stage_1_conditionings.extend(generated_keyframe_conditionings(generated_keyframes, num_frames))

        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
            **self._stage_1_sampler_kwargs(seed),
        )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                color_space=color_space,
            )
        )

        video_state, audio_state = self.stage(
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
        return decoded_video, decoded_audio, num_frames, tiling_config


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params(distilled=True)
    parser = add_generated_keyframes_arg(
        default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True)
    )
    args = parser.parse_args()
    pipeline = DistilledPipeline(
        model_paths=args.model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )
    hdr = resolve_hdr_color_space(images=args.images, hdr=args.hdr)
    vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
    video, audio, num_frames, tiling_config = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        vae_dtype=vae_dtype,
        color_space=hdr,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        tiling_config=AUTO_TILING,
        generated_keyframes=args.num_generated_keyframes,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
        color_space=hdr,
    )


if __name__ == "__main__":
    main()
