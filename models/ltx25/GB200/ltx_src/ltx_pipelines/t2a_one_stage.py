import logging

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
from ltx_core.model.transformer import LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP, LTXAudioOnlyModelConfigurator
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils import get_device
from ltx_pipelines.utils.args import (
    default_1_stage_t2a_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    DurationPredictor,
    PromptEncoder,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.media_io import encode_audio
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration, ModalitySpec, OffloadMode

# Placeholder pixel dimensions used for ``VideoPixelShape`` construction.
# Audio-only generation reads ``frames`` and ``fps`` from the pixel shape via
# ``AudioLatentShape.from_video_pixel_shape`` (height/width are unused).
_AUDIO_ONLY_PLACEHOLDER_RES = 512


class T2AOneStagePipeline:
    """
    Single-stage text-to-audio generation pipeline.
    Generates audio at the target duration in a single diffusion pass with
    classifier-free guidance (CFG) on the audio modality only. The video
    modality is fully absent — the transformer runs audio-only by passing
    ``video=None`` to the ``DiffusionStage``.
    Assumes full non distilled model is provided in the checkpoint_path.
    """

    def __init__(
        self,
        model_paths: ModelPaths,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
    ):
        self.dtype = torch.bfloat16
        self.device = device or get_device()
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
        # Audio-only: build an audio-only transformer (model_configurator) so the video
        # weights are never instantiated, plus a use-case-specific SDOps that restricts
        # checkpoint reads to the audio model's keys, so the video weights are never even
        # read from disk (the loader skips any key the SDOps maps to None).
        self.stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            model_configurator=LTXAudioOnlyModelConfigurator,
            model_sd_ops=LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.audio_decoder = AudioDecoder(
            model_paths.audio_vae(),
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        # None on checkpoints that predate DurationHead (LTX 2.5 / gemma4 only) -- __call__ requires
        # an explicit num_frames in that case instead of crashing deep in a forward pass.
        self.duration_predictor = DurationPredictor.from_checkpoint(
            checkpoint_path=model_paths.duration_head_path,
            dtype=self.dtype,
            device=self.device,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        frame_rate: float,
        num_inference_steps: int,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        max_batch_size: int = 1,
        sigmas: torch.Tensor | None = None,
    ) -> Audio:
        require_num_frames_source(num_frames, self.duration_predictor)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=None,
            enhance_prompt_seed=seed,
        )
        a_context_p = ctx_p.audio_encoding
        a_context_n = ctx_n.audio_encoding

        num_frames = resolve_num_frames(
            num_frames, self.duration_predictor, video_encoding=None, audio_encoding=a_context_p, frame_rate=frame_rate
        )

        sigmas = (sigmas if sigmas is not None else self._scheduler.execute(steps=num_inference_steps)).to(
            dtype=torch.float32, device=self.device
        )

        # Normalize to a guider factory. Plain ``MultiModalGuiderParams`` (the default /
        # CLI case) becomes a simple sigma-independent guider, but callers may also pass
        # their own factory for sigma-dependent guidance; ``FactoryGuidedDenoiser`` always
        # consumes a factory.
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_context_n,
        )

        _, audio_state = self.stage(
            denoiser=FactoryGuidedDenoiser(
                v_context=None,
                a_context=a_context_p,
                video_guider_factory=None,
                audio_guider_factory=audio_guider_factory,
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=_AUDIO_ONLY_PLACEHOLDER_RES,
            height=_AUDIO_ONLY_PLACEHOLDER_RES,
            frames=num_frames,
            fps=frame_rate,
            video=None,
            audio=ModalitySpec(context=a_context_p),
            max_batch_size=max_batch_size,
        )

        return self.audio_decoder(audio_state.latent)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params()
    parser = default_1_stage_t2a_arg_parser(params=params)
    args = parser.parse_args()
    pipeline = T2AOneStagePipeline(
        model_paths=args.model_paths,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
    )
    audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            # Audio-only generation has no video modality, so the video->audio
            # (v2a) cross-modal guidance is meaningless here. 1.0 disables it.
            modality_scale=1.0,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        max_batch_size=args.max_batch_size,
    )

    encode_audio(audio=audio, output_path=args.output_path)


if __name__ == "__main__":
    main()
