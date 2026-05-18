from __future__ import annotations

import logging
from collections.abc import Iterator

import torch

from diffusion.refiner.vendor.ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from diffusion.refiner.vendor.ltx_core.components.noisers import GaussianNoiser
from diffusion.refiner.vendor.ltx_core.components.schedulers import LTX2Scheduler
from diffusion.refiner.vendor.ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
from diffusion.refiner.vendor.ltx_core.loader import LoraPathStrengthAndSDOps
from diffusion.refiner.vendor.ltx_core.loader.registry import Registry
from diffusion.refiner.vendor.ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from diffusion.refiner.vendor.ltx_core.quantization import QuantizationPolicy
from diffusion.refiner.vendor.ltx_core.types import (
    SpatioTemporalScaleFactors,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.args import video_editing_arg_parser
from diffusion.refiner.vendor.ltx_pipelines.utils.blocks import (
    AudioConditioner,
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, detect_params
from diffusion.refiner.vendor.ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from diffusion.refiner.vendor.ltx_pipelines.utils.helpers import (
    audio_latent_from_file,
    get_device,
    video_latent_from_file,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.media_io import (
    encode_video,
    get_videostream_metadata,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.types import ModalitySpec


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

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        distilled: bool = True,
        torch_compile: bool = False,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.distilled = distilled
        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )
        self.audio_conditioner = AudioConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )
        self.stage = DiffusionStage(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
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
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        streaming_prefetch_count: int | None = None,
        max_batch_size: int = 1,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        """Regenerate ``[start_time, end_time]`` of the source video (retake).
        Parameters
        ----------
        video_path : str
            Path to the source video file (must contain video; audio is optional).
        prompt : str
            Text prompt describing the *regenerated* section.
        start_time, end_time : float
            Time window (in seconds) of the section to regenerate.
        seed : int
            Random seed for reproducibility.
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
        tuple[Iterator[torch.Tensor], torch.Tensor]
            ``(video_frames_iterator, audio_waveform)``
        """
        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = self.dtype

        output_shape = get_videostream_metadata(video_path)
        initial_video_latent = self.image_conditioner(
            lambda enc: video_latent_from_file(
                video_encoder=enc,
                file_path=video_path,
                output_shape=output_shape,
                dtype=dtype,
                device=self.device,
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
            enhance_prompt_seed=seed,
            streaming_prefetch_count=streaming_prefetch_count,
        )

        v_context_p, a_context_p = contexts[0].video_encoding, contexts[0].audio_encoding
        video_modality_spec = ModalitySpec(
            context=v_context_p,
            conditionings=(
                [TemporalRegionMask(start_time=start_time, end_time=end_time, fps=output_shape.fps)]
                if regenerate_video
                else []
            ),
            initial_latent=initial_video_latent,
            frozen=not regenerate_video,
        )
        audio_modality_spec = ModalitySpec(
            context=a_context_p,
            conditionings=(
                [TemporalRegionMask(start_time=start_time, end_time=end_time, fps=output_shape.fps)]
                if (initial_audio_latent is not None and regenerate_audio)
                else []
            ),
            initial_latent=initial_audio_latent,
            frozen=initial_audio_latent is not None and not regenerate_audio,
        )
        # Build denoiser
        if self.distilled:
            sigmas = torch.tensor(DISTILLED_SIGMA_VALUES).to(dtype=torch.float32, device=self.device)
            denoiser = SimpleDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
            )
        else:
            sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
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
            streaming_prefetch_count=streaming_prefetch_count,
            max_batch_size=max_batch_size,
        )

        # Decode
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)

        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    """CLI entry point for retake (regenerate a time region)."""
    logging.getLogger().setLevel(logging.INFO)
    parser = video_editing_arg_parser(distilled=True)
    parser.description = "Retake: regenerate a time region of a video with LTX-2."
    args = parser.parse_args()

    if args.start_time >= args.end_time:
        raise ValueError("start_time must be less than end_time")

    # Validate frame count (8k+1) and resolution (multiples of 32) at CLI stage
    video_scale = SpatioTemporalScaleFactors.default()
    src = get_videostream_metadata(args.video_path)
    if (src.frames - 1) % video_scale.time != 0:
        snapped = ((src.frames - 1) // video_scale.time) * video_scale.time + 1
        raise ValueError(
            f"Video frame count must satisfy 8k+1 (e.g. 97, 193). Got {src.frames}; use a video with {snapped} frames."
        )
    if src.width % 32 != 0 or src.height % 32 != 0:
        raise ValueError(f"Video width and height must be multiples of 32. Got {src.width}x{src.height}.")

    pipeline = RetakePipeline(
        checkpoint_path=args.distilled_checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        distilled=args.distilled,
        torch_compile=args.compile,
    )
    params = detect_params(args.distilled_checkpoint_path)
    tiling_config = TilingConfig.default()
    video_iter, audio = pipeline(
        video_path=args.video_path,
        prompt=args.prompt,
        start_time=args.start_time,
        end_time=args.end_time,
        seed=args.seed,
        video_guider_params=params.video_guider_params,
        audio_guider_params=params.audio_guider_params,
        tiling_config=tiling_config,
        streaming_prefetch_count=args.streaming_prefetch_count,
        max_batch_size=args.max_batch_size,
    )
    video_chunks_number = get_video_chunks_number(src.frames, tiling_config)
    encode_video(
        video=video_iter,
        fps=int(src.fps),
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
