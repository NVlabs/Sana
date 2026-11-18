"""DFR (Diffusion Fidelity Rendering): keyframe-slot base, spatial detailing, tiled temporal rounds."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import replace as dataclass_replace
from functools import partial

import torch
from safetensors import safe_open

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByReferenceLatent,
    VideoGeneratedKeyframeSlots,
)
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import VIDEO_SCALE_FACTORS, Audio, LatentState, VideoPixelShape
from ltx_pipelines.dfr_layout import (
    remap_positions_to_local,
    resolve_canvas,
    stitch_tile_latents,
    tile_ranges,
)
from ltx_pipelines.iclora_utils import read_lora_reference_downscale_factor
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    LoraAction,
    default_2_stage_arg_parser,
    resolve_cli_params,
    resolve_existing_path,
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
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    ensure_tiling_config,
    get_device,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration, ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)

# Anchor keyframes carried between temporal rounds are ours, pinned just short of fully clean so a
# tile can still settle its seam frame.
_ANCHOR_KEYFRAME_STRENGTH = 0.95
_TEMPORAL_ANCESTRAL_ETA = 0.5
# Conditioning fps is capped independently of playback fps. RoPE time is ``pixel_frame / fps``, so a
# 120 fps time base halves every token's temporal span versus the trained distribution and the model
# can no longer lay out the 8 pixel frames inside one latent token -- it decodes as a motion spike at
# each latent border followed by a stall. Playback fps is used for decoding only.
_MAX_CONDITIONING_FPS = 60.0


def _keyframe_conditionings_from_latents(
    keyframes: torch.Tensor,
    positions: Sequence[int],
    strength: float,
) -> list[ConditioningItem]:
    """Build ``VideoConditionByKeyframeIndex`` guides from already-encoded keyframe latents."""
    if keyframes.ndim != 5:
        raise ValueError(f"Expected keyframes (B, C, K, H, W), got {tuple(keyframes.shape)}")
    if keyframes.shape[2] != len(positions):
        raise ValueError(f"Expected {len(positions)} keyframe latents, got K={keyframes.shape[2]}")
    return [
        VideoConditionByKeyframeIndex(
            keyframes=keyframes[:, :, index : index + 1],
            frame_idx=int(frame_idx),
            strength=strength,
        )
        for index, frame_idx in enumerate(positions)
    ]


def _slot_initials_from_video(
    video_latent: torch.Tensor,
    positions: Sequence[int],
    temporal_scale: int,
) -> torch.Tensor:
    """Stack the nearest video latent frames as ``(B, C, K, H, W)`` slot seeds."""
    frames = []
    for position in positions:
        index = min(max(round(int(position) / temporal_scale), 0), video_latent.shape[2] - 1)
        frames.append(video_latent[:, :, index : index + 1])
    return torch.cat(frames, dim=2)


def _merge_carry_forward_keyframes(
    anchor_positions: Sequence[int],
    anchor_latents: torch.Tensor | None,
    slot_positions: Sequence[int],
    slot_latents: torch.Tensor | None,
) -> tuple[list[int], torch.Tensor]:
    """Build the next round's anchor bag: carried keyframe stills plus this round's denoised slots.
    Positions must already be on the current round's pixel grid; callers remap (x2) for the next round.
    """
    by_position: dict[int, torch.Tensor] = {}
    for positions, latents, label in (
        (anchor_positions, anchor_latents, "anchor"),
        (slot_positions, slot_latents, "slot"),
    ):
        if not positions:
            continue
        if latents is None:
            raise RuntimeError(f"Missing {label} keyframe latents for carry-forward merge")
        if latents.shape[2] != len(positions):
            raise ValueError(f"{label} latents K={latents.shape[2]} != {len(positions)} positions")
        for index, position in enumerate(positions):
            by_position[int(position)] = latents[:, :, index : index + 1]
    if not by_position:
        raise RuntimeError("Carry-forward keyframe bag is empty")
    ordered = sorted(by_position)
    return ordered, torch.cat([by_position[position] for position in ordered], dim=2)


def _detailing_downscale_factor(lora_path: str) -> int:
    """Prefer LoRA metadata; default to 2 for x2 spatial detailing."""
    try:
        with safe_open(lora_path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            if "reference_downscale_factor" in metadata:
                return int(metadata["reference_downscale_factor"])
    except Exception as exc:
        logger.warning("Failed to read detailing LoRA metadata from %s: %s", lora_path, exc)
    factor = read_lora_reference_downscale_factor(lora_path)
    return factor if factor != 1 else 2


class DFRPipeline:
    """
    DFR pipeline on a keyframe-slot-capable SFT base plus a distilled LoRA.
    Stage 1 (half-res) generates video and keyframe slots on an x8-border segment grid; the half-res
    video is reserved as the IC-LoRA reference while video and slots are spatially latent-upsampled.
    Stage 2 re-denoises at full resolution with the distilled LoRA and an optional x2 detailing
    IC-LoRA. Shipped audio comes from stage 1: stage 2 still runs an audio pass because video needs
    the cross-modal attention, but it re-noises audio under the detailing LoRA and no later stage
    refines it.
    Optional ``temporal_upsample_rounds`` (0-2): each round temporally x2-upsamples, splits the canvas
    into ``2**round`` keyframe-seam tiles, invents mid-segment slots per tile, densifies with ancestral
    Euler, and stitches. The caller always gets ``(num_frames - 1) * 2**rounds + 1`` frames even when
    the canvas padded its tail.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        detailing_lora: list[LoraPathStrengthAndSDOps] | None = None,
        temporal_upsampler_path: str | None = None,
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
        self._distilled_lora = tuple(distilled_lora)
        self._user_loras = tuple(loras)
        self._detailing_lora = tuple(detailing_lora or ())
        self._detailing_downscale = (
            _detailing_downscale_factor(self._detailing_lora[0].path) if self._detailing_lora else 2
        )

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
        stage_loras = (*self._user_loras, *self._distilled_lora)
        self.stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=stage_loras,
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage_detailing = (
            self.stage.with_loras((*stage_loras, *self._detailing_lora)) if self._detailing_lora else self.stage
        )
        self.upsampler = VideoUpsampler(
            model_paths.video_vae(),
            spatial_upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.temporal_upsampler = (
            VideoUpsampler(
                model_paths.video_vae(),
                temporal_upsampler_path,
                self.dtype,
                self.device,
                registry=registry,
                alloc_trim_strategy=alloc_trim_strategy,
            )
            if temporal_upsampler_path
            else None
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
        self.duration_predictor = DurationPredictor.from_checkpoint(
            model_paths.duration_head_path,
            self.dtype,
            self.device,
        )

    def __call__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        temporal_upsample_rounds: int = 0,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio, int, TilingConfig | None]:
        if temporal_upsample_rounds not in (0, 1, 2):
            raise ValueError(f"temporal_upsample_rounds must be 0, 1, or 2, got {temporal_upsample_rounds}")
        if temporal_upsample_rounds > 0 and self.temporal_upsampler is None:
            raise ValueError("temporal_upsample_rounds > 0 requires temporal_upsampler_path")

        require_num_frames_source(num_frames, self.duration_predictor)
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        temporal_scale = VIDEO_SCALE_FACTORS.time

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
        requested_frames = num_frames
        num_frames, _segment, positions = resolve_canvas(num_frames)
        self.stage.assert_generated_keyframes_supported()

        # --- Stage 1: half-res base + keyframe slots -----------------------------------
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
            )
        )
        stage_1_conditionings.append(VideoGeneratedKeyframeSlots(pixel_frame_indices=positions))

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
        )

        reserved_half_res_video = video_state.latent[:1].detach().clone()
        stage_1_audio_latent = audio_state.latent.detach().clone() if audio_state is not None else None
        if video_state.generated_keyframes is None:
            raise RuntimeError("Stage 1 did not return generated_keyframes despite requesting slots")
        upsampled_slot_keyframes = self.upsampler(video_state.generated_keyframes)
        upscaled_video_latent = self.upsampler(reserved_half_res_video)

        # --- Stage 2: spatial detailing ------------------------------------------------
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )
        stage_2_conditionings.append(
            VideoGeneratedKeyframeSlots(pixel_frame_indices=positions, initial_keyframes=upsampled_slot_keyframes)
        )
        if self._detailing_lora:
            stage_2_conditionings.append(
                VideoConditionByReferenceLatent(
                    latent=reserved_half_res_video,
                    downscale_factor=self._detailing_downscale,
                    strength=1.0,
                )
            )

        video_state, audio_state = self.stage_detailing(
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

        # Stage 2's slots become the next round's anchors.
        carry_positions = list(positions)
        carry_keyframes = video_state.generated_keyframes
        current_fps = frame_rate
        temporal_sigmas = DISTILLED_SIGMAS[4:].to(dtype=torch.float32, device=self.device)

        for round_idx in range(1, temporal_upsample_rounds + 1):
            assert self.temporal_upsampler is not None
            if carry_keyframes is None or not carry_positions:
                raise RuntimeError(f"Temporal round {round_idx}: missing carry-forward keyframes")

            video_latent = self.temporal_upsampler(video_state.latent[:1])
            num_frames = 2 * (num_frames - 1) + 1
            current_fps = 2 * current_fps
            # Carried keyframes are single-frame latents, so only their positions scale with the round.
            seam_positions = [2 * position for position in carry_positions]
            anchor_keyframes = carry_keyframes
            seam_to_index = {seam: index for index, seam in enumerate(seam_positions)}
            cond_fps = min(current_fps, _MAX_CONDITIONING_FPS)
            tiles = tile_ranges(seam_positions, num_frames, 2**round_idx, temporal_scale=temporal_scale)

            tile_latents: list[torch.Tensor] = []
            slot_positions: list[int] = []
            slot_latent_slices: list[torch.Tensor] = []

            for tile_index, tile in enumerate(tiles):
                local_frames = (tile.latent_end_exclusive - tile.latent_start - 1) * temporal_scale + 1
                tile_video = video_latent[:, :, tile.latent_start : tile.latent_end_exclusive]

                # Image conditioning is tile-local: ``frame_idx=0`` means this tile's first frame, so
                # re-applying the opening image on a non-first tile would pin the wrong frame onto the
                # seam. Only images that actually fall inside the window are re-attached.
                if tile.pixel_start == 0:
                    tile_images = images
                else:
                    tile_images = [
                        ImageConditioningInput(
                            image.path, image.frame_idx - tile.pixel_start, image.strength, image.crf
                        )
                        for image in images
                        if tile.pixel_start <= image.frame_idx <= tile.pixel_end
                    ]
                round_conditionings = (
                    self.image_conditioner(
                        lambda enc, _images=tile_images: combined_image_conditionings(
                            images=_images,
                            height=height,
                            width=width,
                            video_encoder=enc,
                            dtype=dtype,
                            device=self.device,
                        )
                    )
                    if tile_images
                    else []
                )

                # Every seam in the window is a hard keyframe, including the one at local frame 0.
                anchor_global = list(tile.anchor_kf_global)
                if anchor_global:
                    missing = [position for position in anchor_global if position not in seam_to_index]
                    if missing:
                        raise RuntimeError(f"Anchor seams {missing} missing from the carry-forward bag")
                    anchor_latents = torch.cat(
                        [anchor_keyframes[:, :, seam_to_index[p] : seam_to_index[p] + 1] for p in anchor_global], dim=2
                    )
                    round_conditionings.extend(
                        _keyframe_conditionings_from_latents(
                            anchor_latents,
                            remap_positions_to_local(anchor_global, tile.pixel_start),
                            strength=_ANCHOR_KEYFRAME_STRENGTH,
                        )
                    )

                slot_global = list(tile.slot_kf_global)
                if slot_global:
                    slot_local = remap_positions_to_local(slot_global, tile.pixel_start)
                    round_conditionings.append(
                        VideoGeneratedKeyframeSlots(
                            pixel_frame_indices=slot_local,
                            initial_keyframes=_slot_initials_from_video(tile_video, slot_local, temporal_scale),
                        )
                    )

                tile_state, _ = self.stage(
                    denoiser=SimpleDenoiser(video_context, audio_context),
                    sigmas=temporal_sigmas,
                    noiser=noiser,
                    width=width,
                    height=height,
                    frames=local_frames,
                    fps=cond_fps,
                    video=ModalitySpec(
                        context=video_context,
                        conditionings=round_conditionings,
                        noise_scale=temporal_sigmas[0].item(),
                        initial_latent=tile_video,
                    ),
                    audio=None,
                    stepper=EulerAncestralDiffusionStep(eta=_TEMPORAL_ANCESTRAL_ETA),
                    # Tiles are positionally identical, so a shared ancestral seed would inject
                    # byte-identical noise into every one of them.
                    loop=partial(euler_ancestral_denoising_loop, noise_seed=seed + 1000 * round_idx + tile_index),
                )
                tile_latents.append(tile_state.latent[:1])

                if slot_global:
                    if tile_state.generated_keyframes is None:
                        raise RuntimeError(f"Temporal round {round_idx}: tile produced no keyframe slots")
                    slot_positions.extend(slot_global)
                    slot_latent_slices.append(tile_state.generated_keyframes)

            stitched = stitch_tile_latents(tile_latents, tiles)
            expected_t = (num_frames - 1) // temporal_scale + 1
            if stitched.shape[2] != expected_t:
                raise RuntimeError(f"Stitched latent T={stitched.shape[2]} != expected {expected_t}")
            if not isinstance(video_state, LatentState):
                raise TypeError(f"Expected LatentState, got {type(video_state)}")
            video_state = dataclass_replace(video_state, latent=stitched, generated_keyframes=None)

            slot_latents = torch.cat(slot_latent_slices, dim=2) if slot_latent_slices else None
            if slot_positions and slot_latents is not None:
                # Lead-in segments repeat the previous tile's slots; the earlier tile's version wins.
                first_index: dict[int, int] = {}
                for index, position in enumerate(slot_positions):
                    first_index.setdefault(position, index)
                slot_positions = sorted(first_index)
                slot_latents = torch.cat(
                    [slot_latents[:, :, first_index[p] : first_index[p] + 1] for p in slot_positions], dim=2
                )

            carry_positions, carry_keyframes = _merge_carry_forward_keyframes(
                seam_positions, anchor_keyframes, slot_positions, slot_latents
            )

        # The canvas may have padded its tail, and each round maps N -> 2(N-1)+1, so the caller's
        # contract is ``(requested - 1) * 2**rounds + 1``. ``requested - 1`` is a multiple of the VAE
        # temporal scale, so the trim always lands on a latent boundary.
        target_frames = (requested_frames - 1) * 2**temporal_upsample_rounds + 1
        if target_frames > num_frames:
            raise RuntimeError(f"Target {target_frames} frames exceeds the generated canvas {num_frames}")
        if target_frames != num_frames:
            keep_latents = (target_frames - 1) // temporal_scale + 1
            video_state = dataclass_replace(video_state, latent=video_state.latent[:, :, :keep_latents])
            num_frames = target_frames

        playback_fps = frame_rate * 2**temporal_upsample_rounds
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path),
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=playback_fps),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        if stage_1_audio_latent is None:
            raise RuntimeError("Stage 1 produced no audio latent to ship")
        decoded_audio = self.audio_decoder(stage_1_audio_latent)
        # Audio was generated for the padded canvas, so cut it to the video's duration or the muxed
        # container outlasts the picture.
        video_seconds = num_frames / playback_fps
        audio_samples = min(decoded_audio.waveform.shape[-1], round(video_seconds * decoded_audio.sampling_rate))
        if audio_samples != decoded_audio.waveform.shape[-1]:
            decoded_audio = dataclass_replace(decoded_audio, waveform=decoded_audio.waveform[..., :audio_samples])
        return decoded_video, decoded_audio, num_frames, tiling_config


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params()
    parser = default_2_stage_arg_parser(params=params, supports_auto_duration=True)
    parser.add_argument(
        "--detailing-lora",
        dest="detailing_lora",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="Stage-2 x2 spatial detailing IC-LoRA (path and optional strength). Omit to run without it.",
    )
    parser.add_argument(
        "--temporal-upsampler-path",
        type=resolve_existing_path,
        default=None,
        help="Path to the temporal x2 latent upsampler (required when --temporal-upsample-rounds > 0).",
    )
    parser.add_argument(
        "--temporal-upsample-rounds",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Number of temporal x2 refine rounds (0->base fps, 1->2x with 2 tiles, 2->4x with 4 tiles).",
    )
    args = parser.parse_args()

    pipeline = DFRPipeline(
        model_paths=args.model_paths,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        detailing_lora=args.detailing_lora,
        temporal_upsampler_path=args.temporal_upsampler_path,
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )
    video, audio, num_frames, tiling_config = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=AUTO_TILING,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        temporal_upsample_rounds=args.temporal_upsample_rounds,
    )

    encode_video(
        video=video,
        fps=int(args.frame_rate * (2**args.temporal_upsample_rounds)),
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
    )


if __name__ == "__main__":
    main()
