"""Self-contained validation runner for LTX-2 training.
Encapsulates all validation concerns: model loading, embedding caching,
conditioning media encoding, sample generation (denoising + decoding),
output saving, and W&B logging.
The trainer only needs to call ``run()`` with the current transformer state.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF  # noqa: N812
from torchvision.transforms.functional import to_tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.conditioning.types.mask_cond import VideoConditionByMask
from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.audio_vae.audio_vae import encode_audio as ltx_encode_audio
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import (
    Audio,
    AudioLatentShape,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)
from ltx_trainer import logger
from ltx_trainer.config import ValidationConfig, ValidationSample
from ltx_trainer.gpu_utils import free_gpu_memory_context
from ltx_trainer.model_loader import (
    load_audio_vae_decoder,
    load_audio_vae_encoder,
    load_embeddings_processor,
    load_text_encoder,
    load_video_vae_decoder,
    load_video_vae_encoder,
    load_vocoder,
    read_video_scale_factors,
)
from ltx_trainer.progress import SamplingContext, TrainingProgress
from ltx_trainer.utils import open_image_as_srgb, save_image
from ltx_trainer.video_utils import read_video, save_video

if TYPE_CHECKING:
    from ltx_core.model.transformer import LTXModel

# Shared by both decoder families. The floor is the diffusion decoder (LTX-2.4+), which rejects
# tiles below its per-stage minimum of 7 latent units spatially (224px) and 3 temporally; these
# are the smallest tiles above that which stay valid across the resolutions we validate at.
# Both decoders warn that these overlaps are below their recommended receptive-field halo, so tile
# edges can diverge from an untiled decode.
_DEFAULT_TILING = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=384, tile_overlap_in_pixels=192),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=48, tile_overlap_in_frames=24),
)


def _local_rank_device() -> torch.device:
    """Per-rank CUDA device for early init (before Accelerator exists).
    DDP-safe: ``LOCAL_RANK`` is set by accelerate before trainer init; loading on bare
    ``"cuda"`` would resolve to ``cuda:0`` on every rank and crash with a device mismatch.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(f"cuda:{local_rank}")


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class CachedPromptEmbeddings:
    """Pre-computed text embeddings for a validation prompt."""

    video_context_positive: Tensor
    audio_context_positive: Tensor
    video_context_negative: Tensor | None = None
    audio_context_negative: Tensor | None = None


@dataclass
class CachedConditionMedia:
    """Pre-encoded media for a single validation condition."""

    latent: Tensor
    pixels: Tensor | None = None
    mask: Tensor | None = None


@dataclass
class CachedSampleMedia:
    """Pre-encoded conditioning media for one validation sample.
    Keyed by condition index (position in the sample's conditions list).
    """

    conditions: dict[int, CachedConditionMedia] = field(default_factory=dict)


# =============================================================================
# ValidationRunner
# =============================================================================


class ValidationRunner:
    """Self-contained validation: loads models, caches media, generates samples.
    Lifecycle:
        1. ``__init__``: loads text encoder + embeddings processor, caches prompt
           embeddings, unloads both.  Loads VAE encoder, encodes conditioning media,
           unloads.  Loads VAE decoder / audio decoder / vocoder and keeps them on CPU.
        2. ``run()``: called at each validation step with the current transformer.
           Generates all samples, saves outputs, returns file paths.
    """

    def __init__(
        self,
        config: ValidationConfig,
        model_path: str | Path,
        text_encoder_path: str | Path | None,
        load_text_encoder_in_8bit: bool = False,
    ):
        self._config = config
        self._model_path = Path(model_path)
        # Video VAE compression factors, derived from the checkpoint's embedded VAE config
        # (defaults to 32x32x8 when absent), used for frame/resolution snapping and the
        # latent/pixel coordinate math below.
        self._video_scale_factors = read_video_scale_factors(self._model_path)
        # The config-level dim validators are VAE-agnostic (they can't know the compression
        # factors), so the geometry is checked here, once the factors are known.
        self._validate_sample_geometry()

        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

        self._cached_embeddings = self._cache_prompt_embeddings(text_encoder_path, load_text_encoder_in_8bit)
        self._cached_media = self._encode_conditioning_media()
        self._load_decoder_components()

    def _validate_sample_geometry(self) -> None:
        """Assert every validation sample's geometry is compatible with the VAE compression.
        This replaces the config-level dim validators, which run before the checkpoint (and
        therefore its scale factors) is known and so had to assume the default 32x32x8 layout.
        Width/height must be divisible by the spatial factors and frame counts must be
        VAE-aligned; otherwise the generated geometry would be silently cropped to fit.
        """
        sf = self._video_scale_factors
        for i, sample in enumerate(self._config.samples):
            width, height, frames = sample.video_dims or self._config.video_dims
            if width % sf.width != 0 or height % sf.height != 0:
                raise ValueError(
                    f"Validation sample {i}: dimensions {width}x{height} must be divisible by the VAE "
                    f"spatial factors ({sf.width}x{sf.height})."
                )
            if (frames - 1) % sf.time != 0:
                raise ValueError(
                    f"Validation sample {i}: frames ({frames}) must satisfy (frames - 1) % {sf.time} == 0 "
                    "for the VAE temporal factor."
                )
            for cond in sample.conditions:
                if getattr(cond, "video", None) is None:
                    continue  # audio prefix/suffix use duration, not num_frames
                num_frames = getattr(cond, "num_frames", None)
                if num_frames is None:
                    continue
                if cond.type == "prefix" and (num_frames - 1) % sf.time != 0:
                    raise ValueError(
                        f"Validation sample {i}: prefix num_frames ({num_frames}) must satisfy "
                        f"(num_frames - 1) % {sf.time} == 0 for the VAE temporal factor."
                    )
                if cond.type == "suffix" and num_frames % sf.time != 0:
                    raise ValueError(
                        f"Validation sample {i}: suffix num_frames ({num_frames}) must satisfy "
                        f"num_frames % {sf.time} == 0 for the VAE temporal factor."
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def run(
        self,
        transformer: "LTXModel",
        step: int,
        output_dir: Path,
        device: torch.device,
        progress: TrainingProgress,
        wandb_run: object | None = None,
        work_items: list[tuple[int, bool]] | None = None,
    ) -> list[tuple[int, Path]]:
        """Generate validation samples, save outputs, and optionally log to W&B.
        Args:
            work_items: Optional list of ``(sample_index, save_output)`` tuples for distributed
                validation. Each rank passes its assigned indices so prompts are split across
                ranks. ``save_output=False`` runs the generation but skips writing to disk; this
                is used for FSDP padding so every rank executes the same number of forwards and
                collective ops stay aligned. When ``None``, all samples are generated and saved.
        Returns:
            List of ``(sample_index, path)`` tuples for samples that were saved. Callers can
            ``gather_object`` and sort by index to reconstruct the global ordering.
        """
        samples = self._config.samples
        if not samples:
            return []

        if work_items is None:
            work_items = [(i, True) for i in range(len(samples))]
        if not work_items:
            return []

        inference_steps = self._config.inference_steps
        sampling_ctx = progress.start_sampling(num_prompts=len(work_items), num_steps=inference_steps)

        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True, parents=True)

        results: list[tuple[int, Path]] = []

        for local_i, (sample_idx, save_output) in enumerate(work_items):
            sample = samples[sample_idx]
            sampling_ctx.start_video(local_i)

            cached_embeddings = self._cached_embeddings[sample_idx] if self._cached_embeddings else None
            cached_media = self._cached_media[sample_idx] if self._cached_media else CachedSampleMedia()

            video, audio = self._generate_sample(
                sample=sample,
                cached_embeddings=cached_embeddings,
                cached_media=cached_media,
                transformer=transformer,
                device=device,
                sampling_ctx=sampling_ctx,
            )

            if not save_output:
                continue

            dims = sample.video_dims or self._config.video_dims
            num_frames = dims[2]
            ext = "png" if num_frames == 1 else "mp4"
            out_path = samples_dir / f"step_{step:06d}_{sample_idx + 1}.{ext}"

            if video is not None:
                if num_frames == 1:
                    save_image(video, out_path)
                else:
                    save_video(
                        video_tensor=video,
                        output_path=out_path,
                        fps=self._config.frame_rate,
                        audio=audio,
                        audio_sample_rate=self._vocoder.output_sampling_rate if audio is not None else None,
                        video_format="CFHW",
                    )
                results.append((sample_idx, out_path))
            elif audio is not None:
                out_path = out_path.with_suffix(".wav")
                self._save_audio(audio, out_path)
                results.append((sample_idx, out_path))

        sampling_ctx.cleanup()

        rel_path = samples_dir.relative_to(output_dir)
        logger.info(f"🎥 Validation samples for step {step} saved in {rel_path}")

        if wandb_run is not None and results:
            self.log_to_wandb(wandb_run, [p for _, p in results], step)

        return results

    # --- Initialization: caching ---

    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _cache_prompt_embeddings(
        self,
        text_encoder_path: str | Path | None,
        load_in_8bit: bool,
    ) -> list[CachedPromptEmbeddings]:
        """Load text encoder, encode all validation prompts, cache on CPU, then unload."""
        prompts = [s.prompt for s in self._config.samples]
        if not prompts:
            return []

        init_device = _local_rank_device()

        logger.debug("Loading text encoder for validation embedding caching...")
        text_encoder = load_text_encoder(
            gemma_model_path=text_encoder_path, device=init_device, dtype=torch.bfloat16, load_in_8bit=load_in_8bit
        )

        logger.debug("Loading embeddings processor for validation embedding caching...")
        embeddings_processor = load_embeddings_processor(
            checkpoint_path=self._model_path,
            gemma_model_path=text_encoder_path,
            device=init_device,
            dtype=torch.bfloat16,
        )

        logger.info(f"Pre-computing embeddings for {len(prompts)} validation prompts...")
        cached: list[CachedPromptEmbeddings] = []
        neg_hs, neg_mask = text_encoder.encode([self._config.negative_prompt])[0]
        neg_out = embeddings_processor.process_hidden_states(neg_hs, neg_mask)

        for prompt in prompts:
            pos_hs, pos_mask = text_encoder.encode([prompt])[0]
            pos_out = embeddings_processor.process_hidden_states(pos_hs, pos_mask)

            cached.append(
                CachedPromptEmbeddings(
                    video_context_positive=pos_out.video_encoding.cpu(),
                    audio_context_positive=pos_out.audio_encoding.cpu(),
                    video_context_negative=neg_out.video_encoding.cpu(),
                    audio_context_negative=(
                        neg_out.audio_encoding.cpu() if neg_out.audio_encoding is not None else None
                    ),
                )
            )

        del text_encoder, embeddings_processor
        logger.debug("Validation prompt embeddings cached. Text encoder unloaded.")
        return cached

    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _encode_conditioning_media(self) -> list[CachedSampleMedia]:
        """Load VAE encoders, encode all conditioning media, cache on CPU, then unload."""
        samples = self._config.samples
        if not samples:
            return []

        def _cond_needs_video_encoder(cond: object) -> bool:
            if getattr(cond, "type", None) in ("first_frame", "video_to_audio", "spatial_crop"):
                return True
            return getattr(cond, "video", None) is not None

        def _cond_needs_audio_encoder(cond: object) -> bool:
            if getattr(cond, "type", None) == "audio_to_video":
                return True
            return getattr(cond, "type", None) in ("prefix", "suffix", "mask", "reference") and (
                getattr(cond, "audio", None) is not None
            )

        needs_video_encoder = any(_cond_needs_video_encoder(c) for s in samples for c in s.conditions)
        needs_audio_encoder = any(_cond_needs_audio_encoder(c) for s in samples for c in s.conditions)

        if not needs_video_encoder and not needs_audio_encoder:
            return [CachedSampleMedia() for _ in samples]

        device = _local_rank_device()
        vae_encoder = None
        audio_encoder = None

        if needs_video_encoder:
            logger.debug("Loading VAE encoder for validation media encoding...")
            vae_encoder = load_video_vae_encoder(self._model_path, device="cpu", dtype=torch.bfloat16)
        if needs_audio_encoder:
            logger.debug("Loading audio VAE encoder for validation media encoding...")
            audio_encoder = load_audio_vae_encoder(self._model_path, device="cpu", dtype=torch.bfloat16)

        logger.info(f"Pre-encoding conditioning media for {len(samples)} validation samples...")
        cached: list[CachedSampleMedia] = []

        for sample in samples:
            sample_dims = sample.video_dims or self._config.video_dims
            cached.append(self._encode_sample_conditions(sample, sample_dims, vae_encoder, audio_encoder, device))

        del vae_encoder, audio_encoder
        logger.info("Validation conditioning media cached on CPU.")
        return cached

    def _encode_sample_conditions(
        self,
        sample: ValidationSample,
        dims: tuple[int, int, int],
        vae_encoder: torch.nn.Module | None,
        audio_encoder: torch.nn.Module | None,
        device: torch.device,
    ) -> CachedSampleMedia:
        """Encode all conditioning media for a single validation sample."""
        s_width, s_height, s_num_frames = dims
        sample_media = CachedSampleMedia()

        for cond_idx, cond in enumerate(sample.conditions):
            if cond.type == "first_frame":
                image = self._load_first_frame(Path(cond.image_or_video))
                latent = self._encode_image(image, s_height, s_width, vae_encoder, device)
                sample_media.conditions[cond_idx] = CachedConditionMedia(latent=latent)

            elif cond.type == "reference" and cond.video is not None:
                ref_video, _ = read_video(cond.video, max_frames=s_num_frames)
                preprocessed, pixels = self._preprocess_reference(
                    ref_video,
                    s_height,
                    s_width,
                    self._video_scale_factors,
                    cond.downscale_factor,
                    cond.temporal_scale_factor,
                )
                latent = self._encode_video(preprocessed, vae_encoder, device)
                sample_media.conditions[cond_idx] = CachedConditionMedia(
                    latent=latent, pixels=pixels if cond.include_in_output else None
                )

            elif cond.type == "reference" and getattr(cond, "audio", None) is not None:
                video_duration = s_num_frames / self._config.frame_rate
                latent = self._encode_audio(cond.audio, audio_encoder, device, max_duration=video_duration)
                sample_media.conditions[cond_idx] = CachedConditionMedia(latent=latent)

            elif cond.type in ("prefix", "suffix", "mask", "spatial_crop", "video_to_audio"):
                latent = self._encode_temporal_condition_media(
                    cond=cond,
                    target_width=s_width,
                    target_height=s_height,
                    max_frames=s_num_frames,
                    vae_encoder=vae_encoder,
                    audio_encoder=audio_encoder,
                    device=device,
                )
                mask_tensor = None
                if latent is not None and cond.type == "mask" and hasattr(cond, "mask") and cond.mask is not None:
                    if getattr(cond, "audio", None) is not None:
                        mask_tensor = self._load_audio_mask(cond.mask, s_num_frames, self._config.frame_rate)
                    else:
                        mask_tensor = self._load_and_downsample_mask(cond.mask, s_width, s_height, s_num_frames)
                if latent is not None:
                    sample_media.conditions[cond_idx] = CachedConditionMedia(latent=latent, mask=mask_tensor)

            elif cond.type == "audio_to_video":
                video_duration = s_num_frames / self._config.frame_rate
                latent = self._encode_audio(cond.audio, audio_encoder, device, max_duration=video_duration)
                sample_media.conditions[cond_idx] = CachedConditionMedia(latent=latent)

        return sample_media

    def _load_decoder_components(self) -> None:
        """Load VAE decoder, audio decoder, and vocoder. Kept on CPU until generation."""
        self._vae_decoder = None
        needs_video_decoder = self._config.generate_video or any(
            c.type == "video_to_audio" for s in self._config.samples for c in s.conditions
        )
        if needs_video_decoder:
            logger.debug("Loading video VAE decoder for validation...")
            self._vae_decoder = load_video_vae_decoder(self._model_path, device="cpu", dtype=torch.bfloat16)
            if self._vae_decoder is not None:
                self._vae_decoder.requires_grad_(False)

        self._audio_decoder = None
        self._vocoder = None
        needs_audio_decoder = self._config.generate_audio or any(
            c.type == "audio_to_video" for s in self._config.samples for c in s.conditions
        )
        if needs_audio_decoder:
            logger.debug("Loading audio decoder and vocoder for validation...")
            self._audio_decoder = load_audio_vae_decoder(self._model_path, device="cpu", dtype=torch.bfloat16)
            if self._audio_decoder is not None:
                self._audio_decoder.requires_grad_(False)
            self._vocoder = load_vocoder(self._model_path, device="cpu", dtype=torch.bfloat16)
            if self._vocoder is not None:
                self._vocoder.requires_grad_(False)

    def _encode_temporal_condition_media(
        self,
        cond: object,
        *,
        target_width: int,
        target_height: int,
        max_frames: int,
        vae_encoder: torch.nn.Module | None,
        audio_encoder: torch.nn.Module | None,
        device: torch.device,
    ) -> Tensor | None:
        """Encode media for prefix/suffix/mask/spatial_crop/frozen-video conditions."""
        video_path = getattr(cond, "video", None)
        audio_path = getattr(cond, "audio", None)
        if video_path is None and audio_path is not None:
            max_dur = getattr(cond, "duration", None)
            if max_dur is None and cond.type == "mask":
                max_dur = max_frames / self._config.frame_rate
            return self._encode_audio(
                audio_path,
                audio_encoder,
                device,
                max_duration=max_dur,
                from_end=cond.type == "suffix",
            )
        if video_path is None:
            return None

        video, _ = read_video(video_path, max_frames=max_frames)
        video = self._resize_and_center_crop(video, target_height, target_width)
        requested_frames = getattr(cond, "num_frames", None)

        temporal_factor = self._video_scale_factors.time
        if cond.type == "suffix" and requested_frames is not None:
            # The VAE encoder has temporal receptive fields, so encoding a short
            # suffix clip independently produces different latents than encoding the
            # full video. Encode the full clip and extract the last N latent frames.
            video = rearrange(video, "f c h w -> 1 c f h w")
            valid_frames = (video.shape[2] - 1) // temporal_factor * temporal_factor + 1
            video = video[:, :, :valid_frames]
            latent = self._encode_video(video * 2.0 - 1.0, vae_encoder, device)
            num_suffix_latent_frames = requested_frames // temporal_factor
            return latent[:, :, -num_suffix_latent_frames:]

        if requested_frames is not None:
            requested_frames = min(requested_frames, video.shape[0])
            video = video[:requested_frames]
        video = rearrange(video, "f c h w -> 1 c f h w")
        valid_frames = (video.shape[2] - 1) // temporal_factor * temporal_factor + 1
        video = video[:, :, :valid_frames]
        return self._encode_video(video * 2.0 - 1.0, vae_encoder, device)

    # ------------------------------------------------------------------
    # Unified generation pipeline
    # ------------------------------------------------------------------

    def _generate_sample(
        self,
        sample: ValidationSample,
        cached_embeddings: CachedPromptEmbeddings | None,
        cached_media: CachedSampleMedia,
        transformer: "LTXModel",
        device: torch.device,
        sampling_ctx: SamplingContext,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Generate one sample: build conditioned states, denoise, decode."""
        dims = sample.video_dims or self._config.video_dims
        width, height, num_frames = dims
        seed = sample.seed or self._config.seed
        generate_audio = self._config.generate_audio
        generate_video = self._config.generate_video

        # Determine frozen modalities from conditions
        video_frozen = any(c.type == "video_to_audio" for c in sample.conditions)
        audio_frozen = any(c.type == "audio_to_video" for c in sample.conditions)

        # 1. Prompt embeddings
        v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = self._get_prompt_embeddings(cached_embeddings, device)

        # 2. Build video state
        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        video_state: LatentState | None = None
        video_clean: LatentState | None = None
        video_tools: VideoLatentTools | None = None

        if generate_video or video_frozen:
            video_tools = self._create_video_tools(width, height, num_frames)

            if video_frozen:
                frozen_media = self._find_condition_media(sample, cached_media, "video_to_audio")
                video_state = video_tools.create_initial_state(
                    device=device, dtype=torch.bfloat16, initial_latent=frozen_media.latent.to(device)
                )
                video_state = replace(video_state, denoise_mask=torch.zeros_like(video_state.denoise_mask))
                video_clean = video_state
            else:
                video_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
                video_state = self._apply_video_conditionings(video_state, video_tools, sample, cached_media, device)
                video_clean = video_state
                video_state = noiser(video_state, noise_scale=1.0)

        # 3. Build audio state
        audio_state: LatentState | None = None
        audio_clean: LatentState | None = None
        audio_tools: AudioLatentTools | None = None

        if generate_audio or audio_frozen:
            if audio_frozen:
                frozen_media = self._find_condition_media(sample, cached_media, "audio_to_video")
                frozen_latent = frozen_media.latent.to(device)
                audio_tools = AudioLatentTools(
                    patchifier=self._audio_patchifier,
                    target_shape=AudioLatentShape.from_torch_shape(frozen_latent.shape),
                )
                audio_state = audio_tools.create_initial_state(
                    device=device, dtype=torch.bfloat16, initial_latent=frozen_latent
                )
                audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))
                audio_clean = audio_state
            else:
                audio_tools = self._create_audio_tools(num_frames, self._config.frame_rate)
                audio_state = audio_tools.create_initial_state(device=device, dtype=torch.bfloat16)
                audio_state = self._apply_audio_conditionings(audio_state, audio_tools, sample, cached_media, device)
                audio_clean = audio_state
                audio_state = noiser(audio_state, noise_scale=1.0)

        # 4. Denoising loop
        video_state, audio_state = self._run_denoising(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            video_clean=video_clean,
            audio_clean=audio_clean,
            video_frozen=video_frozen,
            audio_frozen=audio_frozen,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            device=device,
            sampling_ctx=sampling_ctx,
        )

        # 5. Decode modalities (both generated and frozen — frozen audio/video is included in output)
        video_output = self._finalize_modality(video_state, video_tools, self._decode_video, device)
        audio_output = self._finalize_modality(audio_state, audio_tools, self._decode_audio, device)

        # 6. Side-by-side reference output
        if video_output is not None:
            video_output = self._apply_reference_side_by_side(video_output, sample, cached_media)

        return video_output, audio_output

    # ------------------------------------------------------------------
    # Conditioning application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_video_conditionings(
        state: LatentState,
        tools: VideoLatentTools,
        sample: ValidationSample,
        cached_media: CachedSampleMedia,
        device: torch.device,
    ) -> LatentState:
        """Apply all video-targeting conditionings from the sample's conditions list."""
        for cond_idx, cond in enumerate(sample.conditions):
            media = cached_media.conditions.get(cond_idx)
            if media is None:
                continue

            latent = media.latent.to(device=device, dtype=torch.bfloat16)

            if cond.type == "first_frame" or (cond.type == "prefix" and getattr(cond, "video", None) is not None):
                state = VideoConditionByLatentIndex(latent=latent, strength=1.0, latent_idx=0).apply_to(state, tools)

            elif cond.type == "suffix" and getattr(cond, "video", None) is not None:
                _, _, tgt_frames, _, _ = tools.target_shape.to_torch_shape()
                _, _, cond_frames, _, _ = latent.shape
                suffix_start = tgt_frames - cond_frames
                state = VideoConditionByLatentIndex(latent=latent, strength=1.0, latent_idx=suffix_start).apply_to(
                    state, tools
                )

            elif cond.type == "reference" and cond.video is not None:
                state = VideoConditionByReferenceLatent(
                    latent=latent,
                    downscale_factor=cond.downscale_factor,
                    temporal_scale_factor=cond.temporal_scale_factor,
                    strength=1.0,
                ).apply_to(state, tools)

            elif cond.type == "mask":
                mask = media.mask.to(device=device)
                state = VideoConditionByMask(latent=latent, mask=mask, strength=1.0).apply_to(state, tools)

            elif cond.type == "spatial_crop":
                mask = _build_spatial_crop_mask(cond.spatial_region, tools.target_shape, device, tools.scale_factors)
                state = VideoConditionByMask(latent=latent, mask=mask, strength=1.0).apply_to(state, tools)

        return state

    @staticmethod
    def _apply_audio_conditionings(
        state: LatentState,
        tools: AudioLatentTools,
        sample: ValidationSample,
        cached_media: CachedSampleMedia,
        device: torch.device,
    ) -> LatentState:
        """Apply all audio-targeting conditionings from the sample's conditions list."""
        for cond_idx, cond in enumerate(sample.conditions):
            media = cached_media.conditions.get(cond_idx)
            if media is None:
                continue

            latent = media.latent.to(device=device, dtype=torch.bfloat16)

            if cond.type in ("prefix", "suffix") and getattr(cond, "audio", None) is not None:
                tokens = tools.patchifier.patchify(latent)
                num_cond_tokens = tokens.shape[1]

                state = state.clone()
                start = 0 if cond.type == "prefix" else tools.target_shape.token_count() - num_cond_tokens
                stop = start + num_cond_tokens

                state.clean_latent[:, start:stop] = tokens
                state.denoise_mask[:, start:stop] = 0.0

            elif cond.type == "reference" and getattr(cond, "audio", None) is not None:
                tokens = tools.patchifier.patchify(latent)
                ref_shape = AudioLatentShape.from_torch_shape(latent.shape)

                positions = tools.patchifier.get_patch_grid_bounds(
                    output_shape=ref_shape,
                    device=device,
                )

                denoise_mask = torch.zeros(
                    *tokens.shape[:2],
                    1,
                    device=device,
                    dtype=torch.float32,
                )

                state = LatentState(
                    latent=torch.cat([state.latent, torch.zeros_like(tokens, dtype=state.latent.dtype)], dim=1),
                    denoise_mask=torch.cat([state.denoise_mask, denoise_mask], dim=1),
                    positions=torch.cat([state.positions, positions], dim=2),
                    clean_latent=torch.cat([state.clean_latent, tokens], dim=1),
                    attention_mask=None,
                )

            elif cond.type == "mask" and getattr(cond, "audio", None) is not None:
                tokens = tools.patchifier.patchify(latent)
                target_len = state.latent.shape[1]
                tokens = tokens[:, :target_len]

                mask = media.mask.to(device=device)
                if mask.shape[-1] != target_len:
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).float(), size=target_len, mode="nearest"
                    ).squeeze(0)
                    mask = (mask > 0.5).float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)

                m = mask.to(dtype=state.latent.dtype)
                inv = 1 - m

                state = LatentState(
                    latent=state.latent,
                    denoise_mask=state.denoise_mask * inv,
                    positions=state.positions,
                    clean_latent=state.clean_latent * inv + tokens * m,
                    attention_mask=state.attention_mask,
                )

        return state

    @staticmethod
    def _find_condition_media(
        sample: ValidationSample, cached_media: CachedSampleMedia, condition_type: str
    ) -> CachedConditionMedia:
        """Look up the cached media for the first condition matching the given type."""
        for cond_idx, cond in enumerate(sample.conditions):
            if cond.type == condition_type and cond_idx in cached_media.conditions:
                return cached_media.conditions[cond_idx]
        raise ValueError(f"No cached media found for condition type '{condition_type}'")

    @staticmethod
    def _apply_reference_side_by_side(
        video_output: Tensor, sample: ValidationSample, cached_media: CachedSampleMedia
    ) -> Tensor:
        """Concatenate reference video pixels side-by-side with generated output if requested."""
        for cond_idx, cond in enumerate(sample.conditions):
            if cond.type == "reference" and cond.include_in_output:
                media = cached_media.conditions.get(cond_idx)
                if media is not None and media.pixels is not None:
                    ref_pixels = media.pixels
                    output_frames = video_output.shape[1]
                    ref_frames = ref_pixels.shape[1]
                    if ref_frames < output_frames:
                        temporal_sf = cond.temporal_scale_factor
                        if temporal_sf > 1:
                            # VAE-aligned stretch mirroring training-time subsampling:
                            # first frame preserved once (aligns with VAE's causal
                            # first latent), each subsequent ref frame held for
                            # `temporal_sf` output frames.
                            indices = torch.tensor([0] + [1 + (i - 1) // temporal_sf for i in range(1, output_frames)])
                            indices = indices.clamp(max=ref_frames - 1)
                        else:
                            indices = torch.linspace(0, ref_frames - 1, output_frames).round().long()
                        ref_pixels = ref_pixels[:, indices]
                    video_output = _concatenate_videos_side_by_side(ref_pixels, video_output)
        return video_output

    # ------------------------------------------------------------------
    # Denoising loop
    # ------------------------------------------------------------------

    def _run_denoising(  # noqa: PLR0913
        self,
        transformer: "LTXModel",
        video_state: LatentState | None,
        audio_state: LatentState | None,
        video_clean: LatentState | None,
        audio_clean: LatentState | None,
        *,
        video_frozen: bool,
        audio_frozen: bool,
        v_ctx_pos: Tensor,
        a_ctx_pos: Tensor,
        v_ctx_neg: Tensor | None,
        a_ctx_neg: Tensor | None,
        device: torch.device,
        sampling_ctx: SamplingContext,
    ) -> tuple[LatentState | None, LatentState | None]:
        """Run the Euler denoising loop with CFG/STG, handling frozen modalities."""
        cfg = self._config
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=cfg.inference_steps).to(device).float()
        stepper = EulerDiffusionStep()
        video_present = video_state is not None
        audio_present = audio_state is not None
        video_guider = MultiModalGuider(
            self._resolve_video_guider_params(
                cfg,
                video_frozen=video_frozen,
                audio_present=audio_present,
            )
        )
        audio_guider = MultiModalGuider(
            self._resolve_audio_guider_params(
                cfg,
                audio_frozen=audio_frozen,
                video_present=video_present,
            )
        )
        transformer_dtype = next(transformer.parameters()).dtype

        x0_model = X0Model(transformer)

        for step_idx, sigma in enumerate(sigmas[:-1]):
            v_sigma = torch.zeros_like(sigma) if video_frozen else sigma
            a_sigma = torch.zeros_like(sigma) if audio_frozen else sigma

            video = (
                self._modality_from_latent_state(video_state, v_ctx_pos, v_sigma.unsqueeze(0))
                if video_state is not None
                else None
            )
            audio = (
                self._modality_from_latent_state(audio_state, a_ctx_pos, a_sigma.unsqueeze(0))
                if audio_state is not None
                else None
            )

            pos_video, pos_audio = x0_model(video=video, audio=audio, perturbations=None)
            neg_video, neg_audio = 0.0, 0.0
            ptb_video, ptb_audio = 0.0, 0.0
            mod_video, mod_audio = 0.0, 0.0

            # CFG
            video_needs_neg = video is not None and video_guider.do_unconditional_generation()
            audio_needs_neg = audio is not None and audio_guider.do_unconditional_generation()
            if video_needs_neg or audio_needs_neg:
                video_neg_ctx = v_ctx_neg if video_needs_neg and v_ctx_neg is not None else v_ctx_pos
                audio_neg_ctx = a_ctx_neg if audio_needs_neg and a_ctx_neg is not None else a_ctx_pos
                video_neg = replace(video, context=video_neg_ctx) if video is not None else None
                audio_neg = replace(audio, context=audio_neg_ctx) if audio is not None else None
                neg_video, neg_audio = x0_model(video=video_neg, audio=audio_neg, perturbations=None)

            # STG
            stg_perturbation_config = self._build_stg_perturbation_config(
                video_enabled=video is not None and video_guider.do_perturbed_generation(),
                audio_enabled=audio is not None and audio_guider.do_perturbed_generation(),
                stg_blocks=cfg.stg_blocks,
                num_blocks=transformer.num_blocks,
                device=device,
                dtype=transformer_dtype,
            )
            if stg_perturbation_config is not None:
                ptb_video, ptb_audio = x0_model(video=video, audio=audio, perturbations=stg_perturbation_config)

            # Modality guidance
            if (video is not None and video_guider.do_isolated_modality_generation()) or (
                audio is not None and audio_guider.do_isolated_modality_generation()
            ):
                mod_perturbation_config = self._build_modality_perturbation_config(
                    transformer.num_blocks, device, transformer_dtype
                )
                mod_video, mod_audio = x0_model(video=video, audio=audio, perturbations=mod_perturbation_config)

            denoised_video, denoised_audio = pos_video, pos_audio
            if not video_frozen and pos_video is not None:
                denoised_video = video_guider.calculate(pos_video, neg_video, ptb_video, mod_video)
            if not audio_frozen and pos_audio is not None:
                denoised_audio = audio_guider.calculate(pos_audio, neg_audio, ptb_audio, mod_audio)

            # Re-apply conditioning mask
            if denoised_video is not None and video_clean is not None:
                denoised_video = self._post_process_latent(
                    denoised_video,
                    video_state.denoise_mask,
                    video_clean.clean_latent,
                )
            if denoised_audio is not None and audio_clean is not None:
                denoised_audio = self._post_process_latent(
                    denoised_audio,
                    audio_state.denoise_mask,
                    audio_clean.clean_latent,
                )

            # Euler step (skip for frozen modalities)
            if video_state is not None and not video_frozen:
                video_state = replace(
                    video_state,
                    latent=stepper.step(video.latent, denoised_video, sigmas, step_idx),
                )
            if audio_state is not None and not audio_frozen:
                audio_state = replace(
                    audio_state,
                    latent=stepper.step(audio.latent, denoised_audio, sigmas, step_idx),
                )

            sampling_ctx.advance_step()

        return video_state, audio_state

    # --- Finalization (decode) ---

    def _finalize_modality(
        self,
        state: LatentState | None,
        tools: VideoLatentTools | AudioLatentTools | None,
        decode_fn: Callable[[LatentState, torch.device], Tensor],
        device: torch.device,
    ) -> Tensor | None:
        """Clear conditioning tokens, unpatchify, and decode a modality (generated or frozen)."""
        if state is None or tools is None:
            return None
        state = tools.clear_conditioning(state)
        state = tools.unpatchify(state)
        return decode_fn(state, device)

    def _decode_video(self, video_state: LatentState, device: torch.device) -> Tensor:
        """Decode video latents to pixels using tiled VAE decoding."""
        self._vae_decoder.to(device)
        latent = video_state.latent.to(dtype=torch.bfloat16)

        chunks = list(self._vae_decoder.tiled_decode(latent, tiling_config=_DEFAULT_TILING))
        decoded_video = torch.cat(chunks, dim=2)

        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        self._vae_decoder.to("cpu")
        return decoded_video[0].float().cpu()

    def _decode_audio(self, audio_state: LatentState, device: torch.device) -> Tensor:
        """Decode audio latents to waveform via audio VAE + vocoder."""
        self._audio_decoder.to(device)
        first_param = next(self._audio_decoder.parameters(), None)
        decoder_dtype = first_param.dtype if first_param is not None else audio_state.latent.dtype
        latent = audio_state.latent.to(dtype=decoder_dtype, device=device)
        decoded_audio = self._audio_decoder(latent)
        self._audio_decoder.to("cpu")

        self._vocoder.to(device)
        audio_waveform = self._vocoder(decoded_audio)
        self._vocoder.to("cpu")

        return audio_waveform.squeeze(0).float().cpu()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_audio(self, audio: Tensor, path: Path) -> None:
        """Save an audio waveform tensor to a .wav file using stdlib wave."""
        import wave  # noqa: PLC0415

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        pcm = (audio.clamp(-1.0, 1.0).cpu() * 32767).to(torch.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(pcm.shape[0])
            wf.setsampwidth(2)
            wf.setframerate(self._vocoder.output_sampling_rate)
            wf.writeframes(pcm.T.contiguous().numpy().tobytes())

    @staticmethod
    def _encode_audio(
        audio_path: str,
        audio_encoder: torch.nn.Module,
        device: torch.device,
        max_duration: float | None = None,
        *,
        from_end: bool = False,
    ) -> Tensor:
        """Encode an audio file through the audio VAE encoder. Returns latent on CPU."""
        import torchaudio  # noqa: PLC0415

        waveform, sr = torchaudio.load(audio_path)
        if max_duration is not None:
            num_samples = min(round(max_duration * sr), waveform.shape[-1])
            waveform = waveform[:, -num_samples:] if from_end else waveform[:, :num_samples]
        audio = Audio(waveform=waveform.unsqueeze(0), sampling_rate=sr)
        audio_encoder.to(device)
        latent = ltx_encode_audio(audio, audio_encoder)
        audio_encoder.to("cpu")
        return latent.cpu()

    def _load_and_downsample_mask(
        self,
        mask_path: str | Path,
        target_width: int,
        target_height: int,
        target_num_frames: int,
    ) -> Tensor:
        """Load a mask image/video and downsample to latent-space dimensions.
        Returns a binary float tensor of shape [1, F', H', W'] where F', H', W' are
        the latent-space dimensions corresponding to the target video dims.
        """
        sf = self._video_scale_factors
        latent_f = (target_num_frames - 1) // sf.time + 1

        mask_path = Path(mask_path)
        if mask_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            img = to_tensor(open_image_as_srgb(mask_path)).mean(dim=0, keepdim=True)  # [1, H, W]
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(target_height, target_width), mode="nearest"
            ).squeeze(0)  # [1, H, W]
            mask_pixels = img.expand(target_num_frames, -1, -1)  # [F, H, W]
        else:
            frames, _ = read_video(str(mask_path), max_frames=target_num_frames)  # [F, C, H, W]
            frames = frames[:target_num_frames].mean(dim=1)  # grayscale [F, H, W]
            mask_pixels = torch.nn.functional.interpolate(
                frames.unsqueeze(1), size=(target_height, target_width), mode="nearest"
            ).squeeze(1)  # [F, H, W]

        mask_latent = torch.nn.functional.avg_pool2d(
            mask_pixels.unsqueeze(1), kernel_size=(sf.height, sf.width)
        ).squeeze(1)  # [F, H', W']

        # Temporal: max-pool over groups of sf.time frames (conservative — any masked frame masks the group)
        f_spatial = mask_latent.shape[0]
        pad_f = (sf.time - f_spatial % sf.time) % sf.time
        if pad_f > 0:
            mask_latent = torch.nn.functional.pad(mask_latent, (0, 0, 0, 0, 0, pad_f))
        h_prime, w_prime = mask_latent.shape[1], mask_latent.shape[2]
        mask_latent = mask_latent.reshape(-1, sf.time, h_prime, w_prime).amax(dim=1)[:latent_f]

        mask_latent = (mask_latent > 0.5).float()
        return mask_latent.unsqueeze(0)  # [1, F', H', W'] — unpatchified latent space

    @staticmethod
    def _load_audio_mask(
        mask_path: str | Path,
        target_num_frames: int,
        frame_rate: float,
    ) -> Tensor:
        """Load an audio mask (.pt or .wav) and resample to target audio latent length.
        Returns a float tensor of shape [1, num_tokens] in patchified token space.
        """
        mask_path = Path(mask_path)
        if mask_path.suffix == ".pt":
            raw = torch.load(mask_path, map_location="cpu", weights_only=True)
            if isinstance(raw, dict):
                raw = raw.get("mask", next(iter(raw.values())))
            raw = raw.float().flatten()
        else:
            import torchaudio  # noqa: PLC0415

            waveform, _sr = torchaudio.load(mask_path)
            raw = waveform.abs().mean(dim=0)

        target_duration = target_num_frames / frame_rate
        target_shape = AudioLatentShape.from_duration(batch=1, duration=target_duration)
        target_len = target_shape.frames

        resampled = torch.nn.functional.interpolate(
            raw.unsqueeze(0).unsqueeze(0), size=target_len, mode="nearest"
        ).squeeze()
        mask = (resampled > 0.5).float()
        return mask.unsqueeze(0)  # [1, num_tokens]

    def _create_video_tools(self, width: int, height: int, num_frames: int) -> VideoLatentTools:
        """Create VideoLatentTools for the given output dimensions."""
        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=height, width=width, fps=self._config.frame_rate
        )
        return VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(shape=pixel_shape, scale_factors=self._video_scale_factors),
            fps=self._config.frame_rate,
            scale_factors=self._video_scale_factors,
            causal_fix=True,
        )

    def _create_audio_tools(self, num_frames: int, frame_rate: float) -> AudioLatentTools:
        """Create AudioLatentTools for the given video duration."""
        return AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate),
        )

    @staticmethod
    def _get_prompt_embeddings(
        cached: CachedPromptEmbeddings | None, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Move cached prompt embeddings to device, returning (v_pos, a_pos, v_neg, a_neg)."""
        if cached is None:
            raise ValueError("Cached prompt embeddings are required for validation generation")
        return (
            cached.video_context_positive.to(device),
            cached.audio_context_positive.to(device),
            cached.video_context_negative.to(device) if cached.video_context_negative is not None else None,
            cached.audio_context_negative.to(device) if cached.audio_context_negative is not None else None,
        )

    @staticmethod
    def log_to_wandb(wandb_run: object, sample_paths: list[Path], step: int) -> None:
        """Log validation outputs (images, videos, or audio) to Weights & Biases."""
        import wandb  # noqa: PLC0415

        suffix = sample_paths[0].suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".heic", ".webp"):
            media = [wandb.Image(str(path)) for path in sample_paths]
        elif suffix in (".wav", ".mp3", ".flac", ".ogg"):
            media = [wandb.Audio(str(path)) for path in sample_paths]
        else:
            media = [wandb.Video(str(path), format=suffix.lstrip(".")) for path in sample_paths]
        wandb_run.log({"validation_samples": media}, step=step)

    @staticmethod
    def _post_process_latent(denoised: Tensor, denoise_mask: Tensor, clean_latent: Tensor) -> Tensor:
        """Blend denoised output with clean latent according to the denoise mask."""
        return (denoised * denoise_mask + clean_latent.float() * (1 - denoise_mask)).to(denoised.dtype)

    @staticmethod
    def _resolve_video_guider_params(
        cfg: ValidationConfig,
        *,
        video_frozen: bool,
        audio_present: bool,
    ) -> MultiModalGuiderParams:
        """Resolve video guider params for the current validation sample.
        Frozen conditioning video (V2A) and video-only runs without an audio branch use a
        neutral guider, matching ``a2vid_two_stage`` / ``t2a_one_stage`` pipeline call-site patterns.
        """
        if video_frozen:
            return MultiModalGuiderParams()
        modality_scale = cfg.video_modality_guidance_scale if audio_present else 1.0
        return MultiModalGuiderParams(
            cfg_scale=cfg.video_cfg_scale,
            stg_scale=cfg.video_stg_scale,
            stg_blocks=cfg.stg_blocks,
            rescale_scale=cfg.guidance_rescale,
            modality_scale=modality_scale,
        )

    @staticmethod
    def _resolve_audio_guider_params(
        cfg: ValidationConfig,
        *,
        audio_frozen: bool,
        video_present: bool,
    ) -> MultiModalGuiderParams:
        """Resolve audio guider params for the current validation sample.
        Frozen conditioning audio (A2V) uses a neutral guider, matching ``a2vid_two_stage``.
        Audio-only runs without a video branch disable cross-modal guidance (``t2a_one_stage``).
        """
        if audio_frozen:
            return MultiModalGuiderParams()
        modality_scale = cfg.audio_modality_guidance_scale if video_present else 1.0
        return MultiModalGuiderParams(
            cfg_scale=cfg.audio_cfg_scale,
            stg_scale=cfg.audio_stg_scale,
            stg_blocks=cfg.stg_blocks,
            rescale_scale=cfg.guidance_rescale,
            modality_scale=modality_scale,
        )

    @staticmethod
    def _modality_from_latent_state(state: LatentState, context: Tensor, sigma: Tensor) -> Modality:
        """Build a Modality object from a LatentState, text context, and sigma."""
        return Modality(
            enabled=True,
            latent=state.latent,
            sigma=sigma,
            timesteps=state.denoise_mask * sigma,
            positions=state.positions,
            context=context,
            context_mask=None,
        )

    @staticmethod
    def _build_stg_perturbation_config(
        video_enabled: bool,
        audio_enabled: bool,
        stg_blocks: list[int] | None,
        num_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BatchedPerturbationConfig | None:
        """Build STG perturbation config that skips self-attention in the specified blocks."""
        perturbations: list[Perturbation] = []
        if video_enabled:
            perturbations.append(Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=stg_blocks))
        if audio_enabled:
            perturbations.append(Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=stg_blocks))
        if not perturbations:
            return None
        return BatchedPerturbationConfig([PerturbationConfig(perturbations=perturbations)], num_blocks, device, dtype)

    @staticmethod
    def _build_modality_perturbation_config(
        num_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BatchedPerturbationConfig:
        """Build modality guidance perturbations that isolate cross-modal attention."""
        perturbations: list[Perturbation] = [
            Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
        ]
        return BatchedPerturbationConfig([PerturbationConfig(perturbations=perturbations)], num_blocks, device, dtype)

    @staticmethod
    def _load_first_frame(media_path: Path) -> Tensor:
        """Load the first frame from an image or video file as [C, H, W] in [0, 1]."""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        if media_path.suffix.lower() in video_extensions:
            frames, _ = read_video(str(media_path), max_frames=1)
            return frames[0]
        image = open_image_as_srgb(str(media_path))
        return TF.to_tensor(image)

    @staticmethod
    def _resize_and_center_crop(tensor: Tensor, target_height: int, target_width: int) -> Tensor:
        """Resize [N, C, H, W] tensor to cover target dims (preserving aspect ratio) and center-crop."""
        current_height, current_width = tensor.shape[2:]
        if current_height == target_height and current_width == target_width:
            return tensor

        aspect_ratio = current_width / current_height
        target_aspect_ratio = target_width / target_height
        if aspect_ratio > target_aspect_ratio:
            resize_height, resize_width = target_height, int(target_height * aspect_ratio)
        else:
            resize_height, resize_width = int(target_width / aspect_ratio), target_width

        tensor = TF.resize(tensor, size=[resize_height, resize_width], interpolation=InterpolationMode.BICUBIC)
        tensor = tensor.clamp(0, 1)
        h_start = (resize_height - target_height) // 2
        w_start = (resize_width - target_width) // 2
        return tensor[:, :, h_start : h_start + target_height, w_start : w_start + target_width]

    @staticmethod
    def _encode_image(
        image: Tensor, target_height: int, target_width: int, vae_encoder: torch.nn.Module, device: torch.device
    ) -> Tensor:
        """Encode a conditioning image to latent space. Returns [B, C, 1, H', W'] on CPU."""
        image = ValidationRunner._resize_and_center_crop(image.unsqueeze(0), target_height, target_width)
        image = image.unsqueeze(2)  # [B, C, 1, H, W]
        image = (image * 2.0 - 1.0).to(device=device, dtype=torch.bfloat16)
        vae_encoder.to(device)
        encoded = vae_encoder(image)
        vae_encoder.to("cpu")
        return encoded.cpu()

    @staticmethod
    def _preprocess_reference(
        video: Tensor,
        target_height: int,
        target_width: int,
        scale_factors: SpatioTemporalScaleFactors,
        downscale_factor: int = 1,
        temporal_scale_factor: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Preprocess reference video. Returns (preprocessed [-1,1], pixels [0,1])."""
        if target_height % downscale_factor != 0 or target_width % downscale_factor != 0:
            raise ValueError(
                f"Reference target dimensions ({target_height}x{target_width}) must be evenly divisible by "
                f"downscale_factor ({downscale_factor})."
            )
        ref_height = target_height // downscale_factor
        ref_width = target_width // downscale_factor
        if ref_height % scale_factors.height != 0 or ref_width % scale_factors.width != 0:
            raise ValueError(
                f"Scaled reference dimensions ({ref_height}x{ref_width}) must be divisible by the VAE "
                f"spatial factor ({scale_factors.height}). Original: {target_height}x{target_width}, "
                f"downscale_factor: {downscale_factor}"
            )

        video = ValidationRunner._resize_and_center_crop(video, ref_height, ref_width)
        video = rearrange(video, "f c h w -> 1 c f h w")
        valid_frames = (video.shape[2] - 1) // scale_factors.time * scale_factors.time + 1
        video = video[:, :, :valid_frames]

        # VAE-aligned temporal subsampling: keep frame 0, then every Nth frame
        if temporal_scale_factor > 1:
            indices = [0, *list(range(1, video.shape[2], temporal_scale_factor))]
            video = video[:, :, indices]

        pixels = video[0].clone()
        preprocessed = video * 2.0 - 1.0
        return preprocessed, pixels

    @staticmethod
    def _encode_video(video: Tensor, vae_encoder: torch.nn.Module, device: torch.device) -> Tensor:
        """Encode a [B, C, F, H, W] video tensor through the VAE. Returns latent on CPU."""
        vae_encoder.to(device)
        latent = vae_encoder.tiled_encode(video.to(dtype=torch.bfloat16), TilingConfig.default())
        vae_encoder.to("cpu")
        return latent.cpu()


def _build_spatial_crop_mask(
    region: tuple[int, int, int, int],
    target_shape: VideoLatentShape,
    device: torch.device,
    scale_factors: SpatioTemporalScaleFactors,
) -> Tensor:
    """Build a binary mask from a pixel-space spatial region (y1, x1, y2, x2)."""
    y1, x1, y2, x2 = region
    _, _, frames, height, width = target_shape.to_torch_shape()

    def to_latent(v: int, scale: int, max_v: int) -> int:
        return max(0, min(v // scale, max_v))

    ly1 = to_latent(y1, scale_factors.height, height)
    ly2 = to_latent(y2, scale_factors.height, height)
    lx1 = to_latent(x1, scale_factors.width, width)
    lx2 = to_latent(x2, scale_factors.width, width)

    spatial_mask = torch.zeros(height, width, dtype=torch.float32, device=device)
    spatial_mask[ly1:ly2, lx1:lx2] = 1.0

    return spatial_mask.unsqueeze(0).unsqueeze(0).expand(1, frames, -1, -1)  # [1, F, H, W]


def _concatenate_videos_side_by_side(left_video: Tensor, right_video: Tensor) -> Tensor:
    """Concatenate two [C, F, H, W] videos horizontally, matching height and padding frames."""
    left_height, left_width = left_video.shape[2], left_video.shape[3]
    right_height = right_video.shape[2]
    if left_height != right_height:
        scale = right_height / left_height
        new_width = int(left_width * scale)
        c, f, h, w = left_video.shape
        left_video = left_video.reshape(c * f, 1, h, w)
        left_video = TF.resize(left_video, size=[right_height, new_width], interpolation=InterpolationMode.BICUBIC)
        left_video = left_video.clamp(0, 1)
        left_video = left_video.reshape(c, f, right_height, new_width)
    left_frames, right_frames = left_video.shape[1], right_video.shape[1]
    if left_frames < right_frames:
        padding = left_video[:, -1:].expand(-1, right_frames - left_frames, -1, -1)
        left_video = torch.cat([left_video, padding], dim=1)
    elif right_frames < left_frames:
        padding = right_video[:, -1:].expand(-1, left_frames - right_frames, -1, -1)
        right_video = torch.cat([right_video, padding], dim=1)
    return torch.cat([left_video, right_video], dim=3)
