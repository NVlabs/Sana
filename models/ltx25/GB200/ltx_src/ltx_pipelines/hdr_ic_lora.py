"""HDR IC-LoRA pipeline: two-stage video generation with HDR output.
Extends the standard IC-LoRA pipeline with HDR decode via LogC3 inverse
transform.  ``__call__`` returns a **linear HDR float** tensor
``[f, h, w, c]``; tonemapping and EXR saving are the caller's
responsibility.
Text embeddings must be pre-computed externally (e.g. using
``PromptEncoder`` from ``ltx_pipelines.utils.blocks`` with a Gemma text
encoder) and saved as a ``.safetensors`` file with ``video_context``
and ``audio_context`` tensors (via ``safetensors.torch.save_file``).
The path is passed via ``text_embeddings_path``.
Run as a script for batch inference::
    python -m ltx_pipelines.hdr_ic_lora \\
        --input ./videos/ \\
        --output-dir ./hdr-output \\
        --distilled-checkpoint-path /models/ltx-2.3-22b-distilled.safetensors \\
        --spatial-upsampler-path /models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \\
        --hdr-lora /path/to/hdr_lora.safetensors \\
        --text-embeddings /path/to/hdr_scene_emb.safetensors \\
        --num-frames 161
Supports resolutions up to 4K (3840x2160 @ 121 frames on 80 GB,
49 frames on 48 GB).  The caller is responsible for choosing a resolution
and frame count that fits in GPU memory.  See ``--help`` for a reference
table, or use ``ltx_pipelines.utils.vram_budget.max_frames_for_resolution``
to query your specific configuration.
"""

import dataclasses
import logging
from dataclasses import replace
from pathlib import Path

import torch
from einops import rearrange
from safetensors import safe_open

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByReferenceLatent,
)
from ltx_core.devices import empty_device_cache
from ltx_core.hdr import HDRTransfer, to_hdr_linear
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import ModelRegistry, Registry
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.modality_tiling import VideoModalityTilingHelper
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, VideoEncoder
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tiling import DimensionSizeConfig, DimensionTilingConfig, TileCountConfig, TileSizeConfig
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.blocks import (
    DiffusionStage,
    ImageConditioner,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    ensure_tiling_config,
    get_device,
    modality_from_latent_state,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import ResizeMode, align_resolution, load_video_conditioning_hdr
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NUM_FRAMES = 161
MIN_RESOLUTION = 64
ALIGNMENT_DIVISOR = 64

# Conditioning videos whose spatial resolution (H x W) exceeds this value are
# encoded with the tiled encoder. The default (512 x 768) is suitable for
# H100-80GB. On lower-VRAM GPUs pass tiled_vae_encode_pixel_threshold=256*256
# to the pipeline constructor.
TILED_VAE_ENCODE_PIXEL_THRESHOLD = 512 * 768

_DEFAULT_QUANTIZATION = QuantizationKind.FP8_CAST

# Default stage-2 configuration: one refinement phase with modest 2-way tiling
# in every dimension and a short 2-step distilled sigma schedule.
_S2 = STAGE_2_DISTILLED_SIGMA_VALUES

_TILED_2F2H2W_OV8_6 = TileCountConfig(
    frames=DimensionTilingConfig(2, 8),
    height=DimensionTilingConfig(2, 6),
    width=DimensionTilingConfig(2, 6),
)

STAGE2_TILINGS = [_TILED_2F2H2W_OV8_6]
STAGE2_SIGMAS = [[_S2[0], _S2[1], 0.0]]
STAGE2_USE_IC_LORA = [True]


def _encode_tiling_config(tiling_config: TilingConfig) -> TilingConfig:
    if not isinstance(tiling_config, TileSizeConfig):
        return tiling_config
    if not tiling_config.height.is_tiled() and not tiling_config.width.is_tiled():
        return tiling_config
    long = max(tiling_config.height.tile_size, tiling_config.width.tile_size)
    overlap = tiling_config.height.overlap if tiling_config.height.is_tiled() else tiling_config.width.overlap
    return TileSizeConfig(
        frames=tiling_config.frames,
        height=DimensionSizeConfig(tile_size=long, overlap=overlap),
        width=DimensionSizeConfig(tile_size=long, overlap=overlap),
    )


def _clamp_dim_tiling(cfg: DimensionTilingConfig, dim_size: int, axis: str) -> DimensionTilingConfig:
    """Clamp a single dim's tile count and overlap to the latent's extent.
    ``split_by_count`` requires ``overlap < tile_size``; with
    ``tile_size = (dim_size + overlap*(n-1)) // n`` this reduces to
    ``overlap <= dim_size - n``. When the configured overlap exceeds this
    bound it is clamped; if the latent is too small to hold ``n`` tiles
    at all, tiling falls back to a single tile on this axis.
    """
    n = cfg.num_tiles
    if n <= 1:
        return cfg
    if dim_size < n:
        logger.warning(
            "%s tiling: dim_size=%d < num_tiles=%d; falling back to 1 tile on this axis.",
            axis,
            dim_size,
            n,
        )
        return DimensionTilingConfig(1, 0)
    max_overlap = dim_size - n
    if cfg.overlap <= max_overlap:
        return cfg
    logger.warning(
        "%s tiling: overlap=%d exceeds latent bound (%d); clamping to %d.",
        axis,
        cfg.overlap,
        max_overlap,
        max_overlap,
    )
    return DimensionTilingConfig(n, max_overlap)


def _clamp_tile_to_latent(tiling: TileCountConfig, latent_shape: tuple[int, int, int]) -> TileCountConfig:
    """Clamp frame, height, and width tilings to the latent's extents.
    ``latent_shape`` is ``(F, H, W)`` in latent units.
    """
    f, h, w = latent_shape
    return replace(
        tiling,
        frames=_clamp_dim_tiling(tiling.frames, f, "Frame"),
        height=_clamp_dim_tiling(tiling.height, h, "Height"),
        width=_clamp_dim_tiling(tiling.width, w, "Width"),
    )


# Default tiling config (spatial tile 1280 px, overlap 256 px; temporal 32
# frames, overlap 16). On GPUs with < 80 GB VRAM you may need to shrink
# the spatial tile size (e.g. 768) to avoid OOM during VAE decode.
DEFAULT_SPATIAL_TILE = 1280
DEFAULT_SPATIAL_OVERLAP = 256
DEFAULT_TEMPORAL_TILE = 32
DEFAULT_TEMPORAL_OVERLAP = 16

# ---------------------------------------------------------------------------
# HDR LoRA config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HdrLoraConfig:
    """Explicit HDR LoRA parameters.
    Read from LoRA safetensors metadata by :func:`read_hdr_lora_config`, or
    constructed manually for testing.
    """

    hdr_transform: HDRTransfer = HDRTransfer.LOGC3
    reference_downscale_factor: int = 1


def read_hdr_lora_config(lora_path: str) -> HdrLoraConfig | None:
    """Read HDR config from LoRA safetensors metadata.
    Returns ``None`` when the LoRA has no HDR metadata.
    """
    try:
        with safe_open(lora_path, framework="pt") as f:
            metadata = f.metadata() or {}
    except (OSError, ValueError) as e:
        logger.warning("Failed to read metadata from LoRA file '%s': %s", lora_path, e)
        return None

    raw_transform = metadata.get("hdr_transform", "")
    has_hdr = bool(raw_transform or metadata.get("use_hdr_transform"))
    if not has_hdr:
        return None

    # Metadata may store a bare flag ("true") or a transfer name ("logc3").
    transfer = HDRTransfer(raw_transform) if raw_transform and raw_transform != "true" else HDRTransfer.LOGC3
    scale = int(metadata.get("reference_downscale_factor", 1))
    return HdrLoraConfig(hdr_transform=transfer, reference_downscale_factor=scale)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class HDRICLoraPipeline:
    """Two-stage IC-LoRA pipeline with HDR support.
    Same two-stage architecture as ICLoraPipeline (half-res generation + 2x
    upscale refinement), with HDR decode via LogC3 inverse.
    ``__call__`` returns a **linear HDR float** tensor ``[f, h, w, c]``.
    Tonemapping and EXR saving are the caller's responsibility.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        spatial_upsampler_path: str,
        hdr_lora: str | Path,
        text_embeddings_path: str | Path,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | QuantizationKind | None = _DEFAULT_QUANTIZATION,
        registry: Registry | None = None,
        hdr_lora_config: HdrLoraConfig | None = None,
        tiled_vae_encode_pixel_threshold: int = TILED_VAE_ENCODE_PIXEL_THRESHOLD,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ):
        """
        Args:
            model_paths: Resolved component paths (diffusion + video VAE at minimum).
            spatial_upsampler_path: Path to the spatial upsampler checkpoint.
            hdr_lora: Path to the HDR IC-LoRA ``.safetensors`` file.
            text_embeddings_path: Path to pre-computed text embeddings
                (``.safetensors`` file with ``video_context`` and
                ``audio_context`` tensors).
            device: Target device. Auto-detected when ``None``.
            quantization: Quantization policy. Defaults to ``fp8_cast``.
            registry: Optional registry for non-diffusion blocks (VAE encode/decode,
                upsampler). Diffusion stages always use a dedicated full
                :class:`ModelRegistry` that is cleared before decode so
                transformer weights do not pin VRAM.
            hdr_lora_config: Explicit HDR LoRA config override. When ``None``,
                auto-detected from LoRA safetensors metadata.
            tiled_vae_encode_pixel_threshold: Conditioning videos whose spatial
                area (H x W) exceeds this value are encoded with the tiled
                encoder. Default ``512 * 768`` is suitable for 80 GB GPUs.
                Use ``256 * 256`` on GPUs with less VRAM.
            offload_mode: Weight offloading strategy for diffusion stages.
        """
        self.device = device or get_device()
        self._tiled_vae_encode_threshold = tiled_vae_encode_pixel_threshold
        if isinstance(quantization, QuantizationKind):
            quantization = quantization.to_policy(checkpoint_path=model_paths.transformer())
        if offload_mode != OffloadMode.NONE and quantization is not None:
            logger.info("Offload mode enabled — disabling quantization (not supported with layer streaming).")
            quantization = None
        self.dtype = torch.bfloat16

        lora_path = str(Path(hdr_lora).resolve())
        loras = (LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)

        # Load pre-computed text embeddings from safetensors.
        emb_path = Path(text_embeddings_path)
        logger.info("Loading text embeddings from %s", emb_path)
        with safe_open(emb_path, framework="pt", device=str(self.device)) as f:
            self.text_embeddings: tuple[torch.Tensor, torch.Tensor] = (
                f.get_tensor("video_context"),
                f.get_tensor("audio_context"),
            )

        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        # Full cache for diffusion only: cheap per-tile stage_2 rebuilds. Cleared
        # before decode so transformer SDs are not still pinning VRAM.
        self._diffusion_registry = ModelRegistry()
        self.stage_1 = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=loras,
            quantization=quantization,
            registry=self._diffusion_registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage_2 = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=loras,
            quantization=quantization,
            registry=self._diffusion_registry,
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

        # HDR config: explicit override, or auto-detect from LoRA metadata.
        if hdr_lora_config is not None:
            self._hdr_config: HdrLoraConfig | None = hdr_lora_config
        else:
            self._hdr_config = read_hdr_lora_config(lora_path)

        if self._hdr_config is not None:
            logger.info(
                "[HDR IC-LoRA] HDR mode enabled (%s decode)",
                self._hdr_config.hdr_transform.value,
            )

    @property
    def hdr_transform(self) -> HDRTransfer:
        """Active HDR working-space transfer (defaults to LogC3)."""
        return self._hdr_config.hdr_transform if self._hdr_config is not None else HDRTransfer.LOGC3

    @property
    def reference_downscale_factor(self) -> int:
        """Reference video downscale factor from HDR LoRA config."""
        return self._hdr_config.reference_downscale_factor if self._hdr_config is not None else 1

    def __call__(  # noqa: PLR0913
        self,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        video_conditioning: list[tuple[str, float]],
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        high_quality_hdr: bool = False,
        stage2_tilings: list[TileCountConfig] | None = None,
        stage2_sigmas: list[list[float]] | None = None,
        stage2_use_ic_lora: list[bool] | None = None,
    ) -> torch.Tensor:
        """Generate video with IC-LoRA conditioning and HDR output.
        Returns a linear HDR float tensor ``[f, h, w, c]``.
        Args:
            seed: Random seed for reproducibility.
            height: Desired output video height in pixels. Aligned internally
                to the nearest multiple of 64 (rounded up). Decoded output is
                cropped back to this size.
            width: Desired output video width in pixels. Same alignment rules
                as *height*.
            num_frames: Number of frames to generate.
            frame_rate: Output video frame rate.
            video_conditioning: List of (path, strength) tuples for IC-LoRA video conditioning.
            high_quality_hdr: High-quality HDR mode. Duplicates each conditioning
                frame and generates at 2x frame count, then keeps every other
                output frame. Reduces temporal artifacts at the cost of ~2x
                generation time.
        Returns:
            Linear HDR float tensor ``[f, h, w, c]``.
        """
        # In high-quality HDR mode, generate 2*N - 1 frames internally
        # (satisfies (n-1)%8==0 when N itself does), then keep every other frame.
        if high_quality_hdr:
            gen_num_frames = 2 * num_frames - 1
            logger.info("[HDR IC-LoRA] High-quality HDR: %d -> %d internal frames", num_frames, gen_num_frames)
        else:
            gen_num_frames = num_frames
        gen_w, gen_h, crop_w, crop_h = align_resolution(
            width, height, ResizeMode.REFLECT_PAD, divisor=ALIGNMENT_DIVISOR
        )
        if gen_h < MIN_RESOLUTION or gen_w < MIN_RESOLUTION:
            raise ValueError(
                f"Resolution ({width}x{height}) is too small after alignment "
                f"(got {gen_w}x{gen_h}, need at least {MIN_RESOLUTION}x{MIN_RESOLUTION})."
            )
        needs_crop = crop_w != gen_w or crop_h != gen_h
        if needs_crop:
            logger.info(
                "[HDR IC-LoRA] Aligned %dx%d -> %dx%d, will crop to %dx%d",
                width,
                height,
                gen_w,
                gen_h,
                crop_w,
                crop_h,
            )

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=gen_num_frames, height=gen_h, width=gen_w, fps=frame_rate),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        video_context, _ = self.text_embeddings

        # Stage 1: Initial low resolution video generation.
        s1_w, s1_h = gen_w // 2, gen_h // 2

        stage_1_conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                video_conditioning=video_conditioning,
                height=s1_h,
                width=s1_w,
                video_encoder=enc,
                num_frames=gen_num_frames,
                tiling_config=tiling_config,
                high_quality_hdr=high_quality_hdr,
            )
        )

        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        try:
            # HDR is video-only: skip the audio stream to avoid denoising 5B audio params.
            video_state, _ = self.stage_1(
                denoiser=SimpleDenoiser(video_context, None),
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=s1_w,
                height=s1_h,
                frames=gen_num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=video_context,
                    conditionings=stage_1_conditionings,
                ),
            )

            if stage2_tilings is None:
                stage2_tilings = list(STAGE2_TILINGS)
            if stage2_sigmas is None:
                stage2_sigmas = [list(s) for s in STAGE2_SIGMAS]
            if stage2_use_ic_lora is None:
                stage2_use_ic_lora = list(STAGE2_USE_IC_LORA)
            if not (len(stage2_tilings) == len(stage2_sigmas) == len(stage2_use_ic_lora)):
                raise ValueError("stage2_tilings, stage2_sigmas, and stage2_use_ic_lora must have equal length")

            # Stage 2: Upsample and refine at full resolution.
            upscaled_video_latent = self.upsampler(video_state.latent[:1])

            stage_2_conditionings = self.image_conditioner(
                lambda enc: self._create_conditionings(
                    video_conditioning=video_conditioning,
                    height=gen_h,
                    width=gen_w,
                    video_encoder=enc,
                    num_frames=gen_num_frames,
                    tiling_config=tiling_config,
                    high_quality_hdr=high_quality_hdr,
                )
            )
            # Per-tile stage_2(...) rebuilds are cheap with shell caching.
            phase_latent = upscaled_video_latent
            for phase_idx, (tiling, sigmas_list, use_ic) in enumerate(
                zip(stage2_tilings, stage2_sigmas, stage2_use_ic_lora, strict=True)
            ):
                diffusion_tiling = _clamp_tile_to_latent(tiling, tuple(phase_latent.shape[2:5]))
                conditionings = stage_2_conditionings if use_ic else []
                sigma_t = torch.tensor(sigmas_list, dtype=torch.float32, device=self.device)
                logger.info(
                    "[Stage 2 / phase %d] sigmas=%s ic_lora=%s tiling_h=%s tiling_w=%s",
                    phase_idx,
                    sigmas_list,
                    use_ic,
                    diffusion_tiling.height,
                    diffusion_tiling.width,
                )
                phase_latent = self._run_stage2_phase(
                    latent=phase_latent,
                    conditionings=conditionings,
                    tiling=diffusion_tiling,
                    sigmas=sigma_t,
                    v_ctx=video_context,
                    frame_rate=frame_rate,
                    seed=seed,
                )
            final_video_latent = phase_latent
        finally:
            # Drop cached transformer shells/weights so decode (or failure cleanup) can claim VRAM.
            self._diffusion_registry.clear()

        crop_size = (crop_w, crop_h) if needs_crop else None
        return self._decode_video(
            final_video_latent,
            tiling_config,
            generator,
            crop_size,
            high_quality_hdr=high_quality_hdr,
        )

    def _run_stage2_phase(
        self,
        latent: torch.Tensor,
        conditionings: list[ConditioningItem],
        tiling: TileCountConfig,
        sigmas: torch.Tensor,
        v_ctx: torch.Tensor,
        frame_rate: float,
        seed: int,
    ) -> torch.Tensor:
        """Run one stage-2 denoising phase with optional IC-LoRA conditioning.
        Each tile calls ``stage_2(...)`` (build→denoise→dispose) with a tile-sized
        ``ModalitySpec`` for video only (audio is omitted entirely for HDR).
        IC-LoRA conditionings are sliced spatially to match each tile's extent.
        """
        batch, n_channels, n_frames, n_height, n_width = latent.shape
        full_shape = VideoLatentShape(batch=batch, channels=n_channels, frames=n_frames, height=n_height, width=n_width)
        full_tools = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1),
            full_shape,
            frame_rate,
            scale_factors=self.stage_2.video_scale_factors,
        )
        helper = VideoModalityTilingHelper(tiling, full_tools)

        ref_initial = full_tools.create_initial_state(device=self.device, dtype=self.dtype)
        ref_modality = modality_from_latent_state(ref_initial, v_ctx, sigmas[0])
        n_gen = full_tools.target_shape.token_count()
        blend_output = torch.zeros(batch, n_gen, n_channels, device=self.device, dtype=self.dtype)
        patchifier = VideoLatentPatchifier(patch_size=1)
        df = self.reference_downscale_factor

        for tile_idx, tile in enumerate(helper.tiles):
            _, ctx = helper.tile_modality(ref_modality, tile, normalize_positions=True)
            frame_s, height_s, width_s = tile.in_coords
            tile_h = height_s.stop - height_s.start
            tile_w = width_s.stop - width_s.start
            tile_f = frame_s.stop - frame_s.start

            tile_conditionings = [
                VideoConditionByReferenceLatent(
                    latent=cond.latent[
                        :,
                        :,
                        frame_s,
                        slice(height_s.start // df, height_s.stop // df),
                        slice(width_s.start // df, width_s.stop // df),
                    ].to(device=self.device, dtype=self.dtype),
                    downscale_factor=cond.downscale_factor,
                    strength=cond.strength,
                )
                for cond in conditionings
            ]

            sf = self.stage_2.video_scale_factors
            tile_video_state, _ = self.stage_2(
                denoiser=SimpleDenoiser(v_ctx, None),
                sigmas=sigmas,
                noiser=GaussianNoiser(generator=torch.Generator(device=self.device).manual_seed(seed + tile_idx)),
                width=tile_w * sf.width,
                height=tile_h * sf.height,
                frames=(tile_f - 1) * sf.time + 1,
                fps=frame_rate,
                video=ModalitySpec(
                    context=v_ctx,
                    conditionings=tile_conditionings,
                    noise_scale=sigmas[0].item(),
                    initial_latent=latent[:, :, frame_s, height_s, width_s].to(device=self.device, dtype=self.dtype),
                ),
            )

            tile_tokens = patchifier.patchify(tile_video_state.latent)
            blend_output = helper.blend(tile_tokens, tile, ctx, blend_output)

        return full_tools.unpatchify(replace(ref_initial, latent=blend_output)).latent

    def _decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None,
        generator: torch.Generator,
        crop_size: tuple[int, int] | None = None,
        *,
        high_quality_hdr: bool = False,
    ) -> torch.Tensor:
        """Decode latent to HDR video, optionally cropping to target size.
        Args:
            crop_size: ``(width, height)`` to crop decoded frames to, or
                ``None`` to skip cropping.
            high_quality_hdr: When True, keep only every other frame (undoes the
                2x generation applied during high-quality HDR mode).
        Returns:
            Linear HDR float tensor ``[f, h, w, c]``.
        """
        # Always-HDR path: float32 weights + latent so compute/output/tiling stay fp32.
        # to_hdr_linear expects float32 [0, 1] working-space codes.
        latent = latent.to(dtype=torch.float32)
        decoded = torch.cat(
            list(self.video_decoder(latent, tiling_config, generator, dtype=torch.float32)),
            dim=0,
        )
        decoded = rearrange(decoded, "f h w c -> 1 c f h w")
        hdr = to_hdr_linear(decoded, transfer=self.hdr_transform)
        del decoded
        out = rearrange(hdr[0], "c f h w -> f h w c")
        if crop_size is not None:
            out = out[:, : crop_size[1], : crop_size[0], :]
        if high_quality_hdr:
            out = out[::2]
        return out

    def _create_conditionings(
        self,
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        tiling_config: TilingConfig | None = None,
        high_quality_hdr: bool = False,
    ) -> list[ConditioningItem]:
        """Create conditioning items for video generation."""
        conditionings: list[ConditioningItem] = []

        scale = self.reference_downscale_factor
        if scale != 1 and (height % scale != 0 or width % scale != 0):
            raise ValueError(
                f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor ({scale})"
            )
        ref_height = height // scale
        ref_width = width // scale

        # In high-quality HDR mode, load half the frames then duplicate each one.
        load_frame_cap = (num_frames + 1) // 2 if high_quality_hdr else num_frames

        for video_path, strength in video_conditioning:
            video = torch.cat(
                list(
                    load_video_conditioning_hdr(
                        video_path=video_path,
                        height=ref_height,
                        width=ref_width,
                        frame_cap=load_frame_cap,
                        dtype=self.dtype,
                        device=self.device,
                        transfer=self.hdr_transform,
                        resize_mode=ResizeMode.REFLECT_PAD,
                    )
                ),
                dim=2,
            )
            if high_quality_hdr:
                video = video.repeat_interleave(2, dim=2)[:, :, :num_frames, :, :]
            if tiling_config is not None and ref_height * ref_width > self._tiled_vae_encode_threshold:
                encoded_video = video_encoder.tiled_encode(video, _encode_tiling_config(tiling_config))
            else:
                encoded_video = video_encoder(video)

            cond = VideoConditionByReferenceLatent(
                latent=encoded_video,
                downscale_factor=scale,
                strength=strength,
            )
            conditionings.append(cond)

        if video_conditioning:
            logger.info("[HDR IC-LoRA] Added %d video conditioning(s)", len(video_conditioning))

        return conditionings


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _make_tiling_config(
    spatial_tile: int = DEFAULT_SPATIAL_TILE,
    spatial_overlap: int = DEFAULT_SPATIAL_OVERLAP,
    temporal_tile: int = DEFAULT_TEMPORAL_TILE,
    temporal_overlap: int = DEFAULT_TEMPORAL_OVERLAP,
) -> TilingConfig:
    """Build a TilingConfig from explicit sizes.
    The defaults (1280 px spatial tile, 256 px overlap; 32 temporal frames,
    16 overlap) are suitable for H100-80 GB.  On GPUs with less VRAM,
    reduce the spatial tile size (e.g. ``spatial_tile=768``).
    """
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig  # noqa: PLC0415

    return TileSizeConfig(
        height=DimensionSizeConfig(tile_size=spatial_tile, overlap=spatial_overlap),
        width=DimensionSizeConfig(tile_size=spatial_tile, overlap=spatial_overlap),
        frames=DimensionSizeConfig(tile_size=temporal_tile, overlap=temporal_overlap),
    )


_VIDEO_SUFFIXES = {".mp4", ".mov"}


def _collect_videos(input_path: Path) -> list[Path]:
    """Return a list of .mp4/.mov files from *input_path* (file or directory)."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES)
    logger.error("Input %s is not a file or directory", input_path)
    return []


def _process_single_video(  # noqa: PLR0913
    pipeline: HDRICLoraPipeline,
    video_path: Path,
    vid_w: int,
    vid_h: int,
    num_frames: int,
    frame_rate: float,
    output_dir: Path,
    tiling_config: TilingConfig,
    seed: int,
    skip_mp4: bool,
    exr_executor: "ThreadPoolExecutor",  # noqa: F821
    exr_futures: list,
    high_quality_hdr: bool = False,
) -> None:
    """Run inference on a single video: generate EXR frames + optional H.264 .mp4 preview."""
    import gc  # noqa: PLC0415
    import time  # noqa: PLC0415

    from ltx_pipelines.utils.media_io import encode_exr_sequence_to_mp4, save_exr_tensor  # noqa: PLC0415

    output_mp4 = output_dir / f"{video_path.stem}.mp4"
    exr_dir = output_dir / f"{video_path.stem}_exr"

    t0 = time.time()
    hdr_video = pipeline(
        seed=seed,
        height=vid_h,
        width=vid_w,
        num_frames=num_frames,
        frame_rate=frame_rate,
        video_conditioning=[(str(video_path), 1.0)],
        tiling_config=tiling_config,
        high_quality_hdr=high_quality_hdr,
    )

    exr_dir.mkdir(parents=True, exist_ok=True)
    for j in range(hdr_video.shape[0]):
        frame_cpu = hdr_video[j].cpu().clone()
        path = exr_dir / f"frame_{j:05d}.exr"
        exr_futures.append(exr_executor.submit(save_exr_tensor, frame_cpu, str(path), True))

    del hdr_video
    gc.collect()
    empty_device_cache()

    if not skip_mp4:
        # Wait for EXR saves to finish before encoding.
        for fut in exr_futures:
            fut.result()
        logger.info("Encoding H.264 sRGB preview: %s", video_path.name)
        encode_exr_sequence_to_mp4(exr_dir, output_mp4, frame_rate)

    elapsed = time.time() - t0
    logger.info("Decode + encode: %.1fs | %s", elapsed, output_mp4)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> "argparse.ArgumentParser":  # noqa: F821
    """Build the argument parser for HDR IC-LoRA batch inference."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="HDR IC-LoRA inference: EXR frames + tonemapped ProRes .mov.",
        # Prevent ``--hdr`` from abbreviating to ``--hdr-lora`` (native HDR flag is not supported here).
        allow_abbrev=False,
        epilog="""\
Resolution & frame constraints
------------------------------
  * Width and height must each be divisible by 32.
  * Frame count must satisfy (frames - 1) %% 8 == 0.
    Valid counts: 1, 9, 17, 25, ..., 121, 129, 137, 145, 153, 161.

Max frames by resolution (fp8_cast, bfloat16 VAE, tiled decode)
---------------------------------------------------------------
  Resolution       80 GB (H100)    48 GB (A6000)
  ------------------------------------------------
   720p 1280x720    161+ frames     161+ frames
  1080p 1920x1080   161+ frames     161+ frames
  2K    2048x1080   161+ frames     161+ frames
  1440p 2560x1440   161+ frames     137 frames
  4K    3840x2160   121 frames       49 frames
  4K    4096x2160   105 frames       49 frames

  Estimates from ltx_pipelines.utils.vram_budget. Run
    python -c "from ltx_pipelines.utils.vram_budget import \\
      max_frames_for_resolution as mf; print(mf(W, H, vram_gb=GB))"
  to check your specific resolution and GPU.

  * The tiled-encode threshold (%(tiled_threshold)s px) and the default
    tiling config (%(stile)s px spatial tile) are tuned for 80 GB.
    On lower-VRAM GPUs pass --spatial-tile 768 (or smaller).
"""
        % {"tiled_threshold": TILED_VAE_ENCODE_PIXEL_THRESHOLD, "stile": DEFAULT_SPATIAL_TILE},
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Single .mp4 or directory of .mp4 videos.")
    parser.add_argument("--output-dir", required=True, help="Directory for .mov and EXR folders.")
    parser.add_argument("--hdr-lora", required=True, help="HDR IC-LoRA .safetensors file.")
    parser.add_argument("--text-embeddings", required=True, help="Pre-computed text embeddings (.safetensors file).")
    parser.add_argument("--distilled-checkpoint-path", required=True, help="Distilled model checkpoint (.safetensors).")
    parser.add_argument("--spatial-upsampler-path", required=True, help="Spatial upsampler (.safetensors).")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of output frames.  Must satisfy (n-1) %% 8 == 0 (default: {DEFAULT_NUM_FRAMES}).",
    )
    parser.add_argument(
        "--spatial-tile",
        type=int,
        default=DEFAULT_SPATIAL_TILE,
        help=f"Spatial tile size in pixels for tiled VAE decode (default: {DEFAULT_SPATIAL_TILE}). "
        "Reduce on lower-VRAM GPUs (e.g. 768 for 48 GB).",
    )
    parser.add_argument("--skip-mp4", action="store_true", help="Skip H.264 MP4 encoding, only produce EXR.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed (default: 10).")
    parser.add_argument(
        "--offload",
        dest="offload_mode",
        type=OffloadMode,
        default=OffloadMode.NONE,
        choices=list(OffloadMode),
        help=(
            "Weight offloading strategy. "
            "'none' keeps all weights on GPU (default). "
            "'cpu' pins weights in CPU RAM, streams to GPU per layer. "
            "'disk' reads weights from disk on demand (lowest memory). "
            "Example: --offload cpu"
        ),
    )
    parser.add_argument(
        "--high-quality",
        action="store_true",
        help="High-quality HDR mode. Generates at 2x frame count internally "
        "and keeps every other frame for smoother output. ~2x slower.",
    )
    parser.add_argument(
        "--video-vae-path",
        default=None,
        help=(
            "Optional path to a separate video VAE checkpoint (.safetensors). "
            "When set, encoding/decoding use this file instead of "
            "--distilled-checkpoint-path. Decoder kind is selected from file metadata. "
            "Diffusion VAEs require the natten extra "
            "(uv sync --package ltx-core --extra natten)."
        ),
    )
    parser.add_argument(
        "--diffvae-optimization",
        type=DiffVAEMode,
        default=DiffVAEMode.CHUNKED_EAGER,
        choices=list(DiffVAEMode),
        help=(
            "DiffVAE decode optimization preset. "
            "'chunked_eager' (default): deferred stage-4, W-chunks=4, cutlass-fna. "
            "'chunked_compile': same chunking with torch.compile (det stages off). "
            "'combined_compile': combined context, full compile (highest VRAM, fastest warm). "
            "Ignored for convolutional VAEs."
        ),
    )
    return parser
@torch.inference_mode()
def main() -> None:
    """Batch HDR IC-LoRA inference: per-frame EXR + tonemapped ProRes .mov."""
    import time  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from ltx_pipelines.utils.media_io import get_videostream_metadata  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)

    args = _build_arg_parser().parse_args()
    high_quality = args.high_quality
    num_frames = args.num_frames

    tiling_config = _make_tiling_config(spatial_tile=args.spatial_tile)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _collect_videos(input_path)
    if not videos:
        logger.error("No valid videos to process.")
        return
    logger.info("Found %d video(s), generating %d frames each", len(videos), num_frames)

    logger.info("Loading pipeline...")
    # HDR uses precomputed embeddings — no text encoder. CLI boundary builds ModelPaths; the
    # pipeline only consumes that object (and non-path args like LoRA / embeddings file).
    model_paths = ModelPaths.from_monolith(
        args.distilled_checkpoint_path,
        gemma_root=None,
        video_vae_path=args.video_vae_path,
    )
    pipeline = HDRICLoraPipeline(
        model_paths=model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        hdr_lora=args.hdr_lora,
        text_embeddings_path=args.text_embeddings,
        offload_mode=args.offload_mode,
        diffvae_optimization=args.diffvae_optimization,
    )
    logger.info("Pipeline loaded.")

    exr_executor = ThreadPoolExecutor(max_workers=4)
    exr_futures: list = []

    total_t0 = time.time()
    successes = 0

    for i, video_path in enumerate(videos, 1):
        meta = get_videostream_metadata(str(video_path))
        vid_w, vid_h = meta.width, meta.height
        logger.info("%s", "=" * 60)
        logger.info("[%d/%d] %s  (%dx%d, %df)", i, len(videos), video_path.name, vid_w, vid_h, num_frames)

        _process_single_video(
            pipeline=pipeline,
            video_path=video_path,
            vid_w=vid_w,
            vid_h=vid_h,
            num_frames=num_frames,
            frame_rate=meta.fps,
            output_dir=output_dir,
            tiling_config=tiling_config,
            seed=args.seed,
            skip_mp4=args.skip_mp4,
            exr_executor=exr_executor,
            exr_futures=exr_futures,
            high_quality_hdr=high_quality,
        )
        successes += 1

    infer_elapsed = time.time() - total_t0
    logger.info("%s", "=" * 60)
    logger.info("All inference done in %.0fs  (%d/%d OK)", infer_elapsed, successes, len(videos))

    if exr_futures:
        t0 = time.time()
        logger.info("Waiting for %d EXR saves...", len(exr_futures))
        for fut in exr_futures:
            fut.result()
        exr_wait = time.time() - t0
        if exr_wait > 0.1:
            logger.info("EXR save wait: %.1fs", exr_wait)

    logger.info("Total wall time: %.0fs", time.time() - total_t0)


if __name__ == "__main__":
    main()
