# ruff: noqa: PLC0415

"""
Model loader for LTX-2 trainer using the new ltx-core package.
This module provides a unified interface for loading LTX-2 model components
for training, using SingleGPUModelBuilder from ltx-core.
Example usage:
    # Load individual components
    vae_encoder = load_video_vae_encoder("/path/to/checkpoint.safetensors", device="cuda")
    vae_decoder = load_video_vae_decoder("/path/to/checkpoint.safetensors", device="cuda")
    text_encoder = load_text_encoder("/path/to/gemma", device="cuda")
    # Load all components at once
    components = load_model("/path/to/checkpoint.safetensors", text_encoder_path="/path/to/gemma")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ltx_trainer import logger

# Type alias for device specification
Device = str | torch.device

# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.audio_vae import AudioDecoder, AudioEncoder, Vocoder
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ltx_core.text_encoders.gemma import LTXGemmaTextEncoder
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
    from ltx_core.types import SpatioTemporalScaleFactors


def _to_torch_device(device: Device) -> torch.device:
    """Convert device specification to torch.device."""
    return torch.device(device) if isinstance(device, str) else device


def read_video_scale_factors(model_path: str | Path) -> "SpatioTemporalScaleFactors":
    """Read the video VAE compression factors from a checkpoint's embedded config.
    The VAE is not always loaded where these factors are needed (during training the
    latents are precomputed), so they are derived from the checkpoint metadata via
    ``SpatioTemporalScaleFactors.from_model_config``, which yields the default 32x32x8
    layout for checkpoints whose config carries no VAE block list (e.g. audio-only).
    A failure to *read* the metadata is deliberately not swallowed: it means the
    checkpoint path is wrong or the file is corrupt, and silently falling back to the
    default layout would train a non-default VAE (e.g. 16x16x4) on the wrong RoPE
    positions without ever surfacing the error. Fail loudly instead.
    """
    from ltx_core.loader import SafetensorsModelStateDictLoader
    from ltx_core.types import SpatioTemporalScaleFactors

    metadata = SafetensorsModelStateDictLoader().metadata(str(model_path))
    return SpatioTemporalScaleFactors.from_model_config(metadata.get("config", {}))


# =============================================================================
# Individual Component Loaders
# =============================================================================


def load_transformer(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "LTXModel":
    """Load the LTX transformer model.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded LTXModel transformer
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoEncoder":
    """Load the video VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoEncoder
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoDecoder":
    """Load the video VAE decoder (for inference/validation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoDecoder
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import (
        VideoDecoderConfigurator,
        is_diffusion_video_vae,
        video_decoder_sd_ops_for_checkpoint,
    )
    from ltx_core.model.video_vae.transformer import build_cutlass_fna_diffusion_decoder_op

    checkpoint_path = str(checkpoint_path)

    # Pin neighborhood attention to cutlass-fna, which lowers peak VRAM during decode.
    diffusion_vae = is_diffusion_video_vae(checkpoint_path)
    module_ops = (build_cutlass_fna_diffusion_decoder_op(),) if diffusion_vae else ()

    return SingleGPUModelBuilder(
        model_path=checkpoint_path,
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=video_decoder_sd_ops_for_checkpoint(checkpoint_path, diffusion_vae=diffusion_vae),
        module_ops=module_ops,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioEncoder":
    """Load the audio VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights (default bfloat16, but float32 recommended for quality)
    Returns:
        Loaded AudioEncoder
    """
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioDecoder":
    """Load the audio VAE decoder.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded AudioDecoder
    """
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, AudioDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_vocoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "Vocoder":
    """Load the vocoder (for audio waveform generation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded Vocoder
    """
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_text_encoder(
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
) -> "LTXGemmaTextEncoder":
    """Load the Gemma text encoder.
    Args:
        gemma_model_path: Path to Gemma model directory
        device: Device to load model on
        dtype: Data type for model weights
        load_in_8bit: Whether to load the Gemma model in 8-bit precision using bitsandbytes.
    Returns:
        Loaded LTXGemmaTextEncoder
    """
    if not Path(gemma_model_path).is_dir():
        raise ValueError(f"Gemma model path is not a directory: {gemma_model_path}")

    # Use 8-bit loading path if requested
    if load_in_8bit:
        from ltx_trainer.gemma_8bit import load_8bit_gemma

        return load_8bit_gemma(gemma_model_path, dtype, device=device)

    # Standard loading path
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        GemmaTextEncoderConfigurator,
        get_gemma_ops,
    )
    from ltx_core.utils import find_matching_file

    torch_device = _to_torch_device(device)

    gemma_model_folder = find_matching_file(str(gemma_model_path), "model*.safetensors").parent
    gemma_weight_paths = [str(p) for p in gemma_model_folder.rglob("*.safetensors")]
    gemma_sd_ops, gemma_module_ops = get_gemma_ops(str(gemma_model_path))

    text_encoder = SingleGPUModelBuilder(
        model_path=tuple(gemma_weight_paths),
        model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(str(gemma_model_path)),
        model_sd_ops=gemma_sd_ops,
        module_ops=gemma_module_ops,
    ).build(device=torch_device, dtype=dtype)

    return text_encoder


def load_embeddings_processor(
    checkpoint_path: str | Path,
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "EmbeddingsProcessor":
    """Load the embeddings processor (feature extractor + video/audio connectors).
    Args:
        checkpoint_path: Path to the LTX-2 safetensors checkpoint file
        gemma_model_path: Path to the Gemma model directory. Used to read the
            text encoder's hidden_size / num_hidden_layers when sizing the
            feature-extractor projections.
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded EmbeddingsProcessor with feature extractor and connectors
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        EmbeddingsProcessorConfigurator,
    )

    torch_device = _to_torch_device(device)

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(str(gemma_model_path)),
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    ).build(device=torch_device, dtype=dtype)


# =============================================================================
# Combined Component Loader
# =============================================================================


@dataclass
class LtxModelComponents:
    """Container for all LTX-2 model components."""

    transformer: "LTXModel"
    video_vae_encoder: "VideoEncoder | None" = None
    video_vae_decoder: "VideoDecoder | None" = None
    audio_vae_decoder: "AudioDecoder | None" = None
    vocoder: "Vocoder | None" = None
    text_encoder: "LTXGemmaTextEncoder | None" = None
    scheduler: "LTX2Scheduler | None" = None


def load_model(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    with_video_vae_encoder: bool = False,
    with_video_vae_decoder: bool = True,
    with_audio_vae_decoder: bool = True,
    with_vocoder: bool = True,
    with_text_encoder: bool = True,
) -> LtxModelComponents:
    """
    Load LTX-2 model components from a safetensors checkpoint.
    This is a convenience function that loads multiple components at once.
    For loading individual components, use the dedicated functions:
    - load_transformer()
    - load_video_vae_encoder()
    - load_video_vae_decoder()
    - load_audio_vae_decoder()
    - load_vocoder()
    - load_text_encoder()
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        text_encoder_path: Path to Gemma model directory (required if with_text_encoder=True)
        device: Device to load models on ("cuda", "cpu", etc.)
        dtype: Data type for model weights
        with_video_vae_encoder: Whether to load the video VAE encoder (for preprocessing)
        with_video_vae_decoder: Whether to load the video VAE decoder (for inference/validation)
        with_audio_vae_decoder: Whether to load the audio VAE decoder
        with_vocoder: Whether to load the vocoder
        with_text_encoder: Whether to load the text encoder
    Returns:
        LtxModelComponents containing all loaded model components
    """
    from ltx_core.components.schedulers import LTX2Scheduler

    checkpoint_path = Path(checkpoint_path)

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading LTX-2 model from {checkpoint_path}")

    torch_device = _to_torch_device(device)

    # Load transformer
    logger.debug("Loading transformer...")
    transformer = load_transformer(checkpoint_path, torch_device, dtype)

    # Load video VAE encoder
    video_vae_encoder = None
    if with_video_vae_encoder:
        logger.debug("Loading video VAE encoder...")
        video_vae_encoder = load_video_vae_encoder(checkpoint_path, torch_device, dtype)

    # Load video VAE decoder
    video_vae_decoder = None
    if with_video_vae_decoder:
        logger.debug("Loading video VAE decoder...")
        video_vae_decoder = load_video_vae_decoder(checkpoint_path, torch_device, dtype)

    # Load audio VAE decoder
    audio_vae_decoder = None
    if with_audio_vae_decoder:
        logger.debug("Loading audio VAE decoder...")
        audio_vae_decoder = load_audio_vae_decoder(checkpoint_path, torch_device, dtype)

    # Load vocoder
    vocoder = None
    if with_vocoder:
        logger.debug("Loading vocoder...")
        vocoder = load_vocoder(checkpoint_path, torch_device, dtype)

    # Load text encoder
    text_encoder = None
    if with_text_encoder:
        if text_encoder_path is None:
            raise ValueError("text_encoder_path must be provided when with_text_encoder=True")
        logger.debug("Loading Gemma text encoder...")
        text_encoder = load_text_encoder(text_encoder_path, torch_device, dtype)

    # Create scheduler (stateless, no loading needed)
    scheduler = LTX2Scheduler()

    return LtxModelComponents(
        transformer=transformer,
        video_vae_encoder=video_vae_encoder,
        video_vae_decoder=video_vae_decoder,
        audio_vae_decoder=audio_vae_decoder,
        vocoder=vocoder,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
