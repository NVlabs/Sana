#!/usr/bin/env python3

"""
Preprocess a media dataset for LTX-2 training.
Automatically detects dataset columns and processes each according to a convention table.
Column names determine what gets encoded and where outputs go — no per-role CLI flags needed.
Convention table:
    video           → Video VAE    → latents/
    audio           → Audio VAE    → audio_latents/
    reference_video → Video VAE    → reference_latents/
    reference_audio → Audio VAE    → reference_audio_latents/
    video_mask      → (downsample) → video_masks/
    audio_mask      → (downsample) → audio_masks/
    caption         → Text encoder → conditions/
Legacy aliases: media_path → video, ref_media_path → reference_video
Basic usage:
    python scripts/process_dataset.py /path/to/dataset.json --resolution-buckets 768x768x49 \\
        --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma
"""

from pathlib import Path

import typer
from decode_latents import LatentsDecoder
from process_captions import compute_captions_embeddings
from process_videos import (
    compute_audio_latents,
    compute_audio_masks,
    compute_latents,
    compute_scaled_resolution_buckets,
    compute_video_masks,
    detect_dataset_columns,
    parse_resolution_buckets,
)
from rich.console import Console

from ltx_trainer import logger
from ltx_trainer.gpu_utils import free_gpu_memory_context
from ltx_trainer.model_loader import read_video_scale_factors

console = Console()

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Preprocess a media dataset for LTX-2 training. "
    "Automatically detects columns (video, audio, reference_video, reference_audio, caption) "
    "and processes each with the appropriate encoder.",
)

_KNOWN_ROLES = {"video", "audio", "reference_video", "reference_audio", "video_mask", "audio_mask", "caption"}
_LEGACY_ALIASES = {"media_path": "video", "ref_media_path": "reference_video"}


def preprocess_dataset(  # noqa: PLR0912, PLR0913, PLR0915
    dataset_file: str,
    resolution_buckets: list[tuple[int, int, int]] | None,
    model_path: str,
    text_encoder_path: str,
    device: str,
    output_dir: str | None = None,
    video_column: str | None = None,
    caption_column: str | None = None,
    batch_size: int = 1,
    lora_trigger: str | None = None,
    vae_tiling: bool = False,
    decode: bool = False,
    remove_llm_prefixes: bool = False,
    reference_downscale_factor: int = 1,
    reference_temporal_scale_factor: int = 1,
    skip_audio: bool = False,
    audio_durations: list[float] | None = None,
    load_text_encoder_in_8bit: bool = False,
    overwrite: bool = False,
) -> None:
    """Run the preprocessing pipeline with convention-based column detection."""
    _validate_dataset_file(dataset_file)

    # Video VAE compression factors, derived from the checkpoint config (default 32x32x8).
    scale_factors = read_video_scale_factors(model_path)

    # Detect columns and resolve roles
    dataset_columns = detect_dataset_columns(dataset_file)
    roles = _resolve_columns(dataset_columns, video_column, caption_column)

    # Log detected roles
    for role, col in sorted(roles.items()):
        alias_note = f" (alias for '{role}')" if col != role else ""
        logger.info(f"Detected column '{col}'{alias_note} → {role}")

    # Validate: need at least caption
    if "caption" not in roles:
        raise ValueError(
            f"No caption column found. Dataset has columns: {dataset_columns}. "
            f"Expected 'caption' or use --caption-column to specify."
        )

    # Validate: need video or audio
    has_video = "video" in roles
    has_audio = "audio" in roles
    if not has_video and not has_audio:
        raise ValueError(
            f"No media column found. Dataset has columns: {dataset_columns}. "
            f"Expected 'video', 'audio', or 'media_path' (legacy)."
        )

    # Validate: video modes need resolution buckets
    if has_video and not resolution_buckets:
        raise ValueError("--resolution-buckets is required when the dataset has a video column.")

    output_base = Path(output_dir) if output_dir else Path(dataset_file).parent / ".precomputed"

    if lora_trigger:
        logger.info(f'LoRA trigger word "{lora_trigger}" will be prepended to all captions')

    # --- Phase 1: Text encoder ---
    with free_gpu_memory_context():
        compute_captions_embeddings(
            dataset_file=dataset_file,
            output_dir=str(output_base / "conditions"),
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            caption_column=roles["caption"],
            media_column=roles.get("video") or roles.get("audio") or roles["caption"],
            lora_trigger=lora_trigger,
            remove_llm_prefixes=remove_llm_prefixes,
            batch_size=batch_size,
            device=device,
            load_in_8bit=load_text_encoder_in_8bit,
            overwrite=overwrite,
        )

    # --- Phase 2: Video VAE (video, reference_video) ---
    if has_video and resolution_buckets:
        # Determine if audio should be auto-extracted from video files
        auto_audio = not skip_audio and "audio" not in roles

        audio_latents_dir = str(output_base / "audio_latents") if auto_audio else None
        if auto_audio:
            logger.info("Audio will be auto-extracted from video files (use --skip-audio to disable)")

        with free_gpu_memory_context():
            compute_latents(
                dataset_file=dataset_file,
                video_column=roles["video"],
                resolution_buckets=resolution_buckets,
                output_dir=str(output_base / "latents"),
                model_path=model_path,
                batch_size=batch_size,
                device=device,
                vae_tiling=vae_tiling,
                with_audio=auto_audio,
                audio_output_dir=audio_latents_dir,
                overwrite=overwrite,
            )

        # Process reference video if present
        if "reference_video" in roles:
            if reference_downscale_factor > 1 and len(resolution_buckets) > 1:
                raise ValueError(
                    "When using --reference-downscale-factor > 1, only a single resolution bucket is supported."
                )
            if reference_temporal_scale_factor > 1 and len(resolution_buckets) > 1:
                raise ValueError(
                    "When using --reference-temporal-scale-factor > 1, only a single resolution bucket is supported."
                )

            reference_buckets = compute_scaled_resolution_buckets(
                resolution_buckets, reference_downscale_factor, scale_factors
            )
            if reference_downscale_factor > 1:
                logger.info(f"Processing reference videos at 1/{reference_downscale_factor} resolution...")
            if reference_temporal_scale_factor > 1:
                logger.info(
                    f"Temporally subsampling reference videos by {reference_temporal_scale_factor}x "
                    f"(VAE-aligned pattern)..."
                )

            with free_gpu_memory_context():
                compute_latents(
                    dataset_file=dataset_file,
                    main_media_column=roles["video"],
                    video_column=roles["reference_video"],
                    resolution_buckets=reference_buckets,
                    output_dir=str(output_base / "reference_latents"),
                    model_path=model_path,
                    batch_size=batch_size,
                    device=device,
                    vae_tiling=vae_tiling,
                    overwrite=overwrite,
                    temporal_subsample_factor=reference_temporal_scale_factor,
                )

    # --- Phase 2b: Masks (video_mask, audio_mask) — processed after video latents for alignment ---
    if "video_mask" in roles and has_video:
        compute_video_masks(
            dataset_file=dataset_file,
            mask_column=roles["video_mask"],
            latents_dir=str(output_base / "latents"),
            output_dir=str(output_base / "video_masks"),
            main_media_column=roles["video"],
            scale_factors=scale_factors,
        )

    # --- Phase 3: Audio VAE (audio, reference_audio) ---
    audio_roles_to_process = [
        ("audio", "audio_latents"),
        ("reference_audio", "reference_audio_latents"),
    ]
    active_audio_roles = [(role, subdir) for role, subdir in audio_roles_to_process if role in roles]

    if active_audio_roles:
        # Determine audio duration constraint: video bucket → max_duration, or explicit buckets
        max_audio_duration = None
        audio_duration_buckets = None
        if has_video and resolution_buckets:
            max_audio_duration = max(f for f, _h, _w in resolution_buckets) / 25.0
        elif audio_durations:
            audio_duration_buckets = audio_durations

        for role, output_subdir in active_audio_roles:
            with free_gpu_memory_context():
                compute_audio_latents(
                    dataset_file=dataset_file,
                    audio_column=roles[role],
                    output_dir=str(output_base / output_subdir),
                    model_path=model_path,
                    main_media_column=roles.get("video"),
                    max_duration=max_audio_duration,
                    duration_buckets=audio_duration_buckets,
                    device=device,
                    overwrite=overwrite,
                )

    # --- Phase 4: Audio masks (after audio latents exist for temporal alignment) ---
    if "audio_mask" in roles:
        audio_latents_source = output_base / "audio_latents"
        if audio_latents_source.exists():
            compute_audio_masks(
                dataset_file=dataset_file,
                mask_column=roles["audio_mask"],
                audio_latents_dir=str(audio_latents_source),
                output_dir=str(output_base / "audio_masks"),
                main_media_column=roles.get("video") or roles.get("audio"),
            )
        else:
            logger.warning("audio_mask column found but no audio_latents/ — run with audio first")

    # --- Decode for verification ---
    if decode:
        logger.info("Decoding latents for verification...")
        decoder = LatentsDecoder(model_path=model_path, device=device, vae_tiling=vae_tiling, with_audio=has_audio)
        if has_video:
            decoder.decode(output_base / "latents", output_base / "decoded_videos")
        if "reference_video" in roles and (output_base / "reference_latents").exists():
            decoder.decode(output_base / "reference_latents", output_base / "decoded_reference_videos")

    # --- Summary ---
    logger.info(f"Dataset preprocessing complete! Results saved to {output_base}")
    produced = [d.name for d in output_base.iterdir() if d.is_dir() and not d.name.startswith("decoded")]
    logger.info(f"Output directories: {', '.join(sorted(produced))}")


def _validate_dataset_file(dataset_path: str) -> None:
    """Validate that the dataset file exists and has the correct format."""
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")
    if not dataset_file.is_file():
        raise ValueError(f"Dataset path must be a file, not a directory: {dataset_file}")
    if dataset_file.suffix.lower() not in [".csv", ".json", ".jsonl"]:
        raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_file}")


def _resolve_columns(
    dataset_columns: set[str],
    video_column_override: str | None = None,
    caption_column_override: str | None = None,
) -> dict[str, str]:
    """Map canonical role names to actual dataset column names.
    Returns a dict of role → column_name for recognized roles found in the dataset.
    """
    roles: dict[str, str] = {}
    for col in dataset_columns:
        role = _LEGACY_ALIASES.get(col, col)
        if role in _KNOWN_ROLES:
            roles[role] = col

    if video_column_override and video_column_override in dataset_columns:
        roles["video"] = video_column_override
    if caption_column_override and caption_column_override in dataset_columns:
        roles["caption"] = caption_column_override

    return roles


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) with columns matching the convention table",
    ),
    resolution_buckets: str | None = typer.Option(
        default=None,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25"). '
        "Required when dataset has a video column.",
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX-2 checkpoint (.safetensors file)",
    ),
    text_encoder_path: str = typer.Option(
        ...,
        help="Path to Gemma text encoder directory",
    ),
    caption_column: str | None = typer.Option(
        default=None,
        help="Override: treat this column as 'caption' (default: auto-detect 'caption')",
    ),
    video_column: str | None = typer.Option(
        default=None,
        help="Override: treat this column as 'video' (default: auto-detect 'video' or 'media_path')",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for preprocessing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    output_dir: str | None = typer.Option(
        default=None,
        help="Output directory (defaults to .precomputed in dataset directory)",
    ),
    lora_trigger: str | None = typer.Option(
        default=None,
        help="Optional trigger word to prepend to each caption",
    ),
    decode: bool = typer.Option(
        default=False,
        help="Decode and save latents after encoding for verification",
    ),
    remove_llm_prefixes: bool = typer.Option(
        default=False,
        help="Remove LLM prefixes from captions",
    ),
    skip_audio: bool = typer.Option(
        default=False,
        help="Don't extract audio from video files (audio extraction is on by default)",
    ),
    audio_durations: str | None = typer.Option(
        default=None,
        help='Audio duration buckets in seconds for audio-only datasets (e.g. "2.0;4.0;8.0"). '
        "When set, audio files are trimmed to the best matching duration. "
        "Not needed when a video column is present (audio duration derived from video bucket).",
    ),
    with_audio: bool = typer.Option(
        default=False,
        hidden=True,
        help="[DEPRECATED: audio is now on by default, use --skip-audio to disable]",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the Gemma text encoder in 8-bit precision to save GPU memory",
    ),
    reference_downscale_factor: int = typer.Option(
        default=1,
        help="Downscale factor for reference video resolution (e.g., 2 = half resolution for IC-LoRA)",
    ),
    reference_temporal_scale_factor: int = typer.Option(
        default=1,
        help="Temporal subsampling factor for reference videos (e.g., 2 = half frame rate, "
        "VAE-aligned: keeps frame 0, then every Nth frame from frame 1 onwards)",
    ),
    overwrite: bool = typer.Option(
        default=False,
        help="Re-compute every item even if its output exists. Use when rerunning with "
        "changed parameters (different model, resolution, etc.) so stale outputs are replaced.",
    ),
) -> None:
    """Preprocess a media dataset for LTX-2 training.
    See module docstring for the convention table. Audio is auto-extracted from
    video files by default — use --skip-audio to disable.
    For multi-GPU preprocessing, invoke under ``accelerate launch`` -- each process
    will handle an interleaved shard of the dataset.
    """
    # Handle deprecated --with-audio flag
    if with_audio:
        logger.warning(
            "--with-audio is deprecated. Audio extraction is now on by default. Use --skip-audio to disable."
        )

    scale_factors = read_video_scale_factors(model_path)
    parsed_buckets = parse_resolution_buckets(resolution_buckets, scale_factors) if resolution_buckets else None

    if parsed_buckets and len(parsed_buckets) > 1:
        logger.warning("Using multiple resolution buckets. Training batch size must be 1.")

    if reference_downscale_factor < 1:
        raise typer.BadParameter("--reference-downscale-factor must be >= 1")

    if reference_temporal_scale_factor < 1:
        raise typer.BadParameter("--reference-temporal-scale-factor must be >= 1")

    parsed_audio_durations = None
    if audio_durations:
        parsed_audio_durations = [float(d) for d in audio_durations.split(";")]
        if any(d <= 0 for d in parsed_audio_durations):
            raise typer.BadParameter("All audio durations must be positive")

    preprocess_dataset(
        dataset_file=dataset_path,
        resolution_buckets=parsed_buckets,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        device=device,
        output_dir=output_dir,
        video_column=video_column,
        caption_column=caption_column,
        batch_size=batch_size,
        lora_trigger=lora_trigger,
        vae_tiling=vae_tiling,
        decode=decode,
        remove_llm_prefixes=remove_llm_prefixes,
        reference_downscale_factor=reference_downscale_factor,
        reference_temporal_scale_factor=reference_temporal_scale_factor,
        skip_audio=skip_audio,
        audio_durations=parsed_audio_durations,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
