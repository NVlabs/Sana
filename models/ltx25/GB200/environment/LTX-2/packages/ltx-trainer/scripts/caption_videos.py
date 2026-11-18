#!/usr/bin/env python3

"""
Auto-caption videos with audio using multimodal models.
Backends:
- Qwen3-Omni-30B-A3B-Thinking via a local vLLM HTTP server (default,
  ``qwen_omni``). Launch the server once with ``scripts/serve_captioner.py``.
- Gemini Flash 3.5 via Google's API (``gemini_flash``).
The paths in the output file are RELATIVE to the output file's directory,
making the dataset portable.
Basic usage:
    # Launch the captioner server once (separate terminal)
    uv run python scripts/serve_captioner.py
    # Caption a directory
    caption_videos.py videos_dir/ --output captions.json
    # Caption a single video with a custom prompt
    caption_videos.py video.mp4 --output cap.json --instruction "Describe in detail."
Advanced usage:
    # Use Gemini Flash 3.5 (cloud, requires GEMINI_API_KEY)
    caption_videos.py videos_dir/ --captioner-type gemini_flash
    # Gemini with parallel workers
    caption_videos.py videos_dir/ --captioner-type gemini_flash --num-workers 5
    # Talk to a remote vLLM server
    caption_videos.py videos_dir/ --vllm-url http://192.168.1.10:8001/v1
    # Enable Qwen3 chain-of-thought (slower, more detail)
    caption_videos.py videos_dir/ --enable-thinking
"""

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ltx_trainer.captioning import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_VLLM_BASE_URL,
    CaptionerType,
    MediaCaptioningModel,
    create_captioner,
)

VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS
SAVE_INTERVAL = 5

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Auto-caption videos with audio using multimodal models.",
)


class OutputFormat(str, Enum):
    """Available output formats for captions."""

    TXT = "txt"  # Separate files for captions and video paths, one caption / video path per line
    CSV = "csv"  # CSV file with video path and caption columns
    JSON = "json"  # JSON file with video paths as keys and captions as values
    JSONL = "jsonl"  # JSON Lines file with one JSON object per line


def caption_media(
    input_path: Path,
    output_path: Path,
    captioner: MediaCaptioningModel,
    extensions: list[str],
    recursive: bool,
    fps: int,
    output_format: OutputFormat,
    override: bool,
    num_workers: int = 1,
) -> None:
    """Caption videos and images using the provided captioning model.
    Args:
        input_path: Path to input video file or directory
        output_path: Path to output caption file
        captioner: Media captioning model
        extensions: List of media file extensions to include
        recursive: Whether to search subdirectories recursively
        fps: Frames per second to sample from videos (ignored for images)
        output_format: Format to save the captions in
        override: Whether to override existing captions
        num_workers: Number of parallel workers (only for cloud-based captioners like Gemini)
    """

    # Get list of media files to process
    media_files = _get_media_files(input_path, extensions, recursive)

    if not media_files:
        console.print("[bold yellow]No media files found to process.[/]")
        return

    console.print(f"Found [bold]{len(media_files)}[/] media files to process.")

    # Load existing captions and determine which files need processing
    base_dir = output_path.parent.resolve()
    existing_captions = _load_existing_captions(output_path, output_format)
    existing_abs_paths = {str((base_dir / p).resolve()) for p in existing_captions}

    if override:
        media_to_process = media_files
    else:
        media_to_process = [f for f in media_files if str(f.resolve()) not in existing_abs_paths]
        if skipped := len(media_files) - len(media_to_process):
            console.print(f"[bold yellow]Skipping {skipped} media that already have captions.[/]")

    if not media_to_process:
        console.print("[bold yellow]All media already have captions. Use --override to recaption.[/]")
        return

    if num_workers > 1:
        console.print(f"Running with [bold cyan]{num_workers}[/] parallel workers.")

    captions = existing_captions.copy()
    successfully_captioned = 0
    completed_since_save = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    def process_one(media_file: Path) -> tuple[str, str]:
        """Caption a single media file and return (relative_path, caption)."""
        caption = captioner.caption(
            path=media_file,
            fps=fps,
        )
        # Don't resolve the file itself, so a symlinked clip keeps its logical path under the
        # dataset dir instead of jumping to its (possibly external) link target.
        rel_path = str((media_file.parent.resolve() / media_file.name).relative_to(base_dir))
        return rel_path, caption

    with progress:
        task = progress.add_task(
            f"Captioning (workers: {num_workers})" if num_workers > 1 else "Captioning",
            total=len(media_to_process),
        )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_one, f): f for f in media_to_process}

            for future in as_completed(futures):
                media_file = futures[future]
                progress.update(task, description=f"Captioning [bold blue]{media_file.name}[/]")

                try:
                    rel_path, caption = future.result()

                    captions[rel_path] = caption
                    successfully_captioned += 1
                    completed_since_save += 1

                    if completed_since_save >= SAVE_INTERVAL:
                        _save_captions(captions, output_path, output_format)
                        completed_since_save = 0

                except Exception as e:
                    console.print(f"[bold red]Error captioning {media_file.name}: {e}[/]")

                progress.advance(task)

    # Final save with everything accumulated
    _save_captions(captions, output_path, output_format)

    # Print summary
    console.print(
        f"[bold green]✓[/] Captioned [bold]{successfully_captioned}/{len(media_to_process)}[/] media successfully.",
    )


def _get_media_files(
    input_path: Path,
    extensions: list[str] = MEDIA_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    """Get all media files from the input path."""
    input_path = Path(input_path)
    # Normalize extensions to lowercase without dots
    extensions_set = {ext.lower().lstrip(".") for ext in extensions}

    if input_path.is_file():
        # If input is a file, check if it has a valid extension
        if input_path.suffix.lstrip(".").lower() in extensions_set:
            return [input_path]
        else:
            typer.echo(f"Warning: {input_path} is not a recognized media file. Skipping.")
            return []
    elif input_path.is_dir():
        # Find all files and filter by extension case-insensitively
        glob_pattern = "**/*" if recursive else "*"
        media_files = [
            f for f in input_path.glob(glob_pattern) if f.is_file() and f.suffix.lstrip(".").lower() in extensions_set
        ]
        return sorted(media_files)
    else:
        typer.echo(f"Error: {input_path} does not exist.")
        raise typer.Exit(code=1)


def _save_captions(
    captions: dict[str, str],
    output_path: Path,
    format_type: OutputFormat,
) -> None:
    """Save captions to a file in the specified format.
    Args:
        captions: Dictionary mapping media paths to captions
        output_path: Path to save the output file
        format_type: Format to save the captions in
    """
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Saving captions...[/]")

    match format_type:
        case OutputFormat.TXT:
            # Create two separate files for captions and media paths
            captions_file = output_path.with_stem(f"{output_path.stem}_captions")
            paths_file = output_path.with_stem(f"{output_path.stem}_paths")

            with captions_file.open("w", encoding="utf-8") as f:
                for caption in captions.values():
                    f.write(f"{caption}\n")

            with paths_file.open("w", encoding="utf-8") as f:
                for media_path in captions:
                    f.write(f"{media_path}\n")

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{captions_file}[/]")
            console.print(f"[bold green]✓[/] Media paths saved to [cyan]{paths_file}[/]")

        case OutputFormat.CSV:
            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["caption", "media_path"])
                for media_path, caption in captions.items():
                    writer.writerow([caption, media_path])

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")

        case OutputFormat.JSON:
            # Format as list of dictionaries with caption and media_path keys
            json_data = [{"caption": caption, "media_path": media_path} for media_path, caption in captions.items()]

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")

        case OutputFormat.JSONL:
            with output_path.open("w", encoding="utf-8") as f:
                for media_path, caption in captions.items():
                    f.write(json.dumps({"caption": caption, "media_path": media_path}, ensure_ascii=False) + "\n")

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")

        case _:
            raise ValueError(f"Unsupported output format: {format_type}")


def _load_existing_captions(  # noqa: PLR0912
    output_path: Path,
    format_type: OutputFormat,
) -> dict[str, str]:
    """Load existing captions from a file.
    Args:
        output_path: Path to the captions file
        format_type: Format of the captions file
    Returns:
        Dictionary mapping media paths to captions, or empty dict if file doesn't exist
    """
    if not output_path.exists():
        return {}

    console.print(f"[bold blue]Loading existing captions from [cyan]{output_path}[/]...[/]")

    existing_captions = {}

    try:
        match format_type:
            case OutputFormat.TXT:
                # For TXT format, we have two separate files
                captions_file = output_path.with_stem(f"{output_path.stem}_captions")
                paths_file = output_path.with_stem(f"{output_path.stem}_paths")

                if captions_file.exists() and paths_file.exists():
                    captions = captions_file.read_text(encoding="utf-8").splitlines()
                    paths = paths_file.read_text(encoding="utf-8").splitlines()

                    if len(captions) == len(paths):
                        existing_captions = dict(zip(paths, captions, strict=False))

            case OutputFormat.CSV:
                with output_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    # Skip header
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            caption, media_path = row[0], row[1]
                            existing_captions[media_path] = caption

            case OutputFormat.JSON:
                with output_path.open("r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    for item in json_data:
                        if "caption" in item and "media_path" in item:
                            existing_captions[item["media_path"]] = item["caption"]

            case OutputFormat.JSONL:
                with output_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        if "caption" in item and "media_path" in item:
                            existing_captions[item["media_path"]] = item["caption"]

            case _:
                raise ValueError(f"Unsupported output format: {format_type}")

        console.print(f"[bold green]✓[/] Loaded [bold]{len(existing_captions)}[/] existing captions")
        return existing_captions

    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not load existing captions: {e}[/]")
        return {}


@app.command()
def main(  # noqa: PLR0913
    input_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to input video/image file or directory containing media files",
        exists=True,
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Path to output file for captions. Format determined by file extension.",
    ),
    captioner_type: CaptionerType = typer.Option(  # noqa: B008
        CaptionerType.QWEN_OMNI,
        "--captioner-type",
        "-c",
        help="Type of captioner to use. Valid values: 'qwen_omni' (local), 'gemini_flash' (API)",
        case_sensitive=False,
    ),
    vllm_url: str = typer.Option(
        DEFAULT_VLLM_BASE_URL,
        "--vllm-url",
        help=(
            "Base URL of the vLLM OpenAI-compatible server (qwen_omni only). "
            "Launch the server with `uv run python scripts/serve_captioner.py`."
        ),
    ),
    vllm_model: str = typer.Option(
        DEFAULT_QWEN_MODEL,
        "--vllm-model",
        help="Served model identifier on the vLLM server (qwen_omni only).",
    ),
    enable_thinking: bool = typer.Option(
        False,
        "--enable-thinking/--no-thinking",
        help=(
            "Let Qwen3-Omni produce a <think>...</think> chain-of-thought before the caption. "
            "Off by default: ~5x slower with marginal quality benefit and occasional hallucinations."
        ),
    ),
    max_tokens: int = typer.Option(
        4096,
        "--max-tokens",
        help="Maximum new tokens to generate per caption (qwen_omni only).",
    ),
    instruction: str | None = typer.Option(
        None,
        "--instruction",
        "-i",
        help="Custom instruction for the captioning model. If not provided, uses an appropriate default.",
    ),
    extensions: str = typer.Option(
        ",".join(MEDIA_EXTENSIONS),
        "--extensions",
        "-e",
        help="Comma-separated list of media file extensions to process",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search for media files in subdirectories recursively",
    ),
    fps: int = typer.Option(
        2,
        "--fps",
        "-f",
        help=(
            "Frames per second to sample from videos. 2 is a typical default; "
            "lower values use less compute per video. Ignored for images and for the "
            "Gemini backend (which decides its own sampling rate)."
        ),
    ),
    override: bool = typer.Option(
        False,
        "--override",
        help="Whether to override existing captions for media",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        help="API key for Gemini Flash (can also use GOOGLE_API_KEY or GEMINI_API_KEY env var)",
    ),
    num_workers: int = typer.Option(
        1,
        "--num-workers",
        "-w",
        min=1,
        max=10,
        help=(
            "Number of parallel workers for captioning (1-10). "
            "Values above 1 are only supported for cloud-based captioners (gemini_flash). "
            "Using multiple workers with a local model will raise an error."
        ),
    ),
) -> None:
    """Auto-caption videos with audio using multimodal models.
    Backends:
    - ``qwen_omni`` (default): Qwen3-Omni-30B-A3B-Thinking via a local vLLM
      HTTP server. Launch the server once in a separate terminal with
      ``uv run python scripts/serve_captioner.py``. The server stays loaded
      across script invocations.
    - ``gemini_flash``: Google Gemini (``gemini-3.5-flash``) via the google-genai SDK.
      Auth is automatic -- ``GEMINI_API_KEY``/``GOOGLE_API_KEY`` for the Developer API,
      or Google Cloud credentials (gcloud / service account) for Vertex AI with no env vars.
    The paths in the output file will be relative to the output file's directory.
    Examples:
        # Caption videos using the local vLLM server (default)
        caption_videos.py videos_dir/ -o captions.json
        # Point at a remote vLLM server
        caption_videos.py videos_dir/ -o captions.json --vllm-url http://other-host:8001/v1
        # Caption using Gemini Flash 3.5
        caption_videos.py videos_dir/ -o captions.json -c gemini_flash
        # Caption with custom instruction
        caption_videos.py video.mp4 -o captions.json -i "Describe this video in detail"
    """

    # Parallel workers are only supported for the cloud Gemini backend; qwen_omni
    # drives a single shared vLLM server and is captioned serially from here.
    if num_workers > 1 and captioner_type != CaptionerType.GEMINI_FLASH:
        console.print(
            "[bold red]Error:[/] --num-workers > 1 is only supported with "
            "[bold]--captioner-type gemini_flash[/]. Use --num-workers 1 (default) "
            "for the qwen_omni backend."
        )
        raise typer.Exit(code=1)

    # Parse extensions
    ext_list = [ext.strip() for ext in extensions.split(",")]

    # Determine output path and format
    if output is None:
        output_format = OutputFormat.JSON
        if input_path.is_file():  # noqa: SIM108
            # Default to a JSON file with the same name as the input media
            output = input_path.with_suffix(".dataset.json")
        else:
            # Default to a JSON file in the input directory
            output = input_path / "dataset.json"
    else:
        # Determine format from file extension
        output_format = OutputFormat(Path(output).suffix.lstrip(".").lower())

    # Ensure output path is absolute
    output = Path(output).resolve()
    console.print(f"Output will be saved to [bold blue]{output}[/]")

    with console.status("Initializing captioner...", spinner="dots"):
        if captioner_type == CaptionerType.QWEN_OMNI:
            captioner = create_captioner(
                captioner_type=captioner_type,
                base_url=vllm_url,
                model=vllm_model,
                instruction=instruction,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
        elif captioner_type == CaptionerType.GEMINI_FLASH:
            captioner = create_captioner(
                captioner_type=captioner_type,
                api_key=api_key,
                instruction=instruction,
            )
        else:
            raise ValueError(f"Unsupported captioner type: {captioner_type}")

        console.print(f"[bold green]✓[/] {captioner_type.value} captioner ready")

    # Caption media files
    caption_media(
        input_path=input_path,
        output_path=output,
        captioner=captioner,
        extensions=ext_list,
        recursive=recursive,
        fps=fps,
        output_format=output_format,
        override=override,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    app()
