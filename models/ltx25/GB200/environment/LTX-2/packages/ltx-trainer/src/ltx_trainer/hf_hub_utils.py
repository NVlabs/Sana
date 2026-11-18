import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import imageio
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
from rich.progress import Progress, SpinnerColumn, TextColumn

from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig


def push_to_hub(
    weights_path: Path,
    sampled_videos_paths: Optional[List[Path]],
    config: LtxTrainerConfig,
) -> None:
    """Push the trained LoRA weights to HuggingFace Hub."""
    if not config.hub.hub_model_id:
        logger.warning("⚠️ HuggingFace hub_model_id not specified, skipping push to hub")
        return

    api = HfApi()

    # Save original progress bar state
    original_progress_state = are_progress_bars_disabled()
    disable_progress_bars()  # Disable during our custom progress tracking

    try:
        # Try to create repo if it doesn't exist
        try:
            repo = create_repo(
                repo_id=config.hub.hub_model_id,
                repo_type="model",
                exist_ok=True,  # Don't raise error if repo exists
            )
            repo_id = repo.repo_id
            logger.info(f"🤗 Successfully created HuggingFace model repository at: {repo.url}")
        except Exception as e:
            logger.error(f"❌ Failed to create HuggingFace model repository: {e}")
            return

        # Create a single temporary directory for all files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                try:
                    # Copy weights
                    task_copy = progress.add_task("Copying weights...", total=None)
                    weights_dest = temp_path / weights_path.name
                    shutil.copy2(weights_path, weights_dest)
                    progress.update(task_copy, description="✓ Weights copied")

                    # Create model card and save samples
                    task_card = progress.add_task("Creating model card and samples...", total=None)
                    _create_model_card(
                        output_dir=temp_path,
                        videos=sampled_videos_paths,
                        config=config,
                    )
                    progress.update(task_card, description="✓ Model card and samples created")

                    # Upload everything at once
                    task_upload = progress.add_task("Pushing files to HuggingFace Hub...", total=None)
                    api.upload_folder(
                        folder_path=str(temp_path),
                        repo_id=repo_id,
                        repo_type="model",
                    )
                    progress.update(task_upload, description="✓ Files pushed to HuggingFace Hub")
                    logger.info("✅ Successfully pushed files to HuggingFace Hub")

                except Exception as e:
                    logger.error(f"❌ Failed to process and push files to HuggingFace Hub: {e}")
                    raise  # Re-raise to handle in outer try block

    finally:
        # Restore original progress bar state
        if not original_progress_state:
            enable_progress_bars()


def convert_video_to_gif(video_path: Path, output_path: Path) -> None:
    """Convert a video file to GIF format."""
    try:
        # Read the video file
        reader = imageio.get_reader(str(video_path))
        fps = reader.get_meta_data()["fps"]

        # Write GIF file with infinite loop
        writer = imageio.get_writer(
            str(output_path),
            fps=min(fps, 15),  # Cap FPS at 15 for reasonable file size
            loop=0,  # 0 means infinite loop
        )

        for frame in reader:
            writer.append_data(frame)

        writer.close()
        reader.close()
    except Exception as e:
        logger.error(f"Failed to convert video to GIF: {e}")


def _create_model_card(
    output_dir: Union[str, Path],
    videos: Optional[List[Path]],
    config: LtxTrainerConfig,
) -> Path:
    """Generate and save a model card for the trained model."""

    repo_id = config.hub.hub_model_id
    pretrained_model_name_or_path = config.model.model_path
    validation_prompts = config.validation.prompts
    output_dir = Path(output_dir)
    template_path = Path(__file__).parent.parent.parent / "templates" / "model_card.md"

    # Read the template
    template = template_path.read_text()

    # Get model name from repo_id
    model_name = repo_id.split("/")[-1]

    # Get base model information
    base_model_link = str(pretrained_model_name_or_path)
    model_path_str = str(pretrained_model_name_or_path)
    is_url = model_path_str.startswith(("http://", "https://"))

    # For URLs, extract the filename from the URL. For local paths, use the filename stem
    base_model_name = model_path_str.split("/")[-1] if is_url else Path(pretrained_model_name_or_path).name

    # Format validation prompts and create grid layout
    prompts_text = ""
    sample_grid = []

    if validation_prompts and videos:
        prompts_text = "Example prompts used during validation:\n\n"

        # Create samples directory
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True, parents=True)

        # Process videos and create cells
        cells = []
        for i, (prompt, video) in enumerate(zip(validation_prompts, videos, strict=False)):
            if video.exists():
                # Add prompt to text section
                prompts_text += f"- `{prompt}`\n"

                # Convert video to GIF
                gif_path = samples_dir / f"sample_{i}.gif"
                try:
                    convert_video_to_gif(video, gif_path)

                    # Create grid cell with collapsible description
                    cell = (
                        f"![example{i + 1}](./samples/sample_{i}.gif)"
                        "<br>"
                        '<details style="max-width: 300px; margin: auto;">'
                        f"<summary>Prompt</summary>"
                        f"{prompt}"
                        "</details>"
                    )
                    cells.append(cell)
                except Exception as e:
                    logger.error(f"Failed to process video {video}: {e}")

        # Calculate optimal grid dimensions
        num_cells = len(cells)
        if num_cells > 0:
            # Aim for a roughly square grid, with max 4 columns
            num_cols = min(4, num_cells)
            num_rows = (num_cells + num_cols - 1) // num_cols  # Ceiling division

            # Create grid rows
            for row in range(num_rows):
                start_idx = row * num_cols
                end_idx = min(start_idx + num_cols, num_cells)
                row_cells = cells[start_idx:end_idx]
                # Properly format the row with table markers and exact number of cells
                formatted_row = "| " + " | ".join(row_cells) + " |"
                sample_grid.append(formatted_row)

    # Join grid rows with just the content, no headers needed
    grid_text = "\n".join(sample_grid) if sample_grid else ""

    # Fill in the template
    model_card_content = template.format(
        base_model=base_model_name,
        base_model_link=base_model_link,
        model_name=model_name,
        training_type="LoRA fine-tuning" if config.model.training_mode == "lora" else "Full model fine-tuning",
        training_steps=config.optimization.steps,
        learning_rate=config.optimization.learning_rate,
        batch_size=config.optimization.batch_size,
        validation_prompts=prompts_text,
        sample_grid=grid_text,
    )

    # Save the model card directly
    model_card_path = output_dir / "README.md"
    model_card_path.write_text(model_card_content)

    return model_card_path
