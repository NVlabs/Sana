"""Display utilities for training configuration.
This module provides formatted console output for LtxTrainerConfig.
"""

from rich import box
from rich.console import Console
from rich.table import Table

from ltx_trainer.config import LtxTrainerConfig


def print_config(config: LtxTrainerConfig) -> None:
    """Print configuration as a nicely formatted table with sections."""

    def fmt(v: object, max_len: int = 55) -> str:
        """Format any value for display."""
        if v is None:
            return "[dim]—[/]"
        if isinstance(v, bool):
            return "[green]✓[/]" if v else "[dim]✗[/]"
        if isinstance(v, (list, tuple)):
            if not v:
                return "[dim]—[/]"
            return ", ".join(str(x) for x in v)
        s = str(v)
        return s[: max_len - 3] + "..." if len(s) > max_len else s

    cfg = config
    opt = cfg.optimization
    val = cfg.validation
    accel = cfg.acceleration

    # Build sections: list of (section_title, [(key, value), ...])
    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "🎬 Model",
            [
                ("Base", fmt(cfg.model.model_path)),
                ("Text Encoder", fmt(cfg.model.text_encoder_path) or "[dim]Built-in[/]"),
                ("Training Mode", f"[bold green]{cfg.model.training_mode.upper()}[/]"),
                ("Load Checkpoint", fmt(cfg.model.load_checkpoint) if cfg.model.load_checkpoint else "[dim]—[/]"),
            ],
        ),
    ]

    if cfg.lora:
        sections.append(
            (
                "🔗 LoRA",
                [
                    ("Rank / Alpha", f"{cfg.lora.rank} / {cfg.lora.alpha}"),
                    ("Dropout", str(cfg.lora.dropout)),
                    ("Target Modules", fmt(cfg.lora.target_modules)),
                ],
            )
        )

    # Strategy section - include strategy-specific fields
    strategy_items: list[tuple[str, str]] = [("Name", cfg.training_strategy.name)]
    if hasattr(cfg.training_strategy, "with_audio"):
        strategy_items.append(("Audio", fmt(cfg.training_strategy.with_audio)))
    if hasattr(cfg.training_strategy, "first_frame_conditioning_p"):
        strategy_items.append(("First Frame Cond P", str(cfg.training_strategy.first_frame_conditioning_p)))

    sections.append(("🎯 Strategy", strategy_items))

    sections.extend(
        [
            (
                "⚡ Optimization",
                [
                    ("Steps", f"[bold]{opt.steps:,}[/]"),
                    ("Learning Rate", f"{opt.learning_rate:.2e}"),
                    ("Batch Size", str(opt.batch_size)),
                    ("Grad Accumulation", str(opt.gradient_accumulation_steps)),
                    ("Optimizer", opt.optimizer_type),
                    ("Scheduler", opt.scheduler_type),
                    ("Max Grad Norm", str(opt.max_grad_norm)),
                    ("Grad Checkpointing", fmt(opt.enable_gradient_checkpointing)),
                ],
            ),
            (
                "🚀 Acceleration",
                [
                    ("Mixed Precision", accel.mixed_precision_mode or "[dim]—[/]"),
                    ("Quantization", str(accel.quantization) if accel.quantization else "[dim]—[/]"),
                    ("Text Encoder 8bit", fmt(accel.load_text_encoder_in_8bit)),
                    ("Optimizer CPU Offload", fmt(accel.offload_optimizer_during_validation)),
                ],
            ),
            (
                "🎥 Validation",
                [
                    ("Prompts", f"{len(val.prompts)} prompt(s)" if val.prompts else "[dim]—[/]"),
                    ("Interval", f"Every {val.interval} steps" if val.interval else "[dim]Disabled[/]"),
                    ("Video Dims", f"{val.video_dims[0]}x{val.video_dims[1]}, {val.video_dims[2]} frames"),
                    ("Frame Rate", f"{val.frame_rate} fps"),
                    ("Inference Steps", str(val.inference_steps)),
                    ("CFG Scale", f"video={val.video_cfg_scale}; audio={val.audio_cfg_scale}"),
                    (
                        "STG Scale",
                        f"video={val.video_stg_scale}; audio={val.audio_stg_scale}; blocks={fmt(val.stg_blocks)}"
                        if val.video_stg_scale > 0 or val.audio_stg_scale > 0
                        else "[dim]Disabled[/]",
                    ),
                    ("Guidance Rescale", str(val.guidance_rescale)),
                    (
                        "Modality Guidance",
                        f"video={val.video_modality_guidance_scale}; audio={val.audio_modality_guidance_scale}",
                    ),
                    ("Seed", str(val.seed)),
                ],
            ),
            (
                "📂 Data & Output",
                [
                    ("Dataset", fmt(cfg.data.preprocessed_data_root)),
                    ("Dataloader Workers", str(cfg.data.num_dataloader_workers)),
                    ("Output Dir", fmt(cfg.output_dir)),
                    ("Seed", str(cfg.seed)),
                ],
            ),
            (
                "🔌 Integrations",
                [
                    (
                        "Checkpoints",
                        f"Every {cfg.checkpoints.interval} steps (keep {cfg.checkpoints.keep_last_n})"
                        if cfg.checkpoints.interval
                        else "[dim]Disabled[/]",
                    ),
                    ("W&B", f"{cfg.wandb.project}" if cfg.wandb.enabled else "[dim]Disabled[/]"),
                    ("HF Hub", cfg.hub.hub_model_id if cfg.hub.push_to_hub else "[dim]Disabled[/]"),
                ],
            ),
        ]
    )

    # Build table with section headers
    table = Table(
        title="[bold]⚙️  Training Configuration[/]",
        show_header=False,
        box=box.ROUNDED,
        border_style="bright_blue",
        padding=(0, 1),
        title_style="bold bright_blue",
    )
    table.add_column("Key", style="white", width=20)
    table.add_column("Value", style="cyan")

    for i, (section_title, items) in enumerate(sections):
        if i > 0:
            table.add_row("", "")  # Blank line between sections
        table.add_row(f"[bold yellow]{section_title}[/]", "")
        for key, value in items:
            table.add_row(f"  {key}", value)

    console = Console()
    console.print()
    console.print(table)
    console.print()
