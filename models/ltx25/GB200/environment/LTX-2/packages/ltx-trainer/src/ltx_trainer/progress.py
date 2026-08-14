"""Progress tracking for LTX training.
This module provides a unified progress display for training and validation sampling,
encapsulating all Rich progress bar logic in one place.
"""

from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class SamplingContext:
    """Context for validation sampling progress tracking.
    Provides a unified progress display showing current video and denoising step.
    Display format: "Sampling X/Y [████████████] step Z/W"
    The progress bar shows the denoising progress for the current video.
    """

    def __init__(self, progress: Progress | None, task: TaskID | None, num_prompts: int, num_steps: int):
        self._progress = progress
        self._task = task
        self._num_prompts = num_prompts
        self._num_steps = num_steps

    def start_video(self, video_idx: int) -> None:
        """Start tracking a new video (resets step progress)."""
        if self._progress is None or self._task is None:
            return
        # Reset task for new video: completed=0, total=num_steps
        self._progress.reset(self._task, total=self._num_steps)
        self._progress.update(
            self._task,
            completed=0,
            video=f"{video_idx + 1}/{self._num_prompts}",
            info=f"step 0/{self._num_steps}",
        )

    def advance_step(self) -> None:
        """Advance the denoising step by one."""
        if self._progress is None or self._task is None:
            return
        self._progress.advance(self._task)
        completed = int(self._progress.tasks[self._task].completed)
        self._progress.update(self._task, info=f"step {completed}/{self._num_steps}")

    def cleanup(self) -> None:
        """Hide sampling task when done."""
        if self._progress is None or self._task is None:
            return
        self._progress.update(self._task, visible=False)


class TrainingProgress:
    """Manages Rich progress display for training and validation.
    This class encapsulates all progress bar logic, providing a clean interface
    for the trainer to update progress without dealing with Rich internals.
    Usage:
        with TrainingProgress(enabled=True, total_steps=1000) as progress:
            for step in range(1000):
                # ... training step ...
                progress.update_training(loss=0.1, lr=1e-4, step_time=0.5)
                if should_validate:
                    sampling_ctx = progress.start_sampling(num_prompts=3, num_steps=30)
                    sampler = ValidationSampler(..., sampling_context=sampling_ctx)
                    for prompt_idx, prompt in enumerate(prompts):
                        sampling_ctx.start_video(prompt_idx)
                        sampler.generate(...)
                    sampling_ctx.cleanup()
    """

    def __init__(self, enabled: bool, total_steps: int):
        """Initialize progress tracking.
        Args:
            enabled: Whether to display progress bars (False for non-main processes)
            total_steps: Total number of training steps
        """
        self._enabled = enabled
        self._total_steps = total_steps
        self._train_task: TaskID | None = None

        if not enabled:
            self._progress = None
            return

        # Single Progress instance with flexible columns
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[video]}", style="magenta"),
            BarColumn(bar_width=40, style="blue"),
            TextColumn("{task.fields[info]}", style="cyan"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(compact=True),
        )

    def __enter__(self) -> "TrainingProgress":
        """Enter the progress context, starting the live display."""
        if self._progress is not None:
            self._progress.__enter__()
            self._train_task = self._progress.add_task(
                "Training",
                total=self._total_steps,
                video=f"0/{self._total_steps}",
                info="Starting...",
            )
        return self

    def __exit__(self, *args) -> None:
        """Exit the progress context, stopping the live display."""
        if self._progress is not None:
            self._progress.__exit__(*args)

    @property
    def enabled(self) -> bool:
        """Whether progress display is enabled."""
        return self._enabled

    def update_training(
        self,
        *,
        loss: float,
        lr: float,
        step_time: float,
        advance: bool = True,
    ) -> None:
        """Update the training progress display.
        Args:
            loss: Current training loss
            lr: Current learning rate
            step_time: Time taken for this step in seconds
            advance: Whether to advance the progress by one step
        """
        if self._progress is None or self._train_task is None:
            return

        info = f"Loss: {loss:.4f} | LR: {lr:.2e} | {step_time:.2f}s/step"
        self._progress.update(
            self._train_task,
            advance=1 if advance else 0,
            info=info,
        )
        # Update step count in video column
        completed = int(self._progress.tasks[self._train_task].completed)
        self._progress.update(self._train_task, video=f"{completed}/{self._total_steps}")

    def start_sampling(self, num_prompts: int, num_steps: int) -> SamplingContext:
        """Start validation sampling progress tracking.
        Creates a task that shows current video and denoising step progress.
        Format: "Sampling X/Y [████████████] step Z/W"
        Args:
            num_prompts: Number of validation prompts to sample
            num_steps: Number of denoising steps per sample
        Returns:
            SamplingContext for tracking progress (no-op if progress is disabled)
        """
        if self._progress is None:
            # Return a no-op context when progress is disabled
            return SamplingContext(
                progress=None,
                task=None,
                num_prompts=num_prompts,
                num_steps=num_steps,
            )

        task = self._progress.add_task(
            "Sampling",
            total=num_steps,
            completed=0,
            video=f"0/{num_prompts}",
            info=f"step 0/{num_steps}",
        )

        return SamplingContext(
            progress=self._progress,
            task=task,
            num_prompts=num_prompts,
            num_steps=num_steps,
        )
