"""Runtime-independent, one-publication adapter to the native H3 renderer."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StreamRenderInput:
    """One initialization or one ordered semantic batch.

    Attributes:
        world_id: New identity requests a fresh renderer state.
        reference: CPU uint8 HWC RGB appearance image, only until acknowledged.
        prompt: Explicit appearance description on initialization; None thereafter.
        seed: Native random seed used only when starting a world.
        first_frame: Zero-based first semantic frame index, with no gaps.
        semantics: CPU uint8 (T, 768, 1344, 3), exactly frames_wanted images.
            None initializes conditioning without advancing the semantic timeline.
    """

    world_id: int
    reference: np.ndarray | None = None
    prompt: str | None = None
    seed: int = 0
    first_frame: int = 0
    semantics: np.ndarray | None = None


@dataclass(frozen=True)
class StreamRenderResult:
    """Completed renderer facts, including demand for the next step.

    Attributes:
        world_id: Identity successfully applied by the renderer.
        chunk_index: Zero-based completed render chunk, -1 after initialization.
        next_frame: First unconsumed semantic frame index.
        frames_wanted: Exact native semantic input count for the next chunk.
        frames: CPU uint8 (T, 768, 1344, 3) video, or None before publication.
    """

    world_id: int
    chunk_index: int
    next_frame: int
    frames_wanted: int
    frames: np.ndarray | None


class InvalidStream(ValueError):
    """The input does not continue the renderer's current semantic timeline."""


class StreamRenderModel:
    """Keep weights resident and advance one native semantic/render chunk per call.

    New identities require an anchor; repeated identities consume only the next
    requested batch. InvalidStream rejects malformed timelines. Backend errors
    propagate without committing progress and require an explicit new world.
    """

    def __init__(self):
        self.backend = None
        self.world_id = None
        self.next_frame = 0
        self.chunk_index = -1
        self.failed = False

    def load(self, source_root: Path, assets_path: Path) -> None:
        """Load resolved local source and weights inside an existing worker group."""
        from streamrender_h3_backend import StreamRenderBackend

        self.backend = StreamRenderBackend()
        self.backend.load(source_root, assets_path)

    def reset(self) -> None:
        """Release per-drive caches and counters while keeping weights resident."""
        if self.backend is not None:
            self.backend.reset()
        self.world_id = None
        self.next_frame = 0
        self.chunk_index = -1
        self.failed = False

    def generate(self, input: StreamRenderInput) -> StreamRenderResult:
        """Prepare a new world or generate one chunk; preserve native exceptions."""
        if self.backend is None:
            raise RuntimeError("load() has not run")
        if input.world_id != self.world_id:
            if input.reference is None or not input.prompt or not input.prompt.strip():
                raise InvalidStream(
                    "A new world needs an explicit reference image and prompt"
                )
            if input.semantics is not None or input.first_frame != 0:
                raise InvalidStream(
                    "Initialize the world before submitting semantic frames"
                )
            # The request explicitly starts a new world; no failure resets it silently.
            self.reset()
            try:
                self.backend.start(input.reference, input.prompt, input.seed)
            except Exception:
                self.failed = True
                raise
            self.world_id = input.world_id
        else:
            if self.failed:
                raise InvalidStream("A failed renderer must be explicitly restarted")
            if input.reference is not None or input.prompt is not None:
                raise InvalidStream("Reference image and prompt are fixed for a world")
            if input.first_frame != self.next_frame:
                raise InvalidStream(
                    "Missing, duplicated, or out-of-order semantic frames"
                )
            count = self.backend.frames_wanted()
            if input.semantics is None or input.semantics.shape != (
                count,
                768,
                1344,
                3,
            ):
                raise InvalidStream(f"Expected {count} semantic frames at 1344x768")
            if input.semantics.dtype != np.uint8:
                raise InvalidStream("Semantic colors must be lossless uint8 RGB")
            try:
                frames = self.backend.render(input.semantics)
            except Exception:
                # Native encoders may have advanced; retry is not assumed safe.
                self.failed = True
                raise
            self.next_frame += count
            self.chunk_index += 1
            return StreamRenderResult(
                world_id=self.world_id,
                chunk_index=self.chunk_index,
                next_frame=self.next_frame,
                frames_wanted=self.backend.frames_wanted(),
                frames=frames,
            )
        return StreamRenderResult(
            world_id=self.world_id,
            chunk_index=-1,
            next_frame=0,
            frames_wanted=self.backend.frames_wanted(),
            frames=None,
        )
