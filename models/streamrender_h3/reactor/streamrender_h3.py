"""Render uploaded semantic video through the native H3 streaming pipeline."""

import asyncio
import os
from pathlib import Path

import av
from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    TrackPayload,
    UploadedFile,
    connected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner

from streamrender_h3_images import decode_reference
from streamrender_h3_model import (
    StreamRenderInput,
    StreamRenderModel,
    StreamRenderResult,
)
from streamrender_h3_types import StateUpdate, StreamRenderOutput, StreamRenderState
from streamrender_h3_video import SemanticVideo


class StreamRenderH3(ReactorApp):
    """Render recorded semantic video with an explicit reference image and prompt."""

    fps = 24
    state: StreamRenderState

    def load(self, config_path: Path | None) -> None:
        world_size = int(os.environ.get("STREAMRENDER_GPUS", "4"))
        if world_size not in (4, 8):
            raise ValueError("The released Qwen checkpoint requires 4 or 8 GPUs")
        source_root = Path(os.environ["STREAMRENDER_SOURCE"])
        assets_path = Path(
            os.environ.get(
                "STREAMRENDER_ASSETS", str(get_weights_path() / "assets.json")
            )
        )
        self._engine = DistributedRunner(
            StreamRenderModel,
            world_size=world_size,
            load_kwargs={"source_root": source_root, "assets_path": assets_path},
            call_timeout=600.0,
        )
        self._engine.start()

    def _status(self) -> StateUpdate:
        state = self.state
        return StateUpdate(
            world_id=state._world_id,
            active=state._active,
            ready=state._active and state._applied_world_id == state._world_id,
            next_frame=state._next_frame,
            total_frames=state._total_frames,
            completed_chunks=state._chunks,
            output_frames=state._output_frames,
            complete=state._complete,
            unused_tail_frames=state._tail_frames,
            error=state._error,
            last_chunk_seconds=state._last_seconds,
        )

    @event(
        name="start",
        description=(
            "Render a recorded 24 FPS semantic video with its opening reference "
            "and prompt. Inputs stay fixed for this run."
        ),
    )
    async def start(
        self,
        video: UploadedFile = InputField(
            description=(
                "Recorded semantic video, 24 FPS, MP4/MOV/MKV/WebM, up to 256 MiB."
            ),
            moderate=True,
        ),
        image: UploadedFile = InputField(
            description=(
                "Appearance reference aligned to the first semantic video frame, "
                "up to 16 MiB."
            ),
            moderate=True,
        ),
        prompt: str = InputField(
            description="Rendering description, fixed for this video.",
            moderate=True,
            min_length=1,
            max_length=32768,
        ),
        seed: int = InputField(
            default=20260830,
            ge=0,
            le=2147483647,
            description="Random seed for the new render.",
        ),
        max_frames: int = InputField(
            default=0,
            ge=0,
            le=14400,
            description=(
                "Source-frame limit; 0 reads the full video, up to 10 minutes. "
                "Official examples request 1450."
            ),
        ),
    ) -> StateUpdate:
        if not prompt.strip():
            raise CommandError("empty_prompt", "Provide an explicit rendering prompt")
        try:
            reference = await asyncio.to_thread(decode_reference, image.data)
        except (ValueError, OSError) as error:
            raise CommandError("invalid_image", str(error)) from error
        try:
            reader = await asyncio.to_thread(SemanticVideo, video.data, max_frames)
        except ValueError as error:
            raise CommandError("invalid_video", str(error)) from error
        state = self.state
        if state._video is not None:
            state._video.close()
        state._world_id += 1
        state._reference, state._prompt, state._seed = reference, prompt, seed
        state._video, state._total_frames = reader, reader.total_frames
        state._complete = False
        state._error = state._last_seconds = state._pending = None
        state._tail_frames = state._output_frames = 0
        state._next_frame = state._frames_wanted = state._chunks = 0
        state._active = True
        self.output.flush()
        status = self._status()
        await self.send(status)
        return status

    @event(
        name="stop",
        description=(
            "Stop generation and clear playout. Start again to render a fresh video."
        ),
    )
    async def stop_render(self) -> StateUpdate:
        self._engine.reset()
        if self.state._video is not None:
            self.state._video.close()
            self.state._video = None
        self.state._active = False
        self.state._pending = self.state._reference = None
        self.state._applied_world_id = None
        self.output.flush()
        status = self._status()
        await self.send(status)
        return status

    async def process_input(self) -> StreamRenderInput:
        state = self.state
        if not state._active:
            raise ApplicationError("No active render")
        if state._world_id != state._applied_world_id:
            return StreamRenderInput(
                world_id=state._world_id,
                reference=state._reference,
                prompt=state._prompt,
                seed=state._seed,
            )
        if state._pending is None:
            if state._video is None:
                raise ApplicationError("No semantic video selected")
            try:
                state._pending = await asyncio.to_thread(
                    state._video.read, state._frames_wanted
                )
            except (ValueError, av.FFmpegError, OSError) as error:
                state._video.close()
                state._video = None
                state._active = False
                state._error = f"Video decoding failed: {error}"
                self._engine.reset()
                await self.send(self._status())
                raise ApplicationError(state._error) from error
            if state._pending is None:
                state._tail_frames = state._video.tail_frames
                state._video = None
                state._active = False
                state._complete = True
                self._engine.reset()
                # Preserve queued output so the final chunk can finish playing.
                await self.send(self._status())
                raise ApplicationError("Video complete")
        return StreamRenderInput(
            world_id=state._world_id,
            first_frame=state._next_frame,
            semantics=state._pending,
        )

    def generate(self, input: StreamRenderInput) -> StreamRenderResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> StreamRenderOutput | None:
        if outcome.error is not None:
            # A failed native step may have advanced its caches; end the session.
            raise outcome.error
        result: StreamRenderResult = outcome.result
        state = self.state
        state._applied_world_id = result.world_id
        state._reference = state._pending = None
        state._next_frame, state._frames_wanted = (
            result.next_frame,
            result.frames_wanted,
        )
        state._chunks = result.chunk_index + 1
        if result.chunk_index >= 0:
            state._last_seconds = outcome.elapsed
        if result.frames is not None:
            state._output_frames += len(result.frames)
        await self.send(self._status())
        if result.frames is None:
            return None
        return StreamRenderOutput(
            main_video=TrackPayload(
                result.frames,
                metadata=[{"world_id": result.world_id, "chunk": result.chunk_index}]
                * len(result.frames),
            )
        )

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @session_ended
    def on_session_ended(self) -> None:
        if self.state is not None and self.state._video is not None:
            self.state._video.close()
            self.state._video = None
        self._engine.reset()
        self.output.flush()
