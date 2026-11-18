"""Public rendering progress and the generated video track."""

import numpy as np
from reactor_runtime import InputState, MessageField, ModelMessage, Output, Video

from streamrender_h3_video import SemanticVideo


class StreamRenderState(InputState):
    _world_id: int = 0
    _applied_world_id: int | None = None
    _reference: np.ndarray | None = None
    _prompt: str = ""
    _seed: int = 0
    _active: bool = False
    _video: SemanticVideo | None = None
    _pending: np.ndarray | None = None
    _next_frame: int = 0
    _frames_wanted: int = 0
    _chunks: int = 0
    _output_frames: int = 0
    _total_frames: int | None = None
    _tail_frames: int = 0
    _complete: bool = False
    _error: str | None = None
    _last_seconds: float | None = None


class StreamRenderOutput(Output):
    main_video: Video


class StateUpdate(ModelMessage):
    world_id: int = MessageField(description="Identity of the selected rendering run.")
    active: bool = MessageField(description="Whether generation is active.")
    ready: bool = MessageField(
        description="Whether image and text conditioning are prepared."
    )
    next_frame: int = MessageField(
        description="Number of semantic frames consumed successfully."
    )
    total_frames: int | None = MessageField(
        description="Selected source-frame count, when declared by the video container."
    )
    completed_chunks: int = MessageField(
        description="Number of successful native render chunks."
    )
    output_frames: int = MessageField(
        description="Generated frames published on main_video."
    )
    complete: bool = MessageField(
        description=(
            "All complete input chunks have been rendered; "
            "buffered video may still be playing."
        )
    )
    unused_tail_frames: int = MessageField(
        description=(
            "Trailing source frames discarded because they cannot fill a native chunk."
        )
    )
    error: str | None = MessageField(
        description="An input decoding failure that stopped this run, if any."
    )
    last_chunk_seconds: float | None = MessageField(
        description="Runtime-measured duration of the last render chunk."
    )
