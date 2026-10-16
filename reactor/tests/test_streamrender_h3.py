"""CPU contract tests; these do not claim GPU or renderer-quality validation."""

from dataclasses import replace

import numpy as np
import pytest

from streamrender_h3_model import InvalidStream, StreamRenderInput, StreamRenderModel


def test_continuous_clock_survives_integer_json_horizons():
    from streamrender_h3_backend import continuous_time_settings

    diffusion = {
        "sampling_timesteps": {"args": {"T": 1, "num_sampling_steps": 2, "shift": 6}},
        "audio_sampling_timesteps": {
            "args": {"T": 1, "num_sampling_steps": 2, "shift": 1.5}
        },
        "schedule": {"args": {"T": 1, "pred_type": "x_0"}},
        "sampler": {"args": {}},
    }
    settings = {"inference": {"diffusion": diffusion}, "unchanged": object()}
    untouched = settings["unchanged"]
    result = continuous_time_settings(settings)
    for name in ("sampling_timesteps", "audio_sampling_timesteps", "schedule"):
        assert type(result["inference"]["diffusion"][name]["args"]["T"]) is float
        assert result["inference"]["diffusion"][name]["args"]["T"] == 1.0
    assert diffusion["sampling_timesteps"]["args"]["num_sampling_steps"] == 2
    assert diffusion["sampling_timesteps"]["args"]["shift"] == 6
    assert diffusion["audio_sampling_timesteps"]["args"]["shift"] == 1.5
    assert diffusion["schedule"]["args"]["pred_type"] == "x_0"
    assert diffusion["sampler"]["args"] == {}
    assert result["unchanged"] is untouched
    assert continuous_time_settings(result) == result


class Backend:
    def __init__(self):
        self.starts = []
        self.batches = []
        self.failure = None

    def reset(self):
        self.batches = []

    def start(self, reference, prompt, seed):
        self.starts.append((reference, prompt, seed))

    def frames_wanted(self):
        return (3, 5, 2)[len(self.batches) % 3]

    def render(self, frames):
        if self.failure:
            raise self.failure
        self.batches.append(frames.copy())
        return frames[:, :2, :2].copy()


def new_model():
    model = StreamRenderModel()
    model.backend = Backend()
    return model


def anchor(world=1):
    return StreamRenderInput(
        world, np.zeros((768, 1344, 3), np.uint8), "explicit prompt", 42
    )


def test_ten_chunks_preserve_native_variable_counts_and_colors():
    model = new_model()
    result = model.generate(anchor())
    assert result.frames is None and result.chunk_index == -1
    for index in range(10):
        frames = np.full((result.frames_wanted, 768, 1344, 3), index, np.uint8)
        result = model.generate(
            StreamRenderInput(1, first_frame=result.next_frame, semantics=frames)
        )
        assert result.chunk_index == index
        np.testing.assert_array_equal(model.backend.batches[-1], frames)
    assert len(model.backend.starts) == 1


def test_no_implicit_image_or_prompt():
    for request in [StreamRenderInput(1), replace(anchor(), prompt=" ")]:
        with pytest.raises(InvalidStream):
            new_model().generate(request)


def test_no_duplicate_missing_or_lossy_frames():
    model = new_model()
    result = model.generate(anchor())
    frames = np.zeros((result.frames_wanted, 768, 1344, 3), np.uint8)
    for request in [
        StreamRenderInput(1, first_frame=1, semantics=frames),
        StreamRenderInput(1, semantics=frames.astype(np.float32)),
        StreamRenderInput(1, semantics=frames[:-1]),
        replace(anchor(), semantics=frames),
    ]:
        with pytest.raises(InvalidStream):
            model.generate(request)
    assert not model.backend.batches


def test_error_does_not_commit_progress_or_silently_retry():
    model = new_model()
    result = model.generate(anchor())
    request = StreamRenderInput(
        1, semantics=np.zeros((result.frames_wanted, 768, 1344, 3), np.uint8)
    )
    model.backend.failure = RuntimeError("decoder failed")
    with pytest.raises(RuntimeError, match="decoder failed"):
        model.generate(request)
    assert model.chunk_index == -1 and model.next_frame == 0
    with pytest.raises(InvalidStream, match="explicitly restarted"):
        model.generate(request)
    model.backend.failure = None
    result = model.generate(anchor(2))
    assert result.world_id == 2 and result.next_frame == 0


def test_reset_discards_world():
    model = new_model()
    model.generate(anchor())
    model.reset()
    assert model.world_id is None
    with pytest.raises(InvalidStream):
        model.generate(StreamRenderInput(1))


def make_app():
    from unittest.mock import AsyncMock, Mock

    from streamrender_h3 import StreamRenderH3
    from streamrender_h3_types import StreamRenderState

    app = StreamRenderH3()
    app.state = StreamRenderState()
    app._engine = Mock()
    app.send = AsyncMock()
    return app


def test_playout_uses_native_24_fps_for_variable_chunks():
    import asyncio

    from streamrender_h3 import StreamRenderH3
    from streamrender_h3_types import StreamRenderOutput

    assert StreamRenderH3.fps == 24
    app = make_app()
    chunks = []
    app.bind_output(
        broadcast=lambda _: None,
        addressed=lambda *_: None,
        media=chunks.append,
    )
    output = StreamRenderOutput(main_video=np.zeros((8, 2, 2, 3), np.uint8))
    # The native step loop omits compute_time for a class-pinned frame rate.
    asyncio.run(app.emit(output))
    asyncio.run(
        app.emit(StreamRenderOutput(main_video=np.zeros((13, 2, 2, 3), np.uint8)))
    )
    assert [chunk.fps for chunk in chunks] == [24.0, 24.0]


def test_refused_app_steps_do_not_reach_model():
    import asyncio

    from reactor_runtime import ApplicationError

    app = make_app()
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    app.state._active = True
    app.state._applied_world_id = app.state._world_id
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    app._engine.generate.assert_not_called()


def test_app_generate_is_input_only():
    app = make_app()
    request = anchor()
    app.state = None
    app.generate(request)
    app._engine.generate.assert_called_once_with(request)


def test_stop_command_does_not_override_runtime_shutdown():
    import asyncio
    import inspect

    from reactor_runtime import ReactorApp

    from streamrender_h3 import StreamRenderH3

    assert StreamRenderH3.stop is ReactorApp.stop
    assert not inspect.iscoroutinefunction(StreamRenderH3.stop)
    app = make_app()
    app.state._active = True
    result = asyncio.run(app.stop_render())
    assert not result.active
    app._engine.reset.assert_called_once_with()


def test_anchor_until_ack_then_only_semantics():
    import asyncio

    from reactor_runtime import ApplicationError, StepOutcome

    from streamrender_h3_model import StreamRenderResult

    app = make_app()
    app.state._active = True
    app.state._world_id = 1
    app.state._reference = anchor().reference
    app.state._prompt = "selected appearance"
    first = asyncio.run(app.process_input())
    assert asyncio.run(app.process_input()).reference is first.reference
    asyncio.run(
        app.process_output(
            StepOutcome(result=StreamRenderResult(1, -1, 0, 3, None), elapsed=2)
        )
    )
    assert app.state._reference is None
    assert app.state._chunks == 0
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    app.state._pending = np.zeros((3, 768, 1344, 3), np.uint8)
    request = asyncio.run(app.process_input())
    assert request.reference is None and request.prompt is None
    asyncio.run(
        app.process_output(
            StepOutcome(result=StreamRenderResult(1, 0, 3, 5, None), elapsed=0.25)
        )
    )
    assert app.state._chunks == 1 and app.state._last_seconds == 0.25
    assert app.state._next_frame == 3 and app.state._frames_wanted == 5


def test_app_error_never_acknowledges_or_counts_chunk():
    import asyncio

    from reactor_runtime import StepOutcome

    app = make_app()
    with pytest.raises(RuntimeError, match="GPU error"):
        asyncio.run(
            app.process_output(
                StepOutcome(error=RuntimeError("GPU error"), elapsed=0.25)
            )
        )
    assert app.state._chunks == 0 and app.state._applied_world_id is None
    app.send.assert_not_called()


def video_bytes(count=17, rate=24):
    import io

    import av

    file = io.BytesIO()
    with av.open(file, mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
        for index in range(count):
            pixels = np.full((24, 32, 3), index, dtype=np.uint8)
            pixels[:, :16, 0] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return file.getvalue()


def test_video_decode_matches_native_replay_without_skips():
    import io

    import av

    from streamrender_h3_video import SemanticVideo

    data = video_bytes()
    with av.open(io.BytesIO(data)) as container:
        expected = [
            np.asarray(frame.to_image().resize((1344, 768), resample=0))
            for frame in container.decode(video=0)
        ]
    reader = SemanticVideo(data)
    actual = np.concatenate([reader.read(3), reader.read(5), reader.read(8)])
    np.testing.assert_array_equal(actual, np.stack(expected[:16]))
    assert reader.read(3) is None
    assert reader.tail_frames == 1 and reader.container is None


def test_video_limit_and_exact_chunk_boundary():
    from streamrender_h3_video import SemanticVideo

    reader = SemanticVideo(video_bytes(), max_frames=8)
    assert len(reader.read(3)) == 3 and len(reader.read(5)) == 5
    assert reader.read(2) is None and reader.tail_frames == 0
    assert reader.decoded == 8


def test_invalid_video_and_fps_are_refused_before_inference():
    from streamrender_h3_video import SemanticVideo

    for data in [b"", b"not a movie", video_bytes(rate=30)]:
        with pytest.raises(ValueError):
            SemanticVideo(data)
    with pytest.raises(ValueError):
        SemanticVideo(video_bytes(), max_frames=14401)


def upload_inputs():
    import io
    from types import SimpleNamespace

    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", (32, 24)).save(image, "PNG")
    return SimpleNamespace(data=video_bytes()), SimpleNamespace(data=image.getvalue())


def test_start_owns_video_and_rejects_invalid_replacement():
    import asyncio
    from types import SimpleNamespace

    from reactor_runtime import CommandError

    app = make_app()
    video, image = upload_inputs()
    first = asyncio.run(
        app.start(video, image, "original prompt", seed=42, max_frames=8)
    )
    reader = app.state._video
    assert first.world_id == 1 and first.active and not first.ready
    with pytest.raises(CommandError):
        asyncio.run(
            app.start(
                SimpleNamespace(data=b"bad"), image, "prompt", seed=42, max_frames=0
            )
        )
    assert app.state._video is reader and reader.container is not None
    second = asyncio.run(app.start(video, image, "new prompt", seed=2, max_frames=0))
    assert second.world_id == 2 and reader.container is None
    assert app.state._chunks == 0 and app.state._output_frames == 0
    app.on_session_ended()
    assert app.state._video is None


def test_app_reads_requested_batches_and_completes_without_flushing_playout():
    import asyncio
    from unittest.mock import Mock

    from reactor_runtime import ApplicationError, StepOutcome

    from streamrender_h3_model import StreamRenderResult

    app = make_app()
    video, image = upload_inputs()
    asyncio.run(app.start(video, image, "prompt", seed=42, max_frames=8))
    app.output.flush = Mock()
    asyncio.run(
        app.process_output(
            StepOutcome(result=StreamRenderResult(1, -1, 0, 3, None), elapsed=1)
        )
    )
    request = asyncio.run(app.process_input())
    assert request.semantics.shape == (3, 768, 1344, 3)
    assert asyncio.run(app.process_input()).semantics is request.semantics
    asyncio.run(
        app.process_output(
            StepOutcome(
                result=StreamRenderResult(1, 0, 3, 6, np.zeros((8, 2, 2, 3), np.uint8)),
                elapsed=0.4,
            )
        )
    )
    with pytest.raises(ApplicationError, match="complete"):
        asyncio.run(app.process_input())
    assert app.state._complete and not app.state._active
    assert app.state._tail_frames == 5 and app.state._chunks == 1
    assert app.state._output_frames == 8 and app.state._next_frame == 3
    app.output.flush.assert_not_called()
    app._engine.reset.assert_called_once_with()


def test_decode_failure_stops_without_counting_as_completion():
    import asyncio
    from unittest.mock import Mock

    from reactor_runtime import ApplicationError

    app = make_app()
    app.state._active = True
    app.state._applied_world_id = app.state._world_id
    reader = Mock()
    reader.read.side_effect = ValueError("corrupt frame")
    app.state._video = reader
    with pytest.raises(ApplicationError, match="corrupt frame"):
        asyncio.run(app.process_input())
    assert app.state._chunks == 0 and not app.state._complete and not app.state._active
    assert "corrupt frame" in app.state._error
    reader.close.assert_called_once_with()
    app._engine.generate.assert_not_called()


def test_model_modules_import_without_runtime(monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("reactor_runtime"):
            raise AssertionError("Model side imported Runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for name in ("streamrender_h3_model", "streamrender_h3_backend"):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parents[1] / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def test_session_state_does_not_share_input_or_seed():
    from streamrender_h3_types import StreamRenderState

    first = StreamRenderState()
    first._seed = 123
    first._world_id = 7
    first._complete = True
    next_session = StreamRenderState()
    assert next_session._seed == 0 and next_session._world_id == 0
    assert next_session._video is None and not next_session._complete
