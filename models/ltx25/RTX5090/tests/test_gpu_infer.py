from __future__ import annotations

from contextlib import contextmanager

from models.ltx25.RTX5090 import gpu_infer


class FakeAttention:
    def __init__(self) -> None:
        self.contexts = []

    @contextmanager
    def stage2(self, enabled: bool):
        self.contexts.append(enabled)
        yield


class FakeStage:
    def __call__(self, *args, **kwargs):
        return kwargs["loop"](transformer=object())


class KeysOnlyCheckpoint:
    def keys(self):
        return (
            "transformer.keyframes_abs_pos_embedding",
            "transformer.patchify_proj.bias",
        )


def test_keyframe_lookup_accepts_noniterable_safe_open_handle() -> None:
    assert (
        gpu_infer.find_keyframe_embedding_key(KeysOnlyCheckpoint())
        == "transformer.keyframes_abs_pos_embedding"
    )


def test_shared_stage_marks_only_the_second_call_as_stage2(monkeypatch) -> None:
    monkeypatch.setattr(gpu_infer, "sync", lambda: None)
    attention = FakeAttention()
    timings = gpu_infer.Timings()
    stage = gpu_infer.TimedStage("stage", FakeStage(), timings, attention)

    stage(loop=lambda **_kwargs: None)
    stage(loop=lambda **_kwargs: None)

    assert attention.contexts == [False, True]
    assert timings.total("stage_1") >= 0.0
    assert timings.total("stage_2") >= 0.0
