import pytest
import torch
import torch.distributed as dist

from sol_hyperflow.engine import HyperFlowInference
from validate_runtime import digest


@pytest.mark.parametrize("owns_group", [False, True])
def test_close_releases_only_owned_process_groups(monkeypatch, owns_group):
    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(dist, "destroy_process_group", lambda: calls.append("destroy"))
    engine = HyperFlowInference.__new__(HyperFlowInference)
    engine._owns_process_group = owns_group
    engine.close()
    engine.close()
    assert calls == (["barrier", "destroy"] if owns_group else [])
    assert engine._closed and not engine._owns_process_group
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate("A scene.")


def test_warmup_uses_the_requested_prompt_and_image():
    engine = HyperFlowInference.__new__(HyperFlowInference)
    calls = []
    engine.generate = lambda *args, **kwargs: calls.append((args, kwargs))
    image = object()
    engine.warmup(prompt="A particular long scene description.", task="i2v",
                  duration=15, image=image)
    assert calls[0][0] == ("A particular long scene description.",)
    assert calls[0][1]["image"] is image and calls[0][1]["duration"] == 15


def test_validation_digest_handles_bfloat16_and_missing_audio():
    values = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()
    result = digest({"video": [values], "audio": None})
    assert result["audio"] is None
    assert result["video"][0] == digest(values.contiguous())
    changed = values.clone()
    changed[0, 0] = 1
    assert digest(changed)["sha256"] != result["video"][0]["sha256"]
