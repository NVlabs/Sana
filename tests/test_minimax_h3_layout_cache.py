"""CPU regressions for request-local MiniMax-H3 position-layout caching."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture(params=["A100", "H100"])
def adapter(request, tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "minimax_h3"
        / request.param
        / "adapter.py"
    )
    name = f"minimax_h3_layout_test_{request.param}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("H3_SOL_ATTN", "1")
    monkeypatch.setattr(module, "emit_event", lambda *args, **kwargs: None)
    epoch_file = tmp_path / "request_epoch.txt"
    monkeypatch.setenv("H3_REQUEST_EPOCH_FILE", str(epoch_file))
    epoch_file.write_text("request-A\n")
    return module, epoch_file


def _positions():
    return {
        "text_positions": torch.tensor([0, 1]),
        "image_positions": torch.tensor([4, 5, 6, 7]),
        "audio_positions": torch.tensor([2, 3]),
    }


def _begin(module, positions):
    return module.begin_model_forward(
        unique_timesteps=torch.tensor([1000.0]),
        **positions,
        cu_seqlens_host=(0, 8),
        total_tokens=8,
        num_layers=50,
    )


def test_new_request_recomputes_layout_for_reused_buffers(adapter):
    module, epoch_file = adapter
    # Serving tensors can be inference tensors, which have no version counter.
    with torch.inference_mode():
        positions = _positions()
        first = _begin(module, positions)
        addresses = [tensor.data_ptr() for tensor in positions.values()]
        positions["image_positions"].copy_(torch.tensor([2, 3, 6, 7]))
        positions["audio_positions"].copy_(torch.tensor([4, 5]))
        epoch_file.write_text("request-B\n")
        second = _begin(module, positions)

    assert addresses == [tensor.data_ptr() for tensor in positions.values()]
    assert first.prefix_tokens == 4
    assert second.prefix_tokens == second.target_video_start == 6
    assert second.request_epoch == "request-B"
    assert second.request_index == first.request_index + 1
    assert second.step_index == 0


def test_new_request_revalidates_reused_buffers(adapter):
    module, epoch_file = adapter
    with torch.inference_mode():
        positions = _positions()
        _begin(module, positions)
        positions["image_positions"].copy_(torch.tensor([2, 3, 6, 7]))
        positions["audio_positions"].copy_(torch.tensor([3, 5]))
        epoch_file.write_text("request-B\n")
        with pytest.raises(ValueError, match="token positions overlap"):
            _begin(module, positions)


def test_unchanged_request_keeps_cached_host_layout(adapter, monkeypatch):
    module, _ = adapter
    original = module._host_positions
    calls = []

    def materialize(values):
        calls.append(values)
        return original(values)

    monkeypatch.setattr(module, "_host_positions", materialize)
    with torch.inference_mode():
        positions = _positions()
        first = _begin(module, positions)
        assert len(calls) == 3
        second = _begin(module, positions)

    assert len(calls) == 3
    assert second.request_index == first.request_index
    assert second.step_index == 1
    assert second.prefix_tokens == 4
