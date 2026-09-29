"""CPU checks for fixed per-engine schedules; these do not run the DiT."""
from types import SimpleNamespace

import pytest
import torch

from h3_runtime.engine import MiniMaxH3Inference


@pytest.mark.parametrize('steps', [0, 1, True, 2.5])
def test_invalid_schedule_rejected_before_cuda(steps):
    with pytest.raises(ValueError, match='integer >= 2'):
        MiniMaxH3Inference('unused', 'adapter', num_inference_steps=steps)


def test_base_requires_explicit_schedule():
    with pytest.raises(ValueError, match='explicit num_inference_steps'):
        MiniMaxH3Inference('unused', None)


def test_base_rejects_adapter_alpha():
    with pytest.raises(ValueError, match='requires an adapter'):
        MiniMaxH3Inference('unused', None, num_inference_steps=29, adapter_alpha=8)


def test_base_rejects_lora_branch_modes():
    with pytest.raises(ValueError, match='does not use a LoRA branch mode'):
        MiniMaxH3Inference('unused', None, num_inference_steps=29, lora_mode='fused')


def test_generate_uses_each_engine_schedule(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *_: None)
    seen = []
    for points in (5, 9, 29):
        engine = object.__new__(MiniMaxH3Inference)
        engine._num_inference_steps = points
        engine.task = 't2v'
        engine.world_size = 1
        engine.rank = 0
        engine.device = torch.device('cpu')
        engine._adaln_checked = True
        def pipe(**request):
            seen.append(request['num_inference_steps'])
            return SimpleNamespace(videos=[torch.zeros(1)], audio=None)
        engine.pipe = pipe
        engine.generate('test', duration=5)
        engine.generate('test again', duration=5)
        with pytest.raises(AttributeError):
            engine.num_inference_steps = 7
    assert seen == [5, 5, 9, 9, 29, 29]


@pytest.mark.parametrize('adapter,points,expected', [('adapter', None, 5), ('adapter', 9, 9), (None, 29, 29)])
def test_constructor_resolves_schedule_before_model_loading(monkeypatch, adapter, points, expected):
    class StopBeforeCuda(Exception):
        pass
    def stop(_):
        raise StopBeforeCuda
    monkeypatch.setattr(torch.cuda, 'set_device', stop)
    engine = object.__new__(MiniMaxH3Inference)
    with pytest.raises(StopBeforeCuda):
        engine.__init__('unused', adapter, num_inference_steps=points)
    assert engine.num_inference_steps == expected
