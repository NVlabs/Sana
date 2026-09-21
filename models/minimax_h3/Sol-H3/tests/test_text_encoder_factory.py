"""CPU checks for the custom resident conditioner registration hook."""

import pytest
import torch
from h3_runtime.engine import _register_text_encoder


class FakePipeline:
    def __init__(self):
        self.text_encoder = None
        self.registered = []

    def register_components(self, **components):
        self.registered.append(components)
        for name, value in components.items():
            setattr(self, name, value)


def test_none_factory_leaves_the_default_component_unloaded():
    pipe = FakePipeline()
    assert _register_text_encoder(pipe, None, torch.device("cpu")) is None
    assert pipe.text_encoder is None
    assert pipe.registered == []


def test_factory_receives_the_rank_device_and_registers_before_loading():
    pipe = FakePipeline()
    encoder = torch.nn.Linear(4, 4)
    seen = []

    def factory(device):
        seen.append(device)
        return encoder

    assert _register_text_encoder(pipe, factory, torch.device("cuda", 3)) is encoder
    assert seen == [torch.device("cuda", 3)]
    assert pipe.text_encoder is encoder
    assert pipe.registered == [{"text_encoder": encoder}]


def test_factory_must_return_a_module():
    with pytest.raises(TypeError, match="must return a torch.nn.Module"):
        _register_text_encoder(
            FakePipeline(), lambda _device: object(), torch.device("cpu")
        )


def test_registration_failure_is_reported_before_component_loading():
    class RejectingPipeline(FakePipeline):
        def register_components(self, **components):
            pass

    with pytest.raises(RuntimeError, match="did not register"):
        _register_text_encoder(
            RejectingPipeline(),
            lambda _device: torch.nn.Linear(2, 2),
            torch.device("cpu"),
        )
