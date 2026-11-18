import math
from types import SimpleNamespace

import pytest
from diffusers.modular_pipelines.minimax_h3 import before_encoder

from sol_hyperflow import runtime


class Setup:
    def __call__(self, components, state):
        width,height=state.size
        scale = components.config.reference_image_short_edge / min(width, height)
        state.scale=scale
        return components,state


def test_guarded_reference_patch_is_idempotent_and_never_upscales(monkeypatch):
    monkeypatch.setattr(before_encoder,"MiniMaxH3Ref2VASetupStep",Setup)
    original=Setup.__call__
    try:
        runtime.install_reference_resize();first=Setup.__call__
        runtime.install_reference_resize();assert Setup.__call__ is first
        assert Setup.__call__.__wrapped__ is original
        component=SimpleNamespace(config=SimpleNamespace(reference_image_short_edge=2048))
        for size in [(512,256),(2048,4096)]:
            state=SimpleNamespace(size=size);Setup()(component,state)
            assert state.scale==min(1.,math.sqrt(1344*768/(size[0]*size[1])))
    finally:
        Setup.__call__=original


def test_changed_upstream_geometry_is_rejected(monkeypatch):
    class UnknownSetup:
        def __call__(self,components,state):return components,state
    monkeypatch.setattr(before_encoder,"MiniMaxH3Ref2VASetupStep",UnknownSetup)
    with pytest.raises(AssertionError,match="Pinned reference geometry"):
        runtime.install_reference_resize()
