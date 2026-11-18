from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Scheduler

from hyperflow_h3.schedule import DEFAULT_SIGMAS_8STEP
from sol_hyperflow import adaln_cache as cache


class TimeEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear_1=torch.nn.Linear(1,4); self.endpoint=None

    @contextmanager
    def endpoint_context(self, value):
        previous=self.endpoint; self.endpoint=value
        try:yield
        finally:self.endpoint=previous

    def forward(self, value):
        return self.linear_1(value)+self.endpoint[:,None]*0.25


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear=torch.nn.Linear(4,6*2*3)

    def forward(self, value):
        return self.linear(value).reshape(-1,12).chunk(6,dim=-1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.time_embedder=TimeEmbedder()
        self.transformer_blocks=torch.nn.ModuleList([
            torch.nn.ModuleDict({"adaln_proj":Projection()}) for _ in range(2)])

    def time_proj(self, value):
        return value[:,None]


def plans():
    video=MiniMaxH3Scheduler();video.set_shift(12.)
    audio=MiniMaxH3Scheduler();audio.set_shift(3.)
    pipe=SimpleNamespace(scheduler=video,audio_scheduler=audio,keyframe_noise_aug=.999)
    return cache.make_plans(pipe,DEFAULT_SIGMAS_8STEP)


def test_pair_variants_keep_endpoints_and_pinned_conditions():
    values=plans()
    assert list(values)==["none","image","audio","image_audio"]
    assert all(len(plan)==8 for plan in values.values())
    assert len({cache.signature(plan) for plan in values.values()})==4
    t,r,_=values["none"][0]
    assert t.numel()==2 and torch.equal(t,torch.zeros_like(t))
    assert r[0]!=r[1]  # Same initial t, different video/audio endpoints.
    t,r,_=values["image_audio"][2]
    pairs=list(zip(t.tolist(),r.tolist()))
    assert (1.,1.) in pairs
    assert any(abs(a-.999)<1e-6 and a==b for a,b in pairs)


def test_tables_equal_original_projections_across_all_variants_and_steps():
    torch.manual_seed(42);model=Model().eval();values=plans();expected={}
    with torch.no_grad():
        for label,plan in values.items():
            for step,(t,r,_) in enumerate(plan):
                with model.time_embedder.endpoint_context(r):embedding=model.time_embedder(model.time_proj(t))
                for i,block in enumerate(model.transformer_blocks):
                    expected[label,step,i]=torch.cat(block.adaln_proj(embedding),dim=-1).clone()
    report=cache.precompute(model,values)
    assert report["steps"]==8 and report["blocks"]==2 and report["freed_bytes"]>0
    assert all(not list(block.adaln_proj.parameters()) for block in model.transformer_blocks)
    for label in ("image","none","image_audio","audio","image"):
        variant=cache.select_plan(model,values[label])
        for step in range(8):
            model._multi_pair_cursor.set(step,variant)
            for i,block in enumerate(model.transformer_blocks):
                actual=torch.cat(block.adaln_proj(torch.empty(0)),dim=-1)
                reference=expected[label,step,i]
                assert torch.equal(actual[:len(reference)],reference)
                assert torch.count_nonzero(actual[len(reference):])==0
    assert report["hits"]["image"]==2
    with pytest.raises(RuntimeError,match="already installed"):cache.precompute(model,values)


def test_unknown_conditioning_plan_is_rejected():
    model=Model();values=plans();cache.precompute(model,values)
    changed=[tuple(x.clone() for x in triple) for triple in values["image"]]
    changed[0][1][0]+=0.01
    with pytest.raises(RuntimeError,match="not precomputed"):cache.select_plan(model,changed)
