from types import SimpleNamespace

import pytest

from hyperflow_h3.schedule import DEFAULT_SIGMAS_8STEP
from sol_hyperflow import runtime


class Model:
    def __init__(self, name, events):
        self.name=name;self.events=events

    def to(self, device):
        self.events.append((self.name,str(device)));return self


class Pipe:
    common=("text_encoder","vae","audio_vae","tokenizer","processor","scheduler","audio_scheduler")

    def __init__(self, kind, events):
        self.kind=kind;self.events=events;self.components={};self.blocks=object()
        self._component_specs=set(self.common)|{"transformer_ref" if kind=="ref2va" else "transformer"}

    def update_components(self, **values):
        self.components.update(values)
        for key,value in values.items():setattr(self,key,value)

    def load_components(self, names=None, **kwargs):
        if names:
            assert names==["transformer_ref"]
            self.update_components(transformer_ref=Model("reference",self.events))
        else:
            values={name:object() for name in self.common}
            values["audio_vae"]=SimpleNamespace(set_attention_backend=lambda backend:None)
            self.update_components(transformer=Model("main",self.events),**values)

    def to(self, device):
        for name in ("transformer","transformer_ref"):
            if name in self.components:getattr(self,name).to(device)

    def set_progress_bar_config(self, **kwargs):pass


@pytest.mark.parametrize("mode", ["fused","separate"])
def test_three_workflows_share_conditioners_but_keep_two_transformers(monkeypatch, mode):
    events=[];loads=[];vae_installs=[]
    def blocks(kind):return SimpleNamespace(init_pipeline=lambda _:Pipe(kind,events))
    def load(pipe,weights):
        loads.append(pipe.kind);return SimpleNamespace(sigmas=DEFAULT_SIGMAS_8STEP,version="1.0")
    def optimize(pipe,model,metadata,world,reference):
        model._consumer_lora_fusion=SimpleNamespace(enabled=True)
        return object(),{"lora":{"mode":"fused"}}
    monkeypatch.setattr(runtime,"hyperflow_blocks",blocks)
    monkeypatch.setattr(runtime,"load_hyperflow_lora",load)
    monkeypatch.setattr(runtime,"configure_hyperflow_blocks",lambda *args:None)
    monkeypatch.setattr(runtime,"install_reference_resize",lambda:None)
    monkeypatch.setattr(runtime,"install_hook",lambda:None)
    monkeypatch.setattr(runtime,"optimize",optimize)
    monkeypatch.setattr(runtime.MiniMaxH3ModularPipeline,"max_duration",runtime.MiniMaxH3ModularPipeline.max_duration)
    monkeypatch.setattr(runtime.dist,"get_rank",lambda:0)
    monkeypatch.setattr(runtime.torch.cuda,"empty_cache",lambda:None)
    monkeypatch.setattr(runtime.vae_parallel,"install",lambda vae,**kw:vae_installs.append(vae))
    pipes,_,report=runtime.build_all("model","adapter.safetensors","cuda:0",8,lora_mode=mode,trim_conditioner=False)
    main,image,ref=pipes["t2v"],pipes["i2v"],pipes["ref2v"]
    assert loads==["t2va","ref2va"]
    assert main.transformer is image.transformer and ref.transformer_ref is not main.transformer
    for name in Pipe.common:
        assert getattr(main,name) is getattr(image,name) is getattr(ref,name)
    assert vae_installs==[main.vae]
    assert events==[("main","cuda:0"),("main","cpu"),("reference","cuda:0"),("main","cuda:0")]
    assert report["transformer_instances"]==2 and report["steps"]==8
    assert report["compute_quant"]=="none" and report["lora_mode"]==mode
    assert main.transformer._consumer_lora_fusion.enabled==(mode=="fused")


@pytest.mark.parametrize("task,mode,expected",[("t2v","speed",(1,2)),("i2v","speed",(1,2)),
    ("ref2v","speed",(0,0)),("ref2v","quality",(8,0))])
def test_attention_switch_closes_previous_policy_before_reset(monkeypatch,task,mode,expected):
    closed=[];reset=[]
    sparse=SimpleNamespace(dense_steps=8,dense_layers=0,sparse_calls=123)
    sparse._close_request=lambda:closed.append((sparse.dense_steps,sparse.dense_layers))
    monkeypatch.setattr(runtime.ulysses,"reset_row_counts",lambda:reset.append(True))
    runtime.begin_request(sparse,task=task,mode=mode)
    assert closed==[(8,0)] and reset==[True]
    assert (sparse.dense_steps,sparse.dense_layers)==expected
    assert sparse.step==-1 and sparse.layer==0 and sparse._prev_timestep is None
    assert sparse._sparse_at_request_start==123
