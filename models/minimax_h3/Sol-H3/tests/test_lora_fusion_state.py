"""Specialization must preserve PEFT state and reject unsupported adapters."""
from types import SimpleNamespace
import pytest
import torch
from peft import LoraConfig
from peft.tuners.lora import Linear
from h3_runtime.lora_fusion import install, split_linear


def model():
    def linear():
        layer = Linear(torch.nn.Linear(8, 8, dtype=torch.bfloat16), adapter_name='test_adapter',
                       config=LoraConfig(r=4, lora_alpha=4, lora_dropout=0), r=4, lora_alpha=4)
        return layer.to(torch.bfloat16).requires_grad_(False).eval()
    block = SimpleNamespace(attn=SimpleNamespace(to_q=linear(), to_k=linear(), to_v=linear(),
        to_out=[linear(), torch.nn.Dropout(0).eval()]),
        ff=SimpleNamespace(net=[SimpleNamespace(proj=linear()), torch.nn.Identity(), linear()]))
    return SimpleNamespace(transformer_blocks=[block])


def test_install_uninstall_preserves_peft_weights_and_forward():
    m = model()
    layer = m.transformer_blocks[0].attn.to_q
    before = {n:t.clone() for n,t in layer.state_dict().items()}
    forward = layer.forward
    state = install(m)
    assert len(state.targets) == 6
    assert layer.forward == forward
    with torch.no_grad():
        x = torch.randn(2, 8).to(torch.bfloat16)
        actual, delta = split_linear(layer, x)  # CPU uses native PEFT.
        assert delta is None and torch.equal(actual, layer(x))
    with pytest.raises(ValueError, match='already installed'):
        install(m)
    state.uninstall()
    assert not hasattr(layer, '_sol_lora_fusion')
    assert set(before) == set(layer.state_dict())
    for n,t in layer.state_dict().items():
        assert torch.equal(t, before[n])


@pytest.mark.parametrize('change', ['scale', 'training', 'disabled', 'dtype', 'hook'])
def test_reject_unsupported_without_partial_install(change):
    m = model()
    first, last = m.transformer_blocks[0].attn.to_q, m.transformer_blocks[0].ff.net[2]
    if change == 'scale': last.scaling['test_adapter'] = .5
    elif change == 'training': last.train()
    elif change == 'disabled': last.enable_adapters(False)
    elif change == 'dtype': last.lora_A['test_adapter'].float()
    elif change == 'hook': last.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match='requires frozen'):
        install(m)
    assert not hasattr(first, '_sol_lora_fusion')
    assert not hasattr(last, '_sol_lora_fusion')


def test_incompatible_consumer_switch_rejected_before_mutation():
    from h3_runtime.fusion_install import install as install_consumers
    with pytest.raises(ValueError, match='qknorm_rope=True'):
        install_consumers(None, lora=True, qknorm_rope=False)


def test_nondefault_adapter_name_is_supported():
    m = model()
    state = install(m)
    assert len(state.targets) == 6
    assert all(layer.active_adapters == ['test_adapter'] for layer in state.targets)
    state.uninstall()


@pytest.mark.parametrize('owns_group', [False, True])
def test_engine_close_only_releases_its_own_group(monkeypatch, owns_group):
    from h3_runtime import engine as runtime

    calls = []
    monkeypatch.setattr(runtime.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(runtime.dist, 'barrier', lambda: calls.append('barrier'))
    monkeypatch.setattr(runtime.dist, 'destroy_process_group', lambda: calls.append('destroy'))
    engine = runtime.MiniMaxH3Inference.__new__(runtime.MiniMaxH3Inference)
    engine._owns_process_group = owns_group
    engine.close()
    engine.close()
    assert calls == (['barrier', 'destroy'] if owns_group else [])
    assert not engine._owns_process_group
