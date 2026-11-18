"""Synthetic public-format adapter tests; no model weights or prompts required."""
import pytest
import torch
from safetensors.torch import save_file
from h3_runtime.lora import load_lora_branches


def model():
    m = torch.nn.Module()
    m.to_q = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    m.to_v = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    return m.eval()


def checkpoint(tmp_path, *, hybrid=False, bad_shape=False):
    torch.manual_seed(91)
    tensors = {}
    for name in ('to_q', 'to_v'):
        tensors[f'{name}.lora_A.weight'] = torch.randn(4, 8).bfloat16()
        tensors[f'{name}.lora_B.weight'] = torch.randn(8, 3 if bad_shape and name == 'to_v' else 4).bfloat16()
    metadata = {}
    if hybrid:
        metadata = {'format': 'fastvideo-lora-v2', 'rank': '4', 'low_rank_tensors': '4', 'diff_tensors': '1'}
        tensors['to_q.diff_b'] = torch.full((8,), .003, dtype=torch.float32)
    p = tmp_path/'adapter.safetensors'
    save_file(tensors, p, metadata=metadata)
    return p, tensors


@pytest.mark.parametrize('hybrid', [False, True])
def test_separate_preserves_base_and_exact_branch_rounding(tmp_path, hybrid):
    torch.manual_seed(90)
    m = model();weight = m.to_q.weight.clone();bias = m.to_q.bias.clone()
    p, tensors = checkpoint(tmp_path, hybrid=hybrid)
    report = load_lora_branches(m, p, alpha=4)
    assert report['effective_scale'] == 1 and report['pairs'] == 2
    assert torch.equal(m.to_q.base_layer.weight, weight)
    if hybrid: bias = (bias.float() + tensors['to_q.diff_b']).bfloat16()
    assert torch.equal(m.to_q.base_layer.bias, bias)
    x = torch.randn(3, 8).bfloat16()
    expected = torch.nn.functional.linear(x, weight, bias) + torch.nn.functional.linear(
        torch.nn.functional.linear(x, tensors['to_q.lora_A.weight']), tensors['to_q.lora_B.weight'])
    with torch.inference_mode():assert torch.equal(m.to_q(x), expected)
    assert not m.to_q.merged and not m.to_q.training


def test_invalid_pair_does_not_install_any_wrappers(tmp_path):
    m = model();state = {n:p.clone() for n,p in m.state_dict().items()}
    p, _ = checkpoint(tmp_path, bad_shape=True)
    with pytest.raises(ValueError, match='shape mismatch'):load_lora_branches(m, p, alpha=4)
    assert type(m.to_q) is torch.nn.Linear and type(m.to_v) is torch.nn.Linear
    assert all(torch.equal(p,state[n]) for n,p in m.state_dict().items())
