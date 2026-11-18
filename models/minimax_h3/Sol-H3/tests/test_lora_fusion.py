"""Strict consumer-fusion regression; run only in an allocated CUDA environment."""
import os
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA allocation required')


def _exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), (
        int(torch.count_nonzero(a != b)), float((a.float() - b.float()).abs().max()))


def _rand(shape, scale=1):
    return (torch.randn(shape, device='cuda') * scale).to(torch.bfloat16)


def test_swiglu_lora_rounding_and_strides():
    from h3_runtime.fusions import fused_swiglu
    torch.manual_seed(813)
    for width in (35, 14336):
        # Non-contiguous row stride and a small update at BF16 half-ULP boundaries.
        base = _rand((2, 37, 4 * width))[..., :2 * width]
        delta = _rand((2, 37, 4 * width), 0.003)[..., :2 * width]
        base[..., :4] = torch.tensor([1, 1.0078125, -1, 0], device='cuda')
        delta[..., :4] = torch.tensor([0.00390625, 0.00390625, 1, -0.0], device='cuda')
        _exact(fused_swiglu(base, lora=delta), fused_swiglu(base + delta))


def test_attention_output_lora_modulation():
    from h3_runtime.fusions import fused_residual_gate_rmsnorm_modulate as fused
    torch.manual_seed(814)
    for width in (65, 5376):
        base, delta, residual = _rand((2, 37, width)), _rand((2, 37, width), .003), _rand((2, 37, width))
        table = _rand((3, 6 * width), .1)
        shift, scale, gate, _, _, _ = table.chunk(6, -1)
        index = torch.arange(37, device='cuda') % 3
        weight = _rand((width,))
        expected = fused(residual, base + delta, gate, weight, scale, shift, index, 1e-6)
        actual = fused(residual, base, gate, weight, scale, shift, index, 1e-6, lora=delta)
        for a, b in zip(actual, expected):
            _exact(a, b)


def test_ffn_output_lora_gate_rounding():
    from h3_runtime.fusions import fused_lora_gate_residual
    torch.manual_seed(815)
    base, delta, residual = _rand((2, 37, 5376)), _rand((2, 37, 5376), .003), _rand((2, 37, 5376))
    gate = _rand((3, 6 * 5376), .3).chunk(6, -1)[2]
    index = torch.arange(37, device='cuda') % 3
    expected = residual + gate.index_select(0, index) * (base + delta)
    _exact(fused_lora_gate_residual(residual, base, delta, gate, index), expected)
    # A different promoted dtype must preserve the eager fallback.
    _exact(fused_lora_gate_residual(residual, base, delta, gate.float(), index),
           residual + gate.float().index_select(0, index) * (base + delta))


def test_qkv_lora_wire_bytes():
    from h3_runtime.relayout import qknorm_rope_pack_qkv_destination_major as pack
    torch.manual_seed(816)
    rows, heads, dim = 19, 56, 128
    base = _rand((rows, 3, heads, dim))
    delta = _rand(base.shape, .003)
    q, k, v = base.unbind(1)
    dq, dk, dv = delta.unbind(1)
    angle = torch.randn((rows, 64), device='cuda')
    cos, sin = angle.cos(), angle.sin()
    qw, kw = _rand((dim,)), _rand((dim,))
    for world in (1, 2, 4, 8):
        for wire in ('bf16', 'int8_qkv'):
            expected = pack(q + dq, k + dk, v + dv, qw, kw, cos, sin, 1e-6, 1e-6, world, wire_dtype=wire)
            actual = pack(q, k, v, qw, kw, cos, sin, 1e-6, 1e-6, world, wire_dtype=wire, lora=(dq, dk, dv))
            _exact(actual, expected)


@pytest.mark.skipif(os.environ.get('H3_RUN_LARGE_GPU_TESTS') != '1', reason='Explicit large GPU allocation required')
def test_lora_swiglu_64bit_offsets():
    from h3_runtime.fusions import fused_swiglu
    rows, width = 110500, 14336
    base = torch.full((rows, width * 2), .25, dtype=torch.bfloat16, device='cuda')
    delta = torch.full_like(base, .0009765625)
    actual = fused_swiglu(base, lora=delta)
    boundary = 2**31 // (2 * width)
    idx = torch.tensor([0, boundary - 1, boundary, boundary + 1, rows - 1], device='cuda')
    expected = fused_swiglu(base.index_select(0, idx) + delta.index_select(0, idx))
    _exact(actual.index_select(0, idx), expected)


def test_split_linear_nondefault_adapter_matches_native_peft():
    from peft import LoraConfig
    from peft.tuners.lora import Linear
    from h3_runtime.lora_fusion import LoRAFusion, split_linear
    torch.manual_seed(817)
    layer = Linear(torch.nn.Linear(96, 128, dtype=torch.bfloat16, device='cuda'),
                   adapter_name='test_adapter', config=LoraConfig(r=8, lora_alpha=8), r=8, lora_alpha=8)
    layer = layer.to(dtype=torch.bfloat16).requires_grad_(False).eval()
    layer.lora_B['test_adapter'].weight.normal_()
    layer._sol_lora_fusion = LoRAFusion()
    with torch.inference_mode():
        x = _rand((2, 37, 96))
        base, delta = split_linear(layer, x)
        assert delta is not None
        _exact(base + delta, layer(x))
        layer._sol_lora_fusion.enabled = False
        out, delta = split_linear(layer, x)
        assert delta is None
        _exact(out, layer(x))
