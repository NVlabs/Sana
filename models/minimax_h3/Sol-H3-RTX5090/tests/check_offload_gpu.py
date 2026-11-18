"""Run explicitly in the Stage1 environment on one visible RTX 5090."""
import copy
import json

import torch
import triton
import triton.language as tl

from runtime.offload import install_stage1_loader, mutation_scope, validate_device


@triton.jit
def add_one(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offsets, tl.load(x + offsets, offsets < N, 0) + 1, offsets < N)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(32, dtype=torch.bfloat16), requires_grad=False)
        self.register_buffer("_fp8_weight", torch.randn(32, 32).to(torch.float8_e4m3fn), persistent=False)
        self.register_buffer("scale", torch.tensor(0.03125), persistent=False)

    def forward(self, x):
        return x @ (self._fp8_weight.to(x.dtype) * self.scale).T + self.bias


def main():
    validate_device(torch, {"execution": {"offload": "cpu"}})
    torch.manual_seed(42)
    x = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    add_one[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), 256)
    torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
    install_stage1_loader()
    from fastvideo.models.loader import component_loader
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([Block() for _ in range(50)])
    reference = copy.deepcopy(model.transformer_blocks[0]).cuda()
    component_loader.enable_layerwise_offload(model)
    manager = model._layerwise_offload_manager
    first = model.transformer_blocks[0]
    with mutation_scope(first):
        assert first._fp8_weight.is_cuda and first.bias.is_cuda
        first.register_buffer("new_lookup", torch.arange(3, device="cuda", dtype=torch.bfloat16), persistent=False)
    assert first.new_lookup.device.type == "cpu"
    manager.prepare_compilation()
    manager.install_inference_hooks()
    expected = reference(x)
    torch.testing.assert_close(first(x), expected, rtol=0, atol=0)
    first.forward = torch.compile(first.forward, fullgraph=True)
    reference.forward = torch.compile(reference.forward, fullgraph=True)
    expected = reference(x)
    for _ in range(2):
        actual = first(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(t.device.type == "cpu" for t in (*first.parameters(), *first.buffers()))
    for block in model.transformer_blocks[1:]:
        block(x)
    def failing(x):
        assert first._fp8_weight.is_cuda
        raise ValueError("test exception cleanup")
    first.forward = failing
    try:
        first(x)
    except ValueError as error:
        assert str(error) == "test exception cleanup"
    else:
        raise AssertionError("expected test exception")
    assert all(t.device.type == "cpu" for t in (*first.parameters(), *first.buffers()))
    assert manager.stats["active_blocks"] == 0
    assert manager.stats["max_active_blocks"] == 1
    assert manager.stats["fp8_buffer_count"] == 50
    print(json.dumps({"status": "PASS", "torch": torch.__version__, "triton": triton.__version__,
                      "tests": ["triton_jit", "eager_equality", "compiled_equality", "fp8_buffers", "mutation", "exception_cleanup"],
                      "offload": manager.stats}), flush=True)


if __name__ == "__main__":
    main()
