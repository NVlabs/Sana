"""Force every pooling configuration against independently pooled BF16 values."""

import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _reference(k, v):
    batch, tokens, heads, dim = k.shape
    blocks = (tokens + 63) // 64
    lengths = (tokens - 64 * torch.arange(blocks, device=k.device)).clamp(1, 64)
    results = []
    for value, mean in ((k, True), (v, False)):
        padded = torch.zeros((batch, blocks * 64, heads, dim), device=k.device, dtype=torch.float64)
        padded[:, :tokens] = value.double()
        pooled = padded.reshape(batch, blocks, 64, heads, dim).sum(dim=2)
        if mean:
            pooled = pooled / lengths.double()[None, :, None, None]
        results.append(pooled.to(torch.bfloat16))
    return tuple(results)


def _configuration_list(kernel):
    # Read the production autotuner's actual list, then bypass tuning so no
    # timing-dependent selection can hide a numerically different configuration.
    return [(config.num_warps, config.num_stages, dict(config.kwargs)) for config in kernel.configs]


@pytest.mark.parametrize("backend", ["native", "pointer"])
@pytest.mark.parametrize(
    "shape,kind",
    [
        ((1, 63, 1, 128), "dyadic"),
        ((1, 64, 2, 128), "dyadic"),
        ((2, 65, 3, 128), "dyadic"),
        ((2, 129, 2, 128), "dyadic"),
        ((2, 257, 3, 128), "dyadic"),
        ((1, 8193, 1, 128), "normal"),
    ],
)
@torch.inference_mode()
def test_every_pooling_configuration_matches_float64_reference(backend, shape, kind, record_property):
    if backend == "native" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Native descriptor pooling requires TMA")
    batch, tokens, heads, dim = shape
    blocks = (tokens + 63) // 64
    generator = torch.Generator(device="cuda").manual_seed(71)
    if kind == "normal":
        # Include the same unused Q draw as the captured 8193 regression case.
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    else:
        # BF16 dyadic data with cancellation: FP32 can represent every partial
        # sum exactly, whereas BF16 intermediate sums lose low-order bits.
        k, v = [
            (torch.randint(-1024, 1025, shape, device="cuda", generator=generator).float() / 256).to(torch.bfloat16)
            for _ in range(2)
        ]
    originals = (k.clone(), v.clone())
    expected = _reference(k, v)
    if backend == "native":
        from triton.tools.tensor_descriptor import TensorDescriptor

        module = importlib.import_module("sol_attn.preprocess")
        configs = _configuration_list(module._reduce_kc_kernel)
        assert configs == _configuration_list(module._reduce_vc_kernel)
        assert {(warps, stages) for warps, stages, _ in configs} == {
            (warps, stages) for warps in (4, 8) for stages in (1, 2, 3, 4)
        }
        descriptors = [TensorDescriptor.from_tensor(value, [1, 64, 1, dim]) for value in (k, v)]
    else:
        module = importlib.import_module("sol_attn.triton_ref.preprocess")
        configs = _configuration_list(module._reduce_kv_kernel)
        assert {(warps, stages) for warps, stages, _ in configs} == {
            (warps, stages) for warps in (4, 8) for stages in (1, 2)
        }
    record_property("forced_configs", str(configs))
    first = None
    for warps, stages, kwargs in configs:
        launch = dict(kwargs, num_warps=warps, num_stages=stages)
        previous = None
        for repeat in range(2):
            if backend == "native":
                kc, vc = [torch.empty_like(value) for value in expected]
                for kernel, descriptor, output in zip(
                    (module._reduce_kc_kernel, module._reduce_vc_kernel), descriptors, (kc, vc)
                ):
                    kernel.fn[(1, blocks, batch * heads)](
                        descriptor, output, tokens, heads, blocks, dim, 64, dim, **launch
                    )
            else:
                padded_blocks = ((blocks + 63) // 64) * 64
                kc, vc = [
                    torch.full((batch, padded_blocks, heads, dim), 123.0, device="cuda", dtype=torch.bfloat16)
                    for _ in range(2)
                ]
                module._reduce_kv_kernel.fn[(blocks, batch * heads)](
                    k, v, kc, vc, tokens, tokens, padded_blocks, heads, dim, 64, **launch
                )
                for value in (kc, vc):
                    assert torch.all(value[:, blocks:] == 123.0), "Pooling overwrote padded blocks"
                kc, vc = kc[:, :blocks].contiguous(), vc[:, :blocks].contiguous()
            torch.cuda.synchronize()
            actual = (kc, vc)
            label = f"{backend} {shape} {kind} warps={warps} stages={stages} repeat={repeat}"
            for name, result, reference in zip(("KC", "VC"), actual, expected):
                assert torch.isfinite(result).all(), (label, name)
                torch.testing.assert_close(result, reference, rtol=0, atol=0, msg=f"{label} {name} vs float64")
            for reference in (first, previous):
                if reference is not None:
                    for result, other in zip(actual, reference):
                        assert torch.equal(result.view(torch.uint8), other.view(torch.uint8)), label
            first = actual if first is None else first
            previous = actual
        assert torch.equal(k, originals[0]) and torch.equal(v, originals[1]), "Pooling mutated its inputs"
