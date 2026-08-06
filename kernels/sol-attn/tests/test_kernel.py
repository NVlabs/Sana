from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import sol_attn
from sol_attn import interface
from sol_attn.triton_ref import sol_attn as triton_sol_attn


def _inputs(tokens=256, heads=4):
    torch.manual_seed(42)
    q = torch.randn(
        1,
        tokens,
        heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return q, torch.randn_like(q), torch.randn_like(q)


def test_package_uses_isolation_safe_imports():
    package_root = Path(sol_attn.__file__).resolve().parent
    offenders = []
    for path in package_root.rglob("*.py"):
        if path.name.startswith("._"):
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("from sol_attn") or stripped.startswith(
                "import sol_attn"
            ):
                offenders.append(f"{path.relative_to(package_root)}:{line_number}")
    assert offenders == []


def test_backend_dispatch_contract():
    assert interface._backend_for_arch((8, 0), cute_available=True) == "triton"
    assert interface._backend_for_arch((8, 9), cute_available=True) == "triton"
    assert interface._backend_for_arch((9, 0), cute_available=True) == "cute_sm90"
    assert interface._backend_for_arch((10, 0), cute_available=True) == "cute_sm100"
    assert interface._backend_for_arch((12, 0), cute_available=True) == "cute_sm120"
    assert interface._backend_for_arch((9, 0), cute_available=False) == "triton"
    assert interface._backend_for_arch((10, 0), cute_available=False) == "triton"
    assert interface._backend_for_arch((12, 0), cute_available=False) == "triton"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_full_sink_matches_sdpa():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Sol-Attn requires compute capability 8.0 or newer")

    q, k, v = _inputs()
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)
    actual = sol_attn.sol_attn(
        q,
        k,
        v,
        tau=1.0,
        thresh_type="exact",
        sink_start=0,
        sink_tokens=q.shape[1],
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sparse_matches_triton_reference():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Sol-Attn requires compute capability 8.0 or newer")

    q, k, v = _inputs()
    expected = triton_sol_attn(
        q,
        k,
        v,
        tau=1.0,
        thresh_type="exact",
    )
    actual = sol_attn.sol_attn(
        q,
        k,
        v,
        tau=1.0,
        thresh_type="exact",
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)
