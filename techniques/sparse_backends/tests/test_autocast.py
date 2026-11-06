"""Autocast must preserve the internal BF16 threshold-kernel contract."""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _require_sm80():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires SM80 to exercise the Triton pointer backend")


@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_exact_attention_preserves_bf16_under_autocast(autocast_dtype):
    _require_sm80()
    from sol_attn import get_sol_attn_backend, sol_attn

    assert get_sol_attn_backend("cuda") == "triton"
    torch.manual_seed(7)
    shape = (1, 129, 1, 128)
    q, k, v = [
        (torch.randn(shape, device="cuda") * 0.25).to(torch.bfloat16) for _ in range(3)
    ]
    kwargs = {"thresh_type": "exact", "sink_start": 0, "sink_tokens": shape[1]}
    with torch.autocast("cuda", enabled=False):
        expected = sol_attn(q, k, v, **kwargs)
        # The full sink makes dense SDPA an independent mathematical oracle.
        reference = (
            F.scaled_dot_product_attention(
                q.transpose(1, 2).double(),
                k.transpose(1, 2).double(),
                v.transpose(1, 2).double(),
            )
            .transpose(1, 2)
            .to(q.dtype)
        )
    with torch.autocast("cuda", dtype=autocast_dtype):
        actual = sol_attn(q, k, v, **kwargs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)


def test_exact_threshold_avoids_fp16_intermediate_overflow():
    _require_sm80()
    from sol_attn.triton_ref.preprocess import prepare

    shape = (1, 129, 1, 128)
    q = torch.full(shape, 1 / 256, device="cuda", dtype=torch.bfloat16)
    k = torch.full_like(q, 256)
    v = torch.ones_like(q)
    kwargs = {"tau": 1.0, "scale": 128**-0.5, "thresh_type": "exact"}
    # K's second moment exceeds FP16's range. Casting an autocasted matmul
    # result back to BF16 cannot recover the original finite values.
    with torch.autocast("cuda", enabled=False):
        _, _, expected = prepare(q, k, v, **kwargs)
    with torch.autocast("cuda", dtype=torch.float16):
        _, _, actual = prepare(q, k, v, **kwargs)

    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
