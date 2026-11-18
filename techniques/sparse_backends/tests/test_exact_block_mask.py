"""GPU contracts for arbitrary exact Q/KV blocks in the fused SOL kernels."""

import importlib
import math

import pytest
import torch

from sol_attn import sol_attn
from sol_attn.interface import _validate_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.fixture(params=["native", "triton_tma", "triton_pointer"])
def kernel(request, monkeypatch):
    if request.param == "native":
        return sol_attn
    module = importlib.import_module("sol_attn.triton_ref.fwd")
    if request.param == "triton_pointer":
        monkeypatch.setattr(module, "_use_tma", lambda device: False)
    elif torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA requires SM90 or newer")
    return module.sol_attn


def inputs(tokens=513):
    generator = torch.Generator(device="cuda").manual_seed(71)
    return tuple(
        torch.randn(
            1, tokens, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        for _ in range(3)
    )


def hybrid_oracle(q, k, v, mask, tau, thresh_type):
    """Independent token-level attention over exact blocks and expanded centroids."""
    from sol_attn.triton_ref.preprocess import prepare

    tokens = q.shape[1]
    scale = 128**-0.5
    kc, vc, thresholds = prepare(
        q, k, v, tau=tau, scale=scale, thresh_type=thresh_type, tokens=tokens
    )
    result = torch.empty_like(q, dtype=torch.float32)
    protected = mask.tolist()
    for qb in range(len(protected)):
        qs = slice(qb * 64, min((qb + 1) * 64, tokens))
        mean = q[:, qs].float().mean(1)
        keys, values = [], []
        for kb in range(len(protected)):
            ks = slice(kb * 64, min((kb + 1) * 64, tokens))
            score = (mean * kc[:, kb].float()).sum(-1) * (scale * math.log2(math.e))
            exact = score > thresholds[:, qb]
            if abs(qb - kb) <= 1 or protected[qb] or protected[kb]:
                exact = torch.ones_like(exact)
            exact = exact[:, None, :, None]
            keys.append(
                torch.where(exact, k[:, ks].float(), kc[:, kb : kb + 1].float())
            )
            values.append(
                torch.where(
                    exact,
                    v[:, ks].float(),
                    vc[:, kb : kb + 1].float() / (ks.stop - ks.start),
                )
            )
        result[:, qs] = torch.nn.functional.scaled_dot_product_attention(
            q[:, qs].float().transpose(1, 2),
            torch.cat(keys, 1).transpose(1, 2),
            torch.cat(values, 1).transpose(1, 2),
            scale=scale,
        ).transpose(1, 2)
    return result


@pytest.mark.parametrize("tokens,tau", [(63, 1e6), (257, 1e6), (513, 1e6), (513, 1.5)])
@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
@torch.inference_mode()
def test_disjoint_exact_rows_and_columns_match_hybrid_oracle(
    kernel, tokens, tau, thresh_type
):
    q, k, v = inputs(tokens)
    mask = torch.zeros((tokens + 63) // 64, device="cuda", dtype=torch.bool)
    mask[0] = mask[-1] = True
    if mask.numel() > 5:
        mask[4] = True
    actual = kernel(q, k, v, tau=tau, thresh_type=thresh_type, exact_block_mask=mask)
    expected = hybrid_oracle(q, k, v, mask, tau, thresh_type)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@torch.inference_mode()
def test_empty_mask_preserves_legacy_output_and_mutated_mask_uses_current_values(
    kernel,
):
    q, k, v = inputs()
    mask = torch.zeros(10, device="cuda", dtype=torch.bool)[1:]
    assert mask.is_contiguous() and mask.data_ptr() % 16 != 0
    legacy = kernel(q, k, v, tau=1e6, sink_start=128, sink_tokens=64)
    actual = kernel(
        q, k, v, tau=1e6, sink_start=128, sink_tokens=64, exact_block_mask=mask
    )
    torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
    mask.fill_(True)
    actual = kernel(q, k, v, tau=1e6, exact_block_mask=mask)
    dense = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), dense, atol=0.02, rtol=0.02)
    assert not torch.equal(actual, legacy)


@pytest.mark.parametrize(
    "kind", ["list", "dtype", "length", "rank", "stride", "device"]
)
def test_invalid_mask_rejected(kind):
    q, k, v = inputs()
    masks = {
        "list": [True] * 9,
        "dtype": torch.ones(9, device="cuda"),
        "length": torch.ones(8, device="cuda", dtype=torch.bool),
        "rank": torch.ones(1, 9, device="cuda", dtype=torch.bool),
        "stride": torch.ones(18, device="cuda", dtype=torch.bool)[::2],
        "device": torch.ones(9, dtype=torch.bool),
    }
    with pytest.raises((TypeError, ValueError), match="exact_block_mask"):
        _validate_inputs(q, k, v, "diag", exact_block_mask=masks[kind])


@pytest.mark.parametrize("splits", [2, 4])
@torch.inference_mode()
def test_sm90_split_kv_protects_disjoint_query_rows(splits):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("KV splitting requires SM90")
    q, k, v = inputs(65537)
    mask = torch.zeros(1025, device="cuda", dtype=torch.bool)
    mask[0] = mask[512] = mask[-1] = True
    actual = sol_attn(q, k, v, kv_splits=splits, exact_block_mask=mask)
    rows = mask.repeat_interleave(64)[: q.shape[1]]
    dense = torch.nn.functional.scaled_dot_product_attention(
        q[:, rows].float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(actual[:, rows].float(), dense, atol=0.02, rtol=0.02)
    assert torch.isfinite(actual).all()
