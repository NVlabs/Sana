from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sol_attn.interface import _validate_inputs, get_sol_attn_backend, sol_attn
from sol_attn.mps import _sink_block_range
from sol_attn.mps.metal import sol_attn_tiled_mps
from sol_attn.mps.preprocess import _routing_thresholds
from sol_attn_backend import _resolve_kv_splits, sol_attn_supported


def test_mps_backend_selection_does_not_require_hardware():
    assert get_sol_attn_backend("mps") == "metal"


def test_default_backend_selects_metal_on_mps_only_system(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_sol_attn_backend() == "metal"


def test_mps_input_contract_does_not_query_cuda():
    tensor = SimpleNamespace(
        ndim=4,
        shape=(1, 65, 2, 128),
        dtype=torch.bfloat16,
        device=torch.device("mps"),
        is_contiguous=lambda: True,
    )
    assert _validate_inputs(tensor, tensor, tensor, "diag") is None


def test_mps_model_adapter_support_and_auto_split(monkeypatch):
    monkeypatch.setattr(torch.mps, "compile_shader", object(), raising=False)
    tensor = SimpleNamespace(
        ndim=4,
        shape=(1, 65, 2, 128),
        dtype=torch.bfloat16,
        device=torch.device("mps"),
    )
    assert sol_attn_supported(tensor)
    assert _resolve_kv_splits(tensor, "auto") == 1
    assert _resolve_kv_splits(tensor, None) == 1


@pytest.mark.parametrize(
    ("tokens", "sink_start", "sink_tokens", "expected"),
    [
        (65, None, 0, (2, 2)),
        (65, 0, 1, (0, 1)),
        (129, None, 1, (2, 3)),
        (129, 63, 2, (0, 2)),
        (129, 64, 64, (1, 2)),
    ],
)
def test_sink_token_range_converts_to_overlapping_blocks(
    tokens, sink_start, sink_tokens, expected
):
    assert _sink_block_range(tokens, sink_start, sink_tokens) == expected


@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
def test_routing_threshold_matches_score_statistics(thresh_type):
    torch.manual_seed(7)
    q_centroids = torch.randn(2, 3, 5, 4)
    k_centroids = torch.randn(2, 3, 5, 4)
    scale = 0.25
    tau = 1.3
    log2_scale = scale * math.log2(math.e)

    actual = _routing_thresholds(
        q_centroids,
        k_centroids,
        scale,
        tau,
        thresh_type,
    )
    raw_mean = (
        q_centroids * k_centroids.mean(dim=2).unsqueeze(2)
    ).sum(dim=-1)
    if thresh_type == "exact":
        scores = torch.matmul(
            q_centroids,
            k_centroids.transpose(-1, -2),
        )
        raw_variance = scores.var(dim=-1, correction=0)
    else:
        variance = k_centroids.var(dim=2, correction=0)
        raw_variance = (
            q_centroids.square() * variance.unsqueeze(2)
        ).sum(dim=-1)
    expected = raw_mean * log2_scale + tau * torch.sqrt(
        raw_variance * (log2_scale * log2_scale) + 1.0e-6
    )
    torch.testing.assert_close(actual, expected)


MPS_AVAILABLE = (
    torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires MPS compile_shader")
@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
def test_full_sink_metal_matches_dense_sdpa(thresh_type):
    torch.manual_seed(11)
    shape = (1, 129, 2, 128)
    q, k, v = [
        torch.randn(shape, device="mps", dtype=torch.bfloat16)
        for _ in range(3)
    ]

    actual = sol_attn(
        q,
        k,
        v,
        thresh_type=thresh_type,
        sink_start=0,
        sink_tokens=shape[1],
    )
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires MPS compile_shader")
def test_bq32_and_bq64_match_with_tail_tokens():
    torch.manual_seed(13)
    shape = (1, 129, 2, 128)
    q, k, v = [
        torch.randn(shape, device="mps", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    kwargs = {
        "tau": 1.3,
        "thresh_type": "diag",
        "sink_blocks": (0, 1),
    }

    bq32 = sol_attn_tiled_mps(q, k, v, query_block_size=32, **kwargs)
    bq64 = sol_attn_tiled_mps(q, k, v, query_block_size=64, **kwargs)
    torch.testing.assert_close(bq32, bq64, rtol=2e-2, atol=2e-2)