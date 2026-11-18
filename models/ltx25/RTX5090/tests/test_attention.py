from __future__ import annotations

import pytest

from models.ltx25.RTX5090.attention import (
    LAYERS_PER_FORWARD,
    STAGE2_TAUS,
    LTX25Stage2SolAttention,
    route_for_call,
)


def test_stage2_policy_routes_141_sol_calls_and_three_dense_layers() -> None:
    routes = [route_for_call(index) for index in range(3 * LAYERS_PER_FORWARD)]

    assert sum(tau is None for tau, _, _ in routes) == 3
    assert sum(tau is not None for tau, _, _ in routes) == 141
    for forward_index, expected_tau in enumerate(STAGE2_TAUS):
        forward = routes[
            forward_index * LAYERS_PER_FORWARD : (forward_index + 1) * LAYERS_PER_FORWARD
        ]
        assert forward[0] == (None, forward_index, 0)
        assert all(tau == expected_tau for tau, _, _ in forward[1:])


def test_stage2_policy_rejects_an_unexpected_fourth_forward() -> None:
    with pytest.raises(RuntimeError, match="unexpected Stage-2"):
        route_for_call(3 * LAYERS_PER_FORWARD)


def test_attention_dispatch_is_excluded_from_torch_compile() -> None:
    assert getattr(LTX25Stage2SolAttention.__call__, "_torchdynamo_disable", False)
