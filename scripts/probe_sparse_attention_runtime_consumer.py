#!/usr/bin/env python3
"""Probe the PiecewiseAttention runtime consumer for sparse route policies.

This is runtime-glue evidence. It proves the Cosmos3-compatible
``piecewise_attn`` backend can consume the model-agnostic sparse policy layer
when ``piecewise_route_mode`` selects one of the pure policy modes. It does not
claim GPU quality or line-for-line public-original equivalence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "Sol-LTX-Infer/python"
for item in (str(RUNTIME_ROOT), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (  # noqa: E402
    _PIECEWISE_POLICY_ROUTE_MODES,
    _piecewise_policy_block_indices,
    PiecewiseAttentionImpl,
    taylor_error_block_indices,
)


def _fixtures() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(13)
    qc = torch.randn(1, 4, 6, 8)
    kc = torch.randn(1, 4, 8, 8)
    k_var = torch.linspace(0.5, 1.5, steps=8).view(1, 1, 8).expand(1, 4, 8)
    return qc, kc, k_var


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeDevice:
    type = "cuda"


class _FakeTensor:
    device = _FakeDevice()

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def _probe_cross_attention_guard() -> dict[str, object]:
    impl = object.__new__(PiecewiseAttentionImpl)
    impl.dropout = 0.0
    impl.causal = False
    impl.route_mode = "proxy_mask_prediction"
    impl.only_video_self_attention = True
    impl.allow_qk_mismatch = False
    impl.allow_gqa = False
    impl.prefix = "gen_layers.0.cross_attention.attn.impl"
    query = _FakeTensor((1, 384, 32, 64))
    key = _FakeTensor((1, 128, 4, 64))
    value = _FakeTensor((1, 128, 4, 64))
    policy_reason = impl._piecewise_fallback_reason(query, key, value, 0.25)
    _assert(policy_reason == "", "policy route should allow cross-attention shape")

    impl.route_mode = "score"
    score_reason = impl._piecewise_fallback_reason(query, key, value, 0.25)
    _assert(
        score_reason == "qk_sequence_mismatch",
        "score route should keep the original q/k mismatch guard",
    )
    return {
        "policy_route_reason": policy_reason,
        "score_route_reason": score_reason,
    }


def _probe_mode(mode: str) -> dict[str, object]:
    qc, kc, k_var = _fixtures()
    result = _piecewise_policy_block_indices(
        route_mode=mode,
        qc=qc,
        kc=kc,
        k_var=k_var,
        density=0.25,
        scale=0.5,
        step=1,
        layer_idx=2,
        frame_size=2,
    )
    indices = result["indices"]
    _assert(tuple(indices.shape[:3]) == (1, 4, 6), f"{mode} indices shape")
    _assert(indices.dtype == torch.int32, f"{mode} indices dtype")
    _assert(result["mode"] == mode, f"{mode} canonical route mode")
    _assert(result["mask"] is not None, f"{mode} should consume policy mask")
    _assert(0.0 < float(result["density"]) < 1.0, f"{mode} sparse density")
    return {
        "selected_mode": result["selected_mode"],
        "indices_shape": list(indices.shape),
        "density": round(float(result["density"]), 6),
        "reused": bool(result["reused"]),
    }


def _probe_score_fallback() -> dict[str, object]:
    qc, kc, k_var = _fixtures()
    result = _piecewise_policy_block_indices(
        route_mode="score",
        qc=qc,
        kc=kc,
        k_var=k_var,
        density=0.25,
        scale=0.5,
    )
    expected = taylor_error_block_indices(qc, kc, k_var, density=0.25, scale=0.5)
    _assert(result["mask"] is None, "score should keep existing Taylor route")
    _assert(torch.equal(result["indices"], expected), "score route changed")
    return {
        "selected_mode": result["selected_mode"],
        "indices_shape": list(result["indices"].shape),
        "density": round(float(result["density"]), 6),
    }


def _probe_online_reuse() -> dict[str, object]:
    qc, kc, k_var = _fixtures()
    first = _piecewise_policy_block_indices(
        route_mode="online_mask_search_reuse",
        qc=qc,
        kc=kc,
        k_var=k_var,
        density=0.25,
        scale=0.5,
        drift=1.0,
    )
    reused = _piecewise_policy_block_indices(
        route_mode="online_mask_search_reuse",
        qc=qc,
        kc=kc,
        k_var=k_var,
        density=0.25,
        scale=0.5,
        previous_mask=first["mask"],
        drift=0.001,
        reuse_threshold=0.05,
    )
    _assert(bool(reused["reused"]), "online route did not reuse low-drift mask")
    _assert(
        torch.equal(first["indices"], reused["indices"]),
        "reused online route changed indices",
    )
    return {
        "first_reused": bool(first["reused"]),
        "low_drift_reused": bool(reused["reused"]),
        "indices_shape": list(reused["indices"].shape),
    }


def _probe_svg_sample_mse_route() -> dict[str, object]:
    torch.manual_seed(29)
    qc = torch.randn(1, 2, 4, 4)
    kc = torch.randn(1, 2, 5, 4)
    vc = torch.randn(1, 2, 5, 4)
    k_var = torch.ones(1, 2, 5)
    result = _piecewise_policy_block_indices(
        route_mode="spatial_temporal_head_routing",
        qc=qc,
        kc=kc,
        k_var=k_var,
        value_centroids=vc,
        density=0.5,
        scale=0.5,
        frame_size=2,
    )
    _assert(
        result["selected_mode"] == "svg_sample_mse_head_selection",
        "SVG route did not consume value centroids for sample-MSE selection",
    )
    _assert(result["mask"] is not None, "SVG sample-MSE route did not return a mask")
    _assert(tuple(result["indices"].shape[:3]) == (1, 2, 4), "SVG indices shape")
    _assert(
        bool(result["mask"][:, :, :, 0].all().item()),
        "SVG q/k mismatch route did not keep KV-prefix sink block",
    )
    return {
        "selected_mode": result["selected_mode"],
        "density": round(float(result["density"]), 6),
        "indices_shape": list(result["indices"].shape),
        "keeps_prefix_sink": bool(result["mask"][:, :, :, 0].all().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    modes = sorted(_PIECEWISE_POLICY_ROUTE_MODES)
    result = {
        "cross_attention_guard": _probe_cross_attention_guard(),
        "score_fallback": _probe_score_fallback(),
        "policy_modes": {mode: _probe_mode(mode) for mode in modes},
        "online_mask_search_reuse_state": _probe_online_reuse(),
        "svg_sample_mse_route": _probe_svg_sample_mse_route(),
        "status": "pass",
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
