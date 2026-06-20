#!/usr/bin/env python3
"""Runtime behavior probe for Cosmos3 payload-cache glue.

This is intentionally small and tensor-level. It does not claim PAB public
equivalence; it proves the conservative Cosmos3 payload consumer has the basic
miss/hit/shape-guard behavior that makes a later GPU smoke meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SGLANG_PYTHON = ROOT / "Sol-LTX-Infer" / "python"
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))

import torch  # noqa: E402
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (  # noqa: E402
    _Cosmos3PayloadCache,
)
from sglang.multimodal_gen.runtime.efficiency.techniques.payload_cache import (  # noqa: E402
    PABBroadcastController,
)


def make_tensor(shape: tuple[int, ...], offset: float) -> torch.Tensor:
    size = 1
    for dim in shape:
        size *= int(dim)
    return torch.arange(size, dtype=torch.float32).reshape(shape) + float(offset)


def assert_tensor_equal(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(got, expected):
        raise AssertionError(f"{name}: tensor mismatch")


def probe_attention_cache() -> dict[str, object]:
    cache = _Cosmos3PayloadCache(
        scope="attention_broadcast",
        skip_steps=frozenset({1}),
    )
    hidden = make_tensor((1, 2, 3), 0)
    k_und = make_tensor((1, 2, 1, 3), 10)
    v_und = make_tensor((1, 2, 1, 3), 20)
    cos = make_tensor((1, 2, 1, 3), 30)
    sin = make_tensor((1, 2, 1, 3), 40)

    calls = {"count": 0}

    def run_attention():
        calls["count"] += 1
        return make_tensor((1, 2, 3), 100 * calls["count"])

    cache.begin_step(0)
    seed = cache.forward_attention(
        layer_idx=0,
        cache_key="cond",
        current_step=0,
        run_attention=run_attention,
        hidden_states=hidden,
        k_und=k_und,
        v_und=v_und,
        freqs_cos=cos,
        freqs_sin=sin,
    )
    assert calls["count"] == 1

    cache.begin_step(1)
    hit = cache.forward_attention(
        layer_idx=0,
        cache_key="cond",
        current_step=1,
        run_attention=run_attention,
        hidden_states=hidden,
        k_und=k_und,
        v_und=v_und,
        freqs_cos=cos,
        freqs_sin=sin,
    )
    assert calls["count"] == 1
    assert_tensor_equal("attention hit reuses seed payload", hit, seed)

    cache.begin_step(2)
    miss = cache.forward_attention(
        layer_idx=0,
        cache_key="cond",
        current_step=2,
        run_attention=run_attention,
        hidden_states=hidden,
        k_und=k_und,
        v_und=v_und,
        freqs_cos=cos,
        freqs_sin=sin,
    )
    assert calls["count"] == 2
    if torch.equal(miss, seed):
        raise AssertionError("attention non-skip step unexpectedly reused payload")

    return {"calls": calls["count"], "stats": dict(cache.stats)}


def probe_attention_shape_guard() -> dict[str, object]:
    cache = _Cosmos3PayloadCache(
        scope="attention_broadcast",
        skip_steps=frozenset({1}),
    )
    hidden = make_tensor((1, 2, 3), 0)
    hidden_changed = make_tensor((1, 3, 3), 0)
    k_und = make_tensor((1, 2, 1, 3), 10)
    v_und = make_tensor((1, 2, 1, 3), 20)
    cos = make_tensor((1, 2, 1, 3), 30)
    sin = make_tensor((1, 2, 1, 3), 40)

    calls = {"count": 0}

    def run_attention():
        calls["count"] += 1
        seq_len = hidden.shape[1] if calls["count"] == 1 else hidden_changed.shape[1]
        return make_tensor((1, seq_len, 3), 100 * calls["count"])

    cache.begin_step(0)
    cache.forward_attention(
        layer_idx=0,
        cache_key="cond",
        current_step=0,
        run_attention=run_attention,
        hidden_states=hidden,
        k_und=k_und,
        v_und=v_und,
        freqs_cos=cos,
        freqs_sin=sin,
    )
    cache.begin_step(1)
    cache.forward_attention(
        layer_idx=0,
        cache_key="cond",
        current_step=1,
        run_attention=run_attention,
        hidden_states=hidden_changed,
        k_und=k_und,
        v_und=v_und,
        freqs_cos=cos,
        freqs_sin=sin,
    )
    assert calls["count"] == 2
    return {"calls": calls["count"], "stats": dict(cache.stats)}


def probe_block_cache() -> dict[str, object]:
    cache = _Cosmos3PayloadCache(
        scope="block_layer_feature",
        skip_steps=frozenset({1}),
    )
    hidden = make_tensor((1, 2, 3), 0)
    residual = make_tensor((1, 2, 3), 5)
    k_und = make_tensor((1, 2, 1, 3), 10)
    v_und = make_tensor((1, 2, 1, 3), 20)
    cos = make_tensor((1, 2, 1, 3), 30)
    sin = make_tensor((1, 2, 1, 3), 40)

    calls = {"count": 0}

    def run_block():
        calls["count"] += 1
        return (
            make_tensor((1, 2, 3), 100 * calls["count"]),
            make_tensor((1, 2, 3), 200 * calls["count"]),
        )

    cache.begin_step(0)
    seed_hidden, seed_residual = cache.forward_block(
        layer_idx=1,
        cache_key="uncond",
        current_step=0,
        run_block=run_block,
        hidden_states=hidden,
        residual=residual,
        k_und=k_und,
        v_und=v_und,
        freqs_cos=cos,
        freqs_sin=sin,
    )
    cache.begin_step(1)
    hit_hidden, hit_residual = cache.forward_block(
        layer_idx=1,
        cache_key="uncond",
        current_step=1,
        run_block=run_block,
        hidden_states=hidden,
        residual=residual,
        k_und=k_und,
        v_und=v_und,
        freqs_cos=cos,
        freqs_sin=sin,
    )
    assert calls["count"] == 1
    assert_tensor_equal("block hit hidden", hit_hidden, seed_hidden)
    assert_tensor_equal("block hit residual", hit_residual, seed_residual)
    return {"calls": calls["count"], "stats": dict(cache.stats)}


def probe_pab_attention_controller() -> dict[str, object]:
    cache = _Cosmos3PayloadCache(
        scope="attention_broadcast",
        skip_steps=frozenset(),
        mode="pab",
        attention_kind="cross",
        pab_controller=PABBroadcastController(
            steps=8,
            cross_broadcast=True,
            cross_threshold=[100, 900],
            cross_range=3,
        ),
    )
    hidden = make_tensor((1, 2, 3), 0)
    k_und = make_tensor((1, 2, 1, 3), 10)
    v_und = make_tensor((1, 2, 1, 3), 20)
    cos = make_tensor((1, 2, 1, 3), 30)
    sin = make_tensor((1, 2, 1, 3), 40)
    calls = {"count": 0}

    def run_attention():
        calls["count"] += 1
        return make_tensor((1, 2, 3), 100 * calls["count"])

    cache.begin_step(0, timestep_value=950, num_inference_steps=8)
    seed = cache.forward_attention(
        layer_idx=0,
        cache_key="cond",
        current_step=0,
        run_attention=run_attention,
        hidden_states=hidden,
        k_und=k_und,
        v_und=v_und,
        freqs_cos=cos,
        freqs_sin=sin,
    )
    cache.begin_step(1, timestep_value=500, num_inference_steps=8)
    hit = cache.forward_attention(
        layer_idx=0,
        cache_key="cond",
        current_step=1,
        run_attention=run_attention,
        hidden_states=hidden,
        k_und=k_und,
        v_und=v_und,
        freqs_cos=cos,
        freqs_sin=sin,
    )
    assert calls["count"] == 1
    assert_tensor_equal("PAB attention hit", hit, seed)
    return {"calls": calls["count"], "stats": dict(cache.stats)}


def probe_pab_mlp_controller() -> dict[str, object]:
    cache = _Cosmos3PayloadCache(
        scope="block_layer_feature",
        skip_steps=frozenset(),
        mode="pab",
        pab_controller=PABBroadcastController(
            steps=8,
            mlp_broadcast=True,
            mlp_spatial_broadcast_config={500: {"block": [2], "skip_count": 2}},
        ),
        all_timesteps=[500, 499, 498, 497],
    )
    hidden = make_tensor((1, 2, 3), 0)
    calls = {"count": 0}

    def run_mlp():
        calls["count"] += 1
        return make_tensor((1, 2, 3), 100 * calls["count"])

    cache.begin_step(0, timestep_value=500, num_inference_steps=8)
    seed = cache.forward_mlp(
        layer_idx=2,
        cache_key="cond",
        current_step=0,
        run_mlp=run_mlp,
        hidden_states=hidden,
    )
    cache.begin_step(1, timestep_value=499, num_inference_steps=8)
    hit = cache.forward_mlp(
        layer_idx=2,
        cache_key="cond",
        current_step=1,
        run_mlp=run_mlp,
        hidden_states=hidden,
    )
    assert calls["count"] == 1
    assert_tensor_equal("PAB MLP hit", hit, seed)
    return {"calls": calls["count"], "stats": dict(cache.stats)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = {
        "attention_cache": probe_attention_cache(),
        "attention_shape_guard": probe_attention_shape_guard(),
        "block_cache": probe_block_cache(),
        "pab_attention_controller": probe_pab_attention_controller(),
        "pab_mlp_controller": probe_pab_mlp_controller(),
    }
    result["status"] = "pass"

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
