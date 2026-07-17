#!/usr/bin/env python3
"""Tensor-level behavior probes for generic efficiency candidates.

These probes are not public-original equivalence proofs. They pin the local
generic behavior boundary for completed smoke families: whole-step scheduled
reuse, TeaCache-style accumulated signal reuse, and token-prune scoring plus
scatter restoration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from techniques.schedule import at_steps
from techniques.spec import ModelSpec
from techniques.technique import Capability, TechniqueContext
from techniques.methods.payload_cache import PABBroadcastController
from techniques.methods.step_cache import StepCache
from techniques.methods.teacache import TeaCache, TeaCacheResidual
from techniques.methods.token_prune import (
    CatPruneState,
    TokenPrune,
    cat_convergence_stale_indices,
    keep_indices,
)


def assert_equal(name: str, got, expected) -> None:
    if got != expected:
        raise AssertionError(f"{name}: got {got!r}, expected {expected!r}")


def assert_tensor_equal(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(got, expected):
        raise AssertionError(f"{name}: tensor mismatch")


def probe_step_cache() -> dict[str, object]:
    technique = StepCache(skip="1-2", delta_scale=0.5)
    scratch: dict = {}
    calls = {"count": 0}

    def run_step():
        calls["count"] += 1
        return torch.tensor([float(calls["count"])])

    outputs = []
    for step in range(4):
        ctx = TechniqueContext(step=step, stage="denoise", cache_key="step", scratch=scratch)
        outputs.append(float(technique.on_step(ctx, run_step).item()))

    assert_equal("step_cache call count", calls["count"], 2)
    assert_equal("step_cache outputs", outputs, [1.0, 1.0, 1.0, 2.0])
    return {"calls": calls["count"], "outputs": outputs}


def probe_step_cache_delta() -> dict[str, object]:
    technique = StepCache(skip="2", delta_scale=0.5)
    scratch: dict = {}
    values = iter([torch.tensor([1.0]), torch.tensor([3.0])])
    calls = {"count": 0}

    def run_step():
        calls["count"] += 1
        return next(values)

    outputs = []
    for step in range(3):
        ctx = TechniqueContext(step=step, stage="denoise", cache_key="step", scratch=scratch)
        outputs.append(float(technique.on_step(ctx, run_step).item()))

    assert_equal("step_cache_delta call count", calls["count"], 2)
    assert_equal("step_cache_delta outputs", outputs, [1.0, 3.0, 4.0])
    return {"calls": calls["count"], "outputs": outputs}


def probe_teacache() -> dict[str, object]:
    technique = TeaCache(
        threshold=10.0,
        start_step=1,
        coefficients=[1.0, 0.0],
        max_continuous_hits=1,
        periodic_recompute=0,
    )
    scratch: dict = {}
    calls = {"count": 0}
    outputs = []

    def run_step():
        calls["count"] += 1
        return torch.tensor([float(calls["count"])])

    for step, signal in enumerate(
        [torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0])]
    ):
        scratch[("teacache_signal", "step")] = signal
        ctx = TechniqueContext(step=step, stage="denoise", cache_key="step", scratch=scratch)
        outputs.append(float(technique.on_step(ctx, run_step).item()))

    assert_equal("teacache max-hit call count", calls["count"], 2)
    assert_equal("teacache outputs", outputs, [1.0, 1.0, 2.0])
    return {"calls": calls["count"], "outputs": outputs}


def probe_teacache_threshold() -> dict[str, object]:
    technique = TeaCache(
        threshold=0.01,
        start_step=1,
        coefficients=[1.0, 0.0],
        max_continuous_hits=4,
        periodic_recompute=0,
    )
    scratch: dict = {}
    calls = {"count": 0}

    def run_step():
        calls["count"] += 1
        return torch.tensor([float(calls["count"])])

    outputs = []
    for step, signal in enumerate(
        [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([2.0])]
    ):
        scratch[("teacache_signal", "step")] = signal
        ctx = TechniqueContext(step=step, stage="denoise", cache_key="step", scratch=scratch)
        outputs.append(float(technique.on_step(ctx, run_step).item()))

    assert_equal("teacache threshold call count", calls["count"], 2)
    assert_equal("teacache threshold outputs", outputs, [1.0, 2.0, 2.0])
    return {"calls": calls["count"], "outputs": outputs}


def probe_teacache_unlimited_hits() -> dict[str, object]:
    technique = TeaCache(
        threshold=10.0,
        start_step=1,
        coefficients=[1.0, 0.0],
        max_continuous_hits=0,
        periodic_recompute=0,
    )
    scratch: dict = {}
    calls = {"count": 0}

    def run_step():
        calls["count"] += 1
        return torch.tensor([float(calls["count"])])

    outputs = []
    for step, signal in enumerate(
        [torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0])]
    ):
        scratch[("teacache_signal", "step")] = signal
        ctx = TechniqueContext(step=step, stage="denoise", cache_key="step", scratch=scratch)
        outputs.append(float(technique.on_step(ctx, run_step).item()))

    assert_equal("teacache unlimited-hit call count", calls["count"], 1)
    assert_equal("teacache unlimited-hit outputs", outputs, [1.0, 1.0, 1.0, 1.0])
    return {"calls": calls["count"], "outputs": outputs}


def probe_teacache_residual_replay() -> dict[str, object]:
    technique = TeaCacheResidual(
        threshold=10.0,
        start_step=0,
        coefficients=[1.0, 0.0],
        max_continuous_hits=0,
        periodic_recompute=0,
    )
    scratch: dict = {}
    ctx0 = TechniqueContext(step=0, stage="denoise", cache_key="step", scratch=scratch)
    scratch[("teacache_signal", "step")] = torch.tensor([1.0])
    hidden0 = torch.tensor([[10.0, 20.0]])
    h0, carry0 = technique.before_blocks(ctx0, hidden0)
    assert_tensor_equal("teacache residual first compute input", h0, hidden0)
    technique.after_blocks(ctx0, hidden0 + torch.tensor([[1.0, 2.0]]), carry0)

    ctx1 = TechniqueContext(step=1, stage="denoise", cache_key="step", scratch=scratch)
    scratch[("teacache_signal", "step")] = torch.tensor([1.0])
    hidden1 = torch.tensor([[30.0, 40.0]])
    h1, carry1 = technique.before_blocks(ctx1, hidden1)
    assert_equal("teacache residual second carry", carry1[0], "reuse")
    assert_tensor_equal(
        "teacache residual replay output",
        h1,
        torch.tensor([[31.0, 42.0]]),
    )

    ctx2 = TechniqueContext(step=2, stage="denoise", cache_key="step", scratch=scratch)
    scratch[("teacache_signal", "step")] = torch.tensor([1.0])
    scratch[("teacache_force_compute", "step")] = True
    hidden2 = torch.tensor([[50.0, 60.0]])
    h2, carry2 = technique.before_blocks(ctx2, hidden2)
    assert_equal("teacache residual force-final carry", carry2[0], "compute")
    assert_tensor_equal("teacache residual force-final input", h2, hidden2)
    return {
        "reuse_output": h1.tolist(),
        "force_final_carry": carry2[0],
    }


def probe_pab_controller() -> dict[str, object]:
    controller = PABBroadcastController(
        steps=8,
        cross_broadcast=True,
        cross_threshold=[100, 900],
        cross_range=3,
        mlp_broadcast=True,
        mlp_spatial_broadcast_config={500: {"block": [2], "skip_count": 2}},
    )
    count = 0
    flags = []
    for timestep in [950, 500, 500, 500, 50, 500]:
        flag, count = controller.attention_decision("cross", timestep, count)
        flags.append((flag, count))
    assert_equal(
        "PAB cross decisions",
        flags,
        [(False, 1), (True, 2), (True, 3), (False, 4), (False, 5), (True, 6)],
    )

    seed = controller.mlp_decision(
        timestep=500,
        count=0,
        block_idx=2,
        all_timesteps=[500, 499, 498, 497],
        is_temporal=False,
    )
    hit = controller.mlp_decision(
        timestep=499,
        count=seed[1] or 0,
        block_idx=2,
        all_timesteps=[500, 499, 498, 497],
        is_temporal=False,
    )
    assert_equal("PAB MLP seed", seed, (False, 1, True, [500, 498]))
    assert_equal("PAB MLP hit", hit, (True, 0, False, [500, 498]))
    return {
        "cross_flags": flags,
        "mlp_seed": seed,
        "mlp_hit": hit,
    }


def probe_token_prune_indices() -> dict[str, object]:
    hidden = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 4.0],
                [2.0, 0.0],
                [0.0, 3.0],
            ]
        ]
    )
    feat_idx = keep_indices("feature_norm_prune", 4, 0.5, hidden).tolist()
    uniform_idx = keep_indices("shape_stable_compute_mask", 5, 0.4, torch.zeros(1, 5, 2)).tolist()
    assert_equal("feature_norm top indices", feat_idx, [1, 3])
    assert_equal("shape_stable uniform indices", uniform_idx, [0, 2])
    return {"feature_norm": feat_idx, "shape_stable": uniform_idx}


def probe_cat_prune_selector() -> dict[str, object]:
    cached = torch.tensor(
        [[[10.0, 0.0], [9.0, 0.0], [8.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    current = torch.tensor(
        [[[10.0, 0.0], [9.0, 0.0], [8.0, 0.0], [1.0, 0.0], [1.1, 0.0], [1.2, 0.0]]]
    )
    state = CatPruneState()
    seed = cat_convergence_stale_indices(cached, 0.5, state)
    state.labels = torch.tensor([0, 1, 2, 3, 4, 5])
    state.counts = torch.tensor([10.0, 9.0, 8.0, 0.0, 1.0, 2.0])
    selected = cat_convergence_stale_indices(current, 0.5, state)
    assert_equal("CAT seed full", seed.tolist(), [0, 1, 2, 3, 4, 5])
    assert_equal("CAT selected delta clusters", selected.tolist(), [3, 4, 5])
    assert_equal("CAT state calls", state.calls, 1)
    return {
        "seed": seed.tolist(),
        "selected": selected.tolist(),
        "calls": state.calls,
    }


def probe_token_prune_gather_scatter() -> dict[str, object]:
    spec = ModelSpec(
        name="unit",
        capabilities=frozenset({Capability.PRUNABLE_TOKENS}),
        seq_dim=1,
    )
    technique = TokenPrune(
        keep_ratio=0.5,
        method="feature_norm_prune",
        compensation="prev",
        enabled=at_steps("1", True, False),
    )
    scratch: dict = {}
    ctx0 = TechniqueContext(step=0, stage="denoise", spec=spec, cache_key="tok", scratch=scratch)
    seed = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    full, carry = technique.before_blocks(ctx0, seed)
    assert carry is None
    assert_tensor_equal("token_prune seed identity", full, seed)
    restored_seed = technique.after_blocks(ctx0, full + 10.0, carry)
    assert_tensor_equal("token_prune seed restore identity", restored_seed, full + 10.0)

    ctx1 = TechniqueContext(step=1, stage="denoise", spec=spec, cache_key="tok", scratch=scratch)
    current = torch.tensor([[[1.0], [8.0], [2.0], [7.0]]])
    pruned, carry = technique.before_blocks(ctx1, current)
    assert carry is not None
    assert_equal("token_prune pruned shape", tuple(pruned.shape), (1, 2, 1))
    processed = pruned + 100.0
    restored = technique.after_blocks(ctx1, processed, carry)
    assert_equal("token_prune restored shape", tuple(restored.shape), tuple(current.shape))
    # Kept positions are 1 and 3; dropped positions come from previous full seed.
    expected = torch.tensor([[[11.0], [108.0], [13.0], [107.0]]])
    assert_tensor_equal("token_prune restored values", restored, expected)
    return {
        "pruned_shape": list(pruned.shape),
        "restored_shape": list(restored.shape),
        "restored": restored.flatten().tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = {
        "step_cache": probe_step_cache(),
        "step_cache_delta": probe_step_cache_delta(),
        "teacache": probe_teacache(),
        "teacache_threshold": probe_teacache_threshold(),
        "teacache_unlimited_hits": probe_teacache_unlimited_hits(),
        "teacache_residual_replay": probe_teacache_residual_replay(),
        "pab_controller": probe_pab_controller(),
        "token_prune_indices": probe_token_prune_indices(),
        "cat_prune_selector": probe_cat_prune_selector(),
        "token_prune_gather_scatter": probe_token_prune_gather_scatter(),
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
