#!/usr/bin/env python3
"""Independent token-prune gate for the manifest-derived spec path.

Run with:
    ~/lustre/miniconda3/envs/sana/bin/python loops/token_prune/test_token_prune.py
"""

from __future__ import annotations

import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from efficiency import Capability, ModelSpec, TechniqueContext, compose, const  # noqa: E402
from efficiency.techniques.token_prune import TokenPrune  # noqa: E402


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> int:
    torch.manual_seed(0)

    spec = ModelSpec(
        name="TokenPruneFixture",
        capabilities=frozenset({Capability.PRUNABLE_TOKENS}),
        seq_dim=1,
    )
    check("fixture spec has prunable tokens", spec.has(Capability.PRUNABLE_TOKENS))

    hidden = torch.randn(2, 16, 8)

    # Mirror efficiency/selftest.py [4]: keep_ratio >= 1.0 is OFF/identity.
    off_plan = compose([TokenPrune(keep_ratio=1.0)], spec)
    off_ctx = TechniqueContext(step=3, stage="stage2", spec=spec, cache_key="token_prune")
    h2, carries = off_plan.before_blocks(off_ctx, hidden)
    h2 = off_plan.after_blocks(off_ctx, h2, carries)
    check("ratio=1.0 before/after is identity", torch.equal(h2, hidden))

    # Mirror efficiency/selftest.py [5] against a manifest-derived spec fixture.
    tp = TokenPrune(
        keep_ratio=0.5,
        method="feat_norm",
        compensation="prev",
        enabled=const(True),
    )
    on_plan = compose([tp], spec)

    scratch = {}
    seed_ctx = TechniqueContext(
        step=0,
        stage="stage2",
        spec=spec,
        cache_key="token_prune",
        scratch=scratch,
    )
    seeded, seed_carries = on_plan.before_blocks(seed_ctx, hidden)
    check("step0 seed keeps full S=16", seeded.shape[1] == 16)
    check("step0 seed has no gather carry", seed_carries == [(tp, None)])
    seeded = on_plan.after_blocks(seed_ctx, seeded, seed_carries)
    check("step0 scatter keeps full S=16", seeded.shape[1] == 16)

    prune_ctx = TechniqueContext(
        step=1,
        stage="stage2",
        spec=spec,
        cache_key="token_prune",
        scratch=scratch,
    )
    gathered, prune_carries = on_plan.before_blocks(prune_ctx, hidden)
    check("step1 gathers K=8 tokens", gathered.shape[1] == 8)

    scattered = on_plan.after_blocks(prune_ctx, gathered, prune_carries)
    check("step1 scatters back to S=16", scattered.shape[1] == 16)
    check("step1 preserves dtype", scattered.dtype == hidden.dtype)

    print("token_prune manifest-spec independent gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
