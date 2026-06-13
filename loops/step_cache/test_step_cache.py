#!/usr/bin/env python3
"""Independent StepCache verification for the step_cache loop."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from efficiency import TechniqueContext, at_steps, by_stage, compose, const, get_model_spec  # noqa: E402
from efficiency.techniques.step_cache import StepCache  # noqa: E402


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"  PASS  {name}")


def main() -> int:
    print("[step_cache] compose and schedule")
    spec = get_model_spec("Cosmos3")
    check("Cosmos3 spec registered", spec is not None and spec.name == "Cosmos3")

    skip = by_stage({"stage1": at_steps("16-28", True, False)}, default=False)
    step_cache = StepCache(skip=skip, delta_scale=0.0)
    plan = compose([step_cache], spec)
    check("StepCache composes against Cosmos3", plan.techniques == [step_cache])

    check(
        "active stage1 step20 (skip cluster)",
        step_cache.is_active(TechniqueContext(step=20, stage="stage1", spec=spec)),
    )
    check(
        "active stage1 step16 (cluster start)",
        step_cache.is_active(TechniqueContext(step=16, stage="stage1", spec=spec)),
    )
    check(
        "active stage1 step28 (cluster end)",
        step_cache.is_active(TechniqueContext(step=28, stage="stage1", spec=spec)),
    )
    check(
        "inactive stage1 step15",
        not step_cache.is_active(TechniqueContext(step=15, stage="stage1", spec=spec)),
    )
    check(
        "inactive stage1 step29",
        not step_cache.is_active(TechniqueContext(step=29, stage="stage1", spec=spec)),
    )
    check(
        "inactive stage2 step20",
        not step_cache.is_active(TechniqueContext(step=20, stage="stage2", spec=spec)),
    )

    print("[step_cache] OFF == identity")
    torch.manual_seed(0)
    hidden = torch.randn(2, 4)
    calls = {"count": 0}

    def baseline_step():
        calls["count"] += 1
        return hidden.clone()

    off_plan = compose([StepCache(skip=const(False))], spec)
    off_out = off_plan.on_step(
        TechniqueContext(step=20, stage="stage1", spec=spec, cache_key="sample", scratch={}),
        baseline_step,
    )
    check("disabled cache calls run_step once", calls["count"] == 1)
    check("disabled cache returns byte-identical tensor", torch.equal(off_out, hidden))

    print("[step_cache] scheduled reuse")
    scratch = {}
    seed_value = torch.tensor([1.0, 2.0, 3.0])
    seed_calls = {"count": 0}

    def seed_step():
        seed_calls["count"] += 1
        return seed_value.clone()

    seed_ctx = TechniqueContext(
        step=15,
        stage="stage1",
        spec=spec,
        cache_key="sample",
        scratch=scratch,
    )
    seed_out = plan.on_step(seed_ctx, seed_step)
    check("pre-cluster seed computes once", seed_calls["count"] == 1)
    check("pre-cluster seed output is baseline", torch.equal(seed_out, seed_value))

    first_active_value = torch.tensor([4.0, 5.0, 6.0])
    first_active_calls = {"count": 0}

    def first_active_step():
        first_active_calls["count"] += 1
        return first_active_value.clone()

    first_active_ctx = TechniqueContext(
        step=16,
        stage="stage1",
        spec=spec,
        cache_key="sample",
        scratch=scratch,
    )
    first_active_out = plan.on_step(first_active_ctx, first_active_step)
    check("first active cluster step computes to seed cache", first_active_calls["count"] == 1)
    check(
        "first active cluster step returns baseline output",
        torch.equal(first_active_out, first_active_value),
    )

    skipped_calls = {"count": 0}

    def should_skip():
        skipped_calls["count"] += 1
        return torch.tensor([-1.0])

    skip_ctx = TechniqueContext(
        step=17,
        stage="stage1",
        spec=spec,
        cache_key="sample",
        scratch=scratch,
    )
    skip_out = plan.on_step(skip_ctx, should_skip)
    check("subsequent skip-cluster step reuses cached output", torch.equal(skip_out, first_active_value))
    check("skip-cluster step does not call run_step", skipped_calls["count"] == 0)

    post_value = torch.tensor([9.0, 8.0, 7.0])
    post_calls = {"count": 0}

    def post_step():
        post_calls["count"] += 1
        return post_value.clone()

    post_ctx = TechniqueContext(
        step=29,
        stage="stage1",
        spec=spec,
        cache_key="sample",
        scratch=scratch,
    )
    post_out = plan.on_step(post_ctx, post_step)
    check("outside cluster recomputes", post_calls["count"] == 1)
    check("outside cluster returns new baseline output", torch.equal(post_out, post_value))

    print("\n=== step_cache test passed ===")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())
