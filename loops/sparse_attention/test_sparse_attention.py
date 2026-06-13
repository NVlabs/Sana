#!/usr/bin/env python3
"""Independent sparse-attention transform verification.

Run with:
  ~/lustre/miniconda3/envs/sana/bin/python loops/sparse_attention/test_sparse_attention.py
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficiency import Capability, CompositionError, ModelSpec, compose, get_model_spec  # noqa: E402
from efficiency.transforms.sparse_attention import SparseAttention  # noqa: E402


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"ok - {name}")


def local_sparse_spec() -> ModelSpec:
    return ModelSpec(
        name="LocalSparseAttentionFixture",
        capabilities=frozenset(
            {
                Capability.BLOCKS,
                Capability.SWAPPABLE_ATTENTION,
            }
        ),
        get_blocks=lambda transformer: getattr(transformer, "blocks", []),
        seq_dim=1,
    )


def main() -> int:
    cosmos = get_model_spec("Cosmos3")
    check("Cosmos3 spec is registered", cosmos is not None)
    check(
        "Cosmos3 does not declare SWAPPABLE_ATTENTION yet",
        not cosmos.has(Capability.SWAPPABLE_ATTENTION),
    )

    transform = SparseAttention(dense_steps=3, stage2_dense_layers="0")

    try:
        compose([transform], cosmos)
    except CompositionError:
        print("ok - Cosmos3 rejects sparse attention until the seam is wired")
    else:
        raise AssertionError("Cosmos3 should reject sparse attention before wiring")

    spec = local_sparse_spec()
    plan = compose([transform], spec)
    check("sparse-attention transform composes", len(plan.transforms) == 1)
    check("no runtime techniques are installed", len(plan.techniques) == 0)

    env: dict[str, str] = {}
    result = plan.apply_transforms(None, "stage2", env=env)
    check("build transform leaves placeholder transformer unchanged", result is None)

    expected_backends = "transformer=fa,transformer_2=piecewise_attn"
    expected_config = (
        "piecewise_sparsity=0.9,"
        "piecewise_block_size=64,"
        "piecewise_only_video_self_attention=true,"
        "piecewise_stage1_schedule=false,"
        "piecewise_stage1_dense_steps=3,"
        "piecewise_stage2_dense_layers=0,"
        "piecewise_approx_remainder=true,"
        "piecewise_route_mode=score,"
        "piecewise_dense_fallback=fa"
    )
    check(
        "component backend selects transformer_2 piecewise",
        env.get("SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS") == expected_backends,
    )
    check(
        "backend config matches PISA recipe",
        env.get("SGLANG_HQ_ATTENTION_BACKEND_CONFIG") == expected_config,
    )

    print("sparse_attention independent test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
