#!/usr/bin/env python3
"""Independent CPU-only check for the kwl_fusion loop."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from efficiency import Capability, ModelSpec, Seam, TransformPhase, compose  # noqa: E402
from efficiency.transforms.kwl_fusions import KWLFusions  # noqa: E402


EXPECTED_ENV = {
    "SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN": "1",
    "SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX": "1",
    "SGLANG_HQ_KWL_FUSED_QK_ROPE": "1",
    "SGLANG_HQ_KWL_FUSED_RMS_ADALN": "1",
    "SGLANG_HQ_KWL_FUSED_ADALN": "1",
    "SGLANG_HQ_KWL_FUSED_QKNORM_ROPE": "1",
    "SGLANG_HQ_KWL_FUSED_DUAL_MODULATE": "1",
    "SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE": "1",
    "SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL": "1",
    "SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE": "1",
    "SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU": "1",
    "SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT": "1",
    "SGLANG_HQ_KWL_FUSED_AUDIO_QKVG": "1",
    "SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE": "1",
    "SGLANG_HQ_KWL_COMPILE_TILED_VAE": "1",
}


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def smoke_import_reference_ops() -> None:
    path = Path(__file__).resolve().parents[2] / "reference" / "kwl_fusion" / "kwl_ops.py"
    spec = importlib.util.spec_from_file_location("kwl_reference_ops", path)
    check("reference kwl_ops import spec", spec is not None and spec.loader is not None)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    check("reference exposes installer", hasattr(module, "build_official_kwl_module_op"))


def main() -> None:
    spec = ModelSpec(
        name="Cosmos3KWLFixture",
        capabilities=frozenset({Capability.BLOCKS}),
        get_blocks=lambda transformer: getattr(transformer, "gen_layers", ()),
        seq_dim=1,
    )

    transform = KWLFusions()
    check("KWL is build transform", transform.phase == TransformPhase.BUILD)
    check("KWL writes kernel-fusion seam", transform.writes == frozenset({Seam.KERNEL_FUSION}))

    plan = compose([transform], spec)
    check("compose returns one transform", len(plan.transforms) == 1 and not plan.techniques)

    env: dict[str, str] = {}
    sentinel = object()
    result = plan.apply_transforms(sentinel, stage="stage2", env=env)
    check("apply_transforms leaves object unchanged", result is sentinel)
    check("KWL env bundle exact", env == EXPECTED_ENV)
    check("CA dual modulation flag set", env["SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE"] == "1")

    subset_env: dict[str, str] = {}
    subset = compose([KWLFusions(flags=("FUSED_CA_DUAL_MODULATE",))], spec)
    subset.apply_transforms(None, stage="stage2", env=subset_env)
    check(
        "KWL subset env exact",
        subset_env == {"SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE": "1"},
    )

    before = dict(os.environ)
    smoke_import_reference_ops()
    check("smoke import does not mutate process env", dict(os.environ) == before)


if __name__ == "__main__":
    main()
