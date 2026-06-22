#!/usr/bin/env python3
"""Independent CPU-only check for the kwl_fusion loop."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from efficiency import Capability, ModelSpec, Seam, TransformPhase, compose  # noqa: E402
from efficiency.transforms.kwl_fusions import KWLFusions  # noqa: E402


EXPECTED_ENV = {
    "SGLANG_HQ_VARIANT": "kwl",
    "SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN": "0",
    "SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX": "0",
    "SGLANG_HQ_KWL_FUSED_QK_ROPE": "0",
    "SGLANG_HQ_KWL_FUSED_RMS_ADALN": "0",
    "SGLANG_HQ_KWL_FUSED_ADALN": "0",
    "SGLANG_HQ_KWL_FUSED_QKNORM_ROPE": "0",
    "SGLANG_HQ_KWL_FUSED_DUAL_MODULATE": "0",
    "SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE": "0",
    "SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL": "0",
    "SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE": "0",
    "SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU": "0",
    "SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT": "0",
    "SGLANG_HQ_KWL_FUSED_AUDIO_QKVG": "0",
    "SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE": "0",
    "SGLANG_HQ_KWL_COMPILE_TILED_VAE": "0",
}
KWL_KEYS = tuple(key for key in EXPECTED_ENV if key.startswith("SGLANG_HQ_KWL_"))


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


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
    check("KWL generic env bundle exact", env == EXPECTED_ENV)
    check("generic KWL default does not request LTX2 adapter", "SGLANG_HQ_KWL_ADAPTER" not in env)
    check("CA dual modulation flag off by default", env["SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE"] == "0")

    ltx2_env: dict[str, str] = {}
    ltx2 = compose([KWLFusions(kwl_adapter="ltx2")], spec)
    ltx2.apply_transforms(None, stage="stage2", env=ltx2_env)
    expected_ltx2 = {
        "SGLANG_HQ_VARIANT": "kwl",
        "SGLANG_HQ_KWL_ADAPTER": "ltx2",
        **{key: "1" for key in KWL_KEYS},
        "SGLANG_HQ_KWL_COMPILE_CAPTURE_POLICY": "shape_stable_regions",
        "SGLANG_HQ_KWL_COMPILE_CAPTURE_REGIONS": "gate_to_out,tiled_vae",
        "SGLANG_HQ_KWL_COMPILE_CAPTURE_FALLBACK": "eager",
        "SGLANG_HQ_KWL_COMPILE_CAPTURE_PUBLIC_FAMILIES": (
            "CUDA graph,CUTLASS,TransformerEngine"
        ),
        "SGLANG_HQ_KWL_COMPILE_CAPTURE_BOUNDARY": "policy_not_kernel_port",
    }
    check("explicit LTX2 KWL adapter env exact", ltx2_env == expected_ltx2)

    subset_env: dict[str, str] = {}
    subset = compose([KWLFusions(flags=("FUSED_CA_DUAL_MODULATE",))], spec)
    subset.apply_transforms(None, stage="stage2", env=subset_env)
    expected_subset = {"SGLANG_HQ_VARIANT": "kwl", **{key: "0" for key in KWL_KEYS}}
    expected_subset["SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE"] = "1"
    check("KWL subset env exact", subset_env == expected_subset)

    backend_env: dict[str, str] = {}
    backend_policy = compose(
        [
            KWLFusions(
                flags=(),
                attention_backend_component="transformer",
                attention_backend="torch_sdpa",
                attention_backend_fallback="fa",
                backend_policy_name="cosmos3_transformer_torch_sdpa",
            )
        ],
        spec,
    )
    backend_policy.apply_transforms(None, stage="stage2", env=backend_env)
    check(
        "KWL backend policy env set",
        backend_env["SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS"]
        == "transformer=torch_sdpa"
        and backend_env["SGLANG_HQ_KWL_BACKEND_SELECTION_POLICY"]
        == "cosmos3_transformer_torch_sdpa"
        and backend_env["SGLANG_HQ_KWL_BACKEND_SELECTION_FALLBACK"] == "fa",
    )

    before = dict(os.environ)
    check("smoke import does not mutate process env", dict(os.environ) == before)


if __name__ == "__main__":
    main()
