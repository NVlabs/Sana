#!/usr/bin/env python3
"""Check KWL backend/compile probes against their public-reference boundary.

``backend_selection_probe`` and ``compile_graph_capture`` cite public backend
families (FlashAttention, CUTLASS, TransformerEngine) as provenance, but the
local candidates are runtime/backend probes: one selects the Cosmos3 transformer
attention backend, the other enables Cosmos3's torch.compile path. This checker
prevents those probes from being mistaken for public kernel ports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KWL_TRANSFORM = ROOT / "efficiency" / "transforms" / "kwl_fusions.py"
RUNTIME_KWL_TRANSFORM = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/efficiency/transforms/kwl_fusions.py"
)
COSMOS3_RUN_SCRIPT = ROOT / "Sol-LTX-Infer" / "scripts" / "run_cosmos3_sglang.sh"
BACKEND_MANIFEST = ROOT / "candidates" / "kwl_fusion" / "backend_selection_probe.toml"
COMPILE_MANIFEST = ROOT / "candidates" / "kwl_fusion" / "compile_graph_capture.toml"
KWL_FLAGS = (
    "SHARE_BLOCK0_SELF_ATTN",
    "SHARE_GUIDANCE_PREFIX",
    "FUSED_QK_ROPE",
    "FUSED_RMS_ADALN",
    "FUSED_ADALN",
    "FUSED_QKNORM_ROPE",
    "FUSED_DUAL_MODULATE",
    "FUSED_CA_DUAL_MODULATE",
    "FUSED_ADA_VALUES_ALL",
    "FUSED_RESIDUAL_GATE",
    "FUSED_FFN_PROJ_IN_GELU",
    "COMPILE_GATE_TO_OUT",
    "FUSED_AUDIO_QKVG",
    "ENABLE_FUSED_QKNORM_ROPE",
    "COMPILE_TILED_VAE",
)


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def source_checks() -> dict[str, bool]:
    generic = read_text(KWL_TRANSFORM)
    runtime = read_text(RUNTIME_KWL_TRANSFORM)
    run_script = read_text(COSMOS3_RUN_SCRIPT)
    backend_manifest = read_text(BACKEND_MANIFEST)
    compile_manifest = read_text(COMPILE_MANIFEST)
    return {
        "backend_manifest_cites_flashattention_cutlass": "Dao-AILab/flash-attention" in backend_manifest
        and "NVIDIA/cutlass" in backend_manifest,
        "backend_manifest_selects_torch_sdpa": 'attention_backend = "torch_sdpa"' in backend_manifest
        and 'attention_backend_fallback = "fa"' in backend_manifest,
        "kwl_transform_emits_component_backend_policy": "SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS" in generic
        and "class KWLBackendSelectionPlan" in generic
        and "def kwl_backend_selection_plan" in generic
        and "SGLANG_HQ_KWL_BACKEND_SELECTION_POLICY" in generic
        and "SGLANG_HQ_KWL_BACKEND_SELECTION_FALLBACK" in generic,
        "kwl_transform_emits_compile_capture_policy": "class KWLCompileCapturePlan" in generic
        and "def kwl_compile_capture_plan" in generic
        and "SGLANG_HQ_KWL_COMPILE_CAPTURE_REGIONS" in generic
        and "SGLANG_HQ_KWL_COMPILE_CAPTURE_BOUNDARY" in generic,
        "kwl_transform_scopes_ltx2_full_bundle_adapter": 'self.kwl_adapter == "ltx2"' in generic
        and "SGLANG_HQ_KWL_ADAPTER" in generic,
        "runtime_kwl_transform_matches_generic_policy": "SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS" in runtime
        and "class KWLBackendSelectionPlan" in runtime
        and "class KWLCompileCapturePlan" in runtime
        and "SGLANG_HQ_KWL_BACKEND_SELECTION_POLICY" in runtime
        and "SGLANG_HQ_KWL_BACKEND_SELECTION_FALLBACK" in runtime,
        "runtime_kwl_transform_scopes_ltx2_full_bundle_adapter": 'self.kwl_adapter == "ltx2"' in runtime
        and "SGLANG_HQ_KWL_ADAPTER" in runtime,
        "cosmos_run_script_consumes_component_backend_policy": "--component-attention-backends" in run_script
        and "SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS" in run_script,
        "compile_manifest_cites_cutlass_te": "NVIDIA/cutlass" in compile_manifest
        and "NVIDIA/TransformerEngine" in compile_manifest,
        "compile_manifest_selects_compile_flags": 'flags = ["COMPILE_GATE_TO_OUT", "COMPILE_TILED_VAE"]'
        in compile_manifest
        and 'SGLANG_HQ_CUDA_GRAPH_PROBE = "1"' in compile_manifest,
        "kwl_transform_emits_compile_flags": "COMPILE_GATE_TO_OUT" in generic
        and "COMPILE_TILED_VAE" in generic
        and "SGLANG_HQ_KWL_{f}" in generic,
        "cosmos_run_script_consumes_torch_compile_probe": "SGLANG_HQ_CUDA_GRAPH_PROBE" in run_script
        and "SGLANG_HQ_ENABLE_TORCH_COMPILE" in run_script
        and "--enable-torch-compile" in run_script,
        "local_backend_probe_is_not_public_flashattention_or_cutlass_port": 'attention_backend = "torch_sdpa"'
        in backend_manifest,
        "local_compile_probe_is_not_public_te_or_cutlass_kernel_port": "torch.compile" in run_script
        or "--enable-torch-compile" in run_script,
    }


def transform_env_probe() -> dict[str, Any]:
    from techniques import Capability, ModelSpec, compose
    from techniques.transforms.kwl_fusions import (
        KWLFusions,
        kwl_backend_selection_plan,
        kwl_compile_capture_plan,
    )

    spec = ModelSpec(
        name="KWLProbeSpec",
        capabilities=frozenset({Capability.BLOCKS}),
        get_blocks=lambda transformer: getattr(transformer, "blocks", ()),
    )
    generic_env: dict[str, str] = {}
    compose([KWLFusions()], spec).apply_transforms(None, stage="stage2", env=generic_env)
    ltx2_env: dict[str, str] = {}
    compose([KWLFusions(kwl_adapter="ltx2")], spec).apply_transforms(
        None, stage="stage2", env=ltx2_env
    )
    backend_plan = kwl_backend_selection_plan(
        component="transformer",
        preferred_backend="torch_sdpa",
        fallback_backend="fa",
        policy_name="cosmos3_transformer_torch_sdpa",
    )
    backend_env: dict[str, str] = {}
    compose(
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
    ).apply_transforms(None, stage="stage2", env=backend_env)
    compile_plan = kwl_compile_capture_plan(
        ("COMPILE_GATE_TO_OUT", "COMPILE_TILED_VAE")
    )
    compile_env: dict[str, str] = {}
    compose(
        [KWLFusions(flags=("COMPILE_GATE_TO_OUT", "COMPILE_TILED_VAE"))],
        spec,
    ).apply_transforms(None, stage="stage2", env=compile_env)
    flag_keys = [f"SGLANG_HQ_KWL_{flag}" for flag in KWL_FLAGS]
    return {
        "generic_env_has_all_flag_keys": all(key in generic_env for key in flag_keys),
        "generic_env_has_no_enabled_model_specific_flags": all(
            generic_env.get(key) == "0" for key in flag_keys
        ),
        "generic_env_has_no_ltx2_adapter": "SGLANG_HQ_KWL_ADAPTER" not in generic_env,
        "ltx2_adapter_env_has_adapter_marker": ltx2_env.get("SGLANG_HQ_KWL_ADAPTER")
        == "ltx2",
        "ltx2_adapter_env_enables_full_bundle": all(
            ltx2_env.get(key) == "1" for key in flag_keys
        ),
        "backend_plan": {
            "policy_id": backend_plan.policy_id,
            "env": backend_plan.as_env(),
            "compose_env": backend_env,
            "matches_policy_env": all(
                backend_env.get(key) == value
                for key, value in backend_plan.as_env().items()
            ),
        },
        "compile_capture_plan": {
            "regions": list(compile_plan.regions),
            "env": compile_plan.as_env(),
            "compose_env": compile_env,
            "matches_policy_env": all(
                compile_env.get(key) == value
                for key, value in compile_plan.as_env().items()
            ),
        },
    }


def candidate_alignment() -> dict[str, dict[str, Any]]:
    backend = load_toml(BACKEND_MANIFEST)
    compile_ = load_toml(COMPILE_MANIFEST)
    backend_params = backend.get("efficiency", {}).get("params", {})
    compile_params = compile_.get("efficiency", {}).get("params", {})
    return {
        "backend_selection_probe": {
            "manifest": str(BACKEND_MANIFEST),
            "selected_backend": backend_params.get("attention_backend"),
            "fallback_backend": backend_params.get("attention_backend_fallback"),
            "matches_public_flashattention_kernel": False,
            "matches_public_cutlass_kernel": False,
            "matches_full_public_backend_implementation": False,
            "known_difference": (
                "This row selects Cosmos3 transformer=torch_sdpa through the "
                "generic component-backend policy. The pure policy is preserved, "
                "but the public references are backend families; no FlashAttention "
                "or CUTLASS kernel is implemented by this candidate."
            ),
        },
        "compile_graph_capture": {
            "manifest": str(COMPILE_MANIFEST),
            "flags": list(compile_params.get("flags", [])),
            "uses_cosmos3_torch_compile_path": True,
            "matches_public_transformerengine_kernel": False,
            "matches_public_cutlass_graph_capture": False,
            "matches_full_public_compile_or_graph_capture_implementation": False,
            "known_difference": (
                "This row enables Cosmos3's existing torch.compile probe path "
                "and a generic compile/capture-region policy. It does not add a "
                "TransformerEngine fused op, a CUTLASS kernel, or a standalone "
                "CUDA graph capture implementation."
            ),
        },
    }


def probe() -> dict[str, Any]:
    checks = source_checks()
    env_probe = transform_env_probe()
    status = (
        "pass"
        if checks["kwl_transform_emits_component_backend_policy"]
        and checks["kwl_transform_emits_compile_capture_policy"]
        and checks["kwl_transform_scopes_ltx2_full_bundle_adapter"]
        and checks["runtime_kwl_transform_matches_generic_policy"]
        and checks["runtime_kwl_transform_scopes_ltx2_full_bundle_adapter"]
        and env_probe["generic_env_has_all_flag_keys"]
        and env_probe["generic_env_has_no_enabled_model_specific_flags"]
        and env_probe["generic_env_has_no_ltx2_adapter"]
        and env_probe["ltx2_adapter_env_has_adapter_marker"]
        and env_probe["ltx2_adapter_env_enables_full_bundle"]
        and env_probe["backend_plan"]["matches_policy_env"]
        and env_probe["compile_capture_plan"]["matches_policy_env"]
        else "fail"
    )
    return {
        "status": status,
        "public_reference_role": {
            "backend_selection_probe": "FlashAttention/CUTLASS are backend-family provenance, not a local torch_sdpa public port.",
            "compile_graph_capture": "CUTLASS/TransformerEngine motivate backend compile/fusion boundaries, not this torch.compile probe as public equivalence.",
        },
        "checks": checks,
        "transform_env_probe": env_probe,
        "candidate_manifest_alignment": candidate_alignment(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = probe()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
