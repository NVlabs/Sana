#!/usr/bin/env python3
"""Pin the NVFP4/TransformerEngine boundary for model-agnostic candidates.

The NVFP4 candidates cite TransformerEngine, ModelOpt, and CUTLASS. Cosmos3 is a
validation target, not the algorithm boundary. This probe separates the generic
FP4/NVFP4 linear consumer and recipe axes that are already wired from
LTX2-specific TE fused-epilogue glue that does not semantically map to Cosmos3's
FFN and must not remain active in the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNTIME_PYTHON = ROOT / "Sol-LTX-Infer/.conda/ltx23/bin/python"
TE_MANIFEST = ROOT / "candidates/nvfp4_ffn/te_recipe_variant.toml"
CONSERVATIVE_MANIFEST = ROOT / "candidates/nvfp4_ffn/conservative_ffn_nvfp4.toml"
PROFILED_MANIFEST = ROOT / "candidates/nvfp4_ffn/profiled_hot_linear_nvfp4.toml"
DENSE_GUARD_MANIFEST = ROOT / "candidates/nvfp4_ffn/dense_guard_policy.toml"
BACKEND_PADDING_MANIFEST = ROOT / "candidates/nvfp4_ffn/backend_padding_policy.toml"
GENERIC_NVFP4_TRANSFORM = ROOT / "efficiency/transforms/nvfp4_ffn.py"
RUNTIME_NVFP4_TRANSFORM = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/efficiency/transforms/nvfp4_ffn.py"
)
MODELOPT_FP4 = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py"
)
TRANSFORMER_LOAD_UTILS = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py"
)
COSMOS3_MODEL = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py"
)
LTX2_MODEL = (
    ROOT / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py"
)
LAUNCH_CANDIDATE = ROOT / "scripts/launch_candidate.py"


NON_TE_NVFP4_CANDIDATES = {
    "conservative_ffn_nvfp4": CONSERVATIVE_MANIFEST,
    "profiled_hot_linear_nvfp4": PROFILED_MANIFEST,
    "dense_guard_policy": DENSE_GUARD_MANIFEST,
    "backend_padding_policy": BACKEND_PADDING_MANIFEST,
}


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _first_line(value: str) -> str:
    for line in value.splitlines():
        line = line.strip()
        if line:
            return line[:240]
    return ""


def _last_line(value: str) -> str:
    for line in reversed(value.splitlines()):
        line = line.strip()
        if line:
            return line[:240]
    return ""


def transformerengine_import_check() -> dict[str, Any]:
    python = RUNTIME_PYTHON if RUNTIME_PYTHON.exists() else Path(os.environ.get("PYTHON", "python3"))
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    try:
        proc = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import transformer_engine.pytorch as te; "
                    "from transformer_engine.common.recipe import NVFP4BlockScaling; "
                    "print(te.Linear.__name__, NVFP4BlockScaling.__name__)"
                ),
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return {
            "python": str(python),
            "available": False,
            "returncode": None,
            "stdout_first_line": "",
            "stderr_first_line": type(exc).__name__ + ": " + str(exc)[:200],
            "stderr_last_line": type(exc).__name__ + ": " + str(exc)[:200],
        }
    return {
        "python": str(python),
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_first_line": _first_line(proc.stdout),
        "stderr_first_line": _first_line(proc.stderr),
        "stderr_last_line": _last_line(proc.stderr),
    }


def source_checks() -> dict[str, bool]:
    te_manifest = read_text(TE_MANIFEST)
    profiled_manifest = read_text(PROFILED_MANIFEST)
    generic_transform = read_text(GENERIC_NVFP4_TRANSFORM)
    runtime_transform = read_text(RUNTIME_NVFP4_TRANSFORM)
    modelopt = read_text(MODELOPT_FP4)
    load_utils = read_text(TRANSFORMER_LOAD_UTILS)
    cosmos3 = read_text(COSMOS3_MODEL)
    ltx2 = read_text(LTX2_MODEL)
    launch = read_text(LAUNCH_CANDIDATE)
    return {
        "te_manifest_cites_transformerengine": "TransformerEngine" in te_manifest
        and "docs.nvidia.com/deeplearning/transformer-engine" in te_manifest
        and "github.com/NVIDIA/TransformerEngine" in te_manifest,
        "te_manifest_preserves_only_generic_recipe_axis": "row_scaled_activation = true" in te_manifest
        and "fused_proj_in_gelu = false" in te_manifest
        and "fused_proj_out_bias_gate = false" in te_manifest,
        "te_manifest_explicitly_disables_ltx2_adapter": 'te_adapter = ""' in te_manifest,
        "generic_transform_emits_te_recipe_flags": "SGLANG_HQ_NVFP4_ROW_SCALED_ACTIVATION" in generic_transform
        and "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU" in generic_transform
        and "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE" in generic_transform,
        "generic_transform_emits_generic_recipe_flags": "SGLANG_HQ_NVFP4_DISABLE_RHT" in generic_transform
        and "SGLANG_HQ_NVFP4_DISABLE_STOCHASTIC_ROUNDING" in generic_transform
        and "SGLANG_HQ_NVFP4_DISABLE_2D_QUANTIZATION" in generic_transform
        and "SGLANG_HQ_NVFP4_PAD_M_TO" in generic_transform,
        "generic_transform_emits_profile_selector_env": "select_profiled_nvfp4_layers" in generic_transform
        and "SGLANG_HQ_NVFP4_PROFILED_LAYERS" in generic_transform
        and "SGLANG_HQ_NVFP4_PROFILE_SOURCE" in generic_transform,
        "generic_transform_scopes_ltx2_adapter_env": 'self.te_adapter == "ltx2"' in generic_transform
        and 'e["SGLANG_LTX2_TE_NVFP4_VIDEO_FFN"]' in generic_transform,
        "runtime_transform_mirrors_te_recipe_flags": "SGLANG_HQ_NVFP4_ROW_SCALED_ACTIVATION" in runtime_transform
        and "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU" in runtime_transform
        and "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE" in runtime_transform,
        "runtime_transform_mirrors_generic_recipe_flags": "SGLANG_HQ_NVFP4_DISABLE_RHT" in runtime_transform
        and "SGLANG_HQ_NVFP4_DISABLE_STOCHASTIC_ROUNDING" in runtime_transform
        and "SGLANG_HQ_NVFP4_DISABLE_2D_QUANTIZATION" in runtime_transform
        and "SGLANG_HQ_NVFP4_PAD_M_TO" in runtime_transform,
        "runtime_transform_mirrors_profile_selector_env": "select_profiled_nvfp4_layers" in runtime_transform
        and "SGLANG_HQ_NVFP4_PROFILED_LAYERS" in runtime_transform
        and "SGLANG_HQ_NVFP4_PROFILE_SOURCE" in runtime_transform,
        "runtime_transform_scopes_ltx2_adapter_env": 'self.te_adapter == "ltx2"' in runtime_transform
        and 'e["SGLANG_LTX2_TE_NVFP4_VIDEO_FFN"]' in runtime_transform,
        "modelopt_online_fp4_consumer_wired": "_online_quantize_nvfp4_weight" in modelopt
        and "class ModelOptFp4LinearMethod" in modelopt
        and "modelopt_fp4_quantize_activation" in modelopt
        and "modelopt_fp4_apply_quantized_linear" in modelopt,
        "modelopt_profiled_layer_consumer_wired": "self.profiled_layers" in modelopt
        and "SGLANG_HQ_NVFP4_PROFILED_LAYERS" in load_utils
        and "in_profiled_layer" in modelopt,
        "modelopt_dense_step_guard_wired": "_current_forward_step" in modelopt
        and "dense_steps" in modelopt
        and "SGLANG_HQ_NVFP4_DENSE_STEPS" in load_utils,
        "profiled_manifest_uses_profile_scores_not_static_dense_layers": "profile_layer_scores" in profiled_manifest
        and "profile_keep_ratio" in profiled_manifest
        and "dense_layers" not in profiled_manifest,
        "loader_selects_modelopt_fp4_online_consumer": "server_args.quantization == \"modelopt_fp4\"" in load_utils
        and "\"online_quantization\": True" in load_utils,
        "ltx2_has_te_nvfp4_adapter": "def _ltx2_get_te_nvfp4_context" in ltx2
        and "from transformer_engine.common.recipe import NVFP4BlockScaling" in ltx2
        and "import transformer_engine.pytorch as te" in ltx2
        and "_LTX2_TE_NVFP4_LINEAR_CLS = te.Linear" in ltx2
        and "fp8_autocast" in ltx2,
        "ltx2_te_adapter_is_fused_gelu_bias_gate_specific": "ltx2_te_nvfp4_fused_proj_in_gelu" in ltx2
        and "general_gemm" in ltx2
        and "gelu=True" in ltx2
        and "ltx2_bias_residual_gate" in ltx2,
        "cosmos3_ffn_is_bias_free_swiglu": "class Cosmos3GatedMLP" in cosmos3
        and "self.gate_up_proj = MergedColumnParallelLinear" in cosmos3
        and "self.down_proj = RowParallelLinear" in cosmos3
        and cosmos3.count("bias=False") >= 2
        and "self.act_fn = SiluAndMul()" in cosmos3,
        "cosmos3_has_no_te_fused_epilogue_consumer": "transformer_engine" not in cosmos3
        and "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU" not in cosmos3
        and "ltx2_bias_residual_gate" not in cosmos3,
        "launcher_keeps_te_recipe_variant_guarded": "\"te_recipe_variant\"" in launch
        and "online ModelOpt FP4 linear consumer" in launch
        and "Cosmos3-equivalent bias-free SwiGLU" in launch
        and "fused flags are disabled in " in launch
        and "the manifest; only run" in launch,
    }


def transform_env_probe() -> dict[str, Any]:
    from efficiency import Capability, ModelSpec, compose
    from efficiency.transforms.nvfp4_ffn import NVFP4FFN

    spec = ModelSpec(
        name="ManifestNVFP4Probe",
        capabilities=frozenset({Capability.SUPPORTS_NVFP4_LINEAR}),
        seq_dim=1,
    )
    generic_env: dict[str, str] = {}
    compose(
        [
            NVFP4FFN(
                disable_rht=False,
                disable_stochastic_rounding=False,
                disable_2d_quantization=False,
                row_scaled_activation=True,
                fused_proj_in_gelu=True,
                fused_proj_out_bias_gate=True,
                pad_m_to=32,
            )
        ],
        spec,
    ).apply_transforms(None, stage="stage2", env=generic_env)

    ltx2_env: dict[str, str] = {}
    compose(
        [
            NVFP4FFN(
                fused_proj_in_gelu=True,
                fused_proj_out_bias_gate=True,
                pad_m_to=32,
                te_adapter="ltx2",
            )
        ],
        spec,
    ).apply_transforms(None, stage="stage2", env=ltx2_env)

    profile_env: dict[str, str] = {}
    compose(
        [
            NVFP4FFN(
                module_scope="profiled_hot_ffn",
                profile_layer_scores="0-1:0.05,2-29:1.0,30-31:0.05",
                profile_keep_ratio=0.875,
                fallback_policy="bf16_on_profile_miss",
            )
        ],
        spec,
    ).apply_transforms(None, stage="stage2", env=profile_env)

    ltx2_keys = sorted(key for key in generic_env if key.startswith("SGLANG_LTX2_TE_NVFP4_"))
    explicit_ltx2_keys = sorted(key for key in ltx2_env if key.startswith("SGLANG_LTX2_TE_NVFP4_"))
    return {
        "generic_env_has_hq_recipe_flags": all(
            key in generic_env
            for key in (
                "SGLANG_HQ_NVFP4_DISABLE_RHT",
                "SGLANG_HQ_NVFP4_DISABLE_STOCHASTIC_ROUNDING",
                "SGLANG_HQ_NVFP4_DISABLE_2D_QUANTIZATION",
                "SGLANG_HQ_NVFP4_ROW_SCALED_ACTIVATION",
                "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU",
                "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE",
                "SGLANG_HQ_NVFP4_PAD_M_TO",
            )
        ),
        "generic_env_has_no_ltx2_adapter_keys": not ltx2_keys,
        "generic_env_ltx2_keys": ltx2_keys,
        "explicit_ltx2_adapter_has_ltx2_keys": all(
            key in ltx2_env
            for key in (
                "SGLANG_LTX2_TE_NVFP4_VIDEO_FFN",
                "SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU",
                "SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE",
                "SGLANG_LTX2_TE_NVFP4_PAD_M_TO",
            )
        ),
        "explicit_ltx2_adapter_keys": explicit_ltx2_keys,
        "profile_env_has_selector_output": profile_env.get(
            "SGLANG_HQ_NVFP4_PROFILED_LAYERS"
        )
        == "2-29",
        "profile_env_has_derived_dense_guards": profile_env.get(
            "SGLANG_HQ_NVFP4_DENSE_LAYERS"
        )
        == "0-1,30-31",
        "profile_env_source_is_model_agnostic": profile_env.get(
            "SGLANG_HQ_NVFP4_PROFILE_SOURCE"
        )
        == "manifest_layer_scores",
    }


def candidate_alignment(
    runtime_dependency: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    te_data = load_toml(TE_MANIFEST)
    te_params = te_data.get("efficiency", {}).get("params", {})
    profiled_data = load_toml(PROFILED_MANIFEST)
    profiled_params = profiled_data.get("efficiency", {}).get("params", {})
    checks = source_checks()
    env_probe = transform_env_probe()
    non_te_rows: dict[str, dict[str, Any]] = {}
    for cid, manifest in NON_TE_NVFP4_CANDIDATES.items():
        non_te_rows[cid] = {
            "manifest": str(manifest),
            "algorithm_boundary": "ModelOpt/CUTLASS NVFP4 linear path",
            "online_modelopt_consumer_wired": checks["modelopt_online_fp4_consumer_wired"],
            "uses_te_fused_recipe_flags": False,
            "can_claim_cosmos3_runtime_consumer": checks[
                "modelopt_online_fp4_consumer_wired"
            ],
        }
        if cid == "profiled_hot_linear_nvfp4":
            non_te_rows[cid].update(
                {
                    "profile_selector_declared": bool(
                        profiled_params.get("profile_layer_scores")
                    )
                    and bool(profiled_params.get("profile_keep_ratio")),
                    "profile_selector_env_emitted": env_probe[
                        "profile_env_has_selector_output"
                    ]
                    and env_probe["profile_env_has_derived_dense_guards"],
                    "runtime_consumes_profiled_layers": checks[
                        "modelopt_profiled_layer_consumer_wired"
                    ],
                    "uses_static_dense_layer_manifest_param": "dense_layers"
                    in profiled_params,
                    "derived_profiled_layers": "2-29",
                    "derived_dense_guards": "0-1,30-31",
                }
            )

    cosmos3_fused_match = not checks["cosmos3_ffn_is_bias_free_swiglu"]
    te_row = {
        "manifest": str(TE_MANIFEST),
        "algorithm_boundary": (
            "TransformerEngine NVFP4 recipe family with only generic recipe "
            "axes enabled in the manifest; model-specific fused epilogues must "
            "prove their FFN semantics before becoming a candidate claim."
        ),
        "row_scaled_recipe_flag_declared": bool(te_params.get("row_scaled_activation")),
        "fused_proj_in_gelu_flag_declared": bool(te_params.get("fused_proj_in_gelu")),
        "fused_proj_out_bias_gate_flag_declared": bool(
            te_params.get("fused_proj_out_bias_gate")
        ),
        "te_adapter_declared": str(te_params.get("te_adapter", "")),
        "fused_manifest_flags_are_ltx2_shape": bool(
            te_params.get("fused_proj_in_gelu")
            and te_params.get("fused_proj_out_bias_gate")
        ),
        "generic_recipe_axes": [
            name
            for name in ("row_scaled_activation",)
            if bool(te_params.get(name))
        ],
        "ltx2_shaped_fused_epilogue_manifest_flags": [
            name
            for name in ("fused_proj_in_gelu", "fused_proj_out_bias_gate")
            if bool(te_params.get(name))
        ],
        "te_fused_manifest_flag_status": (
            "disabled_until_cosmos3_swiglu_adapter_exists"
            if not te_params.get("fused_proj_in_gelu")
            and not te_params.get("fused_proj_out_bias_gate")
            else "ltx2_shaped_reconciliation_debt_not_generic_recipe"
        ),
        "unblock_requires_manifest_flag_reconciliation": bool(
            te_params.get("fused_proj_in_gelu")
            or te_params.get("fused_proj_out_bias_gate")
        ),
        "row_scaled_activation_status": (
            "generic_recipe_axis_not_te_fused_epilogue_consumer_evidence"
        ),
        "recipe_env_emitted": checks["generic_transform_emits_te_recipe_flags"]
        and checks["generic_transform_emits_generic_recipe_flags"]
        and checks["runtime_transform_mirrors_te_recipe_flags"]
        and checks["runtime_transform_mirrors_generic_recipe_flags"],
        "generic_recipe_env_is_model_agnostic": env_probe[
            "generic_env_has_no_ltx2_adapter_keys"
        ],
        "ltx2_adapter_env_requires_explicit_request": env_probe[
            "explicit_ltx2_adapter_has_ltx2_keys"
        ],
        "ltx2_has_model_specific_te_adapter": checks["ltx2_has_te_nvfp4_adapter"],
        "cosmos3_has_model_specific_te_adapter": not checks[
            "cosmos3_has_no_te_fused_epilogue_consumer"
        ],
        "cosmos3_fused_epilogue_semantics_match_ltx2": cosmos3_fused_match,
        "can_claim_te_public_recipe_consumer_on_cosmos3": (
            checks["ltx2_has_te_nvfp4_adapter"]
            and not checks["cosmos3_has_no_te_fused_epilogue_consumer"]
            and cosmos3_fused_match
        ),
        "transformerengine_runtime_available": bool(
            (runtime_dependency or {}).get("available")
        ),
        "transformerengine_runtime_error": (
            (runtime_dependency or {}).get("stderr_last_line")
            or (runtime_dependency or {}).get("stderr_first_line")
            or ""
        ),
        "model_specific_adapter_status": (
            "not_claimed_on_cosmos3; LTX2 GELU/bias/residual-gate epilogues do "
            "not match Cosmos3 bias-free SwiGLU FFN"
        ),
        "known_difference": (
            "The pure NVFP4/FP4 linear path is consumed through ModelOpt on "
            "Cosmos3. The generic TE recipe flags are model-agnostic; LTX2 "
            "compatibility env is emitted only through an explicit adapter. "
            "The manifest now preserves row-scaled activation as the only "
            "generic TE recipe axis and disables the LTX2-shaped fused "
            "GELU/bias-gate flags. A Cosmos3 TE fused epilogue still cannot be "
            "claimed until its bias-free SwiGLU gate_up_proj -> SiluAndMul -> "
            "down_proj semantics are implemented and validated."
        ),
    }
    return {**non_te_rows, "te_recipe_variant": te_row}


def probe() -> dict[str, Any]:
    checks = source_checks()
    runtime_dependency = transformerengine_import_check()
    env_probe = transform_env_probe()
    critical = [
        "te_manifest_cites_transformerengine",
        "te_manifest_preserves_only_generic_recipe_axis",
        "te_manifest_explicitly_disables_ltx2_adapter",
        "generic_transform_emits_te_recipe_flags",
        "generic_transform_emits_generic_recipe_flags",
        "generic_transform_emits_profile_selector_env",
        "generic_transform_scopes_ltx2_adapter_env",
        "runtime_transform_mirrors_te_recipe_flags",
        "runtime_transform_mirrors_generic_recipe_flags",
        "runtime_transform_mirrors_profile_selector_env",
        "runtime_transform_scopes_ltx2_adapter_env",
        "modelopt_online_fp4_consumer_wired",
        "modelopt_profiled_layer_consumer_wired",
        "modelopt_dense_step_guard_wired",
        "profiled_manifest_uses_profile_scores_not_static_dense_layers",
        "loader_selects_modelopt_fp4_online_consumer",
        "ltx2_has_te_nvfp4_adapter",
        "ltx2_te_adapter_is_fused_gelu_bias_gate_specific",
        "cosmos3_ffn_is_bias_free_swiglu",
        "cosmos3_has_no_te_fused_epilogue_consumer",
        "launcher_keeps_te_recipe_variant_guarded",
    ]
    status = (
        "pass"
        if all(checks[name] for name in critical)
        and env_probe["generic_env_has_hq_recipe_flags"]
        and env_probe["generic_env_has_no_ltx2_adapter_keys"]
        and env_probe["explicit_ltx2_adapter_has_ltx2_keys"]
        and env_probe["profile_env_has_selector_output"]
        and env_probe["profile_env_has_derived_dense_guards"]
        else "fail"
    )
    return {
        "status": status,
        "public_reference_role": {
            "non_te_nvfp4_candidates": (
                "TransformerEngine/ModelOpt/CUTLASS motivate the generic "
                "NVFP4 linear algorithm boundary; Cosmos3 consumes the local "
                "online ModelOpt FP4 linear path."
            ),
            "te_recipe_variant": (
                "TransformerEngine recipe flags are public provenance, but "
                "LTX2 fused epilogues are model-specific adapter code. The "
                "manifest keeps only generic recipe axes active for Cosmos3."
            ),
        },
        "checks": checks,
        "transform_env_probe": env_probe,
        "runtime_dependency": runtime_dependency,
        "candidate_manifest_alignment": candidate_alignment(runtime_dependency),
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
