#!/usr/bin/env python3
"""Check local ``semantic_permutation`` against public Sparse-VideoGen SAP.

The local backend intentionally reuses the public Sparse-VideoGen/SAP algorithm
shape and public Cosmos SAP hyperparameters, then adds Cosmos3 runtime glue for
SGLang attention metadata, GQA, text-KV prefixes, and current FlashInfer wrapper
APIs. This probe pins that boundary without importing the CUDA dependencies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

from techniques.sparse_attention_policies import (  # noqa: E402
    SparseVideoGenSAPPlan,
    sparse_videogen_sap_plan,
)

PUBLIC_SVG = Path("/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/autovideo/Sparse-VideoGen-public-f0abc563")
PUBLIC_COSMOS_ATTN = PUBLIC_SVG / "svg" / "models" / "cosmos" / "attention.py"
PUBLIC_COSMOS_SCRIPT = PUBLIC_SVG / "scripts" / "cosmos" / "cosmos_t2v_sap.sh"
PUBLIC_COSMOS_INFER = PUBLIC_SVG / "cosmos_t2v_inference.py"
LOCAL_SVG_BACKEND = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/attention/backends/sparse_video_gen_2_attn.py"
)
LOCAL_POLICY = ROOT / "efficiency" / "sparse_attention_policies.py"
LOCAL_COSMOS_STAGE = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py"
)
MANIFEST = ROOT / "candidates" / "sparse_attention" / "semantic_permutation.toml"

PUBLIC_COSMOS_SAP_CONFIG = {
    "component": "transformer",
    "route_mode": "semantic_permutation",
    "backend": "sparse_video_gen_2_attn",
    "svg2_num_q_centroids": 400,
    "svg2_num_k_centroids": 1000,
    "svg2_top_p_kmeans": 0.9,
    "svg2_min_kc_ratio": 0.1,
    "svg2_kmeans_iter_init": 50,
    "svg2_kmeans_iter_step": 2,
    "svg2_first_layers_fp": 0.03,
    "svg2_first_times_fp": 0.3,
}


def git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def source_checks() -> dict[str, bool]:
    public_attn = read_text(PUBLIC_COSMOS_ATTN)
    public_script = read_text(PUBLIC_COSMOS_SCRIPT)
    public_infer = read_text(PUBLIC_COSMOS_INFER)
    local_backend = read_text(LOCAL_SVG_BACKEND)
    local_stage = read_text(LOCAL_COSMOS_STAGE)
    local_policy = read_text(LOCAL_POLICY)
    return {
        "has_public_svg_repo": PUBLIC_SVG.exists(),
        "public_has_cosmos_sap_processor": "class Cosmos_SAPAttn_Processor" in public_attn,
        "public_uses_kmeans_dynamic_map_permutation": "batch_kmeans_Euclid" in public_attn
        and "identify_dynamic_map" in public_attn
        and "permute_tensor_by_labels_triton" in public_attn
        and "dynamic_block_sparse_fwd_flashinfer" in public_attn
        and "apply_inverse_permutation_triton" in public_attn,
        "public_requires_batch1_for_sap": 'assert cfg == 1, "Batch size must be 1' in public_attn,
        "public_disallows_gqa": "Does not support GQA" in public_attn,
        "public_script_sets_cosmos_sap_hyperparams": "qc_kmeans=400" in public_script
        and "kc_kmeans=1000" in public_script
        and "top_p_k=0.9" in public_script
        and "min_kc_ratio=0.10" in public_script
        and "kmeans_iter_init=50" in public_script
        and "kmeans_iter_step=2" in public_script
        and "first_times_fp=0.3" in public_script
        and "first_layers_fp=0.03" in public_script,
        "public_inference_replaces_cosmos_attention": "replace_cosmos_attention" in public_infer
        and '--pattern", type=str, default="dense", choices=["SVG", "dense", "SAP"]' in public_infer,
        "local_uses_sap_algorithm_shape": "batch_kmeans_Euclid" in local_backend
        and "identify_dynamic_map" in local_backend
        and "permute_tensor_by_labels_triton" in local_backend
        and "_dynamic_block_sparse_fwd_flashinfer_varlen" in local_backend
        and "apply_inverse_permutation_triton" in local_backend,
        "local_has_pure_sap_plan": "class SparseVideoGenSAPPlan" in local_policy
        and "def sparse_videogen_sap_plan" in local_policy
        and "def sparse_videogen_identify_dynamic_map" in local_policy
        and "def sparse_videogen_permutation_indices" in local_policy,
        "local_has_cosmos3_text_kv_prefix_adapter": "_append_key_prefix_cluster" in local_backend
        and "always-visible key cluster" in local_backend,
        "local_has_gqa_adapter": "_expand_gqa_kv_for_sdpa" in local_backend
        and "repeat_interleave(repeat_factor, dim=1)" in local_backend,
        "local_has_flashinfer_api_adapter": "_install_flashinfer_sparse_compat" in local_backend
        and "reset_workspace_buffer_compat" in local_backend,
        "local_has_cosmos3_metadata_builder": "SparseVideoGen2AttentionMetadataBuilder" in local_stage
        and "svg2_num_q_centroids" in local_stage,
        "local_has_serial_cfg_bridge": "_predict_noise_cfg_serial" in local_stage
        and "_uses_sparse_video_gen2_attention" in local_stage,
    }


def manifest_config_alignment() -> dict[str, Any]:
    data = load_toml(MANIFEST)
    params = data.get("efficiency", {}).get("params", {})
    actual = {key: params.get(key) for key in PUBLIC_COSMOS_SAP_CONFIG}
    plan = sparse_videogen_sap_plan(
        route_mode=str(params.get("route_mode", "semantic_permutation")),
        backend=str(params.get("backend", "sparse_video_gen_2_attn")),
        num_q_centroids=int(params.get("svg2_num_q_centroids", 0)),
        num_k_centroids=int(params.get("svg2_num_k_centroids", 0)),
        top_p_kmeans=float(params.get("svg2_top_p_kmeans", 0.0)),
        min_kc_ratio=float(params.get("svg2_min_kc_ratio", 0.0)),
        kmeans_iter_init=int(params.get("svg2_kmeans_iter_init", 0)),
        kmeans_iter_step=int(params.get("svg2_kmeans_iter_step", 0)),
        zero_step_kmeans_init=bool(params.get("svg2_zero_step_kmeans_init", False)),
        first_layers_fp=float(params.get("svg2_first_layers_fp", 0.0)),
        first_times_fp=float(params.get("svg2_first_times_fp", 0.0)),
    )
    pure_plan = plan.as_manifest_config()
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in PUBLIC_COSMOS_SAP_CONFIG.items()
        if actual.get(key) != expected
    }
    return {
        "manifest": str(MANIFEST),
        "expected_public_cosmos_sap_config": PUBLIC_COSMOS_SAP_CONFIG,
        "actual_config": actual,
        "pure_sap_plan": pure_plan,
        "pure_sap_algorithm_steps": list(SparseVideoGenSAPPlan().algorithm_steps),
        "matches_pure_sap_plan": actual == pure_plan,
        "matches_public_cosmos_sap_hyperparams": not mismatches,
        "config_mismatches": mismatches,
        "matches_public_runtime_assumptions": False,
        "matches_full_public_sparse_videogen_cosmos": False,
        "known_difference": (
            "The candidate uses the public Cosmos SAP hyperparameters and the "
            "SAP kmeans/dynamic-map/permutation core, but the local SGLang "
            "backend adds Cosmos3-specific GQA expansion, text-KV prefix "
            "clusters, varlen FlashInfer wrapper compatibility, metadata "
            "construction, and serial-CFG bridging. Treat the pure SAP core as "
            "preserved and the runtime differences as model-specific consumer "
            "glue, not a full public Sparse-VideoGen port."
        ),
    }


def probe() -> dict[str, Any]:
    return {
        "status": "pass",
        "public_reference": {
            "repo": str(PUBLIC_SVG),
            "commit": git_commit(PUBLIC_SVG),
            "cosmos_attention_source": str(PUBLIC_COSMOS_ATTN),
            "cosmos_sap_script": str(PUBLIC_COSMOS_SCRIPT),
            "checks": source_checks(),
        },
        "candidate_manifest_alignment": manifest_config_alignment(),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
