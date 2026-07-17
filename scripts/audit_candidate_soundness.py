#!/usr/bin/env python3
"""Strict soundness audit for model-agnostic efficiency candidates.

This is the CPU/static gate before GPU fanout. It proves that each candidate has
the required provenance chain, composes against the selected ModelSpec, has an
inspectable launch configuration, and does not accidentally collapse the
model-agnostic boundary into the Cosmos3 runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from techniques.candidate_manifest import (  # noqa: E402
    dry_run_manifest,
    load_toml,
    manifest_dimension,
    manifest_family,
    manifest_id,
    resolve_capabilities,
    schema_errors,
)
from techniques.sparse_attention_policies import canonical_route_mode  # noqa: E402


EXPECTED = {
    "step_cache": {
        "scheduled_step_reuse",
        "teacache_signal_reuse",
        "attention_broadcast",
        "block_layer_feature_cache",
        "adaptive_delta_forecast",
    },
    "token_prune": {
        "feature_norm_prune",
        "shape_stable_compute_mask",
        "tome_merge_restore",
        "region_dynamic_density",
        "cluster_representative_update",
    },
    "sparse_attention": {
        "piecewise_pisa_env",
        "spatial_temporal_head_routing",
        "semantic_permutation",
        "online_mask_search_reuse",
        "proxy_mask_prediction",
        "rotating_anchor_windows",
        "qk_coclustering",
        "headwise_adaptive_budgets",
        "dynamic_pattern_probe",
    },
    "nvfp4_ffn": {
        "conservative_ffn_nvfp4",
        "profiled_hot_linear_nvfp4",
        "te_recipe_variant",
        "dense_guard_policy",
        "backend_padding_policy",
    },
    "kwl_fusion": {
        "env_flag_kwl_bundle",
        "gemm_epilogue_fusion",
        "norm_modulation_residual_fusion",
        "compile_graph_capture",
        "layout_copy_elimination",
        "backend_selection_probe",
    },
}

MODEL_BOUNDARY_FORBIDDEN = (
    "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py",
    "runtime/models/dits/cosmos3video.py",
)


def fail(problems: list[str], path: Path, message: str) -> None:
    problems.append(f"{path.relative_to(ROOT)}: {message}")


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def extract_urls(value: str) -> list[str]:
    return [part.strip(".,)") for part in value.split() if is_url(part.strip(".,)"))]


def check_url(url: str, timeout: float) -> str | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "auto-video-audit"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status < 400:
                return None
            return f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 405}:
            return None
        return f"HTTP {exc.code}"
    except Exception as exc:  # pragma: no cover - network environment varies
        return f"{type(exc).__name__}: {exc}"


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def audit_candidate(
    path: Path, *, check_urls: bool, timeout: float
) -> tuple[str, list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    data = load_toml(path)
    cid = manifest_id(data)
    dim = manifest_dimension(data)
    family = manifest_family(data)

    if not cid:
        fail(problems, path, "missing candidate id")
    if path.parent.name != dim:
        fail(problems, path, f"directory dimension {path.parent.name!r} != manifest dimension {dim!r}")
    if not family:
        fail(problems, path, "missing family")

    for error in schema_errors(data, ROOT):
        fail(problems, path, error)

    refs = data.get("references", {})
    external = refs.get("external", {})
    local = refs.get("local", {})
    for section, key in (("external", "paper"), ("external", "code")):
        urls = extract_urls(str(refs.get(section, {}).get(key, "")))
        if not urls:
            fail(problems, path, f"[references.{section}].{key} must include a URL")
        elif check_urls:
            for url in urls:
                issue = check_url(url, timeout)
                if issue:
                    fail(problems, path, f"external URL failed {url}: {issue}")

    for key in ("generic_impl", "model_adapter_example", "runtime_example"):
        ref = ROOT / str(local.get(key, ""))
        if not ref.exists():
            fail(problems, path, f"[references.local].{key} missing on disk: {local.get(key)}")

    generic = ROOT / str(local.get("generic_impl", ""))
    if generic.exists():
        text = read_text(generic)
        for token in MODEL_BOUNDARY_FORBIDDEN:
            if token in text:
                fail(problems, path, f"generic impl hard-codes Cosmos3 runtime path: {token}")

    caps = data.get("requirements", {}).get("capabilities", [])
    try:
        resolved_caps = resolve_capabilities([str(cap) for cap in caps])
    except ValueError as exc:
        fail(problems, path, str(exc))
        resolved_caps = frozenset()
    if not resolved_caps:
        fail(problems, path, "no resolved required capabilities")

    try:
        payload = dry_run_manifest(data, ROOT)
    except Exception as exc:
        fail(problems, path, f"dry-run failed: {exc}")
        payload = None

    if payload:
        env = {**payload.get("env_preview", {}), **data.get("env", {})}
        if env.get("SGLANG_HQ_CANDIDATE_ID") != cid:
            fail(problems, path, "launch env must expose SGLANG_HQ_CANDIDATE_ID matching [id].name")
        if not payload.get("compose", {}).get("plan"):
            fail(problems, path, "dry-run did not record composed plan")
        if data.get("efficiency", {}).get("kind") == "build_transform":
            if not env:
                fail(problems, path, "build transform has no inspectable env/config")
        if data.get("efficiency", {}).get("kind") == "runtime_technique":
            params = payload.get("runtime_config", {}).get("params", {})
            if not params:
                fail(problems, path, "runtime technique has no inspectable params")
        if cid in {"attention_broadcast", "block_layer_feature_cache"}:
            techniques = payload.get("compose", {}).get("techniques", [])
            params = payload.get("runtime_config", {}).get("params", {})
            expected_scope = (
                "attention_broadcast"
                if cid == "attention_broadcast"
                else "block_layer_feature"
            )
            if techniques != ["payload_cache"]:
                fail(
                    problems,
                    path,
                    "payload-cache candidate must compose as payload_cache, not whole-step step_cache",
                )
            if params.get("scope") != expected_scope:
                fail(problems, path, f"payload-cache scope must be {expected_scope!r}")
            if env.get("SGLANG_HQ_CACHE_SCOPE") != expected_scope:
                fail(problems, path, f"launch env must set SGLANG_HQ_CACHE_SCOPE={expected_scope}")
            payload_mode = str(params.get("mode", env.get("SGLANG_HQ_PAYLOAD_CACHE_MODE", "scheduled")))
            if payload_mode == "pab":
                if env.get("SGLANG_HQ_PAYLOAD_CACHE_MODE") != "pab":
                    fail(problems, path, "PAB payload-cache candidates must export SGLANG_HQ_PAYLOAD_CACHE_MODE=pab")
                if cid == "attention_broadcast":
                    if params.get("attention_kind") != "cross":
                        fail(problems, path, "attention_broadcast must use the public PAB cross-attention controller")
                    if params.get("cross_broadcast") is not True or not params.get("cross_threshold") or not params.get("cross_range"):
                        fail(problems, path, "attention_broadcast must declare public PAB cross threshold/range knobs")
                if cid == "block_layer_feature_cache":
                    if params.get("mlp_broadcast") is not True:
                        fail(problems, path, "block_layer_feature_cache must declare public PAB MLP broadcast")
                    if not params.get("mlp_spatial_broadcast_config"):
                        fail(problems, path, "block_layer_feature_cache must declare public PAB MLP block/skip config")
            elif not env.get("SGLANG_HQ_PAYLOAD_CACHE_SKIP"):
                fail(problems, path, "scheduled payload-cache candidates must expose SGLANG_HQ_PAYLOAD_CACHE_SKIP")
            cosmos_stage = (
                ROOT
                / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py"
            )
            stage_text = read_text(cosmos_stage)
            if (
                '"payload_cache"' not in stage_text
                or "payload_skip" not in stage_text
                or "payload_pab_active" not in stage_text
            ):
                fail(
                    problems,
                    path,
                    "Cosmos3 builder must record scheduled/PAB payload-cache candidates in the efficiency Plan",
                )
            if payload.get("model_spec") == "Cosmos3":
                warnings.append(
                    f"{path.relative_to(ROOT)}: payload_cache has fresh "
                    "public-controller Cosmos3 GPU diagnostics, but quality "
                    "and/or speed failed; do not make a usefulness claim "
                    "without retuning and rerunning"
                )
        if dim == "token_prune":
            if not env.get("SGLANG_HQ_TOKEN_PRUNE_STEPS"):
                fail(
                    problems,
                    path,
                    "token_prune candidate will not install in Cosmos3 without SGLANG_HQ_TOKEN_PRUNE_STEPS",
                )
            method = (
                data.get("efficiency", {})
                .get("params", {})
                .get("method", env.get("SGLANG_HQ_TOKEN_PRUNE_METHOD", ""))
            )
            runtime_token_prune = (
                ROOT
                / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/efficiency/techniques/token_prune.py"
            )
            if method and runtime_token_prune.exists():
                if str(method) not in read_text(runtime_token_prune):
                    fail(
                        problems,
                        path,
                        f"runtime token_prune implementation does not recognize method {method!r}",
                    )
            if cid == "tome_merge_restore":
                tome_probe = ROOT / "scripts/probe_public_tome_alignment.py"
                probe_text = read_text(tome_probe) if tome_probe.exists() else ""
                runtime_text = read_text(runtime_token_prune) if runtime_token_prune.exists() else ""
                for needle in (
                    "bipartite_soft_matching",
                    "scatter_reduce",
                    "matches_public_tome_merge",
                    "tome_bipartite_soft_matching",
                    "merged_values_match",
                    "restored_values_match",
                ):
                    if needle not in probe_text:
                        fail(
                            problems,
                            path,
                            "tome_merge_restore public checker is missing "
                            f"ToMe boundary evidence {needle!r}",
                        )
                for needle in (
                    "tome_bipartite_soft_matching",
                    "plan.merge",
                    "plan.unmerge",
                    "merged_token_indices",
                ):
                    if needle not in runtime_text:
                        fail(
                            problems,
                            path,
                            "runtime token_prune implementation is missing "
                            f"public ToMe merge/unmerge consumer {needle!r}",
                        )
                cosmos_model = (
                    ROOT
                    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py"
                )
                cosmos_text = read_text(cosmos_model) if cosmos_model.exists() else ""
                for needle in ("merged_token_indices", "cos_gen.gather", "sin_gen.gather"):
                    if needle not in cosmos_text:
                        fail(
                            problems,
                            path,
                            "Cosmos3 token-prune adapter is missing ToMe RoPE "
                            f"position bridge {needle!r}",
                        )
            if cid in {
                "feature_norm_prune",
                "shape_stable_compute_mask",
                "region_dynamic_density",
                "cluster_representative_update",
            }:
                token_probe = ROOT / "scripts/probe_public_token_prune_alignment.py"
                probe_text = read_text(token_probe) if token_probe.exists() else ""
                for needle in (
                    "cat_uses_noise_delta",
                    "cat_uses_kmeans_clusters",
                    "cat_uses_staleness_counts",
                    "matches_cat_core_boundary",
                ):
                    if needle not in probe_text:
                        fail(
                            problems,
                            path,
                            "token_prune public checker is missing "
                            f"CAT/ToMeSD boundary evidence {needle!r}",
                        )
            if cid == "cluster_representative_update":
                runtime_text = read_text(runtime_token_prune) if runtime_token_prune.exists() else ""
                for needle in (
                    "CatPruneState",
                    "cat_convergence_stale_indices",
                    "cached_noise_prev",
                    "cat_convergence_stale_cpp",
                ):
                    if needle not in runtime_text:
                        fail(
                            problems,
                            path,
                            "runtime token_prune implementation is missing "
                            f"CAT selector evidence {needle!r}",
                        )
        if dim == "sparse_attention":
            run_script = ROOT / "Sol-LTX-Infer/scripts/run_cosmos3_sglang.sh"
            script_text = read_text(run_script)
            piecewise_backend = (
                ROOT
                / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py"
            )
            piecewise_text = read_text(piecewise_backend)
            for needle in (
                "--component-attention-backends",
                "--attention-backend-config",
            ):
                if needle not in script_text:
                    fail(problems, path, f"Cosmos3 run script does not bridge {needle}")
            route_mode = str(
                data.get("efficiency", {}).get("params", {}).get("route_mode", "score")
            )
            expected_policy = canonical_route_mode(route_mode)
            if env.get("SGLANG_HQ_SPARSE_ROUTE_POLICY") != expected_policy:
                fail(
                    problems,
                    path,
                    "sparse_attention dry-run must expose a pure route policy "
                    f"{expected_policy!r}",
                )
            if not env.get("SGLANG_HQ_SPARSE_ROUTE_FAMILY"):
                fail(
                    problems,
                    path,
                    "sparse_attention dry-run must expose SGLANG_HQ_SPARSE_ROUTE_FAMILY",
                )
            policy_consumed_modes = {
                "spatial_temporal_head_routing",
                "online_mask_search_reuse",
                "proxy_mask_prediction",
                "rotating_anchor_windows",
                "qk_coclustering",
                "headwise_adaptive_budgets",
                "dynamic_pattern_probe",
            }
            if expected_policy in policy_consumed_modes:
                for needle in (
                    "_piecewise_policy_block_indices",
                    "build_sparse_route_mask",
                    expected_policy,
                ):
                    if needle not in piecewise_text:
                        fail(
                            problems,
                            path,
                            "sparse policy is not consumed by piecewise_attn "
                            f"runtime helper; missing {needle!r}",
                        )
                sparse_policy_probe = ROOT / "scripts/probe_public_sparse_policy_alignment.py"
                probe_text = read_text(sparse_policy_probe) if sparse_policy_probe.exists() else ""
                for needle in (
                    "sparge_has_mean_similarity_block_map",
                    "svg_has_spatial_temporal_head_selection",
                    "matches_public_original",
                    "short_gpu_diagnostic_complete",
                    "official_quality_evidence_complete",
                ):
                    if needle not in probe_text:
                        fail(
                            problems,
                            path,
                            "sparse policy public checker is missing public-boundary/"
                            f"diagnostic-vs-quality evidence {needle!r}",
                        )
            if cid == "piecewise_pisa_env":
                pisa_probe = ROOT / "scripts/probe_public_pisa_alignment.py"
                probe_text = read_text(pisa_probe) if pisa_probe.exists() else ""
                params = data.get("efficiency", {}).get("params", {})
                attention_config = env.get("SGLANG_HQ_ATTENTION_BACKEND_CONFIG", "")
                if not isinstance(params, dict) or params.get("route_bias") is not False:
                    fail(
                        problems,
                        path,
                        "piecewise_pisa_env must set route_bias=false for public PISA default route",
                    )
                for key in ("allow_qk_mismatch", "allow_gqa"):
                    if not isinstance(params, dict) or params.get(key) is not True:
                        fail(
                            problems,
                            path,
                            f"piecewise_pisa_env must set {key}=true for Cosmos3 adapter diagnostics",
                        )
                for needle in (
                    "piecewise_route_bias",
                    "piecewise_allow_qk_mismatch",
                    "piecewise_allow_gqa",
                    "SGLANG_PIECEWISE_ATTN_ROUTE_BIAS",
                    "SGLANG_PIECEWISE_ATTN_ALLOW_QK_MISMATCH",
                    "SGLANG_PIECEWISE_ATTN_ALLOW_GQA",
                    "use_bias=route_bias",
                ):
                    if needle not in piecewise_text:
                        fail(
                            problems,
                            path,
                            "piecewise_attn runtime is missing public PISA route-bias guard "
                            f"{needle!r}",
                        )
                if "piecewise_route_bias=false" not in attention_config:
                    fail(
                        problems,
                        path,
                        "piecewise_pisa_env dry-run must export piecewise_route_bias=false",
                    )
                for needle in (
                    "piecewise_allow_qk_mismatch=true",
                    "piecewise_allow_gqa=true",
                ):
                    if needle not in attention_config:
                        fail(
                            problems,
                            path,
                            f"piecewise_pisa_env dry-run must export {needle}",
                        )
                for needle in (
                    "matches_public_default_route",
                    "matches_public_optional_bias_route",
                    "configured_route_bias",
                    "score + torch.log(bias",
                ):
                    if needle not in probe_text:
                        fail(
                            problems,
                            path,
                            "piecewise_pisa_env public checker is missing "
                            f"PISA behavior-boundary evidence {needle!r}",
                        )
        if cid == "teacache_signal_reuse":
            cosmos_stage = (
                ROOT
                / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py"
            )
            stage_text = read_text(cosmos_stage)
            if data.get("efficiency", {}).get("name") != "teacache_residual":
                fail(
                    problems,
                    path,
                    "TeaCache candidate must use teacache_residual rather than whole-step teacache",
                )
            if data.get("env", {}).get("SGLANG_HQ_TEACACHE_REPLAY") != "block_residual":
                fail(
                    problems,
                    path,
                    "TeaCache candidate must export SGLANG_HQ_TEACACHE_REPLAY=block_residual",
                )
            if "SGLANG_HQ_TEACACHE_PERIODIC_RECOMPUTE" not in stage_text:
                fail(
                    problems,
                    path,
                    "Cosmos3 builder ignores SGLANG_HQ_TEACACHE_PERIODIC_RECOMPUTE",
                )
            if "SGLANG_HQ_TEACACHE_COEFFICIENTS" not in stage_text:
                fail(
                    problems,
                    path,
                    "Cosmos3 builder ignores SGLANG_HQ_TEACACHE_COEFFICIENTS",
                )
            if "teacache_residual" not in stage_text:
                fail(
                    problems,
                    path,
                    "Cosmos3 builder cannot select teacache_residual",
                )
            if "max_continuous_hits <= 0" not in read_text(
                ROOT / "efficiency/techniques/teacache.py"
            ):
                fail(
                    problems,
                    path,
                    "generic TeaCache does not treat max_continuous_hits<=0 as no cap",
                )
        if cid in {
            "scheduled_step_reuse",
            "adaptive_delta_forecast",
            "attention_broadcast",
            "block_layer_feature_cache",
        }:
            pab_probe = ROOT / "scripts/probe_public_pab_alignment.py"
            probe_text = read_text(pab_probe) if pab_probe.exists() else ""
            for needle in (
                "uses_timestep_thresholds",
                "uses_count_mod_range",
                "matches_public_pab",
            ):
                if needle not in probe_text:
                    fail(
                        problems,
                        path,
                        "step_cache PAB public checker is missing "
                        f"behavior-boundary evidence {needle!r}",
                    )
        if cid == "semantic_permutation":
            svg_probe = ROOT / "scripts/probe_public_svg_alignment.py"
            probe_text = read_text(svg_probe) if svg_probe.exists() else ""
            for needle in (
                "matches_public_cosmos_sap_hyperparams",
                "matches_public_runtime_assumptions",
                "local_has_cosmos3_text_kv_prefix_adapter",
            ):
                if needle not in probe_text:
                    fail(
                        problems,
                        path,
                        "semantic_permutation public checker is missing "
                        f"Sparse-VideoGen boundary evidence {needle!r}",
                    )
        if dim == "kwl_fusion":
            for flag in ("SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU", "SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT"):
                if flag not in env:
                    fail(problems, path, f"kwl_fusion env preview does not expose {flag}")
            if cid in {
                "env_flag_kwl_bundle",
                "gemm_epilogue_fusion",
                "layout_copy_elimination",
                "norm_modulation_residual_fusion",
            }:
                active_replay_flags = sorted(
                    key
                    for key, value in env.items()
                    if key.startswith("SGLANG_HQ_KWL_") and value == "1"
                )
                if active_replay_flags:
                    fail(
                        problems,
                        path,
                        "Cosmos3-baseline/LTX2-only KWL blocker probes must "
                        f"not keep active replay flags: {active_replay_flags}",
                    )
            if cid in {"backend_selection_probe", "compile_graph_capture"}:
                kwl_probe = ROOT / "scripts/probe_public_kwl_alignment.py"
                probe_text = read_text(kwl_probe) if kwl_probe.exists() else ""
                expected = (
                    (
                        "matches_full_public_backend_implementation",
                        "backend_manifest_selects_torch_sdpa",
                    )
                    if cid == "backend_selection_probe"
                    else (
                        "matches_full_public_compile_or_graph_capture_implementation",
                        "cosmos_run_script_consumes_torch_compile_probe",
                    )
                )
                for needle in expected:
                    if needle not in probe_text:
                        fail(
                            problems,
                            path,
                            "KWL public checker is missing backend/compile "
                            f"boundary evidence {needle!r}",
                        )
            if cid == "backend_selection_probe" and (
                env.get("SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS")
                != "transformer=torch_sdpa"
            ):
                fail(
                    problems,
                    path,
                    "backend_selection_probe must select the Cosmos3 transformer torch_sdpa backend policy",
                )
            if payload.get("model_spec") == "Cosmos3":
                if cid == "backend_selection_probe":
                    warnings.append(
                        f"{path.relative_to(ROOT)}: kwl_fusion has a concrete "
                        "Cosmos3 runtime consumer/policy and completed GPU "
                        "smoke evidence, but torch_sdpa slowed down; the KWL "
                        "checker proves this is backend-compatibility evidence, "
                        "not a public FlashAttention/CUTLASS kernel port"
                    )
                elif cid == "compile_graph_capture":
                    warnings.append(
                        f"{path.relative_to(ROOT)}: kwl_fusion has completed "
                        "Cosmos3 GPU smoke evidence, but quality failed; the "
                        "KWL checker proves this is torch.compile probe wiring, "
                        "not a public CUTLASS/TransformerEngine implementation"
                    )
                elif cid == "gemm_epilogue_fusion":
                    warnings.append(
                        f"{path.relative_to(ROOT)}: kwl_fusion composes and dry-runs, "
                        "with the LTX2-shaped bias+GELU/residual-gate replay "
                        "flags disabled; Cosmos3 already has a different "
                        "SwiGLU/no-bias MLP and fused add+RMSNorm residual "
                        "path, so define a new Cosmos3 algorithmic delta "
                        "before treating a GPU job as meaningful"
                    )
                else:
                    warnings.append(
                        f"{path.relative_to(ROOT)}: kwl_fusion composes and dry-runs, "
                        "with generic KWL defaults separated from explicit "
                        "LTX2 full-bundle replay, but this row is already "
                        "covered by Cosmos3 baseline fused pieces or "
                        "LTX2-only replay flags; do not treat a GPU job as "
                        "meaningful without a new Cosmos3 algorithmic delta"
                    )
        if dim == "nvfp4_ffn":
            if env.get("SGLANG_HQ_ENABLE_TE_NVFP4_FFN") != "1":
                fail(problems, path, "nvfp4_ffn env preview does not enable TE NVFP4 FFN")
            if payload.get("model_spec") == "Cosmos3":
                ltx2_keys = sorted(
                    key for key in env if str(key).startswith("SGLANG_LTX2_TE_NVFP4_")
                )
                if ltx2_keys:
                    fail(
                        problems,
                        path,
                        "Cosmos3 nvfp4_ffn dry-run must not emit LTX2 TE "
                        f"adapter env by default: {ltx2_keys}",
                    )
                if cid == "te_recipe_variant":
                    warnings.append(
                        f"{path.relative_to(ROOT)}: nvfp4_ffn composes and dry-runs, "
                        "and the generic ModelOpt FP4 linear consumer is already "
                        "wired. The generic TE recipe flags are now separated "
                        "from explicit LTX2 adapter env, but a "
                        "Cosmos3-equivalent bias-free SwiGLU fused-epilogue "
                        "adapter is still required before treating this row as "
                        "GPU optimization evidence"
                    )
                elif cid == "conservative_ffn_nvfp4":
                    warnings.append(
                        f"{path.relative_to(ROOT)}: nvfp4_ffn has a Cosmos3 online "
                        "ModelOpt FP4 consumer and cutlass Slurm 3454033 completed "
                        "with quality pass, but denoise speed was much slower than "
                        "matched dense Slurm 3443090"
                    )
                elif cid == "backend_padding_policy":
                    warnings.append(
                        f"{path.relative_to(ROOT)}: nvfp4_ffn backend policy has a "
                        "Cosmos3 online ModelOpt FP4 consumer and CUDNN Slurm "
                        "3454197 completed with quality pass, but denoise speed "
                        "was slower than matched dense Slurm 3443090"
                    )
                elif cid == "profiled_hot_linear_nvfp4":
                    warnings.append(
                        f"{path.relative_to(ROOT)}: nvfp4_ffn profile-selector "
                        "policy derives hot layers and dense guards, has a "
                        "Cosmos3 online ModelOpt FP4 consumer, and cutlass "
                        "Slurm 3454199 completed with quality pass, but denoise "
                        "speed was neutral/slightly slower than matched dense "
                        "Slurm 3443090"
                    )
                elif cid == "dense_guard_policy":
                    warnings.append(
                        f"{path.relative_to(ROOT)}: nvfp4_ffn dense guards have a "
                        "Cosmos3 online ModelOpt FP4 consumer; two-step Slurm "
                        "3454198 verified all-warmup dense fallback, and four-step "
                        "Slurm 3454344 completed with quality pass and modest "
                        "speedup versus matched dense Slurm 3454343"
                    )
                else:
                    warnings.append(
                        f"{path.relative_to(ROOT)}: nvfp4_ffn composes and dry-runs, "
                        "and Cosmos3 now has an online ModelOpt FP4 consumer; "
                        "backend compatibility plus matched GPU quality/speed "
                        "evidence is still required"
                    )

    verification = data.get("verification", {})
    if verification.get("mode") != "gpu":
        fail(problems, path, "verification.mode must be gpu for candidate baselines")
    if dim in {"nvfp4_ffn", "kwl_fusion"} and verification.get("allow_non_bit_exact") is not True:
        fail(problems, path, "non-bit-exact dimension must explicitly allow non-bit-exact verification")

    return cid, problems, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-urls", action="store_true", help="also perform network HEAD checks")
    parser.add_argument("--timeout", type=float, default=8.0, help="per-URL timeout for --check-urls")
    parser.add_argument("--json-out", type=Path, help="optional audit report path")
    args = parser.parse_args()

    paths = sorted((ROOT / "candidates").glob("*/*.toml"))
    problems: list[str] = []
    warnings: list[str] = []
    seen: Counter[str] = Counter()
    by_dim: dict[str, set[str]] = {}

    for path in paths:
        cid, candidate_problems, candidate_warnings = audit_candidate(
            path, check_urls=args.check_urls, timeout=args.timeout
        )
        seen[cid] += 1
        by_dim.setdefault(path.parent.name, set()).add(cid)
        problems.extend(candidate_problems)
        warnings.extend(candidate_warnings)
        print(("FAIL" if candidate_problems else "PASS") + f" {path.relative_to(ROOT)}")
        for problem in candidate_problems:
            print(f"  - {problem}")
        for warning in candidate_warnings:
            print(f"  ! {warning}")

    for cid, count in sorted(seen.items()):
        if count > 1:
            problems.append(f"duplicate candidate id {cid!r}: {count} files")

    for dim, expected_ids in EXPECTED.items():
        actual = by_dim.get(dim, set())
        if actual != expected_ids:
            problems.append(
                f"{dim}: expected {sorted(expected_ids)}, got {sorted(actual)}"
            )

    report = {
        "candidate_count": len(paths),
        "counts_by_dimension": {dim: len(ids) for dim, ids in sorted(by_dim.items())},
        "problems": problems,
        "warnings": warnings,
        "status": "pass" if not problems else "fail",
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n=== candidate soundness audit: {report['status']} ({len(paths)} candidates) ===")
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
