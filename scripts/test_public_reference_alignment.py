#!/usr/bin/env python3
"""Unit checks for public-reference alignment audit helpers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "scripts" / "audit_public_reference_alignment.py"
LAUNCH_PATH = ROOT / "scripts" / "launch_transfeat.py"
COSMOS3_PATH = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py"
)
SVG2_BACKEND_PATH = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/attention/backends/sparse_video_gen_2_attn.py"
)
COSMOS3_RUN_SCRIPT_PATH = ROOT / "Sol-LTX-Infer/scripts/run_cosmos3_sglang.sh"
COSMOS3_DENOISE_STAGE_PATH = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py"
)
COSMOS3_MODEL_PATH = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py"
)
BACKEND_SELECTION_MANIFEST_PATH = (
    ROOT / "transfeat/kwl_fusion/backend_selection_probe.toml"
)
MODELOPT_FP4_PATH = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py"
)
TRANSFORMER_LOAD_UTILS_PATH = (
    ROOT
    / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py"
)
TEACACHE_PROBE_PATH = ROOT / "scripts/probe_public_teacache_alignment.py"
TOME_PROBE_PATH = ROOT / "scripts/probe_public_tome_alignment.py"
PAB_PROBE_PATH = ROOT / "scripts/probe_public_pab_alignment.py"
PISA_PROBE_PATH = ROOT / "scripts/probe_public_pisa_alignment.py"
PIECEWISE_PISA_MANIFEST_PATH = (
    ROOT / "transfeat/sparse_attention/piecewise_pisa_env.toml"
)
TOKEN_PRUNE_PROBE_PATH = ROOT / "scripts/probe_public_token_prune_alignment.py"
SVG_PROBE_PATH = ROOT / "scripts/probe_public_svg_alignment.py"
KWL_PROBE_PATH = ROOT / "scripts/probe_public_kwl_alignment.py"
SPARSE_POLICY_PROBE_PATH = ROOT / "scripts/probe_public_sparse_policy_alignment.py"
NVFP4_PROBE_PATH = ROOT / "scripts/probe_public_nvfp4_alignment.py"
RUNTIME_PYTHON = ROOT / "Sol-LTX-Infer/.conda/ltx23/bin/python"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> int:
    audit = load_module(AUDIT_PATH, "audit_public_reference_alignment_unit")
    launch = load_module(LAUNCH_PATH, "launch_transfeat_unit")

    check(
        "immediate negation suppresses overclaim",
        not audit.contains_unnegated_phrase(
            "this is not a full local port", "full local port"
        ),
    )
    check(
        "unrelated negation does not suppress overclaim",
        audit.contains_unnegated_phrase(
            "not relevant; this is a full local port", "full local port"
        ),
    )
    check(
        "full-public scope maps to full equivalence claim",
        audit.public_equivalence_claim(
            audit.Alignment("full_public_port", "", "", "", "")
        )
        == "full_public_original_equivalence",
    )
    audit_rows = {row["transfeat"]: row for row in audit.audit()[0]}
    check(
        "public audit exposes algorithm boundary and true blocker fields",
        all(
            row.get("algorithm_boundary")
            and row.get("true_blocker")
            and row.get("model_specific_glue_policy")
            for row in audit_rows.values()
        )
        and audit_rows["proxy_mask_prediction"]["algorithm_boundary"]
        == "public_core_preserved_consumer_wired"
        and audit_rows["semantic_permutation"]["algorithm_boundary"]
        == "public_core_preserved_consumer_wired"
        and audit_rows["semantic_permutation"]["true_blocker"]
        == "model_specific_runtime_assumption_mismatch"
        and audit_rows["te_recipe_variant"]["true_blocker"]
        == "cosmos3_te_fused_adapter_semantics_and_dependency"
        and audit_rows["te_recipe_variant"]["purpose"] == "blocker_probe"
        and audit_rows["backend_selection_probe"]["true_blocker"]
        == "not_public_kernel_port_or_quality_speed_missing"
        and audit_rows["backend_selection_probe"]["algorithm_boundary"]
        == "generic_kwl_policy_preserved_not_public_kernel_port",
    )
    check(
        "local-baseline rows are evidence, not frontier or delivery",
        audit_rows["scheduled_step_reuse"]["purpose"] == "evidence"
        and audit_rows["adaptive_delta_forecast"]["purpose"] == "evidence"
        and audit_rows["feature_norm_prune"]["purpose"] == "evidence"
        and audit_rows["region_dynamic_density"]["purpose"] == "evidence"
        and all(
            row["purpose"] in {"evidence", "blocker_probe"}
            for row in audit_rows.values()
            if row["true_blocker"] == "local_baseline_not_public_original_algorithm"
        ),
    )
    local_baseline_note_paths = {
        "adaptive_delta_forecast": ROOT / "transfeat/step_cache/adaptive_delta_forecast.toml",
        "scheduled_step_reuse": ROOT / "transfeat/step_cache/scheduled_step_reuse.toml",
        "feature_norm_prune": ROOT / "transfeat/token_prune/feature_norm_prune.toml",
        "region_dynamic_density": ROOT / "transfeat/token_prune/region_dynamic_density.toml",
    }
    check(
        "local-baseline reference notes mark motivation-only boundary",
        all(
            not audit.local_baseline_reference_note_problems(
                path,
                str(audit.load_toml(path)["references"]["external"]["notes"]),
            )
            for path in local_baseline_note_paths.values()
        )
        and audit.local_baseline_reference_note_problems(
            local_baseline_note_paths["scheduled_step_reuse"],
            "Canonical public cache implementation.",
        ),
    )
    adapter_probe_note_paths = {
        "backend_selection_probe": ROOT
        / "transfeat/kwl_fusion/backend_selection_probe.toml",
        "compile_graph_capture": ROOT
        / "transfeat/kwl_fusion/compile_graph_capture.toml",
    }
    check(
        "adapter/probe reference notes mark not-public boundary",
        all(
            not audit.adapter_probe_reference_note_problems(
                path,
                str(audit.load_toml(path)["references"]["external"]["notes"]),
            )
            for path in adapter_probe_note_paths.values()
        )
        and audit.adapter_probe_reference_note_problems(
            adapter_probe_note_paths["backend_selection_probe"],
            "Canonical backend implementation.",
        ),
    )
    check(
        "adapter/probe rows are evidence, not frontier or delivery",
        all(audit_rows[cid]["purpose"] == "evidence" for cid in adapter_probe_note_paths)
        and all(
            row["purpose"] in {"evidence", "blocker_probe"}
            for row in audit_rows.values()
            if row["true_blocker"] == "model_specific_glue_or_probe_not_algorithm"
        ),
    )
    quality_failed_probe_ids = {
        "attention_broadcast",
        "block_layer_feature_cache",
        "dynamic_pattern_probe",
        "headwise_adaptive_budgets",
        "online_mask_search_reuse",
        "piecewise_pisa_env",
        "proxy_mask_prediction",
        "qk_coclustering",
        "rotating_anchor_windows",
        "spatial_temporal_head_routing",
    }
    check(
        "quality-failed GPU rows are blocker probes, not frontier or delivery",
        all(audit_rows[cid]["purpose"] == "blocker_probe" for cid in quality_failed_probe_ids)
        and all(
            row["purpose"] in {"evidence", "blocker_probe"}
            for row in audit_rows.values()
            if row["true_blocker"]
            in {
                "gpu_quality_and_speed_failed_after_consumer_wired",
                "gpu_quality_failed_after_consumer_wired",
            }
        ),
    )
    kwl_no_delta_paths = {
        "env_flag_kwl_bundle": ROOT / "transfeat/kwl_fusion/env_flag_kwl_bundle.toml",
        "gemm_epilogue_fusion": ROOT / "transfeat/kwl_fusion/gemm_epilogue_fusion.toml",
        "layout_copy_elimination": ROOT
        / "transfeat/kwl_fusion/layout_copy_elimination.toml",
        "norm_modulation_residual_fusion": ROOT
        / "transfeat/kwl_fusion/norm_modulation_residual_fusion.toml",
    }
    check(
        "Cosmos3-baseline/LTX2 rows are blocker probes with no-delta notes",
        all(audit_rows[cid]["purpose"] == "blocker_probe" for cid in kwl_no_delta_paths)
        and all(
            not audit.load_toml(path)
            .get("efficiency", {})
            .get("params", {})
            .get("flags", [])
            for path in kwl_no_delta_paths.values()
        )
        and all(
            not audit.cosmos3_baseline_or_ltx2_note_problems(
                path,
                str(audit.load_toml(path)["references"]["external"]["notes"]),
            )
            for path in kwl_no_delta_paths.values()
        )
        and audit.cosmos3_baseline_or_ltx2_note_problems(
            kwl_no_delta_paths["gemm_epilogue_fusion"],
            "Canonical fused epilogue implementation.",
        ),
    )
    check(
        "delivery rows have no current true blocker",
        all(row["true_blocker"] == "none" for row in audit_rows.values() if row["purpose"] == "delivery")
        and audit_rows["piecewise_pisa_env"]["purpose"] == "blocker_probe"
        and audit_rows["teacache_signal_reuse"]["purpose"] == "frontier"
        and audit_rows["shape_stable_compute_mask"]["purpose"] == "frontier",
    )

    blocked = set(launch.cosmos3_blocked_transfeat_ids())
    expected_blocked = {
        row["transfeat"]
        for row in audit_rows.values()
        if row["cosmos3_status"] in audit.COSMOS3_GPU_UNSUPPORTED_STATUSES
    }
    check("launcher blocklist entries are transfeat ids", blocked <= set(audit.ALIGNMENT))
    check("launcher blocklist matches unsupported statuses", blocked == expected_blocked)

    cosmos3_source = COSMOS3_PATH.read_text()
    check(
        "SparseVideoGen2 uses serial CFG instead of batch-2 CFG",
        "self._uses_sparse_video_gen2_attention()" in cosmos3_source
        and "or teacache_residual_active" in cosmos3_source
        and "return self._predict_noise_cfg_serial(" in cosmos3_source,
    )
    check(
        "SparseVideoGen2 metadata follows Cosmos3 patch padding",
        "svg2_raw_latent_shape" in cosmos3_source
        and "_pad_to_patch_size" in cosmos3_source,
    )
    svg2_backend_source = SVG2_BACKEND_PATH.read_text()
    check(
        "SparseVideoGen2 MAGMA preference is runtime-compatible",
        "reset_linalg_backend" in svg2_backend_source
        and "except RuntimeError as exc" in svg2_backend_source
        and 'backend="cusolver"' in svg2_backend_source,
    )
    check(
        "SparseVideoGen2 handles Cosmos3 GQA before dense or sparse kernels",
        "_expand_gqa_kv_for_sdpa" in svg2_backend_source
        and "repeat_interleave(repeat_factor, dim=1)" in svg2_backend_source
        and "key, value = self._expand_gqa_kv_for_sdpa(query, key, value)"
        in svg2_backend_source,
    )
    check(
        "SparseVideoGen2 bridges Sparse-VideoGen FlashInfer wrapper API drift",
        "_install_flashinfer_sparse_compat" in svg2_backend_source
        and "_vector_sparse_indptr_buffer" in svg2_backend_source
        and "vector_sparse_indices_buffer" in svg2_backend_source
        and "reset_workspace_buffer_compat" in svg2_backend_source,
    )
    check(
        "SparseVideoGen2 sparse launcher supports Cosmos3 text KV prefix",
        "_dynamic_block_sparse_fwd_flashinfer_varlen" in svg2_backend_source
        and "q_seq_len" in svg2_backend_source
        and "kv_seq_len" in svg2_backend_source
        and "block_col_sz.sum(dim=2) == kv_seq_len" in svg2_backend_source,
    )
    run_script_source = COSMOS3_RUN_SCRIPT_PATH.read_text()
    check(
        "Cosmos3 compile_graph_capture env enables torch.compile",
        "SGLANG_HQ_CUDA_GRAPH_PROBE" in run_script_source
        and "SGLANG_HQ_ENABLE_TORCH_COMPILE" in run_script_source
        and "--enable-torch-compile" in run_script_source,
    )
    check(
        "Cosmos3 run script supports explicit diagnostic smoke overrides",
        "COSMOS3_HEIGHT=${COSMOS3_HEIGHT:-720}" in run_script_source
        and "COSMOS3_WIDTH=${COSMOS3_WIDTH:-1280}" in run_script_source
        and "COSMOS3_NUM_FRAMES=${COSMOS3_NUM_FRAMES:-189}" in run_script_source
        and "COSMOS3_NUM_INFERENCE_STEPS=${COSMOS3_NUM_INFERENCE_STEPS:-35}"
        in run_script_source
        and '--perf-dump-path "$OUT_DIR/benchmark.json"' in run_script_source,
    )
    check(
        "Cosmos3 sparse smoke writes piecewise attention stats by default",
        "SGLANG_PIECEWISE_ATTN_STATS_PATH" in run_script_source
        and "$OUT_DIR/piecewise_attn_stats.json" in run_script_source
        and "SGLANG_PIECEWISE_ATTN_STATS_FLUSH_EVERY=1" in run_script_source,
    )
    backend_selection_source = BACKEND_SELECTION_MANIFEST_PATH.read_text()
    check(
        "backend_selection_probe selects a concrete Cosmos3 backend policy",
        'attention_backend_component = "transformer"' in backend_selection_source
        and 'attention_backend = "torch_sdpa"' in backend_selection_source,
    )
    denoise_stage_source = COSMOS3_DENOISE_STAGE_PATH.read_text()
    check(
        "Cosmos3 payload cache scopes do not install whole-step StepCache",
        "payload_cache_scope = cache_scope in" in denoise_stage_source
        and '\"attention_broadcast\"' in denoise_stage_source
        and '\"block_layer_feature\"' in denoise_stage_source
        and 'build_technique(' in denoise_stage_source
        and '"payload_cache"' in denoise_stage_source
        and "payload_skip" in denoise_stage_source
        and "payload_pab_active" in denoise_stage_source
        and "if skip and not payload_cache_scope:" in denoise_stage_source,
    )
    cosmos3_model_source = COSMOS3_MODEL_PATH.read_text()
    check(
        "Cosmos3 attention/block payload cache consumer is wired",
        "class _Cosmos3PayloadCache" in cosmos3_model_source
        and "SGLANG_HQ_CACHE_SCOPE" in cosmos3_model_source
        and "forward_attention(" in cosmos3_model_source
        and "forward_block(" in cosmos3_model_source
        and "payload_cache.forward_block" in cosmos3_model_source
        and "payload_cache.forward_attention" in cosmos3_model_source,
    )
    check(
        "Cosmos3 PAB block-layer mode uses MLP payload replay, not whole-block replay",
        'if self.mode == "pab":' in cosmos3_model_source
        and "return run_block()" in cosmos3_model_source
        and "def forward_mlp(" in cosmos3_model_source
        and "payload_cache.forward_mlp(" in cosmos3_model_source
        and "pab_mlp_next" in cosmos3_model_source,
    )
    nvfp4_probe = load_module(NVFP4_PROBE_PATH, "probe_public_nvfp4_alignment_unit")
    nvfp4_result = nvfp4_probe.probe()
    nvfp4_checks = nvfp4_result["checks"]
    nvfp4_rows = nvfp4_result["transfeat_manifest_alignment"]
    check(
        "NVFP4 public checker separates pure FP4 consumer from TE fused adapter",
        nvfp4_result["status"] == "pass"
        and nvfp4_checks["modelopt_online_fp4_consumer_wired"]
        and nvfp4_checks["modelopt_dense_step_guard_wired"]
        and nvfp4_checks["loader_selects_modelopt_fp4_online_consumer"]
        and nvfp4_checks["te_manifest_preserves_only_generic_recipe_axis"]
        and nvfp4_checks["te_manifest_explicitly_disables_ltx2_adapter"]
        and nvfp4_checks["generic_transform_emits_te_recipe_flags"]
        and nvfp4_checks["generic_transform_emits_generic_recipe_flags"]
        and nvfp4_checks["generic_transform_scopes_ltx2_adapter_env"]
        and nvfp4_checks["runtime_transform_mirrors_te_recipe_flags"]
        and nvfp4_checks["runtime_transform_mirrors_generic_recipe_flags"]
        and nvfp4_checks["runtime_transform_scopes_ltx2_adapter_env"]
        and nvfp4_checks["ltx2_has_te_nvfp4_adapter"]
        and nvfp4_checks["ltx2_te_adapter_is_fused_gelu_bias_gate_specific"]
        and nvfp4_checks["cosmos3_ffn_is_bias_free_swiglu"]
        and nvfp4_checks["cosmos3_has_no_te_fused_epilogue_consumer"]
        and nvfp4_result["transform_env_probe"]["generic_env_has_hq_recipe_flags"]
        and nvfp4_result["transform_env_probe"]["generic_env_has_no_ltx2_adapter_keys"]
        and nvfp4_result["transform_env_probe"]["explicit_ltx2_adapter_has_ltx2_keys"]
        and nvfp4_rows["conservative_ffn_nvfp4"][
            "online_modelopt_consumer_wired"
        ]
        and nvfp4_rows["te_recipe_variant"][
            "row_scaled_recipe_flag_declared"
        ]
        and not nvfp4_rows["te_recipe_variant"][
            "fused_proj_in_gelu_flag_declared"
        ]
        and not nvfp4_rows["te_recipe_variant"][
            "fused_proj_out_bias_gate_flag_declared"
        ]
        and nvfp4_rows["te_recipe_variant"]["te_adapter_declared"] == ""
        and not nvfp4_rows["te_recipe_variant"][
            "fused_manifest_flags_are_ltx2_shape"
        ]
        and nvfp4_rows["te_recipe_variant"]["generic_recipe_axes"]
        == ["row_scaled_activation"]
        and nvfp4_rows["te_recipe_variant"][
            "ltx2_shaped_fused_epilogue_manifest_flags"
        ]
        == []
        and nvfp4_rows["te_recipe_variant"]["te_fused_manifest_flag_status"]
        == "disabled_until_cosmos3_swiglu_adapter_exists"
        and not nvfp4_rows["te_recipe_variant"][
            "unblock_requires_manifest_flag_reconciliation"
        ]
        and nvfp4_rows["te_recipe_variant"]["row_scaled_activation_status"]
        == "generic_recipe_axis_not_te_fused_epilogue_consumer_evidence"
        and not nvfp4_rows["te_recipe_variant"][
            "can_claim_te_public_recipe_consumer_on_cosmos3"
        ]
        and nvfp4_rows["te_recipe_variant"][
            "generic_recipe_env_is_model_agnostic"
        ]
        and nvfp4_rows["te_recipe_variant"][
            "ltx2_adapter_env_requires_explicit_request"
        ]
        and "transformerengine_runtime_available" in nvfp4_rows["te_recipe_variant"]
        and "transformerengine_runtime_error" in nvfp4_rows["te_recipe_variant"]
        and "conservative_ffn_nvfp4" not in launch.COSMOS3_UNSUPPORTED_GPU_REASONS
        and "te_recipe_variant" in launch.COSMOS3_UNSUPPORTED_GPU_REASONS
        and audit_rows["te_recipe_variant"]["purpose"] == "blocker_probe",
    )
    teacache_probe = load_module(TEACACHE_PROBE_PATH, "probe_public_teacache_alignment_unit")
    teacache_result = teacache_probe.probe()
    teacache_row = next(
        row for row in audit.audit()[0] if row["transfeat"] == "teacache_signal_reuse"
    )
    check(
        "TeaCache public checker pins TeaCache4Cosmos controller/residual adapter and short speed gap",
        teacache_result["public_reference"]["checks"]["has_cosmos_coefficients"]
        and teacache_result["core_formula_probe"]["intermediate_core_match"]
        and teacache_result["core_formula_probe"]["runtime_core_match"]
        and teacache_result["core_formula_probe"]["runtime_public_boundary_match"]
        and teacache_result["transfeat_manifest_alignment"]["matches_public_cosmos_profile"]
        and all(teacache_result["cosmos3_adapter_alignment"]["checks"].values())
        and teacache_row["public_equivalence_gap"]
        == "public_controller_residual_adapter_short_quality_pass_speedup_missing",
    )
    tome_probe = load_module(TOME_PROBE_PATH, "probe_public_tome_alignment_unit")
    tome_checks = tome_probe.source_checks()
    tome_result = json.loads(
        subprocess.check_output([str(RUNTIME_PYTHON), str(TOME_PROBE_PATH)], text=True)
    )
    tome_row = next(
        row for row in audit.audit()[0] if row["transfeat"] == "tome_merge_restore"
    )
    tome_probe_source = TOME_PROBE_PATH.read_text()
    tome_behavior = tome_result["behavior_probe"]
    check(
        "ToMe public checker pins public merge/unmerge core and speed gap",
        tome_checks["tome_uses_bipartite_soft_matching"]
        and tome_checks["tome_uses_scatter_reduce_merge"]
        and tome_checks["tome_has_unmerge"]
        and tome_checks["tomesd_uses_random2d_matching"]
        and tome_behavior["merged_values_match"]
        and tome_behavior["restored_values_match"]
        and tome_behavior["matches_public_tome_merge"]
        and "tome_bipartite_soft_matching" in tome_probe_source
        and "merged_values_match" in tome_probe_source
        and "restored_values_match" in tome_probe_source
        and tome_row["public_equivalence_gap"]
        == "public_core_match_short_quality_pass_speedup_missing",
    )
    pab_probe = load_module(PAB_PROBE_PATH, "probe_public_pab_alignment_unit")
    pab_result = pab_probe.probe()
    pab_rows = {
        row["transfeat"]: row
        for row in audit.audit()[0]
        if row["transfeat"]
        in {
            "scheduled_step_reuse",
            "adaptive_delta_forecast",
            "attention_broadcast",
            "block_layer_feature_cache",
        }
    }
    check(
        "PAB public checker pins public controller adapters and short quality failure",
        pab_result["public_reference"]["checks"]["uses_count_mod_range"]
        and pab_result["public_reference"]["checks"]["uses_timestep_thresholds"]
        and pab_result["public_reference"]["checks"]["has_mlp_block_skip_cache"]
        and not pab_result["transfeat_manifest_alignment"]["scheduled_step_reuse"][
            "matches_public_pab"
        ]
        and not pab_result["transfeat_manifest_alignment"]["adaptive_delta_forecast"][
            "matches_public_pab"
        ]
        and pab_result["transfeat_manifest_alignment"]["attention_broadcast"][
            "matches_public_pab_controller"
        ]
        and pab_result["transfeat_manifest_alignment"]["block_layer_feature_cache"][
            "matches_public_pab_controller"
        ]
        and pab_rows["scheduled_step_reuse"]["public_equivalence_gap"]
        == "local_pure_baseline_no_public_original_claim"
        and pab_rows["adaptive_delta_forecast"]["public_equivalence_gap"]
        == "local_pure_baseline_no_public_original_claim"
        and pab_rows["attention_broadcast"]["public_equivalence_gap"]
        == "public_controller_short_gpu_quality_failed"
        and pab_rows["block_layer_feature_cache"]["public_equivalence_gap"]
        == "public_controller_short_gpu_quality_failed",
    )
    pisa_probe = load_module(PISA_PROBE_PATH, "probe_public_pisa_alignment_unit")
    pisa_checks = pisa_probe.source_checks()
    pisa_row = next(
        row for row in audit.audit()[0] if row["transfeat"] == "piecewise_pisa_env"
    )
    pisa_probe_source = PISA_PROBE_PATH.read_text()
    pisa_manifest_source = PIECEWISE_PISA_MANIFEST_PATH.read_text()
    check(
        "PISA public checker pins public-default route boundary and short-quality failure",
        pisa_checks["default_route_uses_qc_kc_topk"]
        and pisa_checks["optional_bias_route_exists"]
        and pisa_checks["has_exact_and_approx_phases"]
        and "use_bias=False" in pisa_probe_source
        and "configured_route_bias" in pisa_probe_source
        and "route_bias = false" in pisa_manifest_source
        and "allow_qk_mismatch = true" in pisa_manifest_source
        and "allow_gqa = true" in pisa_manifest_source
        and pisa_row["cosmos3_status"] == audit.PIECEWISE_PISA_SHORT_GPU_STATUS
        and pisa_row["public_equivalence_gap"]
        == "public_route_short_quality_failed_speedup_present",
    )
    token_prune_probe = load_module(
        TOKEN_PRUNE_PROBE_PATH, "probe_public_token_prune_alignment_unit"
    )
    token_prune_checks = token_prune_probe.source_checks()
    token_prune_result = json.loads(
        subprocess.check_output(
            [str(RUNTIME_PYTHON), str(TOKEN_PRUNE_PROBE_PATH)], text=True
        )
    )
    token_prune_behavior = token_prune_result["behavior_probe"]
    token_prune_alignment = token_prune_behavior["transfeat_manifest_alignment"]
    token_prune_rows = {
        row["transfeat"]: row
        for row in audit.audit()[0]
        if row["transfeat"]
        in {
            "feature_norm_prune",
            "shape_stable_compute_mask",
            "region_dynamic_density",
            "cluster_representative_update",
        }
    }
    check(
        "Token-prune public checker pins CAT selector and remaining mismatches",
        token_prune_checks["cat_uses_noise_delta"]
        and token_prune_checks["cat_uses_kmeans_clusters"]
        and token_prune_checks["cat_uses_staleness_counts"]
        and token_prune_checks["cat_replaces_unselected_from_cache"]
        and token_prune_checks["tomesd_uses_bipartite_matching"]
        and token_prune_checks["tomesd_uses_random2d_matching"]
        and token_prune_checks["local_has_feature_norm"]
        and token_prune_checks["local_has_region_density"]
        and token_prune_checks["local_has_tomesd_random2d_core"]
        and token_prune_checks["local_has_cat_state"]
        and token_prune_checks["local_has_cat_selector"]
        and token_prune_checks["local_has_optional_kmeans_fallback"]
        and token_prune_checks["local_keeps_graph_pooling_out_of_generic_runtime"]
        and token_prune_behavior["fixture"]["torch_kmeans_fallback_valid"]
        and token_prune_alignment["cluster_representative_update"][
            "cat_cache_replacement_matches_boundary"
        ]
        and token_prune_alignment["cluster_representative_update"][
            "cat_cache_selected_tokens_match_current"
        ]
        and token_prune_alignment["cluster_representative_update"][
            "cat_cache_unselected_tokens_match_previous_cache"
        ]
        and token_prune_alignment["shape_stable_compute_mask"][
            "matches_tomesd_random2d_core"
        ]
        and token_prune_rows["cluster_representative_update"]["public_equivalence_gap"]
        == "public_cat_selector_short_quality_pass_speedup_missing"
        and token_prune_rows["shape_stable_compute_mask"]["public_equivalence_gap"]
        == "public_core_match_short_quality_pass_speedup_missing"
        and all(
            token_prune_rows[cid]["public_equivalence_gap"]
            == "local_pure_baseline_no_public_original_claim"
            for cid in (
                "feature_norm_prune",
                "region_dynamic_density",
            )
        ),
    )
    svg_probe = load_module(SVG_PROBE_PATH, "probe_public_svg_alignment_unit")
    svg_result = svg_probe.probe()
    svg_checks = svg_result["public_reference"]["checks"]
    svg_row = next(
        row for row in audit.audit()[0] if row["transfeat"] == "semantic_permutation"
    )
    check(
        "Sparse-VideoGen public checker pins Cosmos SAP core/runtime boundary",
        svg_checks["public_uses_kmeans_dynamic_map_permutation"]
        and svg_checks["public_script_sets_cosmos_sap_hyperparams"]
        and svg_checks["public_requires_batch1_for_sap"]
        and svg_checks["public_disallows_gqa"]
        and svg_checks["local_uses_sap_algorithm_shape"]
        and svg_checks["local_has_pure_sap_plan"]
        and svg_checks["local_has_cosmos3_text_kv_prefix_adapter"]
        and svg_result["transfeat_manifest_alignment"][
            "matches_public_cosmos_sap_hyperparams"
        ]
        and svg_result["transfeat_manifest_alignment"]["matches_pure_sap_plan"]
        and "identify_dynamic_map"
        in svg_result["transfeat_manifest_alignment"]["pure_sap_algorithm_steps"]
        and not svg_result["transfeat_manifest_alignment"][
            "matches_public_runtime_assumptions"
        ]
        and svg_row["public_equivalence_gap"]
        == "public_svg_sap_core_runtime_assumption_mismatch"
        and svg_row["algorithm_boundary"] == "public_core_preserved_consumer_wired"
        and svg_row["purpose"] == "evidence",
    )
    kwl_probe = load_module(KWL_PROBE_PATH, "probe_public_kwl_alignment_unit")
    kwl_result = kwl_probe.probe()
    kwl_checks = kwl_result["checks"]
    kwl_rows = {
        row["transfeat"]: row
        for row in audit.audit()[0]
        if row["transfeat"] in {"backend_selection_probe", "compile_graph_capture"}
    }
    check(
        "KWL public checker pins backend/compile policy boundaries",
        kwl_checks["backend_manifest_selects_torch_sdpa"]
        and kwl_checks["kwl_transform_emits_component_backend_policy"]
        and kwl_checks["kwl_transform_emits_compile_capture_policy"]
        and kwl_checks["cosmos_run_script_consumes_component_backend_policy"]
        and kwl_checks["compile_manifest_selects_compile_flags"]
        and kwl_checks["cosmos_run_script_consumes_torch_compile_probe"]
        and kwl_result["transform_env_probe"]["backend_plan"]["matches_policy_env"]
        and kwl_result["transform_env_probe"]["compile_capture_plan"]["matches_policy_env"]
        and not kwl_result["transfeat_manifest_alignment"][
            "backend_selection_probe"
        ]["matches_full_public_backend_implementation"]
        and not kwl_result["transfeat_manifest_alignment"][
            "compile_graph_capture"
        ]["matches_full_public_compile_or_graph_capture_implementation"]
        and "not a CUTLASS graph-capture"
        in kwl_rows["compile_graph_capture"]["residual_risk"]
        and "generic compile/capture-region policy"
        in kwl_rows["compile_graph_capture"]["residual_risk"]
        and all(
            row["public_equivalence_gap"]
            == "kwl_generic_policy_not_public_kernel_port"
            for row in kwl_rows.values()
        )
        and all(
            row["purpose"] == "evidence"
            for row in kwl_rows.values()
        ),
    )
    sparse_policy_probe = load_module(
        SPARSE_POLICY_PROBE_PATH, "probe_public_sparse_policy_alignment_unit"
    )
    sparse_policy_checks = sparse_policy_probe.source_checks()
    sparse_policy_behavior = json.loads(
        subprocess.check_output(
            [str(RUNTIME_PYTHON), str(SPARSE_POLICY_PROBE_PATH)],
            text=True,
        )
    )["behavior_probe"]
    sparse_policy_rows = {
        row["transfeat"]: row
        for row in audit.audit()[0]
        if row["transfeat"]
        in {
            "spatial_temporal_head_routing",
            "online_mask_search_reuse",
            "proxy_mask_prediction",
            "rotating_anchor_windows",
            "qk_coclustering",
            "headwise_adaptive_budgets",
            "dynamic_pattern_probe",
        }
    }
    check(
        "Sparse-policy public checker pins Sparge/SVG core evidence and local-policy gaps",
        sparse_policy_checks["sparge_has_mean_similarity_block_map"]
        and sparse_policy_checks["sparge_has_fused_quant_mean_similarity_block_map"]
        and sparse_policy_checks["sparge_uses_quantized_qk_cuda_kernel"]
        and sparse_policy_checks["sparge_exposes_block_sparse_mask_api"]
        and sparse_policy_checks["svg_has_spatial_temporal_head_selection"]
        and sparse_policy_checks["svg_has_first_frame_temporal_window"]
        and sparse_policy_checks["has_minference_repo"]
        and sparse_policy_checks["minference_has_dynamic_pattern_bank"]
        and sparse_policy_checks["minference_has_vertical_slash_mask_builder"]
        and sparse_policy_checks["local_has_all_policy_modes"]
        and sparse_policy_checks["local_has_spargeattn_mean_similarity_core"]
        and sparse_policy_checks["local_has_spargeattn_fused_quant_proxy_core"]
        and sparse_policy_checks["local_has_spargeattn_headwise_topk_budget_core"]
        and sparse_policy_checks["local_has_svg_sample_mse_head_selection_core"]
        and sparse_policy_checks["local_has_svg_first_frame_temporal_window_core"]
        and sparse_policy_checks["local_has_svg_cosmos_temporal_permutation_core"]
        and sparse_policy_checks["local_has_minference_dynamic_pattern_bank_core"]
        and sparse_policy_behavior["sparse_videogen_sample_mse_core"][
            "matches_public_core"
        ]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "spatial_temporal_head_routing"
        ]["matches_public_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "spatial_temporal_head_routing"
        ]["matches_svg_sample_mse_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "online_mask_search_reuse"
        ]["refresh_matches_spargeattn_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "online_mask_search_reuse"
        ]["reuse_path_works"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "proxy_mask_prediction"
        ]["matches_public_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "proxy_mask_prediction"
        ]["matches_sparge_fuse_quant_proxy_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "qk_coclustering"
        ]["matches_public_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "headwise_adaptive_budgets"
        ]["matches_sparge_headwise_topk_budget_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "headwise_adaptive_budgets"
        ]["matches_public_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "dynamic_pattern_probe"
        ]["matches_minference_dynamic_pattern_bank_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "dynamic_pattern_probe"
        ]["matches_public_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "rotating_anchor_windows"
        ]["matches_svg_first_frame_temporal_window_core"]
        and sparse_policy_behavior["transfeat_manifest_alignment"][
            "rotating_anchor_windows"
        ]["matches_public_core"]
        and len(sparse_policy_rows) == 7
        and sparse_policy_rows["online_mask_search_reuse"]["public_equivalence_gap"]
        == "public_core_short_quality_failed_speed_negative"
        and sparse_policy_rows["proxy_mask_prediction"]["public_equivalence_gap"]
        == "public_core_short_quality_failed_speed_negative"
        and sparse_policy_rows["qk_coclustering"]["public_equivalence_gap"]
        == "public_core_short_quality_failed_speed_negative"
        and sparse_policy_rows["headwise_adaptive_budgets"]["public_equivalence_gap"]
        == "public_core_short_quality_failed_speed_negative"
        and sparse_policy_rows["spatial_temporal_head_routing"][
            "public_equivalence_gap"
        ]
        == "public_core_short_quality_failed_speed_negative"
        and sparse_policy_rows["dynamic_pattern_probe"]["public_equivalence_gap"]
        == "public_core_short_quality_failed_speed_negative"
        and sparse_policy_rows["rotating_anchor_windows"]["public_equivalence_gap"]
        == "public_core_short_quality_failed_speed_negative"
        and all(
            row["public_equivalence_gap"]
            == "public_checker_mismatch_official_quality_evidence_missing"
            for cid, row in sparse_policy_rows.items()
            if cid
            not in {
                "online_mask_search_reuse",
                "proxy_mask_prediction",
                "qk_coclustering",
                "headwise_adaptive_budgets",
                "spatial_temporal_head_routing",
                "dynamic_pattern_probe",
                "rotating_anchor_windows",
            }
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
