#!/usr/bin/env python3
"""Audit public-reference alignment claims for efficiency candidates.

This is deliberately stricter than a URL liveness check. It records what each
candidate claims relative to its public/canonical references and prevents
launcher-ready probes from being mistaken for full public-reference ports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Alignment:
    scope: str
    public_role: str
    local_claim: str
    cosmos3_status: str
    residual_risk: str


ALIGNMENT: dict[str, Alignment] = {
    "scheduled_step_reuse": Alignment(
        "family_baseline",
        "PAB/VideoSys motivate scheduled reuse; this is a generic whole-step cache baseline.",
        "Implements deterministic skip/reuse through the generic StepCache primitive.",
        "consumer_wired_public_checker_mismatch",
        "Public PAB checker confirms VideoSys broadcasts attention/MLP module outputs using attention type, timestep threshold, and range/count guards; local scheduled_step_reuse skips whole denoiser outputs on an explicit step index set.",
    ),
    "teacache_signal_reuse": Alignment(
        "algorithmic_baseline",
        "TeaCache motivates signal-accumulation reuse from model step signals.",
        "Implements the public Cosmos TeaCache controller coefficients/profile in a generic residual-replay technique consumed by the Cosmos3 block-loop adapter.",
        "consumer_wired_public_controller_residual_adapter_short_gpu_quality_pass_speed_low",
        "Public TeaCache4Cosmos checker confirms the shared rel-L1/poly accumulator core, public Cosmos coefficients/profile, branch-keyed residual replay, and final-step recompute wiring. Short Cosmos3 Slurm 3451216 completed with rank-local residual stats showing reuse=2/8 and forced final compute=1/8 on every rank, quality available, LPIPS mean 0.3483, and Gemini pass, but denoise speedup versus matched dense 8-step Slurm 3451379 is only 1.043x. The exact public AdaLN-modulated signal source and useful speedup are not yet claimed.",
    ),
    "attention_broadcast": Alignment(
        "runtime_adapter_baseline",
        "PAB/VideoSys motivate attention-scope broadcast.",
        "Uses the public PAB cross-attention threshold/range/count controller, then adapts it to Cosmos3 GEN cross-attention payload replay.",
        "consumer_wired_public_controller_short_gpu_quality_failed",
        "Public PAB checker confirms the local cross-attention controller matches the VideoSys PAB decision rule for the probed profile. Short Cosmos3 Slurm 3452366 completed with rank-local payload stats showing hits=192/512 and zero broadcast misses, but quality failed badly (LPIPS mean 0.8010, Gemini fail:high) and denoise speed was slower than the matched dense 8-step baseline (0.953x).",
    ),
    "block_layer_feature_cache": Alignment(
        "runtime_adapter_baseline",
        "PAB/VideoSys motivate block/layer reuse schedules.",
        "Uses the public PAB MLP start-timestep/block/skip-count controller, then adapts it to Cosmos3 GEN MLP-output replay.",
        "consumer_wired_public_controller_short_gpu_quality_failed",
        "Public PAB checker confirms the local MLP controller matches the VideoSys PAB decision rule for the probed profile. Short Cosmos3 Slurm 3452365 completed with rank-local payload stats showing hits=2/512, one saved-next MLP payload, and zero broadcast misses, but quality failed (LPIPS mean 0.0398 with Gemini fail:high) and denoise speed was slower than the matched dense 8-step baseline (0.972x). Because the hit count is very low while Gemini reports a high structural failure, keep this as a quality/consumer-sanity risk until retuned or rerun; do not treat it as public algorithm promotion evidence.",
    ),
    "adaptive_delta_forecast": Alignment(
        "family_baseline",
        "PAB-style reuse motivates forecasted skip steps.",
        "Implements generic whole-step reuse plus a measured output-delta extrapolation.",
        "consumer_wired_public_checker_mismatch",
        "Public PAB checker confirms VideoSys PAB does not define whole-denoiser output-delta forecasting; local adaptive_delta_forecast reached speedup but failed quality.",
    ),
    "feature_norm_prune": Alignment(
        "training_free_baseline",
        "Token-reduction literature motivates feature-magnitude token scoring.",
        "Implements feature-norm scoring, gather/scatter, and previous-hidden compensation.",
        "consumer_wired_public_checker_mismatch",
        "Public CAT/ToMeSD token-prune checker confirms the cited cluster-aware reference uses noise-delta clustering, staleness, and cache replacement, while this local row is plain feature-L2 top-k with previous-hidden compensation; paired quality passed but speedup is below target.",
    ),
    "shape_stable_compute_mask": Alignment(
        "public_algorithm_baseline",
        "ToMeSD motivates random-2D bipartite merge/unmerge with shape restoration.",
        "Implements the public ToMeSD random-2D bipartite merge/unmerge core with deterministic no_rand selection, then restores full sequence shape around the model block loop.",
        "consumer_wired_public_tomesd_random2d_core_short_gpu_quality_pass_speed_negative",
        "Public token-prune checker confirms the local shape_stable_compute_mask path matches the ToMeSD random-2D bipartite merge/unmerge fixture with deterministic no_rand=True selection. The older Slurm 3435277 evidence belongs to the previous deterministic uniform gather/scatter baseline and is historical only. Slurm 3458395 completed the current public-core path with Cosmos3 token_prune installed, quality available, LPIPS mean 0.0310, and Gemini pass, but denoise time 15.3160s was slower than the matched short dense baseline 14.5538s. Full ToMeSD diffusion patching/runtime schedule and useful speedup are not yet claimed.",
    ),
    "tome_merge_restore": Alignment(
        "public_algorithm_baseline",
        "ToMe/ToMeSD motivate merge-and-restore token reduction.",
        "Implements the public ToMe balanced bipartite merge/unmerge core, then restores full sequence shape around the model block loop.",
        "consumer_wired_public_checker_short_gpu_quality_pass_speed_negative",
        "Public ToMe checker now matches the balanced bipartite merge/unmerge fixture, including scatter-reduce merge and unmerge values. Slurm 3445517 diagnosed a Cosmos3 RoPE-position adapter bug for tagged ToMe carries; after the adapter fix, short Cosmos3 Slurm 3446886 completed with quality available, LPIPS mean 0.0199, and Gemini pass, but denoise time 15.4040s was slower than the matched short dense baseline 14.5538s. Full ToMeSD random-2D diffusion integration and useful speedup are not claimed.",
    ),
    "region_dynamic_density": Alignment(
        "family_baseline",
        "Cluster/region-aware pruning motivates density-balanced token scoring.",
        "Normalizes feature-norm scores by coarse local token density.",
        "consumer_wired_public_checker_mismatch",
        "Public CAT checker confirms cluster-aware pruning uses noise-delta cluster selection, staleness counts, and cache replacement; local region_dynamic_density is feature-L2 divided by coarse density. Pairwise Gemini passed, but metrics/speed remain insufficient.",
    ),
    "cluster_representative_update": Alignment(
        "public_algorithm_baseline",
        "Cluster-aware token pruning motivates representative-token updates.",
        "Implements a CAT-style cached-noise delta, cluster-rank, stale-refresh selector in the generic TokenPrune layer, then consumes it through Cosmos3 gather/scatter.",
        "consumer_wired_public_cat_selector_short_gpu_quality_pass_speed_negative",
        "Public CAT checker confirms the local selector matches the cached-delta cluster/staleness fixture, and short Slurm 3453096 confirms the CAT selector method is consumed through Cosmos3 TokenPrune with quality available, LPIPS mean 0.0231, and Gemini pass. Denoise speed is still slower than the matched dense 2-step baseline (0.979x), and the generic runtime intentionally does not claim public SD3 proj_out plus joint-attention KV-cache hooks or torch-geometric graph pooling.",
    ),
    "piecewise_pisa_env": Alignment(
        "runtime_adapter_baseline",
        "PISA provides the canonical piecewise sparse-attention reference family.",
        "Routes the Cosmos3 transformer component to the existing piecewise_attn backend with the public default score top-k selector; Cosmos3 q/k-length and GQA adapter glue is not part of the pure PISA route claim.",
        "consumer_wired_public_route_short_gpu_quality_failed_speed_positive",
        "Public PISA checker confirms piecewise_pisa_env disables the local variance bias and matches the public default qc/kc top-k route boundary. Short Cosmos3 Slurm 3459457 completed after q/k-length and GQA adapter flags with route_mode=score, sparse_calls=128, fallback_calls=0, exact_density=0.5000, and denoise 9.7327s versus dense Slurm 3443090 at 14.5538s (about 1.495x). Matched-short quality is still rejected: LPIPS mean 0.2472, the canonical Gemini collection was inconclusive, and a reduced-frame Gemini retry saved at outputs/nvidia_gemini_retry.json failed high for mosaic/static artifacts; visual spot-check agrees. This is not a full public-original PISA port, and the Cosmos3 adapter glue remains runtime-consumer validation code rather than part of the pure route claim.",
    ),
    "spatial_temporal_head_routing": Alignment(
        "algorithmic_policy_baseline",
        "AdaSpa/Sparse VideoGen motivate spatial/temporal adaptive attention.",
        "Implements a pure Sparse-VideoGen sample-MSE spatial/temporal head-selection core when value centroids are available, with a deterministic block-mask fallback for the current piecewise_attn adapter.",
        "consumer_wired_public_svg_core_short_gpu_quality_failed_speed_negative",
        "Public sparse-policy checker confirms the local pure SVG sample-MSE selector and Cosmos temporal-head permutation boundary. Fresh short Cosmos3 Slurm 3455836 exercised the new selector through piecewise_attn value centroids: selected_route_mode=svg_sample_mse_head_selection, policy_route_calls=128, sparse_calls=128, fallback_calls=0, exact_density=0.2913. Matched-short validation against dense Slurm 3443090 remained negative: denoise 20.3148s versus 14.5538s (0.716x), LPIPS mean 0.2910, and Gemini fail:high. Public FlexAttention/FlashInfer kernels and useful quality/speed are not claimed.",
    ),
    "semantic_permutation": Alignment(
        "algorithmic_policy_baseline",
        "Sparse VideoGen motivates semantic permutation sparse-video attention.",
        "Keeps a pure Sparse-VideoGen SAP plan plus dependency-light dynamic-map/permutation helpers, then consumes them through the Cosmos3 SparseVideoGen2/SAP backend.",
        "consumer_wired_public_svg_sap_core_runtime_mismatch",
        "Public SVG/SAP checker confirms Cosmos SAP hyperparameters, the pure kmeans/dynamic-map/permutation plan, and local dynamic-map/permutation helpers. Full public equivalence is still not claimed because local SGLang adds Cosmos3 GQA, text-KV prefix, varlen FlashInfer, metadata, and CFG glue; GPU quality/performance also failed.",
    ),
    "online_mask_search_reuse": Alignment(
        "algorithmic_policy_baseline",
        "SpargeAttn motivates online mask search and reuse.",
        "Uses the pure SpargeAttn mean-similarity block-map core for refresh, with a drift-gated previous-mask reuse guard; not a full public CUDA-kernel port.",
        "consumer_wired_public_sparge_core_short_gpu_quality_failed_speed_negative",
        "Public sparse-policy checker confirms the refresh path matches SpargeAttn's mean-sim block-selection core and the reuse guard still works. Short Cosmos3 Slurm 3454721 completed with the current core, piecewise stats policy_route_calls=128, sparse_calls=128, fallback_calls=0, but matched-short validation against baseline Slurm 3443090 slowed to 0.532x with LPIPS mean 0.7056 and Gemini fail:high. Full public-original equivalence is still not claimed because SpargeAttn's int8/fp8 CUDA sparse kernels are not ported.",
    ),
    "proxy_mask_prediction": Alignment(
        "algorithmic_policy_baseline",
        "SpargeAttn-style systems motivate proxy-predicted masks.",
        "Uses a pure SpargeAttn fused-quant mean-similarity block-map proxy path without online reuse; not a full public CUDA-kernel port.",
        "consumer_wired_public_sparge_proxy_core_short_gpu_quality_failed_speed_negative",
        "Public sparse-policy checker confirms the current proxy path matches SpargeAttn's mean-sim block-selection core and prepares dependency-light int8/scale proxy artifacts like the public fused-quant path. Runtime adapters still consume only the boolean block mask. Slurm 3457929 was the first GPU attempt for this changed proxy-core path, but it failed before denoising because user-site contamination loaded an incompatible torch package and raised `torchvision::nms` missing. Slurm 3457934 was resubmitted with `PYTHONNOUSERSITE=1` and completed the short Cosmos3 diagnostic with selected_route_mode=spargeattn_quantized_mean_similarity_proxy, policy_route_calls=128, sparse_calls=128, fallback_calls=0, and matched LPIPS mean 0.5896. The canonical Gemini collection was inconclusive, but a reduced-frame retry saved at outputs/nvidia_gemini_retry.json failed high for mosaic/blocking, detail collapse, and static artifacts; denoise was slower than dense Slurm 3443090 (24.9854s vs 14.5538s, about 0.582x). The older Slurm 3442836 evidence belongs to the previous normalized-centroid local proxy and is historical only.",
    ),
    "rotating_anchor_windows": Alignment(
        "algorithmic_policy_baseline",
        "Sparse VideoGen motivates first-frame anchor/sink plus temporal-window sparse video attention.",
        "Uses a dependency-light Sparse-VideoGen first-frame temporal-window mask core consumed by the Cosmos3 boolean block-mask adapter; not a public FlashInfer/FlexAttention kernel port.",
        "consumer_wired_public_svg_temporal_anchor_core_short_gpu_quality_failed_speed_negative",
        "Public sparse-policy checker confirms the local route now matches Sparse-VideoGen's first-frame sink/anchor plus temporal sliding-window mask construction. The old Slurm 3442833 evidence belongs to the previous local rotating-global-anchor policy and is historical only. Fresh Slurm 3459306 completed the changed core with selected_route_mode=svg_first_frame_temporal_window, policy_route_calls=128, sparse_calls=128, fallback_calls=0, exact density 0.3226, and matched LPIPS mean 0.2798; Gemini failed high and denoise was slower than dense Slurm 3443090 (17.7583s vs 14.5538s, about 0.820x).",
    ),
    "qk_coclustering": Alignment(
        "algorithmic_policy_baseline",
        "SpargeAttn motivates Q/K-structure-aware routing.",
        "Uses the pure SpargeAttn Q/K mean-similarity block-map core without online reuse or fused-quant proxy artifacts; not a full public CUDA-kernel port.",
        "consumer_wired_public_sparge_qk_core_short_gpu_quality_failed_speed_negative",
        "Public sparse-policy checker confirms qk_coclustering now matches SpargeAttn's mean-sim block-selection core and selects spargeattn_qk_mean_similarity_block_map in CPU probes. Runtime adapters consume only the boolean block mask and padded block indices; that Cosmos3 glue is validation/runtime-consumer code, not part of the pure algorithm claim. The older Slurm 3442834 evidence belongs to the previous argmax-feature co-cluster local proxy and is historical only. Slurm 3458125 completed the changed public-core path with selected_route_mode=spargeattn_qk_mean_similarity_block_map, policy_route_calls=128, sparse_calls=128, fallback_calls=0, and matched LPIPS mean 0.6841; Gemini failed high and denoise was slower than dense Slurm 3443090 (23.3023s vs 14.5538s, about 0.625x).",
    ),
    "headwise_adaptive_budgets": Alignment(
        "algorithmic_policy_baseline",
        "SpargeAttn motivates per-head sparse hyperparameters and top-k/CDF block-map budgets.",
        "Uses the pure SpargeAttn mean-similarity block-map core with a per-head top-k budget vector; not a full public CUDA-kernel or autotune port.",
        "consumer_wired_public_sparge_headwise_core_short_gpu_quality_failed_speed_negative",
        "Public sparse-policy checker confirms the local route now uses SpargeAttn's per-head top-k surface on the mean-similarity block-map core. The old Slurm 3442835 evidence belongs to the previous head-energy raw-score local policy and is historical only. Fresh Slurm 3458697 completed the current public-core path with selected_route_mode=spargeattn_headwise_topk_budget_block_map, policy_route_calls=128, sparse_calls=128, fallback_calls=0, and matched LPIPS mean 0.6889; Gemini failed high and denoise was slower than dense Slurm 3443090 (24.7521s vs 14.5538s, about 0.588x).",
    ),
    "dynamic_pattern_probe": Alignment(
        "algorithmic_policy_baseline",
        "MInference motivates dynamic sparse pattern banks with A-shape, vertical/slash, and block-sparse routes.",
        "Uses a dependency-light MInference-style pattern bank and per-head dense-error selector; not a public CUDA-kernel or LLM causal-runtime port.",
        "consumer_wired_public_minference_dynamic_core_short_gpu_quality_failed_speed_negative",
        "Public sparse-policy checker confirms the local route now builds A-shape, vertical/slash, and block-sparse masks and selects a per-head pattern through dense-attention reconstruction error. The old Slurm 3442837 evidence belongs to the previous step-cycled score/rotating/local probe and is historical only. Fresh Slurm 3459048 completed the changed MInference-style core with selected_route_mode=minference_dynamic_pattern_bank, policy_route_calls=128, sparse_calls=128, fallback_calls=0, exact density 0.1258, and matched LPIPS mean 0.7938; Gemini failed high and denoise was slower than dense Slurm 3443090 (21.7872s vs 14.5538s, about 0.668x).",
    ),
    "conservative_ffn_nvfp4": Alignment(
        "load_transform_adapter",
        "TransformerEngine/ModelOpt provide the canonical NVIDIA FP4/NVFP4 stack.",
        "Emits load-time NVFP4 FFN scope policy and is consumed by the Cosmos3 online ModelOpt FP4 quantizer.",
        "consumer_wired_online_nvfp4_cutlass_short_gpu_quality_pass_speed_negative",
        "Cosmos3 online NVFP4 reaches real FP4 GEMM. Default Blackwell trtllm hit FlashInfer cubin/header packaging issues, while cutlass Slurm 3454033 completed and quality passed against matched dense Slurm 3443090 (LPIPS mean 0.1218, Gemini pass) but denoise was much slower: 93.5444s versus dense 14.5538s, about 0.156x.",
    ),
    "profiled_hot_linear_nvfp4": Alignment(
        "load_transform_adapter",
        "ModelOpt motivates profiling/PTQ selection of hot linear modules.",
        "Emits profile-derived NVFP4 hot-layer policy and selector-derived dense guards consumed by the Cosmos3 online ModelOpt FP4 quantizer.",
        "consumer_wired_online_nvfp4_profile_selector_short_gpu_quality_pass_speed_neutral",
        "The pure profile selector now derives profiled layers 2-29 and dense guards 0-1,30-31 from manifest layer scores, and the runtime consumes SGLANG_HQ_NVFP4_PROFILED_LAYERS for profiled scopes. This preserves the behavior of cutlass Slurm 3454199, which completed with matched quality pass against dense Slurm 3443090 (LPIPS mean 0.0936, Gemini pass) but denoise speed was neutral/slightly slower: 14.6364s versus dense 14.5538s, about 0.994x.",
    ),
    "te_recipe_variant": Alignment(
        "load_transform_adapter",
        "TransformerEngine exposes FP4 recipe and fused-epilogue options.",
        "Emits generic TE/NVFP4 recipe flags, while the model-agnostic FP4 linear path is consumed through Cosmos3's online ModelOpt FP4 runtime.",
        "cosmos3_te_recipe_fused_adapter_semantics_mismatch",
        "Source access is not the blocker: the pure FP4/NVFP4 linear consumer is wired, the manifest explicitly keeps te_adapter empty, and the generic TE recipe env is separated from LTX2 adapter env. The row-scaled activation flag is the only active generic recipe/scaling axis. The LTX2-shaped fused_proj_in_gelu and fused_proj_out_bias_gate manifest flags are disabled, so any future Cosmos3 TE fused-epilogue claim must first implement a bias-free SwiGLU adapter. The current runtime Python also cannot import TransformerEngine because the CUDNN shared object is missing, so a future Cosmos3 TE adapter would require semantic implementation and dependency repair before GPU validation.",
    ),
    "dense_guard_policy": Alignment(
        "load_transform_adapter",
        "ModelOpt-style dense guards bound FP4 numerical risk.",
        "Emits dense-layer and dense-step guards consumed by the Cosmos3 online ModelOpt FP4 quantizer.",
        "consumer_wired_online_nvfp4_dense_guard_short4_gpu_quality_pass_speed_low",
        "Dense layer guards keep selected layers BF16 at load; dense step guards retain BF16 weights and fall back through forward_context. Two-step Slurm 3454198 verified all-warmup dense fallback, and four-step Slurm 3454344 exercised post-warmup FP4 with matched quality pass against dense Slurm 3454343 (LPIPS mean 0.0414, Gemini pass) and modest denoise speedup: 15.8822s versus dense 17.3309s, about 1.091x.",
    ),
    "backend_padding_policy": Alignment(
        "load_transform_adapter",
        "CUTLASS Blackwell examples motivate backend/padding choices.",
        "Emits backend and M-padding policy consumed by the Cosmos3 online ModelOpt FP4 path.",
        "consumer_wired_online_nvfp4_cudnn_short_gpu_quality_pass_speed_negative",
        "Backend selection now reaches the runtime GEMM selector. CUDNN Slurm 3454197 completed with matched quality pass against dense Slurm 3443090 (LPIPS mean 0.1218, Gemini pass) but denoise was slower: 15.1625s versus dense 14.5538s, about 0.960x.",
    ),
    "env_flag_kwl_bundle": Alignment(
        "build_transform_adapter",
        "FlashAttention/CUTLASS/TE motivate fused-kernel boundaries.",
        "Emits the generic KWL strategy scaffold; LTX2 full-bundle replay now requires an explicit adapter request.",
        "cosmos3_pure_algorithm_subset_already_baseline",
        "Cosmos3 already uses several pure KWL-style kernels unconditionally. The generic KWL transform now defaults all concrete fusion flags to 0 and emits no LTX2 adapter marker; LTX2-only flags such as audio/VAE/guidance sharing still do not map to Cosmos3.",
    ),
    "gemm_epilogue_fusion": Alignment(
        "build_transform_adapter",
        "CUTLASS/TE motivate fused GEMM epilogues.",
        "Keeps the generic KWL scaffold with no active replay flags; the old FFN projection/residual-gate flags are LTX2-shaped and not a Cosmos3 algorithm claim.",
        "cosmos3_pure_algorithm_subset_already_baseline",
        "Cosmos3 already fuses gate/up projection and activation-multiply, while the LTX2 bias+GELU and residual-gate epilogues do not match Cosmos3's SwiGLU/no-bias MLP and fused add+RMSNorm residual threading. The no-delta manifest now keeps those replay flags disabled.",
    ),
    "norm_modulation_residual_fusion": Alignment(
        "build_transform_adapter",
        "TransformerEngine-style fused ops motivate norm/modulation/gate boundaries.",
        "Keeps the generic KWL scaffold with no active replay flags for the LTX2-shaped norm/modulation surfaces.",
        "cosmos3_pure_algorithm_subset_already_baseline",
        "The Cosmos3 GEN path already uses fused add+RMSNorm; AdaLN/modulation flags are LTX2-specific and not a Cosmos3 semantic match. The no-delta manifest now keeps those replay flags disabled.",
    ),
    "compile_graph_capture": Alignment(
        "kwl_generic_policy_baseline",
        "CUDA graph and compiler references motivate capture of stable repeated regions.",
        "Keeps a generic compile/capture-region policy with eager fallback metadata, then consumes it through Cosmos3's torch.compile probe path.",
        "consumer_wired_public_checker_mismatch",
        "Public KWL checker confirms the generic compile/capture-region policy is preserved, but this is not a CUTLASS graph-capture or TransformerEngine fused-kernel implementation. Although the smoke reached 1.1435x, it is non-promotable because quality failed and the local policy is not a public TE/CUTLASS/standalone-capture port.",
    ),
    "layout_copy_elimination": Alignment(
        "build_transform_adapter",
        "Fused attention/GEMM references motivate layout-preserving dataflow.",
        "Keeps the generic KWL scaffold with no active replay flags for the LTX2-shaped sharing surfaces.",
        "cosmos3_pure_algorithm_subset_already_baseline",
        "Cosmos3 already keeps some layout-friendly RoPE/MLP dataflow, while the old LTX2 sharing flags do not identify an extra Cosmos3-only change. The no-delta manifest now keeps those replay flags disabled.",
    ),
    "backend_selection_probe": Alignment(
        "kwl_generic_policy_baseline",
        "FlashAttention/CUTLASS motivate backend choice as a performance lever.",
        "Keeps a generic backend-selection plan with explicit fallback, then consumes it by selecting Cosmos3 transformer=torch_sdpa.",
        "consumer_wired_public_checker_mismatch",
        "Public KWL checker confirms the generic backend-selection policy is preserved and selects Cosmos3 transformer=torch_sdpa, but this is not a public FlashAttention or CUTLASS kernel port; smoke slowed to 0.5942x and is compatibility evidence only.",
    ),
}

OVERCLAIM_PHRASES = (
    "full public-reference port",
    "full local port",
    "complete local port",
    "complete public implementation",
    "complete implementation",
    "matches public implementation",
    "equivalent to public implementation",
    "identical to the public",
    "no difference from public",
    "faithful port",
    "line-for-line port",
    "line-for-line implementation",
)
NEGATION_MARKERS = ("not ", "not a ", "not an ", "no ", "without ")

COSMOS3_CONSUMER_FILES = (
    "Sol-LTX-Infer/scripts/run_cosmos3_sglang.sh",
    "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py",
    "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py",
)
DIMENSION_CONSUMER_NEEDLES = {
    "kwl_fusion": ("SGLANG_HQ_KWL_", "SGLANG_HQ_VARIANT=kwl"),
    "nvfp4_ffn": ("SGLANG_HQ_ENABLE_TE_NVFP4_FFN", "SGLANG_HQ_NVFP4_"),
}
COSMOS3_GPU_UNSUPPORTED_STATUSES = {
    "consumer_missing_on_cosmos3",
    "cosmos3_pure_algorithm_adapter_missing",
    "cosmos3_pure_algorithm_subset_already_baseline",
    "cosmos3_te_recipe_fused_adapter_semantics_mismatch",
    "cosmos3_probe_requires_concrete_backend_policy",
    "launcher_ready_route_label_only",
    "launcher_ready_but_attention_scope_consumer_missing",
    "launcher_ready_but_block_payload_consumer_missing",
    "consumer_wired_dependency_missing",
}
SEMANTIC_PERMUTATION_PUBLIC_COSMOS_CONFIG = {
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
FULL_PUBLIC_EQUIVALENCE_SCOPES = {
    "full_public_port",
    "line_for_line_port",
    "public_original_equivalent",
}
LOCAL_BASELINE_SCOPES = {
    "family_baseline",
    "training_free_baseline",
}
RUNTIME_ADAPTER_OR_PROBE_SCOPES = {
    "runtime_adapter_baseline",
    "build_transform_adapter",
}
KWL_GENERIC_POLICY_SCOPES = {
    "kwl_generic_policy_baseline",
}
SPARSE_POLICY_SHORT_GPU_DIAGNOSTICS = {
    "proxy_mask_prediction": "runs/20260619-200428-proxy_mask_prediction-diag2s-stats",
    "rotating_anchor_windows": "runs/20260619-200428-rotating_anchor_windows-diag2s-stats",
    "qk_coclustering": "runs/20260619-200428-qk_coclustering-diag2s-stats",
}
SPARSE_POLICY_SHORT_GPU_STATUS = (
    "consumer_wired_public_checker_short_gpu_diagnostic_only"
)
SVG_SAMPLE_MSE_SHORT_GPU_STATUS = (
    "consumer_wired_public_svg_core_short_gpu_quality_failed_speed_negative"
)
SVG_SAMPLE_MSE_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260620-025333-spatial_temporal_head_routing-svg-mse-diag2s"
)
SVG_SAMPLE_MSE_SHORT_GPU_BASELINE = "runs/20260619-202938-baseline-diag2s-832x480"
SPARGE_CORE_SHORT_GPU_STATUS = (
    "consumer_wired_public_sparge_core_short_gpu_quality_failed_speed_negative"
)
SPARGE_PROXY_CORE_SHORT_GPU_STATUS = (
    "consumer_wired_public_sparge_proxy_core_short_gpu_quality_failed_speed_negative"
)
SPARGE_QK_CORE_GPU_MISSING_STATUS = (
    "consumer_wired_public_sparge_qk_core_gpu_evidence_missing"
)
SPARGE_QK_CORE_SHORT_GPU_STATUS = (
    "consumer_wired_public_sparge_qk_core_short_gpu_quality_failed_speed_negative"
)
SPARGE_HEADWISE_CORE_GPU_MISSING_STATUS = (
    "consumer_wired_public_sparge_headwise_core_gpu_evidence_missing"
)
SPARGE_HEADWISE_CORE_SHORT_GPU_STATUS = (
    "consumer_wired_public_sparge_headwise_core_short_gpu_quality_failed_speed_negative"
)
MINFERENCE_DYNAMIC_CORE_GPU_MISSING_STATUS = (
    "consumer_wired_public_minference_dynamic_core_gpu_evidence_missing"
)
SVG_TEMPORAL_ANCHOR_CORE_GPU_MISSING_STATUS = (
    "consumer_wired_public_svg_temporal_anchor_core_gpu_evidence_missing"
)
SVG_TEMPORAL_ANCHOR_CORE_SHORT_GPU_STATUS = (
    "consumer_wired_public_svg_temporal_anchor_core_short_gpu_quality_failed_speed_negative"
)
SVG_TEMPORAL_ANCHOR_CORE_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260620-063030-rotating_anchor_windows-rotating-svg-window-core-diag2s-nousersite"
)
MINFERENCE_DYNAMIC_CORE_SHORT_GPU_STATUS = (
    "consumer_wired_public_minference_dynamic_core_short_gpu_quality_failed_speed_negative"
)
MINFERENCE_DYNAMIC_CORE_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260620-060642-dynamic_pattern_probe-dynamic-minference-core-diag2s-nousersite"
)
SPARGE_PROXY_CORE_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260620-041915-proxy_mask_prediction-proxy-fused-core-diag2s-nousersite"
)
SPARGE_QK_CORE_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260620-043934-qk_coclustering-qk-sparge-core-diag2s-nousersite"
)
SPARGE_HEADWISE_CORE_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260620-053721-headwise_adaptive_budgets-headwise-sparge-core-diag2s-nousersite"
)
SPARGE_CORE_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260620-015804-online_mask_search_reuse-sparge-core-diag2s-nousersite"
)
SPARGE_CORE_SHORT_GPU_BASELINE = (
    "runs/20260619-202938-baseline-diag2s-832x480"
)
PIECEWISE_PISA_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260620-065319-piecewise_pisa_env-pisa-density50-diag2s-nousersite"
)
PIECEWISE_PISA_SHORT_GPU_STATUS = (
    "consumer_wired_public_route_short_gpu_quality_failed_speed_positive"
)
TOME_PUBLIC_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260619-214131-tome_merge_restore-tome-public-diag2s-ropefix"
)
TOME_PUBLIC_SHORT_GPU_STATUS = (
    "consumer_wired_public_checker_short_gpu_quality_pass_speed_negative"
)
TOMESD_RANDOM2D_CORE_GPU_MISSING_STATUS = (
    "consumer_wired_public_tomesd_random2d_core_gpu_evidence_missing"
)
TOMESD_RANDOM2D_CORE_SHORT_GPU_STATUS = (
    "consumer_wired_public_tomesd_random2d_core_short_gpu_quality_pass_speed_negative"
)
TOMESD_RANDOM2D_CORE_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260620-050730-shape_stable_compute_mask-shape-random2d-diag2s-nousersite"
)
TEACACHE_PUBLIC_CONTROLLER_STATUS = (
    "consumer_wired_public_controller_residual_adapter_short_gpu_quality_pass_speed_low"
)
TEACACHE_PUBLIC_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260619-223334-teacache_signal_reuse-teacache-residual-normsignal-diag8s"
)
TEACACHE_PUBLIC_SHORT_GPU_BASELINE = (
    "runs/20260619-224056-baseline-dense-diag8s-832x480"
)
PAB_PUBLIC_CONTROLLER_STATUS = (
    "consumer_wired_public_controller_short_gpu_quality_failed"
)
PAB_PUBLIC_SHORT_GPU_DIAGNOSTICS = {
    "attention_broadcast": "runs/20260619-232104-attention_broadcast-pab-cross-public-diag8s",
    "block_layer_feature_cache": "runs/20260619-232104-block_layer_feature_cache-pab-mlp-public-diag8s",
}
PAB_PUBLIC_SHORT_GPU_BASELINE = (
    "runs/20260619-224056-baseline-dense-diag8s-832x480"
)
CAT_PUBLIC_SELECTOR_STATUS = "consumer_wired_public_cat_selector_gpu_evidence_missing"
CAT_PUBLIC_SELECTOR_SHORT_GPU_STATUS = (
    "consumer_wired_public_cat_selector_short_gpu_quality_pass_speed_negative"
)
CAT_PUBLIC_SELECTOR_SHORT_GPU_DIAGNOSTIC = (
    "runs/20260619-235137-cluster_representative_update-cat-selector-diag2s"
)
CAT_PUBLIC_SELECTOR_SHORT_GPU_BASELINE = (
    "runs/20260619-202938-baseline-diag2s-832x480"
)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manifest_id(data: dict[str, Any]) -> str:
    raw = data.get("id")
    if isinstance(raw, dict):
        return str(raw.get("name", ""))
    return str(raw or "")


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def extract_urls(value: str) -> list[str]:
    return [part.strip(".,)") for part in value.split() if is_url(part.strip(".,)"))]


def candidate_text(data: dict[str, Any]) -> str:
    refs = data.get("references", {})
    external = refs.get("external", {}) if isinstance(refs, dict) else {}
    parts = [
        str(data.get("description", "")),
        str(external.get("paper", "")),
        str(external.get("code", "")),
        str(external.get("notes", "")),
    ]
    return "\n".join(parts).lower()


def contains_unnegated_phrase(text: str, phrase: str) -> bool:
    start = 0
    while True:
        idx = text.find(phrase, start)
        if idx < 0:
            return False
        prefix = text[max(0, idx - 32) : idx]
        if not any(prefix.endswith(marker) for marker in NEGATION_MARKERS):
            return True
        start = idx + len(phrase)


def local_baseline_reference_note_problems(path: Path, notes: str) -> list[str]:
    """Local baselines may cite public work as motivation, not as implementation."""
    problems: list[str] = []
    note = notes.lower()
    if "baseline" not in note or "local" not in note:
        problems.append(
            f"{path.relative_to(ROOT)}: local-baseline reference notes must say "
            "this is a local baseline"
        )
    if "not a public" not in note and "not public" not in note:
        problems.append(
            f"{path.relative_to(ROOT)}: local-baseline reference notes must say "
            "the row is not a public implementation"
        )
    if "canonical" in note and "not a public" not in note and "not public" not in note:
        problems.append(
            f"{path.relative_to(ROOT)}: local-baseline reference notes use "
            "'canonical' without an explicit not-public boundary"
        )
    return problems


def adapter_probe_reference_note_problems(path: Path, notes: str) -> list[str]:
    """Runtime adapter/probe rows must not read as public implementations."""
    problems: list[str] = []
    note = notes.lower()
    if "adapter" not in note and "probe" not in note:
        problems.append(
            f"{path.relative_to(ROOT)}: runtime adapter/probe reference notes "
            "must say this is an adapter or probe"
        )
    not_public = (
        "not a public" in note
        or "not public" in note
        or "not a full public" in note
        or "not full public" in note
        or "not a line-for-line" in note
    )
    if not not_public:
        problems.append(
            f"{path.relative_to(ROOT)}: runtime adapter/probe reference notes "
            "must say the row is not a public implementation"
        )
    return problems


def cosmos3_baseline_or_ltx2_note_problems(path: Path, notes: str) -> list[str]:
    """Rows with no new Cosmos3 delta must say why they are probes, not candidates."""
    problems: list[str] = []
    note = notes.lower()
    if "cosmos3" not in note:
        problems.append(
            f"{path.relative_to(ROOT)}: Cosmos3-baseline/LTX2 rows must mention Cosmos3"
        )
    if "ltx2-only" not in note and "ltx2 only" not in note:
        problems.append(
            f"{path.relative_to(ROOT)}: Cosmos3-baseline/LTX2 rows must mark "
            "LTX2-only flags explicitly"
        )
    if "no new cosmos3 algorithm delta" not in note:
        problems.append(
            f"{path.relative_to(ROOT)}: Cosmos3-baseline/LTX2 rows must say "
            "there is no new Cosmos3 algorithm delta"
        )
    if "blocker probe" not in note:
        problems.append(
            f"{path.relative_to(ROOT)}: Cosmos3-baseline/LTX2 rows must say "
            "they are blocker probes"
        )
    return problems


def cosmos3_baseline_or_ltx2_manifest_problems(
    path: Path, data: dict[str, Any]
) -> list[str]:
    """No-delta KWL rows must not keep model-specific replay flags active."""
    params = data.get("efficiency", {}).get("params", {})
    flags = params.get("flags", []) if isinstance(params, dict) else []
    if flags:
        return [
            f"{path.relative_to(ROOT)}: no-delta Cosmos3/LTX2 KWL rows must "
            f"not keep active replay flags {flags!r}"
        ]
    return []


def te_recipe_manifest_boundary_problems(
    path: Path, data: dict[str, Any]
) -> list[str]:
    """Keep only model-agnostic TE recipe axes active in te_recipe_variant."""
    cid = str(data.get("id", {}).get("name", ""))
    if cid != "te_recipe_variant":
        return []
    params = data.get("efficiency", {}).get("params", {})
    if not isinstance(params, dict):
        return [f"{path.relative_to(ROOT)}: te_recipe_variant missing [efficiency.params]"]
    problems: list[str] = []
    if params.get("te_adapter", "") != "":
        problems.append(
            f"{path.relative_to(ROOT)}: te_recipe_variant must not enable a "
            "model-specific TE adapter in the manifest"
        )
    if not params.get("row_scaled_activation"):
        problems.append(
            f"{path.relative_to(ROOT)}: te_recipe_variant should preserve the "
            "generic row_scaled_activation recipe axis"
        )
    active_fused = [
        key
        for key in ("fused_proj_in_gelu", "fused_proj_out_bias_gate")
        if params.get(key)
    ]
    if active_fused:
        problems.append(
            f"{path.relative_to(ROOT)}: te_recipe_variant must not keep "
            f"LTX2-shaped fused epilogue flags active: {active_fused!r}"
        )
    return problems


def cosmos3_runtime_text() -> str:
    chunks: list[str] = []
    for rel in COSMOS3_CONSUMER_FILES:
        path = ROOT / rel
        if path.exists():
            chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks)


def cosmos3_consumes_dimension(runtime_text: str, dimension: str) -> bool:
    needles = DIMENSION_CONSUMER_NEEDLES.get(dimension, ())
    return any(needle in runtime_text for needle in needles)


def launcher_cosmos3_blocklist() -> set[str]:
    module_path = ROOT / "scripts" / "launch_candidate.py"
    spec = importlib.util.spec_from_file_location("launch_candidate_for_audit", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "cosmos3_blocked_candidate_ids"):
        return set(module.cosmos3_blocked_candidate_ids())
    return set(module.COSMOS3_UNSUPPORTED_GPU_REASONS)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def benchmark_stage_seconds(payload: dict[str, Any], name: str) -> float | None:
    if name == "Cosmos3DenoisingStage" and isinstance(payload.get("denoise_s"), (int, float)):
        return float(payload["denoise_s"])
    stage_map = payload.get("stage_seconds")
    if isinstance(stage_map, dict) and isinstance(stage_map.get(name), (int, float)):
        return float(stage_map[name])
    for stage in payload.get("steps") or []:
        if stage.get("name") == name and isinstance(stage.get("duration_ms"), (int, float)):
            return float(stage["duration_ms"]) / 1000.0
    return None


def validate_pab_public_short_gpu_diagnostic(cid: str) -> list[str]:
    run_rel = PAB_PUBLIC_SHORT_GPU_DIAGNOSTICS.get(cid)
    if not run_rel:
        return [f"{cid}: unexpected PAB short GPU diagnostic validation target"]
    run_dir = ROOT / run_rel
    baseline_dir = ROOT / PAB_PUBLIC_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in ("out.mp4", "benchmark.json", "quality.json", "collection.json"):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    expected_job = "3452366" if cid == "attention_broadcast" else "3452365"
    if metadata.get("slurm_job_id") != expected_job:
        errors.append(
            f"{cid}: expected slurm_job_id={expected_job!r}, got {metadata.get('slurm_job_id')!r}"
        )

    stats_files = sorted(outputs.glob("payload_cache_stats.rank*.json"))
    if len(stats_files) != 4:
        errors.append(f"{cid}: expected 4 rank-local payload-cache stats files, got {len(stats_files)}")
    for path in stats_files:
        stats = load_json(path)
        if stats.get("calls") != 512:
            errors.append(f"{cid}: expected calls=512 in {path.name}, got {stats.get('calls')!r}")
        if stats.get("pab_broadcast_misses") != 0:
            errors.append(
                f"{cid}: expected zero PAB broadcast misses in {path.name}, got {stats.get('pab_broadcast_misses')!r}"
            )
        if cid == "attention_broadcast":
            if stats.get("hits") != 192 or stats.get("pab_broadcast_flags") != 192:
                errors.append(
                    f"{cid}: expected hits=192 and pab_broadcast_flags=192 in {path.name}, "
                    f"got hits={stats.get('hits')!r}, flags={stats.get('pab_broadcast_flags')!r}"
                )
        else:
            if stats.get("hits") != 2 or stats.get("pab_mlp_next") != 1:
                errors.append(
                    f"{cid}: expected hits=2 and pab_mlp_next=1 in {path.name}, "
                    f"got hits={stats.get('hits')!r}, pab_mlp_next={stats.get('pab_mlp_next')!r}"
                )

    benchmark = load_json(outputs / "benchmark.json")
    baseline = load_json(baseline_dir / "outputs/benchmark.json")
    cand_denoise = benchmark_stage_seconds(benchmark, "Cosmos3DenoisingStage")
    base_denoise = benchmark_stage_seconds(baseline, "Cosmos3DenoisingStage")
    if cand_denoise is None or base_denoise is None:
        errors.append(f"{cid}: missing matched 8-step denoise timings")
    elif base_denoise / cand_denoise >= 1.0:
        errors.append(
            f"{cid}: status says speed is negative, but matched denoise speedup is {base_denoise / cand_denoise:.4f}x"
        )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: expected blocked_quality, got {quality.get('status')!r}"
        )
    blockers = quality.get("promotion_blockers") or []
    if "nvidia_gemini:fail:high" not in blockers:
        errors.append(f"{cid}: expected nvidia_gemini:fail:high blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "fail":
        errors.append(f"{cid}: expected Gemini fail, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    if not frames.exists() or len(list(frames.glob("*.png"))) != 15:
        count = len(list(frames.glob("*.png"))) if frames.exists() else 0
        errors.append(f"{cid}: expected 15 extracted diagnostic frames, got {count}")
    return errors


def validate_sparse_policy_short_gpu_diagnostic(cid: str) -> list[str]:
    rel = SPARSE_POLICY_SHORT_GPU_DIAGNOSTICS.get(cid)
    if not rel:
        return [f"{cid}: missing sparse-policy short GPU diagnostic run mapping"]

    run_dir = ROOT / rel
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in (
        "out.mp4",
        "benchmark.json",
        "quality.json",
        "collection.json",
        "piecewise_attn_stats.json",
    ):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if not metadata.get("slurm_job_id"):
        errors.append(f"{cid}: diagnostic metadata is missing slurm_job_id")

    stats = load_json(outputs / "piecewise_attn_stats.json")
    if stats.get("policy_route_calls") != 128:
        errors.append(f"{cid}: expected policy_route_calls=128, got {stats.get('policy_route_calls')!r}")
    if stats.get("sparse_calls") != 128:
        errors.append(f"{cid}: expected sparse_calls=128, got {stats.get('sparse_calls')!r}")
    if stats.get("fallback_calls") != 0:
        errors.append(f"{cid}: expected fallback_calls=0, got {stats.get('fallback_calls')!r}")
    route_modes = stats.get("by_route_mode") or {}
    if cid not in route_modes:
        errors.append(f"{cid}: diagnostic stats do not include route mode {cid!r}")

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: short diagnostic should remain blocked_quality, got {quality.get('status')!r}"
        )
    blockers = [str(item) for item in quality.get("promotion_blockers") or []]
    stale_blockers = (
        "baseline_frame_missing",
        "baseline_frames_missing",
        "dependencies_missing",
        "image metric dependencies missing",
    )
    for blocker in blockers:
        if any(needle in blocker for needle in stale_blockers):
            errors.append(f"{cid}: stale short diagnostic quality blocker {blocker!r}")
    if not any(blocker.startswith("nvidia_gemini:") for blocker in blockers):
        errors.append(f"{cid}: expected matched-short Gemini quality blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    frames = outputs / "frames"
    if not frames.exists() or len(list(frames.glob("*.png"))) != 178:
        count = len(list(frames.glob("*.png"))) if frames.exists() else 0
        errors.append(f"{cid}: expected 178 extracted diagnostic frames, got {count}")
    return errors


def validate_svg_sample_mse_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "spatial_temporal_head_routing":
        return [f"{cid}: unexpected SVG sample-MSE short GPU diagnostic validation target"]

    run_dir = ROOT / SVG_SAMPLE_MSE_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / SVG_SAMPLE_MSE_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in (
        "out.mp4",
        "benchmark.json",
        "quality.json",
        "collection.json",
        "piecewise_attn_stats.json",
    ):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3455836":
        errors.append(
            f"{cid}: expected slurm_job_id='3455836', got {metadata.get('slurm_job_id')!r}"
        )
    env_preview = ((metadata.get("candidate_dry_run") or {}).get("env_preview") or {})
    backend_config = str(env_preview.get("SGLANG_HQ_ATTENTION_BACKEND_CONFIG", ""))
    if "piecewise_frame_size=2" not in backend_config:
        errors.append(f"{cid}: diagnostic launch did not carry piecewise_frame_size=2")

    stats = load_json(outputs / "piecewise_attn_stats.json")
    if stats.get("policy_route_calls") != 128:
        errors.append(f"{cid}: expected policy_route_calls=128, got {stats.get('policy_route_calls')!r}")
    if stats.get("sparse_calls") != 128:
        errors.append(f"{cid}: expected sparse_calls=128, got {stats.get('sparse_calls')!r}")
    if stats.get("fallback_calls") != 0:
        errors.append(f"{cid}: expected fallback_calls=0, got {stats.get('fallback_calls')!r}")
    route_modes = stats.get("by_route_mode") or {}
    if cid not in route_modes:
        errors.append(f"{cid}: diagnostic stats do not include route mode {cid!r}")
    by_shape = stats.get("by_shape") or {}
    selected_modes = {
        str(item.get("selected_route_mode"))
        for item in by_shape.values()
        if isinstance(item, dict)
    }
    if selected_modes != {"svg_sample_mse_head_selection"}:
        errors.append(f"{cid}: expected SVG sample-MSE selected route, got {selected_modes!r}")
    densities = [
        item.get("exact_density")
        for item in by_shape.values()
        if isinstance(item, dict) and isinstance(item.get("exact_density"), (int, float))
    ]
    if not densities:
        errors.append(f"{cid}: missing exact density in piecewise stats")
    elif min(float(density) for density in densities) <= 0.12:
        errors.append(f"{cid}: SVG selector should keep prefix/video sinks above nominal density, got {densities!r}")

    collection = load_json(outputs / "collection.json")
    baseline_collection = load_json(baseline_dir / "outputs/collection.json")
    timing = collection.get("timing") or {}
    baseline_timing = baseline_collection.get("timing") or {}
    for key in ("total_s", "denoise_s"):
        if not isinstance(timing.get(key), (int, float)):
            errors.append(f"{cid}: missing diagnostic collection timing {key}")
        if not isinstance(baseline_timing.get(key), (int, float)):
            errors.append(f"{cid}: missing matched baseline collection timing {key}")
    if isinstance(timing.get("denoise_s"), (int, float)) and isinstance(
        baseline_timing.get("denoise_s"), (int, float)
    ):
        if timing["denoise_s"] <= baseline_timing["denoise_s"]:
            errors.append(
                f"{cid}: expected current SVG sample-MSE denoise to remain slower "
                f"than baseline, got {timing['denoise_s']!r} <= {baseline_timing['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: short diagnostic should remain blocked_quality, got {quality.get('status')!r}"
        )
    blockers = [str(item) for item in quality.get("promotion_blockers") or []]
    if "nvidia_gemini:fail:high" not in blockers:
        errors.append(f"{cid}: expected nvidia_gemini:fail:high blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "fail":
        errors.append(f"{cid}: expected Gemini fail, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    frame_count = len(list(frames.glob("*.png"))) if frames.exists() else 0
    if frame_count < 100:
        errors.append(f"{cid}: expected extracted diagnostic frames, got {frame_count}")
    return errors


def validate_svg_temporal_anchor_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "rotating_anchor_windows":
        return [f"{cid}: unexpected SVG temporal-anchor short GPU diagnostic validation target"]

    run_dir = ROOT / SVG_TEMPORAL_ANCHOR_CORE_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / SPARGE_CORE_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in (
        "out.mp4",
        "benchmark.json",
        "quality.json",
        "collection.json",
        "piecewise_attn_stats.json",
    ):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3459306":
        errors.append(
            f"{cid}: expected slurm_job_id='3459306', got {metadata.get('slurm_job_id')!r}"
        )
    env_preview = ((metadata.get("candidate_dry_run") or {}).get("env_preview") or {})
    backend_config = str(env_preview.get("SGLANG_HQ_ATTENTION_BACKEND_CONFIG", ""))
    if "piecewise_frame_size=2" not in backend_config:
        errors.append(f"{cid}: diagnostic launch did not carry piecewise_frame_size=2")
    if "piecewise_policy_family=sparse_videogen_first_frame_temporal_window" not in backend_config:
        errors.append(f"{cid}: diagnostic launch did not carry SVG temporal-window family")

    stats = load_json(outputs / "piecewise_attn_stats.json")
    if stats.get("policy_route_calls") != 128:
        errors.append(f"{cid}: expected policy_route_calls=128, got {stats.get('policy_route_calls')!r}")
    if stats.get("sparse_calls") != 128:
        errors.append(f"{cid}: expected sparse_calls=128, got {stats.get('sparse_calls')!r}")
    if stats.get("fallback_calls") != 0:
        errors.append(f"{cid}: expected fallback_calls=0, got {stats.get('fallback_calls')!r}")
    route_modes = stats.get("by_route_mode") or {}
    if cid not in route_modes:
        errors.append(f"{cid}: diagnostic stats do not include route mode {cid!r}")
    by_shape = stats.get("by_shape") or {}
    selected_modes = {
        str(item.get("selected_route_mode"))
        for item in by_shape.values()
        if isinstance(item, dict)
    }
    if selected_modes != {"svg_first_frame_temporal_window"}:
        errors.append(f"{cid}: expected SVG temporal-window selected route, got {selected_modes!r}")
    densities = [
        item.get("exact_density")
        for item in by_shape.values()
        if isinstance(item, dict) and isinstance(item.get("exact_density"), (int, float))
    ]
    if not densities:
        errors.append(f"{cid}: missing exact density in piecewise stats")
    elif min(float(density) for density in densities) <= 0.2:
        errors.append(f"{cid}: expected first-frame/window density above nominal top-k, got {densities!r}")

    benchmark = load_json(outputs / "benchmark.json")
    baseline = load_json(baseline_dir / "outputs/benchmark.json")
    for key in ("total_s", "denoise_s"):
        if not isinstance(benchmark.get(key), (int, float)):
            errors.append(f"{cid}: missing diagnostic benchmark {key}")
        if not isinstance(baseline.get(key), (int, float)):
            errors.append(f"{cid}: missing matched short baseline benchmark {key}")
    if isinstance(benchmark.get("denoise_s"), (int, float)) and isinstance(
        baseline.get("denoise_s"), (int, float)
    ):
        if benchmark["denoise_s"] <= baseline["denoise_s"]:
            errors.append(
                f"{cid}: expected current SVG temporal-window denoise to remain slower "
                f"than baseline, got {benchmark['denoise_s']!r} <= {baseline['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: short diagnostic should remain blocked_quality, got {quality.get('status')!r}"
        )
    blockers = [str(item) for item in quality.get("promotion_blockers") or []]
    if "nvidia_gemini:fail:high" not in blockers:
        errors.append(f"{cid}: expected nvidia_gemini:fail:high blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "fail":
        errors.append(f"{cid}: expected Gemini fail, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    frame_count = len(list(frames.glob("*.png"))) if frames.exists() else 0
    if frame_count < 15:
        errors.append(f"{cid}: expected extracted diagnostic frames, got {frame_count}")
    return errors


def validate_sparge_core_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "online_mask_search_reuse":
        return [f"{cid}: unexpected Sparge core short GPU diagnostic validation target"]

    run_dir = ROOT / SPARGE_CORE_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / SPARGE_CORE_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in (
        "out.mp4",
        "benchmark.json",
        "quality.json",
        "collection.json",
        "piecewise_attn_stats.json",
    ):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3454721":
        errors.append(
            f"{cid}: expected slurm_job_id='3454721', got {metadata.get('slurm_job_id')!r}"
        )

    stats = load_json(outputs / "piecewise_attn_stats.json")
    if stats.get("policy_route_calls") != 128:
        errors.append(f"{cid}: expected policy_route_calls=128, got {stats.get('policy_route_calls')!r}")
    if stats.get("sparse_calls") != 128:
        errors.append(f"{cid}: expected sparse_calls=128, got {stats.get('sparse_calls')!r}")
    if stats.get("fallback_calls") != 0:
        errors.append(f"{cid}: expected fallback_calls=0, got {stats.get('fallback_calls')!r}")
    route_modes = stats.get("by_route_mode") or {}
    if cid not in route_modes:
        errors.append(f"{cid}: diagnostic stats do not include route mode {cid!r}")

    benchmark = load_json(outputs / "benchmark.json")
    baseline = load_json(baseline_dir / "outputs/benchmark.json")
    for key in ("total_s", "denoise_s"):
        if not isinstance(benchmark.get(key), (int, float)):
            errors.append(f"{cid}: missing diagnostic benchmark {key}")
        if not isinstance(baseline.get(key), (int, float)):
            errors.append(f"{cid}: missing matched short baseline benchmark {key}")
    if isinstance(benchmark.get("denoise_s"), (int, float)) and isinstance(
        baseline.get("denoise_s"), (int, float)
    ):
        if benchmark["denoise_s"] <= baseline["denoise_s"]:
            errors.append(
                f"{cid}: expected current Sparge-core denoise to remain slower "
                f"than baseline, got {benchmark['denoise_s']!r} <= {baseline['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: short diagnostic should remain blocked_quality, got {quality.get('status')!r}"
        )
    blockers = [str(item) for item in quality.get("promotion_blockers") or []]
    if "nvidia_gemini:fail:high" not in blockers:
        errors.append(f"{cid}: expected nvidia_gemini:fail:high blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    elif lpips["mean"] < 0.7:
        errors.append(f"{cid}: expected high LPIPS drift, got mean={lpips['mean']!r}")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "fail":
        errors.append(f"{cid}: expected Gemini fail, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    if not frames.exists() or len(list(frames.glob("*.png"))) != 178:
        count = len(list(frames.glob("*.png"))) if frames.exists() else 0
        errors.append(f"{cid}: expected 178 extracted diagnostic frames, got {count}")
    return errors


def validate_sparge_proxy_core_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "proxy_mask_prediction":
        return [f"{cid}: unexpected Sparge proxy-core short GPU diagnostic validation target"]

    run_dir = ROOT / SPARGE_PROXY_CORE_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / SPARGE_CORE_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in (
        "out.mp4",
        "benchmark.json",
        "quality.json",
        "collection.json",
        "piecewise_attn_stats.json",
    ):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3457934":
        errors.append(
            f"{cid}: expected slurm_job_id='3457934', got {metadata.get('slurm_job_id')!r}"
        )
    launch_text = (run_dir / "launch.sh").read_text(errors="ignore") if (run_dir / "launch.sh").exists() else ""
    if "PYTHONNOUSERSITE=1" not in launch_text:
        errors.append(f"{cid}: diagnostic launch should isolate user-site packages")

    stats = load_json(outputs / "piecewise_attn_stats.json")
    if stats.get("policy_route_calls") != 128:
        errors.append(f"{cid}: expected policy_route_calls=128, got {stats.get('policy_route_calls')!r}")
    if stats.get("sparse_calls") != 128:
        errors.append(f"{cid}: expected sparse_calls=128, got {stats.get('sparse_calls')!r}")
    if stats.get("fallback_calls") != 0:
        errors.append(f"{cid}: expected fallback_calls=0, got {stats.get('fallback_calls')!r}")
    route_modes = stats.get("by_route_mode") or {}
    if cid not in route_modes:
        errors.append(f"{cid}: diagnostic stats do not include route mode {cid!r}")
    by_shape = stats.get("by_shape") or {}
    selected_modes = {
        str(item.get("selected_route_mode"))
        for item in by_shape.values()
        if isinstance(item, dict)
    }
    if selected_modes != {"spargeattn_quantized_mean_similarity_proxy"}:
        errors.append(f"{cid}: expected Sparge proxy selected route, got {selected_modes!r}")

    collection = load_json(outputs / "collection.json")
    baseline_collection = load_json(baseline_dir / "outputs/collection.json")
    timing = collection.get("timing") or {}
    baseline_timing = baseline_collection.get("timing") or {}
    for key in ("total_s", "denoise_s"):
        if not isinstance(timing.get(key), (int, float)):
            errors.append(f"{cid}: missing diagnostic collection timing {key}")
        if not isinstance(baseline_timing.get(key), (int, float)):
            errors.append(f"{cid}: missing matched baseline collection timing {key}")
    if isinstance(timing.get("denoise_s"), (int, float)) and isinstance(
        baseline_timing.get("denoise_s"), (int, float)
    ):
        if timing["denoise_s"] <= baseline_timing["denoise_s"]:
            errors.append(
                f"{cid}: expected current Sparge proxy denoise to remain slower "
                f"than baseline, got {timing['denoise_s']!r} <= {baseline_timing['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: short diagnostic should remain blocked_quality, got {quality.get('status')!r}"
        )
    blockers = [str(item) for item in quality.get("promotion_blockers") or []]
    if not any(blocker.startswith("nvidia_gemini:") for blocker in blockers):
        errors.append(f"{cid}: expected matched-short Gemini quality blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    elif lpips["mean"] < 0.5:
        errors.append(f"{cid}: expected high LPIPS drift, got mean={lpips['mean']!r}")
    retry = load_json(outputs / "nvidia_gemini_retry.json")
    if retry.get("overall") != "fail":
        errors.append(f"{cid}: expected Gemini retry fail, got {retry.get('overall')!r}")
    retry_artifacts = retry.get("new_artifacts") or []
    if not any(
        isinstance(item, dict) and item.get("severity") == "high"
        for item in retry_artifacts
    ):
        errors.append(f"{cid}: expected high-severity Gemini retry artifact, got {retry_artifacts!r}")
    frames = outputs / "frames"
    frame_count = len(list(frames.glob("*.png"))) if frames.exists() else 0
    if frame_count < 16:
        errors.append(f"{cid}: expected extracted diagnostic frames, got {frame_count}")
    return errors


def validate_sparge_qk_core_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "qk_coclustering":
        return [f"{cid}: unexpected Sparge Q/K-core short GPU diagnostic validation target"]

    run_dir = ROOT / SPARGE_QK_CORE_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / SPARGE_CORE_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in (
        "out.mp4",
        "benchmark.json",
        "quality.json",
        "collection.json",
        "piecewise_attn_stats.json",
    ):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3458125":
        errors.append(
            f"{cid}: expected slurm_job_id='3458125', got {metadata.get('slurm_job_id')!r}"
        )
    launch_text = (run_dir / "launch.sh").read_text(errors="ignore") if (run_dir / "launch.sh").exists() else ""
    if "PYTHONNOUSERSITE=1" not in launch_text:
        errors.append(f"{cid}: diagnostic launch should isolate user-site packages")

    stats = load_json(outputs / "piecewise_attn_stats.json")
    if stats.get("policy_route_calls") != 128:
        errors.append(f"{cid}: expected policy_route_calls=128, got {stats.get('policy_route_calls')!r}")
    if stats.get("sparse_calls") != 128:
        errors.append(f"{cid}: expected sparse_calls=128, got {stats.get('sparse_calls')!r}")
    if stats.get("fallback_calls") != 0:
        errors.append(f"{cid}: expected fallback_calls=0, got {stats.get('fallback_calls')!r}")
    route_modes = stats.get("by_route_mode") or {}
    if cid not in route_modes:
        errors.append(f"{cid}: diagnostic stats do not include route mode {cid!r}")
    by_shape = stats.get("by_shape") or {}
    selected_modes = {
        str(item.get("selected_route_mode"))
        for item in by_shape.values()
        if isinstance(item, dict)
    }
    if selected_modes != {"spargeattn_qk_mean_similarity_block_map"}:
        errors.append(f"{cid}: expected Sparge Q/K selected route, got {selected_modes!r}")

    collection = load_json(outputs / "collection.json")
    baseline_collection = load_json(baseline_dir / "outputs/collection.json")
    timing = collection.get("timing") or {}
    baseline_timing = baseline_collection.get("timing") or {}
    for key in ("total_s", "denoise_s"):
        if not isinstance(timing.get(key), (int, float)):
            errors.append(f"{cid}: missing diagnostic collection timing {key}")
        if not isinstance(baseline_timing.get(key), (int, float)):
            errors.append(f"{cid}: missing matched baseline collection timing {key}")
    if isinstance(timing.get("denoise_s"), (int, float)) and isinstance(
        baseline_timing.get("denoise_s"), (int, float)
    ):
        if timing["denoise_s"] <= baseline_timing["denoise_s"]:
            errors.append(
                f"{cid}: expected current Sparge Q/K denoise to remain slower "
                f"than baseline, got {timing['denoise_s']!r} <= {baseline_timing['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: short diagnostic should remain blocked_quality, got {quality.get('status')!r}"
        )
    blockers = [str(item) for item in quality.get("promotion_blockers") or []]
    if "nvidia_gemini:fail:high" not in blockers:
        errors.append(f"{cid}: expected nvidia_gemini:fail:high blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    elif lpips["mean"] < 0.6:
        errors.append(f"{cid}: expected high LPIPS drift, got mean={lpips['mean']!r}")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "fail":
        errors.append(f"{cid}: expected Gemini fail, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    frame_count = len(list(frames.glob("*.png"))) if frames.exists() else 0
    if frame_count < 16:
        errors.append(f"{cid}: expected extracted diagnostic frames, got {frame_count}")
    return errors


def validate_sparge_headwise_core_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "headwise_adaptive_budgets":
        return [f"{cid}: unexpected Sparge headwise-core short GPU diagnostic validation target"]

    run_dir = ROOT / SPARGE_HEADWISE_CORE_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / SPARGE_CORE_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in (
        "out.mp4",
        "benchmark.json",
        "quality.json",
        "collection.json",
        "piecewise_attn_stats.json",
    ):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3458697":
        errors.append(
            f"{cid}: expected slurm_job_id='3458697', got {metadata.get('slurm_job_id')!r}"
        )
    launch_text = (run_dir / "launch.sh").read_text(errors="ignore") if (run_dir / "launch.sh").exists() else ""
    if "PYTHONNOUSERSITE=1" not in launch_text:
        errors.append(f"{cid}: diagnostic launch should isolate user-site packages")

    stats = load_json(outputs / "piecewise_attn_stats.json")
    if stats.get("policy_route_calls") != 128:
        errors.append(f"{cid}: expected policy_route_calls=128, got {stats.get('policy_route_calls')!r}")
    if stats.get("sparse_calls") != 128:
        errors.append(f"{cid}: expected sparse_calls=128, got {stats.get('sparse_calls')!r}")
    if stats.get("fallback_calls") != 0:
        errors.append(f"{cid}: expected fallback_calls=0, got {stats.get('fallback_calls')!r}")
    route_modes = stats.get("by_route_mode") or {}
    if cid not in route_modes:
        errors.append(f"{cid}: diagnostic stats do not include route mode {cid!r}")
    by_shape = stats.get("by_shape") or {}
    selected_modes = {
        str(item.get("selected_route_mode"))
        for item in by_shape.values()
        if isinstance(item, dict)
    }
    if selected_modes != {"spargeattn_headwise_topk_budget_block_map"}:
        errors.append(f"{cid}: expected Sparge headwise selected route, got {selected_modes!r}")

    collection = load_json(outputs / "collection.json")
    baseline_collection = load_json(baseline_dir / "outputs/collection.json")
    timing = collection.get("timing") or {}
    baseline_timing = baseline_collection.get("timing") or {}
    for key in ("total_s", "denoise_s"):
        if not isinstance(timing.get(key), (int, float)):
            errors.append(f"{cid}: missing diagnostic collection timing {key}")
        if not isinstance(baseline_timing.get(key), (int, float)):
            errors.append(f"{cid}: missing matched baseline collection timing {key}")
    if isinstance(timing.get("denoise_s"), (int, float)) and isinstance(
        baseline_timing.get("denoise_s"), (int, float)
    ):
        if timing["denoise_s"] <= baseline_timing["denoise_s"]:
            errors.append(
                f"{cid}: expected current Sparge headwise denoise to remain slower "
                f"than baseline, got {timing['denoise_s']!r} <= {baseline_timing['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: short diagnostic should remain blocked_quality, got {quality.get('status')!r}"
        )
    blockers = [str(item) for item in quality.get("promotion_blockers") or []]
    if "nvidia_gemini:fail:high" not in blockers:
        errors.append(f"{cid}: expected nvidia_gemini:fail:high blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    elif lpips["mean"] < 0.6:
        errors.append(f"{cid}: expected high LPIPS drift, got mean={lpips['mean']!r}")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "fail":
        errors.append(f"{cid}: expected Gemini fail, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    frame_count = len(list(frames.glob("*.png"))) if frames.exists() else 0
    if frame_count < 16:
        errors.append(f"{cid}: expected extracted diagnostic frames, got {frame_count}")
    return errors


def validate_minference_dynamic_core_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "dynamic_pattern_probe":
        return [f"{cid}: unexpected MInference dynamic-core short GPU diagnostic validation target"]

    run_dir = ROOT / MINFERENCE_DYNAMIC_CORE_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / SPARGE_CORE_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in (
        "out.mp4",
        "benchmark.json",
        "quality.json",
        "collection.json",
        "piecewise_attn_stats.json",
    ):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3459048":
        errors.append(
            f"{cid}: expected slurm_job_id='3459048', got {metadata.get('slurm_job_id')!r}"
        )
    launch_text = (run_dir / "launch.sh").read_text(errors="ignore") if (run_dir / "launch.sh").exists() else ""
    if "PYTHONNOUSERSITE=1" not in launch_text:
        errors.append(f"{cid}: diagnostic launch should isolate user-site packages")

    stats = load_json(outputs / "piecewise_attn_stats.json")
    if stats.get("policy_route_calls") != 128:
        errors.append(f"{cid}: expected policy_route_calls=128, got {stats.get('policy_route_calls')!r}")
    if stats.get("sparse_calls") != 128:
        errors.append(f"{cid}: expected sparse_calls=128, got {stats.get('sparse_calls')!r}")
    if stats.get("fallback_calls") != 0:
        errors.append(f"{cid}: expected fallback_calls=0, got {stats.get('fallback_calls')!r}")
    route_modes = stats.get("by_route_mode") or {}
    if cid not in route_modes:
        errors.append(f"{cid}: diagnostic stats do not include route mode {cid!r}")
    by_shape = stats.get("by_shape") or {}
    selected_modes = {
        str(item.get("selected_route_mode"))
        for item in by_shape.values()
        if isinstance(item, dict)
    }
    if selected_modes != {"minference_dynamic_pattern_bank"}:
        errors.append(f"{cid}: expected MInference dynamic selected route, got {selected_modes!r}")
    exact_densities = [
        item.get("exact_density")
        for item in by_shape.values()
        if isinstance(item, dict) and isinstance(item.get("exact_density"), (int, float))
    ]
    if not exact_densities:
        errors.append(f"{cid}: missing exact density in diagnostic stats")
    elif not (0.0 < float(exact_densities[0]) < 1.0):
        errors.append(f"{cid}: expected sparse exact density, got {exact_densities[0]!r}")

    collection = load_json(outputs / "collection.json")
    baseline_collection = load_json(baseline_dir / "outputs/collection.json")
    timing = collection.get("timing") or {}
    baseline_timing = baseline_collection.get("timing") or {}
    for key in ("total_s", "denoise_s"):
        if not isinstance(timing.get(key), (int, float)):
            errors.append(f"{cid}: missing diagnostic collection timing {key}")
        if not isinstance(baseline_timing.get(key), (int, float)):
            errors.append(f"{cid}: missing matched baseline collection timing {key}")
    if isinstance(timing.get("denoise_s"), (int, float)) and isinstance(
        baseline_timing.get("denoise_s"), (int, float)
    ):
        if timing["denoise_s"] <= baseline_timing["denoise_s"]:
            errors.append(
                f"{cid}: expected current MInference dynamic denoise to remain slower "
                f"than baseline, got {timing['denoise_s']!r} <= {baseline_timing['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: short diagnostic should remain blocked_quality, got {quality.get('status')!r}"
        )
    blockers = [str(item) for item in quality.get("promotion_blockers") or []]
    if "nvidia_gemini:fail:high" not in blockers:
        errors.append(f"{cid}: expected nvidia_gemini:fail:high blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    elif lpips["mean"] < 0.7:
        errors.append(f"{cid}: expected very high LPIPS drift, got mean={lpips['mean']!r}")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "fail":
        errors.append(f"{cid}: expected Gemini fail, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    frame_count = len(list(frames.glob("*.png"))) if frames.exists() else 0
    if frame_count < 16:
        errors.append(f"{cid}: expected extracted diagnostic frames, got {frame_count}")
    return errors


def validate_piecewise_pisa_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "piecewise_pisa_env":
        return [f"{cid}: unexpected PISA short GPU diagnostic validation target"]

    run_dir = ROOT / PIECEWISE_PISA_SHORT_GPU_DIAGNOSTIC
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in (
        "out.mp4",
        "benchmark.json",
        "quality.json",
        "collection.json",
        "piecewise_attn_stats.json",
    ):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3459457":
        errors.append(
            f"{cid}: expected slurm_job_id='3459457', got {metadata.get('slurm_job_id')!r}"
        )

    stats = load_json(outputs / "piecewise_attn_stats.json")
    if stats.get("total_calls") != 128:
        errors.append(f"{cid}: expected total_calls=128, got {stats.get('total_calls')!r}")
    if stats.get("sparse_calls") != 128:
        errors.append(f"{cid}: expected sparse_calls=128, got {stats.get('sparse_calls')!r}")
    if stats.get("fallback_calls") != 0:
        errors.append(f"{cid}: expected fallback_calls=0, got {stats.get('fallback_calls')!r}")
    if stats.get("policy_route_calls") != 0:
        errors.append(
            f"{cid}: expected policy_route_calls=0 for public score route, got {stats.get('policy_route_calls')!r}"
        )
    route_modes = stats.get("by_route_mode") or {}
    if "score" not in route_modes:
        errors.append(f"{cid}: diagnostic stats do not include public score route mode")
    shape_stats = stats.get("by_shape") or {}
    exact_densities = [
        value.get("exact_density")
        for value in shape_stats.values()
        if isinstance(value, dict)
    ]
    if not exact_densities or any(
        not isinstance(value, (int, float)) or abs(float(value) - 0.5) > 1e-6
        for value in exact_densities
    ):
        errors.append(f"{cid}: expected exact density 0.5, got {exact_densities!r}")

    benchmark = load_json(outputs / "benchmark.json")
    baseline = load_json(
        ROOT / "runs/20260619-202938-baseline-diag2s-832x480/outputs/benchmark.json"
    )
    for key in ("total_s", "denoise_s"):
        if not isinstance(benchmark.get(key), (int, float)):
            errors.append(f"{cid}: missing diagnostic benchmark {key}")
        if not isinstance(baseline.get(key), (int, float)):
            errors.append(f"{cid}: missing matched short baseline benchmark {key}")
    if isinstance(benchmark.get("denoise_s"), (int, float)) and isinstance(
        baseline.get("denoise_s"), (int, float)
    ):
        if benchmark["denoise_s"] >= baseline["denoise_s"]:
            errors.append(
                f"{cid}: expected short sparse denoise to beat baseline, got "
                f"{benchmark['denoise_s']!r} >= {baseline['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "blocked_quality":
        errors.append(
            f"{cid}: short diagnostic should remain blocked_quality, got {quality.get('status')!r}"
        )
    blockers = [str(item) for item in quality.get("promotion_blockers") or []]
    if not any(blocker.startswith("nvidia_gemini:") for blocker in blockers):
        errors.append(f"{cid}: expected matched-short Gemini quality blocker, got {blockers!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    elif lpips["mean"] > 0.3:
        errors.append(f"{cid}: expected density-0.5 LPIPS mean <= 0.3, got {lpips['mean']!r}")
    retry = load_json(outputs / "nvidia_gemini_retry.json")
    if retry.get("overall") != "fail":
        errors.append(f"{cid}: expected Gemini retry fail, got {retry.get('overall')!r}")
    retry_artifacts = retry.get("new_artifacts") or []
    if not any(
        isinstance(item, dict) and item.get("severity") == "high"
        for item in retry_artifacts
    ):
        errors.append(f"{cid}: expected high-severity Gemini retry artifact, got {retry_artifacts!r}")
    frames = outputs / "frames"
    if not frames.exists() or len(list(frames.glob("*.png"))) != 15:
        count = len(list(frames.glob("*.png"))) if frames.exists() else 0
        errors.append(f"{cid}: expected 15 extracted diagnostic frames, got {count}")
    return errors


def validate_tome_public_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "tome_merge_restore":
        return [f"{cid}: unexpected ToMe short GPU diagnostic validation target"]

    run_dir = ROOT / TOME_PUBLIC_SHORT_GPU_DIAGNOSTIC
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in ("out.mp4", "benchmark.json", "quality.json", "collection.json"):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3446886":
        errors.append(
            f"{cid}: expected slurm_job_id='3446886', got {metadata.get('slurm_job_id')!r}"
        )

    benchmark = load_json(outputs / "benchmark.json")
    baseline = load_json(
        ROOT / "runs/20260619-202938-baseline-diag2s-832x480/outputs/benchmark.json"
    )
    for key in ("total_s", "denoise_s"):
        if not isinstance(benchmark.get(key), (int, float)):
            errors.append(f"{cid}: missing diagnostic benchmark {key}")
        if not isinstance(baseline.get(key), (int, float)):
            errors.append(f"{cid}: missing matched short baseline benchmark {key}")
    if isinstance(benchmark.get("denoise_s"), (int, float)) and isinstance(
        baseline.get("denoise_s"), (int, float)
    ):
        if benchmark["denoise_s"] <= baseline["denoise_s"]:
            errors.append(
                f"{cid}: expected current short ToMe denoise to remain slower "
                f"than baseline, got {benchmark['denoise_s']!r} <= {baseline['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "available":
        errors.append(
            f"{cid}: short diagnostic quality should be available, got {quality.get('status')!r}"
        )
    if quality.get("promotion_blockers"):
        errors.append(f"{cid}: expected no short quality blockers, got {quality.get('promotion_blockers')!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "pass":
        errors.append(f"{cid}: expected Gemini pass, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    if not frames.exists() or len(list(frames.glob("*.png"))) != 178:
        count = len(list(frames.glob("*.png"))) if frames.exists() else 0
        errors.append(f"{cid}: expected 178 extracted diagnostic frames, got {count}")
    return errors


def validate_tomesd_random2d_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "shape_stable_compute_mask":
        return [f"{cid}: unexpected ToMeSD random2D short GPU diagnostic validation target"]

    run_dir = ROOT / TOMESD_RANDOM2D_CORE_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / "runs/20260619-202938-baseline-diag2s-832x480"
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in ("out.mp4", "benchmark.json", "quality.json", "collection.json"):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3458395":
        errors.append(
            f"{cid}: expected slurm_job_id='3458395', got {metadata.get('slurm_job_id')!r}"
        )
    launch = (run_dir / "launch.sh").read_text(errors="ignore") if (run_dir / "launch.sh").exists() else ""
    for needle in (
        "PYTHONNOUSERSITE=1",
        "SGLANG_HQ_TOKEN_PRUNE_METHOD=shape_stable_compute_mask",
        "SGLANG_HQ_TOKEN_PRUNE_KEEP_RATIO=0.75",
        "SGLANG_HQ_TOKEN_PRUNE_STEPS=1-2",
    ):
        if needle not in launch:
            errors.append(f"{cid}: missing launch env {needle!r}")
    run_log = (outputs / "run.log").read_text(errors="ignore") if (outputs / "run.log").exists() else ""
    if "techniques=['token_prune']" not in run_log:
        errors.append(f"{cid}: run log does not prove Cosmos3 installed token_prune")

    collection = load_json(outputs / "collection.json")
    baseline = load_json(baseline_dir / "outputs/collection.json")
    timing = collection.get("timing") or {}
    baseline_timing = baseline.get("timing") or {}
    for key in ("total_s", "denoise_s"):
        if not isinstance(timing.get(key), (int, float)):
            errors.append(f"{cid}: missing collected timing {key}")
        if not isinstance(baseline_timing.get(key), (int, float)):
            errors.append(f"{cid}: missing matched short baseline timing {key}")
    if isinstance(timing.get("denoise_s"), (int, float)) and isinstance(
        baseline_timing.get("denoise_s"), (int, float)
    ):
        if timing["denoise_s"] <= baseline_timing["denoise_s"]:
            errors.append(
                f"{cid}: expected current short ToMeSD random2D denoise to remain slower "
                f"than baseline, got {timing['denoise_s']!r} <= {baseline_timing['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "available":
        errors.append(
            f"{cid}: short diagnostic quality should be available, got {quality.get('status')!r}"
        )
    if quality.get("promotion_blockers"):
        errors.append(f"{cid}: expected no short quality blockers, got {quality.get('promotion_blockers')!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    elif lpips["mean"] > 0.05:
        errors.append(f"{cid}: expected low LPIPS drift, got mean={lpips['mean']!r}")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "pass":
        errors.append(f"{cid}: expected Gemini pass, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    frame_count = len(list(frames.glob("*.png"))) if frames.exists() else 0
    if frame_count < 16:
        errors.append(f"{cid}: expected extracted diagnostic frames, got {frame_count}")
    return errors


def validate_cat_public_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "cluster_representative_update":
        return [f"{cid}: unexpected CAT selector short GPU diagnostic validation target"]

    run_dir = ROOT / CAT_PUBLIC_SELECTOR_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / CAT_PUBLIC_SELECTOR_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in ("out.mp4", "benchmark.json", "quality.json", "collection.json"):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3453096":
        errors.append(
            f"{cid}: expected slurm_job_id='3453096', got {metadata.get('slurm_job_id')!r}"
        )
    params = ((metadata.get("candidate_dry_run") or {}).get("runtime_config") or {}).get("params") or {}
    if params.get("method") != "cat_convergence_stale_cpp":
        errors.append(f"{cid}: expected CAT selector method in metadata, got {params.get('method')!r}")
    if params.get("keep_ratio") != 0.3:
        errors.append(f"{cid}: expected CAT selector keep_ratio 0.3, got {params.get('keep_ratio')!r}")

    launch = (run_dir / "launch.sh").read_text(errors="ignore") if (run_dir / "launch.sh").exists() else ""
    for needle in (
        "SGLANG_HQ_TOKEN_PRUNE_METHOD=cat_convergence_stale_cpp",
        "SGLANG_HQ_TOKEN_PRUNE_KEEP_RATIO=0.3",
        "SGLANG_HQ_TOKEN_PRUNE_STEPS=1-2",
    ):
        if needle not in launch:
            errors.append(f"{cid}: missing launch env {needle!r}")
    run_log = (outputs / "run.log").read_text(errors="ignore") if (outputs / "run.log").exists() else ""
    if "techniques=['token_prune']" not in run_log:
        errors.append(f"{cid}: run log does not prove Cosmos3 installed token_prune")

    collection = load_json(outputs / "collection.json")
    baseline = load_json(baseline_dir / "outputs/benchmark.json")
    timing = collection.get("timing") or {}
    for key in ("total_s", "denoise_s"):
        if not isinstance(timing.get(key), (int, float)):
            errors.append(f"{cid}: missing collected timing {key}")
        if not isinstance(baseline.get(key), (int, float)):
            errors.append(f"{cid}: missing matched short baseline benchmark {key}")
    if isinstance(timing.get("denoise_s"), (int, float)) and isinstance(
        baseline.get("denoise_s"), (int, float)
    ):
        if timing["denoise_s"] <= baseline["denoise_s"]:
            errors.append(
                f"{cid}: expected current short CAT denoise to remain slower "
                f"than baseline, got {timing['denoise_s']!r} <= {baseline['denoise_s']!r}"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "available":
        errors.append(
            f"{cid}: short diagnostic quality should be available, got {quality.get('status')!r}"
        )
    if quality.get("promotion_blockers"):
        errors.append(f"{cid}: expected no short quality blockers, got {quality.get('promotion_blockers')!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "pass":
        errors.append(f"{cid}: expected Gemini pass, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    if not frames.exists() or len(list(frames.glob("*.png"))) != 16:
        count = len(list(frames.glob("*.png"))) if frames.exists() else 0
        errors.append(f"{cid}: expected 16 extracted diagnostic frames, got {count}")
    return errors


def validate_teacache_public_short_gpu_diagnostic(cid: str) -> list[str]:
    if cid != "teacache_signal_reuse":
        return [f"{cid}: unexpected TeaCache short GPU diagnostic validation target"]

    run_dir = ROOT / TEACACHE_PUBLIC_SHORT_GPU_DIAGNOSTIC
    baseline_dir = ROOT / TEACACHE_PUBLIC_SHORT_GPU_BASELINE
    outputs = run_dir / "outputs"
    errors: list[str] = []
    for name in ("out.mp4", "benchmark.json", "quality.json", "collection.json"):
        path = outputs / name
        if not path.exists():
            errors.append(f"{cid}: missing short GPU diagnostic artifact {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"{cid}: empty short GPU diagnostic artifact {path.relative_to(ROOT)}")

    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("candidate_id") != cid:
        errors.append(
            f"{cid}: diagnostic metadata candidate_id={metadata.get('candidate_id')!r}"
        )
    if metadata.get("slurm_job_id") != "3451216":
        errors.append(
            f"{cid}: expected slurm_job_id='3451216', got {metadata.get('slurm_job_id')!r}"
        )

    stats_files = sorted(outputs.glob("teacache_stats.rank*.json"))
    if len(stats_files) != 4:
        errors.append(f"{cid}: expected 4 rank-local TeaCache stats files, got {len(stats_files)}")
    for path in stats_files:
        stats = load_json(path)
        if stats.get("calls") != 8 or stats.get("reuse") != 2:
            errors.append(
                f"{cid}: expected calls=8 and reuse=2 in {path.name}, got "
                f"calls={stats.get('calls')!r}, reuse={stats.get('reuse')!r}"
            )
        if stats.get("compute") != 6 or stats.get("forced") != 1:
            errors.append(
                f"{cid}: expected compute=6 and forced=1 in {path.name}, got "
                f"compute={stats.get('compute')!r}, forced={stats.get('forced')!r}"
            )
        if not isinstance(stats.get("max_indicator"), (int, float)):
            errors.append(f"{cid}: missing max_indicator in {path.name}")

    benchmark = load_json(outputs / "benchmark.json")
    baseline = load_json(baseline_dir / "outputs/benchmark.json")

    def stage_seconds(payload: dict[str, Any], name: str) -> float | None:
        if name == "Cosmos3DenoisingStage" and isinstance(payload.get("denoise_s"), (int, float)):
            return float(payload["denoise_s"])
        stage_map = payload.get("stage_seconds")
        if isinstance(stage_map, dict) and isinstance(stage_map.get(name), (int, float)):
            return float(stage_map[name])
        for stage in payload.get("steps") or []:
            if stage.get("name") == name and isinstance(stage.get("duration_ms"), (int, float)):
                return float(stage["duration_ms"]) / 1000.0
        return None

    cand_denoise = stage_seconds(benchmark, "Cosmos3DenoisingStage")
    base_denoise = stage_seconds(baseline, "Cosmos3DenoisingStage")
    if cand_denoise is None or base_denoise is None:
        errors.append(f"{cid}: missing matched 8-step denoise timings")
    else:
        speedup = base_denoise / cand_denoise
        if speedup <= 1.0:
            errors.append(f"{cid}: expected short TeaCache denoise speedup > 1.0, got {speedup:.4f}x")
        if speedup >= 1.5:
            errors.append(
                f"{cid}: status says speed remains low, but matched denoise speedup is {speedup:.4f}x"
            )

    quality = load_json(outputs / "quality.json")
    if quality.get("status") != "available":
        errors.append(
            f"{cid}: short diagnostic quality should be available, got {quality.get('status')!r}"
        )
    if quality.get("promotion_blockers"):
        errors.append(f"{cid}: expected no short quality blockers, got {quality.get('promotion_blockers')!r}")
    lpips = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    if not isinstance(lpips.get("mean"), (int, float)):
        errors.append(f"{cid}: missing matched-short LPIPS mean")
    gemini = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result") or {}
    if gemini.get("overall") != "pass":
        errors.append(f"{cid}: expected Gemini pass, got {gemini.get('overall')!r}")
    frames = outputs / "frames"
    if not frames.exists() or len(list(frames.glob("*.png"))) != 16:
        count = len(list(frames.glob("*.png"))) if frames.exists() else 0
        errors.append(f"{cid}: expected 16 extracted diagnostic frames, got {count}")
    return errors


def check_semantic_permutation_public_config(
    data: dict[str, Any], path: Path, problems: list[str]
) -> None:
    """Guard the adapter config against drifting from Sparse-VideoGen's Cosmos SAP script."""

    params = data.get("efficiency", {}).get("params", {})
    if not isinstance(params, dict):
        problems.append(f"{path.relative_to(ROOT)}: missing [efficiency.params]")
        return
    for key, expected in SEMANTIC_PERMUTATION_PUBLIC_COSMOS_CONFIG.items():
        actual = params.get(key)
        if isinstance(expected, float):
            try:
                matches = abs(float(actual) - expected) <= 1e-9
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            problems.append(
                f"{path.relative_to(ROOT)}: semantic_permutation {key}={actual!r} "
                f"does not match public Cosmos SAP default {expected!r}"
            )

    code_url = (
        data.get("references", {})
        .get("external", {})
        .get("code", "")
    )
    if "Sparse-VideoGen" not in str(code_url):
        problems.append(
            f"{path.relative_to(ROOT)}: semantic_permutation code reference must cite Sparse-VideoGen"
        )

    generic_impl = (
        data.get("references", {})
        .get("local", {})
        .get("generic_impl", "")
    )
    generic_path = ROOT / str(generic_impl)
    generic_text = generic_path.read_text(errors="ignore") if generic_path.exists() else ""
    policy_text = (ROOT / "efficiency" / "sparse_attention_policies.py").read_text(
        errors="ignore"
    )
    if (
        "sparse_videogen_sap_plan" not in generic_text
        or "class SparseVideoGenSAPPlan" not in policy_text
        or "def sparse_videogen_sap_plan" not in policy_text
        or "def sparse_videogen_identify_dynamic_map" not in policy_text
        or "def sparse_videogen_permutation_indices" not in policy_text
    ):
        problems.append(
            f"{path.relative_to(ROOT)}: semantic_permutation generic implementation "
            "must expose a pure SparseVideoGenSAPPlan instead of only runtime adapter glue"
        )


def check_piecewise_pisa_public_config(
    data: dict[str, Any], path: Path, problems: list[str]
) -> None:
    params = data.get("efficiency", {}).get("params", {})
    if not isinstance(params, dict):
        problems.append(f"{path.relative_to(ROOT)}: missing [efficiency.params]")
        return
    if params.get("route_mode") != "score":
        problems.append(
            f"{path.relative_to(ROOT)}: piecewise_pisa_env route_mode must be public default 'score'"
        )
    if params.get("route_bias") is not False:
        problems.append(
            f"{path.relative_to(ROOT)}: piecewise_pisa_env must set route_bias=false to match public PISA default routing"
        )
    for key in ("allow_qk_mismatch", "allow_gqa"):
        if params.get(key) is not True:
            problems.append(
                f"{path.relative_to(ROOT)}: piecewise_pisa_env must set {key}=true for the Cosmos3 adapter diagnostic"
            )


def check_teacache_public_controller_profile(
    data: dict[str, Any], path: Path, problems: list[str]
) -> None:
    params = data.get("efficiency", {}).get("params", {})
    env = data.get("env", {})
    if not isinstance(params, dict):
        problems.append(f"{path.relative_to(ROOT)}: missing [efficiency.params]")
        return
    probe = load_module(
        ROOT / "scripts/probe_public_teacache_alignment.py",
        "probe_public_teacache_alignment_audit",
    ).probe()
    profile = probe["candidate_manifest_alignment"]
    if not profile["matches_public_cosmos_profile"]:
        problems.append(
            f"{path.relative_to(ROOT)}: TeaCache profile mismatches public Cosmos profile: "
            f"{profile['mismatches']!r}"
        )
    if not probe["core_formula_probe"]["intermediate_core_match"]:
        problems.append(
            f"{path.relative_to(ROOT)}: TeaCache rel-L1/poly controller no longer matches public intermediate decisions"
        )
    adapter_checks = probe["cosmos3_adapter_alignment"]["checks"]
    missing = [key for key, ok in adapter_checks.items() if not ok]
    if missing:
        problems.append(
            f"{path.relative_to(ROOT)}: TeaCache Cosmos3 residual adapter checks failed: {missing!r}"
        )
    coeff_env = str(env.get("SGLANG_HQ_TEACACHE_COEFFICIENTS", ""))
    if not coeff_env:
        problems.append(
            f"{path.relative_to(ROOT)}: missing SGLANG_HQ_TEACACHE_COEFFICIENTS env export"
        )


def check_pab_public_controller_profile(
    data: dict[str, Any], path: Path, problems: list[str]
) -> None:
    cid = str(data.get("id", {}).get("name", ""))
    if cid not in {"attention_broadcast", "block_layer_feature_cache"}:
        return
    params = data.get("efficiency", {}).get("params", {})
    env = data.get("env", {})
    if not isinstance(params, dict):
        problems.append(f"{path.relative_to(ROOT)}: missing [efficiency.params]")
        return
    if params.get("mode") != "pab" or env.get("SGLANG_HQ_PAYLOAD_CACHE_MODE") != "pab":
        problems.append(
            f"{path.relative_to(ROOT)}: public PAB payload-cache candidates must run with mode='pab'"
        )
    probe = load_module(
        ROOT / "scripts/probe_public_pab_alignment.py",
        "probe_public_pab_alignment_audit",
    ).probe()
    alignment = probe["candidate_manifest_alignment"][cid]
    if not alignment.get("matches_public_pab_controller"):
        problems.append(
            f"{path.relative_to(ROOT)}: local PAB controller no longer matches the public VideoSys PAB behavior probe"
        )


def public_equivalence_claim(alignment: Alignment) -> str:
    if alignment.scope in FULL_PUBLIC_EQUIVALENCE_SCOPES:
        return "full_public_original_equivalence"
    return "not_full_public_port"


def public_equivalence_gap(alignment: Alignment) -> str:
    if public_equivalence_claim(alignment) == "full_public_original_equivalence":
        return ""
    if alignment.cosmos3_status == "consumer_missing_on_cosmos3":
        return "runtime_consumer_missing"
    if alignment.cosmos3_status == "cosmos3_pure_algorithm_adapter_missing":
        return "pure_algorithm_adapter_missing"
    if alignment.cosmos3_status == "cosmos3_pure_algorithm_subset_already_baseline":
        return "already_in_cosmos3_baseline_or_ltx2_specific_not_applicable"
    if alignment.cosmos3_status == "cosmos3_te_recipe_fused_adapter_semantics_mismatch":
        return "te_recipe_model_specific_adapter_semantics_mismatch"
    if (
        alignment.cosmos3_status
        in {
            "consumer_wired_online_nvfp4_cutlass_short_gpu_quality_pass_speed_negative",
            "consumer_wired_online_nvfp4_cudnn_short_gpu_quality_pass_speed_negative",
            "consumer_wired_online_nvfp4_profile_selector_short_gpu_quality_pass_speed_neutral",
            "consumer_wired_online_nvfp4_profiled_scope_short_gpu_quality_pass_speed_neutral_static_scope",
            "consumer_wired_online_nvfp4_dense_guard_short4_gpu_quality_pass_speed_low",
        }
    ):
        return "nvfp4_short_quality_pass_speedup_missing"
    if alignment.cosmos3_status.startswith("consumer_wired_online_nvfp4"):
        return "nvfp4_gpu_backend_quality_evidence_missing"
    if alignment.cosmos3_status == "cosmos3_probe_requires_concrete_backend_policy":
        return "probe_requires_concrete_backend_policy"
    if alignment.cosmos3_status == "launcher_ready_route_label_only":
        return "public_algorithm_not_implemented"
    if alignment.cosmos3_status == "launcher_ready_but_attention_scope_consumer_missing":
        return "attention_payload_cache_missing"
    if alignment.cosmos3_status == "launcher_ready_but_block_payload_consumer_missing":
        return "block_payload_cache_missing"
    if alignment.cosmos3_status == "consumer_wired_dry_run_only":
        return "gpu_and_public_behavior_evidence_missing"
    if alignment.cosmos3_status == "consumer_wired_runtime_probe":
        return "gpu_and_public_behavior_evidence_missing"
    if alignment.cosmos3_status == "consumer_wired_smoke_run_complete":
        return "family_specific_public_equivalence_checker_missing"
    if alignment.cosmos3_status == "consumer_wired_public_svg_sap_core_runtime_mismatch":
        return "public_svg_sap_core_runtime_assumption_mismatch"
    if alignment.cosmos3_status == "consumer_wired_public_checker_mismatch":
        if alignment.scope in LOCAL_BASELINE_SCOPES:
            return "local_pure_baseline_no_public_original_claim"
        if alignment.scope in KWL_GENERIC_POLICY_SCOPES:
            return "kwl_generic_policy_not_public_kernel_port"
        if alignment.scope in RUNTIME_ADAPTER_OR_PROBE_SCOPES:
            return "runtime_adapter_or_probe_not_public_original"
        return "public_behavior_checker_mismatch"
    if alignment.cosmos3_status == "consumer_wired_public_checker_gpu_evidence_stale":
        return "public_checker_mismatch_gpu_evidence_missing"
    if alignment.cosmos3_status == SPARGE_CORE_SHORT_GPU_STATUS:
        return "public_core_short_quality_failed_speed_negative"
    if alignment.cosmos3_status == SPARGE_PROXY_CORE_SHORT_GPU_STATUS:
        return "public_core_short_quality_failed_speed_negative"
    if alignment.cosmos3_status == SPARGE_QK_CORE_SHORT_GPU_STATUS:
        return "public_core_short_quality_failed_speed_negative"
    if alignment.cosmos3_status == SPARGE_QK_CORE_GPU_MISSING_STATUS:
        return "public_core_gpu_evidence_missing"
    if alignment.cosmos3_status == SPARGE_HEADWISE_CORE_GPU_MISSING_STATUS:
        return "public_core_gpu_evidence_missing"
    if alignment.cosmos3_status == SPARGE_HEADWISE_CORE_SHORT_GPU_STATUS:
        return "public_core_short_quality_failed_speed_negative"
    if alignment.cosmos3_status == MINFERENCE_DYNAMIC_CORE_GPU_MISSING_STATUS:
        return "public_core_gpu_evidence_missing"
    if alignment.cosmos3_status == MINFERENCE_DYNAMIC_CORE_SHORT_GPU_STATUS:
        return "public_core_short_quality_failed_speed_negative"
    if alignment.cosmos3_status == SVG_TEMPORAL_ANCHOR_CORE_GPU_MISSING_STATUS:
        return "public_core_gpu_evidence_missing"
    if alignment.cosmos3_status == SVG_TEMPORAL_ANCHOR_CORE_SHORT_GPU_STATUS:
        return "public_core_short_quality_failed_speed_negative"
    if alignment.cosmos3_status == SVG_SAMPLE_MSE_SHORT_GPU_STATUS:
        return "public_core_short_quality_failed_speed_negative"
    if alignment.cosmos3_status == CAT_PUBLIC_SELECTOR_STATUS:
        return "public_cat_selector_gpu_evidence_missing"
    if alignment.cosmos3_status == CAT_PUBLIC_SELECTOR_SHORT_GPU_STATUS:
        return "public_cat_selector_short_quality_pass_speedup_missing"
    if alignment.cosmos3_status == PAB_PUBLIC_CONTROLLER_STATUS:
        return "public_controller_short_gpu_quality_failed"
    if alignment.cosmos3_status == TEACACHE_PUBLIC_CONTROLLER_STATUS:
        return "public_controller_residual_adapter_short_quality_pass_speedup_missing"
    if alignment.cosmos3_status == TOME_PUBLIC_SHORT_GPU_STATUS:
        return "public_core_match_short_quality_pass_speedup_missing"
    if alignment.cosmos3_status == TOMESD_RANDOM2D_CORE_GPU_MISSING_STATUS:
        return "public_core_gpu_evidence_missing"
    if alignment.cosmos3_status == TOMESD_RANDOM2D_CORE_SHORT_GPU_STATUS:
        return "public_core_match_short_quality_pass_speedup_missing"
    if alignment.cosmos3_status == PIECEWISE_PISA_SHORT_GPU_STATUS:
        return "public_route_short_quality_failed_speedup_present"
    if alignment.cosmos3_status == SPARSE_POLICY_SHORT_GPU_STATUS:
        return "public_checker_mismatch_official_quality_evidence_missing"
    if alignment.cosmos3_status == "consumer_wired_dependency_missing":
        return "runtime_dependency_missing"
    return "public_equivalence_not_proven"


def next_required_proof(alignment: Alignment) -> str:
    gap = public_equivalence_gap(alignment)
    if gap == "runtime_consumer_missing":
        return (
            "wire the Cosmos3 runtime consumer, prove env/config is consumed in a "
            "GPU smoke run, then add a family-specific public behavior checker"
        )
    if gap == "pure_algorithm_adapter_missing":
        return (
            "implement the pure algorithm as a small Cosmos3 adapter with explicit "
            "fallback, then prove the candidate changes execution in a GPU smoke run"
        )
    if gap == "already_in_cosmos3_baseline_or_ltx2_specific_not_applicable":
        return (
            "do not submit a candidate GPU job as optimization evidence; either keep "
            "the pure baseline code, or define a new Cosmos3-specific algorithmic delta"
        )
    if gap == "te_recipe_model_specific_adapter_semantics_mismatch":
        return (
            "keep the pure ModelOpt/CUTLASS FP4 linear consumer, and only add a "
            "Cosmos3 TE fused-epilogue adapter if it preserves Cosmos3's "
            "bias-free SwiGLU semantics; the LTX2 GELU/bias-gate fused flags "
            "are disabled in the manifest and must remain out of the candidate "
            "claim until such an adapter exists; generic TE recipe env is "
            "already separated from explicit LTX2 adapter env; repair the "
            "current TransformerEngine/CUDNN runtime dependency before any TE "
            "adapter GPU validation"
        )
    if gap == "nvfp4_gpu_backend_quality_evidence_missing":
        return (
            "run backend-specific NVFP4 GPU validation, resolve any FlashInfer "
            "cubin/header backend issue, then collect matched quality and speed evidence"
        )
    if gap == "nvfp4_short_quality_pass_speedup_missing":
        return (
            "do not promote this backend yet; profile the online quantization/GEMM "
            "path or try another backend, then rerun matched quality and speed"
        )
    if gap == "probe_requires_concrete_backend_policy":
        return (
            "choose a concrete Cosmos3 backend policy with dense fallback and quality "
            "criteria before treating GPU runtime as candidate evidence"
        )
    if gap == "public_algorithm_not_implemented":
        return (
            "implement the named public routing/mask policy, prove dense fallback "
            "and quality, then compare behavior against the public reference"
        )
    if gap == "attention_payload_cache_missing":
        return (
            "cache and reuse the intended attention payloads, prove saved compute "
            "and quality, then compare against the PAB/VideoSys behavior boundary"
        )
    if gap == "block_payload_cache_missing":
        return (
            "cache and reuse the intended block/layer payloads, prove saved compute "
            "and quality, then compare against the public block-cache behavior boundary"
        )
    if gap == "gpu_and_public_behavior_evidence_missing":
        return (
            "run a meaningful Cosmos3 GPU smoke test with metrics/quality and add "
            "a family-specific public behavior checker"
        )
    if gap == "runtime_dependency_missing":
        return (
            "install the required runtime dependency, run a meaningful Cosmos3 GPU "
            "smoke test, then compare behavior against the public reference"
        )
    if gap == "family_specific_public_equivalence_checker_missing":
        return (
            "add a family-specific checker that compares the local behavior and "
            "configuration against the cited public reference"
        )
    if gap == "local_pure_baseline_no_public_original_claim":
        return (
            "keep this as a model-agnostic local baseline or replace it with a "
            "separate public-original algorithm; GPU evidence for this row is "
            "baseline/usefulness evidence, not public-equivalence evidence"
        )
    if gap == "runtime_adapter_or_probe_not_public_original":
        return (
            "keep this as runtime-adapter/probe evidence, or implement a real "
            "public backend/runtime algorithm; do not promote model-specific "
            "glue as the model-agnostic algorithm"
        )
    if gap == "kwl_generic_policy_not_public_kernel_port":
        return (
            "keep the generic KWL backend or compile/capture policy, but do "
            "not claim FlashAttention, CUTLASS, TransformerEngine, or standalone "
            "CUDA-graph equivalence without a real public-kernel/runtime port "
            "and matched quality/performance evidence"
        )
    if gap == "public_svg_sap_core_runtime_assumption_mismatch":
        return (
            "keep the pure Sparse-VideoGen SAP dynamic-map/permutation core, "
            "but do not claim full public runtime equivalence until the "
            "Cosmos3 GQA, text-prefix, FlashInfer-varlen, metadata, and CFG "
            "adapter differences are either eliminated or justified with "
            "official-quality GPU validation"
        )
    if gap == "public_behavior_checker_mismatch":
        return (
            "either preserve the current baseline label, or change the profile/runtime "
            "to match the public checker and rerun GPU quality/performance validation"
        )
    if gap == "public_checker_mismatch_gpu_evidence_missing":
        return (
            "rerun Cosmos3 GPU validation with the current public-aligned "
            "algorithm, then collect performance, quality, and blocker evidence"
        )
    if gap == "public_core_gpu_evidence_missing":
        if alignment.cosmos3_status == MINFERENCE_DYNAMIC_CORE_GPU_MISSING_STATUS:
            return (
                "run Cosmos3 GPU validation with the MInference-style dynamic "
                "pattern-bank core, then collect selected-pattern, fallback, "
                "quality, and matched-speed evidence before promotion"
            )
        if alignment.cosmos3_status == SVG_TEMPORAL_ANCHOR_CORE_GPU_MISSING_STATUS:
            return (
                "run Cosmos3 GPU validation with the Sparse-VideoGen first-frame "
                "temporal-window core, then collect selected-mode, fallback, "
                "quality, and matched-speed evidence before promotion"
            )
        return (
            "run Cosmos3 GPU validation with the current public-core algorithm, "
            "then collect matched performance, quality, fallback, and blocker evidence"
        )
    if gap == "public_core_short_quality_failed_speed_negative":
        if alignment.cosmos3_status == SVG_SAMPLE_MSE_SHORT_GPU_STATUS:
            return (
                "keep the pure Sparse-VideoGen sample-MSE selector core, but reject "
                "or retune the Cosmos3 piecewise_attn adapter; do not promote "
                "without matched quality passing and useful speedup"
            )
        if alignment.cosmos3_status == SVG_TEMPORAL_ANCHOR_CORE_SHORT_GPU_STATUS:
            return (
                "keep the pure Sparse-VideoGen first-frame temporal-window core, "
                "but reject or retune the Cosmos3 boolean-mask adapter; do not "
                "promote without matched quality passing and useful speedup"
            )
        if alignment.cosmos3_status == SPARGE_PROXY_CORE_SHORT_GPU_STATUS:
            return (
                "keep the pure SpargeAttn fused-quant proxy block-map core, "
                "but reject or retune the Cosmos3 boolean-mask adapter; do "
                "not promote without matched quality passing and useful speedup"
            )
        if alignment.cosmos3_status == SPARGE_QK_CORE_SHORT_GPU_STATUS:
            return (
                "keep the pure SpargeAttn Q/K mean-similarity block-map core, "
                "but reject or retune the Cosmos3 boolean-mask adapter and "
                "thresholds; do not promote without matched quality passing and "
                "useful speedup"
            )
        if alignment.cosmos3_status == SPARGE_HEADWISE_CORE_SHORT_GPU_STATUS:
            return (
                "keep the pure SpargeAttn per-head top-k mean-similarity "
                "block-map core, but reject or retune the budget proposal and "
                "Cosmos3 boolean-mask adapter; do not promote without matched "
                "quality passing and useful speedup"
            )
        if alignment.cosmos3_status == MINFERENCE_DYNAMIC_CORE_SHORT_GPU_STATUS:
            return (
                "keep the pure MInference-style dynamic pattern-bank core, "
                "but reject or retune the pattern selection and Cosmos3 "
                "boolean-mask adapter; do not promote without matched quality "
                "passing and useful speedup"
            )
        return (
            "keep the pure SpargeAttn mean-similarity block-map core, but reject "
            "or retune the Cosmos3 mask-search/reuse adapter; do not promote "
            "without matched quality passing and useful speedup"
        )
    if gap == "public_cat_selector_gpu_evidence_missing":
        return (
            "keep the public CAT selector-core claim, then rerun Cosmos3 GPU "
            "validation and collect quality/performance evidence; do not claim "
            "full public CAT runtime equivalence without proj_out, joint-attention "
            "KV-cache, and graph-pooling hooks"
        )
    if gap == "public_cat_selector_short_quality_pass_speedup_missing":
        return (
            "keep the public CAT selector-core claim, then tune or reject it "
            "based on matched-baseline speed; do not claim full public CAT "
            "runtime equivalence without proj_out, joint-attention KV-cache, "
            "and graph-pooling hooks"
        )
    if gap == "public_controller_short_gpu_quality_failed":
        return (
            "keep the public PAB controller/adapter claim, but tune or reject it "
            "based on matched-baseline quality and speed; do not promote without "
            "quality passing and useful speedup"
        )
    if gap == "public_controller_residual_adapter_short_quality_pass_speedup_missing":
        return (
            "keep the public TeaCache controller/residual-adapter claim, then tune "
            "or reject it based on matched-baseline speed; do not promote without "
            "useful speedup and official-shape quality evidence"
        )
    if gap == "public_core_match_short_quality_pass_speedup_missing":
        if alignment.cosmos3_status == TOMESD_RANDOM2D_CORE_SHORT_GPU_STATUS:
            return (
                "keep the public ToMeSD random-2D merge/unmerge core claim, "
                "then tune or reject it based on matched-baseline speed; do "
                "not promote without useful speedup and official-shape quality "
                "evidence"
            )
        return (
            "keep the public ToMe core claim, then tune or reject it based on "
            "matched-baseline speed; do not promote without useful speedup and "
            "official-shape quality evidence"
        )
    if alignment.cosmos3_status == PIECEWISE_PISA_SHORT_GPU_STATUS:
        return (
            "keep the public route-boundary claim, tune or relax the Cosmos3 sparse "
            "path only with explicit quality evidence, then rerun matched-baseline "
            "or official-quality GPU validation with fallback stats"
        )
    if gap == "public_checker_mismatch_official_quality_evidence_missing":
        return (
            "keep the local pure policy label or rewrite it to public-original "
            "semantics, then run matched-baseline or official-quality Cosmos3 "
            "GPU validation with performance, quality, fallback, and blocker evidence"
        )
    if gap == "":
        return "none"
    return "define and pass a public-equivalence checker before claiming equivalence"


def algorithm_boundary(alignment: Alignment) -> str:
    gap = public_equivalence_gap(alignment)
    if gap in {
        "public_core_short_quality_failed_speed_negative",
        "public_core_match_short_quality_pass_speedup_missing",
        "public_cat_selector_short_quality_pass_speedup_missing",
        "public_route_short_quality_failed_speedup_present",
        "public_svg_sap_core_runtime_assumption_mismatch",
    }:
        return "public_core_preserved_consumer_wired"
    if gap in {
        "public_controller_short_gpu_quality_failed",
        "public_controller_residual_adapter_short_quality_pass_speedup_missing",
    }:
        return "public_controller_preserved_consumer_wired"
    if gap == "nvfp4_short_quality_pass_speedup_missing":
        return "generic_fp4_consumer_preserved_consumer_wired"
    if gap == "te_recipe_model_specific_adapter_semantics_mismatch":
        return "generic_fp4_consumer_preserved_te_fused_adapter_not_claimed"
    if gap == "local_pure_baseline_no_public_original_claim":
        return "local_pure_baseline_preserved_not_public_original"
    if gap == "runtime_adapter_or_probe_not_public_original":
        return "runtime_adapter_or_probe_only_not_algorithm"
    if gap == "kwl_generic_policy_not_public_kernel_port":
        return "generic_kwl_policy_preserved_not_public_kernel_port"
    if gap == "already_in_cosmos3_baseline_or_ltx2_specific_not_applicable":
        return "cosmos3_baseline_or_ltx2_replay_no_new_algorithm"
    if gap in {"runtime_consumer_missing", "pure_algorithm_adapter_missing"}:
        return "pure_algorithm_or_adapter_incomplete"
    if gap in {
        "public_core_gpu_evidence_missing",
        "public_cat_selector_gpu_evidence_missing",
        "gpu_and_public_behavior_evidence_missing",
        "public_checker_mismatch_gpu_evidence_missing",
        "public_checker_mismatch_official_quality_evidence_missing",
    }:
        return "pure_algorithm_preserved_consumer_wired_gpu_evidence_missing"
    if gap == "":
        return "full_public_original_equivalence_claimed"
    return "needs_manual_algorithm_boundary_review"


def true_blocker(alignment: Alignment) -> str:
    gap = public_equivalence_gap(alignment)
    if gap in {"runtime_consumer_missing", "attention_payload_cache_missing", "block_payload_cache_missing"}:
        return "runtime_consumer_missing"
    if gap in {"pure_algorithm_adapter_missing", "public_algorithm_not_implemented"}:
        return "pure_algorithm_missing_or_mismatched"
    if gap == "already_in_cosmos3_baseline_or_ltx2_specific_not_applicable":
        return "no_new_cosmos3_algorithm_delta_or_ltx2_only_replay"
    if gap == "te_recipe_model_specific_adapter_semantics_mismatch":
        return "cosmos3_te_fused_adapter_semantics_and_dependency"
    if gap == "nvfp4_short_quality_pass_speedup_missing":
        return "gpu_speed_backend_not_useful_after_quality_pass"
    if gap == "runtime_adapter_or_probe_not_public_original":
        return "model_specific_glue_or_probe_not_algorithm"
    if gap == "kwl_generic_policy_not_public_kernel_port":
        return "not_public_kernel_port_or_quality_speed_missing"
    if gap == "public_svg_sap_core_runtime_assumption_mismatch":
        return "model_specific_runtime_assumption_mismatch"
    if gap == "local_pure_baseline_no_public_original_claim":
        return "local_baseline_not_public_original_algorithm"
    if gap in {
        "public_core_short_quality_failed_speed_negative",
        "public_controller_short_gpu_quality_failed",
    }:
        return "gpu_quality_and_speed_failed_after_consumer_wired"
    if gap == "public_route_short_quality_failed_speedup_present":
        return "gpu_quality_failed_after_consumer_wired"
    if gap in {
        "public_core_match_short_quality_pass_speedup_missing",
        "public_cat_selector_short_quality_pass_speedup_missing",
        "public_controller_residual_adapter_short_quality_pass_speedup_missing",
    }:
        return "gpu_speedup_missing_after_quality_pass"
    if gap in {
        "public_core_gpu_evidence_missing",
        "public_cat_selector_gpu_evidence_missing",
        "gpu_and_public_behavior_evidence_missing",
        "public_checker_mismatch_gpu_evidence_missing",
        "public_checker_mismatch_official_quality_evidence_missing",
        "nvfp4_gpu_backend_quality_evidence_missing",
    }:
        return "gpu_or_public_behavior_evidence_missing"
    if gap == "runtime_dependency_missing":
        return "runtime_dependency_missing"
    if gap == "":
        return "none"
    return "public_equivalence_not_proven"


def model_specific_glue_policy(alignment: Alignment) -> str:
    boundary = algorithm_boundary(alignment)
    if boundary in {
        "runtime_adapter_or_probe_only_not_algorithm",
        "cosmos3_baseline_or_ltx2_replay_no_new_algorithm",
        "generic_fp4_consumer_preserved_te_fused_adapter_not_claimed",
    }:
        return "delete_or_reclassify_after_validation_unless_it_is_the_explicit_consumer"
    if boundary.endswith("_consumer_wired") or "consumer_wired" in boundary:
        return "keep_only_minimal_cosmos3_consumer_not_as_algorithm"
    return "keep_pure_algorithm_only"


def effective_alignment(
    cid: str, alignment: Alignment, launcher_blocked: set[str]
) -> Alignment:
    if cid != "semantic_permutation":
        return alignment
    if cid in launcher_blocked:
        return Alignment(
            alignment.scope,
            alignment.public_role,
            alignment.local_claim,
            "consumer_wired_dependency_missing",
            "Launcher readiness check cannot import the required Sparse-VideoGen/FlashInfer/cuVS runtime dependencies.",
        )
    return alignment


def historical_run_artifacts_present() -> bool:
    runs_dir = ROOT / "runs"
    return runs_dir.exists() and any(
        path.is_dir() and path.name != "__pycache__" for path in runs_dir.iterdir()
    )


def audit() -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    problems: list[str] = []
    effective: dict[str, Alignment] = {}
    paths = sorted((ROOT / "candidates").glob("*/*.toml"))
    seen: set[str] = set()
    runtime_text = cosmos3_runtime_text()
    try:
        launcher_blocked = launcher_cosmos3_blocklist()
    except Exception as exc:
        launcher_blocked = set()
        problems.append(f"could not read launcher Cosmos3 GPU blocklist: {exc}")
    validate_historical_gpu = historical_run_artifacts_present()

    for path in paths:
        data = load_toml(path)
        cid = manifest_id(data)
        if cid not in ALIGNMENT:
            problems.append(f"{path.relative_to(ROOT)}: missing public-reference alignment entry")
            continue
        seen.add(cid)
        alignment = effective_alignment(cid, ALIGNMENT[cid], launcher_blocked)
        effective[cid] = alignment

        refs = data.get("references", {})
        external = refs.get("external", {}) if isinstance(refs, dict) else {}
        urls = extract_urls(str(external.get("paper", ""))) + extract_urls(
            str(external.get("code", ""))
        )
        if not urls:
            problems.append(f"{path.relative_to(ROOT)}: external references have no URL")

        text = candidate_text(data)
        for phrase in OVERCLAIM_PHRASES:
            if contains_unnegated_phrase(text, phrase):
                problems.append(
                    f"{path.relative_to(ROOT)}: overclaims public equivalence with phrase {phrase!r}"
                )

        purpose = str(data.get("purpose", ""))
        gap = public_equivalence_gap(alignment)
        if alignment.cosmos3_status == "consumer_missing_on_cosmos3" and purpose == "delivery":
            problems.append(
                f"{path.relative_to(ROOT)}: purpose=delivery but Cosmos3 consumer is missing"
            )
        blocker = true_blocker(alignment)
        if purpose == "delivery" and blocker != "none":
            problems.append(
                f"{path.relative_to(ROOT)}: purpose=delivery but current true_blocker is {blocker!r}"
            )
        if (
            gap == "local_pure_baseline_no_public_original_claim"
            and purpose not in {"evidence", "blocker_probe"}
        ):
            problems.append(
                f"{path.relative_to(ROOT)}: local baseline without a public-original "
                f"algorithm claim must use purpose=evidence or blocker_probe, got {purpose!r}"
            )
        if gap == "local_pure_baseline_no_public_original_claim":
            problems.extend(
                local_baseline_reference_note_problems(
                    path, str(external.get("notes", ""))
                )
            )
        if gap == "runtime_adapter_or_probe_not_public_original":
            if purpose not in {"evidence", "blocker_probe"}:
                problems.append(
                    f"{path.relative_to(ROOT)}: runtime adapter/probe row "
                    f"without a public-original algorithm claim must use "
                    f"purpose=evidence or blocker_probe, got {purpose!r}"
                )
            problems.extend(
                adapter_probe_reference_note_problems(
                    path, str(external.get("notes", ""))
                )
            )
        if gap == "kwl_generic_policy_not_public_kernel_port":
            if purpose not in {"evidence", "blocker_probe"}:
                problems.append(
                    f"{path.relative_to(ROOT)}: generic KWL policy row without "
                    f"a public kernel/runtime port must use purpose=evidence "
                    f"or blocker_probe, got {purpose!r}"
                )
        if gap == "public_svg_sap_core_runtime_assumption_mismatch":
            if purpose not in {"evidence", "blocker_probe"}:
                problems.append(
                    f"{path.relative_to(ROOT)}: Sparse-VideoGen SAP row with "
                    f"public core but model-specific runtime assumptions must "
                    f"use purpose=evidence or blocker_probe, got {purpose!r}"
                )
        if gap == "te_recipe_model_specific_adapter_semantics_mismatch":
            if purpose not in {"evidence", "blocker_probe"}:
                problems.append(
                    f"{path.relative_to(ROOT)}: TE recipe row with model-specific "
                    f"fused-adapter semantics mismatch must use purpose=evidence "
                    f"or blocker_probe, got {purpose!r}"
                )
        if gap in {
            "public_core_short_quality_failed_speed_negative",
            "public_route_short_quality_failed_speedup_present",
            "public_controller_short_gpu_quality_failed",
        }:
            if purpose not in {"evidence", "blocker_probe"}:
                problems.append(
                    f"{path.relative_to(ROOT)}: quality-failed GPU evidence row "
                    f"must use purpose=evidence or blocker_probe, got {purpose!r}"
                )
        if gap == "already_in_cosmos3_baseline_or_ltx2_specific_not_applicable":
            if purpose not in {"blocker_probe", "evidence"}:
                problems.append(
                    f"{path.relative_to(ROOT)}: Cosmos3-baseline/LTX2-only row "
                    f"must use purpose=blocker_probe or evidence, got {purpose!r}"
                )
            problems.extend(cosmos3_baseline_or_ltx2_manifest_problems(path, data))
            problems.extend(
                cosmos3_baseline_or_ltx2_note_problems(
                    path, str(external.get("notes", ""))
                )
            )
        problems.extend(te_recipe_manifest_boundary_problems(path, data))
        if alignment.cosmos3_status == "consumer_missing_on_cosmos3" and cosmos3_consumes_dimension(
            runtime_text, path.parent.name
        ):
            problems.append(
                f"{path.relative_to(ROOT)}: alignment says consumer_missing_on_cosmos3, "
                f"but Cosmos3 runtime appears to consume {path.parent.name} env/config"
            )
        if alignment.cosmos3_status.endswith("route_label_only") and purpose == "delivery":
            problems.append(
                f"{path.relative_to(ROOT)}: purpose=delivery but candidate is route-label-only"
            )
        if validate_historical_gpu:
            if alignment.cosmos3_status == SPARSE_POLICY_SHORT_GPU_STATUS:
                problems.extend(validate_sparse_policy_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == SVG_SAMPLE_MSE_SHORT_GPU_STATUS:
                problems.extend(validate_svg_sample_mse_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == SVG_TEMPORAL_ANCHOR_CORE_SHORT_GPU_STATUS:
                problems.extend(validate_svg_temporal_anchor_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == SPARGE_CORE_SHORT_GPU_STATUS:
                problems.extend(validate_sparge_core_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == SPARGE_PROXY_CORE_SHORT_GPU_STATUS:
                problems.extend(validate_sparge_proxy_core_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == SPARGE_QK_CORE_SHORT_GPU_STATUS:
                problems.extend(validate_sparge_qk_core_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == SPARGE_HEADWISE_CORE_SHORT_GPU_STATUS:
                problems.extend(validate_sparge_headwise_core_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == MINFERENCE_DYNAMIC_CORE_SHORT_GPU_STATUS:
                problems.extend(validate_minference_dynamic_core_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == PIECEWISE_PISA_SHORT_GPU_STATUS:
                problems.extend(validate_piecewise_pisa_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == TOME_PUBLIC_SHORT_GPU_STATUS:
                problems.extend(validate_tome_public_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == TOMESD_RANDOM2D_CORE_SHORT_GPU_STATUS:
                problems.extend(validate_tomesd_random2d_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == CAT_PUBLIC_SELECTOR_SHORT_GPU_STATUS:
                problems.extend(validate_cat_public_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == TEACACHE_PUBLIC_CONTROLLER_STATUS:
                problems.extend(validate_teacache_public_short_gpu_diagnostic(cid))
            if alignment.cosmos3_status == PAB_PUBLIC_CONTROLLER_STATUS:
                problems.extend(validate_pab_public_short_gpu_diagnostic(cid))
        if cid == "piecewise_pisa_env":
            check_piecewise_pisa_public_config(data, path, problems)
        if cid == "teacache_signal_reuse":
            check_teacache_public_controller_profile(data, path, problems)
        if cid in {"attention_broadcast", "block_layer_feature_cache"}:
            check_pab_public_controller_profile(data, path, problems)
        if cid == "semantic_permutation":
            check_semantic_permutation_public_config(data, path, problems)
        equivalence_claim = public_equivalence_claim(alignment)
        if equivalence_claim == "full_public_original_equivalence":
            problems.append(
                f"{path.relative_to(ROOT)}: claims full public-original equivalence, "
                "but this audit has no family-specific public diff/behavior checker"
            )

        rows.append(
            {
                "candidate": cid,
                "dimension": path.parent.name,
                "purpose": purpose,
                "scope": alignment.scope,
                "equivalence_claim": equivalence_claim,
                "public_equivalence_gap": public_equivalence_gap(alignment),
                "algorithm_boundary": algorithm_boundary(alignment),
                "true_blocker": true_blocker(alignment),
                "model_specific_glue_policy": model_specific_glue_policy(alignment),
                "next_required_proof": next_required_proof(alignment),
                "public_role": alignment.public_role,
                "local_claim": alignment.local_claim,
                "cosmos3_status": alignment.cosmos3_status,
                "residual_risk": alignment.residual_risk,
            }
        )

    missing = sorted(set(ALIGNMENT) - seen)
    if missing:
        problems.append(f"alignment entries without candidate manifests: {missing}")

    expected_blocked = {
        cid
        for cid, alignment in effective.items()
        if alignment.cosmos3_status in COSMOS3_GPU_UNSUPPORTED_STATUSES
    }
    missing_blocks = sorted(expected_blocked - launcher_blocked)
    extra_blocks = sorted(launcher_blocked - expected_blocked)
    if missing_blocks:
        problems.append(
            f"launcher does not block unsupported Cosmos3 GPU candidates: {missing_blocks}"
        )
    if extra_blocks:
        problems.append(
            f"launcher blocks candidates not marked unsupported by public alignment: {extra_blocks}"
        )

    return rows, problems


def write_markdown(path: Path, rows: list[dict[str, str]], problems: list[str]) -> None:
    lines = [
        "# Public Reference Alignment",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "This audit distinguishes public-reference provenance from full public",
        "implementation equivalence. Unless a row explicitly says otherwise, the",
        "candidate is a model-agnostic baseline, local pure policy, runtime",
        "adapter, or probe, not a line-for-line port of the public implementation.",
        "",
        f"Status: {'pass' if not problems else 'fail'}",
        "",
        "## Summary",
        "",
        "- Public references are used as canonical provenance for the method family.",
        "- Local implementations are scoped to the `local_claim` column.",
        "- Source availability is not treated as a blocker; the blocker is whether",
        "  the pure algorithm exists, is consumed by Cosmos3, and has evidence that",
        "  it changes the intended computation.",
        "- The `Algorithm Boundary`, `True Blocker`, and `Glue Policy` columns make",
        "  the durable pure-algorithm claim explicit and keep model-specific",
        "  reproduction wiring out of the algorithm claim.",
        "- Model-specific reproduction glue may be used as temporary test wiring,",
        "  but must not be promoted into the model-agnostic algorithm claim.",
        "- Any Cosmos3 status outside `consumer_wired_*` means a GPU job would not",
        "  yet prove the intended optimization as candidate evidence.",
        "- KWL/NVFP4 rows distinguish pure algorithm gaps from Cosmos3-baseline",
        "  fused pieces, LTX2-only replay flags, and NVFP4 backend/recipe evidence gaps.",
        "- Rows whose Cosmos3 status is not meaningful GPU evidence are cross-checked",
        "  against the launcher blocklist.",
        "- No current row is classified as full public-original equivalence; this",
        "  script fails if such a claim appears without a stronger checker.",
        "- A 0/N full-public count is an end-to-end equivalence guard, not an",
        "  implementation-completion score: rows can preserve a public core,",
        "  controller, or generic policy while still being `not_full_public_port`",
        "  until the full public runtime assumptions and quality/performance proof",
        "  are also established.",
        "",
        "## Candidate Matrix",
        "",
        "| Candidate | Dimension | Purpose | Scope | Equivalence Claim | Public Equivalence Gap | Algorithm Boundary | True Blocker | Glue Policy | Next Required Proof | Public Role | Cosmos3 Status | Local Claim | Residual Risk |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {candidate} | {dimension} | {purpose} | {scope} | {equivalence_claim} | {public_equivalence_gap} | {algorithm_boundary} | {true_blocker} | {model_specific_glue_policy} | {next_required_proof} | {public_role} | {cosmos3_status} | {local_claim} | {residual_risk} |".format(
                **{k: v.replace("|", "\\|") for k, v in row.items()}
            )
        )

    lines.extend(["", "## Problems", ""])
    if problems:
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, help="optional JSON report path")
    parser.add_argument("--markdown-out", type=Path, help="optional Markdown report path")
    args = parser.parse_args()

    rows, problems = audit()
    report = {
        "candidate_count": len(rows),
        "problems": problems,
        "rows": rows,
        "status": "pass" if not problems else "fail",
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        write_markdown(args.markdown_out, rows, problems)

    for row in rows:
        print(
            ("PASS" if not any(row["candidate"] in p for p in problems) else "FAIL")
            + f" {row['dimension']}/{row['candidate']} {row['scope']} {row['cosmos3_status']}"
        )
    print(f"\n=== public reference alignment: {report['status']} ({len(rows)} candidates) ===")
    if problems:
        for problem in problems:
            print(f"  - {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
