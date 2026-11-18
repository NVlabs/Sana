#!/usr/bin/env bash
# LTX-2.3 OPTIMIZED technique env — the shipped full acceleration stack.
#
# Five techniques compose here:
#   1. KWL operator fusion      (algorithm-lossless kernel fusion + compile)
#   2. stage-1 SCSP step-skip cache
#   3. stage-2 PISA piecewise sparse attention
#   4. NVFP4 video FFN          (load-time FP4 quantization)
#   5. stage-2 midpoint token prune
#
# As with the baseline arm every knob carries an explicit value, including the
# ones that stay OFF. The upstream enable_kwl_env() deliberately leaves several
# fusions disabled — same-noise ablations showed fused RMS/AdaLN and fused
# QKNorm+RoPE are NOT lossless for this HQ setup, so they live in the
# experimental set only. Writing the zeros out keeps that decision visible here
# instead of hiding it in an unset variable.
#
# Provenance: expanded from enable_fullopt_env() in the SGLang runtime
# scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh (pinned commit in
# models/ltx23/model.toml). Published speedup: 2.40x (site_docs/pipelines/ltx.md).
# The upstream launcher comment claims 2.47x for the same stack; the two numbers
# have never been reconciled in-repo, so treat 2.40x as the citable figure and
# re-measure before quoting either.

# --- Attention math path -----------------------------------------------------
export SGLANG_LTX2_OFFICIAL_FA4_ATTENTION="${SGLANG_LTX2_OFFICIAL_FA4_ATTENTION:-1}"

# --- 1. KWL operator fusion --------------------------------------------------
# On: proven not to change generated frames for this HQ setup.
export SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN="${SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN:-1}"
export SGLANG_LTX2_SHARE_GUIDANCE_PREFIX="${SGLANG_LTX2_SHARE_GUIDANCE_PREFIX:-1}"
export SGLANG_LTX2_FUSED_QK_ROPE="${SGLANG_LTX2_FUSED_QK_ROPE:-1}"
export SGLANG_LTX2_FUSED_RMS_ADALN="${SGLANG_LTX2_FUSED_RMS_ADALN:-1}"
export SGLANG_LTX2_FUSED_ADALN="${SGLANG_LTX2_FUSED_ADALN:-1}"
export SGLANG_LTX2_FUSED_QKNORM_ROPE="${SGLANG_LTX2_FUSED_QKNORM_ROPE:-1}"
export SGLANG_LTX2_FUSED_DUAL_MODULATE="${SGLANG_LTX2_FUSED_DUAL_MODULATE:-1}"
export SGLANG_LTX2_FUSED_CA_DUAL_MODULATE="${SGLANG_LTX2_FUSED_CA_DUAL_MODULATE:-1}"
export SGLANG_LTX2_FUSED_ADA_VALUES_ALL="${SGLANG_LTX2_FUSED_ADA_VALUES_ALL:-1}"
export SGLANG_LTX2_FUSED_RESIDUAL_GATE="${SGLANG_LTX2_FUSED_RESIDUAL_GATE:-1}"
export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU="${SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU:-1}"
export SGLANG_LTX2_COMPILE_GATE_TO_OUT="${SGLANG_LTX2_COMPILE_GATE_TO_OUT:-1}"
export SGLANG_LTX2_FUSED_AUDIO_QKVG="${SGLANG_LTX2_FUSED_AUDIO_QKVG:-1}"
export SGLANG_ENABLE_FUSED_QKNORM_ROPE="${SGLANG_ENABLE_FUSED_QKNORM_ROPE:-1}"
export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER="${SGLANG_LTX2_COMPILE_TILED_VAE_DECODER:-1}"
export SGLANG_LTX2_VAE_COMPILE_MODE="${SGLANG_LTX2_VAE_COMPILE_MODE:-max-autotune-no-cudagraphs}"
# Off by design — not part of the validated stack. Do not flip these to chase a
# number without re-running the same-noise ablation that retired them.
export SGLANG_LTX2_COMPILE_MARK_STEP_BEGIN=0
export SGLANG_LTX2_COMPILE_PREWARM_PERTURBATION_MASKS=0
export SGLANG_LTX2_FUSED_MODULATE=0
export SGLANG_LTX2_FUSED_QKNORM=0
export SGLANG_LTX2_FUSED_ADA_VALUES=0
export SGLANG_LTX2_FUSED_ADA_DIRECT=0
export SGLANG_LTX2_FUSED_Q_GATE=0
export SGLANG_LTX2_FUSED_QKV=0
export SGLANG_LTX2_FUSED_KV=0
export SGLANG_LTX2_FUSED_GELU_INPLACE=0
export SGLANG_LTX2_COMPILE_GATE_TO_OUT_RESIDUAL=0
export SGLANG_LTX2_COMPILE_A2V_GATE_TO_OUT=0
export SGLANG_LTX2_COMPILE_VAE_DECODER=0

# --- 2. stage-1 SCSP step-skip cache ----------------------------------------
# Replaces TeaCache for LTX-2.3; TeaCache is unused on this pipeline.
export SGLANG_LTX2_STAGE1_CACHE_CORE_ENABLED=1
export SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET="${SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET:-8of15_last_29calls}"
export SGLANG_LTX2_STAGE1_CACHE_CORE_CACHE_DEVICE="${SGLANG_LTX2_STAGE1_CACHE_CORE_CACHE_DEVICE:-default}"

# --- 3. stage-2 PISA piecewise sparse attention ------------------------------
# transformer_2 only; layers 0-1 stay dense.
export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
export SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE=false
export SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS=0
export SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY:-0.9}"
export SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY:-0.9}"
export SGLANG_PIECEWISE_ATTN_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_DENSE_LAYERS:-none}"
export SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS:-none}"
export SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS:-0-1}"
export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"
export SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK="${SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK:-fa}"

# --- 4. NVFP4 video FFN ------------------------------------------------------
export SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=1
export SGLANG_LTX2_TE_NVFP4_DISABLE_RHT="${SGLANG_LTX2_TE_NVFP4_DISABLE_RHT:-1}"
export SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING="${SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING:-1}"
export SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION="${SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION:-1}"
export SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU="${SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU:-0}"
export SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE="${SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE:-0}"

# --- 5. stage-2 midpoint token prune ----------------------------------------
# Keep 50% of video tokens at refine steps 1-2.
export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_RATIO="${SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_RATIO:-0.5}"
export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_METHOD="${SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_METHOD:-feat_norm}"
export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_STEPS="${SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_STEPS:-1,2}"

# --- validated-run extras ----------------------------------------------------
export SGLANG_LTX2_PREPROJECT_PROMPTS="${SGLANG_LTX2_PREPROJECT_PROMPTS:-1}"
export SGLANG_LTX2_CACHE_ROPE_EMB="${SGLANG_LTX2_CACHE_ROPE_EMB:-1}"

# --- generate CLI overrides derived from the PISA settings above -------------
LTX23_COMPONENT_ATTENTION_BACKENDS="${SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS:-transformer=fa,transformer_2=piecewise_attn}"
LTX23_ATTENTION_BACKEND_CONFIG="piecewise_sparsity=${SGLANG_PIECEWISE_ATTN_SPARSITY},piecewise_block_size=${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE},piecewise_only_video_self_attention=${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF},piecewise_stage1_schedule=false,piecewise_stage1_dense_steps=0,piecewise_stage1_start_sparsity=${SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY},piecewise_stage1_end_sparsity=${SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY},piecewise_dense_layers=${SGLANG_PIECEWISE_ATTN_DENSE_LAYERS},piecewise_stage1_dense_layers=${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS},piecewise_stage2_dense_layers=${SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS},piecewise_approx_remainder=${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER},piecewise_route_mode=${SGLANG_PIECEWISE_ATTN_ROUTE_MODE},piecewise_dense_fallback=${SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK}"
LTX23_CACHE_ALGO="stage1_cache_core"
LTX23_VARIANT="fullopt"
