#!/usr/bin/env bash
# ported from Sol-LTX-Infer scripts/slurm_ltx23_best_nvfp4_video_attn_ffn_sglquant_1080p10s.sh @ 29d0d9e
# Reference-only LTX-2.3 NVFP4 recipe. Do not execute from this autovideo loop.
# The real path requires CUDA, FlashInfer/FP4 support, quantized transformer
# component directories, and the source checkout environment used by the LTX run.
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -J ltx23-fp4-sglq
#SBATCH -o outputs/slurm/ltx23-fp4-sglq-%j.out
#SBATCH -e outputs/slurm/ltx23-fp4-sglq-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN=1
export SGLANG_LTX2_FUSED_ADALN=1
export SGLANG_LTX2_FUSED_QKNORM_ROPE=1
export SGLANG_LTX2_FUSED_DUAL_MODULATE=1
export SGLANG_LTX2_FUSED_ADA_VALUES_ALL=1
export SGLANG_LTX2_FUSED_RESIDUAL_GATE=1
export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=1
export SGLANG_LTX2_COMPILE_GATE_TO_OUT=1
export SGLANG_LTX2_FUSED_AUDIO_QKVG=1
export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=1
export SGLANG_LTX2_VAE_COMPILE_MODE="${SGLANG_LTX2_VAE_COMPILE_MODE:-max-autotune-no-cudagraphs}"
export SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=1
export SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND="${SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND:-cudnn}"
export SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND="${SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND:-flashinfer}"
export SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU="${SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU:-1}"
export SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE="${SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE:-1}"
export SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE="${SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE:-0}"

PROMPT="${PROMPT:-A cinematic aerial shot of clouds moving across a mountain ridge at sunrise}"
OUT_DIR="${OUT_DIR:-outputs/ltx23-dev-1080p10s-nvfp4-video-attn-ffn-flashinfer-fp4biasgelu-projoutgate-best-pipeline}"
mkdir -p outputs/slurm "$OUT_DIR"

if [[ "${SGLANG_DIFFUSION_LTX2_EVENT_PROFILE:-0}" == "1" ]]; then
  export SGLANG_DIFFUSION_LTX2_PROFILE_PATH="${SGLANG_DIFFUSION_LTX2_PROFILE_PATH:-$OUT_DIR/ltx2_event_profile.json}"
fi

.conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path Lightricks/LTX-2.3 \
  --backend auto \
  --pipeline-class-name LTX2TwoStagePipeline \
  --num-gpus 1 \
  --performance-mode speed \
  --ltx2-two-stage-device-mode resident \
  --warmup true \
  --warmup-steps 30 \
  --height 1088 \
  --width 1920 \
  --num-frames 241 \
  --fps 24 \
  --seed 42 \
  --num-inference-steps 30 \
  --guidance-scale 3.0 \
  --prompt "$PROMPT" \
  --return-file-paths-only true \
  --component-paths.transformer outputs/ltx23-selective-nvfp4-video-attn-ffn-transformer-mat \
  --component-paths.transformer_2 outputs/ltx23-selective-nvfp4-video-attn-ffn-stage2-lora-transformer-mat \
  --output-file-path "$OUT_DIR/out.mp4" \
  --perf-dump-path "$OUT_DIR/perf.json"

.conda/ltx23/bin/python - <<'PY2'
import json
import os

out_dir = os.environ.get(
    "OUT_DIR",
    "outputs/ltx23-dev-1080p10s-nvfp4-video-attn-ffn-flashinfer-fp4biasgelu-projoutgate-best-pipeline",
)
path = os.path.join(out_dir, "perf.json")
d = json.load(open(path))
steps = {x["name"]: x["duration_ms"] for x in d.get("steps", [])}
summary = {
    "output_dir": out_dir,
    "fp4_quantize_backend": os.environ.get("SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND"),
    "fp4_gemm_backend": os.environ.get("SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND"),
    "fp4_fused_proj_in_bias_gelu": os.environ.get("SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU"),
    "fp4_fused_proj_out_bias_gate": os.environ.get("SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE"),
    "fp4_fused_attn_to_out_bias_gate": os.environ.get("SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE"),
    "total_s": d.get("total_duration_ms", 0) / 1000,
    "denoise_s": steps.get("LTX2AVDenoisingStage", 0) / 1000,
    "refine_s": steps.get("LTX2RefinementStage", 0) / 1000,
    "dit_s": (steps.get("LTX2AVDenoisingStage", 0) + steps.get("LTX2RefinementStage", 0)) / 1000,
    "decode_s": steps.get("LTX2AVDecodingStage", 0) / 1000,
    "speedup_vs_59_332": 59.332 / (d.get("total_duration_ms", 0) / 1000),
}
open(os.path.join(out_dir, "summary.json"), "w").write(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY2
