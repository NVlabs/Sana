#!/usr/bin/env bash
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/run.py}"
: "${LTX25_PIPELINE:?LTX25_PIPELINE must be bf16 or nvfp4}"
: "${LTX25_LTX_ROOT:?Set LTX25_LTX_ROOT to the pinned LTX-2 checkout}"
: "${LTX25_WEIGHTS_ROOT:?Set LTX25_WEIGHTS_ROOT to the LTX-2.5 model snapshot}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$LTX25_LTX_ROOT/.venv/bin/python}"
PROMPT_FILE="${LTX25_PROMPT_FILE:-$REPO_ROOT/models/ltx25/prompts/p01_multishot.txt}"

case "${LTX25_WORKLOAD:-4k5s}" in
  4k5s)
    WIDTH=3840; HEIGHT=2176; FRAMES=121; TOKENS=130560
    ;;
  1080p20s)
    WIDTH=1920; HEIGHT=1088; FRAMES=481; TOKENS=124440
    ;;
  *)
    echo "unknown LTX25_WORKLOAD=${LTX25_WORKLOAD:-}" >&2
    exit 2
    ;;
esac

BF16="$LTX25_WEIGHTS_ROOT/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
NVFP4="$LTX25_WEIGHTS_ROOT/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
TEXT="$LTX25_WEIGHTS_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE="$LTX25_WEIGHTS_ROOT/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO_VAE="$LTX25_WEIGHTS_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER="$LTX25_WEIGHTS_ROOT/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_ROOT${LTX25_EXTRA_PYTHONPATH:+:$LTX25_EXTRA_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR"
RUNNER_ARGS=(--pipeline "$LTX25_PIPELINE" --metrics "$OUT_DIR/benchmark.json")
OFFICIAL_ARGS=()
TRANSFORMER="$BF16"

case "$LTX25_PIPELINE" in
  bf16)
    OFFICIAL_ARGS+=(--offload cpu)
    ;;
  nvfp4)
    TRANSFORMER="$NVFP4"
    OFFICIAL_ARGS+=(--quantization nvfp4-prequant)
    TABLE="${LTX25_ADALN_TABLE:-$REPO_ROOT/.cache/ltx25/rtx5090/${LTX25_WORKLOAD:-4k5s}/exact_adaln.pt}"
    if [[ ! -s "$TABLE" ]]; then
      "$PYTHON_BIN" -m models.ltx25.RTX5090.build_exact_adaln \
        --checkpoint "$NVFP4" --tokens "$TOKENS" --output "$TABLE"
    fi
    RUNNER_ARGS+=(--nvfp4-embedding-source "$BF16" --exact-adaln-table "$TABLE")
    ;;
  *)
    echo "unknown LTX25_PIPELINE=$LTX25_PIPELINE" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m models.ltx25.RTX5090.gpu_infer \
  "${RUNNER_ARGS[@]}" -- \
  --transformer-path "$TRANSFORMER" \
  --text-encoder-path "$TEXT" \
  --video-vae-path "$VIDEO_VAE" \
  --audio-vae-path "$AUDIO_VAE" \
  --spatial-upsampler-path "$UPSCALER" \
  "${OFFICIAL_ARGS[@]}" \
  --width "$WIDTH" --height "$HEIGHT" --num-frames "$FRAMES" \
  --frame-rate 24 --seed "${LTX25_SEED:-42}" \
  --prompt "$(< "$PROMPT_FILE")" --output-path "$OUT_DIR/out.mp4"
