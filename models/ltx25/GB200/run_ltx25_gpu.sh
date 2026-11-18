#!/usr/bin/env bash
# LTX-2.5 4xGB200 runtime entry. The dense and fullopt configs use this same
# body so workload, model paths, prompts, and artifact handling cannot drift.
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/run.py}"
: "${LTX25_VARIANT:?LTX25_VARIANT must be dense or fullopt}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
WEIGHTS_ROOT="${LTX25_WEIGHTS_ROOT:-/lustre/fsw/portfolios/nvr/users/yitongl/pretrained_models/LTX-2.5-public}"

if [[ "$PYTHON_BIN" == */* && "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="$REPO_ROOT/$PYTHON_BIN"
elif [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[error] no usable Python environment: ${PYTHON_BIN:-<empty>}" >&2
  echo "[error] run models/ltx25/GB200/setup_env.sh or override PYTHON_BIN" >&2
  exit 1
fi

if [[ "${LTX25_WORLD_SIZE:-4}" != "4" ]]; then
  echo "[error] the GB200 runtime is validated only with LTX25_WORLD_SIZE=4" >&2
  exit 2
fi
if [[ "${LTX_SOL_STAGE1:-0}" != "0" ]]; then
  echo "[error] this delivery intentionally excludes SOL sparse attention" >&2
  exit 2
fi

case "${LTX25_PROFILE:-default5s}" in
  default5s)
    WIDTH=1536; HEIGHT=1024; FRAMES=121
    ;;
  4k5s)
    WIDTH=3840; HEIGHT=2176; FRAMES=121
    ;;
  1080p20s)
    WIDTH=1920; HEIGHT=1088; FRAMES=481
    ;;
  *)
    echo "[error] unknown LTX25_PROFILE=${LTX25_PROFILE:-}" >&2
    exit 2
    ;;
esac

TRANSFORMER="${LTX25_TRANSFORMER:-$WEIGHTS_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors}"
TEXT_ENCODER="${LTX25_TEXT_ENCODER:-$WEIGHTS_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors}"
VIDEO_VAE="${LTX25_VIDEO_VAE:-$WEIGHTS_ROOT/vae/ltx-2.5-video-vae-conv-bf16.safetensors}"
AUDIO_VAE="${LTX25_AUDIO_VAE:-$WEIGHTS_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors}"
UPSCALER="${LTX25_UPSCALER:-$WEIGHTS_ROOT/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors}"
DISTILLED_LORA="${LTX25_DISTILLED_LORA:-$WEIGHTS_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16-1.0.safetensors}"
if [[ ! -e "$DISTILLED_LORA" ]]; then
  DISTILLED_LORA="$WEIGHTS_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
fi

PROMPT_SPEC="${LTX25_PROMPT_FILES:-models/ltx25/prompts/p01_multishot.txt}"
IFS=: read -r -a PROMPT_ITEMS <<< "$PROMPT_SPEC"
PROMPT_FILES=""
RESOLVED_PROMPTS=()
for prompt_path in "${PROMPT_ITEMS[@]}"; do
  if [[ "$prompt_path" != /* ]]; then
    prompt_path="$REPO_ROOT/$prompt_path"
  fi
  RESOLVED_PROMPTS+=("$prompt_path")
  if [[ -z "$PROMPT_FILES" ]]; then
    PROMPT_FILES="$prompt_path"
  else
    PROMPT_FILES="$PROMPT_FILES:$prompt_path"
  fi
done
export LTX25_PROMPT_FILES="$PROMPT_FILES"

for required in "$PYTHON_BIN" "$TRANSFORMER" "$TEXT_ENCODER" "$VIDEO_VAE" \
  "$AUDIO_VAE" "$UPSCALER" "$DISTILLED_LORA" "${RESOLVED_PROMPTS[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing LTX-2.5 runtime asset: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"
COMPILE_CACHE_ROOT="${LTX25_COMPILE_CACHE_ROOT:-.cache/ltx25/gb200-sm100_py31311_torch2110-cu130_triton360/ltx25-7954dcb-ccedf84-stage-scope-v1}"
if [[ "$COMPILE_CACHE_ROOT" != /* ]]; then
  COMPILE_CACHE_ROOT="$REPO_ROOT/$COMPILE_CACHE_ROOT"
fi
COMPILE_CACHE="$COMPILE_CACHE_ROOT/${LTX25_VARIANT}/${LTX25_PROFILE:-default5s}"
mkdir -p "$COMPILE_CACHE"/{inductor,triton,cuda}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$HERE/ltx_src:$HERE/environment/LTX-2/packages/ltx-kernels/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE/inductor"
export TRITON_CACHE_DIR="$COMPILE_CACHE/triton"
export CUDA_CACHE_PATH="$COMPILE_CACHE/cuda"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

export LTX_S1_PARALLEL="${LTX25_S1_PARALLEL:-sp}"
export LTX_CFG_ORDER="${LTX25_CFG_ORDER:-0,3,1,2}"
export LTX_STACK_CACHE="${LTX25_CACHE:-off}"
export LTX_CACHE_THRESH="${LTX25_CACHE_THRESHOLD:-0.08}"
export LTX_CACHE_WARMUP="${LTX25_CACHE_WARMUP:-1}"
export LTX_CACHE_MAXCONSEC="${LTX25_CACHE_MAX_CONSECUTIVE:-10}"
export LTX_SOL_STAGE1=0
export LTX_TIME_FILE="$OUT_DIR/timing"
export LTX_TIME_FORWARD=0
export LTX_TIME_STEPS=0

COMPILE_ARGS=()
if [[ "${LTX25_COMPILE:-0}" == "1" ]]; then
  COMPILE_ARGS=(--compile mode=max-autotune-no-cudagraphs fullgraph=false capture=false)
fi

echo "[ltx25] host=$(hostname) variant=$LTX25_VARIANT profile=${LTX25_PROFILE:-default5s}"
echo "[ltx25] final=${WIDTH}x${HEIGHT} stage1=$((WIDTH / 2))x$((HEIGHT / 2)) frames=$FRAMES fps=${LTX25_FPS:-24}"
echo "[ltx25] Stage1=${LTX25_STAGE1_STEPS:-30} steps/$LTX_S1_PARALLEL Stage2=2 updates/2x2-TDP VAE=2x2-distributed"
echo "[ltx25] cache=$LTX_STACK_CACHE threshold=$LTX_CACHE_THRESH compile=${LTX25_COMPILE:-0} SOL=off"
echo "[ltx25] persistent_compile_cache=$COMPILE_CACHE"
nvidia-smi -L

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$HERE/gpu_infer.py" \
  --transformer-path "$TRANSFORMER" \
  --text-encoder-path "$TEXT_ENCODER" \
  --video-vae-path "$VIDEO_VAE" \
  --audio-vae-path "$AUDIO_VAE" \
  --spatial-upsampler-path "$UPSCALER" \
  --distilled-lora "$DISTILLED_LORA" \
  --width "$WIDTH" --height "$HEIGHT" --num-frames "$FRAMES" \
  --frame-rate "${LTX25_FPS:-24}" --num-inference-steps "${LTX25_STAGE1_STEPS:-30}" \
  --seed "${LTX25_SEED:-42}" "${COMPILE_ARGS[@]}" \
  --prompt "Prompt text is supplied by LTX25_PROMPT_FILES" \
  --output-path "$OUT_DIR/out.mp4"
