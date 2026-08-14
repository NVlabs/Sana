#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[error] run the refiner only inside a four-GPU Slurm allocation" >&2
  exit 2
fi

: "${OUT_DIR:?OUT_DIR must be set by scripts/run.py}"
: "${INPUT_ROOT:?set INPUT_ROOT to the directory containing the input manifest assets}"
: "${MANIFEST:?set MANIFEST to the one-row input JSON manifest}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for the refined MP4}"
: "${METADATA_DIR:?set METADATA_DIR for benchmark and validation JSON}"

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
readonly WEIGHTS_ROOT="${LTX25_REFINER_WEIGHTS_ROOT:-/lustre/fsw/portfolios/nvr/users/yitongl/pretrained_models/LTX-2.5-public}"

require_fixed() {
  local name="$1"
  local expected="$2"
  local actual="${!name:-}"
  if [[ "$actual" != "$expected" ]]; then
    echo "[error] fixed refiner contract requires ${name}=${expected}; got ${actual:-<unset>}" >&2
    exit 2
  fi
}

require_fixed LTX25_REFINER_WORLD_SIZE 4
require_fixed LTX25_REFINER_PARALLELISM head_context
require_fixed LTX25_REFINER_PARALLEL_DEGREE 4
require_fixed LTX25_REFINER_PARAMETER_REPLICATION full
require_fixed LTX25_REFINER_SELF_ATTN_HEAD_SHARDS 4
require_fixed LTX25_REFINER_SELF_ATTN_TOKEN_SCOPE full_sequence
require_fixed LTX25_REFINER_WIDTH 1920
require_fixed LTX25_REFINER_HEIGHT 1088
require_fixed LTX25_REFINER_FRAME_COUNT 241
require_fixed LTX25_REFINER_FPS 24
require_fixed LTX25_REFINER_BATCH_SIZE 1
require_fixed LTX25_REFINER_SIGMAS 0.909375,0.725,0.421875,0.0
require_fixed LTX25_REFINER_TAUS 1.0,1.25,1.5
require_fixed LTX25_REFINER_TRANSFORMER_LAYERS 48
require_fixed LTX25_REFINER_DENSE_SELF_ATTN_LAYERS 0
require_fixed LTX25_REFINER_SOL_SELF_ATTN_LAYERS 1-47
require_fixed LTX25_REFINER_CROSS_ATTN dense
require_fixed LTX25_REFINER_SOL_THRESH_TYPE diag
require_fixed LTX25_REFINER_SOL_KV_SPLITS auto
require_fixed LTX25_REFINER_DTYPE bfloat16
require_fixed LTX25_REFINER_LORA_STRENGTH 0.8
require_fixed LTX25_REFINER_CACHE 0
require_fixed LTX25_REFINER_COMPILE 0
require_fixed LTX25_REFINER_OFFLOAD 0
require_fixed LTX25_REFINER_QUANTIZATION none
require_fixed LTX25_REFINER_SINK 0
require_fixed LTX25_REFINER_REORDER 0
require_fixed LTX25_REFINER_WARMUP_REQUESTS 1
require_fixed LTX25_REFINER_MEASURE_REQUESTS 1
require_fixed LTX25_REFINER_WARMUP_INDEX 0
require_fixed LTX25_REFINER_SAMPLE_INDEX 0
require_fixed LTX25_REFINER_TAEHV_SOURCE_COMMIT 32ac0146b11007cda5a57b60a3b35653361fb8a4
require_fixed LTX25_REFINER_TAEHV_WEIGHT_SHA256 007788e6b9cb7f77e8589ae30ba7456b119d38b0d017e1d349c1c1d11e3d6339
require_fixed SOL_ATTN_STRICT 1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] missing shared repository environment: $PYTHON_BIN" >&2
  exit 1
fi

readonly TRANSFORMER="$WEIGHTS_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
readonly TEXT_ENCODER="$WEIGHTS_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
readonly VIDEO_VAE="$WEIGHTS_ROOT/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
readonly UPSAMPLER="$WEIGHTS_ROOT/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
LORA="$WEIGHTS_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16-1.0.safetensors"
if [[ ! -f "$LORA" ]]; then
  LORA="$WEIGHTS_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
fi
readonly LORA
readonly TAEHV_SOURCE="$HERE/vendor/taehv/taehv.py"
readonly TAEHV_CHECKPOINT="${LTX25_REFINER_TAEHV_WEIGHT}"

for required in "$TRANSFORMER" "$TEXT_ENCODER" "$VIDEO_VAE" "$UPSAMPLER" "$LORA" \
  "$TAEHV_SOURCE" "$TAEHV_CHECKPOINT" "$INPUT_ROOT" "$MANIFEST"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing refiner runtime asset: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR" "$METADATA_DIR" "$OUT_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SOL_ATTN_STRICT=1
export PYTHONPATH="$HERE:$REPO_ROOT/models/ltx25/GB200/ltx_src:$REPO_ROOT/models/ltx25/GB200/environment/LTX-2/packages/ltx-kernels/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[ltx25-refiner] fixed workload=1920x1088/241f/24fps Stage2=3 updates"
echo "[ltx25-refiner] parallelism=4-way-head-context parameters=full-replica attention=Sol-SM100"
nvidia-smi -L

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "$HERE/refiner_head_cp.py" \
  --input-root "$INPUT_ROOT" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --metadata-dir "$METADATA_DIR" \
  --transformer "$TRANSFORMER" \
  --text-encoder "$TEXT_ENCODER" \
  --video-vae "$VIDEO_VAE" \
  --upsampler "$UPSAMPLER" \
  --refiner-lora "$LORA" \
  --taehv-source "$TAEHV_SOURCE" \
  --taehv-checkpoint "$TAEHV_CHECKPOINT"
