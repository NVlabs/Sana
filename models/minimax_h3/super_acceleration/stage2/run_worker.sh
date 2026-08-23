#!/usr/bin/env bash
set -euo pipefail

readonly pair_id=${SLURM_PROCID:?launch as the two-task Stage-2 step}
(( pair_id >= 0 && pair_id < 2 )) || { echo "invalid pair ${pair_id}" >&2; exit 2; }
: "${E2E_HOST_SUPER_ROOT:?}" "${E2E_STAGE2_RUN_ROOT:?}" "${E2E_STAGE2_CACHE_ROOT:?}"
: "${E2E_SOL_REPO:?}" "${E2E_STAGE2_PYTHON:?}" "${E2E_LTX_WEIGHTS:?}"
: "${E2E_AUTH_TOKEN:?}" "${E2E_HANDOFF_TRANSPORT:?}" "${E2E_HANDOFF_PORT_BASE:?}"

case ${E2E_HANDOFF_TRANSPORT} in
  tcp) endpoint="tcp://127.0.0.1:$((E2E_HANDOFF_PORT_BASE + pair_id))" ;;
  unix) endpoint="unix://${E2E_SOCKET_DIR:?}/pair_${pair_id}.sock" ;;
  *) echo "invalid transport ${E2E_HANDOFF_TRANSPORT}" >&2; exit 2 ;;
esac

readonly cache=${E2E_STAGE2_CACHE_ROOT}/stage2/1344x768_121f_sol_compile_tile${E2E_INPUT_VAE_TEMPORAL_TILE}/pair_${pair_id}
export TORCHINDUCTOR_CACHE_DIR=${cache}/inductor
export TRITON_CACHE_DIR=${cache}/triton
export CUDA_CACHE_PATH=${cache}/cuda
export CUTE_DSL_CACHE_DIR=${cache}/cute_dsl
export XDG_CACHE_HOME=${cache}/xdg
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${CUTE_DSL_CACHE_DIR}" "${XDG_CACHE_HOME}"
exec 9>"${cache}.lock"
flock -n 9 || {
  echo "Stage-2 compile cache is in use: ${cache}" >&2
  exit 3
}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SOL_ATTN_STRICT=1 H3_LTX_SOURCE_FRAMES=124 H3_LTX_OUTPUT_FRAMES=121
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset CC CXX CPP HOSTCC HOSTCXX CUDAHOSTCXX NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
readonly profile=${E2E_SOL_REPO}/models/ltx2.5-refiner/GB200
export PYTHONPATH=${E2E_HOST_SUPER_ROOT}:${E2E_HOST_SUPER_ROOT}/stage2:${profile}:${E2E_SOL_REPO}/models/ltx25/GB200/ltx_src:${E2E_SOL_REPO}/models/ltx25/GB200/environment/LTX-2/packages/ltx-kernels/src:${E2E_SOL_REPO}${PYTHONPATH:+:${PYTHONPATH}}

readonly output=${E2E_STAGE2_RUN_ROOT}/pair_${pair_id}/stage2
mkdir -p "${output}"
[[ ! -e ${output}/benchmark.json ]] || { echo "refusing existing Stage-2 benchmark" >&2; exit 2; }

exec "${E2E_STAGE2_PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  "${E2E_HOST_SUPER_ROOT}/stage2/stage2_server.py" \
  --endpoint "${endpoint}" \
  --auth-token "${E2E_AUTH_TOKEN}" \
  --pair-id "${pair_id}" \
  --handoff-mode "${E2E_HANDOFF_MODE:?}" \
  --hot-repeats "${E2E_HOT_REPEATS}" \
  --template-manifest "${E2E_REFINER_TEMPLATE_MANIFEST}" \
  --first-frame-root "${E2E_REFINER_FIRST_FRAME_ROOT}" \
  --output-dir "${output}/videos" \
  --metadata-path "${output}/benchmark.json" \
  --compile-cache-root "${cache}" \
  --input-vae-temporal-tile-mode "${E2E_INPUT_VAE_TEMPORAL_TILE}" \
  --transformer "${E2E_LTX_WEIGHTS}/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --text-encoder "${E2E_LTX_WEIGHTS}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae "${E2E_LTX_WEIGHTS}/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae "${E2E_LTX_WEIGHTS}/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --upsampler "${E2E_LTX_WEIGHTS}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --refiner-lora "${E2E_LTX_WEIGHTS}/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --taehv-source "${E2E_SOL_REPO}/models/ltx2.5-refiner/GB200/vendor/taehv/taehv.py" \
  --taehv-checkpoint "${E2E_LTX_WEIGHTS}/taehv/taeltx2_3_wide.pth"
