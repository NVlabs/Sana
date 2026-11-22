#!/bin/bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra training args]" >&2
    exit 1
fi
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANA_DIR="${SANA_ROBOT_ROOT:-$(python -c 'import pathlib, sana_robot; print(pathlib.Path(sana_robot.__file__).resolve().parents[1])')}"
config="${SANA_ROBOT_CONFIG:-${SANA_DIR}/configs/robodojo_ar.yaml}"
data_dir="${SCRIPT_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_dir="${SCRIPT_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"

if [[ ! -f "${data_dir}/sana_robot_zip/meta.json" || ! -f "${data_dir}/normalization.json" ]]; then
    echo "[SANA_Robot] no dataset at ${data_dir}; build it with" >&2
    echo "  bash process_data.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type}" >&2
    exit 1
fi

arguments=(
    --work_dir "${ckpt_dir}"
    --data.path "${data_dir}/sana_robot_zip"
    --data.normalization "${data_dir}/normalization.json"
    --seed "${seed}"
)
if [[ -d "${data_dir}/latents" ]]; then
    arguments+=(--data.latent_store "${data_dir}/latents")
fi
if [[ -n "${SANA_ROBOT_PRETRAINED:-}" ]]; then
    arguments+=(--checkpoint "${SANA_ROBOT_PRETRAINED}")
fi
if [[ -n "${SANA_ROBOT_VAE:-}" ]]; then
    arguments+=(--vae "${SANA_ROBOT_VAE}")
fi
if [[ -n "${SANA_ROBOT_TEXT_ENCODER:-}" ]]; then
    arguments+=(--text_encoder "${SANA_ROBOT_TEXT_ENCODER}")
fi

mkdir -p "${ckpt_dir}"
cp "${data_dir}/sana_robot_zip/meta.json" "${ckpt_dir}/meta.json"
cp "${data_dir}/normalization.json" "${ckpt_dir}/normalization.json"

IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export GPUS="${#gpu_ids[@]}"

cd "${SANA_DIR}"
bash scripts/train/train_ar.sh "${config}" "${arguments[@]}" "${@:7}"
